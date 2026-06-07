"""melody-miner Gradio 통합 데모.

입력 WAV (+ 선택: 목표 화자 reference) 업로드 → 반주 생성 + 음성 변환 →
보컬 / 반주 / 최종 믹스를 한 번에 재생.

체크포인트는 **자동 고정**됩니다: ``checkpoints/m2a/`` 의 ckpt를 M2A로,
``checkpoints/tnp/`` 의 ckpt를 TNP로 자동 선택해 시작 시 로드합니다.
(원하면 ``--m2a-checkpoint`` / ``--tnp-checkpoint`` 로 직접 지정 가능.)

실행:
    python app.py                 # checkpoints/m2a, checkpoints/tnp 자동 사용
    python app.py --share
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
import asyncio
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # Uvicorn sometimes overrides the loop policy back to Proactor. 
    # Monkeypatch the known Windows Proactor bug where browser audio streaming disconnects crash the loop.
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport
        _original_ccl = _ProactorBasePipeTransport._call_connection_lost
        def _silenced_ccl(self, exc):
            try:
                _original_ccl(self, exc)
            except ConnectionResetError as e:
                if getattr(e, 'winerror', None) == 10054:
                    pass
                else:
                    raise
        _ProactorBasePipeTransport._call_connection_lost = _silenced_ccl
    except ImportError:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from orchestration import DEFAULT_CONFIG, REPO_ROOT
from orchestration.accompaniment import load_m2a
from orchestration.config import GenParams, OrchestrationConfig, MixParams, VoiceParams
from orchestration.pipeline import run_full
from orchestration.voice import preload_tnp
from m2a_transformer.utils.logger import logger

try:
    import gradio as gr
except ImportError:
    raise ImportError("Gradio가 필요합니다: pip install 'melody_miner[demo]'  (또는 pip install gradio)")

CKPT_ROOT = REPO_ROOT / "checkpoints"
_CKPT_EXTS = {".ckpt", ".pt", ".pth", ".safetensors"}
_STATE: dict = {}


def _gpu_total_gb() -> float:
    """Total VRAM of the active CUDA device in GiB (0 if no CUDA)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def _warmup_sample() -> tuple[Path | None, list[Path]]:
    """A real (input, references) pair for the full GPU warmup, if present."""
    refs_dir = REPO_ROOT / "references"
    for cand in ("source.wav", "context.wav", "target.wav"):
        p = refs_dir / cand
        if p.exists():
            tgt = refs_dir / "target.wav"
            refs = [tgt] if (tgt.exists() and _STATE.get("tnp_ckpt")) else []
            return p, refs
    return None, []


def _warmup(full: bool = False) -> None:
    """Warm cold paths at startup so the first generation isn't slow.

    Always (cheap, VRAM-safe):
      • build + cache the TNP model (removes the ~5-10s B0 from the first click)
      • warm basic-pitch's ONNX session on a tiny tone (CPU - no VRAM)

    ``full=True`` (only on big-VRAM machines): additionally run ONE real
    pipeline pass to warm the GPU encoder paths (ContentVec/crepe/Vocos, M2A
    CUDA kernels) so even the FIRST generation is ~fully warm. This is skipped
    by default because, on a 4 GB GPU, pre-built TNP weights + a dummy inference
    peak past VRAM (OOM); the dev box uses the cheap path, the big-VRAM demo box
    passes ``--full-warmup`` (or it auto-enables when VRAM is large).
    """
    import shutil
    import tempfile

    import numpy as np
    from scipy.io import wavfile

    from orchestration.accompaniment import transcribe

    logger.info(f"[warmup] 워밍업 시작 (full={full}).")

    # 1) Build + cache the TNP model (GPU weights, no inference).
    try:
        preload_tnp(_STATE.get("tnp_ckpt"))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[warmup] TNP 모델 로드 실패(무시): {e}")

    if _STATE.get("model") is None:
        logger.info("[warmup] M2A 미로드 - 워밍업 일부 생략.")
        return

    # 2) Warm basic-pitch's ONNX session on a tiny tone (CPU - no VRAM impact).
    d = Path(tempfile.mkdtemp(prefix="mm_warm_"))
    try:
        sr = 16_000
        t = np.arange(int(1.0 * sr)) / sr
        wav = d / "warm.wav"
        wavfile.write(str(wav), sr, (0.3 * np.sin(2 * np.pi * 220.0 * t) * 32767).astype(np.int16))
        transcribe(wav, _STATE["model"].cfg, d / "warm.mid")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[warmup] basic-pitch 워밍업 실패(무시): {e}")
    finally:
        shutil.rmtree(d, ignore_errors=True)

    # 3) Full GPU warmup - one real pass (big VRAM only).
    if full:
        sample, refs = _warmup_sample()
        if sample is None:
            logger.info("[warmup] full 워밍업용 샘플 WAV 없음 - GPU 추론 워밍업 생략.")
        else:
            wd = Path(tempfile.mkdtemp(prefix="mm_warm_full_"))
            try:
                cfg = OrchestrationConfig(
                    m2a_checkpoint=_STATE["m2a_ckpt"], m2a_config=_STATE["config_path"],
                    tnp_checkpoint=_STATE.get("tnp_ckpt"), references=refs,
                )
                run_full(sample, wd, cfg, model=_STATE["model"])
                logger.info("[warmup] full GPU 워밍업 완료 - 첫 생성도 빠르게 동작합니다.")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[warmup] full GPU 워밍업 실패(무시): {e}")
            finally:
                shutil.rmtree(wd, ignore_errors=True)

    logger.info("[warmup] 완료.")


def _default_ckpt(folder: str) -> Path | None:
    """First checkpoint file under checkpoints/<folder>/ (or None)."""
    d = CKPT_ROOT / folder
    if not d.exists():
        return None
    files = sorted(p for p in d.iterdir()
                   if p.is_file() and p.suffix.lower() in _CKPT_EXTS)
    return files[0] if files else None


# Chars that break Gradio's /file= serving URL → browser preview fails.
_UNSAFE_RE = re.compile(r"[#&?%+;]")
_SAFE_DIR = Path(tempfile.gettempdir()) / "mm_safe_uploads"


def _safe_path(path):
    """Copy an uploaded file to a URL-safe filename so its browser preview works.

    Gradio serves audio/files to the browser via a /file=<path> URL; characters
    like #, &, ?, %, + in the filename break that URL so the preview never loads
    (server-side processing by path still works). On upload we copy the file to a
    sanitised name and hand THAT back to the component, so the preview re-renders
    from a safe URL. No-op when the name is already safe.
    """
    if not path:
        return path
    p = Path(path)
    if not _UNSAFE_RE.search(p.name):
        return str(p)
    _SAFE_DIR.mkdir(parents=True, exist_ok=True)
    dst = _SAFE_DIR / _UNSAFE_RE.sub("_", p.name)
    try:
        shutil.copy2(p, dst)
        return str(dst)
    except Exception:  # noqa: BLE001
        return str(p)


_MODE_LABELS = {
    "M2A + TNP (전체: 반주+음성변환)": "full",
    "M2A만 (반주 생성)": "m2a",
    "TNP만 (음성 변환)": "tnp",
}


_OUT7 = (None,) * 6 + ("",)  # melody, accomp, vocal, final, melody_midi, accomp_midi, status


def _err(msg: str):
    return (None,) * 6 + (msg,)


def _run(input_wav, midi_file, references, mode_label, transcriber,
         denoise, temperature, top_p, cfg_w, avoid, outname, postprocess,
         mel_sharpen, sharpen_t, sharpen_f, saturation_db):
    mode = _MODE_LABELS.get(mode_label, "full")

    # M2A mode accepts a MIDI file (skips transcription) OR audio; others need audio.
    if mode == "m2a" and midi_file:
        src = midi_file
    else:
        src = input_wav
    if not src:
        return _err("⚠️ 입력(오디오 또는 MIDI)을 올려주세요.")

    if mode in ("full", "m2a") and not _STATE.get("model"):
        return _err("⚠️ M2A 체크포인트가 로드되지 않았습니다 (checkpoints/m2a/ 확인).")
    if mode == "tnp" and not _STATE.get("tnp_ckpt"):
        return _err("⚠️ TNP 체크포인트가 없습니다 (checkpoints/tnp/ 확인).")

    name = (outname or "").strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("output") / name
    refs = [Path(r) for r in (references or [])]

    config = OrchestrationConfig(
        m2a_checkpoint=_STATE.get("m2a_ckpt") or Path("none"),
        m2a_config=_STATE["config_path"],
        tnp_checkpoint=_STATE.get("tnp_ckpt"),
        references=refs,
        gen=GenParams(temperature=temperature, top_p=top_p, cfg_w=cfg_w,
                      avoid_note_penalty=avoid, denoise=denoise,
                      transcriber=transcriber),
        voice=VoiceParams(mel_sharpen=mel_sharpen,
                          sharpen_t_alpha=sharpen_t, sharpen_f_alpha=sharpen_f),
        mix=MixParams(postprocess_vocal=postprocess,
                      vocal_saturation_db=saturation_db),
    )
    try:
        r = run_full(src, out_dir, config, model=_STATE.get("model"), mode=mode)
    except Exception as e:  # noqa: BLE001
        return _err(f"❌ 오류: {e}")

    parts = [f"✅ 완료 (모드: {mode})"]
    if r["tempo"] is not None:
        src_kind = "MIDI 입력" if r.get("is_midi_input") else "오디오→전사"
        parts.append(f"{src_kind} · 템포 {r['tempo']:.0f} BPM")
    if mode in ("full", "tnp"):
        parts.append(f"음성변환 {'적용' if r['vc_applied'] else '건너뜀 (' + str(r['vc_reason']) + ')'}")
    parts.append(f"저장 output/{name}/")
    status = " | ".join(parts)

    def s(p):
        return str(p) if p else None
    return (s(r["melody_wav"]), s(r["accomp_wav"]), s(r.get("vocal_post") or r["vocal_wav"]), s(r["final_mix"]),
            s(r["melody_midi"]), s(r["accomp_midi"]), status)


def _visibility(mode_label):
    """Return gr.update(visible=...) for each mode-dependent component."""
    mode = _MODE_LABELS.get(mode_label, "full")
    m2a_on = mode in ("full", "m2a")     # Branch A active
    tnp_on = mode in ("full", "tnp")     # Branch B active
    mix_on = mode in ("full", "m2a")     # a mix is produced
    return (
        gr.update(visible=(mode == "m2a")),   # midi_in (MIDI input only in m2a)
        gr.update(visible=tnp_on),            # refs
        gr.update(visible=tnp_on),            # ref_preview
        gr.update(visible=tnp_on),            # tnp_params group
        gr.update(visible=m2a_on),            # m2a_params group
        gr.update(visible=m2a_on),            # melody_out
        gr.update(visible=m2a_on),            # accomp_out
        gr.update(visible=tnp_on),            # vocal_out
        gr.update(visible=mix_on),            # final_out
        gr.update(visible=m2a_on),            # melody_midi_out
        gr.update(visible=m2a_on),            # accomp_midi_out
    )


def build_ui() -> "gr.Blocks":
    m2a = _STATE.get("m2a_ckpt")
    tnp = _STATE.get("tnp_ckpt")
    ckpt_info = (
        f"**M2A**: `{m2a.name if m2a else '없음 (checkpoints/m2a/ 비어있음)'}`"
        + (" ✅ 로드됨" if _STATE.get("model") else " ⚠️ 미로드") + "  \n"
        f"**TNP**: `{tnp.name if tnp else '없음 → 음성변환 건너뜀'}`"
    )
    labels = list(_MODE_LABELS.keys())

    with gr.Blocks(title="melody-miner") as demo:
        gr.Markdown(
            "# 🎤🎹 melody-miner\n"
            "WAV(또는 MIDI) → **반주 생성**(M2A) + **음성 변환**(TNP).\n\n"
            f"> 체크포인트 자동 고정. {ckpt_info}"
        )

        # ── 모드 (맨 위) ──────────────────────────────────────────────────────
        mode = gr.Radio(
            choices=labels, value=labels[0], label="모드 (선택에 따라 입력/출력이 바뀝니다)",
            info="M2A+TNP=반주+음성변환 믹스 · M2A만=반주(중간산출물 확인용) · TNP만=음성변환 보컬",
        )

        with gr.Row():
            with gr.Column():
                inp = gr.Audio(label="입력 오디오 (노래/허밍)", sources=["upload", "microphone"],
                               type="filepath")
                midi_in = gr.File(label="또는 입력 MIDI (M2A만 - 전사 생략, MIDI→MIDI 검증)",
                                  file_types=[".mid", ".midi"], visible=False)
                refs = gr.File(label="목표 화자 reference WAV (TNP/전체 모드, 여러 개 가능)",
                               file_count="multiple", file_types=[".wav", ".flac"])
                ref_preview = gr.Audio(label="첫 번째 레퍼런스 미리듣기", interactive=False, visible=True)
                with gr.Group() as tnp_params:
                    gr.Markdown("**음성 변환 파라미터 (TNP)**")
                    mel_sharpen = gr.Checkbox(
                        label="mel 샤프닝 (물먹은 음질 완화)", value=True,
                        info="예측 mel 과평활을 보정. 끄면 원래의 뭉개진 소리로 돌아감.")
                    sharpen_t = gr.Slider(
                        0.0, 3.0, step=0.1, value=1.0, label="샤프닝 — 시간축(temporal)",
                        info="프레임간 움직임 복원. 과하면 거칠어짐.")
                    sharpen_f = gr.Slider(
                        0.0, 2.0, step=0.1, value=0.5, label="샤프닝 — 주파수축(spectral)",
                        info="하모닉/포먼트 대비 복원.")
                with gr.Group() as m2a_params:
                    gr.Markdown("**반주 생성 파라미터 (M2A)**")
                    transcriber = gr.Dropdown(
                        ["basic-pitch", "crepe"], value="basic-pitch",
                        label="WAV→MIDI 전사기",
                        info="basic-pitch=다성 AMT · crepe=단성(솔로 보컬 분절 적음)",
                    )
                    denoise = gr.Checkbox(label="전사 전 노이즈 제거", value=False)
                    temperature = gr.Slider(0.5, 2.0, step=0.05, value=1.0, label="Temperature")
                    top_p = gr.Slider(0.5, 1.0, step=0.01, value=0.95, label="Top-p")
                    cfg_w = gr.Slider(0.0, 5.0, step=0.1, value=0.0, label="CFG Weight (0=off)")
                    avoid = gr.Slider(0.0, 8.0, step=0.5, value=0.0, label="Avoid-note Penalty")
                with gr.Group() as mix_params:
                    gr.Markdown("**믹싱 / 후처리 파라미터**")
                    postprocess = gr.Checkbox(label="보컬 후처리 적용 (기계음 완화용 부스트/리버브)", value=True)
                    saturation = gr.Slider(
                        0.0, 24.0, step=1.0, value=0.0,
                        label="Saturation drive (dB, 0=off)",
                        info="하모닉 추가로 둔탁함(물먹음) 완화·밝기 부여. 과하면 거칠어짐. jitter엔 효과 없음.")
                outname = gr.Textbox(label="저장 폴더명 (비우면 타임스탬프)", value="")
                btn = gr.Button("🎵 생성", variant="primary", size="lg")
            with gr.Column():
                status = gr.Textbox(label="상태", interactive=False)
                melody_out = gr.Audio(label="02 멜로디 (전사/입력 MIDI → WAV, 들어보기)", type="filepath")
                accomp_out = gr.Audio(label="03 생성 반주 WAV", type="filepath")
                final_out = gr.Audio(label="05 최종 믹스 (입력+반주 또는 변환보컬+반주)", type="filepath")
                vocal_out = gr.Audio(label="04 변환 보컬 WAV", type="filepath", visible=False)
                with gr.Accordion("MIDI 다운로드 (디버깅)", open=False):
                    melody_midi_out = gr.File(label="02 멜로디 MIDI")
                    accomp_midi_out = gr.File(label="03 반주 MIDI")

        # references comes in as list of temp file objects → paths
        def _paths(files):
            return [f.name if hasattr(f, "name") else f for f in (files or [])]

        def _midi_path(f):
            return f.name if (f is not None and hasattr(f, "name")) else f
            
        def _preview_ref(files):
            if files and len(files) > 0:
                f = files[0]
                return _safe_path(f.name if hasattr(f, "name") else f)
            return None

        # On upload, replace the input audio with a URL-safe copy so its preview
        # plays even when the original filename has #, &, … (server still works).
        inp.upload(fn=_safe_path, inputs=inp, outputs=inp)
        refs.change(fn=_preview_ref, inputs=refs, outputs=ref_preview)

        # mode → toggle visibility
        mode.change(
            fn=_visibility, inputs=mode,
            outputs=[midi_in, refs, ref_preview, tnp_params, m2a_params,
                     melody_out, accomp_out,
                     vocal_out, final_out, melody_midi_out, accomp_midi_out],
        )

        btn.click(
            fn=lambda i, mf, rf, m, tr, d, t, p, c, a, o, pp, ms, st, sfa, sat:
                _run(i, _midi_path(mf), _paths(rf), m, tr, d, t, p, c, a, o, pp,
                     ms, st, sfa, sat),
            inputs=[inp, midi_in, refs, mode, transcriber,
                    denoise, temperature, top_p, cfg_w, avoid, outname, postprocess,
                    mel_sharpen, sharpen_t, sharpen_f, saturation],
            outputs=[melody_out, accomp_out, vocal_out, final_out,
                     melody_midi_out, accomp_midi_out, status],
        )
    return demo


def main() -> None:
    p = argparse.ArgumentParser(description="melody-miner Gradio demo")
    p.add_argument("--m2a-checkpoint", default=None,
                   help="M2A 체크포인트(선택). 생략 시 checkpoints/m2a/ 의 ckpt 자동 사용.")
    p.add_argument("--tnp-checkpoint", default=None,
                   help="TNP 체크포인트(선택). 생략 시 checkpoints/tnp/ 의 ckpt 자동 사용.")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--share", action="store_true")
    p.add_argument("--no-warmup", action="store_true",
                   help="시작 시 모델 워밍업을 건너뜀(첫 생성이 느려질 수 있음).")
    p.add_argument("--full-warmup", action="store_true",
                   help="시작 시 GPU 추론까지 워밍업(큰 VRAM 권장 - 첫 생성도 빠름).")
    p.add_argument("--no-full-warmup", action="store_true",
                   help="VRAM이 커도 full 워밍업을 강제로 끔(가벼운 워밍업만).")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()

    _STATE["config_path"] = Path(args.config)

    # M2A: explicit arg, else auto-fixed from checkpoints/m2a/
    m2a_ckpt = Path(args.m2a_checkpoint) if args.m2a_checkpoint else _default_ckpt("m2a")
    _STATE["m2a_ckpt"] = m2a_ckpt
    _STATE["model"] = load_m2a(m2a_ckpt, _STATE["config_path"]) if m2a_ckpt else None
    if m2a_ckpt:
        print(f"[melody-miner] M2A 고정: {m2a_ckpt}")
    else:
        print("[melody-miner] ⚠️ checkpoints/m2a/ 에 ckpt가 없습니다.")

    # TNP: explicit arg, else auto-fixed from checkpoints/tnp/ (None → VC 건너뜀)
    tnp_ckpt = Path(args.tnp_checkpoint) if args.tnp_checkpoint else _default_ckpt("tnp")
    _STATE["tnp_ckpt"] = tnp_ckpt
    print(f"[melody-miner] TNP 고정: {tnp_ckpt if tnp_ckpt else '없음 (음성변환 건너뜀)'}")

    # Warm at startup so the first generation isn't cold.
    # full GPU warmup: forced by --full-warmup, or auto when VRAM is large
    # (≥ ~7 GiB), unless disabled. Skipped entirely with --no-warmup.
    if not args.no_warmup:
        vram = _gpu_total_gb()
        full = args.full_warmup or (vram >= 7.0 and not args.no_full_warmup)
        logger.info(f"[warmup] GPU VRAM≈{vram:.1f}GiB → full={'on' if full else 'off'}")
        _warmup(full=full)

    demo = build_ui()
    launch_kwargs = dict(server_name=args.host, server_port=args.port, share=args.share)
    try:
        # Gradio 6 moved `theme` from Blocks(...) to launch(...).
        demo.launch(theme=gr.themes.Soft(), **launch_kwargs)
    except TypeError:
        # Older Gradio (<6): theme isn't a launch() kwarg - launch without it.
        demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
