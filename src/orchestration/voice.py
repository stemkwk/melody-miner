"""Branch B — WAV → target-speaker vocal, wrapping the vendored TNP converter.

Instead of calling ``convert.py:convert(args)`` (which rebuilds the heavy
VoiceConversionModel — HuBERT/ContentVec/Vocos — on EVERY call, ~7s wasted), we
build the model ONCE and cache it (``_MODEL_CACHE``), then reuse the vendored
model's public methods (``compute_context`` / ``convert_chunk_streaming``) and
convert.py's helpers (``load_audio`` / ``extract_f0_stats``) for each run. Stage
timings are logged so you can see where time goes.

Graceful fallback: if no TNP checkpoint or no reference audio is supplied (or
the files are missing), voice conversion is SKIPPED and the input WAV is used as
the vocal stem so the full pipeline still runs end-to-end. The heavy stack is
imported lazily so accompaniment-only runs never pull it in.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import torch

from . import ensure_tnp_on_path
from .timing import timed
from m2a_transformer.utils.logger import logger

# Sample-rate / chunking constants (mirror tnp convert.py).
SAMPLE_RATE = 16_000
VOCODER_SR = 24_000
CHUNK_SAMPLES = 16_000 * 4  # 4-second chunks (content encoder runs at 16 kHz)

# Inference-time mel sharpening — band-aid for predicted-mel over-smoothing.
# The TNP regression mel carries ~0.5× temporal / ~0.7× spectral detail of real
# mel ("watery"); unsharp-masking in the log-mel domain restores contrast.
# Tune with scripts/analysis/diagnose_vocoder_wobble.py, then update these. False = off.
MEL_SHARPEN = True
MEL_SHARPEN_T_ALPHA = 1.0   # temporal unsharp strength
MEL_SHARPEN_F_ALPHA = 0.5   # spectral unsharp strength

# Cache the expensive model across calls, keyed by (checkpoint, device).
_MODEL_CACHE: dict = {}


def _refs_ok(references) -> list[Path]:
    return [Path(r) for r in (references or []) if Path(r).exists()]


def preload_tnp(tnp_checkpoint: Path | None, device: torch.device | None = None):
    """Build + cache the TNP model ahead of time (e.g. at server startup).

    Returns the cached model, or None if no usable checkpoint. Subsequent
    ``run_voice`` calls then skip the ~5s model build (B0).
    """
    if tnp_checkpoint is None or not Path(tnp_checkpoint).exists():
        return None
    ensure_tnp_on_path()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _get_model(Path(tnp_checkpoint), device)


def _get_model(tnp_checkpoint: Path, device: torch.device):
    """Build + load VoiceConversionModel once, then return the cached instance."""
    key = (str(Path(tnp_checkpoint).resolve()), str(device))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        logger.info("TNP 모델 캐시 적중 — 재로드 생략")
        return cached

    from core.model import VoiceConversionModel  # vendored tnp

    with timed("B0 TNP 모델 로드 (최초 1회, 이후 캐시)"):
        model = VoiceConversionModel(device=device)
        ckpt = torch.load(str(tnp_checkpoint), map_location=device, weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        model.eval()

    _MODEL_CACHE[key] = model
    return model


def run_voice(
    input_wav: Path,
    out_vocal: Path,
    tnp_checkpoint: Path | None = None,
    references=None,
    mel_sharpen: bool = MEL_SHARPEN,
    sharpen_t_alpha: float = MEL_SHARPEN_T_ALPHA,
    sharpen_f_alpha: float = MEL_SHARPEN_F_ALPHA,
) -> dict:
    """Convert ``input_wav`` to the target speaker → ``out_vocal``.

    ``mel_sharpen`` / ``sharpen_*_alpha`` control the inference-time mel
    sharpening band-aid (see module constants); applied per call so the cached
    model can be re-tuned without rebuilding.

    Returns ``{"vocal_wav": Path, "vc_applied": bool, "reason": str|None}``.
    """
    input_wav = Path(input_wav)
    out_vocal = Path(out_vocal)
    out_vocal.parent.mkdir(parents=True, exist_ok=True)

    refs = _refs_ok(references)
    ckpt_ok = tnp_checkpoint is not None and Path(tnp_checkpoint).exists()

    # ---- Fallback guard: missing checkpoint or reference --------------------
    if not ckpt_ok or not refs:
        reason = "TNP 체크포인트 없음" if not ckpt_ok else "목표 화자 reference WAV 없음"
        logger.warning(f"음성 변환 건너뜀 ({reason}) — 입력 WAV를 보컬 스템으로 사용합니다.")
        shutil.copy2(input_wav, out_vocal)
        return {"vocal_wav": out_vocal, "vc_applied": False, "reason": reason}

    # ---- Run TNP voice conversion (cached model) ----------------------------
    ensure_tnp_on_path()
    try:
        from convert import load_audio, extract_f0_stats  # vendored tnp helpers
        import soundfile as sf
    except Exception as e:  # noqa: BLE001
        logger.exception("TNP 모듈 import 실패")
        shutil.copy2(input_wav, out_vocal)
        return {"vocal_wav": out_vocal, "vc_applied": False, "reason": f"import 실패: {e}"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"TNP 음성 변환 — device={device} refs={len(refs)} ckpt={tnp_checkpoint}")

    try:
        model = _get_model(Path(tnp_checkpoint), device)

        # Inference-time mel sharpening (band-aid for predicted-mel over-smoothing).
        model.MEL_SHARPEN = mel_sharpen
        model.MEL_SHARPEN_T_ALPHA = sharpen_t_alpha
        model.MEL_SHARPEN_F_ALPHA = sharpen_f_alpha
        if mel_sharpen:
            logger.info(f"mel 샤프닝 ON (t={sharpen_t_alpha}, f={sharpen_f_alpha})")

        with timed("B1 화자 컨텍스트 계산 (reference)"):
            ref_audios = [load_audio(str(r), device) for r in refs]
            C = model.compute_context(ref_audios)
            tgt_f0 = extract_f0_stats(torch.cat(ref_audios, dim=-1), model)

        with timed("B2 음성 변환 (source→target, 청크 처리)"):
            source = load_audio(str(input_wav), device)
            T_total = source.shape[-1]
            src_f0 = extract_f0_stats(source, model)
            f0_stats = ((src_f0[0], src_f0[1], tgt_f0[0], tgt_f0[1])
                        if (src_f0 and tgt_f0) else None)

            denoised = model.content_encoder._denoise(source)
            chunks = []
            for start in range(0, T_total, CHUNK_SAMPLES):
                end = min(start + CHUNK_SAMPLES, T_total)
                chunk = denoised[:, start:end]
                if chunk.shape[-1] < 400:  # HuBERT needs ≥ 1 frame
                    break
                wav_out = model.convert_chunk_streaming(chunk, C, f0_stats=f0_stats)
                chunks.append(wav_out.squeeze(0))

            if not chunks:
                raise RuntimeError("출력이 없습니다 (입력이 너무 짧음).")

            out = torch.cat(chunks, dim=-1).cpu().clamp(-1.0, 1.0)
            sf.write(str(out_vocal), out[0].numpy(), VOCODER_SR)
    except Exception as e:  # noqa: BLE001
        logger.exception("TNP 변환 실패 — 입력 WAV로 폴백합니다.")
        shutil.copy2(input_wav, out_vocal)
        return {"vocal_wav": out_vocal, "vc_applied": False, "reason": f"변환 실패: {e}"}

    return {"vocal_wav": out_vocal, "vc_applied": True, "reason": None}
