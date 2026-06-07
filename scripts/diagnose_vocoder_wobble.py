"""음성 변환의 '물먹은/흔들리는' 음질 저하 원인 확정 진단.

지금까지의 청취 결과로 좁혀진 사실:
  • copysynth (진짜 mel → Vocos)        = 깨끗  → Vocos(A) 무죄
  • F0 디코딩 viterbi/argmax 교체         = 차이 없음 → F0(B) 무죄
  • convert (TNP 예측 mel → Vocos)       = 물먹음/흔들림

남는 원인은 하나 — **TNP가 예측한 mel 자체(C)**. "물먹은 소리"는 regression
(L1/L2)으로 학습한 mel 예측기의 과평활(over-smoothing) 전형이다.

이 스크립트는 그걸 '귀'가 아니라 '눈'으로 확정한다. **같은 입력 보컬**에 대해
  (1) 입력 자신의 진짜 log-mel        — 깨끗/또렷 (기준)
  (2) TNP 가 예측한 변환 log-mel      — 흐릿하면 C 확정
을 같은 스케일로 나란히 그리고, 디테일(분산/프레임간 변화량) 수치를 찍는다.
음색은 달라도(소스 화자 → 타깃 화자) 내용·운율이 같으므로 '선명도' 비교는 유효.

reference(짧은 2초)가 아니라 input/ 의 실제 긴 보컬을 --source 로 주는 것이 핵심.

출력 (output/diag/):
  copysynth_input.wav    입력 → 진짜 mel → Vocos (대조: Vocos 단독 = 깨끗해야 정상)
  convert.wav            입력 → 변환 (현 파이프라인, f0 시프트 ON)
  mel_real_input.png     입력 자신의 진짜 log-mel
  mel_pred_convert.png   TNP 예측 변환 log-mel  ← 흐릿하면 C
  mel_compare.png        둘을 같은 스케일로 위아래 비교

재학습·가중치 수정 없음. 단일 패스(청크 없음). 긴 파일은 --max-seconds 로 트림.

사용:
  python scripts/diagnose_vocoder_wobble.py \
      --source "input/pop-female-vocal-clean-melody_144bpm_A_major.wav" \
      --reference references/context.wav \
      --max-seconds 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from orchestration import ensure_tnp_on_path  # noqa: E402

ensure_tnp_on_path()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from convert import load_audio, extract_f0_stats, SAMPLE_RATE  # noqa: E402  (vendored tnp)
from core.model import VoiceConversionModel  # noqa: E402

VOCODER_SR = 24_000


def _save_wav(wav: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = wav.detach().cpu().squeeze().clamp(-1.0, 1.0).numpy()
    sf.write(str(path), arr, VOCODER_SR)
    print(f"  ✔ {path.name}")


def _mel_img(mel_cl: torch.Tensor) -> np.ndarray:
    """[1, T_mel, N_MELS] log-mel → [N_MELS, T] numpy for imshow."""
    return mel_cl.detach().cpu().squeeze(0).numpy().T


def _detail_stats(mel_cl: torch.Tensor) -> tuple[float, float]:
    """과평활 지표: (mel 빈 방향 분산 평균, 프레임간 변화량 평균).

    값이 작을수록 흐릿/평활. 진짜 mel 대비 예측 mel 이 크게 낮으면 over-smoothing.
    """
    m = mel_cl.detach().cpu().squeeze(0)                    # [T, N_MELS]
    spectral_contrast = m.var(dim=-1).mean().item()        # 프레임 내 주파수 디테일
    temporal_delta = (m[1:] - m[:-1]).abs().mean().item()  # 프레임간 변화량
    return spectral_contrast, temporal_delta


@torch.no_grad()
def predict_convert_mel(model, src16k, C, f0_stats) -> torch.Tensor:
    """변환 경로에서 Vocos 직전의 TNP 예측 mel 만 뽑는다. [1, T_mel, N_MELS]."""
    ce = model.content_encoder
    denoised = ce._denoise(src16k)
    content = ce(denoised, skip_denoise=True, f0_stats=f0_stats)
    return model.tnp(C, content)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tnp-checkpoint", default="checkpoints/tnp/latest.pt")
    ap.add_argument("--reference", nargs="+", default=["references/context.wav"],
                    help="목표 화자 reference WAV(들) — compute_context 용")
    ap.add_argument("--source", default="references/source.wav",
                    help="변환할 입력 보컬. input/ 의 긴 실제 보컬을 권장.")
    ap.add_argument("--max-seconds", type=float, default=20.0,
                    help="긴 입력 트림 상한(초). 단일 패스 OOM 방지. 0=무제한.")
    ap.add_argument("--out", default="output/diag")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  source={Path(args.source).name}  out={out_dir}\n")

    print("[load] TNP 모델 빌드 + 체크포인트 로드 …")
    model = VoiceConversionModel(device=device)
    ckpt = torch.load(args.tnp_checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    src = load_audio(args.source, device)                  # [1, T] @16k
    if args.max_seconds and args.max_seconds > 0:
        cap = int(args.max_seconds * SAMPLE_RATE)
        if src.shape[-1] > cap:
            print(f"      입력 {src.shape[-1] / SAMPLE_RATE:.1f}s → {args.max_seconds:.0f}s 로 트림")
            src = src[:, :cap]
    refs = [load_audio(r, device) for r in args.reference]

    # ── 컨텍스트 / F0 통계 ───────────────────────────────────────────────────
    C = model.compute_context(refs)
    src_f0 = extract_f0_stats(src, model)
    tgt_f0 = extract_f0_stats(torch.cat(refs, dim=-1), model)
    f0_stats = ((src_f0[0], src_f0[1], tgt_f0[0], tgt_f0[1])
                if (src_f0 and tgt_f0) else None)

    # ── Vocos 단독 대조군 ────────────────────────────────────────────────────
    print("\n[wav] copysynth(input) — Vocos 단독 대조군:")
    real_mel_cf = model._compute_mel_channels_first(src)   # [1, 100, T_mel]
    _save_wav(model.vocoder(real_mel_cf), out_dir / "copysynth_input.wav")

    # ── mel 비교: 입력 자신의 진짜 mel  vs  TNP 예측 변환 mel (같은 내용) ─────
    print("\n[mel] 스펙트로그램 추출 + 비교 …")
    real_mel = model._compute_mel(src)                      # [1, T_mel, 100] log
    pred_mel = predict_convert_mel(model, src, C, f0_stats)  # [1, T_mel, 100] log

    rc, rd = _detail_stats(real_mel)
    pc, pd = _detail_stats(pred_mel)
    print(f"      진짜 mel(입력) : 주파수디테일(var)={rc:.4f}  프레임간변화={rd:.4f}")
    print(f"      예측 mel(변환) : 주파수디테일(var)={pc:.4f}  프레임간변화={pd:.4f}")
    print(f"      비율 예측/진짜  : 디테일={pc / rc:.2f}×  변화={pd / rd:.2f}×  (<<1.0 이면 과평활)")

    ri, pi = _mel_img(real_mel), _mel_img(pred_mel)
    vmin = float(min(ri.min(), pi.min()))
    vmax = float(max(ri.max(), pi.max()))

    for img, name, title in [(ri, "mel_real_input.png", "REAL input log-mel"),
                             (pi, "mel_pred_convert.png", "PRED convert log-mel")]:
        plt.figure(figsize=(11, 3))
        plt.imshow(img, origin="lower", aspect="auto", interpolation="nearest",
                   vmin=vmin, vmax=vmax)
        plt.title(title); plt.xlabel("frame"); plt.ylabel("mel bin"); plt.colorbar()
        plt.tight_layout(); plt.savefig(out_dir / name, dpi=120); plt.close()
        print(f"  ✔ {name}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    for ax, img, title in [(axes[0], ri, "REAL input (clean → Vocos = OK)"),
                           (axes[1], pi, "PRED convert (← 흐릿하면 C: mel 과평활)")]:
        im = ax.imshow(img, origin="lower", aspect="auto", interpolation="nearest",
                       vmin=vmin, vmax=vmax)
        ax.set_title(title); ax.set_ylabel("mel bin")
        fig.colorbar(im, ax=ax)
    axes[1].set_xlabel("frame")
    plt.tight_layout(); plt.savefig(out_dir / "mel_compare.png", dpi=120); plt.close()
    print("  ✔ mel_compare.png")

    # ── mel 샤프닝 스윕: 예측 mel 을 unsharp 로 또렷하게 → A/B ───────────────
    # 목표: 디테일 비율을 1.0 쪽으로 끌어올린다. 단 1.0 을 넘기면 과샤프닝(거칠어짐).
    print("\n[sharp] mel 샤프닝 스윕 — convert_sharp_*.wav 로 A/B:")
    print(f"        baseline(off)            : 디테일={pc / rc:.2f}× 변화={pd / rd:.2f}×")
    sweep = [
        (0.0, 0.0),   # off (baseline)
        (1.0, 0.5),
        (1.5, 0.7),
        (2.5, 1.0),
    ]
    for ta, fa in sweep:
        model.MEL_SHARPEN = (ta > 0 or fa > 0)
        model.MEL_SHARPEN_T_ALPHA = ta
        model.MEL_SHARPEN_F_ALPHA = fa
        sharp_mel = model._sharpen_mel(pred_mel)
        sc, sd = _detail_stats(sharp_mel)
        wav = model.vocoder(sharp_mel.transpose(1, 2))
        tag = "off" if model.MEL_SHARPEN is False else f"t{ta}_f{fa}"
        _save_wav(wav, out_dir / f"convert_sharp_{tag}.wav")
        if model.MEL_SHARPEN:
            print(f"        t_alpha={ta:<4} f_alpha={fa:<4}: 디테일={sc / rc:.2f}× 변화={sd / rd:.2f}×")
    model.MEL_SHARPEN = False  # restore

    print("\n=== 판독 ===")
    print("  • convert_sharp_*.wav 들을 듣고, 디테일 비율이 ~1.0 에 가까우면서")
    print("    거칠어지지 않는 마지막 설정을 고른다.")
    print("  • 고른 t_alpha/f_alpha 를 voice.py 의 MEL_SHARPEN 블록에 반영하면")
    print("    실제 파이프라인(run.py)에 적용됨.")


if __name__ == "__main__":
    main()
