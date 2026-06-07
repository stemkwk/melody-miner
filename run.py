"""melody-miner CLI — WAV → accompaniment generation + voice conversion → one mix.

Examples
--------
# Branch A only (works even before TNP is ready — input vocal + generated accompaniment)
python run.py --input song.wav \
    --m2a-checkpoint "checkpoints/m2a/best-epoch=007-val_loss=0.8431.ckpt" \
    --out output/run1

# Full (after TNP checkpoint + target-speaker reference are ready)
python run.py --input song.wav \
    --m2a-checkpoint "checkpoints/m2a/best-epoch=007-val_loss=0.8431.ckpt" \
    --tnp-checkpoint checkpoints/tnp/best.pt \
    --reference references/target1.wav references/target2.wav \
    --out output/run2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `import orchestration` work when run from the repo root without install.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from orchestration import DEFAULT_CONFIG
from orchestration.config import GenParams, OrchestrationConfig, MixParams
from orchestration.pipeline import run_full


def build_config(args) -> OrchestrationConfig:
    gen = GenParams(
        temperature=args.temperature,
        top_p=args.top_p,
        cfg_w=args.cfg_w,
        avoid_note_penalty=args.avoid_note_penalty,
        denoise=args.denoise,
        cond_tracks=[t.strip() for t in args.cond_tracks.split(",") if t.strip()],
        transcriber=args.transcriber,
    )
    mix = MixParams(
        target_sr=args.target_sr,
        vocal_gain=args.vocal_gain,
        accomp_gain=args.accomp_gain,
        postprocess_vocal=not args.no_postprocess,
    )
    return OrchestrationConfig(
        m2a_checkpoint=args.m2a_checkpoint,
        m2a_config=args.config,
        tnp_checkpoint=args.tnp_checkpoint,
        references=args.reference or [],
        gen=gen,
        mix=mix,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="melody-miner: WAV → 반주+음성변환 → 믹스")
    p.add_argument("--input", required=True, help="입력 WAV (노래/허밍)")
    p.add_argument("--out", default="output/run", help="출력 폴더")
    p.add_argument("--mode", choices=["full", "m2a", "tnp"], default="full",
                   help="full=반주+음성변환 믹스 · m2a=반주만 · tnp=음성변환만")
    p.add_argument("--m2a-checkpoint", default=None,
                   help="M2A 반주 모델 체크포인트(.ckpt). full/m2a 모드에 필요.")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="M2A config.yaml")
    p.add_argument("--tnp-checkpoint", default=None,
                   help="TNP 음성변환 체크포인트(.pt). 생략 시 VC 건너뜀.")
    p.add_argument("--reference", nargs="+", default=None,
                   help="목표 화자 reference WAV(들). 생략 시 VC 건너뜀.")
    # generation params
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", dest="top_p", type=float, default=None)
    p.add_argument("--cfg-w", dest="cfg_w", type=float, default=0.0)
    p.add_argument("--avoid-note-penalty", dest="avoid_note_penalty",
                   type=float, default=None)
    p.add_argument("--transcriber", choices=["basic-pitch", "crepe"], default="basic-pitch",
                   help="WAV→MIDI 전사기. crepe=단성(솔로 보컬 분절 적음).")
    p.add_argument("--denoise", action="store_true", help="전사 전 노이즈 제거")
    p.add_argument("--cond-tracks", dest="cond_tracks", default="melody")
    # mixing params
    p.add_argument("--target-sr", dest="target_sr", type=int, default=44_100)
    p.add_argument("--vocal-gain", dest="vocal_gain", type=float, default=0.8)
    p.add_argument("--accomp-gain", dest="accomp_gain", type=float, default=0.6)
    p.add_argument("--no-postprocess", action="store_true", help="보컬 후처리(리버브/EQ) 비활성화")
    args = p.parse_args()

    if args.mode in ("full", "m2a") and not args.m2a_checkpoint:
        p.error(f"--mode {args.mode} 에는 --m2a-checkpoint 가 필요합니다.")
    if args.mode == "tnp" and not args.tnp_checkpoint:
        p.error("--mode tnp 에는 --tnp-checkpoint 가 필요합니다.")
    if not args.m2a_checkpoint:           # tnp-only: satisfy config dataclass
        args.m2a_checkpoint = "none"

    config = build_config(args)
    result = run_full(args.input, args.out, config, mode=args.mode)

    print("\n=== 완료 ===")
    print(f"모드      : {result['mode']}")
    print(f"출력 폴더 : {result['out_dir']}")
    if result["tempo"] is not None:
        print(f"템포      : {result['tempo']:.1f} BPM")
    if args.mode in ("full", "tnp"):
        print(f"음성 변환 : {'적용' if result['vc_applied'] else '건너뜀 (' + str(result['vc_reason']) + ')'}")
    print(f"보컬      : {result['vocal_wav']}")
    print(f"반주 WAV  : {result['accomp_wav']}")
    print(f"최종 믹스 : {result['final_mix']}")


if __name__ == "__main__":
    main()
