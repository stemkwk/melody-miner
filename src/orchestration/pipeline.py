"""Full orchestration: input WAV → (Branch A ∥ Branch B) → mixed WAV.

Output layout under ``out_dir``:
    01_input.wav           입력 사본
    02_melody.mid          전사된 멜로디 MIDI
    03_accompaniment.mid   생성 반주 MIDI
    03_accompaniment.wav   생성 반주 WAV (사운드폰트 있을 때)
    04_vocal.wav           변환 보컬 (또는 VC 미적용 시 입력 폴백)
    05_final_mix.wav       ← 최종 결과물 (보컬 + 반주)
"""
from __future__ import annotations

import shutil
from pathlib import Path

from m2a_transformer.utils.logger import logger

from .accompaniment import M2aModel, load_m2a, run_accompaniment
from .config import GenParams, OrchestrationConfig, MixParams, VoiceParams
from .mixing import mix_stems
from .timing import timed
from .voice import run_voice


MODES = ("full", "m2a", "tnp")


def run_full(
    input_wav: Path,
    out_dir: Path,
    config: OrchestrationConfig,
    model: M2aModel | None = None,
    mode: str = "full",
) -> dict:
    """Run the pipeline in one of three modes.

    mode="full"  Branch A (반주) + Branch B (음성변환) → 믹스(변환보컬 + 반주)
    mode="m2a"   Branch A only - 반주 생성 → 믹스(원본 입력 + 반주). 음성변환 X
    mode="tnp"   Branch B only - 음성 변환된 보컬만. 반주/믹스 X

    ``model`` (preloaded M2A) is reused when given; loaded on demand only when
    the mode needs it (full/m2a). Returns a status dict with all output paths
    (unused stages are None).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    input_wav = Path(input_wav)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Input may be audio (WAV/…) or a MIDI file (m2a mode only).
    is_midi_input = input_wav.suffix.lower() in (".mid", ".midi")

    # Preserve the input alongside the outputs (keep its extension).
    input_copy = out_dir / ("01_input" + input_wav.suffix.lower())
    if Path(input_copy).resolve() != input_wav.resolve():
        shutil.copy2(input_wav, input_copy)

    need_m2a = mode in ("full", "m2a")
    need_tnp = mode in ("full", "tnp")

    if need_m2a and model is None:
        with timed("M2A 체크포인트 로드"):
            model = load_m2a(config.m2a_checkpoint, config.m2a_config)

    import time
    _t0 = time.perf_counter()
    logger.info(f"파이프라인 모드: {mode}")

    acc: dict | None = None
    voc: dict | None = None

    # ---- Branch A: accompaniment -------------------------------------------
    if need_m2a:
        with timed("Branch A - 반주 생성"):
            acc = run_accompaniment(input_wav, model, out_dir, config.gen)

    # ---- Branch B: voice conversion ----------------------------------------
    vocal_post = None
    if need_tnp:
        with timed("Branch B - 음성 변환"):
            voc = run_voice(
                input_wav, out_dir / "04_vocal.wav",
                tnp_checkpoint=config.tnp_checkpoint,
                references=config.references,
                mel_sharpen=config.voice.mel_sharpen,
                sharpen_t_alpha=config.voice.sharpen_t_alpha,
                sharpen_f_alpha=config.voice.sharpen_f_alpha,
            )
        # Vocal post-processing → SEPARATE file (keep raw 04_vocal.wav for A/B).
        if voc and voc.get("vc_applied") and config.mix.postprocess_vocal:
            from .mixing import postprocess_vocal_file
            vocal_post = postprocess_vocal_file(
                voc["vocal_wav"], out_dir / "04_vocal_post.wav",
                saturation_db=config.mix.vocal_saturation_db)
            if vocal_post is not None:
                logger.info(f"보컬 후처리(EQ/Reverb) → {vocal_post}")

    # ---- Merge --------------------------------------------------------------
    # full → 변환보컬(후처리본 우선) + 반주 / m2a → 원본입력(또는 렌더된 멜로디) + 반주 / tnp → 믹스 없음
    final_mix = None
    accomp_wav = acc["accomp_wav"] if acc else None
    if accomp_wav is not None:
        if voc is not None:
            vocal_stem = vocal_post or voc["vocal_wav"]   # full: 후처리본 우선
        elif not is_midi_input:
            vocal_stem = input_copy                    # m2a + 오디오 입력: 원본 WAV
        else:
            vocal_stem = acc["melody_wav"]             # m2a + MIDI 입력: 렌더된 멜로디
        if vocal_stem is None:
            logger.warning("믹스용 보컬 스템이 없어 믹스를 건너뜁니다.")
            accomp_wav_for_mix = None
        else:
            accomp_wav_for_mix = accomp_wav
    else:
        accomp_wav_for_mix = None

    if accomp_wav_for_mix is not None:
        with timed("Merge - 믹스"):
            final_mix = mix_stems(
                vocal_stem, accomp_wav, out_dir / "05_final_mix.wav",
                target_sr=config.mix.target_sr,
                vocal_gain=config.mix.vocal_gain,
                accomp_gain=config.mix.accomp_gain,
                peak_ceiling=config.mix.peak_ceiling,
                normalize_stems=config.mix.normalize_stems,
            )
        logger.info(f"최종 믹스 생성: {final_mix}")
    elif need_m2a:
        logger.warning("반주 WAV가 없어(사운드폰트 부재) 믹스를 건너뜁니다.")

    logger.info(f"■ 전체 파이프라인 완료 ({time.perf_counter() - _t0:.2f}s)")

    return {
        "mode": mode,
        "input": input_copy,
        "is_midi_input": is_midi_input,
        "melody_wav": acc["melody_wav"] if acc else None,
        "melody_midi": acc["melody_midi"] if acc else None,
        "accomp_midi": acc["accomp_midi"] if acc else None,
        "accomp_wav": accomp_wav,
        "vocal_wav": voc["vocal_wav"] if voc else None,
        "vocal_post": vocal_post,
        "vc_applied": voc["vc_applied"] if voc else False,
        "vc_reason": voc["reason"] if voc else None,
        "final_mix": final_mix,
        "tempo": acc["tempo"] if acc else None,
        "out_dir": out_dir,
    }


__all__ = ["run_full", "load_m2a", "M2aModel", "OrchestrationConfig",
           "GenParams", "VoiceParams", "MixParams"]
