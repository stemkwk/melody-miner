"""Stem mixing — combine a full vocal take with the generated accompaniment.

Unlike basic looping strategies (which *loops* a short melody
phrase to cover the accompaniment), here both stems are full-length takes that
derive from the SAME input WAV and therefore already share the input's t=0
timeline. So we only need: sample-rate alignment + zero-pad to the longer
length + a weighted sum. No looping, no time-warping.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

try:
    from pedalboard import Pedalboard, Reverb, HighShelfFilter, Distortion
    HAS_PEDALBOARD = True
except ImportError:
    HAS_PEDALBOARD = False


def to_f32(arr: np.ndarray) -> np.ndarray:
    """Normalise any integer/float WAV array to float32 in [-1, 1].

    Ported utility — scipy.io.wavfile.read
    returns different dtypes by bit depth, and a fixed int16 divisor would clip
    int32 audio or near-silence uint8 audio.
    """
    if arr.dtype == np.uint8:
        return (arr.astype(np.float32) - 128.0) / 128.0
    if np.issubdtype(arr.dtype, np.signedinteger):
        return arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    return arr.astype(np.float32)


def _read_mono_f32(path: Path) -> tuple[int, np.ndarray]:
    sr, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)  # mix to mono
    return sr, to_f32(data)


def _resample_to(sig: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return sig
    return resample(sig, int(round(len(sig) * sr_to / sr_from)))


def apply_vocal_postprocessing(vocal: np.ndarray, sr: int,
                               saturation_db: float = 0.0) -> np.ndarray:
    """Apply optional saturation → high-shelf air boost → light reverb.

    ``saturation_db`` (>0) drives a tanh saturation that adds harmonics — a
    band-aid for the predicted-mel dullness ("물먹은"): the extra harmonics add
    perceived brightness/presence and psychoacoustically mask the missing
    high-frequency detail. It is memoryless, so it does NOT fix temporal wobble.
    0 = off.

    Reverb is kept light (wet 0.15) because the accompaniment already carries its
    own reverb from the M2A DSP chain — stacking two heavy reverbs muddies the mix.

    Built per call (not cached) so ``saturation_db`` can be tuned between runs.
    """
    if not HAS_PEDALBOARD:
        from m2a_transformer.utils.logger import logger
        logger.warning("pedalboard not installed. Skipping vocal post-processing.")
        return vocal

    # pedalboard expects shape (channels, samples)
    if vocal.ndim == 1:
        vocal = vocal.reshape(1, -1)

    effects = []
    if saturation_db and saturation_db > 0.0:
        effects.append(Distortion(drive_db=float(saturation_db)))  # tanh waveshaper
    effects.append(HighShelfFilter(cutoff_frequency_hz=4000.0, gain_db=5.0))
    effects.append(Reverb(room_size=0.35, damping=0.5, wet_level=0.15, dry_level=1.0))

    processed = Pedalboard(effects)(vocal, sr)
    return processed[0]  # back to 1D


def postprocess_vocal_file(in_path: Path, out_path: Path,
                           saturation_db: float = 0.0) -> Path | None:
    """Read a vocal WAV, apply post-processing, write to ``out_path``.

    Keeps the raw input file intact (writes a SEPARATE file) so you can A/B the
    converted vocal vs. the post-processed one. Returns ``out_path`` or None if
    pedalboard is unavailable.
    """
    if not HAS_PEDALBOARD:
        return None
    sr, y = _read_mono_f32(Path(in_path))
    y_proc = apply_vocal_postprocessing(y, sr, saturation_db=saturation_db)
    
    # [FIX] 하드 클리핑 및 볼륨 밸런스 붕괴 방지:
    # EQ(HighShelf) 등으로 인해 피크가 1.0을 넘은 상태에서 np.clip만 적용하면
    # 깎여나간 피크로 인해 평균 볼륨(RMS)이 비정상적으로 커져 반주와의 믹스 밸런스가 무너짐.
    y_proc = _normalize_peak(y_proc, 0.99)
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(out_path), sr, (np.clip(y_proc, -1.0, 1.0) * 32767).astype(np.int16))
    return out_path


def _normalize_peak(sig: np.ndarray, target: float) -> np.ndarray:
    """Scale a signal so its peak hits ``target`` (no-op if silent)."""
    peak = float(np.max(np.abs(sig))) if len(sig) else 0.0
    if peak > 1e-6:
        return sig * (target / peak)
    return sig


def mix_stems(
    vocal_wav: Path,
    accomp_wav: Path,
    out_path: Path,
    target_sr: int = 44_100,
    vocal_gain: float = 0.8,
    accomp_gain: float = 0.6,
    peak_ceiling: float = 0.99,
    normalize_stems: bool = True,
    stem_peak: float = 0.95,
) -> Path:
    """Mix vocal + accompaniment WAVs into one int16 WAV at ``target_sr``.

    Both stems are resampled to ``target_sr`` and zero-padded to the longer
    length. When ``normalize_stems`` is True, each stem is first peak-normalised
    to ``stem_peak`` so wildly different source levels (e.g. a quiet single-line
    violin melody render vs. a loud compressed accompaniment, or a raw vocal)
    don't drown each other out — the gains then set the *actual* balance. Without
    this, a quiet melody stem is inaudible under the accompaniment.

    The summed mix is peak-normalised to ``peak_ceiling`` (scale, not clip) if it
    overshoots; a final clip is only a safety net. Returns ``out_path``.
    """
    sr_v, vocal = _read_mono_f32(Path(vocal_wav))
    sr_a, accomp = _read_mono_f32(Path(accomp_wav))

    vocal = _resample_to(vocal, sr_v, target_sr)
    accomp = _resample_to(accomp, sr_a, target_sr)

    n = max(len(vocal), len(accomp), 1)
    vocal = np.pad(vocal, (0, n - len(vocal)))
    accomp = np.pad(accomp, (0, n - len(accomp)))

    if normalize_stems:
        vocal = _normalize_peak(vocal, stem_peak)
        accomp = _normalize_peak(accomp, stem_peak)

    mixed = vocal * vocal_gain + accomp * accomp_gain

    # Peak-normalise to avoid clipping (scale, don't clip).
    peak = float(np.max(np.abs(mixed))) if n else 0.0
    if peak > peak_ceiling:
        mixed = mixed * (peak_ceiling / peak)
    mixed = np.clip(mixed, -1.0, 1.0)  # safety net only

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(out_path), target_sr, (mixed * 32767).astype(np.int16))
    return out_path
