"""Monophonic WAV→MIDI transcription via CREPE (torchcrepe) + note segmentation.

basic-pitch is a *polyphonic* AMT model: on a solo vocal it fragments a single
sung note into many short notes whenever vibrato/bends cross a semitone boundary.
For solo voice the right tool is a monophonic pitch tracker + note segmentation:

    audio → CREPE f0 + periodicity → median-smooth → semitone-quantise
          → segment (min duration + short-gap bridging) → notes

Median smoothing + semitone rounding absorb vibrato and slow bends into a single
held note, so the melody fed to M2A is clean. torchcrepe ships with the project
(TNP uses it), so no new dependency.
"""
from __future__ import annotations

from pathlib import Path

from m2a_transformer.utils.logger import logger


def _segment(pitch_int, frame_dt: float, min_note_ms: float, max_gap_ms: float):
    """Run-length segment a per-frame MIDI-pitch array (-1 = unvoiced) into notes.

    Short unvoiced gaps (≤ ``max_gap_ms``) inside an otherwise steady pitch are
    bridged (vibrato/consonant dropouts); notes shorter than ``min_note_ms`` are
    dropped. Returns a list of (pitch, start_frame, end_frame_exclusive).
    """
    max_gap = int(round((max_gap_ms / 1000.0) / frame_dt))
    min_len = max(1, int(round((min_note_ms / 1000.0) / frame_dt)))
    n = len(pitch_int)
    notes = []
    cur_pitch = None
    cur_start = 0
    gap = 0
    for i in range(n):
        p = int(pitch_int[i])
        if cur_pitch is None:
            if p >= 0:
                cur_pitch, cur_start, gap = p, i, 0
        elif p == cur_pitch:
            gap = 0
        elif p < 0:                      # unvoiced — maybe a short bridgeable gap
            gap += 1
            if gap > max_gap:
                notes.append((cur_pitch, cur_start, i - gap + 1))
                cur_pitch, gap = None, 0
        else:                            # a different stable pitch → new note
            notes.append((cur_pitch, cur_start, i - gap))
            cur_pitch, cur_start, gap = p, i, 0
    if cur_pitch is not None:
        notes.append((cur_pitch, cur_start, n - gap))
    return [nt for nt in notes if (nt[2] - nt[1]) >= min_len]


def transcribe_crepe(
    audio_path: Path,
    out_midi: Path,
    *,
    fmin: float = 65.0,
    fmax: float = 1000.0,
    periodicity_threshold: float = 0.5,
    min_note_ms: float = 100.0,
    max_gap_ms: float = 80.0,
    smooth_frames: int = 5,
    hop_length: int = 160,           # 10 ms @ 16 kHz
    model: str = "full",
    program: int = 40,               # violin (matches melody track program)
    tempo_bpm: float = 120.0,
    device: str | None = None,
) -> Path:
    """Transcribe a monophonic audio file to a single-track melody MIDI."""
    import numpy as np
    import torch
    import torchcrepe
    import librosa
    import pretty_midi
    from scipy.ndimage import median_filter

    sr = 16_000
    audio, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    audio_t = torch.from_numpy(audio).float()[None]  # [1, samples]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pitch, periodicity = torchcrepe.predict(
        audio_t, sr, hop_length, fmin, fmax,
        model=model, return_periodicity=True,
        batch_size=512, device=device,
    )
    pitch = pitch[0].cpu().numpy()          # [frames] Hz
    periodicity = periodicity[0].cpu().numpy()

    # Voiced gate + vibrato/bend smoothing → semitone-quantised contour.
    periodicity = median_filter(periodicity, size=3)
    voiced = (periodicity > periodicity_threshold) & (pitch > 0)
    midi_vals = np.zeros_like(pitch)
    midi_vals[pitch > 0] = 69.0 + 12.0 * np.log2(pitch[pitch > 0] / 440.0)
    midi_s = median_filter(midi_vals, size=smooth_frames)
    pitch_int = np.where(voiced, np.clip(np.round(midi_s), 0, 127).astype(int), -1)

    frame_dt = hop_length / sr
    notes = _segment(pitch_int, frame_dt, min_note_ms, max_gap_ms)

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm)
    inst = pretty_midi.Instrument(program=program, name="MELODY")
    for p, s, e in notes:
        inst.notes.append(pretty_midi.Note(
            velocity=90, pitch=int(p), start=s * frame_dt, end=e * frame_dt))
    pm.instruments.append(inst)
    out_midi = Path(out_midi)
    out_midi.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_midi))
    logger.info(
        f"CREPE 전사: {len(notes)} notes (model={model}, "
        f"voiced {voiced.mean()*100:.0f}%) → {out_midi}")
    return out_midi
