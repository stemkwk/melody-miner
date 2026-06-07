"""Chord and key extraction from MIDI files."""
from __future__ import annotations

from pathlib import Path

from m2a_transformer.utils.logger import logger

_CHORD_TEMPLATES: dict[int, frozenset[int]] = {
    0:  frozenset([0, 4, 7]),           # maj
    1:  frozenset([0, 3, 7]),           # min
    2:  frozenset([0, 4, 7, 10]),       # dom7
    3:  frozenset([0, 4, 7, 11]),       # maj7
    4:  frozenset([0, 3, 7, 10]),       # min7
    5:  frozenset([0, 3, 6]),           # dim
    6:  frozenset([0, 4, 8]),           # aug
    7:  frozenset([0, 2, 4, 7]),        # add9  (sus2 collapses here)
    8:  frozenset([0, 5, 7]),           # sus4
    9:  frozenset([0, 3, 6, 9]),        # dim7
    10: frozenset([0, 3, 6, 10]),       # hdim7
    11: frozenset([0, 4, 7, 10, 2]),    # dom9
}

_MATCH_THRESHOLD = 0.75  # default; overridden per-call via chord_match_threshold


def _match_chord(
    pitch_classes: frozenset[int],
    n_qualities: int,
    chord_match_threshold: float = _MATCH_THRESHOLD,
) -> "tuple[int, int] | None":
    """Return (root_0_11, quality_idx) for the best-matching chord above threshold.

    Returns None when no candidate reaches chord_match_threshold.
    n_qualities: 9 for Lakh (core), 12 for Slakh (full).
    """
    if len(pitch_classes) < 2:
        return None

    best_score = chord_match_threshold - 1e-9
    best: "tuple[int, int] | None" = None

    for root in range(12):
        relative = frozenset((p - root) % 12 for p in pitch_classes)
        for q_idx, template in _CHORD_TEMPLATES.items():
            if q_idx >= n_qualities:
                continue
            score = len(relative & template) / len(template)
            if score > best_score:
                best_score = score
                best = (root, q_idx)

    return best


def _extract_chords_from_midi(
    midi_path: Path,
    cfg,
    n_qualities: int,
    chord_match_threshold: float = _MATCH_THRESHOLD,
) -> "dict[tuple[int, int], tuple[int, int] | None]":
    """Beat-level chord extraction → (bar, pos) chord map.

    Only positions with a confident match (score ≥ _MATCH_THRESHOLD) are added.
    """
    try:
        import miditoolkit
        midi = miditoolkit.MidiFile(str(midi_path))
    except Exception as exc:
        logger.debug(f"Chord extraction failed for {midi_path.name}: {exc}")
        return {}

    tpb = midi.ticks_per_beat
    all_notes: list[tuple[int, int, int]] = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            all_notes.append((n.start, n.end, n.pitch))

    if not all_notes:
        return {}

    max_beat      = int(max(n[1] for n in all_notes) / tpb) + 1
    beats_per_bar = 4

    notes_by_start: list[tuple[int, int, int]] = sorted(all_notes, key=lambda n: n[0])
    active: list[tuple[int, int, int]] = []
    ptr = 0
    n_total = len(notes_by_start)

    chord_map: dict[tuple[int, int], tuple[int, int] | None] = {}
    for beat in range(max_beat):
        beat_start = beat * tpb
        beat_end   = (beat + 1) * tpb

        while ptr < n_total and notes_by_start[ptr][0] < beat_end:
            active.append(notes_by_start[ptr])
            ptr += 1

        active = [n for n in active if n[1] > beat_start]

        active_pcs = frozenset(n[2] % 12 for n in active)
        result = _match_chord(active_pcs, n_qualities, chord_match_threshold)
        if result is not None:
            bar = beat // beats_per_bar
            pos = (beat % beats_per_bar) * cfg.resolution
            chord_map[(bar, pos)] = result

    return chord_map


def _estimate_key_from_midi(midi_path: Path) -> "tuple[int, int] | None":
    """Estimate key via pitch-class profile (Krumhansl-Schmuckler heuristic).

    Returns (root_0_11, mode) or None on failure.
    """
    try:
        import miditoolkit
        midi = miditoolkit.MidiFile(str(midi_path))
    except Exception:
        return None

    pc_count = [0.0] * 12
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            pc_count[n.pitch % 12] += (n.end - n.start)

    total = sum(pc_count)
    if total <= 0:
        return None
    pc = [x / total for x in pc_count]

    major = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    def _corr(profile: list[float], shift: int) -> float:
        rotated = profile[shift:] + profile[:shift]
        mean_p = sum(pc) / 12
        mean_r = sum(rotated) / 12
        num = sum((pc[i] - mean_p) * (rotated[i] - mean_r) for i in range(12))
        dp  = sum((pc[i] - mean_p) ** 2 for i in range(12)) ** 0.5
        dr  = sum((rotated[i] - mean_r) ** 2 for i in range(12)) ** 0.5
        if dp * dr < 1e-9:
            return 0.0
        return num / (dp * dr)

    best_corr  = -999.0
    best_root  = 0
    best_mode  = 0
    for root in range(12):
        for mode, profile in enumerate([major, minor]):
            c = _corr(profile, root)
            if c > best_corr:
                best_corr = c
                best_root = root
                best_mode = mode

    return (best_root, best_mode)
