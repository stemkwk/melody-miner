"""POP909 dataset preprocessing: chord/key parsing and shard encoding."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from m2a_transformer.utils.logger import logger
from m2a_transformer.utils.midi_io import midi_to_events
from m2a_transformer.tokenizer import BaseTokenizer, REMITokenizer
from m2a_transformer.preprocessing.shards import _safe_name, _save_shard
from m2a_transformer.preprocessing.chords import _extract_chords_from_midi
from m2a_transformer.preprocessing.melody import _melody_coverage

# ---------------------------------------------------------------------------
# Harte notation tables
# ---------------------------------------------------------------------------

_HARTE_ROOT_TO_SEMITONE: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}

_HARTE_QUALITY_TO_IDX_FULL: dict[str, int] = {
    "maj": 0, "M": 0, "": 0,
    "min": 1, "m": 1,
    "7": 2, "dom7": 2,
    "maj7": 3, "M7": 3,
    "min7": 4, "m7": 4,
    "dim": 5, "o": 5,
    "aug": 6, "+": 6,
    "add9": 7, "2": 7, "sus2": 7,
    "sus4": 8, "sus": 8,
    "dim7": 9, "o7": 9,
    "hdim7": 10, "m7b5": 10,
    "9": 11, "dom9": 11,
}

_HARTE_QUALITY_TO_IDX_CORE: dict[str, int] = {
    "maj": 0, "M": 0, "": 0,
    "min": 1, "m": 1,
    "7": 2, "dom7": 2,
    "maj7": 3, "M7": 3,
    "min7": 4, "m7": 4,
    "dim": 5, "dim7": 5, "hdim7": 5, "o": 5, "o7": 5, "m7b5": 5,
    "aug": 6, "+": 6,
    "add9": 7, "2": 7, "sus2": 7,
    "sus4": 8, "sus": 8,
}

# ---------------------------------------------------------------------------
# Chord / key file parsers
# ---------------------------------------------------------------------------

def _parse_pop909_chord_file(
    chord_file: Path,
    n_qualities: int,
    midi_path: "Path | None" = None,
    tempo_bpm: float = 120.0,
) -> "dict[tuple[int, int], tuple[int, int] | None]":
    """Parse POP909 chord_midi.txt → (bar, pos) chord map."""
    import numpy as np

    chord_map: dict[tuple[int, int], tuple[int, int] | None] = {}
    try:
        lines = chord_file.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        logger.debug(f"Cannot read {chord_file}: {exc}")
        return chord_map

    beats_per_bar = 4
    resolution    = 4

    tick_to_time: "np.ndarray | None" = None
    tpb = 480
    if midi_path is not None:
        try:
            import miditoolkit
            _mid = miditoolkit.MidiFile(str(midi_path))
            tick_to_time = _mid.get_tick_to_time_mapping()
            tpb = _mid.ticks_per_beat
        except Exception:
            tick_to_time = None

    def _sec_to_bar_pos(sec: float) -> "tuple[int, int]":
        if tick_to_time is not None:
            tick = int(np.searchsorted(tick_to_time, sec, side="left"))
            tick = min(tick, len(tick_to_time) - 1)
            beat_idx = tick // tpb
        else:
            sec_per_beat = 60.0 / max(tempo_bpm, 1.0)
            beat_idx = round(sec / sec_per_beat)
        bar = beat_idx // beats_per_bar
        pos = (beat_idx % beats_per_bar) * resolution
        return bar, pos

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            start_sec = float(parts[0])
        except ValueError:
            continue
        chord_str = parts[2]
        bar, pos = _sec_to_bar_pos(start_sec)

        if chord_str in ("N", "X", ""):
            chord_map[(bar, pos)] = None
            continue

        root_str, quality_str = (chord_str.split(":", 1) if ":" in chord_str
                                 else (chord_str, ""))
        quality_str = quality_str.split("/")[0]
        root   = _HARTE_ROOT_TO_SEMITONE.get(root_str, -1)
        q_map  = _HARTE_QUALITY_TO_IDX_FULL if n_qualities >= 12 else _HARTE_QUALITY_TO_IDX_CORE
        q_idx  = q_map.get(quality_str, -1)

        if root < 0 or q_idx < 0 or q_idx >= n_qualities:
            chord_map[(bar, pos)] = None
        else:
            chord_map[(bar, pos)] = (root, q_idx)

    return chord_map


def _parse_pop909_key_file(key_file: Path) -> "tuple[int, int] | None":
    """Parse POP909 key_audio.txt → (root_0_11, mode_0_1) or None."""
    _KEY_ROOT = {
        "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
        "E": 4, "Fb": 4, "F": 5, "F#": 6, "Gb": 6,
        "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10,
        "B": 11, "Cb": 11,
    }
    _MODE = {"major": 0, "maj": 0, "M": 0, "minor": 1, "min": 1, "m": 1}

    try:
        lines = key_file.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        logger.debug(f"Cannot read {key_file}: {exc}")
        return None

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 3 and ":" in parts[2]:
            key_str = parts[2]
        elif len(parts) >= 2 and ":" in parts[1]:
            key_str = parts[1]
        elif len(parts) >= 1 and ":" in parts[0]:
            key_str = parts[0]
        else:
            continue
        root_str, mode_str = key_str.split(":", 1)
        root = _KEY_ROOT.get(root_str, -1)
        mode = _MODE.get(mode_str.lower(), -1)
        if root >= 0 and mode >= 0:
            return (root, mode)

    return None


def _parse_cl_chord_csv(
    csv_path: Path,
    n_qualities: int,
    cl_midi_path: "Path | None" = None,
) -> "tuple[dict[tuple[int, int], tuple[int, int] | None], tuple[int, int] | None]":
    """Parse a POP909-CL chord_symbol.csv into a chord map and dominant key."""
    import csv as _csv

    _CL_Q: dict[str, int] = {
        "M":    0, "m":    1, "D7":   2, "M7":   3, "m7":   4,
        "o":    5, "+":    6, "sus2": 7, "sus4": 8, "o7":   9, "/o7":  10,
    }

    chord_map: dict[tuple[int, int], tuple[int, int] | None] = {}
    key_votes: dict[tuple[int, int], int] = {}

    beats_per_bar     = 4
    resolution        = 4
    positions_per_bar = beats_per_bar * resolution

    if cl_midi_path is None or not cl_midi_path.exists():
        logger.debug(
            f"CL MIDI not found ({cl_midi_path}); skipping chord_symbol.csv "
            f"for {csv_path.parent.name}. Re-run: "
            "python scripts/download_pop909.py --also_cl --force_cl"
        )
        return chord_map, None

    preroll_beats = 0.0
    try:
        import miditoolkit as _mtk
        _cl = _mtk.MidiFile(str(cl_midi_path))
        _piano = _cl.instruments[0]
        if _piano.notes:
            _first_tick = min(n.start for n in _piano.notes)
            preroll_beats = _first_tick / max(_cl.ticks_per_beat, 1)
    except Exception as exc:
        logger.debug(f"Cannot read CL MIDI {cl_midi_path}: {exc}")
        return chord_map, None

    try:
        text = csv_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.debug(f"Cannot read {csv_path}: {exc}")
        return chord_map, None

    for row in _csv.DictReader(text.splitlines()):
        try:
            offset_qb = float(row["offset_qb"])
        except (KeyError, ValueError):
            continue
        actual_qb   = offset_qb + preroll_beats
        pos_in_grid = round(actual_qb * resolution)
        bar = pos_in_grid // positions_per_bar
        pos = pos_in_grid % positions_per_bar

        quality_str = (row.get("quality") or "").strip()
        root_str    = (row.get("root")    or "").strip()

        if quality_str in ("N", ""):
            chord_map[(bar, pos)] = None
            continue

        root  = _HARTE_ROOT_TO_SEMITONE.get(root_str, -1)
        q_idx = _CL_Q.get(quality_str, -1)

        if root < 0 or q_idx < 0 or q_idx >= n_qualities:
            chord_map[(bar, pos)] = None
        else:
            chord_map[(bar, pos)] = (root, q_idx)

        key_str = (row.get("local_key") or "").strip()
        if key_str:
            mode     = 0 if key_str[0].isupper() else 1
            root_key = _HARTE_ROOT_TO_SEMITONE.get(key_str.capitalize(), -1)
            if root_key >= 0:
                k = (root_key, mode)
                key_votes[k] = key_votes.get(k, 0) + 1

    dominant_key = max(key_votes, key=lambda k: key_votes[k]) if key_votes else None
    return chord_map, dominant_key

# ---------------------------------------------------------------------------
# Finder and encoder
# ---------------------------------------------------------------------------

def _find_pop909_midis(root: Path) -> list[Path]:
    found = sorted(root.rglob("*.mid")) + sorted(root.rglob("*.midi"))
    return [p for p in found
            if "versions" not in {part.lower() for part in p.parts}
            and not p.stem.endswith("_cl")]


def _encode_one(
    midi_path: Path,
    tokenizer: BaseTokenizer,
    cond_tracks: list[str],
    target_tracks: list[str],
    out_dir: Path,
    name_prefix: str = "",
    min_melody_coverage: float = 0.0,
) -> Optional[str]:
    """Encode one POP909 (or generic) MIDI with GT chord/key annotations."""
    stem = name_prefix + _safe_name(midi_path)
    if (out_dir / f"{stem}.pt").exists():
        return stem

    try:
        events, tempo = midi_to_events(midi_path, tokenizer.cfg)
    except Exception as e:
        logger.warning(f"Skipping {midi_path.name}: {e}")
        return None
    if not events:
        return None

    cond_track = cond_tracks[0] if cond_tracks else "melody"
    if min_melody_coverage > 0.0:
        cov = _melody_coverage(events, cond_track)
        if cov < min_melody_coverage:
            logger.debug(
                f"{midi_path.name}: skipping — melody coverage {cov:.1%} "
                f"< {min_melody_coverage:.0%}"
            )
            return None

    chord_map = None
    key_root  = None
    key_mode  = None
    n_q = tokenizer.cfg.chord_qualities

    song_dir = midi_path.parent
    if isinstance(tokenizer, REMITokenizer):
        cl_csv = song_dir / "chord_symbol.csv"
        cl_midi = song_dir / f"{song_dir.name}_cl.mid"
        if cl_csv.exists() and cl_midi.exists():
            cm, key_result = _parse_cl_chord_csv(cl_csv, n_q, cl_midi_path=cl_midi)
            if cm:
                chord_map = cm
            if key_result is not None:
                key_root, key_mode = key_result
        else:
            chord_file = song_dir / "chord_midi.txt"
            key_file   = song_dir / "key_audio.txt"
            if chord_file.exists():
                cm = _parse_pop909_chord_file(chord_file, n_q, midi_path=midi_path)
                chord_map = cm or None
            if key_file.exists():
                kresult = _parse_pop909_key_file(key_file)
                if kresult is not None:
                    key_root, key_mode = kresult

    ids, mask = tokenizer.encode_song(
        events,
        condition_tracks=cond_tracks,
        target_tracks=target_tracks,
        tempo_bpm=tempo,
        chord_map=chord_map,
        key_root=key_root,
        key_mode=key_mode,
    )
    if sum(mask) < 8:
        return None

    stem = name_prefix + _safe_name(midi_path)
    _save_shard(
        out_dir, stem, ids, mask,
        key_root=key_root if key_root is not None else -1,
        key_mode=key_mode if key_mode is not None else -1,
    )
    return stem
