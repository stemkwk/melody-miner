"""Slakh2100 dataset preprocessing with metadata-based melody extraction."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from m2a_transformer.utils.logger import logger
from m2a_transformer.tokenizer import BaseTokenizer, NoteEvent, REMITokenizer
from m2a_transformer.preprocessing.shards import _safe_name, _save_shard
from m2a_transformer.preprocessing.chords import (
    _extract_chords_from_midi, _estimate_key_from_midi, _MATCH_THRESHOLD,
)
from m2a_transformer.preprocessing.melody import (
    _lakh_track_events,
    _melody_coverage,
    _instrument_to_events,
    _MAX_BARS,
)
from m2a_transformer.preprocessing import melody as _melody

# inst_class values in Slakh metadata.yaml that indicate a melody instrument.
# Synth Lead / Pipe (flute etc.) / Reed (sax, clarinet) / Brass (trumpet etc.)
_SLAKH_MELODY_CLASSES: frozenset[str] = frozenset(
    ["Synth Lead", "Pipe", "Reed", "Brass"]
)

# Tiebreaker priority when mono_rate is equal (higher = preferred).
_MELODY_CLASS_PRIORITY: dict[str, int] = {
    "Synth Lead": 3,
    "Pipe":       2,
    "Reed":       1,
    "Brass":      0,
}


def _find_slakh_dirs(root: Path) -> list[Path]:
    return sorted({p.parent for p in root.rglob("all_src.mid")})


def _slakh_instrument_events(
    song_dir: Path, cfg
) -> Optional[tuple[list[NoteEvent], float]]:
    """Extract melody from Slakh using metadata.yaml inst_class labels (near-GT).

    Loads individual per-stem MIDI files (MIDI/SXX.mid) to avoid cross-track
    fingerprint matching. Among melody-class stems, selects the one with the
    highest mono_rate. Non-drum remaining stems become accompaniment.

    Returns (events, tempo_bpm) or None when no melody-class stem is found
    (caller should fall back to _lakh_track_events).
    """
    meta_path = song_dir / "metadata.yaml"
    if not meta_path.exists():
        return None

    try:
        import yaml
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    stems_meta = meta.get("stems", {})

    # Collect valid (stem_id, midi_path, inst_class) tuples for non-drum stems.
    all_stems:      list[tuple[str, Path]] = []
    mel_candidates: list[tuple[str, Path, str]] = []
    for stem_id, info in stems_meta.items():
        if info.get("is_drum", False):
            continue
        mid_path = song_dir / "MIDI" / f"{stem_id}.mid"
        if not mid_path.exists():
            continue
        all_stems.append((stem_id, mid_path))
        ic = info.get("inst_class", "")
        if ic in _SLAKH_MELODY_CLASSES:
            mel_candidates.append((stem_id, mid_path, ic))

    if not mel_candidates:
        return None

    # Pick the melody-class stem by (mono_rate DESC, class_priority DESC).
    best_stem_id: Optional[str] = None
    best_key = (-1.0, -1)
    for stem_id, mid_path, ic in mel_candidates:
        try:
            import miditoolkit
            m = miditoolkit.MidiFile(str(mid_path))
        except Exception:
            continue
        if not m.instruments or not m.instruments[0].notes:
            continue
        notes = sorted(m.instruments[0].notes, key=lambda n: n.start)
        mono = (1.0 if len(notes) < 2
                else sum(1 for a, b in zip(notes, notes[1:]) if a.end <= b.start) / (len(notes) - 1))
        key = (mono, _MELODY_CLASS_PRIORITY.get(ic, 0))
        if key > best_key:
            best_key = key
            best_stem_id = stem_id

    if best_stem_id is None:
        return None

    # Get tempo from all_src.mid (individual stem MIDIs rarely carry tempo maps).
    tempo_bpm = 120.0
    all_src = song_dir / "all_src.mid"
    if all_src.exists():
        try:
            import miditoolkit
            tmp = miditoolkit.MidiFile(str(all_src))
            if tmp.tempo_changes:
                tempo_bpm = tmp.tempo_changes[0].tempo
        except Exception:
            pass

    # Build NoteEvents from every non-drum stem.
    res = cfg.resolution
    ppb = cfg.positions_per_bar
    events: list[NoteEvent] = []

    for stem_id, mid_path in all_stems:
        try:
            import miditoolkit
            m = miditoolkit.MidiFile(str(mid_path))
        except Exception:
            continue
        if not m.instruments:
            continue
        inst = m.instruments[0]
        if not inst.notes:
            continue
        tpb = m.ticks_per_beat

        # Guard against 32-bit tick overflow.
        max_tick = max(n.end for n in inst.notes)
        if int(max_tick * res / max(tpb, 1)) // ppb > _MAX_BARS:
            logger.debug(f"{song_dir.name}/{stem_id}: skipping stem — tick overflow")
            continue

        track = "melody" if stem_id == best_stem_id else "accompaniment"
        events.extend(_instrument_to_events(inst, track, cfg, tpb))

    if not events:
        return None

    return events, tempo_bpm


def _encode_slakh_one(
    song_dir: Path,
    tokenizer: BaseTokenizer,
    cond_tracks: list[str],
    target_tracks: list[str],
    out_dir: Path,
    name_prefix: str = "slakh_",
    min_melody_coverage: float = 0.0,
    mm_models: Optional[list] = None,
    miner_fallback: str = "weight",
    slakh_melody: str = "instrument",
    min_stem_notes: int = 4,
    chord_match_threshold: float = _MATCH_THRESHOLD,
) -> Optional[str]:
    """Encode one Slakh song. Chord: full 12-quality. Key: auto-estimate.

    slakh_melody choices:
      'instrument' — use metadata.yaml inst_class (GT-quality). Songs with no
                     melody-class stem (~27%) fall back to the shared
                     miner/weight selector instead of being dropped.
      'miner'      — use midi-miner classifier via _lakh_track_events.
      'weight'     — use mono_rate weight heuristic via _lakh_track_events.
    """
    stem = _safe_name(song_dir)
    h    = hashlib.md5(str(song_dir).encode("utf-8", errors="replace")).hexdigest()[:8]
    if len(name_prefix) + len(stem) + 9 > 120:
        stem = stem[:max(0, 120 - len(name_prefix) - 9)]
    raw_name = f"{name_prefix}{stem}_{h}"
    if (out_dir / f"{raw_name}.pt").exists():
        return raw_name

    all_src = song_dir / "all_src.mid"
    if not all_src.exists():
        midis = sorted(song_dir.glob("*.mid"))
        if not midis:
            return None
        all_src = midis[0]

    result = None
    method = "instrument"
    if slakh_melody == "instrument":
        # metadata.yaml inst_class labels (near-GT). Returns None when the song
        # has no melody-class stem (~27%) — fall through to the shared selector.
        result = _slakh_instrument_events(song_dir, tokenizer.cfg)

    if result is None:
        result = _lakh_track_events(
            all_src, tokenizer.cfg,
            min_notes=min_stem_notes,
            mm_models=mm_models, miner_fallback=miner_fallback,
            min_melody_coverage=min_melody_coverage,
            chord_match_threshold=chord_match_threshold,
        )
        method = _melody._LAST_METHOD   # "miner" / "weight" from the fallback
    if result is None:
        return None
    events, tempo = result

    cond_track = cond_tracks[0] if cond_tracks else "melody"
    if min_melody_coverage > 0.0:
        cov = _melody_coverage(events, cond_track)
        if cov < min_melody_coverage:
            logger.debug(
                f"{song_dir.name}: skipping — melody coverage {cov:.1%} "
                f"< {min_melody_coverage:.0%}"
            )
            return None

    chord_map = None
    key_root  = None
    key_mode  = None

    if isinstance(tokenizer, REMITokenizer):
        extracted = _extract_chords_from_midi(
            all_src, tokenizer.cfg, n_qualities=12,
            chord_match_threshold=chord_match_threshold,
        )
        chord_map = extracted or None
        kresult   = _estimate_key_from_midi(all_src)
        if kresult is not None:
            key_root, key_mode = kresult

    ids, mask = tokenizer.encode_song(
        events, cond_tracks, target_tracks,
        tempo_bpm=tempo, chord_map=chord_map,
        key_root=key_root, key_mode=key_mode,
    )
    if sum(mask) < 8:
        return None

    _save_shard(
        out_dir, raw_name, ids, mask,
        key_root=key_root if key_root is not None else -1,
        key_mode=key_mode if key_mode is not None else -1,
        method=method,
    )
    return raw_name
