"""Lakh MIDI Clean dataset preprocessing."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from m2a_transformer.utils.logger import logger
from m2a_transformer.tokenizer import BaseTokenizer, REMITokenizer
from m2a_transformer.preprocessing.shards import _safe_name, _save_shard
from m2a_transformer.preprocessing.chords import (
    _extract_chords_from_midi, _estimate_key_from_midi, _MATCH_THRESHOLD,
)
from m2a_transformer.preprocessing.melody import _lakh_track_events, _melody_coverage
from m2a_transformer.preprocessing import melody as _melody


def _find_lakh_midis(root: Path) -> list[Path]:
    return sorted(root.rglob("*.mid")) + sorted(root.rglob("*.midi"))


def _encode_lakh_one(
    midi_path: Path,
    tokenizer: BaseTokenizer,
    cond_tracks: list[str],
    target_tracks: list[str],
    out_dir: Path,
    name_prefix: str = "lakh_",
    min_melody_coverage: float = 0.0,
    min_stem_notes: int = 4,
    chord_match_threshold: float = _MATCH_THRESHOLD,
    mm_models: Optional[list] = None,
    miner_fallback: str = "weight",
) -> Optional[str]:
    """Encode one Lakh MIDI. Chord: core 9-quality. Key: auto-estimate."""
    stem = _safe_name(midi_path)
    h    = hashlib.md5(str(midi_path).encode("utf-8", errors="replace")).hexdigest()[:8]
    if len(name_prefix) + len(stem) + 9 > 120:
        stem = stem[:max(0, 120 - len(name_prefix) - 9)]
    raw_name = f"{name_prefix}{stem}_{h}"
    if (out_dir / f"{raw_name}.pt").exists():
        return raw_name

    result = _lakh_track_events(
        midi_path, tokenizer.cfg,
        min_notes=min_stem_notes,
        mm_models=mm_models, miner_fallback=miner_fallback,
        min_melody_coverage=min_melody_coverage,
        chord_match_threshold=chord_match_threshold,
    )
    if result is None:
        return None
    events, tempo = result

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

    if isinstance(tokenizer, REMITokenizer):
        extracted = _extract_chords_from_midi(
            midi_path, tokenizer.cfg, n_qualities=9,
            chord_match_threshold=chord_match_threshold,
        )
        chord_map = extracted or None
        kresult   = _estimate_key_from_midi(midi_path)
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
        method=_melody._LAST_METHOD,
    )
    return raw_name
