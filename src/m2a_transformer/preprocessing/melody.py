"""Melody track selection: weight heuristic and midi-miner classifier."""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Optional

from m2a_transformer.utils.logger import logger
from m2a_transformer.tokenizer import NoteEvent

_GM_BASS_PROGRAMS: frozenset[int] = frozenset(range(32, 40))
_GM_MELODY_HINT_PROGRAMS: frozenset[int] = frozenset(
    list(range(40, 44)) + list(range(56, 64)) +
    list(range(64, 80)) + list(range(80, 88))
)

_DEFAULT_MM_MODELS = Path(r"C:\Users\hojun\mm_models")

# (note_count, median_pitch_int) — stable across miditoolkit / PrettyMIDI
_MinerFP = tuple[int, int]

# Provenance of the most recent _lakh_track_events selection: "miner" / "weight".
# Read by the encoder immediately after the call (safe: one file per process at a
# time). Persisted into the shard so runs can report the miner/weight breakdown.
_LAST_METHOD: "str | None" = None

_MAX_BARS = 2000  # ~10 min at 120 BPM 4/4; guards against 32-bit tick overflow


def _load_mm_models(mm_models_dir: Path) -> list:
    """Load midi-miner RandomForest models (call once; ~520 MB total).

    Requires midi-miner to be installed: pip install -e <midi-miner-repo>
    See pyproject.toml [project.optional-dependencies] prepare section.
    """
    import track_separate as ts
    import logging as _lg
    ts.logger = _lg.getLogger("mm")
    ts.logger.addHandler(_lg.NullHandler())
    names = ("melody_model", "bass_model", "chord_model", "drum_model")
    models = []
    for name in names:
        m = pickle.load(open(mm_models_dir / name, "rb"))
        for est in m.estimators_:
            if not hasattr(est, "monotonic_cst"):
                est.monotonic_cst = None
        models.append(m)
    return models


def _miner_melody_fp(midi_path: Path, mm_models: list) -> Optional[_MinerFP]:
    """Return (note_count, median_pitch) of the midi-miner-selected melody track."""
    import track_separate as ts
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            feats, pm = ts.cal_file_features(str(midi_path))
        except Exception:
            return None
    if feats is None or pm is None:
        return None
    try:
        df = ts.add_labels(feats)
        df = ts.predict_labels(df, *mm_models)
    except Exception:
        return None
    mel_rows = df[df["melody_predict"] == True]
    if len(mel_rows) == 0:
        return None
    row_idx = int(mel_rows.index[0])
    if row_idx >= len(pm.instruments):
        return None
    inst = pm.instruments[row_idx]
    if not inst.notes:
        return None
    pitches = sorted(n.pitch for n in inst.notes)
    return (len(pitches), pitches[len(pitches) // 2])


def _melody_coverage(events: list, cond_track: str = "melody") -> float:
    """Return the fraction of bars that contain at least one melody note."""
    mel_bars = {e.bar for e in events if e.track == cond_track}
    all_bars  = {e.bar for e in events}
    return len(mel_bars) / max(len(all_bars), 1)


def _instrument_to_events(inst, track: str, cfg, tpb: int) -> list[NoteEvent]:
    """Convert a miditoolkit instrument's notes to NoteEvents."""
    res    = cfg.resolution
    ppb    = cfg.positions_per_bar
    plo    = cfg.pitch_min
    phi    = cfg.pitch_max
    dur_hi = cfg.duration_max
    out: list[NoteEvent] = []
    for n in inst.notes:
        sp = int(round(n.start * res / tpb))
        ep = int(round(n.end   * res / tpb))
        out.append(NoteEvent(
            track=track,
            bar=sp // ppb, position=sp % ppb,
            pitch=max(plo, min(phi, n.pitch)),
            duration=max(1, min(dur_hi, ep - sp)),
            velocity=max(1, min(127, n.velocity)),
        ))
    return out


def _quick_coverage(inst, tpb: int, ppb: int, res: int) -> float:
    """Bar coverage of an instrument: distinct bars with notes / total span bars.

    Used to skip sparse weight-selected tracks before committing to them.
    A track that plays in only 3 out of 40 bars has coverage 0.075.
    """
    if not inst.notes:
        return 0.0
    bars = {int(round(n.start * res / tpb)) // ppb for n in inst.notes}
    span_bars = max(bars) + 1
    return len(bars) / span_bars


def _lakh_track_events(
    midi_path: Path, cfg, min_notes: int = 4,
    mm_models: Optional[list] = None, miner_fallback: str = "weight",
    min_melody_coverage: float = 0.0,
    chord_match_threshold: float = 0.75,
) -> Optional[tuple[list[NoteEvent], float]]:
    """Extract NoteEvents from a Lakh-style MIDI.

    Selects melody track via weight heuristic or midi-miner classifier.
    When weight is used (primary or fallback), iterates ranked candidates and
    skips any that fall below min_melody_coverage, so sparse tracks are
    bypassed in favour of the next best candidate.
    Returns (events, tempo_bpm) or None if the file should be skipped.
    """
    try:
        import miditoolkit
        midi = miditoolkit.MidiFile(str(midi_path))
    except Exception as exc:
        logger.debug(f"miditoolkit failed on {midi_path.name}: {exc}")
        return None

    tempo_bpm = midi.tempo_changes[0].tempo if midi.tempo_changes else 120.0
    tpb  = midi.ticks_per_beat
    res  = cfg.resolution
    ppb  = cfg.positions_per_bar

    non_drum = [i for i in midi.instruments if not i.is_drum and len(i.notes) >= min_notes]
    if len(non_drum) < 2:
        return None

    _all_note_ends = [n.end for inst in non_drum for n in inst.notes]
    if _all_note_ends:
        _max_tick_est = max(_all_note_ends)
        _max_bar_est  = int(_max_tick_est * res / max(tpb, 1)) // ppb
        if _max_bar_est > _MAX_BARS:
            logger.debug(
                f"{midi_path.name}: skipping — max_bar_est={_max_bar_est:,} > {_MAX_BARS} "
                f"(likely 32-bit tick overflow)"
            )
            return None

    def _med_pitch(inst):
        p = sorted(n.pitch for n in inst.notes)
        return float(p[len(p) // 2])

    def _mono_rate(inst):
        notes = sorted(inst.notes, key=lambda n: n.start)
        if len(notes) < 2:
            return 1.0
        return sum(1 for a, b in zip(notes, notes[1:]) if a.end <= b.start) / (len(notes)-1)

    def _density(inst):
        span  = max(n.end for n in inst.notes) - min(n.start for n in inst.notes)
        beats = span / max(tpb, 1)
        return len(inst.notes) / max(beats, 1.0)

    bass    = [i for i in non_drum if i.program in _GM_BASS_PROGRAMS]
    melodic = [i for i in non_drum if i.program not in _GM_BASS_PROGRAMS]
    if len(melodic) < 2:
        melodic, bass = non_drum, []

    mps = [_med_pitch(i) for i in melodic]
    mrs = [_mono_rate(i)  for i in melodic]
    mds = [_density(i)    for i in melodic]
    lo_p, hi_p = min(mps), max(mps)
    lo_d, hi_d = min(mds), max(mds)
    pr = max(hi_p - lo_p, 1.0)
    dr = max(hi_d - lo_d, 1.0)

    def _score(idx):
        return (0.10 * (mps[idx]-lo_p)/pr + 0.80 * mrs[idx]
                + 0.05 * (1-(mds[idx]-lo_d)/dr)
                + 0.05 * (1 if melodic[idx].program in _GM_MELODY_HINT_PROGRAMS else 0))

    # ── melody instrument selection ────────────────────────────────────────
    melody_inst = None
    method = "weight"   # provenance; set to "miner" below if the miner picks it

    if mm_models is not None:
        fp = _miner_melody_fp(midi_path, mm_models)
        if fp is not None:
            for inst in non_drum:
                pitches = sorted(n.pitch for n in inst.notes)
                if pitches and (len(pitches), pitches[len(pitches) // 2]) == fp:
                    melody_inst = inst
                    method = "miner"
                    break
        if melody_inst is None:
            if miner_fallback == "skip":
                return None
            # fallback to weight heuristic below

    if melody_inst is None:
        # Iterate weight-ranked candidates; skip any that are too sparse.
        for idx in sorted(range(len(melodic)), key=_score, reverse=True):
            inst = melodic[idx]
            if (min_melody_coverage > 0.0
                    and _quick_coverage(inst, tpb, ppb, res) < min_melody_coverage):
                continue
            melody_inst = inst
            break
        if melody_inst is None:
            return None

    # All other non-drum instruments → single "accompaniment" stream.
    track_for: dict[int, str] = {}
    track_for[id(melody_inst)] = "melody"
    for inst in non_drum:
        if id(inst) != id(melody_inst):
            track_for[id(inst)] = "accompaniment"

    events: list[NoteEvent] = []
    for inst in non_drum:
        track = track_for.get(id(inst), cfg.tracks[-1])
        if track not in cfg.tracks:
            track = cfg.tracks[-1]
        events.extend(_instrument_to_events(inst, track, cfg, tpb))

    globals()["_LAST_METHOD"] = method
    return events, tempo_bpm
