"""MIDI ↔ NoteEvent conversion.

Wraps miditoolkit so the rest of the project deals only with the typed
NoteEvent intermediate representation. POP909-specific track-name parsing
lives here too."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Sequence

import miditoolkit

from m2a_transformer.config import TokenizerConfig
from m2a_transformer.tokenizer import NoteEvent


# POP909 tracks are named "MELODY" / "BRIDGE" / "PIANO" (upper-cased).
# We normalize anything we see against this map; unknown names fall back to
# "piano" so generic MIDI files can be loaded as a single-target song.
_POP909_NAME_MAP: Dict[str, str] = {
    "MELODY": "melody",
    "BRIDGE": "accompaniment",
    "PIANO":  "accompaniment",
    "LEAD":   "melody",
    "VOCAL":  "melody",
}


def _ticks_to_positions(tick: int, ticks_per_beat: int, resolution: int) -> int:
    return int(round(tick * resolution / ticks_per_beat))


def midi_to_events(
    path: str | Path,
    cfg: TokenizerConfig,
    track_name_override: Dict[str, str] | None = None,
) -> tuple[List[NoteEvent], float]:
    """Read a MIDI file into NoteEvents quantized at the tokenizer resolution.

    Returns (events, tempo_bpm). Unknown instrument names map to the last
    configured track ("accompaniment" by default) so non-POP909 MIDIs still load."""
    name_map = {**_POP909_NAME_MAP, **(track_name_override or {})}
    midi = miditoolkit.MidiFile(str(path))
    tpb = midi.ticks_per_beat
    tempo_bpm = midi.tempo_changes[0].tempo if midi.tempo_changes else 120.0
    pos_per_bar = cfg.positions_per_bar
    res = cfg.resolution
    default_track = cfg.tracks[-1]

    events: List[NoteEvent] = []
    for inst in midi.instruments:
        track = name_map.get((inst.name or "").strip().upper(), default_track)
        if track not in cfg.tracks:
            track = default_track
        for n in inst.notes:
            start_pos = _ticks_to_positions(n.start, tpb, res)
            end_pos = _ticks_to_positions(n.end,   tpb, res)
            dur = max(1, end_pos - start_pos)
            bar = start_pos // pos_per_bar
            pos_in_bar = start_pos % pos_per_bar
            events.append(NoteEvent(
                track=track,
                bar=bar,
                position=pos_in_bar,
                pitch=n.pitch,
                duration=dur,
                velocity=n.velocity,
            ))

    return events, tempo_bpm


_DEFAULT_PROGRAMS: Dict[str, int] = {
    "melody":        40,   # violin
    "accompaniment":  0,   # acoustic grand piano
}


def events_to_midi(
    events: Sequence[NoteEvent],
    cfg: TokenizerConfig,
    tempo_bpm: float = 120.0,
    ticks_per_beat: int = 480,
    programs: Dict[str, int] | None = None,
) -> miditoolkit.MidiFile:
    """Render NoteEvents to a miditoolkit MidiFile (one Instrument per track).

    *programs* maps logical track name → GM program number (0–127).
    Falls back to built-in defaults when not provided.
    """
    midi = miditoolkit.MidiFile(ticks_per_beat=ticks_per_beat)
    midi.tempo_changes = [miditoolkit.TempoChange(tempo=tempo_bpm, time=0)]

    program_for = {**_DEFAULT_PROGRAMS, **(programs or {})}
    inst_for: Dict[str, miditoolkit.Instrument] = {}
    for tr in cfg.tracks:
        inst = miditoolkit.Instrument(program=program_for.get(tr, 0), name=tr.upper())
        inst_for[tr] = inst
        midi.instruments.append(inst)

    ticks_per_pos = ticks_per_beat // cfg.resolution
    for e in events:
        start_tick = (e.bar * cfg.positions_per_bar + e.position) * ticks_per_pos
        end_tick = start_tick + e.duration * ticks_per_pos
        inst_for.get(e.track, inst_for[cfg.tracks[-1]]).notes.append(
            miditoolkit.Note(velocity=e.velocity, pitch=e.pitch,
                             start=start_tick, end=end_tick)
        )
    return midi


def humanize_midi(
    midi: miditoolkit.MidiFile,
    velocity_std: int = 6,
    timing_std_ms: float = 8.0,
    duration_std_ms: float = 5.0,
) -> miditoolkit.MidiFile:
    """Add subtle randomness to velocity and timing to reduce mechanical feel.

    Parameters
    ----------
    velocity_std:    gaussian std for velocity jitter (±units, clamped 1–127)
    timing_std_ms:   gaussian std for note-on timing jitter in milliseconds
    duration_std_ms: gaussian std for note duration jitter in milliseconds
    """
    tpb = midi.ticks_per_beat
    tempo = midi.tempo_changes[0].tempo if midi.tempo_changes else 120.0
    ms_per_tick = (60000.0 / tempo) / tpb  # BPM → ms/tick: (60000 ms/min) / BPM / (ticks/beat)

    timing_std_ticks = int(timing_std_ms / ms_per_tick)
    duration_std_ticks = int(duration_std_ms / ms_per_tick)

    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            note.velocity = max(1, min(127,
                note.velocity + int(random.gauss(0, velocity_std))))

            t_jitter = int(random.gauss(0, timing_std_ticks)) if timing_std_ticks else 0
            d_jitter = int(random.gauss(0, duration_std_ticks)) if duration_std_ticks else 0
            note.start = max(0, note.start + t_jitter)
            note.end = max(note.start + 1, note.end + t_jitter + d_jitter)

    return midi
