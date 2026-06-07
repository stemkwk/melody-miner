"""Synthetic toy song generator (CI smoke test)."""
from __future__ import annotations

import random
from typing import List

from m2a_transformer.tokenizer import NoteEvent


def _synthesize_song(seed: int, n_bars: int = 16) -> tuple[list[NoteEvent], float, int, int]:
    rng = random.Random(seed)
    tempo = rng.choice([80, 100, 120, 140])
    scale = [60, 62, 64, 65, 67, 69, 71, 72]
    chord_roots = [60, 65, 67, 64]
    events: List[NoteEvent] = []
    for bar in range(n_bars):
        for q in range(4):
            events.append(NoteEvent(
                track="melody", bar=bar, position=q * 4,
                pitch=rng.choice(scale), duration=4, velocity=rng.randint(70, 100),
            ))
        root = chord_roots[bar % len(chord_roots)]
        for offs in (0, 4, 7):
            events.append(NoteEvent(
                track="accompaniment", bar=bar, position=0,
                pitch=root + offs, duration=16, velocity=70,
            ))
    return events, tempo, 0, 0  # key_root=0 (C), key_mode=0 (major)
