"""REMI-style tokenizer with fully relative harmonic encoding.

Design rationale
----------------
All harmonic information is expressed relative to the piece's key:

  Chord root  → SCALE_DEGREE_{0-11}   semitones above key root (mod 12)
  Chord type  → QUALITY_{name}        one of 12 chord qualities
  Note pitch  → CHROMA_{0-11}         pitch class relative to key root
                OCTAVE_{1-9}          abs_pitch // 12  (register)

The KEY_{root}_{mode} token (24 variants) is the sole absolute harmonic anchor.

Key invariance proof
---------------------
Given transposition by N semitones (new_key_root = (key_root+N)%12):
  new_CHROMA      = ((P+N)%12 - (K+N)%12) % 12  =  (P%12 - K%12) % 12  =  CHROMA  ✓
  new_SCALE_DEG   = ((C+N)%12 - (K+N)%12) % 12  =  (C%12 - K%12) % 12  =  SD      ✓

CHROMA and SCALE_DEGREE are invariant. Only KEY root and OCTAVE need updating
during pitch augmentation.

Vocabulary layout (173 tokens)
--------------------------------
  [0]          PAD
  [1]          BOS
  [2]          SEP           (melody→accompaniment block boundary)
  [3]          EOS
  [4]          BAR
  [5..20]      POS_0..POS_15           (16 positions per bar)
  [21..36]     TEMPO_0..TEMPO_15       (16 bins, log-scale 50–200 BPM)
  [37..38]     TRACK_melody/accompaniment
  [39..50]     CHROMA_0..CHROMA_11     pitch class relative to key root
  [51..59]     OCTAVE_1..OCTAVE_9      abs_pitch // 12
  [60..91]     DUR_1..DUR_32
  [92..123]    VEL_0..VEL_31
  [124..135]   SCALE_DEGREE_0..11      chord root relative to key root
  [136..147]   QUALITY_{name} × 12    chord quality
  [148]        CHORD_N                 no chord / unknown
  [149..172]   KEY_{r}_maj/min × 24   global key anchor

Total: 173 tokens.

Tempo log-scale rationale
--------------------------
Human tempo JND ≈ 4-6%; 50→200 BPM spans log2(200/50)=2 octaves.
16 log-spaced bins give ~9% spacing (slightly above JND), covering all
musical tempo categories (Larghetto→Prestissimo) without the density
imbalance of linear binning (which over-resolves 50-100 BPM and
under-resolves 150-200 BPM).

Sequence format — TEMPORAL INTERLEAVING
----------------------------------------
  <BOS> KEY_{r}_{mode} TEMPO_{bin}
    BAR  [SCALE_DEGREE_x QUALITY_y | CHORD_N]        ← shared across tracks
    POS_n
      TRACK_melody   CHROMA_c OCTAVE_o DUR_d VEL_v   ← condition (mask=False)
      TRACK_acc      CHROMA_c OCTAVE_o DUR_d VEL_v   ← target    (mask=True)
    POS_n ...
    BAR  ...
  <EOS>

At each (bar, position) time step the condition track's notes appear first,
followed immediately by the target track's notes.  This places the melody
and accompaniment notes that are harmonically simultaneous adjacent in the
token sequence, giving the model direct causal access to the melody context
when predicting each accompaniment note.

The <SEP> token is emitted as the melody→accompaniment block boundary in
the interleaved format.  CFG (classifier-free guidance) is not supported in
interleaved mode; set cfg_w=0 in inference config.

Augmentation contract
---------------------
To transpose by N semitones (dataset._augment):
  1. Update KEY root:      new_root = (key_root + N) % 12
  2. Update OCTAVE tokens: new_abs  = old_abs + N  (clamped); new_OCTAVE = new_abs // 12
  3. CHROMA tokens:        unchanged (key-invariant — see proof above)
  4. SCALE_DEGREE tokens:  unchanged (key-invariant)

Chord map format (prepare_data → encode_song)
----------------------------------------------
  chord_map: dict[(bar, pos_in_resolution_units), (chord_root_0_11, quality_idx) | None]
  None  → emit CHORD_N
  (root, q) → emit SCALE_DEGREE_{(root-key_root)%12}  QUALITY_{q}
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type

from m2a_transformer.config import TokenizerConfig
from m2a_transformer.utils.logger import logger


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: 12 chord qualities in token-index order.
#: Indices 0-8 (core): reliably extractable from any MIDI source.
#:   add9 (index 7) absorbs sus2 — add9 retains the major 3rd.
#: Indices 9-11 (extended / Slakh-tier): dim7, hdim7, dom9.
CHORD_QUALITIES: List[str] = [
    "maj",    # 0
    "min",    # 1
    "dom7",   # 2
    "maj7",   # 3
    "min7",   # 4
    "dim",    # 5
    "aug",    # 6
    "add9",   # 7  (sus2 collapsed here)
    "sus4",   # 8
    "dim7",   # 9
    "hdim7",  # 10
    "dom9",   # 11
]

# Harmonic "avoid notes": pitch-class intervals (semitones above the chord ROOT)
# that clash with a chord tone — typically a note a half-step above the 3rd (the
# natural 11 over a major 3rd) which forms a b9 and sounds dissonant when held.
# Applied at inference as a SOFT logit penalty (not a hard mask) so passing /
# colour tones still survive. Only the textbook-clear cases are listed; qualities
# whose avoid notes are context-dependent (min, dim, aug, …) are intentionally
# omitted to avoid stripping legitimate harmonic colour.
CHORD_AVOID_INTERVALS: Dict[str, frozenset] = {
    "maj":  frozenset({5}),   # natural 4/11 vs the major 3rd
    "maj7": frozenset({5}),
    "dom7": frozenset({5}),
    "dom9": frozenset({5}),
    "add9": frozenset({5}),
    "sus4": frozenset({4}),   # major 3rd vs the suspended 4th
}

CHORD_ROOTS_STR: List[str] = [
    "C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B",
]

KEY_MODES_STR: List[str] = ["maj", "min"]

#: OCTAVE token range: abs_pitch // 12 for MIDI 21-108.
OCTAVE_MIN: int = 1   # MIDI 21 (A0) → 21 // 12 = 1
OCTAVE_MAX: int = 9   # MIDI 108 (C8) → 108 // 12 = 9


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class BaseTokenizer(ABC):
    cfg: TokenizerConfig

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...
    @property
    @abstractmethod
    def pad_id(self) -> int: ...
    @property
    @abstractmethod
    def bos_id(self) -> int: ...
    @property
    @abstractmethod
    def sep_id(self) -> int: ...
    @property
    @abstractmethod
    def eos_id(self) -> int: ...

    @abstractmethod
    def encode_song(
        self,
        events: "Sequence[NoteEvent]",
        condition_tracks: Sequence[str],
        target_tracks: Sequence[str],
        tempo_bpm: float | None = None,
        chord_map: "dict[tuple[int, int], tuple[int, int] | None] | None" = None,
        key_root: int | None = None,
        key_mode: int | None = None,
    ) -> Tuple[List[int], List[bool]]: ...

    @abstractmethod
    def decode(self, ids: Iterable[int]) -> "List[NoteEvent]": ...

    # --- pitch class / register helpers (relative encoding) -----------------
    @property
    @abstractmethod
    def chroma_min_id(self) -> int: ...
    @property
    @abstractmethod
    def chroma_max_id(self) -> int: ...
    @property
    @abstractmethod
    def octave_min_id(self) -> int: ...
    @property
    @abstractmethod
    def octave_max_id(self) -> int: ...
    @property
    @abstractmethod
    def octave_base(self) -> int: ...

    # --- velocity / duration / tempo helpers --------------------------------
    @property
    @abstractmethod
    def vel_min_id(self) -> int: ...
    @property
    @abstractmethod
    def vel_max_id(self) -> int: ...
    @property
    @abstractmethod
    def dur_min_id(self) -> int: ...
    @property
    @abstractmethod
    def dur_max_id(self) -> int: ...
    @property
    @abstractmethod
    def tempo_min_id(self) -> int: ...
    @property
    @abstractmethod
    def tempo_max_id(self) -> int: ...

    # --- harmony helpers ----------------------------------------------------
    @property
    @abstractmethod
    def sd_min_id(self) -> int: ...
    @property
    @abstractmethod
    def sd_max_id(self) -> int: ...
    @property
    @abstractmethod
    def quality_min_id(self) -> int: ...
    @property
    @abstractmethod
    def quality_max_id(self) -> int: ...
    @property
    @abstractmethod
    def chord_n_id(self) -> int: ...
    @property
    @abstractmethod
    def key_min_id(self) -> int: ...
    @property
    @abstractmethod
    def key_max_id(self) -> int: ...

    @abstractmethod
    def key_token_id(self, root: int, mode: int) -> int: ...

    # --- structural token id helpers ----------------------------------------
    @property
    @abstractmethod
    def bar_id(self) -> int: ...
    @property
    @abstractmethod
    def pos_id_range(self) -> Tuple[int, int]: ...
    @property
    @abstractmethod
    def track_id_range(self) -> Tuple[int, int]: ...

    # --- CFG helper ---------------------------------------------------------
    def make_uncond_prompt(self, prompt_ids) -> "list[int]":
        import torch as _torch
        ids = prompt_ids.tolist() if isinstance(prompt_ids, _torch.Tensor) else list(prompt_ids)
        try:
            sep_idx = ids.index(self.sep_id)
        except ValueError:
            return ids
        ids = ids[:]
        for i in range(1, sep_idx):
            ids[i] = self.pad_id
        return ids


_TOKENIZER_REGISTRY: Dict[str, Type[BaseTokenizer]] = {}


def register_tokenizer(name: str) -> Callable[[Type[BaseTokenizer]], Type[BaseTokenizer]]:
    def deco(cls: Type[BaseTokenizer]) -> Type[BaseTokenizer]:
        key = name.lower()
        if key in _TOKENIZER_REGISTRY:
            raise ValueError(f"Tokenizer '{name}' already registered")
        _TOKENIZER_REGISTRY[key] = cls
        return cls
    return deco


def build_tokenizer(cfg: TokenizerConfig) -> BaseTokenizer:
    key = cfg.name.lower()
    if key not in _TOKENIZER_REGISTRY:
        raise KeyError(f"Unknown tokenizer '{cfg.name}'. Known: {sorted(_TOKENIZER_REGISTRY)}")
    return _TOKENIZER_REGISTRY[key](cfg)


def available_tokenizers() -> list[str]:
    return sorted(_TOKENIZER_REGISTRY)


# ---------------------------------------------------------------------------
# NoteEvent
# ---------------------------------------------------------------------------

@dataclass
class NoteEvent:
    track: str
    bar: int
    position: int          # 0..positions_per_bar-1
    pitch: int             # MIDI pitch (absolute)
    duration: int          # resolution units
    velocity: int          # 1..127

    def __post_init__(self):
        self.duration = max(1, int(self.duration))
        self.velocity = max(1, min(127, int(self.velocity)))


# ---------------------------------------------------------------------------
# REMITokenizer — relative harmonic encoding
# ---------------------------------------------------------------------------

@register_tokenizer("remi_v1")
class REMITokenizer(BaseTokenizer):
    """REMI tokenizer with fully relative harmonic encoding.

    See module docstring for full vocabulary layout and design rationale.
    """

    PAD = "<PAD>"
    BOS = "<BOS>"
    SEP = "<SEP>"
    EOS = "<EOS>"

    def __init__(self, cfg: TokenizerConfig):
        self.cfg = cfg
        self.id_to_token: List[str] = []
        self.token_to_id: Dict[str, int] = {}

        def _add(tok: str) -> None:
            if tok in self.token_to_id:
                raise ValueError(f"Duplicate token: {tok}")
            self.token_to_id[tok] = len(self.id_to_token)
            self.id_to_token.append(tok)

        def _add_range(prefix: str, iterable) -> "tuple[int, int]":
            first = len(self.id_to_token)
            for v in iterable:
                _add(f"{prefix}_{v}")
            return first, len(self.id_to_token) - 1

        # ---- Specials --------------------------------------------------------
        for t in (self.PAD, self.BOS, self.SEP, self.EOS):
            _add(t)

        # ---- Structure -------------------------------------------------------
        _add("BAR")
        _add_range("POS",   range(cfg.positions_per_bar))
        self._tempo_min_id,   self._tempo_max_id   = _add_range("TEMPO",        range(cfg.tempo_bins))

        # ---- Tracks ----------------------------------------------------------
        for name in cfg.tracks:
            _add(f"TRACK_{name}")

        # ---- Pitch: CHROMA + OCTAVE (relative encoding) ----------------------
        self._chroma_min_id,  self._chroma_max_id  = _add_range("CHROMA",       range(12))
        self._octave_min_id,  self._octave_max_id  = _add_range("OCTAVE",       range(OCTAVE_MIN, OCTAVE_MAX + 1))

        # ---- Duration / Velocity ---------------------------------------------
        self._dur_min_id,     self._dur_max_id     = _add_range("DUR",          range(cfg.duration_min, cfg.duration_max + 1))
        self._vel_min_id,     self._vel_max_id     = _add_range("VEL",          range(cfg.velocity_bins))

        # ---- Harmony: SCALE_DEGREE + QUALITY + CHORD_N -----------------------
        n_q = min(len(CHORD_QUALITIES), max(1, cfg.chord_qualities))
        self._n_chord_qualities = n_q

        self._sd_min_id,      self._sd_max_id      = _add_range("SCALE_DEGREE", range(12))
        self._quality_min_id, self._quality_max_id = _add_range("QUALITY",      (CHORD_QUALITIES[i] for i in range(n_q)))

        _add("CHORD_N")
        self._chord_n_id = len(self.id_to_token) - 1

        # ---- Key token (optional) --------------------------------------------
        if cfg.use_key_tokens:
            self._key_min_id, self._key_max_id = _add_range(
                "KEY", (f"{r}_{mode}" for mode in KEY_MODES_STR for r in range(12))
            )
        else:
            self._key_min_id = self._key_max_id = -1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.BOS]

    @property
    def sep_id(self) -> int:
        return self.token_to_id[self.SEP]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.EOS]

    def tid(self, token: str) -> int:
        return self.token_to_id[token]

    @property
    def chroma_min_id(self) -> int:
        return self._chroma_min_id

    @property
    def chroma_max_id(self) -> int:
        return self._chroma_max_id

    @property
    def octave_min_id(self) -> int:
        return self._octave_min_id

    @property
    def octave_max_id(self) -> int:
        return self._octave_max_id

    @property
    def octave_base(self) -> int:
        return OCTAVE_MIN

    @property
    def vel_min_id(self) -> int:
        return self._vel_min_id

    @property
    def vel_max_id(self) -> int:
        return self._vel_max_id

    @property
    def dur_min_id(self) -> int:
        return self._dur_min_id

    @property
    def dur_max_id(self) -> int:
        return self._dur_max_id

    @property
    def tempo_min_id(self) -> int:
        return self._tempo_min_id

    @property
    def tempo_max_id(self) -> int:
        return self._tempo_max_id

    @property
    def sd_min_id(self) -> int:
        return self._sd_min_id

    @property
    def sd_max_id(self) -> int:
        return self._sd_max_id

    @property
    def quality_min_id(self) -> int:
        return self._quality_min_id

    @property
    def quality_max_id(self) -> int:
        return self._quality_max_id

    @property
    def chord_n_id(self) -> int:
        return self._chord_n_id

    @property
    def key_min_id(self) -> int:
        return self._key_min_id

    @property
    def key_max_id(self) -> int:
        return self._key_max_id

    def key_token_id(self, root: int, mode: int) -> int:
        if self._key_min_id < 0:
            return -1
        return self._key_min_id + mode * 12 + root

    # ------------------------------------------------------------------
    # Structure / content id helpers
    # ------------------------------------------------------------------
    @property
    def bar_id(self) -> int:
        return self.token_to_id["BAR"]

    @property
    def pos_id_range(self) -> Tuple[int, int]:
        first = self.token_to_id["POS_0"]
        last  = self.token_to_id[f"POS_{self.cfg.positions_per_bar - 1}"]
        return first, last

    @property
    def track_id_range(self) -> Tuple[int, int]:
        first = self.token_to_id[f"TRACK_{self.cfg.tracks[0]}"]
        last  = self.token_to_id[f"TRACK_{self.cfg.tracks[-1]}"]
        return first, last

    def structural_ids(self) -> List[int]:
        ids = [self.bar_id]
        lo, hi = self.pos_id_range
        ids.extend(range(lo, hi + 1))
        lo, hi = self.track_id_range
        ids.extend(range(lo, hi + 1))
        ids.extend(range(self._tempo_min_id, self._tempo_max_id + 1))
        return ids

    def content_ids(self) -> List[int]:
        ids  = list(range(self._chroma_min_id,  self._chroma_max_id  + 1))
        ids += list(range(self._octave_min_id,  self._octave_max_id  + 1))
        ids += list(range(self._dur_min_id,     self._dur_max_id     + 1))
        ids += list(range(self._vel_min_id,     self._vel_max_id     + 1))
        return ids

    def build_token_weight_vector(
        self, struct_weight: float, content_weight: float,
        pos_weight: float | None = None,
    ) -> "list[float]":
        """Per-id weight vector for CE loss scaling.

        Weight assignment:
          structural (BAR/TRACK/TEMPO)      → struct_weight
          POS_n (rhythm — WHEN a note plays)→ pos_weight (defaults to struct_weight)
          content (CHROMA/OCTAVE/DUR/VEL)   → content_weight
          SCALE_DEGREE / QUALITY            → content_weight  (harmonic decisions)
          CHORD_N                           → struct_weight   (placeholder)
          KEY_*                             → struct_weight   (global anchor)
          specials (PAD/BOS/SEP/EOS)        → 1.0

        POS tokens are split out of the structural bucket: they encode the
        rhythmic placement of notes, so up-weighting them (pos_weight >
        struct_weight) teaches the model WHEN to play without touching chord
        size (which the polyphony boost controls independently).
        """
        if pos_weight is None:
            pos_weight = struct_weight
        w = [1.0] * self.vocab_size
        for tid in self.structural_ids():
            w[tid] = struct_weight
        for tid in self.content_ids():
            w[tid] = content_weight
        # POS gets its own weight (override the struct_weight set above)
        pos_lo, pos_hi = self.pos_id_range
        for tid in range(pos_lo, pos_hi + 1):
            w[tid] = pos_weight
        for tid in range(self._sd_min_id, self._sd_max_id + 1):
            w[tid] = content_weight
        for tid in range(self._quality_min_id, self._quality_max_id + 1):
            w[tid] = content_weight
        w[self._chord_n_id] = struct_weight
        if self._key_min_id >= 0:
            for tid in range(self._key_min_id, self._key_max_id + 1):
                w[tid] = struct_weight
        return w

    # ------------------------------------------------------------------
    # Velocity / tempo binning
    # ------------------------------------------------------------------
    def _velocity_bin(self, vel: int) -> int:
        import math
        vel = max(1, min(127, vel))
        b = int(math.log2(vel) / math.log2(128) * self.cfg.velocity_bins)
        return min(self.cfg.velocity_bins - 1, max(0, b))

    def _velocity_from_bin(self, b: int) -> int:
        import math
        b = max(0, min(self.cfg.velocity_bins - 1, b))
        return max(1, min(127, round(2 ** ((b + 0.5) * 7.0 / self.cfg.velocity_bins))))

    def tempo_bin(self, bpm: float) -> int:
        import math
        lo, hi = self.cfg.tempo_min, self.cfg.tempo_max
        bpm = max(lo, min(hi, float(bpm)))
        b = int((math.log(bpm) - math.log(lo)) / (math.log(hi) - math.log(lo)) * self.cfg.tempo_bins)
        return min(self.cfg.tempo_bins - 1, max(0, b))

    def tempo_from_bin(self, b: int) -> float:
        import math
        b = max(0, min(self.cfg.tempo_bins - 1, b))
        lo, hi = self.cfg.tempo_min, self.cfg.tempo_max
        return math.exp(math.log(lo) + (b + 0.5) / self.cfg.tempo_bins * (math.log(hi) - math.log(lo)))

    # ------------------------------------------------------------------
    # Pitch encoding / decoding helpers
    # ------------------------------------------------------------------
    def _pitch_to_chroma_octave(self, pitch: int, key_root: int) -> Tuple[int, int]:
        """Absolute MIDI pitch → (chroma_token_id, octave_token_id)."""
        chroma = (pitch % 12 - key_root) % 12
        octave = pitch // 12
        octave = max(OCTAVE_MIN, min(OCTAVE_MAX, octave))
        return (
            self._chroma_min_id + chroma,
            self._octave_min_id + (octave - OCTAVE_MIN),
        )

    def _chroma_octave_to_pitch(self, chroma_id: int, octave_id: int, key_root: int) -> int:
        """(chroma_token_id, octave_token_id) → absolute MIDI pitch."""
        chroma  = chroma_id  - self._chroma_min_id            # 0-11
        octave  = octave_id  - self._octave_min_id + OCTAVE_MIN  # actual octave
        abs_pc  = (chroma + key_root) % 12
        return octave * 12 + abs_pc

    # ------------------------------------------------------------------
    # Encoding (bar-block temporal interleaving)
    # ------------------------------------------------------------------
    def _group_notes(
        self, events: Sequence[NoteEvent], tracks: "set[str]"
    ) -> "dict[int, dict[int, dict[str, list[NoteEvent]]]]":
        """bar → position → track → sorted note list (pitch/range filtered)."""
        from collections import defaultdict

        _MAX_BAR_GUARD = 2000
        grouped: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for e in events:
            if e.track not in tracks:
                continue
            if e.pitch < self.cfg.pitch_min or e.pitch > self.cfg.pitch_max:
                continue
            if e.bar > _MAX_BAR_GUARD:
                logger.warning(
                    f"encode_song: skipping note at bar={e.bar} > {_MAX_BAR_GUARD} "
                    f"(tick overflow / corrupted MIDI). track={e.track}"
                )
                continue
            grouped[e.bar][e.position][e.track].append(e)
        return grouped

    def _emit_notes_at(
        self,
        grouped,
        bar: int,
        track_order: Sequence[str],
        kr: int,
        ids: List[int],
        mask: List[bool],
        is_target: bool,
        chord_map: "dict | None" = None,
        chord_callback: "Any | None" = None,
    ) -> None:
        """Emit POS / TRACK / note tokens for every position of `bar`.

        When `chord_map` and `chord_callback` are supplied (accompaniment
        section only), a new chord token pair is emitted before any POS token
        at positions > 0 where the chord changes.  The callback's own dedup
        logic ensures no redundant tokens are emitted when the chord is stable.
        """
        bar_positions = grouped.get(bar, {})
        for pos in sorted(bar_positions.keys()):
            # Mid-bar chord injection: check for chord changes at beat boundaries
            # (pos > 0 since the bar-head chord is already emitted by the caller).
            if chord_callback is not None and chord_map is not None and pos > 0:
                chord_callback(chord_map.get((bar, pos), "UNSET"))
            pos_tracks = bar_positions[pos]
            emitted_pos = False
            for track in track_order:
                if track not in pos_tracks:
                    continue
                if not emitted_pos:
                    ids.append(self.tid(f"POS_{pos}")); mask.append(is_target)
                    emitted_pos = True
                ids.append(self.tid(f"TRACK_{track}")); mask.append(is_target)
                for e in sorted(pos_tracks[track], key=lambda e: e.pitch):
                    dur = min(self.cfg.duration_max,
                              max(self.cfg.duration_min, e.duration))
                    chroma_id, octave_id = self._pitch_to_chroma_octave(e.pitch, kr)
                    ids.append(chroma_id);              mask.append(is_target)
                    ids.append(octave_id);              mask.append(is_target)
                    ids.append(self.tid(f"DUR_{dur}"));  mask.append(is_target)
                    ids.append(self.tid(f"VEL_{self._velocity_bin(e.velocity)}"))
                    mask.append(is_target)

    def encode_song(
        self,
        events: Sequence[NoteEvent],
        condition_tracks: Sequence[str],
        target_tracks: Sequence[str],
        tempo_bpm: float | None = None,
        chord_map: "dict[tuple[int, int], tuple[int, int] | None] | None" = None,
        key_root: int | None = None,
        key_mode: int | None = None,
    ) -> Tuple[List[int], List[bool]]:
        """Return (token_ids, target_mask) in bar-block interleaving format.

        The song is split into blocks of ``cfg.lookahead_bars`` consecutive bars.
        Each block is laid out melody-first then accompaniment::

            BAR  POS TRACK_mel [notes]  POS TRACK_mel [notes]   ← block melody
            SEP
            [chord] POS TRACK_acc [notes]  ...                  ← block accompaniment
            (BAR [chord] POS TRACK_acc [notes] ... for 2nd+ bars of the block)

        The model therefore sees the FULL melody of the block before generating
        that block's accompaniment — a ``lookahead_bars``-bar anticipation window.

        Mask: melody section + SEP → False (condition / no loss).
              accompaniment section (incl. its BAR/POS/chord) → True (target).

        key_root: 0-11 (C=0 … B=11). Defaults to 0 (C) when None.
        key_mode: 0=major, 1=minor.  Defaults to 0 when None.
        chord_map: see module docstring; emitted in the accompaniment section.
        """
        kr = int(key_root) if key_root is not None else 0
        km = int(key_mode) if key_mode is not None else 0
        N = max(1, int(getattr(self.cfg, "lookahead_bars", 1)))

        cond_set = set(condition_tracks)
        tgt_set = set(target_tracks)
        track_order = list(condition_tracks) + [
            t for t in target_tracks if t not in cond_set
        ]

        mel_grouped = self._group_notes(events, cond_set)
        acc_grouped = self._group_notes(events, tgt_set)

        # Bars that contain ANY note (melody or accompaniment), chronological.
        active_bars = sorted(set(mel_grouped.keys()) | set(acc_grouped.keys()))

        ids: List[int] = [self.bos_id]
        mask: List[bool] = [False]
        if self._key_min_id >= 0 and key_root is not None:
            ids.append(self.key_token_id(kr, km)); mask.append(False)
        if tempo_bpm is not None:
            ids.append(self.tid(f"TEMPO_{self.tempo_bin(tempo_bpm)}")); mask.append(False)

        # ---- chord emission helper (accompaniment section only) -------------
        last_chord_sig: object = "UNSET"

        def _reset_chord() -> None:
            nonlocal last_chord_sig
            last_chord_sig = "UNSET"

        def _append_chord(chord_val: object) -> None:
            nonlocal last_chord_sig
            if chord_val == "UNSET" or chord_val == last_chord_sig:
                return
            last_chord_sig = chord_val
            if chord_val is None:
                ids.append(self._chord_n_id); mask.append(True)
            else:
                root, q_idx = chord_val  # type: ignore[misc]
                if q_idx >= self._n_chord_qualities:
                    ids.append(self._chord_n_id); mask.append(True)
                    return
                sd = (root - kr) % 12
                ids.append(self._sd_min_id + sd);          mask.append(True)
                ids.append(self._quality_min_id + q_idx);  mask.append(True)

        # ---- iterate blocks of N consecutive (active) bars ------------------
        i = 0
        while i < len(active_bars):
            block_start = active_bars[i]
            block_idx = block_start // N
            # collect every active bar that falls into this block
            bars_in_block: List[int] = []
            while i < len(active_bars) and active_bars[i] // N == block_idx:
                bars_in_block.append(active_bars[i])
                i += 1

            # ---- melody section: BAR + melody notes for each bar ----------
            for bar in bars_in_block:
                ids.append(self.bar_id); mask.append(False)
                self._emit_notes_at(mel_grouped, bar, track_order, kr,
                                    ids, mask, is_target=False)

            # ---- SEP: melody → accompaniment boundary for this block ------
            ids.append(self.sep_id); mask.append(False)

            # ---- accompaniment section: chord + notes for each bar --------
            _reset_chord()
            for j, bar in enumerate(bars_in_block):
                if j > 0:
                    ids.append(self.bar_id); mask.append(True)
                if chord_map is not None:
                    _append_chord(chord_map.get((bar, 0), "UNSET"))
                # Pass chord_map so _emit_notes_at can inject mid-bar chord
                # changes at beat positions > 0 (dedup in _append_chord ensures
                # no redundant tokens when the chord does not change).
                self._emit_notes_at(acc_grouped, bar, track_order, kr,
                                    ids, mask, is_target=True,
                                    chord_map=chord_map,
                                    chord_callback=_append_chord if chord_map is not None else None)

        ids.append(self.eos_id); mask.append(False)
        return ids, mask

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    def decode(self, ids: Iterable[int]) -> List[NoteEvent]:
        """Turn token ids back into NoteEvents (bar-block interleaving format).

        Extracts key_root from the first KEY_* token in the sequence.
        Falls back to key_root=0 (C) if no KEY token is present.

        State machine
        -------------
        Bars are numbered 0-indexed from the first BAR token. Within a block:
          • Melody section BAR tokens advance the absolute bar counter and are
            recorded in ``block_bars``.
          • SEP switches to the accompaniment section and rewinds the bar
            counter to the block's first bar.
          • Accompaniment-section BAR tokens step through ``block_bars``; once
            exhausted, the next BAR starts a new block's melody section.

        This keeps melody and accompaniment of the same bar aligned to the same
        bar index. CHORD_N / SCALE_DEGREE / QUALITY tokens are skipped.
        """
        id_list = [i for i in ids if 0 <= i < self.vocab_size]

        # Extract key_root from first KEY token
        key_root = 0
        if self._key_min_id >= 0:
            for tid in id_list:
                if self._key_min_id <= tid <= self._key_max_id:
                    key_root = (tid - self._key_min_id) % 12
                    break

        events: List[NoteEvent] = []
        cur_track = self.cfg.tracks[0]
        cur_bar   = -1            # first melody BAR brings this to 0
        cur_pos   = 0
        in_acc    = False         # False = melody section, True = accompaniment
        block_bars: List[int] = []
        acc_ptr   = 0
        pending: Dict[str, int] = {}

        def _flush() -> None:
            if "chroma_id" in pending and "octave_id" in pending and "dur" in pending:
                pitch = self._chroma_octave_to_pitch(
                    pending["chroma_id"], pending["octave_id"], key_root
                )
                pitch = max(self.cfg.pitch_min, min(self.cfg.pitch_max, pitch))
                events.append(NoteEvent(
                    track=cur_track,
                    bar=max(0, cur_bar),
                    position=cur_pos,
                    pitch=pitch,
                    duration=pending["dur"],
                    velocity=self._velocity_from_bin(pending.get("vel", 16)),
                ))
            pending.clear()

        for tid in id_list:
            tok = self.id_to_token[tid]
            if tok == self.EOS:
                break
            if tok in (self.PAD, self.BOS):
                continue
            if tok == self.SEP:
                # Melody → accompaniment boundary: rewind to block's first bar.
                _flush()
                in_acc = True
                acc_ptr = 0
                if block_bars:
                    cur_bar = block_bars[0]
                cur_pos = 0
                continue
            if tok.startswith("TRACK_"):
                _flush()
                cur_track = tok[len("TRACK_"):]
            elif tok == "BAR":
                _flush()
                if not in_acc:
                    cur_bar += 1
                    block_bars.append(cur_bar)
                elif acc_ptr + 1 < len(block_bars):
                    acc_ptr += 1
                    cur_bar = block_bars[acc_ptr]
                else:
                    # Accompaniment bars exhausted → new block's melody section.
                    in_acc = False
                    cur_bar = (block_bars[-1] if block_bars else cur_bar) + 1
                    block_bars = [cur_bar]
                    acc_ptr = 0
                cur_pos = 0
            elif tok.startswith("POS_"):
                _flush()
                cur_pos = int(tok.split("_", 1)[1])
            elif tok.startswith("CHROMA_"):
                if "chroma_id" in pending:
                    _flush()
                pending["chroma_id"] = tid
            elif tok.startswith("OCTAVE_"):
                pending["octave_id"] = tid
            elif tok.startswith("DUR_"):
                pending["dur"] = int(tok.split("_", 1)[1])
            elif tok.startswith("VEL_"):
                pending["vel"] = int(tok.split("_", 1)[1])
                _flush()
            # SCALE_DEGREE / QUALITY / CHORD_N / KEY / TEMPO: skip in note decode
        _flush()
        return events
