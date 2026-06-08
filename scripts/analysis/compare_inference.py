"""Accompaniment inference comparison with quantitative metrics.

출력 폴더 구조
--------------
  output/compare_<song>/
    midi/
      00_melody_only.mid
      01_ground_truth.mid
      02_gen_<tag>.mid  ...
    wav/
      00_melody_only.wav
      01_ground_truth.wav
      02_gen_<tag>.wav  ...
      05_comparison.wav       ← 전체 비교 파일 (섹션당 clip_sec초)
    metrics/
      report.txt              ← 사람이 읽는 종합 리포트
      metrics.json            ← 전체 수치 (기계 처리용)
      metrics.csv             ← 스프레드시트용
      plots/
        01_note_density.png
        02_pitch_class_histogram.png
        03_duration_distribution.png
        04_velocity_distribution.png
        05_radar_chart.png

사용법
------
    # 기본 (T/CFG 비교)
    python scripts/compare_inference.py --song 001 --checkpoint ...

    # BPM 비교 모드
    python scripts/compare_inference.py --song 001 --checkpoint ... --tempo_compare

    # 특정 BPM 지정
    python scripts/compare_inference.py --song 001 --checkpoint ... --tempo 120

    # 분석만 (이미 midi/ 폴더 있을 때) — 생성 건너뜀
    python scripts/compare_inference.py --song 001 --checkpoint ... --metrics_only
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import textwrap
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import miditoolkit

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR.parent.parent / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from jam_transformer.config import load_config, TokenizerConfig
from jam_transformer.lightning_module import JamTransformerLightning
from jam_transformer.utils.logger import logger
from jam_transformer.utils.midi_io import events_to_midi, midi_to_events
from jam_transformer.pipeline import estimate_key_from_midi
from jam_transformer.tokenizer import BaseTokenizer, NoteEvent, build_tokenizer
from jam_transformer.utils.overrides import apply_overrides


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def compute_metrics(
    events: List[NoteEvent],
    gt_events: Optional[List[NoteEvent]] = None,
    n_bars: int = 1,
) -> Dict:
    """Compute structural + (optionally) reference-based metrics.

    Structural metrics (always):
      n_notes, notes_per_bar, pitch_mean/std/range, unique_pitches,
      unique_pitch_classes, pitch_entropy_bits, velocity_mean/std,
      duration_mean/std (in position grid units), polyphony_rate,
      interval_mean_semitones, pitch_class_hist (12-dim list)

    Reference metrics (when gt_events is provided):
      vs_gt_pc_cosine     — pitch-class histogram cosine similarity [0,1]
      vs_gt_density_ratio — gen_notes / gt_notes
      vs_gt_pitch_jaccard — pitch-set Jaccard (unique MIDI notes)
      vs_gt_onset_jaccard — (bar,pos) onset-grid Jaccard
    """
    if not events:
        return {"n_notes": 0, "error": "empty"}

    pitches    = np.array([e.pitch    for e in events])
    velocities = np.array([e.velocity for e in events])
    durations  = np.array([e.duration for e in events])
    bars       = np.array([e.bar      for e in events])

    n_notes         = len(events)
    n_bars_covered  = int(bars.max() - bars.min() + 1)
    notes_per_bar   = n_notes / max(n_bars, 1)

    pc_hist = np.zeros(12)
    for p in pitches:
        pc_hist[p % 12] += 1
    pc_norm = pc_hist / pc_hist.sum()
    pc_nz   = pc_norm[pc_norm > 0]
    pitch_entropy = float(-np.sum(pc_nz * np.log2(pc_nz)))

    onsets       = [(e.bar, e.position) for e in events]
    onset_cnt    = Counter(onsets)
    poly_count   = sum(1 for c in onset_cnt.values() if c > 1)
    polyphony    = poly_count / max(len(onset_cnt), 1)

    sorted_ev = sorted(events, key=lambda e: (e.bar, e.position, e.pitch))
    intervals = [abs(sorted_ev[i+1].pitch - sorted_ev[i].pitch)
                 for i in range(len(sorted_ev) - 1)]

    result: Dict = {
        "n_notes":                 n_notes,
        "n_bars_covered":          n_bars_covered,
        "notes_per_bar":           round(notes_per_bar, 2),
        "pitch_mean":              round(float(pitches.mean()), 1),
        "pitch_std":               round(float(pitches.std()),  1),
        "pitch_range":             int(pitches.max() - pitches.min()),
        "unique_pitches":          int(len(np.unique(pitches))),
        "unique_pitch_classes":    int((pc_hist > 0).sum()),
        "pitch_entropy_bits":      round(pitch_entropy, 3),
        "velocity_mean":           round(float(velocities.mean()), 1),
        "velocity_std":            round(float(velocities.std()),  1),
        "duration_mean_pos":       round(float(durations.mean()), 2),
        "duration_std_pos":        round(float(durations.std()),  2),
        "polyphony_rate":          round(polyphony, 3),
        "interval_mean_semitones": round(float(np.mean(intervals)) if intervals else 0.0, 2),
        "pitch_class_hist":        pc_norm.tolist(),
    }

    if gt_events:
        gt_p    = np.array([e.pitch for e in gt_events])
        gt_pc   = np.zeros(12)
        for p in gt_p:
            gt_pc[p % 12] += 1
        gt_norm = gt_pc / max(gt_pc.sum(), 1)

        # cosine similarity of 12-dim PC histogram
        denom = (np.linalg.norm(pc_norm) * np.linalg.norm(gt_norm) + 1e-9)
        pc_cos = float(np.dot(pc_norm, gt_norm) / denom)

        gen_onset_set = set(onsets)
        gt_onset_set  = set((e.bar, e.position) for e in gt_events)
        onset_jac = len(gen_onset_set & gt_onset_set) / max(
            len(gen_onset_set | gt_onset_set), 1)

        gen_pset = set(int(p) for p in pitches)
        gt_pset  = set(int(p) for p in gt_p)
        pitch_jac = len(gen_pset & gt_pset) / max(len(gen_pset | gt_pset), 1)

        result.update({
            "vs_gt_pc_cosine":     round(pc_cos,     4),
            "vs_gt_density_ratio": round(n_notes / max(len(gt_events), 1), 3),
            "vs_gt_pitch_jaccard": round(pitch_jac,  4),
            "vs_gt_onset_jaccard": round(onset_jac,  4),
        })

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def _col(v, w=10) -> str:
    """Right-align a value in a fixed-width column."""
    return str(v).rjust(w)


def build_report(
    song_id: str,
    base_tempo: float,
    n_bars: int,
    labels: List[str],     # e.g. ["GT", "Gen T=1.0", ...]
    all_metrics: List[Dict],
    generated_at: str,
) -> str:
    W = 14
    sep = "=" * (16 + W * len(labels))
    thin = "-" * (16 + W * len(labels))

    hdr = "".join(_col(lb[:W-1], W) for lb in labels)

    def row(name: str, key: str, fmt: str = "{}") -> str:
        vals = ""
        for m in all_metrics:
            v = m.get(key, "N/A")
            if v != "N/A":
                try:
                    v = fmt.format(v)
                except Exception:
                    v = str(v)
            vals += _col(v, W)
        return f"  {name:<20}{vals}"

    # PC histogram dominant notes
    def dominant_pc(m: Dict, n: int = 3) -> str:
        hist = m.get("pitch_class_hist", [0]*12)
        top  = sorted(range(12), key=lambda i: -hist[i])[:n]
        return "/".join(NOTE_NAMES[i] for i in top)

    pc_dominant_row = f"  {'Dominant PCs':<20}" + "".join(
        _col(dominant_pc(m), W) for m in all_metrics
    )

    lines = [
        sep,
        f"  COMPARISON REPORT  Song {song_id} | {base_tempo:.0f} BPM | {n_bars} bars",
        f"  Generated: {generated_at}",
        sep,
        "",
        "  [ Structural Metrics ]",
        thin,
        f"  {'Metric':<20}{hdr}",
        thin,
        row("Notes",            "n_notes"),
        row("Notes / bar",      "notes_per_bar",        "{:.1f}"),
        row("Pitch mean",       "pitch_mean",            "{:.1f}"),
        row("Pitch std",        "pitch_std",             "{:.1f}"),
        row("Pitch range",      "pitch_range"),
        row("Unique pitches",   "unique_pitches"),
        row("Unique PCs",       "unique_pitch_classes"),
        row("Pitch entropy",    "pitch_entropy_bits",    "{:.2f}"),
        pc_dominant_row,
        row("Velocity mean",    "velocity_mean",         "{:.1f}"),
        row("Velocity std",     "velocity_std",          "{:.1f}"),
        row("Dur mean (grid)",  "duration_mean_pos",     "{:.2f}"),
        row("Dur std (grid)",   "duration_std_pos",      "{:.2f}"),
        row("Polyphony rate",   "polyphony_rate",        "{:.3f}"),
        row("Interval mean st", "interval_mean_semitones", "{:.2f}"),
        thin,
        "",
    ]

    # Reference metrics (skip GT column itself)
    ref_labels  = labels[1:]   # skip "GT"
    ref_metrics = all_metrics[1:]
    if ref_metrics and "vs_gt_pc_cosine" in ref_metrics[0]:
        ref_hdr = "".join(_col(lb[:W-1], W) for lb in ref_labels)
        def ref_row(name: str, key: str, fmt: str = "{}") -> str:
            vals = ""
            for m in ref_metrics:
                v = m.get(key, "N/A")
                if v != "N/A":
                    try:
                        v = fmt.format(v)
                    except Exception:
                        v = str(v)
                vals += _col(v, W)
            return f"  {name:<20}{vals}"

        lines += [
            "  [ Reference Metrics vs. Ground Truth ]",
            thin,
            f"  {'Metric':<20}{ref_hdr}",
            thin,
            ref_row("PC cosine sim",   "vs_gt_pc_cosine",     "{:.4f}"),
            ref_row("Density ratio",   "vs_gt_density_ratio", "{:.3f}"),
            ref_row("Pitch Jaccard",   "vs_gt_pitch_jaccard", "{:.4f}"),
            ref_row("Onset Jaccard",   "vs_gt_onset_jaccard", "{:.4f}"),
            thin,
            "",
            "  Notes:",
            "    PC cosine sim   : pitch-class histogram cosine similarity [0=different, 1=identical]",
            "    Density ratio   : gen_notes / gt_notes  (1.0 = same density as GT)",
            "    Pitch Jaccard   : intersection/union of used MIDI note sets",
            "    Onset Jaccard   : intersection/union of (bar,position) onset grids",
            thin,
        ]

    lines.append(sep)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _safe_color_cycle(n: int):
    import matplotlib.pyplot as plt
    prop = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return [prop[i % len(prop)] for i in range(n)]


def plot_note_density(labels, all_metrics, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = [m.get("notes_per_bar", 0) for m in all_metrics]
    colors = _safe_color_cycle(len(labels))
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.set_ylabel("Notes per bar")
    ax.set_title("Note Density Comparison")
    ax.set_ylim(0, max(vals) * 1.3 + 1)
    plt.xticks(rotation=15, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pitch_class_histogram(labels, all_metrics, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    n = len(labels)
    x = np.arange(12)
    width = 0.8 / n
    colors = _safe_color_cycle(n)
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, (lb, m) in enumerate(zip(labels, all_metrics)):
        hist = m.get("pitch_class_hist", [0]*12)
        ax.bar(x + i * width - 0.4 + width/2, hist, width,
               label=lb, color=colors[i], edgecolor="black", linewidth=0.4, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(NOTE_NAMES)
    ax.set_ylabel("Proportion")
    ax.set_title("Pitch Class Distribution")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_duration_distribution(labels, events_list, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    colors = _safe_color_cycle(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    for lb, evs, col in zip(labels, events_list, colors):
        if not evs:
            continue
        durs = [e.duration for e in evs]
        max_dur = max(durs)
        bins = np.arange(0.5, max_dur + 1.5)
        ax.hist(durs, bins=bins, density=True, label=lb, color=col,
                alpha=0.55, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Duration (grid positions)")
    ax.set_ylabel("Density")
    ax.set_title("Note Duration Distribution")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_velocity_distribution(labels, events_list, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    colors = _safe_color_cycle(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    for lb, evs, col in zip(labels, events_list, colors):
        if not evs:
            continue
        vels = [e.velocity for e in evs]
        ax.hist(vels, bins=20, range=(0, 128), density=True, label=lb,
                color=col, alpha=0.55, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Velocity (0–127)")
    ax.set_ylabel("Density")
    ax.set_title("Velocity Distribution")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_radar(labels, all_metrics, out_path: Path) -> None:
    """Radar chart with normalized structural metrics."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch  # noqa: F401

    radar_keys = [
        ("Notes/bar",         "notes_per_bar",           None),
        ("Pitch entropy",     "pitch_entropy_bits",       None),
        ("Pitch range",       "pitch_range",              None),
        ("Unique PCs",        "unique_pitch_classes",     None),
        ("Polyphony",         "polyphony_rate",           None),
        ("Interval mean",     "interval_mean_semitones",  None),
    ]
    n_axes = len(radar_keys)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    # Normalise each axis to [0, 1] across all variants
    raw = {k: [m.get(k, 0) for m in all_metrics] for _, k, _ in radar_keys}
    ranges = {k: (min(vs), max(vs)) for k, vs in raw.items()}

    def _norm(k, v):
        lo, hi = ranges[k]
        return (v - lo) / (hi - lo + 1e-9)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    colors = _safe_color_cycle(len(labels))

    for lb, m, col in zip(labels, all_metrics, colors):
        vals = [_norm(k, m.get(k, 0)) for _, k, _ in radar_keys]
        vals += vals[:1]
        ax.plot(angles, vals, color=col, linewidth=1.5, label=lb)
        ax.fill(angles, vals, color=col, alpha=0.12)

    ax.set_thetagrids(np.degrees(angles[:-1]), [n for n, *_ in radar_keys], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7)
    ax.set_title("Structural Metrics Radar\n(normalised across variants)", pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_plots(
    labels: List[str],
    all_metrics: List[Dict],
    events_list: List[List[NoteEvent]],
    plots_dir: Path,
) -> List[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        logger.warning("matplotlib not available — skipping plots.")
        return []

    plots_dir.mkdir(parents=True, exist_ok=True)
    created = []
    tasks = [
        ("01_note_density.png",        lambda p: plot_note_density(labels, all_metrics, p)),
        ("02_pitch_class_histogram.png",lambda p: plot_pitch_class_histogram(labels, all_metrics, p)),
        ("03_duration_distribution.png",lambda p: plot_duration_distribution(labels, events_list, p)),
        ("04_velocity_distribution.png",lambda p: plot_velocity_distribution(labels, events_list, p)),
        ("05_radar_chart.png",          lambda p: plot_radar(labels, all_metrics, p)),
    ]
    for fname, fn in tasks:
        p = plots_dir / fname
        try:
            fn(p)
            created.append(p)
            logger.info(f"  plot: {p.name}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  plot failed ({fname}): {e}")
    return created


# ─────────────────────────────────────────────────────────────────────────────
# WAV rendering + audio FX
# ─────────────────────────────────────────────────────────────────────────────

_GM_DLS = Path("C:/Windows/system32/drivers/gm.dls")


def apply_audio_fx(
    wav_path: Path,
    *,
    room_size: float = 0.35,
    wet_level: float = 0.20,
    dry_level: float = 0.85,
    threshold_db: float = -18.0,
    ratio: float = 4.0,
    limiter_ceiling_db: float = -1.0,
) -> bool:
    """Apply a Reverb → Compressor → Limiter chain with Pedalboard.

    Parameters
    ----------
    wav_path : Path
        The WAV file to process **in-place**.
    room_size : float
        Reverb room size [0, 1].  0.35 gives a natural small-room ambience.
    wet_level : float
        Reverb wet signal proportion.  0.20 adds space without muddiness.
    dry_level : float
        Reverb dry signal proportion.  Keep > wet for clarity.
    threshold_db : float
        Compressor threshold.  -18 dB catches dynamic peaks in piano/strings.
    ratio : float
        Compressor ratio.  4:1 is gentle — audibly evens out the dynamics.
    limiter_ceiling_db : float
        Hard limiter output ceiling.  -1 dBFS prevents inter-sample clipping.

    Returns
    -------
    bool
        True on success, False if pedalboard / soundfile is unavailable or the
        file does not exist (caller can safely ignore the return value).
    """
    if not wav_path.exists():
        return False
    try:
        from pedalboard import Pedalboard, Reverb, Compressor, Limiter  # type: ignore
        import soundfile as sf
    except ImportError:
        return False  # pedalboard not installed — silently skip

    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    # Pedalboard expects (channels, samples)
    audio_ch = audio.T

    board = Pedalboard([
        Reverb(
            room_size=room_size,
            wet_level=wet_level,
            dry_level=dry_level,
            damping=0.5,
            width=0.9,
        ),
        Compressor(
            threshold_db=threshold_db,
            ratio=ratio,
            attack_ms=10.0,
            release_ms=200.0,
        ),
        Limiter(threshold_db=limiter_ceiling_db),
    ])

    processed = board(audio_ch, sr)          # shape: (channels, samples)
    sf.write(str(wav_path), processed.T, sr)
    return True


def midi_to_wav(
    midi_path: Path,
    wav_path: Path,
    soundfont: Optional[Path] = None,
    sr: int = 44100,
) -> str:
    try:
        import pretty_midi
        import soundfile as sf_lib
    except ImportError:
        raise RuntimeError("pretty_midi / soundfile not installed")

    sf_path: Optional[str] = None
    if soundfont and soundfont.exists():
        sf_path = str(soundfont)
    elif _GM_DLS.exists():
        sf_path = str(_GM_DLS)

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio = pm.fluidsynth(fs=sr, sf2_path=sf_path)
        backend = f"FluidSynth ({Path(sf_path).name if sf_path else 'built-in'})"
    except Exception:
        audio = pm.synthesize(fs=sr)
        backend = "pretty_midi (sine)"

    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    sf_lib.write(str(wav_path), audio, sr)
    return backend


def _load_wav(wav_path: Path) -> Tuple[np.ndarray, int]:
    import soundfile as sf
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, sr


def build_comparison_wav(
    sections: List[Tuple[str, Path]],
    out_path: Path,
    gap_sec: float = 1.5,
    clip_sec: Optional[float] = 25.0,
) -> None:
    import soundfile as sf
    if not sections:
        return

    ref_sr: Optional[int] = None
    loaded: List[Tuple[str, np.ndarray]] = []
    for label, wav_path in sections:
        if not wav_path.exists():
            continue
        audio, sr = _load_wav(wav_path)
        ref_sr = ref_sr or sr
        if clip_sec:
            audio = audio[:int(clip_sec * sr)]
        loaded.append((label, audio))

    if not loaded:
        return

    sr = ref_sr
    gap       = np.zeros(int(gap_sec * sr), dtype=np.float32)
    beep_t    = np.linspace(0, 0.07, int(0.07 * sr), endpoint=False)
    beep      = (0.22 * np.sin(2 * np.pi * 880 * beep_t)).astype(np.float32)
    short_gap = np.zeros(int(0.1 * sr), dtype=np.float32)

    chunks: List[np.ndarray] = []
    for idx, (_, audio) in enumerate(loaded):
        n_beeps = idx + 1
        marker = np.concatenate(
            [np.concatenate([beep, short_gap]) for _ in range(n_beeps)] + [short_gap]
        )
        chunks += [marker, audio, gap]

    combined = np.concatenate(chunks).astype(np.float32)
    peak = np.abs(combined).max()
    if peak > 0:
        combined /= peak
    sf.write(str(out_path), combined, sr)
    clip_note = f", {clip_sec:.0f}s/section" if clip_sec else ""
    logger.info(f"Comparison WAV: {out_path.name}  ({len(combined)/sr:.1f}s total{clip_note})")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint / model helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_lit(ckpt_path: str, cfg, vocab_size: int) -> JamTransformerLightning:
    lit = JamTransformerLightning(config=cfg, vocab_size=vocab_size, total_steps=1)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_sd = ckpt.get("state_dict", ckpt)
    if any("._orig_mod." in k for k in raw_sd) and not hasattr(lit.model, "_orig_mod"):
        raw_sd = {k.replace("._orig_mod.", ".", 1): v for k, v in raw_sd.items()}
        logger.info("Stripped _orig_mod. prefix (torch.compile artifact).")
    missing, unexpected = lit.load_state_dict(raw_sd, strict=False)
    if missing or unexpected:
        logger.warning(f"State dict: {len(missing)} missing, {len(unexpected)} unexpected.")
    else:
        logger.info("State dict loaded cleanly.")
    return lit


def _build_prompt(
    midi_path: Path,
    tokenizer: BaseTokenizer,
    cond_tracks: List[str],
    tempo_override: Optional[float] = None,
) -> Tuple[torch.Tensor, float]:
    events, midi_tempo = midi_to_events(midi_path, tokenizer.cfg)
    tempo = tempo_override if tempo_override is not None else midi_tempo
    key_root: Optional[int] = None
    key_mode: Optional[int] = None
    kresult = estimate_key_from_midi(midi_path)
    if kresult is not None:
        key_root, key_mode = kresult
    ids, _ = tokenizer.encode_song(
        events, condition_tracks=cond_tracks, target_tracks=[], tempo_bpm=tempo,
        key_root=key_root, key_mode=key_mode,
    )
    if ids and ids[-1] == tokenizer.eos_id:
        ids = ids[:-1]
    return torch.tensor(ids, dtype=torch.long), tempo


def _generate(
    lit: JamTransformerLightning,
    tokenizer: BaseTokenizer,
    prompt: torch.Tensor,
    device: torch.device,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    cfg_w: float = 0.0,
    structural_suppression: float = 0.0,
) -> List[int]:
    prompt = prompt.to(device)
    uncond_ids = None
    if cfg_w > 0.0:
        uncond_ids = torch.tensor(
            tokenizer.make_uncond_prompt(prompt), dtype=torch.long, device=device,
        )
    out = lit.model.generate(
        prompt, max_new_tokens=max_new_tokens, eos_id=tokenizer.eos_id,
        temperature=temperature, top_k=top_k, top_p=top_p,
        uncond_prompt_ids=uncond_ids, cfg_w=cfg_w,
        structural_suppression=structural_suppression,
        vel_id_range=(tokenizer.vel_min_id, tokenizer.vel_max_id),
        struct_ids=tokenizer.structural_ids(),
    )[0].cpu().tolist()
    sep_positions = [i for i, t in enumerate(out) if t == tokenizer.sep_id]
    target_start = sep_positions[-1] + 1 if sep_positions else 0
    return out[target_start:]


def _max_bar(events: Sequence[NoteEvent]) -> int:
    return max((e.bar for e in events), default=0)


def _make_midi(
    melody: List[NoteEvent],
    target: List[NoteEvent],
    cfg_tok: TokenizerConfig,
    tempo: float,
) -> miditoolkit.MidiFile:
    return events_to_midi([*melody, *target], cfg_tok, tempo_bpm=tempo)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(__doc__.split("사용법")[0]),
    )
    parser.add_argument("--song",            required=True)
    parser.add_argument("--checkpoint",      required=True)
    parser.add_argument("--config",          default="configs/config.yaml")
    parser.add_argument("--pop909_dir",      default="data/raw/POP909")
    parser.add_argument("--out_dir",         default=None)
    parser.add_argument("--max_bars",        type=int,   default=None)
    parser.add_argument("--max_new_tokens",  type=int,   default=512)
    parser.add_argument("--tempo",           type=float, default=None,
                        help="BPM override (default: from MIDI)")
    parser.add_argument("--tempo_compare",   action="store_true",
                        help="Generate slow/original/fast BPM variants")
    parser.add_argument("--clip_sec",        type=float, default=25.0,
                        help="Seconds per section in comparison WAV (0=full)")
    parser.add_argument("--struct_suppression", type=float, default=None,
                        help="Polyphony hack: subtract from BAR/POS logits "
                             "after each note triple. Default: inference.structural_suppression.")
    parser.add_argument("--no_wav",          action="store_true")
    parser.add_argument("--no_fx",           action="store_true",
                        help="Skip Pedalboard audio FX (reverb/compressor/limiter). "
                             "Default: apply FX when pedalboard is installed.")
    parser.add_argument("--metrics_only",    action="store_true",
                        help="Skip generation; compute metrics from existing MIDIs")
    parser.add_argument("--sr",              type=int,   default=44100)
    parser.add_argument("--soundfont",       default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="SECTION.KEY=VALUE")
    args = parser.parse_args()

    # ── paths ──────────────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    if args.overrides:
        apply_overrides(cfg, args.overrides)

    song_id   = args.song.zfill(3)
    midi_path = Path(args.pop909_dir) / song_id / f"{song_id}.mid"
    if not midi_path.exists():
        logger.error(f"MIDI not found: {midi_path}"); sys.exit(1)

    out_dir    = Path(args.out_dir) if args.out_dir else Path("output") / f"compare_{song_id}"
    midi_dir   = out_dir / "midi"
    wav_dir    = out_dir / "wav"
    metrics_dir= out_dir / "metrics"
    plots_dir  = metrics_dir / "plots"
    for d in (midi_dir, wav_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    soundfont_path = Path(args.soundfont) if args.soundfont else None
    tokenizer      = build_tokenizer(cfg.tokenizer)
    cond_tracks    = ["melody"]
    accom_tracks   = [t for t in cfg.tokenizer.tracks if t != "melody"]

    # ── source MIDI ────────────────────────────────────────────────────────────
    all_events, midi_tempo = midi_to_events(midi_path, cfg.tokenizer)
    base_tempo = args.tempo if args.tempo else midi_tempo
    melody_ev  = [e for e in all_events if e.track in cond_tracks]
    gt_ev      = [e for e in all_events if e.track in accom_tracks]

    if args.max_bars is not None:
        melody_ev = [e for e in melody_ev if e.bar < args.max_bars]
        gt_ev     = [e for e in gt_ev     if e.bar < args.max_bars]

    n_bars = _max_bar(melody_ev) + 1
    logger.info(f"Song {song_id} | {base_tempo:.0f} BPM | {n_bars} bars | "
                f"melody {len(melody_ev)} notes | GT accom {len(gt_ev)} notes")

    # ── generation variants definition ────────────────────────────────────────
    if args.tempo_compare:
        t_slow, t_orig, t_fast = (
            max(50,  int(base_tempo * 0.70)),
            int(base_tempo),
            min(200, int(base_tempo * 1.40)),
        )
        VARIANTS = [
            dict(label=f"Slow {t_slow}BPM",  tag=f"bpm{t_slow}",
                 temp=1.0, top_k=64, top_p=0.95, cfg_w=0.0, bpm=float(t_slow)),
            dict(label=f"Orig {t_orig}BPM",  tag=f"bpm{t_orig}",
                 temp=1.0, top_k=64, top_p=0.95, cfg_w=0.0, bpm=float(t_orig)),
            dict(label=f"Fast {t_fast}BPM",  tag=f"bpm{t_fast}",
                 temp=1.0, top_k=64, top_p=0.95, cfg_w=0.0, bpm=float(t_fast)),
        ]
    else:
        VARIANTS = [
            dict(label="Gen T=1.0", tag="t10_k64",
                 temp=1.0, top_k=64, top_p=0.95, cfg_w=0.0, bpm=base_tempo),
            dict(label="Gen T=0.7", tag="t07_k64",
                 temp=0.7, top_k=64, top_p=0.90, cfg_w=0.0, bpm=base_tempo),
            dict(label="Gen CFG2",  tag="cfg20",
                 temp=1.0, top_k=64, top_p=0.95, cfg_w=2.0, bpm=base_tempo),
        ]

    # ── generation (or load from disk if --metrics_only) ──────────────────────
    gen_results: List[Tuple[str, str, List[NoteEvent], float]] = []

    if args.metrics_only:
        logger.info("--metrics_only: loading events from existing MIDIs in midi/")
        for v in VARIANTS:
            mid_p = midi_dir / f"02_gen_{v['tag']}.mid"
            if mid_p.exists():
                evs, _ = midi_to_events(mid_p, cfg.tokenizer)
                target_evs = [e for e in evs if e.track in accom_tracks]
                gen_results.append((v["label"], v["tag"], target_evs, float(v["bpm"])))
            else:
                logger.warning(f"  missing: {mid_p}")
    else:
        lit = _load_lit(args.checkpoint, cfg, tokenizer.vocab_size)
        lit.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lit.to(device)
        logger.info(f"Device: {device}")

        for v in VARIANTS:
            v_bpm = float(v["bpm"])
            logger.info(f"Generating [{v['label']}] ...")
            prompt_ids, _ = _build_prompt(
                midi_path, tokenizer, cond_tracks, tempo_override=v_bpm,
            )
            if args.max_bars is not None:
                total_bars = _max_bar(all_events) + 1
                frac = args.max_bars / max(total_bars, 1)
                prompt_ids = prompt_ids[:max(4, int(len(prompt_ids) * frac))]
            struct_supp = (args.struct_suppression
                           if args.struct_suppression is not None
                           else getattr(cfg.inference, "structural_suppression", 0.0))
            with torch.no_grad():
                ids = _generate(
                    lit, tokenizer, prompt_ids, device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=float(v["temp"]), top_k=int(v["top_k"]),
                    top_p=float(v["top_p"]), cfg_w=float(v["cfg_w"]),
                    structural_suppression=float(struct_supp),
                )
            evs = tokenizer.decode(ids)
            gen_results.append((v["label"], v["tag"], evs, v_bpm))
            logger.info(f"  -> {len(evs)} notes @ {v_bpm:.0f} BPM")

    # ── write MIDI files ───────────────────────────────────────────────────────
    def _save_midi(name: str, mel: list, tgt: list, bpm: float) -> Path:
        mid_p = midi_dir / f"{name}.mid"
        _make_midi(mel, tgt, cfg.tokenizer, bpm).dump(str(mid_p))
        return mid_p

    mid_melody = _save_midi("00_melody_only",  melody_ev, [],    base_tempo)
    mid_gt     = _save_midi("01_ground_truth", melody_ev, gt_ev, base_tempo)
    gen_midis  = []
    for i, (label, tag, evs, v_bpm) in enumerate(gen_results):
        gen_midis.append(_save_midi(f"{i+2:02d}_gen_{tag}", melody_ev, evs, v_bpm))

    # ── WAV rendering ──────────────────────────────────────────────────────────
    render_backend: Optional[str] = None
    wav_section_pairs: List[Tuple[str, Path]] = []

    if not args.no_wav:
        logger.info("Rendering WAV files ...")
        all_midis = [
            ("00 Melody only", mid_melody),
            ("01 Ground Truth", mid_gt),
        ] + [(f"{i+2:02d} {label}", mp)
             for i, (label, tag, evs, _) in enumerate(gen_results)
             for mp in [gen_midis[i]]]

        # Check pedalboard availability once
        _fx_available = False
        if not args.no_fx:
            try:
                import pedalboard  # noqa: F401
                _fx_available = True
                logger.info("  Pedalboard FX: reverb + compressor + limiter (--no_fx to disable)")
            except ImportError:
                logger.info("  pedalboard not installed — raw FluidSynth output. "
                            "Install with: pip install pedalboard")

        for label, mid_p in all_midis:
            wav_p = wav_dir / mid_p.with_suffix(".wav").name
            try:
                backend = midi_to_wav(mid_p, wav_p, soundfont_path, args.sr)
                render_backend = render_backend or backend
                if _fx_available:
                    apply_audio_fx(wav_p)
                    backend += " + FX"
                wav_section_pairs.append((label, wav_p))
                logger.info(f"  {wav_p.name}  [{backend}]")
            except RuntimeError as e:
                logger.error(f"  FAILED {mid_p.name}: {e}")

        if wav_section_pairs:
            comp_wav = wav_dir / "05_comparison.wav"
            clip = args.clip_sec if args.clip_sec and args.clip_sec > 0 else None
            build_comparison_wav(wav_section_pairs, comp_wav, clip_sec=clip)

    # ── metrics ────────────────────────────────────────────────────────────────
    logger.info("Computing metrics ...")

    # Collect (label, events) for GT + all gen variants
    labels_all     = ["GT"] + [r[0] for r in gen_results]
    events_all     = [gt_ev] + [r[2] for r in gen_results]
    all_metrics    = []
    for i, (lb, evs) in enumerate(zip(labels_all, events_all)):
        gt_ref = gt_ev if i > 0 else None
        m = compute_metrics(evs, gt_events=gt_ref, n_bars=n_bars)
        m["label"] = lb
        all_metrics.append(m)

    # metrics.json
    json_path = metrics_dir / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"song": song_id, "tempo_bpm": base_tempo, "n_bars": n_bars,
                   "metrics": all_metrics}, f, indent=2, ensure_ascii=False)

    # metrics.csv — one row per variant, key metrics as columns
    csv_keys = [
        "label", "n_notes", "notes_per_bar", "pitch_mean", "pitch_std",
        "pitch_range", "unique_pitches", "unique_pitch_classes",
        "pitch_entropy_bits", "velocity_mean", "velocity_std",
        "duration_mean_pos", "duration_std_pos", "polyphony_rate",
        "interval_mean_semitones",
        "vs_gt_pc_cosine", "vs_gt_density_ratio",
        "vs_gt_pitch_jaccard", "vs_gt_onset_jaccard",
    ]
    csv_path = metrics_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_keys, extrasaction="ignore")
        writer.writeheader()
        for m in all_metrics:
            writer.writerow({k: m.get(k, "") for k in csv_keys})

    # report.txt
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = build_report(song_id, base_tempo, n_bars, labels_all, all_metrics, now)
    report_path = metrics_dir / "report.txt"
    report_path.write_text(report, encoding="utf-8")

    # plots
    plot_paths = generate_plots(labels_all, all_metrics, events_all, plots_dir)

    # ── print summary ──────────────────────────────────────────────────────────
    print()
    print(report)
    print()

    # file tree
    print("  [ Output structure ]")
    print(f"  {out_dir}/")
    print(f"    midi/   {len(list(midi_dir.glob('*.mid')))} MIDI files")
    if not args.no_wav:
        print(f"    wav/    {len(list(wav_dir.glob('*.wav')))} WAV files"
              + (f"  [{render_backend}]" if render_backend else ""))
    print(f"    metrics/")
    print(f"      report.txt   <- human-readable")
    print(f"      metrics.json <- full data")
    print(f"      metrics.csv  <- spreadsheet")
    if plot_paths:
        print(f"      plots/  {len(plot_paths)} PNG charts")
    print(f"  {out_dir.resolve()}")


if __name__ == "__main__":
    main()
