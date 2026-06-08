"""Controllability sweep: structural_suppression vs. polyphony rate.

structural_suppression 값을 0.0 → 3.0 사이에서 변화시키면서 생성된 반주의
Polyphony Rate (화음 발생 비율)가 어떻게 변하는지를 정량적으로 측정합니다.

이 실험의 목적
--------------
"모델이 무작위로 화음을 생성하는 것이 아니라, 연구자가 지정한 파라미터에
따라 결정론적으로 제어됨"을 보고서/발표에서 증명하기 위한 실험입니다.
파라미터 값 증가에 비례하여 Polyphony Rate가 단조 증가하면 제어 가능성 입증.

출력
----
  <out_dir>/
    controllability_metrics.csv    ← suppression별 수치 (Excel 열람 가능)
    controllability_polyphony.png  ← polyphony rate 꺾은선 그래프
    controllability_notes_per_bar.png  ← notes/bar 꺾은선 그래프
    controllability_combined.png   ← 두 지표 동시 표시 (보고서용)

사용법
------
    # 기본 (기본 sweep 범위: 0.0~3.0, 7점)
    python scripts/controllability_sweep.py \\
        --song 001 \\
        --checkpoint checkpoints/best.ckpt

    # sweep 범위 커스텀 (더 세밀한 간격)
    python scripts/controllability_sweep.py \\
        --song 001 --checkpoint checkpoints/best.ckpt \\
        --supp_values 0.0 0.3 0.6 0.9 1.2 1.5 1.8 2.1 2.5 3.0

    # 통계 안정성 향상: 각 설정 3회 반복 후 평균
    python scripts/controllability_sweep.py \\
        --song 001 --checkpoint checkpoints/best.ckpt \\
        --n_repeats 3
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR.parent.parent / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from jam_transformer.config import load_config
from jam_transformer.lightning_module import JamTransformerLightning
from jam_transformer.utils.logger import logger
from jam_transformer.utils.midi_io import midi_to_events
from jam_transformer.utils.overrides import apply_overrides
from jam_transformer.tokenizer import NoteEvent, build_tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_lit(ckpt_path: str, cfg, vocab_size: int) -> JamTransformerLightning:
    lit = JamTransformerLightning(config=cfg, vocab_size=vocab_size, total_steps=1)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_sd = ckpt.get("state_dict", ckpt)
    if any("._orig_mod." in k for k in raw_sd) and not hasattr(lit.model, "_orig_mod"):
        raw_sd = {k.replace("._orig_mod.", ".", 1): v for k, v in raw_sd.items()}
        logger.info("Stripped _orig_mod. prefix.")
    lit.load_state_dict(raw_sd, strict=False)
    return lit


def _generate_once(
    lit: JamTransformerLightning,
    tokenizer,
    prompt_ids: torch.Tensor,
    device: torch.device,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    structural_suppression: float,
) -> List[NoteEvent]:
    """One generation pass → NoteEvent list (accompaniment only)."""
    prompt = prompt_ids.to(device)
    out = lit.model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        eos_id=tokenizer.eos_id,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        structural_suppression=structural_suppression,
        vel_id_range=(tokenizer.vel_min_id, tokenizer.vel_max_id),
        struct_ids=tokenizer.structural_ids(),
    )[0].cpu().tolist()

    sep_positions = [i for i, t in enumerate(out) if t == tokenizer.sep_id]
    target_ids = out[sep_positions[-1] + 1 :] if sep_positions else out
    return tokenizer.decode(target_ids)


def _polyphony_rate(events: List[NoteEvent]) -> float:
    """Fraction of onset positions that contain >= 2 simultaneous notes."""
    if not events:
        return 0.0
    onset_cnt = Counter((e.bar, e.position) for e in events)
    poly = sum(1 for c in onset_cnt.values() if c >= 2)
    return poly / max(len(onset_cnt), 1)


def _pitch_entropy(events: List[NoteEvent]) -> float:
    if not events:
        return 0.0
    pc_hist = np.zeros(12)
    for e in events:
        pc_hist[e.pitch % 12] += 1
    pc_norm = pc_hist / pc_hist.sum()
    nz = pc_norm[pc_norm > 0]
    return float(-np.sum(nz * np.log2(nz)))


def _measure(events: List[NoteEvent], n_bars: int) -> Dict[str, float]:
    return {
        "n_notes":       len(events),
        "notes_per_bar": len(events) / max(n_bars, 1),
        "polyphony_rate":  _polyphony_rate(events),
        "pitch_entropy":   _pitch_entropy(events),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot_line(
    x: List[float],
    ys: Dict[str, List[float]],
    title: str,
    xlabel: str,
    out_path: Path,
    gt_lines: Optional[Dict[str, float]] = None,
) -> None:
    """Generic line chart with optional GT horizontal reference lines."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (label, vals) in enumerate(ys.items()):
        ax.plot(x, vals, "o-", color=colors[i], label=label, linewidth=2, markersize=6)

    if gt_lines:
        gt_colors = ["#d62728", "#e377c2", "#17becf"]
        for i, (label, val) in enumerate(gt_lines.items()):
            ax.axhline(val, linestyle="--", color=gt_colors[i % len(gt_colors)],
                       linewidth=1.4, alpha=0.75, label=f"{label} (GT)")

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(axis="y", alpha=0.35)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"  plot: {out_path.name}")


def _plot_combined(
    x: List[float],
    poly_mean: List[float],
    poly_std: Optional[List[float]],
    npb_mean: List[float],
    npb_std: Optional[List[float]],
    gt_poly: Optional[float],
    gt_npb: Optional[float],
    out_path: Path,
) -> None:
    """Two-panel figure: polyphony rate (top) + notes/bar (bottom)."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    kw = dict(linewidth=2, markersize=6, color="#1f77b4")

    ax1.plot(x, poly_mean, "o-", **kw, label="Mean")
    if poly_std:
        ax1.fill_between(x,
                         [m - s for m, s in zip(poly_mean, poly_std)],
                         [m + s for m, s in zip(poly_mean, poly_std)],
                         alpha=0.2, color="#1f77b4", label="±1 std")
    if gt_poly is not None:
        ax1.axhline(gt_poly, linestyle="--", color="#d62728",
                    linewidth=1.5, label=f"GT ({gt_poly:.3f})")
    ax1.set_ylabel("Polyphony Rate", fontsize=10)
    ax1.set_title("Controllability: structural_suppression vs. Music Metrics",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    ax2.plot(x, npb_mean, "s-", **kw, label="Mean")
    if npb_std:
        ax2.fill_between(x,
                         [m - s for m, s in zip(npb_mean, npb_std)],
                         [m + s for m, s in zip(npb_mean, npb_std)],
                         alpha=0.2, color="#1f77b4", label="±1 std")
    if gt_npb is not None:
        ax2.axhline(gt_npb, linestyle="--", color="#d62728",
                    linewidth=1.5, label=f"GT ({gt_npb:.2f})")
    ax2.set_xlabel("structural_suppression", fontsize=10)
    ax2.set_ylabel("Notes per bar", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"  plot: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controllability sweep: structural_suppression vs polyphony rate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--song",       required=True,
                        help="POP909 song ID (e.g. 001)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--pop909_dir", default="data/raw/POP909")
    parser.add_argument("--out_dir",    default=None,
                        help="Output directory (default: analysis/controllability_<song>)")
    parser.add_argument(
        "--supp_values", nargs="+", type=float,
        default=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        help="structural_suppression values to sweep (space-separated floats)",
    )
    parser.add_argument("--n_repeats",     type=int,   default=1,
                        help="Repeat each setting N times and report mean ± std. "
                             "Default=1 (fast); 3 recommended for publication.")
    parser.add_argument("--max_new_tokens",type=int,   default=512)
    parser.add_argument("--temperature",   type=float, default=1.0)
    parser.add_argument("--top_k",         type=int,   default=64)
    parser.add_argument("--top_p",         type=float, default=0.95)
    parser.add_argument("--seed",          type=int,   default=42,
                        help="RNG seed for reproducibility.")
    parser.add_argument("--max_bars",      type=int,   default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="SECTION.KEY=VALUE")
    args = parser.parse_args()

    # ── setup ──────────────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    if args.overrides:
        apply_overrides(cfg, args.overrides)

    song_id   = args.song.zfill(3)
    midi_path = Path(args.pop909_dir) / song_id / f"{song_id}.mid"
    if not midi_path.exists():
        raise SystemExit(f"MIDI not found: {midi_path}")

    out_dir = (Path(args.out_dir) if args.out_dir
               else Path("analysis") / f"controllability_{song_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(cfg.tokenizer)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Song {song_id} | device={device} | sweep={args.supp_values}")
    logger.info(f"n_repeats={args.n_repeats} | max_new_tokens={args.max_new_tokens}")

    # ── load model ─────────────────────────────────────────────────────────────
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    lit = _load_lit(args.checkpoint, cfg, tokenizer.vocab_size)
    lit.eval()
    lit.to(device)

    # ── build prompt (once) ────────────────────────────────────────────────────
    all_events, midi_tempo = midi_to_events(midi_path, cfg.tokenizer)
    if args.max_bars is not None:
        all_events = [e for e in all_events if e.bar < args.max_bars]
    melody_ev = [e for e in all_events if e.track == "melody"]
    gt_ev     = [e for e in all_events if e.track != "melody"]
    n_bars    = max((e.bar for e in melody_ev), default=0) + 1

    ids, _ = tokenizer.encode_song(
        all_events, condition_tracks=["melody"], target_tracks=[], tempo_bpm=midi_tempo,
    )
    if ids and ids[-1] == tokenizer.eos_id:
        ids = ids[:-1]
    prompt_ids = torch.tensor(ids, dtype=torch.long)

    # GT metrics (reference lines in plots)
    gt_metrics = _measure(gt_ev, n_bars)
    logger.info(f"GT  | polyphony={gt_metrics['polyphony_rate']:.3f} | "
                f"notes/bar={gt_metrics['notes_per_bar']:.2f} | "
                f"n_notes={gt_metrics['n_notes']}")

    # ── sweep ──────────────────────────────────────────────────────────────────
    rows: List[Dict] = []

    for supp in args.supp_values:
        repeat_metrics: List[Dict] = []
        for rep in range(args.n_repeats):
            seed = args.seed + rep
            torch.manual_seed(seed)
            with torch.no_grad():
                events = _generate_once(
                    lit, tokenizer, prompt_ids, device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    structural_suppression=supp,
                )
            m = _measure(events, n_bars)
            repeat_metrics.append(m)
            logger.info(
                f"  supp={supp:.2f} rep={rep+1}/{args.n_repeats} | "
                f"poly={m['polyphony_rate']:.3f} | "
                f"notes/bar={m['notes_per_bar']:.2f} | "
                f"n={m['n_notes']}"
            )

        # Aggregate
        keys = list(repeat_metrics[0].keys())
        agg: Dict = {"suppression": supp}
        for k in keys:
            vals = [rm[k] for rm in repeat_metrics]
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"]  = float(np.std(vals)) if len(vals) > 1 else 0.0
        rows.append(agg)

        logger.info(
            f"supp={supp:.2f} MEAN | "
            f"poly={agg['polyphony_rate_mean']:.3f}±{agg['polyphony_rate_std']:.3f} | "
            f"notes/bar={agg['notes_per_bar_mean']:.2f}"
        )

    # ── write CSV ──────────────────────────────────────────────────────────────
    csv_path = out_dir / "controllability_metrics.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"CSV: {csv_path}")

    # ── plots ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        logger.warning("matplotlib not installed — skipping plots.")
        return

    x         = [r["suppression"]           for r in rows]
    poly_mean = [r["polyphony_rate_mean"]    for r in rows]
    poly_std  = [r["polyphony_rate_std"]     for r in rows] if args.n_repeats > 1 else None
    npb_mean  = [r["notes_per_bar_mean"]     for r in rows]
    npb_std   = [r["notes_per_bar_std"]      for r in rows] if args.n_repeats > 1 else None
    ent_mean  = [r["pitch_entropy_mean"]     for r in rows]

    # Combined (보고서용 메인 그림)
    _plot_combined(
        x, poly_mean, poly_std, npb_mean, npb_std,
        gt_poly=gt_metrics["polyphony_rate"],
        gt_npb=gt_metrics["notes_per_bar"],
        out_path=out_dir / "controllability_combined.png",
    )

    # Individual plots
    _plot_line(
        x,
        {"Polyphony Rate": poly_mean},
        title="structural_suppression vs. Polyphony Rate",
        xlabel="structural_suppression",
        out_path=out_dir / "controllability_polyphony.png",
        gt_lines={"GT": gt_metrics["polyphony_rate"]},
    )
    _plot_line(
        x,
        {"Notes per bar": npb_mean},
        title="structural_suppression vs. Notes per Bar",
        xlabel="structural_suppression",
        out_path=out_dir / "controllability_notes_per_bar.png",
        gt_lines={"GT": gt_metrics["notes_per_bar"]},
    )
    _plot_line(
        x,
        {"Pitch entropy (bits)": ent_mean},
        title="structural_suppression vs. Pitch Entropy",
        xlabel="structural_suppression",
        out_path=out_dir / "controllability_entropy.png",
        gt_lines={"GT": gt_metrics["pitch_entropy"]},
    )

    # ── summary ────────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Controllability Sweep — Song {song_id}")
    logger.info(f"{'Suppression':>12}  {'Polyphony':>10}  {'Notes/bar':>10}")
    logger.info("-" * 40)
    for r in rows:
        logger.info(
            f"  {r['suppression']:8.2f}    "
            f"{r['polyphony_rate_mean']:8.3f}    "
            f"{r['notes_per_bar_mean']:8.2f}"
        )
    logger.info(f"  {'GT':8s}    "
                f"{gt_metrics['polyphony_rate']:8.3f}    "
                f"{gt_metrics['notes_per_bar']:8.2f}")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
