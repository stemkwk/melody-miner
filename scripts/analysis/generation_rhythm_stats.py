"""Autoregressive-generation rhythm/harmony diagnostic.

The W&B/CSV metrics are *teacher-forced* (model predicts each token given the
GROUND-TRUTH prefix). They were healthy (pos0~0.11 == GT) yet the audible
output collapsed to beat-1 — the classic teacher-forced vs free-running gap
(exposure bias). This script measures the metrics that actually matter: the
distribution of the model's OWN autoregressively generated accompaniment.

For each test song it:
  1. reads melody + original accompaniment,
  2. encodes the ground-truth song -> (ids, mask) -> GT stats,
  3. autoregressively GENERATES accompaniment from the melody, encodes the
     result the same way -> GEN stats,
and reports GT vs GEN (mean over songs). A large pos0/entropy gap = the
collapse is a *generation-time* problem, not a learned-distribution problem.

Usage
-----
    python scripts/analysis/generation_rhythm_stats.py \
        --checkpoint "checkpoints/best-epoch=009-val_loss=0.7623.ckpt" \
        --pop909 data/raw/POP909 -n 20

    # sweep sampling temperature to see if it drives the collapse
    python scripts/analysis/generation_rhythm_stats.py --checkpoint ... --temperature 0.7
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent / "src"))

from jam_transformer.config import load_config                      # noqa: E402
from jam_transformer.tokenizer import build_tokenizer               # noqa: E402
from jam_transformer.pipeline import (                              # noqa: E402
    load_checkpoint, generate_accompaniment, estimate_key_from_midi,
)
from jam_transformer.utils.midi_io import midi_to_events, events_to_midi  # noqa: E402


def compute_stats(ids, mask, tok) -> dict:
    """Mirror lightning_module._quality_metrics, on a single (ids, mask) seq.

    x=ids[:-1], y=ids[1:], m=mask[1:] reproduces the (input, target, loss_mask)
    triple the training metric uses, so numbers are directly comparable to the
    CSV's gt_* columns.
    """
    ids = np.asarray(ids)
    mask = np.asarray(mask, dtype=bool)
    x, y, m = ids[:-1], ids[1:], mask[1:]
    yf, xf = y[m], x[m]

    plo, phi = tok.pos_id_range
    clo, chi = tok.chroma_min_id, tok.chroma_max_id
    vlo, vhi = tok.vel_min_id, tok.vel_max_id

    def profile(arr, lo, hi):
        sel = arr[(arr >= lo) & (arr <= hi)] - lo
        if sel.size == 0:
            return None
        c = np.bincount(sel, minlength=hi - lo + 1).astype(float)
        p = c / c.sum()
        nz = p[p > 0]
        return float(p[0]), float(-(nz * np.log(nz)).sum())

    pos = profile(yf, plo, phi)
    chr = profile(yf, clo, chi)
    after_vel = (xf >= vlo) & (xf <= vhi)
    stack = (float(((yf[after_vel] >= clo) & (yf[after_vel] <= chi)).mean())
             if after_vel.any() else np.nan)
    # back_half_share: fraction of onsets (POS tokens) landing in the 2nd half
    # of the bar (positions >= ppb/2). Low value => front-loaded bars.
    ppb = phi - plo + 1
    pos_ids = yf[(yf >= plo) & (yf <= phi)] - plo
    back = float((pos_ids >= ppb // 2).mean()) if pos_ids.size else np.nan
    return {
        "pos0_share":    pos[0] if pos else np.nan,
        "pos_entropy":   pos[1] if pos else np.nan,
        "back_half_share": back,
        "chroma_entropy": chr[1] if chr else np.nan,
        "stack_rate":    stack,
        "n_onsets":      int(((yf >= plo) & (yf <= phi)).sum()),
    }


def _mean(rows, key):
    vals = [r[key] for r in rows if r is not None and not np.isnan(r[key])]
    return float(np.mean(vals)) if vals else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--pop909", default="data/raw/POP909",
                    help="dir of multi-track MIDIs (melody + accompaniment)")
    ap.add_argument("-n", "--num_songs", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=None,
                    help="override sampling temperature (default: cfg.inference)")
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--max_bars", type=int, default=0,
                    help="cap each song to its first N bars (0 = full song). "
                         "Speeds up autoregressive generation for larger samples.")
    ap.add_argument("--avoid_penalty", type=float, default=None,
                    help="avoid-note soft penalty (harmonic lever; default cfg).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tok = build_tokenizer(cfg.tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit = load_checkpoint(args.checkpoint, cfg, tok.vocab_size)
    lit.eval(); lit.to(device)

    temp = args.temperature if args.temperature is not None else cfg.inference.temperature
    top_p = args.top_p if args.top_p is not None else cfg.inference.top_p
    avoid = args.avoid_penalty if args.avoid_penalty is not None else getattr(cfg.inference, "avoid_note_penalty", 0.0)
    print(f"device={device}  temperature={temp}  top_p={top_p}  avoid_penalty={avoid}")
    print(f"checkpoint={args.checkpoint}\n")

    # POP909 layout: <root>/NNN/NNN.mid  (also has NNN_cl.mid — skip duplicates).
    songs = sorted(p for p in Path(args.pop909).glob("*/*.mid")
                   if p.stem == p.parent.name)[:args.num_songs]
    if not songs:
        # fall back to a flat directory of .mid files
        songs = sorted(Path(args.pop909).glob("*.mid"))[:args.num_songs]
    if not songs:
        print(f"No .mid files under {args.pop909}"); return

    gt_rows, gen_rows = [], []
    print(f"{'song':<24}{'GT pos0':>9}{'GEN pos0':>10}{'GT posEnt':>11}{'GEN posEnt':>12}")
    for s in songs:
        try:
            events, tempo = midi_to_events(s, cfg.tokenizer)
            k = estimate_key_from_midi(s)
            kr, km = (k if k else (0, 0))
            if args.max_bars > 0:
                events = [e for e in events if e.bar < args.max_bars]
            ids, mask = tok.encode_song(events, ["melody"], ["accompaniment"],
                                        tempo_bpm=tempo, key_root=kr, key_mode=km)
            gt = compute_stats(ids, mask, tok)

            # Generation input: melody only. When capping bars, write a truncated
            # melody MIDI so generation runs over the same window as GT.
            if args.max_bars > 0:
                mel = [e for e in events if e.track == "melody"]
                gen_in = Path(tempfile.mktemp(suffix=".mid"))
                events_to_midi(mel, cfg.tokenizer, tempo_bpm=tempo).dump(str(gen_in))
            else:
                gen_in = s
            gen_midi, gtempo = generate_accompaniment(
                gen_in, cfg, lit, tok, ["melody"], temperature=temp, top_p=top_p,
                avoid_note_penalty=args.avoid_penalty)
            if args.max_bars > 0:
                Path(gen_in).unlink(missing_ok=True)
            tmp = Path(tempfile.mktemp(suffix=".mid"))
            gen_midi.dump(str(tmp))
            gev, _ = midi_to_events(tmp, cfg.tokenizer)
            gids, gmask = tok.encode_song(gev, ["melody"], ["accompaniment"],
                                          tempo_bpm=gtempo, key_root=kr, key_mode=km)
            gen = compute_stats(gids, gmask, tok)
            tmp.unlink(missing_ok=True)

            gt_rows.append(gt); gen_rows.append(gen)
            print(f"{s.parent.name + '/' + s.name:<24}"
                  f"{gt['pos0_share']:>9.3f}{gen['pos0_share']:>10.3f}"
                  f"{gt['pos_entropy']:>11.2f}{gen['pos_entropy']:>12.2f}")
        except Exception as e:  # noqa: BLE001
            print(f"{s.name:<24}  SKIP ({type(e).__name__}: {e})")

    if not gen_rows:
        print("\nNo songs processed successfully."); return

    print("\n" + "=" * 60)
    print(f"AGGREGATE over {len(gen_rows)} songs   "
          f"(temp={temp}, top_p={top_p}, avoid={avoid})")
    print("=" * 60)
    hdr = f"{'metric':<18}{'GT':>10}{'GENERATED':>12}{'gap':>10}"
    print(hdr); print("-" * len(hdr))
    for key, label in (("pos0_share", "pos0_share"),
                       ("pos_entropy", "pos_entropy"),
                       ("back_half_share", "back_half_share"),
                       ("stack_rate", "stack_rate"),
                       ("chroma_entropy", "chroma_entropy")):
        g, e = _mean(gt_rows, key), _mean(gen_rows, key)
        print(f"{label:<18}{g:>10.3f}{e:>12.3f}{e - g:>10.3f}")
    print("\nInterpretation:")
    print("  pos0_share  ↑↑ in GEN vs GT  -> beat-1 collapse at GENERATION time")
    print("  pos_entropy ↓↓ in GEN vs GT  -> onsets concentrated on few positions")
    print("  (teacher-forced CSV baseline was pos0~0.11, pos_entropy~2.64)")


if __name__ == "__main__":
    main()
