"""Sample N random tokenized shards and turn them back into listenable MIDI.

This is the human-friendly QA pass on the preprocessed data:

    python scripts/inspect_data.py
    # → inspection/<timestamp>/  containing:
    #     report.md           summary table + pitch histograms
    #     <shard>.raw.mid     decoded from .pt with no augmentation
    #     <shard>.aug.mid     decoded after the train-time pitch-transpose
    #                         (skipped if augment.pitch_transpose_semitones=0)
    #     <shard>.uncond.mid  melody condition replaced by PAD (CFG preview)
    #                         (skipped if augment.condition_dropout_prob=0)
    #     <shard>.cond.mid    melody-only (the condition the model is prompted with)
    #     <shard>.tgt.mid     bridge+piano only (what the model is asked to generate)

Why both raw and aug? The model never actually sees the .pt tokens verbatim
during training — it sees them after `JamTokenDataset._augment()` shifts the
pitches. Listening to both makes that delta audible, which is the only way to
notice "the augmentation is silently transposing into nonsense keys" or
"augmentation is off when it should be on".

This script is re-runnable: every invocation creates a fresh
`inspection/<timestamp>/` directory; nothing is overwritten.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch

from jam_transformer.config import AppConfig, load_config
from jam_transformer.dataset import JamTokenDataset
from jam_transformer.utils.logger import logger
from jam_transformer.utils.midi_io import events_to_midi
from jam_transformer.utils.overrides import apply_overrides
from jam_transformer.tokenizer import BaseTokenizer, NoteEvent, build_tokenizer


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------
def _split_by_track(events: list[NoteEvent], tracks: Sequence[str]) -> dict[str, list[NoteEvent]]:
    return {tr: [e for e in events if e.track == tr] for tr in tracks}


def _pitch_histogram(events: list[NoteEvent]) -> dict[int, int]:
    return dict(sorted(Counter(e.pitch % 12 for e in events).items()))


def _decode_with_tempo(tokenizer: BaseTokenizer, ids: list[int]) -> tuple[list[NoteEvent], float | None]:
    """Decode tokens; also recover the tempo from the prefix TEMPO_* token if
    one is present (currently `encode_song` emits at most one)."""
    tempo: float | None = None
    if hasattr(tokenizer, "tempo_from_bin"):
        for tid in ids:
            tok = tokenizer.id_to_token[tid] if 0 <= tid < tokenizer.vocab_size else ""
            if tok.startswith("TEMPO_"):
                try:
                    tempo = tokenizer.tempo_from_bin(int(tok.split("_", 1)[1]))
                except Exception:
                    pass
                break
    return tokenizer.decode(ids), tempo


# ---------------------------------------------------------------------------
# Per-shard inspection
# ---------------------------------------------------------------------------
def _emit_midis(
    name: str,
    out_dir: Path,
    tokenizer: BaseTokenizer,
    cfg: AppConfig,
    ids: torch.Tensor,
    mask: torch.Tensor,
    suffix: str,
    tempo_override: float | None = None,
) -> dict:
    """Decode `ids` to MIDI and also emit per-role split files. Returns a
    dict of stats used by the markdown report.

    `tempo_override` is used for the uncond variant where the TEMPO token was
    PAD-ded out by condition dropout — we pass the original song tempo so the
    exported MIDI plays at the correct speed."""
    id_list = ids.tolist()
    events, tempo = _decode_with_tempo(tokenizer, id_list)
    tempo = tempo_override if (tempo is None and tempo_override is not None) else (tempo or 120.0)

    tracks = cfg.tokenizer.tracks
    per_track = _split_by_track(events, tracks)

    # Combined MIDI (everything).
    midi = events_to_midi(events, cfg.tokenizer, tempo_bpm=tempo)
    midi.dump(str(out_dir / f"{name}.{suffix}.mid"))

    # Condition-only (melody) and target-only (bridge+piano) for A/B listening.
    cond_tracks  = ["melody"] if "melody" in tracks else tracks[:1]
    tgt_tracks   = [t for t in tracks if t not in cond_tracks]
    cond_events  = [e for e in events if e.track in cond_tracks]
    tgt_events   = [e for e in events if e.track in tgt_tracks]
    if cond_events:
        events_to_midi(cond_events, cfg.tokenizer, tempo_bpm=tempo) \
            .dump(str(out_dir / f"{name}.{suffix}.cond.mid"))
    if tgt_events:
        events_to_midi(tgt_events, cfg.tokenizer, tempo_bpm=tempo) \
            .dump(str(out_dir / f"{name}.{suffix}.tgt.mid"))

    target_mask_pct = 100.0 * float(mask.float().mean()) if mask.numel() else 0.0
    return {
        "name": name,
        "suffix": suffix,
        "tempo": tempo,
        "n_tokens": int(ids.numel()),
        "target_mask_pct": target_mask_pct,
        "per_track_notes": {tr: len(per_track.get(tr, [])) for tr in tracks},
        "per_track_pitch_range": {
            tr: (min(e.pitch for e in per_track[tr]),
                 max(e.pitch for e in per_track[tr]))
            for tr in tracks if per_track[tr]
        },
        "per_track_pitch_class_hist": {
            tr: _pitch_histogram(per_track[tr])
            for tr in tracks if per_track[tr]
        },
    }


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
_PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _format_pc_hist(hist: dict[int, int]) -> str:
    parts = [f"{_PC_NAMES[pc]}={n}" for pc, n in sorted(hist.items())]
    return ", ".join(parts) if parts else "(empty)"


def _write_report(report_path: Path, cfg: AppConfig, meta: dict, rows: list[dict],
                  data_dir: Path, augmented: bool) -> None:
    lines: list[str] = []
    lines.append(f"# Data inspection — {dt.datetime.now().isoformat(timespec='seconds')}\n")
    lines.append(f"- data dir         : `{data_dir}`")
    lines.append(f"- vocab size       : {meta.get('vocab_size')}")
    lines.append(f"- fingerprint      : `{meta.get('tokenizer_fingerprint')}`")
    lines.append(f"- cond tracks      : {meta.get('cond_tracks')}")
    lines.append(f"- target tracks    : {meta.get('target_tracks')}")
    lines.append(f"- pitch transpose  : ±{cfg.augment.pitch_transpose_semitones} semitones "
                 f"({'enabled' if cfg.augment.pitch_transpose_semitones > 0 else 'disabled'})")
    lines.append(f"- inspected w/ aug : {augmented}\n")

    # Per-shard sections.
    for row in rows:
        lines.append(f"## `{row['name']}` ({row['suffix']})\n")
        lines.append(f"- tempo            : {row['tempo']:.1f} BPM")
        lines.append(f"- token count      : {row['n_tokens']}")
        lines.append(f"- target mask cov. : {row['target_mask_pct']:.1f}% "
                     f"({'cond=' + str(round(100 - row['target_mask_pct'], 1)) + '%'})")
        lines.append("")
        lines.append("| Track | Notes | Pitch range | Pitch-class hist |")
        lines.append("|---|---|---|---|")
        for tr in cfg.tokenizer.tracks:
            n = row["per_track_notes"].get(tr, 0)
            rng = row["per_track_pitch_range"].get(tr)
            rng_str = f"{rng[0]}–{rng[1]}" if rng else "—"
            hist = row["per_track_pitch_class_hist"].get(tr, {})
            lines.append(f"| {tr} | {n} | {rng_str} | {_format_pc_hist(hist)} |")
        lines.append("")
        lines.append("Files:")
        lines.append(f"- `{row['name']}.{row['suffix']}.mid` — everything together")
        if row["suffix"] not in ("uncond",):
            lines.append(f"- `{row['name']}.{row['suffix']}.cond.mid` — melody only (model input)")
            lines.append(f"- `{row['name']}.{row['suffix']}.tgt.mid` — accompaniment only (model target)")
        else:
            lines.append(f"  *(condition portion PAD-ed; visualises CFG unconditional branch)*")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Sample tokenized shards back into listenable MIDI.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--n", type=int, default=4,
                        help="Number of random shards to sample.")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for shard selection + augmentation. "
                             "Omit for a fresh random sample each run.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output dir. Defaults to inspection/<timestamp>/.")
    parser.add_argument("--no_augment", action="store_true",
                        help="Skip the augmented variant (only emit raw MIDIs).")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="SECTION.KEY=VALUE",
                        help="Override any config field (e.g. for testing larger "
                             "transpose ranges than the YAML's default).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.overrides:
        apply_overrides(cfg, args.overrides)
    tokenizer = build_tokenizer(cfg.tokenizer)

    data_dir = Path(args.data_dir)
    meta_path = data_dir / "_dataset_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"Not a prepared dataset: {data_dir} (missing _dataset_meta.json)")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    shards = sorted(p for p in data_dir.glob("*.pt") if not p.name.startswith("_"))
    if not shards:
        raise SystemExit(f"No .pt shards under {data_dir}")
    if args.seed is not None:
        random.seed(args.seed)
        # Also seed torch + Python's random for the dataset's _augment call.
        torch.manual_seed(args.seed)
    sample = random.sample(shards, k=min(args.n, len(shards)))

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("inspection") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"writing {len(sample)} samples to {out_dir}")

    # Decode raw .pt directly.
    rows: list[dict] = []
    for p in sample:
        data = torch.load(p, map_location="cpu", weights_only=False)
        ids: torch.Tensor = data["ids"]
        mask: torch.Tensor = data["mask"]
        name = p.stem
        rows.append(_emit_midis(name, out_dir, tokenizer, cfg, ids, mask, suffix="raw"))

    # Decode after applying the train-time augmentation. We construct a
    # one-off dataset only to reuse its `_augment` method on the **full**
    # token sequence (not the 1024-token training chunk) so the user can hear
    # the whole song in the transposed key.
    has_pitch_aug = cfg.augment.pitch_transpose_semitones > 0
    has_cond_dropout = cfg.augment.condition_dropout_prob > 0.0

    if not args.no_augment and (has_pitch_aug or has_cond_dropout):
        aug_ds = JamTokenDataset(data_dir, cfg, tokenizer, train=True)
        for p in sample:
            data = torch.load(p, map_location="cpu", weights_only=False)
            ids_full: torch.Tensor = data["ids"].long()
            mask_full: torch.Tensor = data["mask"].bool()

            if has_pitch_aug:
                # Pitch-only augmentation: temporarily zero out dropout so we
                # hear the transposition alone on the "aug" file.
                orig_drop = cfg.augment.condition_dropout_prob
                cfg.augment.condition_dropout_prob = 0.0
                ids_aug = aug_ds._augment(ids_full.clone())
                cfg.augment.condition_dropout_prob = orig_drop
                rows.append(_emit_midis(p.stem, out_dir, tokenizer, cfg,
                                        ids_aug, mask_full, suffix="aug"))

            if has_cond_dropout:
                # Unconditional preview: force condition dropout (PAD melody).
                uncond_ids = torch.tensor(
                    tokenizer.make_uncond_prompt(ids_full), dtype=torch.long
                )
                # Recover the original tempo before it was PAD-ded out.
                _, orig_tempo = _decode_with_tempo(tokenizer, ids_full.tolist())
                rows.append(_emit_midis(p.stem, out_dir, tokenizer, cfg,
                                        uncond_ids, mask_full, suffix="uncond",
                                        tempo_override=orig_tempo))

    report_path = out_dir / "report.md"
    _write_report(report_path, cfg, meta, rows, data_dir,
                  augmented=not args.no_augment and (has_pitch_aug or has_cond_dropout))
    logger.info(f"report: {report_path}")
    logger.info(f"Inspection bundle ready: {out_dir}")
    logger.info(f"Open any *.mid in your DAW or MuseScore to listen.")
    logger.info(f"Read {report_path.name} for the summary.")


if __name__ == "__main__":
    main()
