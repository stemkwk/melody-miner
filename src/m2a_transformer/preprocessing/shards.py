"""Shard I/O and dataset index management."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch

from m2a_transformer.tokenizer import BaseTokenizer

META_FILENAME = "_dataset_meta.json"


def tokenizer_fingerprint(cfg) -> str:
    return hashlib.sha256(
        json.dumps(asdict(cfg), sort_keys=True).encode()
    ).hexdigest()[:16]


def write_meta(
    out_dir: Path,
    tokenizer: BaseTokenizer,
    cond_tracks: list[str],
    target_tracks: list[str],
    new_shard_names: Optional[list[str]] = None,
) -> None:
    index_path = out_dir / "_chunk_index.json"
    if new_shard_names is not None:
        shard_lens: dict[str, int] = {}
        if index_path.exists():
            try:
                shard_lens = json.loads(index_path.read_text(encoding="utf-8"))
            except Exception:
                shard_lens = {}
        for stem in new_shard_names:
            p = out_dir / f"{stem}.pt"
            if p.exists():
                data = torch.load(p, map_location="cpu", weights_only=False)
                mask = data["mask"]
                nz   = mask.nonzero()
                sep  = int(nz[0].item()) if nz.numel() > 0 else int(mask.numel())
                shard_lens[f"{stem}.pt"] = {"n": int(data["ids"].numel()), "sep": sep}
    else:
        shard_lens = {}
        for p in sorted(out_dir.glob("*.pt")):
            if p.name.startswith("_"):
                continue
            data = torch.load(p, map_location="cpu", weights_only=False)
            mask = data["mask"]
            nz   = mask.nonzero()
            sep  = int(nz[0].item()) if nz.numel() > 0 else int(mask.numel())
            shard_lens[p.name] = {"n": int(data["ids"].numel()), "sep": sep}

    meta = {
        "vocab_size":             tokenizer.vocab_size,
        "tokenizer_config":       asdict(tokenizer.cfg),
        "tokenizer_fingerprint":  tokenizer_fingerprint(tokenizer.cfg),
        "n_shards":               len(shard_lens),
        "cond_tracks":            cond_tracks,
        "target_tracks":          target_tracks,
    }
    (out_dir / META_FILENAME).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    index_path.write_text(
        json.dumps(shard_lens, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _save_shard(
    out_dir: Path,
    name: str,
    ids: list[int],
    mask: list[bool],
    key_root: int = -1,
    key_mode: int = -1,
    method: "str | None" = None,
) -> None:
    torch.save(
        {
            "ids":      torch.tensor(ids,  dtype=torch.long),
            "mask":     torch.tensor(mask, dtype=torch.bool),
            "name":     name,
            "key_root": key_root,
            "key_mode": key_mode,
            "method":   method,   # melody-selection provenance: miner/weight/instrument
        },
        out_dir / f"{name}.pt",
    )


def _safe_name(p: Path) -> str:
    return p.stem.replace(" ", "_").replace("/", "_")
