"""Dataset over pre-tokenized .pt shards (relative harmonic encoding).

Each .pt shard is a dict::

    "ids":      LongTensor of token ids
    "mask":     BoolTensor; True where the position is accompaniment target
    "name":     str (source song id)
    "key_root": int (0-11, or -1 if unknown)
    "key_mode": int (0=major, 1=minor, or -1 if unknown)

key_root / key_mode are stored in the shard so every windowed chunk can
correctly augment OCTAVE tokens (which depend on the absolute key to
reconstruct).  See tokenizer module for the key-invariance proof.

Augmentation changes vs the absolute-encoding version
------------------------------------------------------
Pitch transposition now only needs to:
  1. Shift the KEY token root by `offset` (mod 12).
  2. Update OCTAVE tokens for notes whose abs_pitch crosses an octave
     boundary (new_octave = (old_abs + offset) // 12).
  CHROMA and SCALE_DEGREE tokens are mathematically invariant to
  transposition and must NOT be modified.

The range `pitch_transpose_semitones` can be reduced to ~5 (vs 12 before)
since all harmonic content is already key-relative in the .pt shards.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from m2a_transformer.config import AppConfig
from m2a_transformer.tokenizer import BaseTokenizer
from m2a_transformer.utils.logger import logger


META_FILENAME = "_dataset_meta.json"


def _fingerprint(cfg) -> str:
    return hashlib.sha256(
        json.dumps(asdict(cfg), sort_keys=True).encode()
    ).hexdigest()[:16]


def _total_ram_gb() -> Optional[float]:
    """Best-effort total system RAM in GB, no external deps (no psutil)."""
    try:
        import os
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:  # Linux/macOS
            return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
    except Exception:
        pass
    try:  # Windows
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        ms = _MS(); ms.dwLength = ctypes.sizeof(_MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return ms.ullTotalPhys / 1e9
    except Exception:
        return None


def _shard_cache_budget_gb(config) -> float:
    """Total shard-cache budget (GB) from the RAM tier matching detected RAM.
    Highest ram_gte_gb ≤ RAM wins; falls back to a conservative 2 GB."""
    tiers = sorted(getattr(config.env_scaling, "ram_tiers", []) or [],
                   key=lambda t: t.ram_gte_gb, reverse=True)
    if not tiers:
        return 2.0
    ram = _total_ram_gb()
    ram = 0.0 if ram is None else ram          # unknown → smallest tier
    for t in tiers:
        if ram >= t.ram_gte_gb:
            return float(t.cache_gb)
    return float(tiers[-1].cache_gb)


def load_dataset_meta(data_dir: str | Path) -> Optional[dict]:
    p = Path(data_dir) / META_FILENAME
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def assert_data_matches_config(data_dir: str | Path, tokenizer: BaseTokenizer) -> None:
    meta = load_dataset_meta(data_dir)
    if meta is None:
        logger.warning(
            f"No {META_FILENAME} found under {data_dir}. "
            "Re-run prepare_data.py with the current config."
        )
        return
    have = _fingerprint(tokenizer.cfg)
    want = meta.get("tokenizer_fingerprint", "")
    if have != want:
        raise SystemExit(
            f"Tokenizer fingerprint mismatch.\n"
            f"  data dir: {data_dir} → {want}\n"
            f"  current:  configs/... → {have}\n"
            f"Re-run prepare_data.py with the current config."
        )
    if meta.get("vocab_size") != tokenizer.vocab_size:
        raise SystemExit(
            f"Vocab size mismatch (data={meta.get('vocab_size')} "
            f"vs current={tokenizer.vocab_size}). Re-prepare the data."
        )


class M2aTokenDataset(Dataset):
    """Loads tokenized songs from `data_dir/*.pt` and returns fixed-length crops."""

    def __init__(
        self,
        data_dir: str | Path,
        config: AppConfig,
        tokenizer: BaseTokenizer,
        train: bool = True,
    ):
        self.config    = config
        self.tokenizer = tokenizer
        self.train     = train
        self.max_len   = config.tokenizer.max_seq_len
        self.data_dir  = Path(data_dir)

        all_shards: List[Path] = sorted(
            p for p in self.data_dir.glob("*.pt") if not p.name.startswith("_")
        )

        # ---- Song-level train/val split (hash-based, deterministic) ----------
        # Only applied when there are enough shards (guard: keeps unit-test
        # tmp_path dirs — typically 1-3 shards — in the legacy stride-only mode
        # so existing tests that assert train[0]==val[0] continue to pass).
        val_ratio = float(getattr(config.training, "val_ratio", 0.0))
        _MIN_SHARDS_FOR_SPLIT = 10

        if val_ratio > 0.0 and len(all_shards) >= _MIN_SHARDS_FOR_SPLIT:
            def _is_val(name: str) -> bool:
                h = int(hashlib.sha256(f"42:{name}".encode()).hexdigest(), 16)
                return (h % 10000) < int(val_ratio * 10000)

            if train:
                self.shards = [p for p in all_shards if not _is_val(p.name)]
            else:
                self.shards = [p for p in all_shards if _is_val(p.name)]
            logger.debug(
                f"Song-level split (val_ratio={val_ratio:.0%}): "
                f"{'train' if train else 'val'} = {len(self.shards)}/{len(all_shards)} shards"
            )
        else:
            self.shards = all_shards

        if not self.shards:
            logger.warning(f"No .pt shards found under {self.data_dir}")

        self._chunks: List[Tuple[int, int]] = []
        stride = max(1, self.max_len // 2) if train else self.max_len

        cache_path = self.data_dir / "_chunk_index.json"
        shard_lens: Optional[dict] = None
        if cache_path.exists():
            try:
                shard_lens = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Couldn't read {cache_path}: {e}. Falling back.")

        for si, shard in enumerate(self.shards):
            if shard_lens is not None and shard.name in shard_lens:
                entry = shard_lens[shard.name]
                if isinstance(entry, dict):
                    n        = int(entry["n"])
                    cond_len = int(entry.get("sep", 0))
                else:
                    # legacy format: plain int (no sep info)
                    n        = int(entry)
                    cond_len = 0
            else:
                data     = torch.load(shard, map_location="cpu", weights_only=False)
                n        = int(data["ids"].numel())
                nz       = data["mask"].nonzero()
                cond_len = int(nz[0].item()) if nz.numel() > 0 else n

            if n < 8:
                continue
            for o in range(0, max(1, n - 1), stride):
                # Skip windows that end entirely within the condition (melody-only)
                # — those produce target_mask=all-False, contributing zero loss.
                if cond_len > 0 and o + self.max_len <= cond_len:
                    continue
                self._chunks.append((si, o))

        self._poly_scores: Optional[List[float]] = None
        if train and getattr(config.training, "polyphony_sample_weight_alpha", 0.0) > 0.0:
            self._poly_scores = self._compute_polyphony_scores()

    def __len__(self) -> int:
        return len(self._chunks)

    # ------------------------------------------------------------------
    # Polyphony scoring (CHROMA tokens replace PITCH tokens)
    # ------------------------------------------------------------------
    def _compute_polyphony_scores(self) -> List[float]:
        pmin    = self.tokenizer.chroma_min_id
        pmax    = self.tokenizer.chroma_max_id
        bar_id  = self.tokenizer.bar_id
        pos_lo, pos_hi = self.tokenizer.pos_id_range
        tr_lo,  tr_hi  = self.tokenizer.track_id_range

        def _score_segment(ids_seg: torch.Tensor) -> float:
            n_seg, n_poly, n_in_seg = 0, 0, 0
            for tid in ids_seg.tolist():
                if tid == bar_id or (pos_lo <= tid <= pos_hi) or (tr_lo <= tid <= tr_hi):
                    if n_in_seg > 0:
                        n_seg += 1
                        if n_in_seg >= 2:
                            n_poly += 1
                    n_in_seg = 0
                elif pmin <= tid <= pmax:
                    n_in_seg += 1
            if n_in_seg > 0:
                n_seg += 1
                if n_in_seg >= 2:
                    n_poly += 1
            return n_poly / max(n_seg, 1)

        cache: dict = {}
        scores: List[float] = []
        for si, start in self._chunks:
            if si not in cache:
                data = torch.load(self.shards[si], map_location="cpu", weights_only=False)
                cache[si] = data["ids"]
            ids_seg = cache[si][start: start + self.max_len].long()
            scores.append(_score_segment(ids_seg))
        return scores

    def get_sample_weights(self) -> Optional["torch.Tensor"]:
        """Return per-chunk sampling weights for WeightedRandomSampler, or None
        if all weights are uniform (→ use plain shuffle instead).

        Combines two independent axes (both optional):
          1. **Polyphony** (polyphony_sample_weight_alpha > 0): up-weight
             chunks with denser chords.
          2. **Source balance** (source_weight_* != 1.0): correct the
             lakh/pop909/slakh imbalance without discarding data.
        """
        import torch as _t
        weights: Optional[_t.Tensor] = None

        # ---- 1. Polyphony weights ------------------------------------------
        alpha = float(getattr(self.config.training, "polyphony_sample_weight_alpha", 0.0))
        if alpha > 0.0 and self._poly_scores is not None:
            scores  = _t.tensor(self._poly_scores, dtype=_t.float32)
            weights = (scores + 0.01).pow(alpha)

        # ---- 2. Source weights ---------------------------------------------
        sw_pop909 = float(getattr(self.config.training, "source_weight_pop909", 1.0))
        sw_slakh  = float(getattr(self.config.training, "source_weight_slakh",  1.0))
        sw_lakh   = float(getattr(self.config.training, "source_weight_lakh",   1.0))

        # Only build a source-weight tensor when:
        #   (a) the configured weights differ across sources, AND
        #   (b) the actual chunks span at least two distinct weights.
        # This ensures that datasets where ALL shards share one prefix (or
        # no prefix) return None rather than a trivially-uniform tensor,
        # which keeps `train.py` in the cheaper `shuffle=True` path.
        if not (sw_pop909 == sw_slakh == sw_lakh):
            src_w: List[float] = []
            for si, _ in self._chunks:
                stem = self.shards[si].stem
                if stem.startswith("pop909"):
                    src_w.append(sw_pop909)
                elif stem.startswith("slakh"):
                    src_w.append(sw_slakh)
                elif stem.startswith("lakh"):
                    src_w.append(sw_lakh)
                else:
                    src_w.append(1.0)
            if len(set(src_w)) > 1:          # at least two distinct weights
                sw_tensor = _t.tensor(src_w, dtype=_t.float32)
                weights = sw_tensor if weights is None else weights * sw_tensor

        return weights

    # ------------------------------------------------------------------
    # Augmentation — relative encoding version
    # ------------------------------------------------------------------
    def _augment(self, ids: torch.Tensor, key_root: int = 0,
                 target_mask: "torch.Tensor | None" = None) -> torch.Tensor:
        """Apply train-time augmentations.

        1. **Pitch transposition** (pitch_transpose_semitones):
           a. Shift KEY token root by `offset` (mod 12).
           b. Update OCTAVE tokens where abs_pitch crosses octave boundary.
           CHROMA and SCALE_DEGREE are invariant — not modified.

        2. **Velocity jitter** (velocity_jitter_bins): per-token.
        3. **Tempo jitter** (tempo_jitter_bins): single delta for all TEMPO_*.
        4. **Duration jitter** (duration_jitter_bins): per-token.
        5. **Condition dropout** (condition_dropout_prob): PAD melody section.
        """
        aug = getattr(self.config, "augment", None)
        if aug is None:
            return ids
        import random as _r

        # ---- 1. Pitch transposition -----------------------------------------
        half = int(getattr(aug, "pitch_transpose_semitones", 0) or 0)
        if half > 0:
            kmin = self.tokenizer.key_min_id
            kmax = self.tokenizer.key_max_id
            omin = self.tokenizer.octave_min_id
            omax = self.tokenizer.octave_max_id
            obase = self.tokenizer.octave_base
            cmin = self.tokenizer.chroma_min_id
            pitch_lo = self.config.tokenizer.pitch_min
            pitch_hi = self.config.tokenizer.pitch_max

            # Vectorized: detect all adjacent (CHROMA, OCTAVE) token pairs.
            # A valid note is encoded as ids[i-1] ∈ [cmin, omin) followed by
            # ids[i] ∈ [omin, omax].
            prev_ids = ids[:-1]
            curr_ids = ids[1:]
            valid_pairs = (
                (prev_ids >= cmin) & (prev_ids < omin) &
                (curr_ids >= omin) & (curr_ids <= omax)
            )

            if valid_pairs.any():
                chromas    = prev_ids[valid_pairs] - cmin
                octaves    = curr_ids[valid_pairs] - omin + obase
                abs_pitches = octaves * 12 + (chromas + key_root) % 12

                lo = -int(min(half, int(abs_pitches.min().item()) - pitch_lo))
                hi =  int(min(half, pitch_hi - int(abs_pitches.max().item())))
                if hi >= lo:
                    offset = _r.randint(lo, hi)
                    if offset != 0:
                        ids = ids.clone()
                        # 1a. Update KEY root (vectorized)
                        if kmin >= 0:
                            key_mask = (ids >= kmin) & (ids <= kmax)
                            if key_mask.any():
                                rel      = ids[key_mask] - kmin
                                mode     = rel // 12
                                new_root = (rel % 12 + offset) % 12
                                ids[key_mask] = kmin + mode * 12 + new_root
                        # 1b. Update OCTAVE tokens (CHROMA unchanged) — vectorized
                        new_abs  = (abs_pitches + offset).clamp(pitch_lo, pitch_hi)
                        new_o    = new_abs // 12
                        new_oids = (omin + (new_o - obase)).clamp(omin, omax)
                        oct_pos  = valid_pairs.nonzero(as_tuple=False).squeeze(1) + 1
                        ids[oct_pos] = new_oids

        # ---- 2. Velocity jitter --------------------------------------------
        vel_jitter = int(getattr(aug, "velocity_jitter_bins", 0) or 0)
        if vel_jitter > 0:
            vmin = self.tokenizer.vel_min_id
            vmax = self.tokenizer.vel_max_id
            is_vel = (ids >= vmin) & (ids <= vmax)
            if bool(is_vel.any()):
                n_vel = int(is_vel.sum().item())
                deltas = torch.randint(-vel_jitter, vel_jitter + 1, (n_vel,))
                new_ids = ids.clone()
                vel_pos = is_vel.nonzero(as_tuple=False).squeeze(1)
                new_ids[vel_pos] = (ids[vel_pos] + deltas).clamp(vmin, vmax)
                ids = new_ids

        # ---- 3. Tempo jitter -----------------------------------------------
        tempo_jitter = int(getattr(aug, "tempo_jitter_bins", 0) or 0)
        if tempo_jitter > 0:
            tmin = self.tokenizer.tempo_min_id
            tmax = self.tokenizer.tempo_max_id
            is_tempo = (ids >= tmin) & (ids <= tmax)
            if bool(is_tempo.any()):
                offset = _r.randint(-tempo_jitter, tempo_jitter)
                if offset != 0:
                    ids = ids.clone()
                    ids[is_tempo] = (ids[is_tempo] + offset).clamp(tmin, tmax)

        # ---- 4. Duration jitter --------------------------------------------
        dur_jitter = int(getattr(aug, "duration_jitter_bins", 0) or 0)
        if dur_jitter > 0:
            dmin = self.tokenizer.dur_min_id
            dmax = self.tokenizer.dur_max_id
            is_dur = (ids >= dmin) & (ids <= dmax)
            if bool(is_dur.any()):
                n_dur = int(is_dur.sum().item())
                deltas = torch.randint(-dur_jitter, dur_jitter + 1, (n_dur,))
                dur_pos = is_dur.nonzero(as_tuple=False).squeeze(1)
                ids = ids.clone()
                ids[dur_pos] = (ids[dur_pos] + deltas).clamp(dmin, dmax)

        # ---- 5. Condition dropout (CFG) ------------------------------------
        # When it fires, blank the ENTIRE melody condition so the model learns a
        # true unconditional mode — this keeps inference-time CFG (cond vs uncond)
        # in-distribution. NOTE: chunks contain multiple [melody]→SEP→[acc] blocks;
        # the previous version padded only the first block, leaving later blocks'
        # melody intact (never trained a fully-unconditional chunk).
        p_drop = float(getattr(aug, "condition_dropout_prob", 0.0) or 0.0)
        if p_drop > 0.0 and _r.random() < p_drop:
            pad_id = self.tokenizer.pad_id
            sep_id = self.tokenizer.sep_id
            if target_mask is not None:
                # PAD every condition token (mask==0) across all blocks; keep the
                # BOS + SEP boundaries and all accompaniment (mask==1) untouched.
                drop = (~target_mask.bool()) & (ids != sep_id) & (ids != self.tokenizer.bos_id)
                if bool(drop.any()):
                    ids = ids.clone()
                    ids[drop] = pad_id
            else:
                # Legacy fallback (mask unavailable, e.g. inspect_data preview):
                # pad only the first block's melody.
                sep_pos = (ids == sep_id).nonzero(as_tuple=False)
                if sep_pos.numel() > 0:
                    sep_idx = int(sep_pos[0].item())
                    if sep_idx > 1:
                        ids = ids.clone()
                        ids[1:sep_idx] = pad_id

        return ids

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _init_shard_cache(self) -> None:
        """LRU shard cache bounded by the RAM-tier budget, split per worker.

        Each DataLoader worker process holds its own cache; dividing the total
        budget by the live worker count keeps the SUM across workers bounded
        (full 18k-shard dataset ≈ 2.8 GB; unbounded caching × workers OOMs small
        boxes). num_workers=0 → main process → divisor 1.
        """
        from collections import OrderedDict
        self._shard_cache: "OrderedDict" = OrderedDict()
        self._cache_bytes = 0
        total_gb = _shard_cache_budget_gb(self.config)
        wi = torch.utils.data.get_worker_info()
        nw = wi.num_workers if (wi is not None and wi.num_workers) else 1
        self._cache_budget = max(64 * 1024 * 1024, int(total_gb * 1e9 / nw))

    @staticmethod
    def _entry_bytes(entry) -> int:
        ids_full, mask_full = entry[0], entry[1]
        return (ids_full.element_size() * ids_full.nelement()
                + mask_full.element_size() * mask_full.nelement())

    def _load(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        shard_idx, start = self._chunks[idx]
        if not hasattr(self, "_shard_cache"):
            self._init_shard_cache()
        cache = self._shard_cache
        entry = cache.get(shard_idx)
        if entry is not None:
            cache.move_to_end(shard_idx)               # mark most-recently-used
        else:
            data = torch.load(self.shards[shard_idx], map_location="cpu", weights_only=False)
            entry = (
                data["ids"].long(),
                data["mask"].bool(),
                int(data.get("key_root", -1)),
                int(data.get("key_mode", -1)),
            )
            cache[shard_idx] = entry
            self._cache_bytes += self._entry_bytes(entry)
            # evict least-recently-used shards until within budget (keep ≥1)
            while self._cache_bytes > self._cache_budget and len(cache) > 1:
                _, old = cache.popitem(last=False)
                self._cache_bytes -= self._entry_bytes(old)
        ids_full, mask_full, key_root, key_mode = entry
        end  = start + self.max_len
        ids  = ids_full[start:end].clone()
        mask = mask_full[start:end].clone()
        return ids, mask, key_root, key_mode

    def __getitem__(self, idx: int):
        ids, target_mask, key_root, key_mode = self._load(idx)
        kr = key_root if key_root >= 0 else 0  # fallback C when unknown

        if self.train:
            ids = self._augment(ids, key_root=kr, target_mask=target_mask)

        pad_id = self.tokenizer.pad_id
        cur = ids.shape[0]
        if cur < self.max_len:
            pad = self.max_len - cur
            ids = torch.cat([ids, torch.full((pad,), pad_id, dtype=ids.dtype)])
            target_mask = torch.cat([
                target_mask, torch.zeros(pad, dtype=target_mask.dtype)
            ])

        x = ids[:-1].contiguous()
        y = ids[1:].contiguous()
        loss_mask = target_mask[1:].contiguous() & (y != pad_id)
        return x, y, loss_mask
