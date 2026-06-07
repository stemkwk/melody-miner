"""Environment / hardware auto-tuning. Mirrors the structure of the spectrogram
project so the same YAML pattern works."""
from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import torch

from m2a_transformer.config import EnvScalingConfig


def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _env_label() -> str:
    if _is_colab():
        return "Google Colab"
    if os.name == "nt":
        return "Windows Local"
    return "Linux Server"


def _num_workers(env_cfg: EnvScalingConfig) -> int:
    if os.name == "nt":
        return env_cfg.num_workers_windows
    if _is_colab():
        return env_cfg.num_workers_colab
    cpus = os.cpu_count() or 1
    return max(1, min(cpus - 1, env_cfg.num_workers_server_max))


def _tier_scale(env_cfg: EnvScalingConfig, vram_gb: float) -> int:
    for t in env_cfg.tiers:
        if vram_gb >= t.vram_gte_gb:
            return t.batch_scale
    return env_cfg.fallback_batch_scale


# ---------------------------------------------------------------------------
# Flash-Attention-aware batch / grad_accum helpers
# ---------------------------------------------------------------------------

def _flash_attn_available() -> bool:
    """Return True if the current CUDA GPU supports Flash Attention (SM ≥ 8.0).

    Flash Attention was introduced in Ampere (SM 8.0).  Turing (T4, SM 7.5)
    and older cards fall back to the full O(seq²) attention matrix path.
    """
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_properties(0).major >= 8


def _max_batch_no_flash(
    n_heads: int,
    seq_len: int,
    vram_gib: float,
    budget_fraction: float = 0.15,
) -> int:
    """Upper-bound on batch size when the full O(seq²) attention matrix is used.

    Limits the score tensor  (B × H × T × T × 4 bytes)  to *budget_fraction*
    of total VRAM so there is room for weights, activations, and optimiser state.

    Default budget_fraction=0.15 is conservative:
        T4 14.56 GiB × 0.15 = 2.18 GiB  →  max_batch = 10  (effective: 8 after divisor fit)
    """
    budget_bytes = budget_fraction * vram_gib * (1024 ** 3)
    per_sample_bytes = n_heads * seq_len * seq_len * 4  # float32 attention scores
    return max(1, int(budget_bytes / per_sample_bytes))


def _fit_batch_to_budget(effective_batch: int, max_bs: int) -> Tuple[int, int]:
    """Return *(batch_size, grad_accum)* such that:

        batch_size × grad_accum == effective_batch   and   batch_size ≤ max_bs

    Chooses the *largest* valid batch_size (exact divisor of effective_batch).
    """
    for bs in range(min(effective_batch, max_bs), 0, -1):
        if effective_batch % bs == 0:
            return bs, effective_batch // bs
    return 1, effective_batch


def get_env_optimizations(
    base_batch_size: int,
    base_lr: float,
    env_cfg: EnvScalingConfig,
    seq_len: int = 0,
    n_heads: int = 0,
) -> Dict[str, Any]:
    """Return hardware-tuned training hyper-parameters.

    Parameters
    ----------
    base_batch_size, base_lr:
        Values from config before any scaling.
    env_cfg:
        The ``env_scaling`` config block.
    seq_len, n_heads:
        Sequence length and attention-head count of the model.  When provided,
        the function detects whether Flash Attention is available and
        automatically adjusts *batch_size* + *grad_accum* so that the O(seq²)
        attention matrix fits within 15 % of VRAM (Turing / T4 workaround).
        Pass 0 to disable this check (e.g. CPU-only or unit tests).

    Returns
    -------
    dict with keys:
        batch_size        – per-step batch (may be reduced from base)
        grad_accum        – gradient accumulation steps (≥ 1)
        learning_rate     – linearly scaled with batch_scale tier
        batch_scale       – VRAM-tier multiplier (1 for most 16 GB cards)
        flash_attn_available – bool
        num_workers, persistent_workers, prefetch_factor, precision, env_name
    """
    opts: Dict[str, Any] = {
        "batch_size": base_batch_size,
        "learning_rate": base_lr,
        "num_workers": _num_workers(env_cfg),
        "persistent_workers": False,
        "prefetch_factor": None,
        "precision": env_cfg.default_precision,
        "env_name": _env_label(),
        "grad_accum": 1,
        "flash_attn_available": True,
    }

    if torch.cuda.is_available():
        vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        scale = _tier_scale(env_cfg, vram_gib)
        opts["batch_size"] = base_batch_size * scale
        opts["learning_rate"] = base_lr * scale
        # Warmup steps cover the same amount of *data* regardless of batch size.
        # Larger batch → fewer steps per epoch → divide warmup proportionally so
        # the warmup phase doesn't consume a disproportionate fraction of training.
        opts["batch_scale"] = scale

        # ── Flash-Attention check ─────────────────────────────────────────
        has_flash = _flash_attn_available()
        opts["flash_attn_available"] = has_flash
        if not has_flash and seq_len > 0 and n_heads > 0:
            # Full O(seq²) attention matrix path: cap batch size so the score
            # tensor (B × H × T × T × fp32) stays within 15 % of total VRAM.
            max_bs = _max_batch_no_flash(n_heads, seq_len, vram_gib)
            effective_batch = opts["batch_size"]
            if max_bs < effective_batch:
                bs, accum = _fit_batch_to_budget(effective_batch, max_bs)
                opts["batch_size"] = bs
                opts["grad_accum"] = accum
                # LR is already correct: it was scaled for effective_batch and
                # we keep that LR because accumulation restores the effective batch.
    else:
        opts["precision"] = env_cfg.cpu_precision
        opts["batch_scale"] = 1
        opts["flash_attn_available"] = False

    if opts["num_workers"] > 0:
        opts["persistent_workers"] = True
        opts["prefetch_factor"] = env_cfg.prefetch_factor
    if "batch_scale" not in opts:
        opts["batch_scale"] = 1
    return opts
