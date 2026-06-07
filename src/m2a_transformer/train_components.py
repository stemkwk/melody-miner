"""Optimizer / scheduler registries.

Add a new optimizer or scheduler by decorating a factory function. The string
names appearing in `training.optimizer` / `training.scheduler` are matched
against the registry."""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable

import torch

from m2a_transformer.config import TrainingConfig


_OPTIMIZER_REGISTRY: Dict[str, Callable[..., torch.optim.Optimizer]] = {}
_SCHEDULER_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_optimizer(name: str):
    def deco(fn):
        key = name.lower()
        if key in _OPTIMIZER_REGISTRY:
            raise ValueError(f"Optimizer '{name}' already registered")
        _OPTIMIZER_REGISTRY[key] = fn
        return fn
    return deco


def register_scheduler(name: str):
    def deco(fn):
        key = name.lower()
        if key in _SCHEDULER_REGISTRY:
            raise ValueError(f"Scheduler '{name}' already registered")
        _SCHEDULER_REGISTRY[key] = fn
        return fn
    return deco


def build_optimizer(
    name: str,
    params: Iterable[torch.nn.Parameter],
    *,
    training_cfg: TrainingConfig,
) -> torch.optim.Optimizer:
    key = name.lower()
    if key not in _OPTIMIZER_REGISTRY:
        raise KeyError(f"Optimizer '{name}' not registered.")
    return _OPTIMIZER_REGISTRY[key](
        params,
        learning_rate=training_cfg.learning_rate,
        beta1=training_cfg.beta1,
        beta2=training_cfg.beta2,
        weight_decay=training_cfg.weight_decay,
    )


def build_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    *,
    training_cfg: TrainingConfig,
    total_steps: int,
) -> Any:
    key = name.lower()
    if key not in _SCHEDULER_REGISTRY:
        raise KeyError(f"Scheduler '{name}' not registered.")
    return _SCHEDULER_REGISTRY[key](
        optimizer,
        warmup_steps=training_cfg.warmup_steps,
        total_steps=total_steps,
        min_lr_factor=training_cfg.min_lr_factor,
    )


# ---------------------------------------------------------------------------
# Built-ins
# ---------------------------------------------------------------------------
@register_optimizer("adamw")
def _adamw(params, *, learning_rate, beta1, beta2, weight_decay, **_):
    return torch.optim.AdamW(
        params, lr=learning_rate, betas=(beta1, beta2),
        weight_decay=weight_decay,
    )


@register_optimizer("adamw_fused")
def _adamw_fused(params, *, learning_rate, beta1, beta2, weight_decay, **_):
    """Fused-CUDA AdamW — 5–10% lower optimizer overhead (single CUDA kernel).

    Uses PyTorch's native ``fused=True`` flag, which merges the per-parameter
    update loop into one CUDA kernel call.

    **AMP + grad_clip note**: PyTorch-Lightning's AMP plugin blocks gradient
    clipping on fused optimizers via ``_step_supports_amp_scaling`` check.
    This is bypassed in ``M2aTransformerLightning.configure_gradient_clipping``
    which clips directly using ``clip_grad_norm_`` (PL guarantees gradients
    are already unscaled before that hook fires, so the math is correct).

    On CPU ``fused`` is unavailable; silently falls back to the standard path."""
    fused = torch.cuda.is_available()
    opt = torch.optim.AdamW(
        params, lr=learning_rate, betas=(beta1, beta2),
        weight_decay=weight_decay, fused=fused,
    )
    return opt


@register_scheduler("cosine_warmup")
def _cosine_warmup(optimizer, *, warmup_steps, total_steps, min_lr_factor):
    """Linear warmup → cosine decay to `min_lr_factor * peak_lr`."""
    warmup_steps = max(1, warmup_steps)
    total_steps  = max(warmup_steps + 1, total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        progress = min(1.0, progress)
        return min_lr_factor + (1.0 - min_lr_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@register_scheduler("constant")
def _constant(optimizer, *, warmup_steps, total_steps, min_lr_factor):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
