"""Decoder-only Transformer with RoPE.

Designed to be small enough (~30 M params at the default config) to smoke-test
on a free Colab T4, but to scale up to ~100 M without code changes by adjusting
d_model / n_layers / d_ff in the YAML. We rely on torch.nn.functional.
scaled_dot_product_attention so FlashAttention is used automatically when the
hardware supports it.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _grad_ckpt

from m2a_transformer.config import ModelConfig


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    def deco(cls: Type[nn.Module]) -> Type[nn.Module]:
        key = name.lower()
        if key in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered")
        _MODEL_REGISTRY[key] = cls
        return cls
    return deco


def build_model(config: ModelConfig, vocab_size: int) -> nn.Module:
    key = config.name.lower()
    if key not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{config.name}'. Known: {sorted(_MODEL_REGISTRY)}"
        )
    return _MODEL_REGISTRY[key](config, vocab_size)


# ---------------------------------------------------------------------------
# Rotary position embeddings
# ---------------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    """Standard RoPE — caches cos / sin up to the longest seq seen so far."""

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Pre-compute cos/sin for ALL positions up to max_seq_len in fp32.
        # Storing as registered buffers that are NEVER reassigned means the
        # buffer address stays fixed across forward calls — required for
        # torch.compile(mode="reduce-overhead") which uses CUDAGraphs and
        # disallows in-place overwrites of captured tensors.
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq.float())            # (max_seq_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)             # (max_seq_len, head_dim)
        self.register_buffer("_cos", emb.cos(), persistent=False)   # fp32, fixed
        self.register_buffer("_sin", emb.sin(), persistent=False)   # fp32, fixed

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k: (B, H, T, head_dim)
        T = q.shape[-2]
        # Slice only — buffers never change shape or address → CUDAGraphs-safe.
        # Cast to q's dtype (bf16 in mixed-precision training) on the fly.
        # `offset` is the number of tokens already in the KV cache so that
        # each new token gets its correct absolute position embedding.
        cos = self._cos[offset:offset + T].to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)  # (1,1,T,D)
        sin = self._sin[offset:offset + T].to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, use_rope: bool,
                 max_seq_len: int = 2048):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = dropout
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len) if use_rope else None

    def forward(
        self, x: torch.Tensor, kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            offset = kv_cache[0].shape[2] if kv_cache is not None else 0
            q, k = self.rope(q, k, offset=offset)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)

        new_cache = (k, v)

        # Build attention mask:
        #   • Training / first fill (kv_cache is None, T == seq_len): use Flash's
        #     built-in causal kernel — fastest path.
        #   • Single-step decode (kv_cache set, T == 1): no mask needed; every
        #     cached key is visible to the single query.
        #   • Forced-prefix decode (kv_cache set, T > 1): must apply a
        #     block-causal mask so query i cannot attend to future query j > i.
        is_causal = False
        attn_mask = None
        if kv_cache is None:
            is_causal = True
        elif T > 1:
            # q[i] (absolute pos T_past+i) may attend to k[j] iff j <= T_past+i.
            T_past = k.shape[2] - T
            qi = torch.arange(T, device=x.device).unsqueeze(1)           # (T, 1)
            kj = torch.arange(T_past + T, device=x.device).unsqueeze(0)  # (1, T_total)
            attn_mask = (kj <= T_past + qi).unsqueeze(0).unsqueeze(0)    # (1,1,T,T_total) bool

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, -1)
        return self.proj(attn), new_cache


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.use_rope,
                                         max_seq_len=getattr(cfg, "max_seq_len", 2048))
        self.drop1 = nn.Dropout(cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x, kv_cache=None):
        h, new_cache = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + self.drop1(h)
        x = x + self.ff(self.ln2(x))
        return x, new_cache


# ---------------------------------------------------------------------------
# Top-level decoder
# ---------------------------------------------------------------------------
@register_model("decoder_transformer_v1")
class DecoderTransformer(nn.Module):
    """Token-only decoder. Input: (B, T) LongTensor → logits (B, T, vocab)."""

    def __init__(self, cfg: ModelConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        # Learned absolute positional embedding is only used if RoPE is off.
        self.pos_emb: Optional[nn.Embedding] = None
        if not cfg.use_rope:
            self.pos_emb = nn.Embedding(8192, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,
        kv_caches: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        B, T = idx.shape
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            offset = 0 if kv_caches is None else kv_caches[0][0].shape[2]
            pos = torch.arange(offset, offset + T, device=idx.device)
            x = x + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)

        new_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            cache_i = kv_caches[i] if kv_caches is not None else None
            if self.cfg.gradient_checkpointing and self.training and cache_i is None:
                x, new_cache = _grad_ckpt(block, x, None, use_reentrant=False)
            else:
                x, new_cache = block(x, kv_cache=cache_i)
            new_caches.append(new_cache)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, new_caches

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        eos_id: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        # ---- Classifier-Free Guidance --------------------------------
        uncond_prompt_ids: Optional[torch.Tensor] = None,
        cfg_w: float = 0.0,
        # ---- Structural suppression (polyphony hack) ------------------
        structural_suppression: float = 0.0,
        vel_id_range: Optional[Tuple[int, int]] = None,
        struct_ids: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Greedy / top-k / top-p sampling with a fresh KV cache.

        Classifier-Free Guidance (CFG):
            If `uncond_prompt_ids` is provided **and** `cfg_w > 0`, we run
            conditional and unconditional sequences in a single batched
            forward pass at every step and blend the logits:

                logits_cfg = logits_uncond + cfg_w * (logits_cond - logits_uncond)

            * cfg_w = 0  → pure conditional (uncond_prompt_ids ignored)
            * cfg_w = 1  → same as conditional (guidance = conditional)
            * cfg_w > 1  → amplified conditional guidance (typical: 1.5–3.0)

            `uncond_prompt_ids` should be identical in length to `prompt_ids`
            with the condition portion (positions 1..sep_idx-1) replaced by
            the pad token, matching what `M2aTokenDataset._augment` does
            during training.  The inference script can build it by calling
            `tokenizer.make_uncond_prompt(prompt_ids)`.

        Structural suppression (polyphony hack):
            When the last sampled token is a VEL_* (end of a PITCH/DUR/VEL
            triple), the model has to choose between starting a new note at
            the SAME position (another PITCH_*, → polyphonic) and advancing
            to the next position (POS_*, BAR, → monophonic).  We subtract
            `structural_suppression` from the BAR/POS logits at exactly
            those decision points so the sampler is biased toward stacking
            another pitch.  Pass `vel_id_range=(lo, hi)` and `struct_ids=[..]`
            from the tokenizer to enable.  `structural_suppression=0` disables.
        """
        self.eval()
        device = prompt_ids.device
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        use_cfg = (
            uncond_prompt_ids is not None
            and cfg_w > 0.0
        )

        # Pre-bake structural-suppression state so we don't rebuild tensors
        # every step. struct_logit_mask: True on BAR/POS ids → subtract penalty.
        use_struct_supp = (
            structural_suppression > 0.0
            and vel_id_range is not None
            and struct_ids is not None
        )
        if use_struct_supp:
            vel_lo, vel_hi = vel_id_range
            struct_idx = torch.tensor(list(struct_ids), dtype=torch.long, device=device)
        else:
            vel_lo = vel_hi = -1

        if use_cfg:
            if uncond_prompt_ids.dim() == 1:
                uncond_prompt_ids = uncond_prompt_ids.unsqueeze(0)
            # Batch both prompts: index 0 = conditional, index 1 = unconditional.
            both_ids = torch.cat([prompt_ids, uncond_prompt_ids], dim=0)  # (2, T)
            logits, caches = self.forward(both_ids, kv_caches=None)
            # Blend: logits_uncond + cfg_w * (logits_cond - logits_uncond) → (1, V)
            next_logits = (
                logits[1:2, -1, :] + cfg_w * (logits[0:1, -1, :] - logits[1:2, -1, :])
            )
            generated = prompt_ids                              # track cond branch only
        else:
            logits, caches = self.forward(prompt_ids, kv_caches=None)
            generated = prompt_ids
            next_logits = logits[:, -1, :]                      # (1, V)

        for _ in range(max_new_tokens):
            # Structural suppression: if the LAST generated token is a VEL_*,
            # subtract `structural_suppression` from the BAR/POS logits so the
            # sampler prefers to add another note at the same position.
            if use_struct_supp:
                last_tok_id = int(generated[0, -1].item())
                if vel_lo <= last_tok_id <= vel_hi:
                    next_logits = next_logits.clone()
                    next_logits[:, struct_idx] -= structural_suppression

            # next_logits is always (1, V) at this point (see assignments above).
            next_tok = self._sample(next_logits, temperature, top_k, top_p)  # (1, 1)
            generated = torch.cat([generated, next_tok], dim=1)
            if (next_tok == eos_id).all():
                break

            if use_cfg:
                # Feed the same new token to both branches.
                next_tok_both = next_tok.expand(2, 1)           # (2, 1)
                logits, caches = self.forward(next_tok_both, kv_caches=caches)
                next_logits = (
                    logits[1:2, -1, :] + cfg_w * (logits[0:1, -1, :] - logits[1:2, -1, :])
                )                                               # (1, V)
            else:
                logits, caches = self.forward(next_tok, kv_caches=caches)
                next_logits = logits[:, -1, :]                  # (1, V)

        return generated

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / max(temperature, 1e-5)

        if top_k and top_k > 0:
            k = min(top_k, logits.shape[-1])
            kth_vals = torch.topk(logits, k, dim=-1).values[..., -1, None]
            logits = torch.where(logits < kth_vals,
                                 torch.full_like(logits, float("-inf")),
                                 logits)

        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
            cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            # Shift cumsum right so the token that first crosses top_p is included.
            keep = torch.zeros_like(cum, dtype=torch.bool)
            keep[..., 0] = True                  # always keep most-probable token
            keep[..., 1:] = cum[..., :-1] < top_p
            mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, sorted_idx, keep)
            logits = torch.where(mask, logits, torch.full_like(logits, float("-inf")))

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
