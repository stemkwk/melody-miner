"""Lightning training wrapper.

Cross-entropy on token logits, with an optional mask that zeroes the loss on
condition (melody-prefix) positions so the model is graded only on the
accompaniment it is supposed to *generate*."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from m2a_transformer.config import AppConfig
from m2a_transformer.utils.logger import logger
from m2a_transformer.model import build_model
from m2a_transformer.tokenizer import REMITokenizer, build_tokenizer
from m2a_transformer.train_components import build_optimizer, build_scheduler


class M2aTransformerLightning(pl.LightningModule):
    def __init__(self, config: AppConfig, vocab_size: int, total_steps: int = 100_000):
        super().__init__()
        self.save_hyperparameters(config.to_dict())
        self.config = config
        self.total_steps = total_steps
        self.model = build_model(config.model, vocab_size)
        self.vocab_size = vocab_size

        # ------------------------------------------------------------------
        # Token-type loss weights + polyphony boost setup
        # ------------------------------------------------------------------
        # We build a (vocab_size,) weight vector once and register it as a
        # non-persistent buffer so it follows .to(device) but stays out of
        # the checkpoint. Polyphony detection needs the vel id range and the
        # pitch id range, which we also cache here.
        tok = build_tokenizer(config.tokenizer)
        tcfg = config.training
        w_struct  = float(getattr(tcfg, "loss_struct_weight",  1.0))
        w_content = float(getattr(tcfg, "loss_content_weight", 1.0))
        w_pos     = float(getattr(tcfg, "loss_pos_weight", w_struct))
        self.register_buffer(
            "token_weight",
            torch.tensor(tok.build_token_weight_vector(w_struct, w_content, w_pos),
                         dtype=torch.float32),
            persistent=False,
        )
        self._vel_min_id    = int(tok.vel_min_id)
        self._vel_max_id    = int(tok.vel_max_id)
        self._chroma_min_id = int(tok.chroma_min_id)
        self._chroma_max_id = int(tok.chroma_max_id)
        self.polyphony_loss_boost = float(getattr(tcfg, "polyphony_loss_boost", 1.0))
        self.polyphony_max_stack  = int(getattr(tcfg, "polyphony_max_stack", 0))
        struct_ids = tok.structural_ids()
        self._struct_min_id = int(min(struct_ids))
        self._struct_max_id = int(max(struct_ids))
        # Cached ranges for the W&B harmony/rhythm quality metrics (val only).
        self._pos_min_id, self._pos_max_id = tok.pos_id_range
        self._ppb = int(config.tokenizer.positions_per_bar)

        # Optional torch.compile. We do it here (not in setup()) so the
        # compiled module is what Lightning's checkpoint code serialises.
        # Wrapped models are still safe to load_from_checkpoint thanks to
        # PyTorch's `_orig_mod` attribute on compiled modules.
        if config.model.compile:
            try:
                self.model = torch.compile(
                    self.model, mode=config.model.compile_mode
                )
                logger.info(f"torch.compile enabled (mode={config.model.compile_mode}).")
            except Exception as e:    # noqa: BLE001
                logger.warning(f"torch.compile failed — running in eager mode. Reason: {e}")

    # ------------------------------------------------------------------
    # Scheduled sampling (exposure-bias fix)
    # ------------------------------------------------------------------
    # Teacher forcing always feeds the GROUND-TRUTH previous token, so the model
    # never learns to recover from its own mistakes — at inference it drifts
    # (beat-1 collapse / front-loaded bars) even though the teacher-forced
    # conditional is correct. Scheduled sampling (Bengio et al. 2015) feeds the
    # model its OWN prediction for some accompaniment input tokens, with an
    # annealed probability, so it practises the autoregressive condition during
    # training. p=0 reproduces plain teacher forcing exactly (no extra forward).
    def _scheduled_sampling_prob(self) -> float:
        if not self.training:
            return 0.0
        tcfg = self.config.training
        pmax = float(getattr(tcfg, "scheduled_sampling_max_prob", 0.0))
        if pmax <= 0.0:
            return 0.0
        start  = int(getattr(tcfg, "scheduled_sampling_start_epoch", 0))
        warmup = max(1, int(getattr(tcfg, "scheduled_sampling_warmup_epochs", 1)))
        e = int(self.current_epoch)
        if e < start:
            return 0.0
        return pmax * min(1.0, (e - start + 1) / warmup)

    @torch.no_grad()
    def _maybe_scheduled_sampling(self, x, loss_mask):
        """Return x with some accompaniment INPUT tokens replaced by the model's
        own argmax prediction of them (prob p, annealed). Melody/condition tokens
        are never touched. p=0 returns x unchanged with no extra forward pass."""
        p = self._scheduled_sampling_prob()
        if p <= 0.0:
            return x
        logits1, _ = self.model(x)                       # teacher-forced probe
        pred1 = logits1.argmax(dim=-1)                   # pred1[:,t] predicts x[:,t+1]
        shifted = torch.zeros_like(x)
        shifted[:, 1:] = pred1[:, :-1]                   # model's guess of x[:,t]
        # x[:,t] is an accompaniment token iff its own target slot (loss_mask at
        # t-1) is True — only those are eligible for replacement.
        acc_input = torch.zeros_like(loss_mask, dtype=torch.bool)
        acc_input[:, 1:] = loss_mask.bool()[:, :-1]
        replace = acc_input & (torch.rand(x.shape, device=x.device) < p)
        return torch.where(replace, shifted, x)

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------
    def _compute_loss(self, batch, return_logits: bool = False):
        """Cross-entropy with token-type weighting + polyphony loss boost.

        Weight composition (multiplicative, per target position):
          base mask        : 1 on target positions, 0 elsewhere
          token-type weight: structural (BAR/POS/TRACK/TEMPO) vs content
                             (CHROMA/OCTAVE/DUR/VEL) — see TrainingConfig.loss_*_weight
          polyphony boost  : applied when target == CHROMA AND previous input
                             token == VEL (i.e. a chord-stacking decision)

        ppl is reported on the *raw* CE (no weights) so it stays comparable
        across runs that change the loss weights.
        """
        x, y, loss_mask = batch
        model_input = self._maybe_scheduled_sampling(x, loss_mask)  # x unless SS active
        logits, _ = self.model(model_input)                 # (B, T, V)
        flat_logits = logits.reshape(-1, self.vocab_size)
        flat_y      = y.reshape(-1)
        flat_x      = model_input.reshape(-1)               # context the model actually saw
        # `reduction='none'` so we can apply the mask.
        per_token = F.cross_entropy(flat_logits, flat_y, reduction="none")

        if self.config.training.mask_condition_loss:
            base_mask = loss_mask.reshape(-1).float()
        else:
            base_mask = (flat_y != 0).float()                # PAD id = 0

        # 1) Per-token type weight (vocab-sized lookup keyed by target id).
        type_w = self.token_weight[flat_y]                   # (B*T,)

        # 2) Polyphony boost: target is CHROMA AND previous input was VEL
        #    → this position decides to stack another note at the same (bar, pos).
        #    Stack-depth cap: count how many consecutive VEL→CHROMA stack
        #    decisions have occurred since the last structural token (BAR/POS/
        #    TRACK/TEMPO, which marks a position advance). Once the count exceeds
        #    polyphony_max_stack the boost is zeroed — the model stops getting
        #    rewarded for dumping more notes at the same position. Vectorised as
        #    a segmented cumulative sum (reset at structural tokens).
        if self.polyphony_loss_boost != 1.0:
            is_chroma_target = (flat_y >= self._chroma_min_id) & (flat_y <= self._chroma_max_id)
            is_vel_prev      = (flat_x >= self._vel_min_id)    & (flat_x <= self._vel_max_id)
            poly_mask = (is_chroma_target & is_vel_prev).float()

            if self.polyphony_max_stack > 0:
                B, T = x.shape
                inc   = (is_vel_prev & is_chroma_target).view(B, T).long()
                reset = ((flat_x >= self._struct_min_id) &
                         (flat_x <= self._struct_max_id)).view(B, T)
                # running count of stacks, reset to 0 at each structural token.
                # cum is non-decreasing → most-recent reset has the largest cum
                # among resets so far, so cummax recovers the segment baseline.
                cum = inc.cumsum(dim=1)
                reset_cum = torch.where(reset, cum, torch.zeros_like(cum))
                baseline  = torch.cummax(reset_cum, dim=1).values
                stack_count = (cum - baseline).view(-1)
                within_cap  = (stack_count <= self.polyphony_max_stack).float()
                poly_mask = poly_mask * within_cap

            poly_w = 1.0 + (self.polyphony_loss_boost - 1.0) * poly_mask
            type_w = type_w * poly_w

        weight = base_mask * type_w                          # (B*T,)
        denom  = weight.sum().clamp(min=1.0)
        loss   = (per_token * weight).sum() / denom

        # ppl reported on RAW (mask-only) CE for run-comparability.
        with torch.no_grad():
            raw_denom = base_mask.sum().clamp(min=1.0)
            raw_loss  = (per_token * base_mask).sum() / raw_denom
            ppl = torch.exp(raw_loss).item()
        if return_logits:
            return loss, ppl, logits
        return loss, ppl

    # ------------------------------------------------------------------
    # Quality metrics for W&B (rhythm + harmony), validation-only.
    # Cheap teacher-forced proxies: compare the model's argmax predictions to
    # the ground-truth tokens on accompaniment-target positions. No expensive
    # autoregressive sampling. These surface the beat-1 / cluster collapse
    # EARLY (within a few epochs) instead of only after listening.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _quality_metrics(self, x, y, logits, loss_mask) -> dict:
        out = {
            "val_rhythm/pred_pos0_share":     0.0,  # model onset share on beat-1 pos (collapse ↑)
            "val_rhythm/gt_pos0_share":       0.0,  # data baseline (~0.19)
            "val_rhythm/pred_pos_entropy":    0.0,  # spread of onsets over 16 positions (collapse ↓)
            "val_rhythm/gt_pos_entropy":      0.0,
            "val_rhythm/pred_stack_rate":     0.0,  # P(stack another note | after VEL) (cluster ↑)
            "val_rhythm/gt_stack_rate":       0.0,
            "val_harmony/pred_chroma_entropy": 0.0,  # pitch-class diversity (degenerate harmony ↓)
            "val_harmony/gt_chroma_entropy":   0.0,
        }
        preds = logits.argmax(dim=-1)
        m  = loss_mask.reshape(-1).bool()
        xf = x.reshape(-1)[m]
        yf = y.reshape(-1)[m]
        pf = preds.reshape(-1)[m]

        plo, phi = self._pos_min_id, self._pos_max_id
        clo, chi = self._chroma_min_id, self._chroma_max_id
        vlo, vhi = self._vel_min_id, self._vel_max_id

        def _profile(ids, lo, hi, nbins):
            sel = ids[(ids >= lo) & (ids <= hi)] - lo
            if sel.numel() == 0:
                return None
            c = torch.bincount(sel, minlength=nbins).float()
            p = c / c.sum()
            nz = p[p > 0]
            return p[0].item(), float(-(nz * nz.log()).sum())

        for tag, ids in (("pred", pf), ("gt", yf)):
            prof = _profile(ids, plo, phi, self._ppb)
            if prof is not None:
                out[f"val_rhythm/{tag}_pos0_share"]  = prof[0]
                out[f"val_rhythm/{tag}_pos_entropy"] = prof[1]
            ce = _profile(ids, clo, chi, chi - clo + 1)
            if ce is not None:
                out[f"val_harmony/{tag}_chroma_entropy"] = ce[1]

        after_vel = (xf >= vlo) & (xf <= vhi)
        if bool(after_vel.any()):
            def _stack_rate(ids):
                return float(((ids >= clo) & (ids <= chi)).float().mean())
            out["val_rhythm/pred_stack_rate"] = _stack_rate(pf[after_vel])
            out["val_rhythm/gt_stack_rate"]   = _stack_rate(yf[after_vel])
        return out

    def training_step(self, batch, batch_idx):
        loss, ppl = self._compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_ppl",  ppl,  prog_bar=False, on_step=True, on_epoch=True)
        opt = self.optimizers()
        if isinstance(opt, list):
            opt = opt[0]
        if opt is not None:
            self.log("lr", opt.param_groups[0]["lr"], prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, ppl, logits = self._compute_loss(batch, return_logits=True)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_ppl",  ppl,  prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        # Harmony + rhythm quality metrics → W&B / CSV (collapse early-warning).
        x, y, loss_mask = batch
        metrics = self._quality_metrics(x, y, logits, loss_mask)
        # pos0_share is the headline collapse signal → show on the progress bar.
        self.log("val_rhythm/pred_pos0_share", metrics.pop("val_rhythm/pred_pos0_share"),
                 prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        for k, v in metrics.items():
            self.log(k, v, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    # ------------------------------------------------------------------
    # Gradient clipping
    # ------------------------------------------------------------------
    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ):
        """Bypass PyTorch-Lightning's AMP-plugin gradient-clipping block.

        PL's AMP precision plugin (bf16-mixed / 16-mixed) raises RuntimeError
        when ``gradient_clip_val > 0`` and the optimizer has
        ``_step_supports_amp_scaling = True`` (set by PyTorch's fused AdamW).
        PL incorrectly infers that the optimizer handles its own gradient
        unscaling — true for apex's FusedAdam, but NOT for PyTorch's native
        fused AdamW, which only fuses the weight-update kernel.

        PL docs guarantee that gradients are already unscaled by the precision
        plugin before this hook fires, so calling ``clip_grad_norm_`` here
        operates on correct fp32 gradients in both AMP and non-AMP setups.
        This override also works transparently for the standard ``adamw``
        optimizer (the base class call would work too, but this is explicit).
        """
        clip_val = float(gradient_clip_val or 0.0)
        if clip_val > 0.0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_val)

    # ------------------------------------------------------------------
    # Optim
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        tcfg = self.config.training
        opt = build_optimizer(tcfg.optimizer, self.parameters(), training_cfg=tcfg)
        sch = build_scheduler(
            tcfg.scheduler, opt, training_cfg=tcfg, total_steps=self.total_steps,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "interval": "step"},
        }
