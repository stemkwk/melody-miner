"""Core inference pipeline helpers shared by CLI and web demo."""
from __future__ import annotations

from pathlib import Path

import torch

from m2a_transformer.utils.logger import logger
from m2a_transformer.utils.midi_io import events_to_midi, humanize_midi, midi_to_events
from m2a_transformer.tokenizer import BaseTokenizer


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class _InferenceModel:
    """Thin wrapper around the raw model for inference without pytorch-lightning."""

    def __init__(self, model):
        self.model = model

    def parameters(self):
        return self.model.parameters()

    def to(self, device):
        self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self


def load_checkpoint(ckpt_path: str, cfg, vocab_size: int) -> _InferenceModel:
    """Load a checkpoint for inference without requiring pytorch-lightning.

    Handles Lightning checkpoints (state_dict keys prefixed with ``model.``)
    and torch.compile'd checkpoints (``_orig_mod.`` prefix) transparently.
    """
    from m2a_transformer.model import build_model

    model = build_model(cfg.model, vocab_size)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_sd = ckpt.get("state_dict", ckpt)

    # Lightning wraps the model as self.model — strip that prefix.
    if any(k.startswith("model.") for k in raw_sd):
        raw_sd = {k[len("model."):]: v for k, v in raw_sd.items() if k.startswith("model.")}

    # torch.compile adds _orig_mod. between the module and its children.
    if any(k.startswith("_orig_mod.") for k in raw_sd):
        raw_sd = {k[len("_orig_mod."):]: v for k, v in raw_sd.items()}
        logger.info("Detected torch.compile checkpoint — stripped '_orig_mod.' prefix.")

    missing, unexpected = model.load_state_dict(raw_sd, strict=False)
    if missing:
        logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    if not missing and not unexpected:
        logger.info("State dict loaded cleanly.")
    ckpt_vocab = model.tok_emb.weight.shape[0]
    if ckpt_vocab != vocab_size:
        raise RuntimeError(
            f"Vocab size mismatch: checkpoint has {ckpt_vocab} tokens "
            f"but current tokenizer has {vocab_size}. "
            "Rebuild the checkpoint or use the matching tokenizer config."
        )
    return _InferenceModel(model)


def estimate_key_from_midi(midi_path: Path) -> tuple[int, int] | None:
    """Estimate global key via Krumhansl-Schmuckler pitch-class profile.

    Returns (root_0_11, mode_0_1) or None on failure.
    """
    try:
        import miditoolkit
        midi = miditoolkit.MidiFile(str(midi_path))
    except Exception:
        return None

    pc_count = [0.0] * 12
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            pc_count[n.pitch % 12] += (n.end - n.start)

    total = sum(pc_count)
    if total <= 0:
        return None
    pc = [x / total for x in pc_count]

    major = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    def _corr(profile: list[float], shift: int) -> float:
        rotated = profile[shift:] + profile[:shift]
        mean_p = sum(pc) / 12
        mean_r = sum(rotated) / 12
        num = sum((pc[i] - mean_p) * (rotated[i] - mean_r) for i in range(12))
        dp = sum((pc[i] - mean_p) ** 2 for i in range(12)) ** 0.5
        dr = sum((rotated[i] - mean_r) ** 2 for i in range(12)) ** 0.5
        return 0.0 if dp * dr < 1e-9 else num / (dp * dr)

    best_corr, best_root, best_mode = -999.0, 0, 0
    for root in range(12):
        for mode, profile in enumerate([major, minor]):
            c = _corr(profile, root)
            if c > best_corr:
                best_corr, best_root, best_mode = c, root, mode
    return (best_root, best_mode)


def _note_to_tokens(
    e: "object",
    tokenizer: BaseTokenizer,
    key_root: int,
) -> list[int]:
    """Encode a single NoteEvent into [CHROMA, OCTAVE, DUR, VEL] token ids."""
    # e is a NoteEvent with .pitch, .duration, .velocity
    dur = min(tokenizer.cfg.duration_max,
              max(tokenizer.cfg.duration_min, int(e.duration)))  # type: ignore[attr-defined]
    chroma_id, octave_id = tokenizer._pitch_to_chroma_octave(  # type: ignore[attr-defined]
        int(e.pitch), key_root  # type: ignore[attr-defined]
    )
    vel_bin = tokenizer.tid(  # type: ignore[attr-defined]
        f"VEL_{tokenizer._velocity_bin(int(e.velocity))}"  # type: ignore[attr-defined]
    )
    return [chroma_id, octave_id, tokenizer.tid(f"DUR_{dur}"), vel_bin]  # type: ignore[attr-defined]


def generate_accompaniment(
    melody_midi: Path,
    cfg,
    lit,
    tokenizer: BaseTokenizer,
    cond_tracks: list[str],
    tempo_override: float | None = None,
    track_name_override: dict[str, str] | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    max_new_tokens: int | None = None,
    cfg_w: float = 0.0,
    structural_suppression: float | None = None,
    avoid_note_penalty: float | None = None,
) -> tuple:
    """Full generation pipeline (bar-block interleaving): melody → (midi, tempo).

    Inference strategy
    ------------------
    The song is processed one block of ``cfg.lookahead_bars`` bars at a time.
    For each block we FORCE the block's melody tokens (BAR / POS / TRACK_mel /
    notes) followed by a forced SEP, then let the model GENERATE that block's
    accompaniment.  This matches the bar-block training distribution: the model
    always sees the full melody of the block before writing its accompaniment.

    Returns (miditoolkit.MidiFile, tempo_bpm).
    """
    from collections import defaultdict
    from m2a_transformer.model import DecoderTransformer

    icfg = cfg.inference
    temperature = temperature if temperature is not None else icfg.temperature
    top_k      = top_k      if top_k      is not None else icfg.top_k
    top_p      = top_p      if top_p      is not None else icfg.top_p
    _max_new_override = max_new_tokens   # finalised below once song length is known
    struct_supp = (structural_suppression if structural_suppression is not None
                   else getattr(icfg, "structural_suppression", 0.0))

    device = next(lit.parameters()).device
    N = max(1, int(getattr(cfg.tokenizer, "lookahead_bars", 1)))

    # ------------------------------------------------------------------ #
    # 1. Load melody and estimate key / tempo
    # ------------------------------------------------------------------ #
    melody_events, midi_tempo = midi_to_events(
        melody_midi, tokenizer.cfg, track_name_override
    )
    if not melody_events:
        raise ValueError(f"No notes found in {melody_midi}")

    tempo = tempo_override if tempo_override is not None else midi_tempo

    key_root: int | None = None
    key_mode: int | None = None
    kresult = estimate_key_from_midi(melody_midi)
    if kresult is not None:
        key_root, key_mode = kresult
        logger.info(
            f"Key estimated: {_NOTE_NAMES[key_root]} "
            f"{'major' if key_mode == 0 else 'minor'}"
        )
    kr = key_root if key_root is not None else 0

    # ------------------------------------------------------------------ #
    # 2. Group melody events by bar → position
    # ------------------------------------------------------------------ #
    melody_cond_track = cond_tracks[0] if cond_tracks else "melody"
    mel_by_bar: dict = defaultdict(lambda: defaultdict(list))
    for e in melody_events:
        if e.track == melody_cond_track:
            mel_by_bar[e.bar][e.position].append(e)

    active_bars = sorted(mel_by_bar.keys())
    if not active_bars:
        raise ValueError(f"No melody notes on track '{melody_cond_track}' in {melody_midi}")
    first_bar, last_bar = active_bars[0], active_bars[-1]

    # ------------------------------------------------------------------ #
    # 3. Build header token ids (BOS / KEY / TEMPO)
    # ------------------------------------------------------------------ #
    header_ids: list[int] = [tokenizer.bos_id]
    if tokenizer._key_min_id >= 0 and key_root is not None:  # type: ignore[attr-defined]
        header_ids.append(tokenizer.key_token_id(kr, key_mode or 0))
    header_ids.append(tokenizer.tid(f"TEMPO_{tokenizer.tempo_bin(tempo)}"))  # type: ignore[attr-defined]

    # CFG: run a parallel UNconditional branch whose melody/condition is PADded
    # (BOS+SEP kept), matching training condition-dropout. Each step blends:
    #   guided = uncond + cfg_w * (cond - uncond).   cfg_w == 0 disables it.
    use_cfg = cfg_w is not None and cfg_w > 0.0
    pad_id = tokenizer.pad_id
    header_ids_u: list[int] = [header_ids[0]] + [pad_id] * (len(header_ids) - 1)

    model_max = getattr(cfg.model, "max_seq_len", 4096)

    # Arbitrary-length default: when the caller does not cap generation, the
    # budget scales with the melody so even long songs are fully accompanied.
    # An explicit max_new_tokens still caps generation.
    if _max_new_override is not None:
        max_new = _max_new_override
    else:
        max_new = max(icfg.max_new_tokens, 64 * (last_bar - first_bar + 1))

    # Sliding-window context bound. The model was TRAINED on sequences up to
    # tokenizer.max_seq_len; to generate arbitrarily long songs without pushing
    # RoPE past its trained range, we keep the KV cache inside this window and
    # rebuild it from the BOS/KEY/TEMPO header anchor + the most-recent tokens
    # whenever it grows past the bound. Long-range memory is intentionally
    # dropped — bar-block harmony is local, so only recent context matters.
    # Reserve RoPE headroom: a slide fires only AFTER the cache exceeds
    # ctx_window, so the cache can momentarily overshoot by up to one forced
    # block before being rebuilt. Keep ctx_window below the RoPE buffer
    # (model_max) by that headroom so the overshoot never indexes past the
    # precomputed cos/sin tables.
    _rope_headroom = min(512, model_max // 4)
    ctx_window = min(int(getattr(cfg.tokenizer, "max_seq_len", model_max)),
                     max(1, model_max - _rope_headroom))
    _ctx_margin = max(64, ctx_window // 8)
    ctx_keep = max(1, ctx_window - len(header_ids) - _ctx_margin)

    logger.info(
        f"Bar-block generation | tempo={tempo:.1f} BPM | lookahead={N} bar(s) | "
        f"bars {first_bar}..{last_bar} | max_new={max_new} | ctx_window={ctx_window}"
    )

    # ------------------------------------------------------------------ #
    # 4. Structural-suppression state + token id shortcuts
    # ------------------------------------------------------------------ #
    use_struct_supp = (
        struct_supp > 0.0 and tokenizer.vel_min_id >= 0 and tokenizer.vel_max_id >= 0
    )
    if use_struct_supp:
        vel_lo, vel_hi = tokenizer.vel_min_id, tokenizer.vel_max_id
        struct_idx_t = torch.tensor(
            tokenizer.structural_ids(), dtype=torch.long, device=device
        )
    else:
        vel_lo = vel_hi = -1

    bar_id = tokenizer.bar_id
    sep_id = tokenizer.sep_id
    eos_id = tokenizer.eos_id

    # ------------------------------------------------------------------ #
    # 4b. Avoid-note soft penalty state
    # ------------------------------------------------------------------ #
    # Track the model-generated chord (SCALE_DEGREE + QUALITY) and subtract a
    # soft penalty from CHROMA logits that are "avoid notes" against it (e.g. the
    # natural 11 over a major 3rd). Soft, not a hard mask → colour/passing tones
    # survive; only sustained clashes are discouraged.
    avoid_pen = (avoid_note_penalty if avoid_note_penalty is not None
                 else getattr(icfg, "avoid_note_penalty", 0.0))
    use_avoid = (avoid_pen > 0.0 and tokenizer.sd_min_id >= 0
                 and tokenizer.quality_min_id >= 0 and tokenizer.chroma_min_id >= 0)
    cur_sd: int | None = None   # chord root pitch-class relative to key (0-11)
    cur_q:  int | None = None   # quality index (None = no active chord / CHORD_N)
    if use_avoid:
        from m2a_transformer.tokenizer import CHORD_AVOID_INTERVALS
        chroma_lo = tokenizer.chroma_min_id
        sd_lo, sd_hi = tokenizer.sd_min_id, tokenizer.sd_max_id
        q_lo, q_hi = tokenizer.quality_min_id, tokenizer.quality_max_id
        chord_n_id = tokenizer.chord_n_id
        quality_avoid = [
            CHORD_AVOID_INTERVALS.get(
                tokenizer.id_to_token[q_lo + i].replace("QUALITY_", ""), frozenset())
            for i in range(q_hi - q_lo + 1)
        ]

    # ------------------------------------------------------------------ #
    # 5. Step helper (single forward, returns last-position logits)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _step(tok_ids: list[int], kv_caches, uncond_ids: list[int] | None = None):
        if use_cfg and uncond_ids is not None:
            # Batch row 0 = conditional, row 1 = unconditional; blend last logits.
            t = torch.tensor([tok_ids, uncond_ids], dtype=torch.long, device=device)
            logits, new_caches = lit.model(t, kv_caches=kv_caches)
            lc = logits[0:1, -1, :]
            lu = logits[1:2, -1, :]
            return lu + cfg_w * (lc - lu), new_caches
        t = torch.tensor([tok_ids], dtype=torch.long, device=device)
        logits, new_caches = lit.model(t, kv_caches=kv_caches)
        return logits[:, -1, :], new_caches

    all_ids: list[int] = list(header_ids)
    all_ids_u: list[int] = list(header_ids_u)   # parallel uncond history (CFG only)
    last_logits, kv_caches = _step(header_ids, None,
                                   header_ids_u if use_cfg else None)
    generated_count = 0
    done = False

    @torch.no_grad()
    def _maybe_slide(last_logits, kv_caches):
        """Keep the KV cache within the trained context window.

        When the cache grows past ``ctx_window`` we rebuild it from the header
        anchor (BOS/KEY/TEMPO) plus the most-recent ``ctx_keep`` tokens via a
        single fresh forward. This re-rotates every kept key at in-range RoPE
        positions (naively truncating pre-rotated keys would corrupt relative
        positions) and yields the next-token logits for the latest token.
        Under CFG the unconditional branch is rebuilt in lockstep.
        """
        if kv_caches is None or kv_caches[0][0].shape[2] <= ctx_window:
            return last_logits, kv_caches
        window_ids = header_ids + all_ids[-ctx_keep:]
        if use_cfg:
            window_ids_u = header_ids_u + all_ids_u[-ctx_keep:]
            return _step(window_ids, None, window_ids_u)
        return _step(window_ids, None)

    # ------------------------------------------------------------------ #
    # 6. Bar-block generation loop
    # ------------------------------------------------------------------ #
    bar = first_bar
    while bar <= last_bar and not done:
        # bars belonging to the current N-aligned block (capped at last_bar)
        block_idx = bar // N
        bars_in_block: list[int] = []
        b = bar
        while b <= last_bar and b // N == block_idx:
            bars_in_block.append(b)
            b += 1

        # ---- Force melody section + SEP ---------------------------------
        forced: list[int] = []
        for bb in bars_in_block:
            forced.append(bar_id)
            for pos in sorted(mel_by_bar[bb].keys()):
                forced.append(tokenizer.tid(f"POS_{pos}"))
                forced.append(tokenizer.tid(f"TRACK_{melody_cond_track}"))
                for e in sorted(mel_by_bar[bb][pos], key=lambda e: e.pitch):
                    forced.extend(_note_to_tokens(e, tokenizer, kr))
        forced.append(sep_id)
        all_ids.extend(forced)
        if use_cfg:
            # uncond: PAD the whole forced melody block, keep only the SEP boundary
            forced_u = [pad_id] * (len(forced) - 1) + [sep_id]
            all_ids_u.extend(forced_u)
            last_logits, kv_caches = _step(forced, kv_caches, forced_u)
        else:
            last_logits, kv_caches = _step(forced, kv_caches)
        last_logits, kv_caches = _maybe_slide(last_logits, kv_caches)

        # ---- Generate accompaniment for this block ----------------------
        # The acc section spans len(bars_in_block) bars → (M-1) internal BARs
        # are allowed; the M-th BAR signals the next block (stop, discard it).
        M = len(bars_in_block)
        acc_bars_seen = 0
        block_budget = min(max_new - generated_count, 64 * M)

        for _ in range(block_budget):
            logits_use = last_logits
            _cloned = False
            if use_struct_supp and all_ids and (vel_lo <= all_ids[-1] <= vel_hi):
                logits_use = logits_use.clone(); _cloned = True
                logits_use[:, struct_idx_t] -= struct_supp
            if use_avoid and cur_sd is not None and cur_q is not None and quality_avoid[cur_q]:
                if not _cloned:
                    logits_use = logits_use.clone(); _cloned = True
                for _iv in quality_avoid[cur_q]:
                    logits_use[:, chroma_lo + ((cur_sd + _iv) % 12)] -= avoid_pen

            next_tok = DecoderTransformer._sample(logits_use, temperature, top_k, top_p)
            next_id = int(next_tok.item())
            if use_avoid:                       # track the active chord
                if sd_lo <= next_id <= sd_hi:
                    cur_sd = next_id - sd_lo
                elif q_lo <= next_id <= q_hi:
                    cur_q = next_id - q_lo
                elif next_id == chord_n_id:
                    cur_q = None                # explicit "no chord" → no penalty

            if next_id == eos_id:
                done = True
                break

            if next_id == bar_id:
                if acc_bars_seen < M - 1:
                    # internal bar of a multi-bar block → keep it, advance
                    acc_bars_seen += 1
                    all_ids.append(next_id)
                    if use_cfg: all_ids_u.append(next_id)
                    generated_count += 1
                    last_logits, kv_caches = _step(
                        [next_id], kv_caches, [next_id] if use_cfg else None)
                    last_logits, kv_caches = _maybe_slide(last_logits, kv_caches)
                    continue
                # block's accompaniment finished → discard this BAR, next block
                break

            # SEP should not appear inside generated accompaniment; treat as stop.
            if next_id == sep_id:
                break

            all_ids.append(next_id)
            if use_cfg: all_ids_u.append(next_id)
            generated_count += 1
            last_logits, kv_caches = _step(
                [next_id], kv_caches, [next_id] if use_cfg else None)
            last_logits, kv_caches = _maybe_slide(last_logits, kv_caches)
            if generated_count >= max_new:
                done = True
                break

        bar = bars_in_block[-1] + 1

    # ------------------------------------------------------------------ #
    # 7. Decode → NoteEvents, shift acc bars to absolute, render MIDI
    # ------------------------------------------------------------------ #
    decoded = tokenizer.decode(all_ids)
    target_track_name = cfg.tokenizer.tracks[-1]   # "accompaniment"
    # decode() numbers bars 0-indexed from the first BAR (= first_bar absolute).
    acc_events = []
    for e in decoded:
        if e.track == target_track_name:
            e.bar = e.bar + first_bar
            acc_events.append(e)

    midi = events_to_midi(
        [*melody_events, *acc_events], cfg.tokenizer,
        tempo_bpm=tempo,
        programs=cfg.midi_output.programs,
    )

    hcfg = cfg.humanize
    if hcfg.enabled:
        midi = humanize_midi(
            midi,
            velocity_std=hcfg.velocity_std,
            timing_std_ms=hcfg.timing_std_ms,
            duration_std_ms=hcfg.duration_std_ms,
        )

    logger.info(
        f"Generated {len(acc_events)} accompaniment notes "
        f"({generated_count} tokens) over {last_bar - first_bar + 1} bars."
    )
    return midi, tempo
