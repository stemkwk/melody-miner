"""Branch A — melody WAV → accompaniment, wrapping the vendored M2A Transformer.

Thin reuse layer over ``m2a_transformer``'s public helpers (the same ones
previous implementations):
    load_config / build_tokenizer / load_checkpoint
    audio_to_midi  (WAV → melody MIDI, basic-pitch)
    generate_accompaniment  (melody MIDI → accompaniment MIDI)
    render_midi_to_wav + apply_dsp  (MIDI → WAV)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from . import DEFAULT_CONFIG  # noqa: F401  (ensures src is on sys.path)
from .config import GenParams
from .timing import timed

# vendored M2A imports (src is on sys.path via orchestration/__init__.py)
from m2a_transformer.config import load_config
from m2a_transformer.tokenizer import build_tokenizer
from m2a_transformer.pipeline import generate_accompaniment, load_checkpoint
from m2a_transformer.utils.audio import audio_to_midi, render_midi_to_wav, apply_dsp
from m2a_transformer.utils.midi_io import midi_to_events, events_to_midi
from m2a_transformer.utils.logger import logger


def extract_cond_midi(src_midi: Path, cfg, cond_tracks, out_path: Path) -> Path:
    """Write a MIDI containing ONLY the conditioning (melody) track(s).

    Extract a single-instrument MIDI track. A basic-pitch
    transcription has a generic instrument whose track name is NOT "melody"
    (it maps to "accompaniment"), so generation would raise "No melody notes
    on track 'melody'". Fallback: if no event matches ``cond_tracks``, relabel
    every note as the melody track so generation always has a conditioning
    signal.
    """
    from dataclasses import replace
    events, tempo = midi_to_events(Path(src_midi), cfg.tokenizer)
    target = cond_tracks[0] if cond_tracks else "melody"
    cond = set(cond_tracks) if cond_tracks else {"melody"}
    mel = [e for e in events if e.track in cond]
    if not mel:
        logger.warning(
            f"'{target}' 트랙을 찾지 못했습니다 — 전사 노트 전체를 멜로디로 간주합니다."
        )
        mel = [replace(e, track=target) for e in events]
    midi = events_to_midi(mel, cfg.tokenizer, tempo_bpm=tempo)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    midi.dump(str(out_path))
    return out_path


@dataclass
class M2aModel:
    """Loaded M2A state — build once, reuse across runs."""
    cfg: object
    lit: object
    tokenizer: object


def load_m2a(m2a_checkpoint: Path, config_path: Path = DEFAULT_CONFIG) -> M2aModel:
    cfg = load_config(str(config_path))
    tokenizer = build_tokenizer(cfg.tokenizer)
    lit = load_checkpoint(str(m2a_checkpoint), cfg, tokenizer.vocab_size)
    lit.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit.to(device)
    logger.info(f"M2A checkpoint loaded on {device}: {m2a_checkpoint}")
    return M2aModel(cfg=cfg, lit=lit, tokenizer=tokenizer)


def transcribe(input_wav: Path, cfg, out_midi: Path, denoise: bool = False,
               transcriber: str = "basic-pitch") -> Path:
    """WAV → melody MIDI.

    transcriber="basic-pitch" (default, polyphonic AMT) or "crepe" (monophonic
    CREPE f0 + note segmentation — far less fragmentation on solo vocals).
    """
    acfg = cfg.audio_input
    out_midi.parent.mkdir(parents=True, exist_ok=True)

    if transcriber == "crepe":
        from .transcribe_crepe import transcribe_crepe
        # Use a vocal-friendly range (config min_frequency defaults to C1, too low).
        fmin = max(float(acfg.min_frequency or 65.0), 50.0)
        fmax = min(float(acfg.max_frequency or 1000.0), 1500.0)
        transcribe_crepe(Path(input_wav), out_midi, fmin=fmin, fmax=fmax)
        return out_midi

    audio_to_midi(
        Path(input_wav), out_midi,
        denoise=denoise or acfg.denoise,
        onset_threshold=acfg.onset_threshold,
        frame_threshold=acfg.frame_threshold,
        min_note_length_ms=acfg.min_note_length_ms,
        min_frequency=acfg.min_frequency,
        max_frequency=acfg.max_frequency,
    )
    return out_midi


def render(midi_path: Path, cfg, out_wav: Path) -> Path | None:
    """MIDI → WAV via FluidSynth + DSP chain. Returns None if no soundfont."""
    icfg = cfg.inference
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    render_midi_to_wav(Path(midi_path), out_wav, icfg.soundfont, icfg.sample_rate)
    if not out_wav.exists():
        return None
    try:
        apply_dsp(out_wav, out_wav, cfg.dsp)
    except ImportError:
        logger.warning("pedalboard not installed — skipping DSP effects.")
    return out_wav


_MIDI_EXTS = {".mid", ".midi"}


def _split_tracks(midi_obj, cond_tracks, cfg):
    """Split a combined M2A output MIDI into (melody_only, accomp_only).

    ``generate_accompaniment`` returns ONE MIDI containing both the conditioning
    melody and the generated accompaniment (``events_to_midi`` makes one
    Instrument per track, named after the track in UPPER case — "MELODY" /
    "ACCOMPANIMENT"). We deep-copy and keep only the matching instrument so the
    accompaniment can be rendered on its own (no melody bleeding in), and the
    melody is rendered with its configured program (e.g. violin) for a consistent
    timbre.
    """
    import copy
    mel_name = (cond_tracks[0] if cond_tracks else "melody").upper()
    acc_name = cfg.tokenizer.tracks[-1].upper()

    def keep(names: set[str]):
        m = copy.deepcopy(midi_obj)
        m.instruments = [i for i in m.instruments
                         if (getattr(i, "name", "") or "").upper() in names]
        return m

    return keep({mel_name}), keep({acc_name})


def run_accompaniment(
    input_path: Path,
    model: M2aModel,
    out_dir: Path,
    gen: GenParams | None = None,
) -> dict:
    """Branch A: (WAV→transcription | direct MIDI) → melody → accompaniment → WAV.

    Input may be **audio (WAV/…) or a MIDI file**. MIDI input skips basic-pitch
    (transcription) entirely — useful for MIDI→MIDI verification. The melody
    (transcription or input MIDI) is ALSO rendered to ``02_melody.wav`` so you
    can listen to exactly what was fed into generation.

    Writes ``02_melody.mid`` / ``02_melody.wav`` / ``03_accompaniment.mid`` /
    ``03_accompaniment.wav`` into ``out_dir``. Returns paths + tempo + is_midi.
    """
    gen = gen or GenParams()
    cfg, lit, tokenizer = model.cfg, model.lit, model.tokenizer
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(input_path)
    is_midi = input_path.suffix.lower() in _MIDI_EXTS
    melody_midi = out_dir / "02_melody.mid"

    if is_midi:
        with timed("A1 입력 MIDI 사용 (전사 생략)"):
            import shutil
            shutil.copy2(input_path, melody_midi)
    else:
        with timed(f"A1 전사 (WAV→MIDI, {gen.transcriber})"):
            transcribe(input_path, cfg, melody_midi, gen.denoise, gen.transcriber)

    # Relabel notes onto the conditioning track so generation always has a
    # melody to condition on (basic-pitch tracks aren't named "melody").
    cond_midi = extract_cond_midi(
        melody_midi, cfg, gen.cond_tracks, out_dir / "02_melody_cond.mid")

    with timed("A2 반주 생성 (M2A Transformer)"):
        accomp_midi_obj, tempo = generate_accompaniment(
            melody_midi=cond_midi,
            cfg=cfg, lit=lit, tokenizer=tokenizer,
            cond_tracks=gen.cond_tracks,
            tempo_override=gen.tempo_override,
            temperature=gen.temperature,
            top_p=gen.top_p,
            top_k=gen.top_k,
            cfg_w=gen.cfg_w,
            avoid_note_penalty=gen.avoid_note_penalty,
        )

    # Split the combined output → pure accompaniment vs. melody (so 03 has NO
    # melody bleeding in, and the melody is rendered with its own instrument).
    melody_only, accomp_only = _split_tracks(accomp_midi_obj, gen.cond_tracks, cfg)
    accomp_midi = out_dir / "03_accompaniment.mid"           # pure accompaniment
    accomp_only.dump(str(accomp_midi))
    accomp_midi_obj.dump(str(out_dir / "03_accompaniment_full.mid"))  # combined (ref)
    melody_render_src = out_dir / "02_melody_render.mid"
    melody_only.dump(str(melody_render_src))

    # Render melody (consistent timbre = configured program) and pure accompaniment.
    with timed("A1b 멜로디 렌더 (동일 음색)"):
        melody_wav = render(melody_render_src, cfg, out_dir / "02_melody.wav")
    with timed("A3 렌더 (순수 반주, 멜로디 제외)"):
        accomp_wav = render(accomp_midi, cfg, out_dir / "03_accompaniment.wav")
    if accomp_wav is None:
        logger.warning("No soundfont — accompaniment WAV skipped (MIDI only).")

    return {
        "melody_midi": melody_midi,
        "melody_wav": melody_wav,
        "accomp_midi": accomp_midi,
        "accomp_wav": accomp_wav,
        "tempo": tempo,
        "is_midi": is_midi,
    }
