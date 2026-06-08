"""Orchestration-level config for melody-miner.

This is intentionally small — the heavy model hyperparameters live in the
vendored M2A ``configs/config.yaml`` (loaded via ``m2a_transformer.config``).
Here we only hold paths and the few knobs the orchestrator itself owns
(mixing gains, output sample rate, generation overrides).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import DEFAULT_CONFIG


@dataclass
class GenParams:
    """Optional overrides forwarded to M2A ``generate_accompaniment``.

    ``None`` means "use the value from configs/config.yaml inference section".
    """
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    cfg_w: float = 0.0
    avoid_note_penalty: float | None = None
    denoise: bool = False
    cond_tracks: list[str] = field(default_factory=lambda: ["melody"])
    transcriber: str = "basic-pitch"   # WAV→MIDI: "basic-pitch" | "crepe"
    tempo_override: float | None = None


@dataclass
class VoiceParams:
    """Branch B (TNP voice conversion) inference knobs.

    Mel sharpening is a band-aid for predicted-mel over-smoothing — see
    ``VoiceConversionModel._sharpen_mel`` and scripts/analysis/diagnose_vocoder_wobble.py.
    """
    mel_sharpen: bool = True
    sharpen_t_alpha: float = 1.0   # temporal unsharp strength
    sharpen_f_alpha: float = 0.5   # spectral unsharp strength


@dataclass
class MixParams:
    target_sr: int = 44_100
    vocal_gain: float = 0.8
    accomp_gain: float = 0.6
    peak_ceiling: float = 0.99   # peak-normalise above this instead of clipping
    normalize_stems: bool = True  # match stem levels before applying gains
    postprocess_vocal: bool = True
    # tanh saturation drive (dB) in the vocal post chain — adds harmonics to mask
    # predicted-mel dullness. 0 = off. Only applies when postprocess_vocal is True.
    vocal_saturation_db: float = 0.0


@dataclass
class OrchestrationConfig:
    """Top-level run configuration."""
    m2a_checkpoint: Path
    m2a_config: Path = field(default_factory=lambda: Path(DEFAULT_CONFIG))
    tnp_checkpoint: Path | None = None
    references: list[Path] = field(default_factory=list)
    gen: GenParams = field(default_factory=GenParams)
    voice: VoiceParams = field(default_factory=VoiceParams)
    mix: MixParams = field(default_factory=MixParams)

    def __post_init__(self) -> None:
        self.m2a_checkpoint = Path(self.m2a_checkpoint)
        self.m2a_config = Path(self.m2a_config)
        if self.tnp_checkpoint is not None:
            self.tnp_checkpoint = Path(self.tnp_checkpoint)
        self.references = [Path(r) for r in self.references]
