"""Audio I/O utilities for m2a_transformer.

Four entry points:
  record_from_mic()    — record WAV from the default microphone (Enter to stop)
  denoise_audio()      — spectral noise reduction via noisereduce
  audio_to_midi()      — transcribe WAV/MP3/FLAC/… to MIDI via basic-pitch
  render_midi_to_wav() — render a MIDI file to WAV via FluidSynth
"""
from __future__ import annotations

from pathlib import Path

from m2a_transformer.utils.logger import logger


def record_from_mic(out_wav: Path, sample_rate: int = 44100) -> None:
    """Record from the default microphone until the user presses Enter.

    Saves a 16-bit mono WAV to *out_wav*.
    Requires: pip install 'm2a_transformer[audio]'  (sounddevice, scipy)
    """
    try:
        import sounddevice as sd
        import numpy as np
        from scipy.io import wavfile
    except ImportError:
        raise ImportError(
            "sounddevice and scipy are required for microphone recording. "
            "Run: pip install 'm2a_transformer[audio]'"
        )

    chunks: list = []

    def _callback(indata, frames, time, status):  # noqa: ARG001
        chunks.append(indata.copy())

    print("\n[REC] 마이크 녹음 시작 — 멜로디를 연주하세요.")
    print("[REC] 끝나면 Enter를 누르세요.\n")
    with sd.InputStream(samplerate=sample_rate, channels=1,
                        dtype="float32", callback=_callback):
        input()

    audio = np.concatenate(chunks, axis=0)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(out_wav), sample_rate, (audio * 32767).astype("int16"))
    duration = len(audio) / sample_rate
    logger.info(f"Recorded {duration:.1f}s  →  {out_wav}")


def denoise_audio(audio_path: Path, out_wav: Path) -> None:
    """Reduce stationary background noise using spectral gating (noisereduce).

    Estimates the noise profile from the first 0.5 s of the recording and
    subtracts it across the whole file — effective for room hum, fan noise,
    and mic self-noise.
    Requires: pip install 'm2a_transformer[audio]'  (noisereduce, scipy)
    """
    try:
        import noisereduce as nr
        import numpy as np
        from scipy.io import wavfile
    except ImportError:
        raise ImportError(
            "noisereduce and scipy are required for denoising. "
            "Run: pip install 'm2a_transformer[audio]'"
        )

    sr, data = wavfile.read(str(audio_path))
    audio = data.astype(np.float32) / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    reduced = nr.reduce_noise(y=audio, sr=sr)

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(out_wav), sr, (reduced * 32767).astype("int16"))
    logger.info(f"Denoised audio → {out_wav}")


def audio_to_midi(
    audio_path: Path,
    out_midi: Path,
    denoise: bool = False,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_length_ms: float = 58.0,
    min_frequency: float | None = 32.7,
    max_frequency: float | None = 2093.0,
) -> None:
    """Transcribe an audio file (WAV/MP3/FLAC/…) to MIDI using basic-pitch.

    basic-pitch is a polyphonic neural AMT model; weights ship with the package
    so no separate download is needed.

    All transcription parameters map 1-to-1 to basic-pitch's predict() args.
    Requires: pip install 'm2a_transformer[audio]'
    """
    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH
    except ImportError:
        raise ImportError(
            "basic-pitch is not installed. "
            "Run: pip install 'm2a_transformer[audio]'  (or: pip install basic-pitch)"
        )

    src = audio_path
    if denoise:
        import tempfile
        tmp = Path(tempfile.mktemp(suffix=".wav"))
        denoise_audio(audio_path, tmp)
        src = tmp

    logger.info(f"Transcribing audio → MIDI: {src}")
    _, midi_data, _ = predict(
        str(src),
        ICASSP_2022_MODEL_PATH,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_note_length_ms,
        minimum_frequency=min_frequency,
        maximum_frequency=max_frequency,
    )
    out_midi.parent.mkdir(parents=True, exist_ok=True)
    midi_data.write(str(out_midi))
    logger.info(f"Transcribed MIDI saved: {out_midi}")

    if denoise and src != audio_path:
        src.unlink(missing_ok=True)


_SOUNDFONT_SEARCH_PATHS = [
    # Project-local (works on all platforms — drop any .sf2 here)
    "soundfonts/FluidR3_GM.sf2",
    "soundfonts/GeneralUser.sf2",
    "soundfonts/default.sf2",
    # Linux (apt install fluid-soundfont-gm / musescore-soundfont-gm)
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/FluidR3_GS.sf2",
    "/usr/share/sounds/sf2/TimGM6mb.sf2",
    # macOS (Homebrew fluid-synth)
    "/usr/local/share/sounds/sf2/FluidR3_GM.sf2",
    "/opt/homebrew/share/sounds/sf2/FluidR3_GM.sf2",
    # Windows — choco install fluidsynth
    "C:/tools/fluidsynth/share/soundfonts/default.sf2",
    # Common user locations
    "~/soundfonts/FluidR3_GM.sf2",
    "~/soundfonts/GeneralUser.sf2",
    "~/soundfonts/default.sf2",
]


def _find_soundfont(configured: str | None) -> str | None:
    """Return the first usable soundfont path, or None with a helpful message."""
    if configured and Path(configured).exists():
        return configured

    for candidate in _SOUNDFONT_SEARCH_PATHS:
        p = Path(candidate).expanduser()
        if p.exists():
            logger.info(f"Auto-detected soundfont: {p}")
            return str(p)

    logger.warning(
        "No soundfont found — WAV render skipped.\n"
        "  Install one of:\n"
        "    Linux:  sudo apt install fluid-soundfont-gm\n"
        "    macOS:  brew install fluid-synth  (includes FluidR3_GM)\n"
        "    Manual: download GeneralUser GS (.sf2) and set inference.soundfont in config.yaml"
    )
    return None


def apply_dsp(wav_path: Path, out_path: Path, dsp_cfg) -> None:
    """Apply Reverb → Compressor → Limiter chain via pedalboard.

    Writes processed audio to *out_path* (can be the same as *wav_path*).
    Requires: pip install 'm2a_transformer[render]'  (pedalboard, soundfile)
    """
    try:
        import soundfile as sf
        from pedalboard import Compressor, Limiter, Pedalboard, Reverb
    except ImportError:
        raise ImportError(
            "pedalboard and soundfile are required for DSP effects. "
            "Run: pip install 'm2a_transformer[render]'"
        )

    if not dsp_cfg.enabled:
        if wav_path != out_path:
            import shutil
            shutil.copy2(wav_path, out_path)
        return

    audio, sr = sf.read(str(wav_path), always_2d=True)  # (samples, channels)
    audio = audio.T.astype("float32")                   # (channels, samples)

    effects = []
    if dsp_cfg.reverb:
        effects.append(Reverb(
            room_size=dsp_cfg.reverb_room_size,
            damping=dsp_cfg.reverb_damping,
            wet_level=dsp_cfg.reverb_wet_level,
            dry_level=dsp_cfg.reverb_dry_level,
        ))
    if dsp_cfg.compressor:
        effects.append(Compressor(
            threshold_db=dsp_cfg.compressor_threshold_db,
            ratio=dsp_cfg.compressor_ratio,
            attack_ms=dsp_cfg.compressor_attack_ms,
            release_ms=dsp_cfg.compressor_release_ms,
        ))
    if dsp_cfg.limiter:
        effects.append(Limiter(threshold_db=dsp_cfg.limiter_threshold_db))

    board = Pedalboard(effects)
    processed = board(audio, sr).T  # back to (samples, channels)

    sf.write(str(out_path), processed, sr)
    logger.info(f"DSP applied ({len(effects)} effects) → {out_path}")


def render_midi_to_wav(midi_path: Path, wav_path: Path,
                       soundfont: str, sample_rate: int) -> None:
    """Render a MIDI file to WAV using FluidSynth.

    Strategy (in order):
      1. pyfluidsynth midi_to_audio()     — newest API
      2. pyfluidsynth file audio driver   — reliable on all platforms
      3. fluidsynth CLI via subprocess    — last resort
    Requires: pip install 'm2a_transformer[render]'  (pyfluidsynth)
    *soundfont* is tried first; if missing, common system paths are searched.
    """
    sf_path = _find_soundfont(soundfont)
    if sf_path is None:
        return

    try:
        import fluidsynth
    except ImportError:
        logger.warning("pyfluidsynth not installed; skipping WAV render.")
        return

    # ── 방법 1: midi_to_audio() (pyfluidsynth 최신) ─────────────────────────
    try:
        fs = fluidsynth.Synth(samplerate=float(sample_rate))
        fs.sfload(sf_path)
        if hasattr(fs, "midi_to_audio"):
            fs.midi_to_audio(str(midi_path), str(wav_path))
            fs.delete()
            logger.info(f"Rendered WAV → {wav_path}")
            return
        fs.delete()
    except Exception as e:
        logger.debug(f"midi_to_audio 실패: {e}")

    # ── 방법 2: Player + get_samples() 오프라인 렌더 ─────────────────────────
    # 드라이버를 시작하지 않고(no realtime/file driver) sample 타이밍으로 직접
    # 오디오를 끌어와 WAV로 씁니다. 기존 file-driver 방식은 백그라운드 스레드가
    # 출력 파일을 무음으로 덮어쓰는 레이스가 있어 (간헐 무음) 폐기.
    try:
        import numpy as _np
        from scipy.io import wavfile as _wav

        fs = fluidsynth.Synth(samplerate=float(sample_rate))
        fs.setting("player.timing-source", "sample")  # 소비한 샘플 수로 진행 → 오프라인
        fs.setting("synth.lock-memory", 0)
        fs.sfload(sf_path)

        player = fluidsynth.Player(fs)
        player.add(str(midi_path))
        player.play()

        sr = int(sample_rate)
        chunks = []
        max_chunks = 20 * 60  # ~20분 안전 상한
        while (player.get_status() == fluidsynth.FLUID_PLAYER_PLAYING
               and len(chunks) < max_chunks):
            chunks.append(fs.get_samples(sr))         # 1초치 stereo int16 (길이 2*sr)
        chunks.append(fs.get_samples(sr * 2))         # note release/reverb tail
        player.stop()
        fs.delete()

        if chunks:
            buf = _np.concatenate(chunks).astype(_np.int16).reshape(-1, 2)
            if int(_np.max(_np.abs(buf.astype(_np.int32)))) > 1:   # 무음 가드
                _wav.write(str(wav_path), sr, buf)
                if wav_path.exists():
                    logger.info(f"Rendered WAV → {wav_path}")
                    return
            else:
                logger.debug("get_samples produced silence — falling back to CLI")
    except Exception as e:
        logger.debug(f"offline get_samples 실패: {e}")

    # ── 방법 3: fluidsynth CLI subprocess ────────────────────────────────────
    import shutil
    import subprocess

    fluidsynth_bin = shutil.which("fluidsynth")
    if fluidsynth_bin is None:
        logger.warning(
            "WAV 렌더링 실패 — pyfluidsynth API와 CLI 모두 사용 불가.\n"
            "  Windows: winget install FluidSynth.FluidSynth 후 재시도"
        )
        return

    cmd = [
        fluidsynth_bin, "-ni",
        "-F", str(wav_path),
        "-r", str(sample_rate),
        sf_path, str(midi_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not wav_path.exists():
        logger.warning(f"fluidsynth CLI 실패:\n{result.stderr}")
        return

    logger.info(f"Rendered WAV → {wav_path}")
