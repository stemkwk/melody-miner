"""Infrastructure utilities (logging, MIDI I/O, audio, hardware tuning).

These are supporting tools, not the core academic contribution.
For the core algorithm, see:  model.py  tokenizer.py  dataset.py  pipeline.py

Submodules:
    utils.logger    — Loguru-based logger
    utils.midi_io   — MIDI ↔ NoteEvent conversion
    utils.audio     — audio recording / rendering / transcription
    utils.hardware  — environment / hardware auto-tuning
    utils.overrides — CLI config override parser
"""
