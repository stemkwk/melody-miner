"""melody-miner — WAV → (accompaniment generation + voice conversion) → one mix.

From a single input WAV, run two branches and merge them into one track:
  Branch A: WAV → MIDI transcription → M2A Transformer accompaniment → accompaniment WAV
  Branch B: WAV → TNP voice conversion (target speaker) → converted vocal WAV
  Merge:    converted vocal + generated accompaniment → mixed WAV

The two source projects are vendored under ``src/``, and this package
orchestrates them.
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Vendored import paths
# ---------------------------------------------------------------------------
# repo root = .../melody-miner  (src/orchestration/__init__.py → parents[2])
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
M2A_DIR = SRC_DIR / "m2a_transformer"            # package dir
TNP_DIR = SRC_DIR / "tnp_voice_conversion"       # bare-module dir

# `import m2a_transformer` needs src/ on sys.path. Always required.
_src_path = str(SRC_DIR)
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)


def ensure_tnp_on_path() -> Path:
    """Add the vendored TNP repo dir to sys.path (lazy — only when VC runs).

    TNP exposes bare top-level modules (``convert``, ``core``, ``dataset`` …),
    so we only inject it right before voice conversion to avoid polluting the
    namespace during accompaniment-only runs.
    """
    p = str(TNP_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    return TNP_DIR


DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"

__all__ = ["REPO_ROOT", "SRC_DIR", "M2A_DIR", "TNP_DIR",
           "DEFAULT_CONFIG", "ensure_tnp_on_path"]
