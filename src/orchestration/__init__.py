"""melody-miner — WAV → (반주 생성 + 음성 변환) → 믹스 1개 파이프라인.

하나의 입력 WAV에서 두 갈래를 돌려 한 곡으로 합친다:
  Branch A: WAV → MIDI 전사 → M2A Transformer 반주 생성 → 반주 WAV
  Branch B: WAV → TNP voice conversion(목표 화자) → 변환 보컬 WAV
  Merge:    변환 보컬 + 생성 반주 → 믹스 WAV

두 원본 프로젝트는 ``src/`` 아래에 소스 복사(vendoring)되어 있고,
이 패키지가 둘을 오케스트레이션한다.
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
