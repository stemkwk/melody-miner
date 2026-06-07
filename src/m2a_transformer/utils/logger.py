"""Loguru-based logger. Import as `from m2a_transformer.utils.logger import logger`.

`attach_file_sink(path)` mirrors all subsequent log records to a file so logs
survive crashes / disconnected sessions on rented GPUs."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _logger

_logger.remove()
_logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | "
           "<cyan>{name}</cyan>:<cyan>{line}</cyan> - {message}",
)

logger = _logger


def attach_file_sink(path: str | Path, level: str = "INFO") -> int:
    """Add a rotating-file sink. Returns the loguru handler id so callers can
    detach it later if needed. Safe to call multiple times — each call adds an
    additional sink."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return logger.add(
        str(p),
        level=level,
        rotation="50 MB",
        retention=5,
        enqueue=True,        # crash-safe across processes
        backtrace=True,
        diagnose=False,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {name}:{line} - {message}",
    )
