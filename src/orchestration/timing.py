"""Tiny stage-timing helper — logs a start/done (Xs) line around a block of work."""
from __future__ import annotations

import time
from contextlib import contextmanager

from m2a_transformer.utils.logger import logger


@contextmanager
def timed(label: str):
    logger.info(f"▶ {label} 시작")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.info(f"■ {label} 완료 ({time.perf_counter() - t0:.2f}s)")
