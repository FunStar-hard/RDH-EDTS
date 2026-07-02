"""Logging configuration helper."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "threshold_schnorr",
    log_file: Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a logger that writes to *stderr* and optionally to *log_file*."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger