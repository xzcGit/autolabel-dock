"""Centralized logging configuration for AutoLabel Dock."""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

_configured = False


def setup_logging(log_dir: Path | None = None) -> None:
    """Configure application-wide logging with console and rotating file handlers.

    Args:
        log_dir: Directory for log files. Defaults to ~/.autolabel/logs/.
    """
    global _configured
    if _configured:
        return

    if log_dir is None:
        log_dir = Path.home() / ".autolabel" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = "[%(asctime)s] [%(levelname)-5s] [%(name)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console handler — INFO level
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Rotating file handler — DEBUG level, 5MB × 3 backups
    log_file = log_dir / "autolabel.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    _configured = True
