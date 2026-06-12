"""Tests for logging configuration."""
from __future__ import annotations

import logging
import logging.handlers

from src.utils.logging_config import setup_logging


def test_setup_logging_creates_log_dir(tmp_path):
    """setup_logging should create the log directory."""
    log_dir = tmp_path / "logs"
    _reset_logging()
    setup_logging(log_dir=log_dir)
    assert log_dir.exists()
    _reset_logging()


def test_setup_logging_adds_handlers(tmp_path):
    """Root logger should get console + file handlers."""
    log_dir = tmp_path / "logs"
    _reset_logging()
    setup_logging(log_dir=log_dir)
    root = logging.getLogger()
    handler_types = [type(h) for h in root.handlers]
    assert logging.StreamHandler in handler_types
    assert logging.handlers.RotatingFileHandler in handler_types
    _reset_logging()


def test_setup_logging_idempotent(tmp_path):
    """Calling setup_logging twice should not duplicate handlers."""
    log_dir = tmp_path / "logs"
    _reset_logging()
    setup_logging(log_dir=log_dir)
    count = len(logging.getLogger().handlers)
    setup_logging(log_dir=log_dir)
    assert len(logging.getLogger().handlers) == count
    _reset_logging()


def test_log_messages_written_to_file(tmp_path):
    """Messages should appear in the log file."""
    log_dir = tmp_path / "logs"
    _reset_logging()
    setup_logging(log_dir=log_dir)
    test_logger = logging.getLogger("test.logging_config")
    test_logger.info("hello_from_test")
    # Flush handlers
    for h in logging.getLogger().handlers:
        h.flush()
    log_file = log_dir / "autolabel.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "hello_from_test" in content
    _reset_logging()


def test_third_party_loggers_suppressed(tmp_path):
    """Ultralytics and PIL loggers should be at WARNING level."""
    log_dir = tmp_path / "logs"
    _reset_logging()
    setup_logging(log_dir=log_dir)
    assert logging.getLogger("ultralytics").level == logging.WARNING
    assert logging.getLogger("PIL").level == logging.WARNING
    _reset_logging()


def _reset_logging():
    """Reset logging state between tests."""
    import src.utils.logging_config as mod
    mod._configured = False
    root = logging.getLogger()
    for h in root.handlers[:]:
        h.close()
        root.removeHandler(h)
    root.setLevel(logging.WARNING)
