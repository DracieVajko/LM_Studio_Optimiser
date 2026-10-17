"""Logging must not multiply records or leak fixtures into production logs.

Two defects motivated these tests:

1. ``setup_logging()`` appended a fresh ``FileHandler`` on every call. CLI
   entrypoints and tests both call it, so an N-times-called process wrote
   every record N times - that is how a single session produced a 57 MB
   ``optimizer.log``.
2. Tests resolved ``config.storage.logs_dir`` to the real ``data/logs``, so
   fixture records (``"model": "m"``, ...) were appended to the production log.
"""

import logging
from pathlib import Path

import pytest

from lm_optimizer.logging_config import setup_logging


def _managed_handlers() -> list[logging.Handler]:
    return [
        h
        for h in logging.getLogger().handlers
        if getattr(h, "_optimizer_managed", False)
    ]


class TestSetupLoggingIdempotent:
    def test_repeated_calls_keep_one_handler(self):
        for _ in range(10):
            setup_logging()

        assert len(_managed_handlers()) == 1, (
            f"setup_logging() accumulated {_managed_handlers().__len__()} handlers; "
            "each one writes every record again"
        )

    def test_no_record_is_written_more_than_once(self, tmp_path):
        from lm_optimizer.logging_config import get_logger

        log = tmp_path / "dup.log"
        setup_logging()
        for h in _managed_handlers():
            if isinstance(h, logging.FileHandler):
                h.close()

        import lm_optimizer.logging_config as lc

        log_dir = lc.config.storage.logs_dir
        lc.config.storage.logs_dir = tmp_path
        try:
            setup_logging()
            get_logger("t").info("marker-line", once=True)
            for h in _managed_handlers():
                h.flush()
            text = (tmp_path / "optimizer.log").read_text(encoding="utf-8", errors="replace")
        finally:
            lc.config.storage.logs_dir = log_dir

        assert text.count("marker-line") == 1, (
            f"record appeared {text.count('marker-line')}x in the log file"
        )
        assert log.exists() is False


class TestLogIsolation:
    def test_logs_dir_redirected_to_tmp(self):
        """The autouse fixture must move logs_dir out of the project tree."""
        from lm_optimizer.config import config

        assert "data\\logs" not in str(config.storage.logs_dir).replace("/", "\\"), (
            f"tests would write into production: {config.storage.logs_dir}"
        )

    def test_setup_logging_writes_into_tmp_path(self, tmp_path):
        from lm_optimizer.config import config
        from lm_optimizer.logging_config import get_logger

        setup_logging()
        get_logger("iso").info("fixture-only-record", model="m")

        for h in _managed_handlers():
            h.flush()

        produced = Path(config.storage.logs_dir) / "optimizer.log"
        assert produced.exists(), f"handler did not point at {produced}"
        assert str(tmp_path) in str(produced), (
            f"handler wrote outside tmp_path: {produced}"
        )
        assert "fixture-only-record" in produced.read_text(
            encoding="utf-8", errors="replace"
        )

        # Clean up so later tests do not inherit this handler.
        root = logging.getLogger()
        for h in _managed_handlers():
            root.removeHandler(h)
            h.close()
