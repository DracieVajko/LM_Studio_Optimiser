"""Structured logging configuration."""

import logging
import sys

import structlog
from structlog.types import EventDict, WrappedLogger

from .config import config


def add_log_level(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add log level to event dict."""
    event_dict["level"] = method_name.upper()
    return event_dict


def add_timestamp(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add ISO timestamp to event dict."""
    import datetime

    event_dict["timestamp"] = datetime.datetime.now(datetime.UTC).isoformat()
    return event_dict


def setup_logging() -> None:
    """Configure structured logging."""
    log_dir = config.storage.logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "optimizer.log"

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        add_log_level,
        add_timestamp,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if config.logging.format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, config.logging.level),
    )

    # File handler for persistent logs
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, config.logging.level))
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(getattr(logging, config.logging.level))


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)
