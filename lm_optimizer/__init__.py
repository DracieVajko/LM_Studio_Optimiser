"""LM Studio Auto Optimizer - Main package."""

from .config import config
from .logging_config import get_logger, setup_logging

__version__ = "1.3.2"
__all__ = ["config", "get_logger", "setup_logging"]
