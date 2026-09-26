#!/usr/bin/env python3
"""Web UI entry point for LM Studio Optimizer."""

import uvicorn

from lm_optimizer.config import config
from lm_optimizer.logging_config import setup_logging

# Setup logging before importing app
setup_logging()

if __name__ == "__main__":
    uvicorn.run(
        "lm_optimizer.api.main:app",
        host=config.web_ui.host,
        port=config.web_ui.port,
        log_level="info",
        reload=False,
    )
