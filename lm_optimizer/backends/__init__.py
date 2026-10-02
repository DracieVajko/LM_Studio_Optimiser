"""Backend seam packages (split-ready: one subpackage per backend)."""

from lm_optimizer.backends.base import BackendClient, assert_conforms

__all__ = ["BackendClient", "assert_conforms"]
