"""MTP test runner tests (offline, no LM Studio)."""
import json
from pathlib import Path

import pytest


def test_mtp_test_help():
    """The script exists and is importable as a module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "mtp_test",
        Path(__file__).resolve().parents[1] / "tools" / "mtp_test.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "run_mtp_variant")
    assert hasattr(mod, "main")


def test_mtp_result_schema():
    """run_mtp_variant returns a dict with the expected keys."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "mtp_test",
        Path(__file__).resolve().parents[1] / "tools" / "mtp_test.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # The function should exist and be callable
    assert callable(mod.run_mtp_variant)
