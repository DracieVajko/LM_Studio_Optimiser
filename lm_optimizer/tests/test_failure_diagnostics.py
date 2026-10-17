"""Task 11: repeated-failure diagnostics (Mixtral-style cascade).

Signature grouping + circuit-breaker + duration display. Mock-only.
"""
from datetime import datetime


def _cfg(ctx=4096, flash=True, error=None, status="failed", duration_ms=0.0, score=None):
    from lm_optimizer.domain.models import (
        ConfigurationResult, ConfigurationStatus, LoadConfiguration,
    )

    return ConfigurationResult(
        config=LoadConfiguration(context_length=ctx, flash_attention=flash),
        context_length=ctx, status=ConfigurationStatus(status),
        metrics=[], tested_at=datetime.now(), error=error,
        duration_ms=duration_ms, score=score,
    )


class TestSignatureGrouping:
    def test_same_cause_groups(self):
        from lm_optimizer.services.failure_states import normalize_error_signature

        assert normalize_error_signature("CUDA OOM ctx=8192") == \
            normalize_error_signature("cuda oom ctx=16384")

    def test_different_causes_split(self):
        from lm_optimizer.services.failure_states import normalize_error_signature

        assert normalize_error_signature("CUDA OOM") != \
            normalize_error_signature("connection refused")

    def test_empty_signature(self):
        from lm_optimizer.services.failure_states import normalize_error_signature

        assert normalize_error_signature(None) == "unknown"
        assert normalize_error_signature("   ") == "unknown"


class TestCircuitBreaker:
    def _configs(self):
        return [
            _cfg(ctx=4096, error="CUDA OOM ctx=4096", duration_ms=1200.0),
            _cfg(ctx=8192, error="CUDA OOM ctx=8192", duration_ms=2300.0),
            _cfg(ctx=16384, error="CUDA OOM ctx=16384", duration_ms=3100.0),
        ]

    def test_trips_on_repeated_identical_across_configs(self):
        from lm_optimizer.services.failure_states import should_stop_repeating

        verdict = should_stop_repeating(self._configs())
        assert verdict["trip"] is True
        assert verdict["count"] == 3

    def test_no_trip_with_baseline_success(self):
        from lm_optimizer.services.failure_states import should_stop_repeating

        cfgs = self._configs() + [_cfg(ctx=2048, status="passed", score=0.5)]
        verdict = should_stop_repeating(cfgs)
        assert verdict["trip"] is False

    def test_flash_failure_never_trips_global(self):
        from lm_optimizer.services.failure_states import should_stop_repeating

        cfgs = [_cfg(ctx=c, flash=False, error="flash attention not supported")
                for c in (4096, 8192, 16384, 32768, 65536)]
        verdict = should_stop_repeating(cfgs)
        assert verdict["trip"] is False
        assert verdict["reason"] == "flash-specific"

    def test_timing_display_rows(self):
        from lm_optimizer.services.failure_states import failure_signature_rows

        rows = failure_signature_rows(self._configs())
        text = "\n".join(rows)
        assert "3x" in text
        assert "6600" in text  # total duration_ms summed
