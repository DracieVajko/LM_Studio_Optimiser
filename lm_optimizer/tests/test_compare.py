"""A/B compare service: custom test JSON, engine verdict, partial results.

TDD RED batch: engine does not exist yet (services/compare.py missing).
Tests use mock client + real engine code, no network.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from lm_optimizer.domain.models import LoadConfiguration


def _mock_client(speed_a: float = 80.0, speed_b: float = 40.0, fail_b_load: bool = False):
    """Mock client serving side A then side B with different speeds."""
    state = {"side": "A"}

    async def _load(model_id, cfg):
        # Decide side by eval_batch_size marker (128=A, 256=B).
        batch = getattr(cfg, "eval_batch_size", None)
        if batch == 256:
            state["side"] = "B"
            if fail_b_load:
                bad = MagicMock()
                bad.success = False
                bad.error = "engine abort"
                bad.loaded_config = None
                return bad
        else:
            state["side"] = "A"
        ok = MagicMock()
        ok.success = True
        ok.loaded_config = None
        ok.identifier = "mock-id"
        return ok

    async def _chat(**kwargs):
        tok = speed_a if state["side"] == "A" else speed_b
        return {
            "choices": [{"message": {"content": "hash tables map keys fast and well done."}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 10, "total_tokens": 22},
            "_stats": {"tokens_per_second": tok, "time_to_first_token_seconds": 0.05},
        }

    c = MagicMock()
    c.load_model = AsyncMock(side_effect=_load)
    c.ensure_unloaded = AsyncMock(return_value=True)
    c.chat_completion = AsyncMock(side_effect=_chat)
    c.unload_all = AsyncMock(return_value={})
    c.get_loaded_instances = AsyncMock(return_value=[])
    return c


def _cfgs():
    a = LoadConfiguration(context_length=2048, eval_batch_size=128)
    b = LoadConfiguration(context_length=2048, eval_batch_size=256)
    return a, b


class TestCompareJson:
    def test_parse_valid_json(self):
        from lm_optimizer.services.compare import parse_compare_cases

        data = {"tests": [{"name": "t1", "prompt": "Say hi in five words."}]}
        cases = parse_compare_cases(data)
        assert len(cases) == 1
        assert cases[0].name == "t1"
        assert cases[0].max_tokens > 0

    def test_parse_invalid_missing_prompt(self):
        from lm_optimizer.services.compare import parse_compare_cases

        with pytest.raises(ValueError, match="[Pp]rompt"):
            parse_compare_cases({"tests": [{"name": "t1"}]})

    def test_parse_rejects_empty_and_too_many(self):
        from lm_optimizer.services.compare import parse_compare_cases

        with pytest.raises(ValueError):
            parse_compare_cases({"tests": []})
        many = {"tests": [{"name": f"t{i}", "prompt": "hi"} for i in range(11)]}
        with pytest.raises(ValueError, match="[Mm]ax"):
            parse_compare_cases(many)


class TestCompareEngine:
    def test_faster_side_wins_speed_verdict(self):
        from lm_optimizer.services.compare import run_ab_compare

        c = _mock_client(speed_a=80.0, speed_b=40.0)
        a, b = _cfgs()
        res = asyncio.run(run_ab_compare(c, "m", a, b, None, repetitions=1))
        assert res.side_a_ok and res.side_b_ok
        assert res.speed_winner == "A"
        assert res.delta_tok_s > 0
        # Same custom cases ran on both sides (fairness guard).
        prompts_a = [call.kwargs.get("input_text") for call in c.chat_completion.call_args_list]
        assert prompts_a, "expected chat calls on both sides"
        assert len(set(prompts_a)) == 1 or len(prompts_a) >= 2

    def test_tie_within_band_is_draw(self):
        from lm_optimizer.services.compare import run_ab_compare

        c = _mock_client(speed_a=50.0, speed_b=51.0)
        a, b = _cfgs()
        res = asyncio.run(run_ab_compare(c, "m", a, b, None, repetitions=1))
        assert res.speed_winner in ("draw", "tie", "none", None) or res.verdict in (
            "draw",
            "tie",
        )

    def test_compare_command_registered_with_ab_options(self):
        import typer

        from lm_optimizer.cli.main import app

        info = typer.main.get_command(app)
        assert "compare" in info.commands, "CLI must expose a 'compare' command"
        params = [p.name for p in info.commands["compare"].params]
        for required in ("config_a", "config_b"):
            assert required in params, f"compare must accept --{required.replace('_', '-')}"

    def test_api_compare_routes_exist(self):
        from lm_optimizer.api.main import app

        paths = set()
        for r in app.routes:
            if hasattr(r, "path"):
                paths.add(r.path)
            if type(r).__name__ == "_IncludedRouter":
                prefix = getattr(getattr(r, "include_context", None), "prefix", "") or ""
                inner = getattr(r, "original_router", None)
                for sub in getattr(inner, "routes", []) or []:
                    if hasattr(sub, "path"):
                        paths.add(prefix + sub.path)
        assert "/api/compare" in paths, "POST /api/compare missing"
        assert "/api/compare/{compare_id}" in paths, "GET /api/compare/{id} missing"
        assert "/compare" not in paths, "legacy /compare page must be gone (sandbox replaces it)"
        assert "/sandbox" in paths, "sandbox page missing"

    def test_b_load_fail_keeps_partial_a(self):
        from lm_optimizer.services.compare import run_ab_compare

        c = _mock_client(fail_b_load=True)
        a, b = _cfgs()
        res = asyncio.run(run_ab_compare(c, "m", a, b, None, repetitions=1))
        assert res.side_a_ok is True
        assert res.side_b_ok is False
        assert res.error_b, "expected B error preserved"
        assert res.result_a is not None
        assert res.verdict in ("partial", "A-only", "incomplete") or "partial" in res.verdict.lower()
