"""Web resilience: UI endpoints must degrade, never 500, when LM Studio is down.

RED batch: /api/status and /api/models currently propagate connect errors
as HTTP 500, leaving dashboard as a white page with menu only.
"""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient


def _client_no_lm(monkeypatch):
    """Make every get_lm_client() raise like an unreachable LM Studio."""
    import lm_optimizer.api.routes as routes

    async def _raise():
        raise ConnectionError("LM Studio unreachable")

    monkeypatch.setattr(routes, "get_lm_client", _raise)


class TestOfflineDegradation:
    def test_status_degrades_to_disconnected(self, monkeypatch):
        from lm_optimizer.api.main import app

        _client_no_lm(monkeypatch)
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/api/status")
            assert r.status_code == 200, f"status must degrade, got {r.status_code}"
            assert r.json()["lm_studio"]["connected"] is False

    def test_models_degrades_to_empty(self, monkeypatch):
        from lm_optimizer.api.main import app

        _client_no_lm(monkeypatch)
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/api/models")
            assert r.status_code == 200, f"models must degrade, got {r.status_code}"
            assert r.json()["models"] == []

    def test_pages_still_render_offline(self, monkeypatch):
        from lm_optimizer.api.main import app

        _client_no_lm(monkeypatch)
        with TestClient(app, raise_server_exceptions=False) as c:
            for path in ("/", "/history", "/sandbox", "/settings"):
                r = c.get(path)
                assert r.status_code == 200, path
                assert "main-content" in r.text, path


class TestReadEndpointsAreLoadFree:
    def test_status_models_getmodel_never_echo_load(self, monkeypatch):
        """Read endpoints must not load any model (echo probe >> UI timeout)."""
        import lm_optimizer.api.routes as routes
        from lm_optimizer.api.main import app

        seen: list = []

        class FakeClient:
            async def health_check(self):
                return False

            async def list_models(self, **kwargs):
                return []

            async def get_model(self, model_id):
                return None

            async def close(self):
                pass

        async def fake_create(url, echo_probe=True):
            seen.append(echo_probe)
            return FakeClient()

        monkeypatch.setattr(routes, "create_client", fake_create)
        with TestClient(app, raise_server_exceptions=False) as c:
            assert c.get("/api/status").status_code == 200
            assert c.get("/api/models").status_code == 200
            assert c.get("/api/models/whatever").status_code == 404
        assert seen, "expected client creation"
        assert all(e is False for e in seen), f"read endpoints echo-loaded: {seen}"


class TestListEndpointsSurviveCorruptRows:
    def test_runs_skips_unreadable_row(self, monkeypatch):
        import lm_optimizer.api.routes as routes
        from lm_optimizer.api.main import app

        monkeypatch.setattr(routes.run_repo, "list_all", lambda *a, **k: [object(), object()])
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/api/runs")
            assert r.status_code == 200, f"one bad row must not kill /runs, got {r.status_code}"
            assert r.json()["runs"] == []

    def test_configurations_skips_unreadable_row(self, monkeypatch):
        from types import SimpleNamespace

        import lm_optimizer.api.routes as routes
        from lm_optimizer.api.main import app

        monkeypatch.setattr(
            routes.run_repo, "get", lambda *a, **k: SimpleNamespace(configurations=[object()])
        )
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/api/runs/00000000-0000-0000-0000-000000000000/configurations")
            assert r.status_code == 200, f"got {r.status_code}"
            assert r.json()["configurations"] == []


class TestScorelessRowsPassThrough:
    def test_score_none_accepted(self):
        """Scoreless configs (speed probes, quality_failed) must serialize."""
        from lm_optimizer.api.routes import _convert_config_result
        from lm_optimizer.domain.models import (
            ConfigurationResult,
            ConfigurationStatus,
            LoadConfiguration,
        )

        from uuid import uuid4

        r = ConfigurationResult(
            config=LoadConfiguration(context_length=2048),
            status=ConfigurationStatus.PASSED,
            run_id=uuid4(),
        )
        r.score = None
        out = _convert_config_result(r)
        assert out.score is None

    def test_all_local_scripts_versioned(self):
        """Every local <script src> must carry ?v=N (no stale-cache trap)."""
        import re
        from pathlib import Path

        base = Path(__file__).parent.parent
        unversioned = []
        for f in (base / "ui" / "templates").glob("*.html"):
            for m in re.finditer(r"""src="(/static/js/[\w-]+\.js)(\?v=\d+)?\"""", f.read_text()):
                if not m.group(2):
                    unversioned.append(f"{f.name}: {m.group(1)}")
        assert not unversioned, f"unversioned scripts: {unversioned}"


class TestApplyValidation:
    def test_apply_unknown_model_404_without_load(self, monkeypatch):
        """Apply must validate the model first (no blind load, no echo)."""
        from unittest.mock import MagicMock

        import lm_optimizer.api.routes as routes
        from lm_optimizer.api.main import app

        loads: list = []

        class FakeClient:
            async def get_model(self, model_id):
                return None

            async def load_model(self, *args, **kwargs):
                loads.append(1)
                ok = MagicMock()
                ok.success = True
                ok.identifier = "x"
                return ok

            async def close(self):
                pass

        async def fake_create(url, echo_probe=True):
            assert echo_probe is False, "apply must be load-free until validated"
            return FakeClient()

        monkeypatch.setattr(routes, "create_client", fake_create)
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.post("/api/apply", json={"model_id": "no-such-model", "config": {}})
            assert r.status_code == 404, f"got {r.status_code}: {r.text[:200]}"
            assert loads == [], "must not attempt load for unknown model"


class TestRunListCounts:
    def _run(self, status):
        from lm_optimizer.domain.models import OptimizationRun

        return OptimizationRun(status=status)

    def test_list_runs_carries_counts_and_stale(self, monkeypatch):
        import lm_optimizer.api.routes as routes
        from lm_optimizer.api.main import app
        from lm_optimizer.domain.models import RunStatus

        run = self._run(RunStatus.RUNNING)
        monkeypatch.setattr(routes.run_repo, "list_all", lambda *a, **k: [run])
        monkeypatch.setattr(routes.run_repo, "count_configurations", lambda *a, **k: 3)
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/api/runs")
            assert r.status_code == 200
            row = r.json()["runs"][0]
            assert row["config_count"] == 3
            assert row["stale"] is True, "running without checkpoint must be stale"

    def test_paused_and_success_never_stale(self, monkeypatch):
        import lm_optimizer.api.routes as routes
        from lm_optimizer.api.main import app
        from lm_optimizer.domain.models import RunStatus

        runs = [self._run(RunStatus.PAUSED), self._run(RunStatus.SUCCESS)]
        monkeypatch.setattr(routes.run_repo, "list_all", lambda *a, **k: runs)
        monkeypatch.setattr(routes.run_repo, "count_configurations", lambda *a, **k: 0)
        with TestClient(app, raise_server_exceptions=False) as c:
            rows = c.get("/api/runs").json()["runs"]
            assert [x["stale"] for x in rows] == [False, False]


class TestConfigDetailShape:
    def _cfg(self):
        from uuid import uuid4

        from lm_optimizer.domain.models import (
            BenchmarkMetrics,
            ConfigurationResult,
            ConfigurationStatus,
            LoadConfiguration,
        )

        r = ConfigurationResult(
            config=LoadConfiguration(context_length=2048, flash_attention=True),
            status=ConfigurationStatus.QUALITY_FAILED,
            run_id=uuid4(),
        )
        r.score = None
        r.generation = {"style": "balanced", "reasoning": "off", "phase": "quality"}
        r.metrics = [
            BenchmarkMetrics(
                test_name="coding_task",
                category="coding",
                success=True,
                generation_tok_s=50.0,
                output_text="def find_duplicates(nums): return sorted(set([x for x in nums if nums.count(x) > 1])).",
            )
        ]
        return r

    def test_generation_passthrough(self):
        from lm_optimizer.api.routes import _convert_config_result

        out = _convert_config_result(self._cfg())
        assert out.generation == {"style": "balanced", "reasoning": "off", "phase": "quality"}
        assert out.config.flash_attention is True
        assert out.score is None

    def test_enrich_adds_full_output_and_per_test_quality(self):
        from lm_optimizer.api.routes import _convert_config_result, _enrich_config_detail

        cfg = self._cfg()
        out = _convert_config_result(cfg)
        enriched = _enrich_config_detail(out.model_dump(), cfg, "test-model")
        m = enriched["metrics"][0]
        assert m["output_text"] == cfg.metrics[0].output_text, "full output, not truncated"
        assert "quality_overall" in m, "per-test quality recomputed"


class TestThinkingPlumbing:
    def test_native_thinking_captured(self):
        import asyncio
        from unittest.mock import AsyncMock

        from lm_optimizer.services.lm_studio import LMStudioClient

        client = LMStudioClient(base_url="http://127.0.0.1:1234")

        class Resp:
            def json(self):
                return {
                    "output": [
                        {"type": "reasoning", "content": "let me think..."},
                        {"type": "message", "content": "The answer."},
                    ],
                    "stats": {"input_tokens": 10, "total_output_tokens": 5},
                }

        client._request_with_retry = AsyncMock(return_value=Resp())
        out = asyncio.run(
            client.chat_completion(model="m", input_text="hi", temperature=0.3, max_output_tokens=32)
        )
        assert out["choices"][0]["message"]["content"] == "The answer."
        assert out.get("thinking_text") == "let me think..."

    def test_single_case_carries_thinking_and_prompt(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from lm_optimizer.domain.models import BenchmarkCase
        from lm_optimizer.services.benchmark import BenchmarkService

        async def _chat(**kwargs):
            return {
                "choices": [{"message": {"content": "Answer here, done."}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
                "thinking_text": "internal reasoning trace",
                "_stats": {"tokens_per_second": 50.0, "time_to_first_token_seconds": 0.05},
            }

        c = MagicMock()
        c.chat_completion = AsyncMock(side_effect=_chat)
        svc = BenchmarkService(c)
        case = BenchmarkCase(
            name="t1", category="instruction", prompt="Explain X.", max_tokens=64, temperature=0.3
        )
        m = asyncio.run(svc._run_single_case("m", case))
        assert m.success
        assert m.thinking_text == "internal reasoning trace"
        assert m.prompt == "Explain X."

    def test_api_exposes_thinking_and_prompt(self):
        from lm_optimizer.api.routes import _convert_config_result
        from lm_optimizer.domain.models import BenchmarkMetrics

        cfg_run = TestConfigDetailShape._cfg(self)
        cfg_run.metrics[0].thinking_text = "why this answer"
        cfg_run.metrics[0].prompt = "Write code."
        out = _convert_config_result(cfg_run)
        m = out.metrics[0]
        assert m["thinking_text"] == "why this answer"
        assert m["prompt"] == "Write code."


class TestConfigJsonPage:
    def test_unknown_config_404(self):
        from fastapi.testclient import TestClient

        from lm_optimizer.api.main import app

        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/results/00000000-0000-0000-0000-000000000000/configs/00000000-0000-0000-0000-000000000000")
            assert r.status_code == 404, r.status_code


class TestCapabilityCache:
    def test_second_connect_skips_probing(self):
        import asyncio
        from unittest.mock import AsyncMock

        from lm_optimizer.services import lm_studio as mod
        from lm_optimizer.services.lm_studio import LMStudioClient

        mod._CAPS_CACHE.clear()
        calls = {"post": 0}

        class FakeResp:
            def json(self):
                return {"models": []}

        async def fake_request(method, path, **kwargs):
            if method == "GET":
                return FakeResp()
            calls["post"] += 1
            raise ConnectionError("probe must not run on cache hit")

        async def go(url):
            client = LMStudioClient(base_url=url)
            client._request_with_retry = AsyncMock(side_effect=fake_request)
            await client.connect(echo_probe=False)
            await client.close()
            return client

        url = "http://127.0.0.1:19999"
        asyncio.run(go(url))
        first_posts = calls["post"]
        assert first_posts > 0, "first connect must probe"
        asyncio.run(go(url))
        assert calls["post"] == first_posts, "second connect must use cache"
        mod._CAPS_CACHE.clear()


class TestStartupWarmup:
    def test_warmup_connects_load_free_and_never_raises(self, monkeypatch):
        import lm_optimizer.api.routes as routes

        calls: list = []

        class FakeClient:
            async def close(self):
                calls.append("close")

        async def fake_get_client(echo_probe=True):
            calls.append(echo_probe)
            return FakeClient()

        async def boom(echo_probe=True):
            raise ConnectionError("down")

        import asyncio

        monkeypatch.setattr(routes, "get_lm_client", fake_get_client)
        assert asyncio.run(routes.warm_capability_cache()) is True
        assert calls[0] is False, "warmup must be load-free"
        assert "close" in calls

        monkeypatch.setattr(routes, "get_lm_client", boom)
        assert asyncio.run(routes.warm_capability_cache()) is False


class TestNavUnified:
    def test_every_page_has_canonical_nav(self):
        from pathlib import Path

        base = Path(__file__).parent.parent
        pages = ["dashboard.html", "history.html", "results.html", "settings.html",
                 "sandbox.html", "json.html"]
        for name in pages:
            text = (base / "ui" / "templates" / name).read_text()
            assert 'viewBox="0 0 24 24"' in text, f"{name} missing logo"
            assert 'id="lm-status"' in text, f"{name} missing status badge"
            assert "theme-toggle" in text, f"{name} missing theme toggle"
            assert "navstatus.js?v=" in text, f"{name} missing status poller"
            for href in ('href="/"', 'href="/history"', 'href="/sandbox"', 'href="/settings"'):
                assert href in text, f"{name} missing {href}"


class TestUnloadGuard:
    def _client(self, stubborn=False):
        from unittest.mock import AsyncMock, MagicMock

        state = {"loaded": ["model-a"]}

        async def _unload_all():
            if not stubborn:
                state["loaded"] = []
            return {"model-a": not stubborn}

        async def _instances():
            return [
                {"instance_id": "i1", "model": m, "config": {}}
                for m in state["loaded"]
            ]

        c = MagicMock()
        c.unload_all = AsyncMock(side_effect=_unload_all)
        c.get_loaded_instances = AsyncMock(side_effect=_instances)
        return c

    def test_clean_host_passes(self):
        import asyncio

        from lm_optimizer.services.unload_guard import assert_unloaded

        asyncio.run(assert_unloaded(self._client(), purpose="test"))

    def test_stubborn_instance_raises(self):
        import asyncio

        import pytest

        from lm_optimizer.services.unload_guard import UnloadNotClean, assert_unloaded

        with pytest.raises(UnloadNotClean, match="[Kk]eep Model in Memory"):
            asyncio.run(
                assert_unloaded(self._client(stubborn=True), purpose="test", attempts=2)
            )


class TestMatrixFailFast:
    def test_dirty_row_raises(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        import pytest

        from lm_optimizer.domain.models import LoadConfiguration
        from lm_optimizer.services.matrix import run_one
        from lm_optimizer.services.unload_guard import UnloadNotClean

        async def _chat(**kwargs):
            return {
                "choices": [{"message": {"content": "hi there buddy"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
                "_stats": {"tokens_per_second": 50.0, "time_to_first_token_seconds": 0.05},
            }

        ok_load = MagicMock()
        ok_load.success = True
        ok_load.loaded_config = None
        c = MagicMock()
        c.load_model = AsyncMock(return_value=ok_load)
        c.chat_completion = AsyncMock(side_effect=_chat)
        c.ensure_unloaded = AsyncMock(return_value=True)
        # Host stays dirty no matter what: unload sweep is a no-op here.
        c.unload_all = AsyncMock(return_value={})
        c.get_loaded_instances = AsyncMock(return_value=[
            {"instance_id": "stuck", "model": "m", "config": {}}
        ])
        with pytest.raises(UnloadNotClean):
            asyncio.run(run_one(c, "m", LoadConfiguration(context_length=2048)))


class TestFlashLessonPersists:
    def _opt(self):
        from unittest.mock import AsyncMock, MagicMock

        from lm_optimizer.domain.models import ModelIdentity, OptimizationRun
        from lm_optimizer.services.optimizer import AdaptiveOptimizer, OptimizationState
        from lm_optimizer.services.search_space import SearchSpace

        opt = AdaptiveOptimizer.__new__(AdaptiveOptimizer)
        opt.client = MagicMock()
        opt.benchmark = MagicMock()
        opt.quality = MagicMock()
        opt.search_generator = MagicMock()
        opt.state = OptimizationState(
            run=OptimizationRun(model=ModelIdentity(id="m", name="m")),
            search_space=SearchSpace(),
        )
        opt._progress_cb = None
        return opt

    def _failed(self, error):
        from lm_optimizer.domain.models import ConfigurationResult

        r = ConfigurationResult()
        r.error = error
        return r

    def test_lesson_persists_to_settings(self):
        from lm_optimizer.database.repositories import settings_repo

        opt = self._opt()
        opt._note_flash_requirement(
            self._failed("V Cache Quantization requires flash attention (KV-Q4)")
        )
        assert opt.state.flash_required is True
        assert settings_repo.get("flash_required") == "1"

    def test_other_errors_do_not_set_flag(self):
        from lm_optimizer.database.repositories import settings_repo

        opt = self._opt()
        opt._note_flash_requirement(self._failed("OOM allocating KV cache"))
        assert opt.state.flash_required is False
        assert settings_repo.get("flash_required") is None

    def test_prepare_applies_persisted_advice(self):
        from lm_optimizer.database.repositories import settings_repo

        settings_repo.set("flash_required", "1")
        opt = self._opt()
        opt._apply_persisted_flash_advice()
        assert opt.state.flash_required is True

    def test_flash_off_success_clears_flag(self):
        from lm_optimizer.database.repositories import settings_repo
        from lm_optimizer.domain.models import LoadConfiguration

        settings_repo.set("flash_required", "1")
        opt = self._opt()
        opt.state.flash_required = True
        opt._clear_flash_requirement(LoadConfiguration(flash_attention=False))
        assert opt.state.flash_required is False
        assert settings_repo.get("flash_required") == "0"


class TestDisplayTotal:
    def _state(self, tested, planned, estimate=5040):
        from types import SimpleNamespace

        space = SimpleNamespace(estimate_size=lambda: estimate)
        return SimpleNamespace(tested_configs=[None] * tested,
                               planned_total=planned, search_space=space)

    def test_prefers_planned(self):
        from lm_optimizer.services.optimizer import _display_total

        assert _display_total(self._state(5, 25)) == 25

    def test_never_below_tested(self):
        from lm_optimizer.services.optimizer import _display_total

        assert _display_total(self._state(30, 25)) == 30

    def test_legacy_falls_back_to_estimate(self):
        from lm_optimizer.services.optimizer import _display_total

        assert _display_total(self._state(2, 0)) == 5040
        assert _display_total(self._state(0, 0, estimate=0)) == 1

    def test_completed_run_reports_full_bar(self):
        # End-of-run display contract: CLI sets the bar to 100 explicitly;
        # _display_total never exceeds tested reality mid-run.
        from lm_optimizer.services.optimizer import _display_total

        assert _display_total(self._state(25, 30)) == 30


class TestGenerationProfiles:
    def test_known_and_unknown_families(self):
        from lm_optimizer.services.generation_profiles import profiles_for

        qwen = profiles_for("qwen3.8-9b-distill", "qwen35")
        assert set(qwen) == {"precision", "chat", "creative"}
        assert qwen["precision"]["ours"]["temperature"] == 0.1
        assert qwen["precision"]["publisher"]["temperature"] == 0.6
        unknown = profiles_for("some-future-model-x", None)
        assert unknown["chat"]["publisher"] is None
        assert unknown["precision"]["ours"]["temperature"] == 0.1

    def test_every_number_has_source(self):
        from lm_optimizer.services.generation_profiles import profiles_for

        for mid, arch in [("qwen3.8-9b-distill", "qwen35"),
                          ("mistralai/ministral-3-3b", "mistral3"),
                          ("google/gemma-4-12b", "gemma4"),
                          ("openai/gpt-oss-20b", "gpt-oss"),
                          ("mystery", None)]:
            for _name, prof in profiles_for(mid, arch).items():
                assert prof["ours"].get("source"), (mid, _name)
                pub = prof.get("publisher")
                if pub is not None:
                    assert pub.get("source"), (mid, _name)


class TestSamplingPassthrough:
    def test_top_p_top_k_reach_chat(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from lm_optimizer.domain.models import BenchmarkCase
        from lm_optimizer.services.benchmark import BenchmarkService

        seen = {}

        async def _chat(**kwargs):
            seen.update({k: kwargs.get(k) for k in ("temperature", "top_p", "top_k")})
            return {
                "choices": [{"message": {"content": "hash tables map keys fast."}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
                "_stats": {"tokens_per_second": 50.0, "time_to_first_token_seconds": 0.05},
            }

        c = MagicMock()
        c.chat_completion = AsyncMock(side_effect=_chat)
        svc = BenchmarkService(c)
        case = BenchmarkCase(
            name="short_instruction", category="instruction", prompt="Hi.",
            max_tokens=64, temperature=0.6, top_p=0.95, top_k=20,
        )
        m = asyncio.run(svc._run_single_case("m", case))
        assert m.success
        assert seen == {"temperature": 0.6, "top_p": 0.95, "top_k": 20}

    def test_defaults_stay_none(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from lm_optimizer.domain.models import BenchmarkCase
        from lm_optimizer.services.benchmark import BenchmarkService

        seen = {}

        async def _chat(**kwargs):
            seen.update({k: kwargs.get(k) for k in ("top_p", "top_k")})
            return {
                "choices": [{"message": {"content": "hash tables map keys fast."}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
                "_stats": {"tokens_per_second": 50.0, "time_to_first_token_seconds": 0.05},
            }

        c = MagicMock()
        c.chat_completion = AsyncMock(side_effect=_chat)
        svc = BenchmarkService(c)
        case = BenchmarkCase(
            name="short_instruction", category="instruction", prompt="Hi.", max_tokens=64,
        )
        asyncio.run(svc._run_single_case("m", case))
        assert seen == {"top_p": None, "top_k": None}


class TestSamplingSweep:
    def test_combos_from_publisher(self):
        from lm_optimizer.services.sampling_sweep import sampling_combos

        combos = sampling_combos("qwen3.8-9b-distill", "qwen35")
        assert [c["profile"] for c in combos] == ["precision", "chat", "creative"]
        prec = combos[0]
        assert (prec["temperature"], prec["top_p"], prec["top_k"]) == (0.6, 0.95, 20)

    def test_runner_scores_each_combo(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from lm_optimizer.domain.models import LoadConfiguration
        from lm_optimizer.services.benchmark import BenchmarkConfig, BenchmarkService
        from lm_optimizer.services.quality import QualityEvaluator
        from lm_optimizer.services.sampling_sweep import run_sampling_sweep

        async def _chat(**kwargs):
            return {
                "choices": [{"message": {"content": "hash tables map keys fast and well."}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 10, "total_tokens": 22},
                "_stats": {"tokens_per_second": 40.0, "time_to_first_token_seconds": 0.05},
            }

        c = MagicMock()
        ok = MagicMock()
        ok.success = True
        ok.loaded_config = None
        ok.identifier = "x"
        c.load_model = AsyncMock(return_value=ok)
        c.ensure_unloaded = AsyncMock(return_value=True)
        c.chat_completion = AsyncMock(side_effect=_chat)
        svc = BenchmarkService(
            c, benchmark_config=BenchmarkConfig(repetitions=1, warmup_repetitions=0))
        res = asyncio.run(run_sampling_sweep(
            svc, QualityEvaluator(), "m",
            LoadConfiguration(context_length=2048), 2048,
            [{"profile": "chat", "temperature": 0.7, "top_p": 0.8, "top_k": 20}],
            repetitions=1,
        ))
        assert len(res["combos"]) == 1
        assert res["combos"][0]["quality"] is not None
        assert res["combos"][0]["tok_s"] > 0

    def test_sample_sweep_command_registered(self):
        import typer

        from lm_optimizer.cli.main import app

        info = typer.main.get_command(app)
        assert "sample-sweep" in info.commands
        params = [p.name for p in info.commands["sample-sweep"].params]
        assert "config" in params


class TestUnifiedMemory:
    def test_darwin_reports_unified_without_nvidia(self, monkeypatch):
        import platform as _platform

        import lm_optimizer.services.hostguard as hg

        def _no_subprocess(*a, **k):
            raise AssertionError("nvidia-smi must not run on Darwin")

        monkeypatch.setattr(_platform, "system", lambda: "Darwin")
        monkeypatch.setattr(hg.subprocess, "run", _no_subprocess)
        snap = hg.gpu_free_mb()
        assert snap and snap[0].get("unified_memory") is True
        lines = hg.format_snapshot({"gpus": snap, "mem": {}, "verified_empty": True})
        assert any("unified" in line.lower() for line in lines)

    def test_nvidia_path_unchanged(self, monkeypatch):
        import lm_optimizer.services.hostguard as hg

        snap = hg.gpu_free_mb()
        assert isinstance(snap, list)


class TestDarkMode:
    def test_theme_wired_everywhere(self):
        from pathlib import Path

        base = Path(__file__).parent.parent
        css = (base / "ui" / "static" / "css" / "app.css").read_text()
        assert ".dark body" in css, "dark surface overrides missing"
        assert ".dark .card" in css, "dark card overrides missing"
        for f in (base / "ui" / "templates").glob("*.html"):
            text = f.read_text()
            assert "theme.js?v=" in text, f"{f.name} missing theme.js"
            assert "theme-toggle" in text, f"{f.name} missing toggle button"
        theme = (base / "ui" / "static" / "js" / "theme.js").read_text()
        assert "prefers-color-scheme" in theme, "must follow OS theme by default"
        assert "localStorage" in theme, "must persist choice"


class TestJsSyntax:
    def test_all_ui_scripts_parse(self):
        """Every UI script must be parseable (a syntax error whitescreens pages)."""
        import shutil
        import subprocess
        from pathlib import Path

        node = shutil.which("node")
        if not node:
            pytest.skip("node not available")
        js_dir = Path(__file__).parent.parent / "ui" / "static" / "js"
        failures = []
        for f in sorted(js_dir.glob("*.js")):
            r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
            if r.returncode != 0:
                failures.append(f"{f.name}: {r.stderr.strip().splitlines()[1] if r.stderr else ''}")
        assert not failures, f"JS syntax errors: {failures}"

    def test_asset_versions_consistent(self):
        """All ?v=N cache-busters must share one version and target real files."""
        import re
        from pathlib import Path

        base = Path(__file__).parent.parent
        versions: set[str] = set()
        missing: list[str] = []
        targets = list((base / "ui" / "templates").glob("*.html")) + list(
            (base / "ui" / "static" / "js").glob("*.js")
        )
        for f in targets:
            for m in re.finditer(r"[\"'](?:\./|/static/js/)([\w-]+\.js)\?v=(\d+)", f.read_text()):
                versions.add(m.group(2))
                if not (base / "ui" / "static" / "js" / m.group(1)).exists():
                    missing.append(f"{f.name} -> {m.group(1)}")
        assert len(versions) == 1, f"mixed asset versions: {sorted(versions)}"
        assert not missing, f"dangling JS references: {missing}"


class TestConfigDetailRoute:
    def test_single_config_route_has_no_double_prefix(self):
        """GET /api/runs/{run}/configurations/{cfg} must exist (was /api/api/...)."""
        from lm_optimizer.api.main import app

        paths = set()
        for r in app.routes:
            if hasattr(r, "path"):
                paths.add(r.path)
        assert "/api/api/runs/{run_id}/configurations/{config_id}" not in paths
