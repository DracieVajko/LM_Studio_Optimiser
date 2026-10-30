"""Foreign-model watchdog: detection, protection, opt-in unload, fail-closed."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from lm_optimizer.services.foreign_watchdog import (
    BACKEND_LMSTUDIO,
    BACKEND_OLLAMA,
    ForeignModelWatchdog,
    LoadedModel,
    build_watchdog,
    normalize_expected,
)


def _lmstudio(instances):
    client = MagicMock()
    client.get_loaded_instances = AsyncMock(return_value=instances)
    client.unload_model = AsyncMock(return_value=True)
    return client


def _inst(model, iid="i1", config=None):
    return {"model": model, "instance_id": iid, "config": config or {}}


def _ollama(names):
    client = MagicMock()
    client.refresh_running_models = AsyncMock(return_value=set(names))
    client.unload = AsyncMock(return_value=True)
    return client


class TestDetection:
    @pytest.mark.asyncio
    async def test_foreign_model_detected(self):
        client = _lmstudio([_inst("openai/gpt-oss-20b"), _inst("qwen/qwen3-coder-30b", "i2")])
        wd = build_watchdog(client, BACKEND_LMSTUDIO, model_under_test="openai/gpt-oss-20b")

        r = await wd.check()

        assert r.inspected
        assert not r.clean
        assert [f.model.model_id for f in r.foreign] == ["qwen/qwen3-coder-30b"]
        assert r.foreign[0].model.instance_id == "i2"
        assert "FOREIGN" in r.summary()

    @pytest.mark.asyncio
    async def test_clean_when_only_expected_loaded(self):
        client = _lmstudio([_inst("openai/gpt-oss-20b")])
        wd = build_watchdog(client, BACKEND_LMSTUDIO, model_under_test="openai/gpt-oss-20b")

        r = await wd.check()

        assert r.clean
        assert r.foreign == []
        assert [m.model_id for m in r.expected] == ["openai/gpt-oss-20b"]

    @pytest.mark.asyncio
    async def test_echo_probe_model_is_protected(self):
        """connect() legitimately loads the smallest model; never flag it."""
        client = _lmstudio([_inst("openai/gpt-oss-20b"), _inst("prism-ml/bonsai-27b", "i2")])
        wd = build_watchdog(
            client,
            BACKEND_LMSTUDIO,
            model_under_test="openai/gpt-oss-20b",
            echo_probe_model="prism-ml/bonsai-27b",
        )

        r = await wd.check()

        assert r.clean, r.summary()

    @pytest.mark.asyncio
    async def test_match_is_case_and_tag_insensitive(self):
        client = _lmstudio([_inst("openai/gpt-oss-20b"), _inst("Qwen/Qwen3-Coder-30B:latest", "i2")])
        wd = build_watchdog(client, BACKEND_LMSTUDIO, model_under_test="openai/gpt-oss-20b")

        r = await wd.check()

        assert [f.model.model_id for f in r.foreign] == ["Qwen/Qwen3-Coder-30B:latest"]

    @pytest.mark.asyncio
    async def test_ollama_tags_match_bare_name(self):
        client = _ollama(["qwen3:8b", "other:latest"])
        wd = build_watchdog(client, BACKEND_OLLAMA, model_under_test="qwen3:8b")

        r = await wd.check()

        assert [f.model.model_id for f in r.foreign] == ["other:latest"]


class TestFailClosed:
    @pytest.mark.asyncio
    async def test_uninspectable_host_is_unknown_not_clean(self):
        client = MagicMock()
        client.get_loaded_instances = AsyncMock(side_effect=ConnectionError("host down"))

        wd = build_watchdog(client, BACKEND_LMSTUDIO, model_under_test="m")
        r = await wd.check()

        assert r.inspected is False
        assert r.clean is False, "a host we cannot inspect is never 'clean'"
        assert "UNKNOWN" in r.summary()

    @pytest.mark.asyncio
    async def test_no_loaded_models_is_clean(self):
        client = _lmstudio([])
        wd = build_watchdog(client, BACKEND_LMSTUDIO, model_under_test="m")

        r = await wd.check()

        assert r.inspected and r.clean
        assert r.expected == []


class TestUnloadIsOptIn:
    @pytest.mark.asyncio
    async def test_detect_only_by_default_never_unloads(self):
        client = _lmstudio([_inst("model-a"), _inst("qwen/qwen3-coder-30b", "i2")])
        wd = build_watchdog(client, BACKEND_LMSTUDIO, model_under_test="model-a")

        r = await wd.check()

        assert r.unloaded == []
        client.unload_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_unload_when_enabled(self):
        client = _lmstudio([_inst("model-a"), _inst("qwen/qwen3-coder-30b", "i2")])
        wd = ForeignModelWatchdog(
            client,
            BACKEND_LMSTUDIO,
            expected_models=["model-a"],
            allow_unload=True,
        )

        r = await wd.check()

        assert r.unloaded == ["qwen/qwen3-coder-30b"]
        client.unload_model.assert_awaited_once_with(
            model_id="qwen/qwen3-coder-30b", identifier="i2"
        )

    @pytest.mark.asyncio
    async def test_never_unloads_the_expected_model(self):
        client = _lmstudio([_inst("model-a"), _inst("model-b", "i2")])
        wd = ForeignModelWatchdog(
            client, BACKEND_LMSTUDIO, expected_models=["model-a"], allow_unload=True
        )

        r = await wd.check()

        assert r.unloaded == ["model-b"]
        for call in client.unload_model.await_args_list:
            assert call.kwargs["model_id"] != "model-a"

    @pytest.mark.asyncio
    async def test_failed_unload_is_reported_not_swallowed(self):
        client = _lmstudio([_inst("model-a"), _inst("model-x", "i2")])
        client.unload_model = AsyncMock(return_value=False)

        wd = ForeignModelWatchdog(
            client, BACKEND_LMSTUDIO, expected_models=["model-a"], allow_unload=True
        )
        r = await wd.check()

        assert r.unloaded == []
        assert r.unload_failed == ["model-x"]
        assert "unload_failed" in r.summary()

    @pytest.mark.asyncio
    async def test_unload_exception_is_contained(self):
        client = _lmstudio([_inst("model-a"), _inst("model-x", "i2")])
        client.unload_model = AsyncMock(side_effect=RuntimeError("boom"))

        wd = ForeignModelWatchdog(
            client, BACKEND_LMSTUDIO, expected_models=["model-a"], allow_unload=True
        )
        r = await wd.check()  # must not raise

        assert r.unload_failed == ["model-x"]

    @pytest.mark.asyncio
    async def test_ollama_unload_path(self):
        client = _ollama(["model-a", "rogue:latest"])
        wd = ForeignModelWatchdog(
            client, BACKEND_OLLAMA, expected_models=["model-a"], allow_unload=True
        )

        r = await wd.check()

        assert r.unloaded == ["rogue:latest"]
        client.unload.assert_awaited_once_with("rogue:latest")


class TestExpectedSet:
    def test_normalize_drops_empties(self):
        # Tag stripping happens in LoadedModel.key(); the expected set keeps
        # the bare name so both sides compare identically.
        assert "" not in normalize_expected(["", None])
        assert normalize_expected(["Model-A", None]) == {"model-a"}

    @pytest.mark.asyncio
    async def test_set_expected_replaces(self):
        client = _lmstudio([_inst("model-b")])
        wd = build_watchdog(client, BACKEND_LMSTUDIO, model_under_test="model-a")
        assert (await wd.check()).foreign

        wd.set_expected(["model-b"])
        r = await wd.check()
        assert r.clean

    @pytest.mark.asyncio
    async def test_protect_adds_without_dropping(self):
        client = _lmstudio([_inst("model-a"), _inst("probe"), _inst("rogue")])
        wd = build_watchdog(client, BACKEND_LMSTUDIO, model_under_test="model-a")
        wd.protect(["probe"])

        r = await wd.check()

        assert [f.model.model_id for f in r.foreign] == ["rogue"]

    @pytest.mark.asyncio
    async def test_wait_until_clean_retries(self):
        calls = {"n": 0}

        async def _instances():
            calls["n"] += 1
            if calls["n"] < 3:
                return [_inst("model-a"), _inst("rogue")]
            return [_inst("model-a")]

        client = MagicMock()
        client.get_loaded_instances = _instances
        client.unload_model = AsyncMock(return_value=True)

        wd = ForeignModelWatchdog(client, BACKEND_LMSTUDIO, expected_models=["model-a"])
        r = await wd.wait_until_clean(attempts=5, delay_s=0)

        assert r.clean
        assert len(wd.history) == 3

    @pytest.mark.asyncio
    async def test_worst_reports_last_offending_report(self):
        client = _lmstudio([_inst("rogue")])
        wd = ForeignModelWatchdog(client, BACKEND_LMSTUDIO, expected_models=["model-a"])

        await wd.check()
        worst = wd.worst()

        assert worst is not None
        assert worst.foreign[0].model.model_id == "rogue"
        assert worst.checked_at is not None