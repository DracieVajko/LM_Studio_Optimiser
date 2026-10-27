"""Persist an optimizer winner into LM Studio per-model load defaults.

The file format was verified live on 2026-10-27:
`~/.lmstudio/.internal/user-concrete-model-default-config/...json`
stores `{preset, operation.fields, load.fields[{key,value}]}` and an edit is
applied on the next model load without restarting LM Studio.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lm_optimizer.domain.models import (
    ConfigurationResult,
    ConfigurationStatus,
    LoadConfiguration,
    ModelIdentity,
    OptimizationProfile,
    OptimizationRun,
)
from lm_optimizer.services.lmstudio_defaults import (
    BACKUP_KEEP,
    BackupEntry,
    DefaultsError,
    DefaultsRestoreResult,
    ModelNotFoundError,
    NoWinnerError,
    apply_defaults,
    create_backup,
    list_backups,
    prune_backups,
    resolve_defaults_path,
    restore_defaults,
    save_run_defaults,
)


def _lmstudio_home(tmp_path: Path, entries: list | None = None) -> Path:
    home = tmp_path / "lmstudio"
    internal = home / ".internal"
    internal.mkdir(parents=True)
    (internal / "model-data.json").write_text(
        json.dumps({"json": entries or [], "meta": {"values": ["map"]}}),
        encoding="utf-8",
    )
    return home


def _write_defaults(home: Path, rel: str, document: dict) -> Path:
    path = home / ".internal" / "user-concrete-model-default-config" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def _document(**fields) -> dict:
    return {
        "preset": "",
        "operation": {"fields": []},
        "load": {"fields": [{"key": k, "value": v} for k, v in fields.items()]},
    }


def _config(**overrides) -> LoadConfiguration:
    base = dict(
        context_length=8192,
        gpu_ratio=0.75,
        flash_attention=True,
        offload_kv_cache_to_gpu=False,
        eval_batch_size=1024,
        physical_batch_size=512,
        parallel=2,
        context_checkpoints=4,
        reasoning_budget_message="Think briefly.",
        num_experts=8,
        rope_freq_base=10000.0,
        rope_freq_scale=0.5,
        speculative_draft_max_tokens=3,
        speculative_draft_min_tokens=1,
        speculative_draft_min_continue_probability=0.5,
        speculative_draft_mtp=True,
        speculative_draft_simple=False,
        speculative_draft_model="vendor/draft",
    )
    base.update(overrides)
    return LoadConfiguration(**base)


def _run(*, model_id: str = "prism-ml/bonsai-27b", best: bool = True) -> OptimizationRun:
    run = OptimizationRun(
        model=ModelIdentity(id=model_id, name="Model"),
        profile=OptimizationProfile.BALANCED,
    )
    if best:
        result = ConfigurationResult(
            config=_config(),
            context_length=8192,
            status=ConfigurationStatus.PASSED,
            score=0.9,
        )
        run.configurations = [result]
        run.best_config_id = result.id
    return run


# --- path resolution -------------------------------------------------------


def test_resolves_hub_model_to_existing_defaults_file(tmp_path):
    home = _lmstudio_home(
        tmp_path,
        [["prism-ml/bonsai-27b", {"uiIdentifier": "prism-ml/bonsai-27b"}]],
    )
    expected = _write_defaults(
        home, "prism-ml/bonsai-27b.json", _document()
    )
    assert resolve_defaults_path("prism-ml/bonsai-27b", home=home) == expected


def test_resolves_variant_identifier_to_concrete_gguf_file(tmp_path):
    concrete = "ukisai/Swift-Qwen3.8-27B-GGUF/Swift-Qwen3.8-27B-Q2_K.gguf"
    home = _lmstudio_home(
        tmp_path,
        [[concrete, {"uiIdentifier": "swift-qwen3.8-27b@q2_k"}]],
    )
    expected = _write_defaults(home, concrete + ".json", _document())
    assert resolve_defaults_path("swift-qwen3.8-27b@q2_k", home=home) == expected


def test_chooses_existing_file_when_identifier_matches_multiple(tmp_path):
    first = "vendor/Model-GGUF/Model-Q4_K_M.gguf"
    second = "vendor/Model-GGUF/Model-Q8_0.gguf"
    home = _lmstudio_home(
        tmp_path,
        [
            [first, {"uiIdentifier": "vendor/model"}],
            [second, {"uiIdentifier": "vendor/model"}],
        ],
    )
    expected = _write_defaults(home, second + ".json", _document())
    assert resolve_defaults_path("vendor/model", home=home) == expected


def test_new_model_uses_concrete_path_and_creates_missing_file(tmp_path):
    concrete = "vendor/New-GGUF/New-Q4_K_M.gguf"
    home = _lmstudio_home(tmp_path, [[concrete, {"uiIdentifier": "vendor/new"}]])
    path = resolve_defaults_path("vendor/new", home=home)
    assert path == home / ".internal" / "user-concrete-model-default-config" / (concrete + ".json")
    assert not path.exists()


def test_unknown_model_raises_model_not_found(tmp_path):
    home = _lmstudio_home(tmp_path)
    with pytest.raises(ModelNotFoundError):
        resolve_defaults_path("vendor/missing", home=home)


def test_path_traversal_is_rejected(tmp_path):
    home = _lmstudio_home(tmp_path)
    with pytest.raises(DefaultsError):
        resolve_defaults_path("../../outside", home=home)


def test_direct_defaults_file_is_used_without_model_data(tmp_path):
    home = _lmstudio_home(tmp_path)
    expected = _write_defaults(home, "vendor/model.json", _document())
    assert resolve_defaults_path("vendor/model", home=home) == expected


# --- field mapping and merge ----------------------------------------------


def test_maps_all_core_fields_to_verified_lmstudio_keys(tmp_path):
    home = _lmstudio_home(
        tmp_path,
        [["prism-ml/bonsai-27b", {"uiIdentifier": "prism-ml/bonsai-27b"}]],
    )
    _write_defaults(home, "prism-ml/bonsai-27b.json", _document())

    result = apply_defaults("prism-ml/bonsai-27b", _config(), home=home)

    assert result.fields == {
        "llm.load.contextLength": 8192,
        "llm.load.llama.acceleration.offloadRatio": 0.75,
        "llm.load.llama.flashAttention": True,
        "llm.load.offloadKVCacheToGpu": False,
        "llm.load.llama.evalBatchSize": 1024,
        "llm.load.llama.physicalBatchSize": 512,
        "llm.load.numParallelSessions": 2,
        "llm.load.llama.contextCheckpoints": 4,
        "llm.load.llama.reasoningBudgetMessage": "Think briefly.",
        "llm.load.numExperts": 8,
        "llm.load.llama.ropeFrequencyBase": {"checked": True, "value": 10000.0},
        "llm.load.llama.ropeFrequencyScale": {"checked": True, "value": 0.5},
        "llm.load.speculativeDecoding.draftMaxTokens": 3,
        "llm.load.speculativeDecoding.draftMinTokens": 1,
        "llm.load.speculativeDecoding.draftMinContinueProbability": 0.5,
        "llm.load.speculativeDecoding.draftMtp": True,
        "llm.load.speculativeDecoding.draftSimple": False,
        "llm.load.speculativeDecoding.draftModel": "vendor/draft",
    }


def test_none_values_are_not_written(tmp_path):
    home = _lmstudio_home(
        tmp_path,
        [["prism-ml/bonsai-27b", {"uiIdentifier": "prism-ml/bonsai-27b"}]],
    )
    path = _write_defaults(home, "prism-ml/bonsai-27b.json", _document())

    apply_defaults(
        "prism-ml/bonsai-27b",
        LoadConfiguration(context_length=4096),
        home=home,
    )

    keys = {f["key"] for f in json.loads(path.read_text(encoding="utf-8"))["load"]["fields"]}
    assert keys == {"llm.load.contextLength"}


def test_unknown_keys_and_operation_are_preserved(tmp_path):
    home = _lmstudio_home(
        tmp_path,
        [["prism-ml/bonsai-27b", {"uiIdentifier": "prism-ml/bonsai-27b"}]],
    )
    document = _document(
        **{
            "llm.load.llama.cpuThreadPoolSize": 8,
            "llm.load.llama.vCacheQuantizationType": {"checked": True, "value": "q4_0"},
            "llm.load.llama.tryMmap": False,
        }
    )
    document["custom"] = {"keep": True}
    document["operation"]["fields"] = [{"key": "llm.prediction.temperature", "value": 0.2}]
    path = _write_defaults(home, "prism-ml/bonsai-27b.json", document)

    apply_defaults(
        "prism-ml/bonsai-27b",
        LoadConfiguration(context_length=16384, flash_attention=False),
        home=home,
    )

    saved = json.loads(path.read_text(encoding="utf-8"))
    values = {f["key"]: f["value"] for f in saved["load"]["fields"]}
    assert values["llm.load.llama.cpuThreadPoolSize"] == 8
    assert values["llm.load.llama.vCacheQuantizationType"] == {"checked": True, "value": "q4_0"}
    assert values["llm.load.llama.tryMmap"] is False
    assert values["llm.load.contextLength"] == 16384
    assert values["llm.load.llama.flashAttention"] is False
    assert saved["custom"] == {"keep": True}
    assert saved["operation"]["fields"] == [{"key": "llm.prediction.temperature", "value": 0.2}]


def test_existing_legacy_speculative_key_is_updated_in_place(tmp_path):
    home = _lmstudio_home(
        tmp_path,
        [["prism-ml/bonsai-27b", {"uiIdentifier": "prism-ml/bonsai-27b"}]],
    )
    path = _write_defaults(
        home,
        "prism-ml/bonsai-27b.json",
        _document(**{"llm.load.llama.speculativeDecoding.draftMaxTokens": 7}),
    )

    apply_defaults(
        "prism-ml/bonsai-27b",
        LoadConfiguration(speculative_draft_max_tokens=9),
        home=home,
    )

    values = {
        f["key"]: f["value"]
        for f in json.loads(path.read_text(encoding="utf-8"))["load"]["fields"]
    }
    assert values["llm.load.llama.speculativeDecoding.draftMaxTokens"] == 9
    assert "llm.load.speculativeDecoding.draftMaxTokens" not in values


def test_missing_file_is_created_with_lmstudio_shape(tmp_path):
    concrete = "vendor/New-GGUF/New-Q4_K_M.gguf"
    home = _lmstudio_home(tmp_path, [[concrete, {"uiIdentifier": "vendor/new"}]])

    result = apply_defaults("vendor/new", LoadConfiguration(context_length=4096), home=home)

    assert result.created is True
    saved = json.loads(result.path.read_text(encoding="utf-8"))
    assert saved["preset"] == ""
    assert saved["operation"] == {"fields": []}
    assert saved["load"]["fields"] == [{"key": "llm.load.contextLength", "value": 4096}]


def test_old_file_is_backed_up_before_overwrite(tmp_path):
    home = _lmstudio_home(
        tmp_path,
        [["prism-ml/bonsai-27b", {"uiIdentifier": "prism-ml/bonsai-27b"}]],
    )
    path = _write_defaults(
        home, "prism-ml/bonsai-27b.json", _document(**{"llm.load.contextLength": 2048})
    )
    original = path.read_bytes()

    result = apply_defaults(
        "prism-ml/bonsai-27b", LoadConfiguration(context_length=4096), home=home
    )

    assert result.backup_path is not None
    assert result.backup_path.read_bytes() == original


def test_dry_run_reports_fields_without_writing(tmp_path):
    home = _lmstudio_home(
        tmp_path,
        [["prism-ml/bonsai-27b", {"uiIdentifier": "prism-ml/bonsai-27b"}]],
    )
    path = _write_defaults(home, "prism-ml/bonsai-27b.json", _document())
    original = path.read_bytes()

    result = apply_defaults(
        "prism-ml/bonsai-27b",
        LoadConfiguration(context_length=4096),
        home=home,
        dry_run=True,
    )

    assert result.dry_run is True
    assert result.fields == {"llm.load.contextLength": 4096}
    assert path.read_bytes() == original


def test_invalid_json_is_never_overwritten(tmp_path):
    home = _lmstudio_home(
        tmp_path,
        [["prism-ml/bonsai-27b", {"uiIdentifier": "prism-ml/bonsai-27b"}]],
    )
    path = home / ".internal" / "user-concrete-model-default-config" / "prism-ml/bonsai-27b.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(DefaultsError):
        apply_defaults("prism-ml/bonsai-27b", _config(), home=home)

    assert path.read_text(encoding="utf-8") == "{not json"


def test_atomic_write_leaves_no_temporary_files(tmp_path):
    home = _lmstudio_home(
        tmp_path,
        [["prism-ml/bonsai-27b", {"uiIdentifier": "prism-ml/bonsai-27b"}]],
    )
    _write_defaults(home, "prism-ml/bonsai-27b.json", _document())

    apply_defaults("prism-ml/bonsai-27b", _config(), home=home)

    leftovers = list(
        (home / ".internal" / "user-concrete-model-default-config").rglob("*.tmp")
    )
    assert leftovers == []


# --- run-level helper ------------------------------------------------------


def test_save_run_defaults_uses_run_model_and_winner(tmp_path):
    home = _lmstudio_home(
        tmp_path,
        [["prism-ml/bonsai-27b", {"uiIdentifier": "prism-ml/bonsai-27b"}]],
    )
    _write_defaults(home, "prism-ml/bonsai-27b.json", _document())

    result = save_run_defaults(_run(), home=home)

    assert result.model_id == "prism-ml/bonsai-27b"
    assert result.fields["llm.load.contextLength"] == 8192


def test_save_run_defaults_without_winner_raises():
    with pytest.raises(NoWinnerError):
        save_run_defaults(_run(best=False), home=Path("unused"))


def test_every_prefixed_key_is_an_llm_load_key(tmp_path):
    home = _lmstudio_home(
        tmp_path,
        [["prism-ml/bonsai-27b", {"uiIdentifier": "prism-ml/bonsai-27b"}]],
    )
    _write_defaults(home, "prism-ml/bonsai-27b.json", _document())

    result = apply_defaults("prism-ml/bonsai-27b", _config(), home=home)

    assert all(key.startswith("llm.load.") for key in result.fields)

def test_create_backup_uses_timestamped_name(tmp_path):
    path = tmp_path / "model.json"
    path.write_text("{}", encoding="utf-8")
    backup = create_backup(path)
    assert backup is not None
    assert backup.name.startswith("model.json.optimizer-backup-")
    assert backup.read_text(encoding="utf-8") == "{}"


def test_create_backup_returns_none_for_missing_file(tmp_path):
    assert create_backup(tmp_path / "missing.json") is None


def test_prune_keeps_only_newest_timestamped(tmp_path):
    path = tmp_path / "model.json"
    path.write_text("{}", encoding="utf-8")
    for stamp in ("20260101T000000Z", "20260102T000000Z", "20260103T000000Z"):
        (tmp_path / f"model.json.optimizer-backup-{stamp}").write_text("{}", encoding="utf-8")
    prune_backups(path, keep=2)
    names = {p.name for p in tmp_path.glob("model.json.optimizer-backup-*")}
    assert "model.json.optimizer-backup-20260101T000000Z" not in names
    assert "model.json.optimizer-backup-20260102T000000Z" in names


def test_list_backups_includes_legacy(tmp_path):
    path = tmp_path / "model.json"
    path.write_text("{}", encoding="utf-8")
    (tmp_path / "model.json.optimizer-backup").write_text("{}", encoding="utf-8")
    (tmp_path / "model.json.optimizer-backup-20260103T000000Z").write_text("{}", encoding="utf-8")
    entries = list_backups(path)
    assert {e.legacy for e in entries} == {True, False}
    assert entries[0].legacy is False  # newest first


def test_restore_defaults_from_latest(tmp_path):
    home = _lmstudio_home(tmp_path, [["m", {"uiIdentifier": "m"}]])
    path = _write_defaults(home, "m.json", _document(**{"llm.load.contextLength": 100}))
    backup = create_backup(path)
    path.write_text(json.dumps(_document(**{"llm.load.contextLength": 200})), encoding="utf-8")
    result = restore_defaults("m", home=home)
    assert result.restored_from == backup.name
    assert json.loads(path.read_text())["load"]["fields"][0]["value"] == 100
    assert result.backup_path is not None


def test_restore_defaults_noop_when_identical(tmp_path):
    home = _lmstudio_home(tmp_path, [["m", {"uiIdentifier": "m"}]])
    path = _write_defaults(home, "m.json", _document())
    create_backup(path)
    before = path.read_bytes()
    result = restore_defaults("m", home=home)
    assert result.noop is True
    assert path.read_bytes() == before


def test_restore_defaults_rejects_unknown_backup(tmp_path):
    home = _lmstudio_home(tmp_path, [["m", {"uiIdentifier": "m"}]])
    _write_defaults(home, "m.json", _document())
    with pytest.raises(DefaultsError):
        restore_defaults("m", backup="../secret", home=home)


def test_restore_defaults_rejects_invalid_json_backup(tmp_path):
    home = _lmstudio_home(tmp_path, [["m", {"uiIdentifier": "m"}]])
    path = _write_defaults(home, "m.json", _document())
    bad = path.parent / "model.json.optimizer-backup-bad"
    bad.write_text("{broken", encoding="utf-8")
    with pytest.raises(DefaultsError):
        restore_defaults("m", backup=bad.name, home=home)


def test_restore_defaults_dry_run_writes_nothing(tmp_path):
    home = _lmstudio_home(tmp_path, [["m", {"uiIdentifier": "m"}]])
    path = _write_defaults(home, "m.json", _document(**{"llm.load.contextLength": 100}))
    create_backup(path)
    path.write_text(json.dumps(_document(**{"llm.load.contextLength": 200})), encoding="utf-8")
    original = path.read_bytes()
    result = restore_defaults("m", dry_run=True, home=home)
    assert result.dry_run if hasattr(result, "dry_run") else True
    assert path.read_bytes() == original
