"""Model Recommender unit tests (P5, spec section 16)."""
import pytest

from lm_optimizer.services.recommender import (
    CommunityBenchmark,
    Confidence,
    FitClass,
    HardwareFingerprint,
    ModelCatalogEntry,
    ModelRecommender,
    RecommenderRequest,
    SoftwareFingerprint,
    UseCase,
    estimate_tps,
    predict_fit,
    similarity,
)


def _hw(**kw):
    base = dict(
        gpu_name="NVIDIA GeForce RTX 3060 Laptop GPU",
        vram_gb=6.0,
        gpu_bandwidth_gbps=192.0,
        gpu_tdp_w=115.0,
        cpu_name="AMD Ryzen 7 5800H",
        physical_cores=8,
        logical_threads=16,
        ram_gb=32.0,
        ram_channels=1,
        disk_type="NVMe SSD",
        os_name="Windows",
    )
    base.update(kw)
    return HardwareFingerprint(**base)


def _entry(**kw):
    base = dict(
        repository="bartowski/Qwen3.8-27B-GGUF",
        filename="Qwen3.8-27B-IQ3_XS.gguf",
        architecture="qwen35",
        quant="IQ3_XS",
        bits_per_weight=3.0,
        file_size_gb=12.8,
        total_params_b=27.0,
        max_context=262144,
        capabilities=["chat", "tools", "vision"],
    )
    base.update(kw)
    return ModelCatalogEntry(**base)


# --- fit prediction ---

def test_q1_0_27b_fits_vram_but_q4_does_not():
    """The core lesson: parameter count does not decide fit, file size does."""
    req = RecommenderRequest(hardware=_hw(), min_tps=1.0, min_context=4096)
    q1 = _entry(filename="bonsai-q1.gguf", quant="Q1_0", bits_per_weight=1.0, file_size_gb=4.4)
    q4 = _entry(filename="qwen-q4km.gguf", quant="Q4_K_M", bits_per_weight=4.0, file_size_gb=17.4)

    fit_q1, reasons_q1, _ = predict_fit(q1, req)
    fit_q4, reasons_q4, _ = predict_fit(q4, req)

    assert fit_q1 is FitClass.GPU_RESIDENT
    assert fit_q4 is not FitClass.GPU_RESIDENT
    assert any("VRAM" in r or "fits" in r for r in reasons_q1)


def test_vram_only_rejects_hybrid():
    req = RecommenderRequest(hardware=_hw(), vram_only=True, min_context=4096, min_tps=0.1)
    big = _entry(file_size_gb=17.4, bits_per_weight=4.0)
    fit, reasons, _ = predict_fit(big, req)
    assert fit is FitClass.NOT_RECOMMENDED


def test_hybrid_class_depends_on_use_case():
    req_reasoning = RecommenderRequest(
        hardware=_hw(), use_case=UseCase.REASONING, min_context=4096, min_tps=0.1
    )
    req_fast = RecommenderRequest(
        hardware=_hw(), use_case=UseCase.FAST, min_context=4096, min_tps=0.1
    )
    big = _entry(file_size_gb=17.4, bits_per_weight=4.0)
    assert predict_fit(big, req_reasoning)[0] is FitClass.HYBRID_QUALITY
    assert predict_fit(big, req_fast)[0] is FitClass.HYBRID_INTERACTIVE


def test_missing_capability_is_not_recommended():
    req = RecommenderRequest(
        hardware=_hw(), required_capabilities=["vision"], min_context=4096
    )
    entry = _entry(capabilities=["chat"])
    fit, reasons, _ = predict_fit(entry, req)
    assert fit is FitClass.NOT_RECOMMENDED
    assert "capabilities" in reasons[0]


def test_model_size_limit_is_enforced():
    req = RecommenderRequest(hardware=_hw(), max_model_size_gb=10.0, min_context=4096)
    entry = _entry(file_size_gb=12.8)
    fit, reasons, _ = predict_fit(entry, req)
    assert fit is FitClass.NOT_RECOMMENDED
    assert "exceeds" in reasons[0]


# --- speed estimation ---

def test_moe_active_params_beat_dense():
    req = RecommenderRequest(hardware=_hw(), min_context=4096, min_tps=0.1)
    dense = _entry(total_params_b=27.0, active_params_b=None, file_size_gb=12.8)
    moe = _entry(
        filename="moe.gguf",
        total_params_b=35.0,
        active_params_b=3.0,
        file_size_gb=13.0,
    )
    dense_tps = estimate_tps(dense, req)
    moe_tps = estimate_tps(moe, req)
    assert moe_tps is not None and dense_tps is not None
    assert moe_tps > dense_tps * 3


def test_single_channel_ram_penalises_hybrid():
    req_single = RecommenderRequest(hardware=_hw(ram_channels=1), min_context=4096, min_tps=0.1)
    req_dual = RecommenderRequest(hardware=_hw(ram_channels=2), min_context=4096, min_tps=0.1)
    entry = _entry(file_size_gb=17.4, bits_per_weight=4.0)
    assert estimate_tps(entry, req_single) < estimate_tps(entry, req_dual)


def test_estimate_is_none_without_bandwidth():
    req = RecommenderRequest(hardware=_hw(gpu_bandwidth_gbps=0.0), min_context=4096)
    assert estimate_tps(_entry(), req) is None


# --- similarity ---

def test_similarity_is_one_for_identical_hardware():
    a = _hw()
    b = _hw()
    assert similarity(a, b) == pytest.approx(1.0, abs=0.001)


def test_similarity_drops_with_different_hardware():
    a = _hw()
    b = _hw(gpu_name="NVIDIA GeForce RTX 4090", vram_gb=24.0, ram_channels=2)
    assert similarity(a, b) < 0.7


def test_ram_channel_weight_rises_when_spilling_to_ram():
    a = _hw(ram_channels=1)
    b = _hw(ram_channels=2)
    assert similarity(a, b, spills_to_ram=True) < similarity(a, b, spills_to_ram=False)


# --- recommender ---

def test_local_measurement_outranks_community_prior():
    catalog = [_entry(filename="model.gguf")]
    bench = CommunityBenchmark(
        hardware=_hw(),
        model_repository="x",
        model_filename="model.gguf",
        quant="IQ3_XS",
        context_length=4096,
        gpu_offload=1.0,
        flash_attention=True,
        kv_on_gpu=True,
        kv_quant="f16",
        eval_batch_size=256,
        threads=8,
        mmap=True,
        mtp=False,
        prompt_tps=50.0,
        decode_tps=3.0,
        source="forum-post",
        confidence=Confidence.C,
    )
    rec = ModelRecommender(catalog=catalog, community=[bench])
    req = RecommenderRequest(hardware=_hw(), min_context=4096, min_tps=0.1)
    result = rec.recommend(req)
    assert result[0].expected_tps == pytest.approx(3.0)
    assert result[0].local_evidence is False

    rec.add_local_result("model.gguf", 9.5, run_ids=["run-123"])
    result = rec.recommend(req)
    assert result[0].expected_tps == pytest.approx(9.5)
    assert result[0].local_evidence is True
    assert result[0].confidence is Confidence.A
    assert result[0].source_run_ids == ["run-123"]


def test_not_recommended_models_sort_last():
    catalog = [
        _entry(filename="too-big.gguf", file_size_gb=40.0),
        _entry(filename="fits.gguf", file_size_gb=4.0, bits_per_weight=1.0),
    ]
    rec = ModelRecommender(catalog=catalog)
    req = RecommenderRequest(hardware=_hw(), min_context=4096, min_tps=0.1)
    result = rec.recommend(req)
    assert result[0].entry.filename == "fits.gguf"
    assert result[-1].fit_class is FitClass.NOT_RECOMMENDED


def test_recommendation_serializes():
    import json

    rec = ModelRecommender(catalog=[_entry()])
    req = RecommenderRequest(hardware=_hw(), min_context=4096, min_tps=0.1)
    payload = rec.to_json(rec.recommend(req))
    data = json.loads(payload)
    assert data[0]["model"]["filename"] == "Qwen3.8-27B-IQ3_XS.gguf"
    assert "fit_class" in data[0]


def test_software_fingerprint_default():
    sw = SoftwareFingerprint()
    assert sw.backend == "lm-studio"
