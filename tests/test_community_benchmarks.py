"""Community benchmark database tests (P5, spec 16.4)."""
import pytest

from lm_optimizer.services.community_benchmarks import COMMUNITY_BENCHMARKS
from lm_optimizer.services.recommender import (
    Confidence,
    FitClass,
    HardwareFingerprint,
    ModelCatalogEntry,
    ModelRecommender,
    RecommenderRequest,
)


def test_database_is_populated():
    assert COMMUNITY_BENCHMARKS
    for bench in COMMUNITY_BENCHMARKS:
        assert bench.decode_tps > 0
        assert bench.quant
        assert bench.model_filename
        assert bench.hardware.vram_gb > 0
        assert isinstance(bench.confidence, Confidence)


def test_every_confidence_grade_is_used():
    grades = {b.confidence.value for b in COMMUNITY_BENCHMARKS}
    assert {"A", "B", "C", "D"} & grades


def test_mtp_slowdown_evidence_is_present():
    """MTP must not be assumed a win: a record where MTP slowed the model."""
    mtp_records = [b for b in COMMUNITY_BENCHMARKS if b.mtp]
    assert mtp_records
    # Group ALL records by the base model (the MTP build has a "-mtp"
    # suffix) so the OFF and ON variants of the same model can be compared.
    by_model: dict[tuple[str, str], list] = {}
    for bench in COMMUNITY_BENCHMARKS:
        base = bench.model_filename.replace("-mtp", "")
        key = (base, bench.quant)
        by_model.setdefault(key, []).append(bench)
    slowdowns = [
        group
        for group in by_model.values()
        if len(group) > 1
        and any(g.mtp for g in group)
        and any(not g.mtp for g in group)
    ]
    assert slowdowns, "expected at least one MTP-off vs MTP-on comparison"
    # And at least one case where MTP actually SLOWED the model down.
    slowed = [
        group
        for group in slowdowns
        if any(
            g.mtp and any(not o.mtp and o.decode_tps > g.decode_tps for o in group)
            for g in group
        )
    ]
    assert slowed, "expected evidence that MTP can slow a model down"


def test_quant_ladder_ordering_is_preserved():
    """Q3 is slower than Q4 in the batiai M4 Max ladder (dequant overhead)."""
    def _tps(filename: str) -> float:
        for bench in COMMUNITY_BENCHMARKS:
            if bench.model_filename == filename:
                return bench.decode_tps
        return 0.0

    q3 = _tps("qwen3.8-27b:q3")
    q4 = _tps("qwen3.8-27b:q4")
    iq4 = _tps("qwen3.8-27b:iq4")
    assert q3 < q4
    assert q3 < iq4


def test_community_prior_is_used_when_no_local_measurement():
    entry = ModelCatalogEntry(
        repository="batiai/qwen3.8-27b",
        filename="qwen3.8-27b:q4",
        architecture="qwen35",
        quant="Q4_K_M",
        bits_per_weight=4.0,
        file_size_gb=16.5,
        total_params_b=27.0,
    )
    rec = ModelRecommender(catalog=[entry])
    hw = HardwareFingerprint(
        gpu_name="Apple M4 Max",
        vram_gb=128.0,
        gpu_bandwidth_gbps=546.0,
        cpu_name="Apple M4 Max",
        physical_cores=14,
        logical_threads=14,
        ram_gb=128.0,
        ram_channels=1,
        disk_type="NVMe SSD",
        os_name="macOS",
    )
    request = RecommenderRequest(hardware=hw, min_context=16384, min_tps=5.0)
    result = rec.recommend(request)
    assert result[0].local_evidence is False
    assert result[0].expected_tps == pytest.approx(15.9)
    assert result[0].confidence is Confidence.B
    assert any("community prior" in r for r in result[0].reasons)


def test_local_measurement_beats_community_prior():
    entry = ModelCatalogEntry(
        repository="batiai/qwen3.8-27b",
        filename="qwen3.8-27b:q4",
        architecture="qwen35",
        quant="Q4_K_M",
        bits_per_weight=4.0,
        file_size_gb=16.5,
        total_params_b=27.0,
    )
    rec = ModelRecommender(catalog=[entry])
    rec.add_local_result("qwen3.8-27b:q4", 1.2, run_ids=["run-x"])
    hw = HardwareFingerprint(
        gpu_name="Apple M4 Max", vram_gb=128.0, ram_gb=128.0, ram_channels=1,
        cpu_name="Apple M4 Max", physical_cores=14, logical_threads=14,
        gpu_bandwidth_gbps=546.0, disk_type="NVMe SSD", os_name="macOS",
    )
    result = rec.recommend(RecommenderRequest(hardware=hw, min_context=16384, min_tps=1.0))
    assert result[0].expected_tps == pytest.approx(1.2)
    assert result[0].local_evidence is True
    assert result[0].confidence is Confidence.A
