"""Model Recommender: catalog, fit prediction, similarity, community priors.

Implements the master spec (LM_STUDIO_OPTIMISER_MASTER_SPEC_SK.md) section 16:

- 16.1 Inputs: hardware fingerprint (GPU/VRAM, CPU, RAM + CHANNELS, disk),
  OS/backend, use case, minimum TPS, minimum context, VRAM-only vs hybrid
  allowed, quality preference, max model size, required capabilities.
- 16.2 Output classes: GPU-resident, Hybrid interactive, Hybrid quality-first,
  Not recommended.
- 16.3 Exact model record: repository, filename, SHA256, architecture,
  total/active params, quant, bits/weight, file size, max context,
  capabilities, expected fit, source/confidence, suggested starting config.
- 16.4 Community benchmark database with confidence grades A-D. LOCAL
  measurements always outrank community priors.
- 16.5 Similarity: GPU architecture, VRAM, GPU bandwidth/TDP, CPU, RAM
  channels/bandwidth, OS, backend, runtime version. Models spilling into
  RAM raise the weight of RAM bandwidth/channel similarity.

The recommender NEVER hardcodes a VRAM limit and never treats parameter
count as a fit predictor: Bonsai 27B Q1_0 (4.4 GB) runs on a 6 GB card
while a 27B Q4 (17 GB) does not - file size and bits/weight decide.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# --- Fit classes (spec 16.2) ----------------------------------------------


class FitClass(str, Enum):
    GPU_RESIDENT = "gpu_resident"
    HYBRID_INTERACTIVE = "hybrid_interactive"
    HYBRID_QUALITY = "hybrid_quality"
    NOT_RECOMMENDED = "not_recommended"


class UseCase(str, Enum):
    GENERAL = "general"
    CODING = "coding"
    REASONING = "reasoning"
    FAST = "fast"
    LONG_CONTEXT = "long_context"
    TOOLS = "tools"


class Confidence(str, Enum):
    """Community benchmark confidence grade (spec 16.4)."""

    A = "A"  # raw JSON + exact hash + full settings
    B = "B"  # complete command/config + benchmark output
    C = "C"  # hardware/model/quant/context/TPS
    D = "D"  # community claim with missing settings


#: Local measurements always win over community priors (spec 16.4).
_CONFIDENCE_PRIOR: dict[str, float] = {
    Confidence.A.value: 0.85,
    Confidence.B.value: 0.70,
    Confidence.C.value: 0.45,
    Confidence.D.value: 0.20,
}


# --- Hardware / software fingerprint (spec 16.1, 17.1) -------------------


@dataclass
class HardwareFingerprint:
    """Everything that decides whether a model is usable here."""

    gpu_name: str = ""
    vram_gb: float = 0.0
    gpu_bandwidth_gbps: float = 0.0
    gpu_tdp_w: float = 0.0
    cpu_name: str = ""
    physical_cores: int = 0
    logical_threads: int = 0
    ram_gb: float = 0.0
    ram_channels: int = 1
    disk_type: str = ""
    os_name: str = ""

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class SoftwareFingerprint:
    backend: str = "lm-studio"
    runtime: str = ""
    os_version: str = ""


# --- Model record (spec 16.3) --------------------------------------------


@dataclass
class ModelCatalogEntry:
    """One exact model/quant record."""

    repository: str
    filename: str
    architecture: str
    quant: str
    bits_per_weight: float
    file_size_gb: float
    total_params_b: float
    active_params_b: float | None = None
    max_context: int = 0
    sha256: str = ""
    capabilities: list[str] = field(default_factory=list)
    source: str = "community"
    confidence: Confidence = Confidence.C
    suggested_config: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    #: LM Studio model id this entry corresponds to (used to attach LOCAL
    #: measurements - the strongest evidence - to the entry).
    model_id: str = ""

    @property
    def is_moe(self) -> bool:
        return bool(self.active_params_b and self.active_params_b < self.total_params_b * 0.5)

    def as_dict(self) -> dict[str, Any]:
        return {
            "repository": self.repository,
            "filename": self.filename,
            "architecture": self.architecture,
            "quant": self.quant,
            "bits_per_weight": self.bits_per_weight,
            "file_size_gb": self.file_size_gb,
            "total_params_b": self.total_params_b,
            "active_params_b": self.active_params_b,
            "max_context": self.max_context,
            "sha256": self.sha256,
            "capabilities": list(self.capabilities),
            "source": self.source,
            "confidence": self.confidence.value,
            "suggested_config": dict(self.suggested_config),
            "notes": self.notes,
            "model_id": self.model_id,
        }


# --- Community benchmark database (spec 16.4) ----------------------------


@dataclass
class CommunityBenchmark:
    """One community-reported measurement."""

    hardware: HardwareFingerprint
    model_repository: str
    model_filename: str
    quant: str
    context_length: int
    gpu_offload: float
    flash_attention: bool
    kv_on_gpu: bool
    kv_quant: str
    eval_batch_size: int
    threads: int
    mmap: bool
    mtp: bool
    prompt_tps: float
    decode_tps: float
    ttft_s: float | None = None
    source: str = ""
    confidence: Confidence = Confidence.C

    def as_dict(self) -> dict[str, Any]:
        return {
            "hardware": self.hardware.as_dict(),
            "model_repository": self.model_repository,
            "model_filename": self.model_filename,
            "quant": self.quant,
            "context_length": self.context_length,
            "gpu_offload": self.gpu_offload,
            "flash_attention": self.flash_attention,
            "kv_on_gpu": self.kv_on_gpu,
            "kv_quant": self.kv_quant,
            "eval_batch_size": self.eval_batch_size,
            "threads": self.threads,
            "mmap": self.mmap,
            "mtp": self.mtp,
            "prompt_tps": self.prompt_tps,
            "decode_tps": self.decode_tps,
            "ttft_s": self.ttft_s,
            "source": self.source,
            "confidence": self.confidence.value,
        }


# --- Similarity (spec 16.5) -----------------------------------------------


def similarity(
    a: HardwareFingerprint, b: HardwareFingerprint, spills_to_ram: bool = False
) -> float:
    """0..1 similarity between two hardware fingerprints.

    RAM-channel similarity is weighted higher for models that spill into
    system RAM (spec 16.5: "models spilling to RAM raise the weight of RAM
    bandwidth/channel similarity").
    """
    score = 0.0
    weight = 0.0

    def _add(value: float, w: float) -> None:
        nonlocal score, weight
        score += value * w
        weight += w

    # GPU architecture / name
    if a.gpu_name and b.gpu_name:
        _add(1.0 if a.gpu_name.strip().lower() == b.gpu_name.strip().lower() else 0.0, 2.0)
    # VRAM
    if a.vram_gb and b.vram_gb:
        ratio = min(a.vram_gb, b.vram_gb) / max(a.vram_gb, b.vram_gb)
        _add(ratio, 2.0)
    # GPU bandwidth
    if a.gpu_bandwidth_gbps and b.gpu_bandwidth_gbps:
        ratio = min(a.gpu_bandwidth_gbps, b.gpu_bandwidth_gbps) / max(
            a.gpu_bandwidth_gbps, b.gpu_bandwidth_gbps
        )
        _add(ratio, 1.5)
    # CPU cores
    if a.physical_cores and b.physical_cores:
        ratio = min(a.physical_cores, b.physical_cores) / max(a.physical_cores, b.physical_cores)
        _add(ratio, 1.0)
    # RAM capacity
    if a.ram_gb and b.ram_gb:
        ratio = min(a.ram_gb, b.ram_gb) / max(a.ram_gb, b.ram_gb)
        _add(ratio, 1.0)
    # RAM channels - heavier when the model spills into RAM
    if a.ram_channels and b.ram_channels:
        _add(
            1.0 if a.ram_channels == b.ram_channels else 0.0,
            2.0 if spills_to_ram else 0.75,
        )
    # OS
    if a.os_name and b.os_name:
        _add(1.0 if a.os_name.lower() == b.os_name.lower() else 0.0, 0.5)
    return (score / weight) if weight else 0.0


# --- Fit prediction (spec 16.2) ------------------------------------------


@dataclass
class RecommenderRequest:
    """What the user wants (spec 16.1)."""

    hardware: HardwareFingerprint
    software: SoftwareFingerprint = field(default_factory=SoftwareFingerprint)
    use_case: UseCase = UseCase.GENERAL
    min_tps: float = 5.0
    min_context: int = 8192
    vram_only: bool = False
    quality_priority: bool = False
    max_model_size_gb: float = 0.0
    required_capabilities: list[str] = field(default_factory=list)
    #: KV cache quantization in the runtime. "f16" = 1.0x, "q8_0" = 0.5x,
    #: "q4_0" = 0.25x of the FP16 KV size. This is a RUNTIME FACT (set in
    #: the LM Studio GUI, not per request), so it is part of the request
    #: context, never a searched dimension.
    kv_cache_quantization: str = "q4_0"

    @property
    def kv_size_factor(self) -> float:
        """Multiplier on the FP16 KV cache size."""
        return {
            "f16": 1.0,
            "fp16": 1.0,
            "q8_0": 0.5,
            "q5_1": 0.3125,
            "q5_0": 0.3125,
            "q4_1": 0.25,
            "q4_0": 0.25,
        }.get((self.kv_cache_quantization or "f16").lower(), 1.0)


@dataclass
class Recommendation:
    """One recommendation with its honest reasoning."""

    entry: ModelCatalogEntry
    fit_class: FitClass
    expected_tps: float | None
    confidence: Confidence
    reasons: list[str] = field(default_factory=list)
    local_evidence: bool = False
    source_run_ids: list[str] = field(default_factory=list)
    #: Context the local measurement was taken at (None = unknown). The
    #: comparison is only meaningful against the requested context.
    measured_context: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.entry.as_dict(),
            "fit_class": self.fit_class.value,
            "expected_tps": self.expected_tps,
            "confidence": self.confidence.value,
            "reasons": list(self.reasons),
            "local_evidence": self.local_evidence,
            "source_run_ids": list(self.source_run_ids),
            "measured_context": self.measured_context,
        }


#: Runtime overhead (weights + KV cache + llama.cpp buffers) per GB of file.
#: Measured on this class of hardware; deliberately conservative.
_VRAM_OVERHEAD_GB = 1.2
#: KV cache size in GB per token per 1B parameters (FP16 K+V, GQA).
#: Derivation for a 27B dense model with 64 layers / 8 KV heads / 128 dim:
#: 2 (K+V) * 64 * 8 * 128 * 2 bytes = 256 KiB per token -> 0.25 MB/token,
#: i.e. 9.3e-6 GB per token per 1B parameters.
#: Sanity check: 27B at 4096 tokens -> ~1.0 GB; at 262144 tokens -> ~65 GB
#: (which is exactly why 256K contexts REQUIRE KV quantization).
_KV_GB_PER_TOKEN_PER_B = 9.3e-6


def predict_fit(
    entry: ModelCatalogEntry, request: RecommenderRequest
) -> tuple[FitClass, list[str], float | None]:
    """Predict the fit class of a catalog entry for this request.

    Returns (fit_class, reasons, expected_tps).

    The predictor uses FILE SIZE (not parameter count) and MoE active
    parameters for the speed estimate. A dense 27B at Q4 needs ~17 GB; a
    27B at Q1 needs ~4.4 GB - the same parameter count, a completely
    different fit.
    """
    reasons: list[str] = []
    hw = request.hardware
    vram = max(0.0, hw.vram_gb)
    ram = max(0.0, hw.ram_gb)

    # --- Capability check ---
    missing = [
        cap
        for cap in request.required_capabilities
        if cap not in (entry.capabilities or [])
    ]
    if missing:
        reasons.append(f"missing capabilities: {missing}")
        return FitClass.NOT_RECOMMENDED, reasons, None

    if request.max_model_size_gb and entry.file_size_gb > request.max_model_size_gb:
        reasons.append(
            f"file {entry.file_size_gb:.1f} GB exceeds the {request.max_model_size_gb:.1f} GB limit"
        )
        return FitClass.NOT_RECOMMENDED, reasons, None

    # --- Memory fit ---
    weights_gb = entry.file_size_gb
    kv_gb = (
        _KV_GB_PER_TOKEN_PER_B
        * entry.total_params_b
        * request.min_context
        * request.kv_size_factor
    )
    total_gb = weights_gb + kv_gb + 0.4  # + runtime overhead
    fits_vram = total_gb <= vram if vram else False
    fits_ram = total_gb <= (vram + ram) if (vram or ram) else False

    if fits_vram:
        reasons.append(f"fits in VRAM: {total_gb:.1f} GB <= {vram:.1f} GB")
        fit_class = FitClass.GPU_RESIDENT
    elif fits_ram and not request.vram_only:
        reasons.append(
            f"hybrid: {total_gb:.1f} GB needs {total_gb - vram:.1f} GB of system RAM"
        )
        fit_class = (
            FitClass.HYBRID_QUALITY
            if request.quality_priority or request.use_case == UseCase.REASONING
            else FitClass.HYBRID_INTERACTIVE
        )
    else:
        reasons.append(
            f"does not fit: {total_gb:.1f} GB needed "
            f"(VRAM {vram:.1f} GB + RAM {ram:.1f} GB)"
        )
        return FitClass.NOT_RECOMMENDED, reasons, None

    # --- Speed estimate ---
    expected = estimate_tps(entry, request)
    if expected is not None and expected < request.min_tps:
        reasons.append(
            f"estimated {expected:.1f} tok/s is below the required {request.min_tps:.1f} tok/s"
        )
        if fit_class is FitClass.GPU_RESIDENT:
            fit_class = FitClass.HYBRID_QUALITY
    elif expected is not None:
        reasons.append(f"estimated {expected:.1f} tok/s")
    return fit_class, reasons, expected


#: Real-world efficiency of the bandwidth bound, measured locally on this
#: class of hardware (RTX 3060 Laptop + llama.cpp + partial offload):
#: a fully GPU-resident model reaches ~45% of the theoretical bandwidth,
#: a hybrid model only ~20% (PCIe transfer + layer synchronisation).
GPU_RESIDENT_EFFICIENCY = 0.45
HYBRID_EFFICIENCY = 0.20
#: Effective system RAM bandwidth (GB/s): one DDR4 channel ~25 GB/s peak,
#: two channels ~45 GB/s.
RAM_BW_SINGLE_CHANNEL = 25.0
RAM_BW_DUAL_CHANNEL = 45.0


def estimate_tps(
    entry: ModelCatalogEntry, request: RecommenderRequest
) -> float | None:
    """Rough decode TPS estimate.

    MoE models decode at roughly the speed of their ACTIVE parameters (plus
    expert-routing overhead); dense models at the speed of the full weight
    set over the available bandwidth. The estimate is a PLANNING aid, never
    a substitute for a local measurement - local evidence always wins.
    """
    hw = request.hardware
    if not hw.gpu_bandwidth_gbps or not hw.vram_gb:
        return None

    effective_b = entry.active_params_b if entry.is_moe else entry.total_params_b
    # Bits actually touched per token (weights are read every token).
    bytes_per_token = effective_b * 1e9 * (entry.bits_per_weight / 8.0)
    if entry.is_moe:
        bytes_per_token *= 1.4  # expert routing + gather overhead

    weights_gb = entry.file_size_gb
    kv_gb = (
        _KV_GB_PER_TOKEN_PER_B
        * entry.total_params_b
        * request.min_context
        * request.kv_size_factor
    )
    total_gb = weights_gb + kv_gb + 0.4

    if total_gb <= hw.vram_gb:
        # Fully resident: VRAM bandwidth bound.
        bandwidth_gbs = hw.gpu_bandwidth_gbps * GPU_RESIDENT_EFFICIENCY
    else:
        # Hybrid: the spilled fraction of the weights comes from system RAM.
        spilled = max(0.0, total_gb - hw.vram_gb)
        spill_fraction = min(1.0, spilled / total_gb)
        ram_bw = RAM_BW_DUAL_CHANNEL if hw.ram_channels >= 2 else RAM_BW_SINGLE_CHANNEL
        bandwidth_gbs = (
            (1.0 - spill_fraction) * hw.gpu_bandwidth_gbps
            + spill_fraction * ram_bw
        ) * HYBRID_EFFICIENCY

    if bandwidth_gbs <= 0:
        return None
    return bandwidth_gbs * 1e9 / bytes_per_token


# --- Recommender -----------------------------------------------------------


class ModelRecommender:
    """Ranks catalog entries for a request.

    Local measurements (a `local_results` map of filename -> measured TPS)
    always outrank community priors, per spec 16.4.
    """

    def __init__(
        self,
        catalog: list[ModelCatalogEntry] | None = None,
        community: list[CommunityBenchmark] | None = None,
    ) -> None:
        self.catalog = list(catalog or [])
        if community is None:
            # Default: the shipped community benchmark database.
            try:
                from lm_optimizer.services.community_benchmarks import (
                    COMMUNITY_BENCHMARKS,
                )

                self.community = list(COMMUNITY_BENCHMARKS)
            except Exception:  # noqa: BLE001 - the database is optional
                self.community = []
        else:
            self.community = list(community)
        #: filename -> measured decode TPS on THIS machine.
        self.local_results: dict[str, float] = {}
        #: filename -> run ids that produced the local measurement.
        self.local_sources: dict[str, list[str]] = {}
        #: filename -> context the local measurement was taken at.
        self.local_contexts: dict[str, int] = {}

    def add_local_result(
        self,
        filename: str,
        decode_tps: float,
        run_ids: list[str] | None = None,
        context_length: int | None = None,
    ) -> None:
        """Register a local measurement (highest priority evidence)."""
        self.local_results[filename] = decode_tps
        if run_ids:
            self.local_sources[filename] = list(run_ids)
        if context_length:
            self.local_contexts[filename] = int(context_length)

    def add_local_measurement(
        self,
        model_id: str,
        decode_tps: float,
        run_ids: list[str] | None = None,
        context_length: int | None = None,
    ) -> None:
        """Register a local measurement addressed by LM Studio model id.

        Every catalog entry pointing at the same model id inherits it.
        """
        matched = False
        for entry in self.catalog:
            if entry.model_id and entry.model_id == model_id:
                self.add_local_result(entry.filename, decode_tps, run_ids, context_length)
                matched = True
        if not matched:
            # Unknown catalog entry - keep the evidence under the raw id so
            # it is still visible in the report.
            self.add_local_result(model_id, decode_tps, run_ids, context_length)

    def community_prior(
        self, entry: ModelCatalogEntry, request: RecommenderRequest
    ) -> tuple[float | None, Confidence, str]:
        """Best community prior for an entry: (decode_tps, confidence, source)."""
        best: tuple[float | None, Confidence, str] = (None, Confidence.D, "")
        for bench in self.community:
            if bench.model_filename != entry.filename:
                continue
            if bench.quant != entry.quant:
                continue
            sim = similarity(bench.hardware, request.hardware, spills_to_ram=True)
            if sim < 0.5:
                continue
            prior = _CONFIDENCE_PRIOR.get(bench.confidence.value, 0.2) * sim
            if best[0] is None or prior > _CONFIDENCE_PRIOR.get(best[1].value, 0.2):
                best = (bench.decode_tps, bench.confidence, bench.source)
        return best

    def recommend(self, request: RecommenderRequest) -> list[Recommendation]:
        """Rank every catalog entry for the request."""
        out: list[Recommendation] = []
        for entry in self.catalog:
            fit_class, reasons, expected = predict_fit(entry, request)

            local_tps = self.local_results.get(entry.filename)
            community_tps, community_conf, community_src = self.community_prior(
                entry, request
            )
            local_evidence = local_tps is not None
            measured_ctx = self.local_contexts.get(entry.filename)
            if local_evidence:
                expected = local_tps
                confidence = Confidence.A
                ctx_note = f" @ctx {measured_ctx}" if measured_ctx else ""
                reasons.append(
                    f"local measurement: {local_tps:.2f} tok/s{ctx_note} (highest priority)"
                )
            elif community_tps is not None:
                expected = community_tps
                confidence = community_conf
                reasons.append(
                    f"community prior ({community_conf.value}, {community_src or 'unknown source'}): "
                    f"{community_tps:.1f} tok/s"
                )
            else:
                confidence = entry.confidence

            out.append(
                Recommendation(
                    entry=entry,
                    fit_class=fit_class,
                    expected_tps=expected,
                    confidence=confidence,
                    reasons=reasons,
                    local_evidence=local_evidence,
                    source_run_ids=self.local_sources.get(entry.filename, []),
                    measured_context=measured_ctx,
                )
            )

        out.sort(
            key=lambda r: (
                r.fit_class is not FitClass.NOT_RECOMMENDED,
                r.expected_tps or 0.0,
            ),
            reverse=True,
        )
        return out

    def to_json(self, recommendations: list[Recommendation]) -> str:
        return json.dumps([r.as_dict() for r in recommendations], indent=2, ensure_ascii=False)
