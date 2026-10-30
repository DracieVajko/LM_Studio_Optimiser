"""Community benchmark database (spec 16.4).

Records measured results from OTHER people on similar hardware. Every entry
carries a confidence grade:

- A: raw JSON + exact hash + full settings
- B: complete command/config + benchmark output
- C: hardware/model/quant/context/TPS
- D: community claim with missing settings

Local measurements always outrank these priors (spec 16.4: "local results
have priority over community priors"). The similarity score decides how
much a record is trusted for the current machine (spec 16.5).

Sources (retrieved 2026-10-21):
- batiai/qwen3.8-27b ollama tags (M4 Max 128 GB, Metal, ctx 16384, warm)
- batiai/qwen3.6-27b ollama tags (M4 Max 128 GB, Metal)
- atomic.chat Qwen 3.8 27B guide (4x RTX 5090, eval_neutral held-out)
- Qwen/Qwen3.8-27B discussion #65 (d4nnyy, 4x RTX 5090, same harness)
- specpicks RTX 3060 12 GB community runs
- llama.cpp benchmark repo (RTX 3060 12 GB, CUDA)
- kvmem/llama.cpp v0.16.0-rc1 (RTX 5060 Ti 16 GB, MTP3 256K)
- georuffus/Qwen3.8-27B-i1-IQ4_KS_KT (16 GB target, q4_0 KV cache)
- vmarcelo/Swift-Qwen3.8-27B-MIX (RX 9070 XT, Vulkan, 32K ctx, MTP on)
- Sudo su benchmark book (public X post, 6/8/12/16/24 GB class)
"""

from __future__ import annotations

from lm_optimizer.services.recommender import (
    CommunityBenchmark,
    Confidence,
    HardwareFingerprint,
)

_M4_MAX = HardwareFingerprint(
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

_M4_MINI = HardwareFingerprint(
    gpu_name="Apple M4 mini",
    vram_gb=24.0,
    gpu_bandwidth_gbps=150.0,
    cpu_name="Apple M4 mini",
    physical_cores=10,
    logical_threads=10,
    ram_gb=24.0,
    ram_channels=1,
    disk_type="NVMe SSD",
    os_name="macOS",
)

_RTX_3060_12 = HardwareFingerprint(
    gpu_name="NVIDIA GeForce RTX 3060",
    vram_gb=12.0,
    gpu_bandwidth_gbps=360.0,
    gpu_tdp_w=170.0,
    cpu_name="unknown",
    physical_cores=0,
    logical_threads=0,
    ram_gb=32.0,
    ram_channels=1,
    disk_type="SSD",
    os_name="Windows",
)

_RTX_5060_TI = HardwareFingerprint(
    gpu_name="NVIDIA GeForce RTX 5060 Ti",
    vram_gb=16.0,
    gpu_bandwidth_gbps=448.0,
    cpu_name="unknown",
    physical_cores=0,
    logical_threads=0,
    ram_gb=32.0,
    ram_channels=2,
    disk_type="NVMe SSD",
    os_name="Linux",
)

_RTX_5090 = HardwareFingerprint(
    gpu_name="NVIDIA GeForce RTX 5090",
    vram_gb=32.0,
    gpu_bandwidth_gbps=1792.0,
    gpu_tdp_w=575.0,
    cpu_name="unknown",
    physical_cores=0,
    logical_threads=0,
    ram_gb=128.0,
    ram_channels=2,
    disk_type="NVMe SSD",
    os_name="Linux",
)

_RX_9070_XT = HardwareFingerprint(
    gpu_name="AMD Radeon RX 9070 XT",
    vram_gb=16.0,
    gpu_bandwidth_gbps=645.0,
    cpu_name="unknown",
    physical_cores=0,
    logical_threads=0,
    ram_gb=32.0,
    ram_channels=2,
    disk_type="NVMe SSD",
    os_name="Linux",
)

_V100 = HardwareFingerprint(
    gpu_name="NVIDIA Tesla V100",
    vram_gb=32.0,
    gpu_bandwidth_gbps=900.0,
    cpu_name="unknown",
    physical_cores=0,
    logical_threads=0,
    ram_gb=64.0,
    ram_channels=2,
    disk_type="SSD",
    os_name="Linux",
)


def _bench(
    hardware: HardwareFingerprint,
    repo: str,
    filename: str,
    quant: str,
    context: int,
    decode: float,
    prompt: float,
    source: str,
    confidence: Confidence,
    gpu_offload: float = 1.0,
    mtp: bool = False,
    kv_quant: str = "f16",
    threads: int = 0,
) -> CommunityBenchmark:
    return CommunityBenchmark(
        hardware=hardware,
        model_repository=repo,
        model_filename=filename,
        quant=quant,
        context_length=context,
        gpu_offload=gpu_offload,
        flash_attention=True,
        kv_on_gpu=True,
        kv_quant=kv_quant,
        eval_batch_size=2048,
        threads=threads,
        mmap=True,
        mtp=mtp,
        prompt_tps=prompt,
        decode_tps=decode,
        source=source,
        confidence=confidence,
    )


#: The community benchmark database.
COMMUNITY_BENCHMARKS: list[CommunityBenchmark] = [
    # --- batiai: Qwen3.8-27B quant ladder (M4 Max, Metal, ctx 16384) ---
    _bench(_M4_MAX, "batiai/qwen3.8-27b", "qwen3.8-27b:q2", "Q2_K_S", 16384,
           17.3, 65.1, "batiai ollama tag q2", Confidence.B),
    _bench(_M4_MAX, "batiai/qwen3.8-27b", "qwen3.8-27b:iq3", "IQ3_XXS", 16384,
           14.5, 63.4, "batiai ollama tag iq3", Confidence.B),
    _bench(_M4_MAX, "batiai/qwen3.8-27b", "qwen3.8-27b:q3", "Q3_K_M", 16384,
           13.0, 63.0, "batiai ollama tag q3 (slowest!)", Confidence.B),
    _bench(_M4_MAX, "batiai/qwen3.8-27b", "qwen3.8-27b:iq4", "IQ4_XS", 16384,
           15.5, 60.0, "batiai ollama tag iq4 (default pick)", Confidence.B),
    _bench(_M4_MAX, "batiai/qwen3.8-27b", "qwen3.8-27b:q4", "Q4_K_M", 16384,
           15.9, 60.0, "batiai ollama tag q4", Confidence.B),
    _bench(_M4_MAX, "batiai/qwen3.8-27b", "qwen3.8-27b:q6", "Q6_K", 16384,
           11.5, 55.0, "batiai ollama tag q6", Confidence.B),
    # --- batiai: Qwen3.6-27B (M4 Max) ---
    _bench(_M4_MAX, "batiai/qwen3.6-27b", "qwen3.6-27b:iq3", "IQ3_XXS", 8192,
           17.83, 108.7, "batiai ollama tag iq3", Confidence.B),
    _bench(_M4_MAX, "batiai/qwen3.6-27b", "qwen3.6-27b:q3", "Q3_K_M", 8192,
           15.30, 111.7, "batiai ollama tag q3", Confidence.B),
    _bench(_M4_MAX, "batiai/qwen3.6-27b", "qwen3.6-27b:q4", "Q4_K_M", 8192,
           16.56, 114.5, "batiai ollama tag q4 (fastest on Metal)", Confidence.B),
    # --- RTX 3060 12 GB class (closest desktop sibling of the 6 GB laptop) ---
    _bench(_RTX_3060_12, "bartowski/Qwen3.6-27B-GGUF", "Qwen3.6-27B-Q3_K_M.gguf",
           "Q3_K_M", 8192, 7.5, 300.0,
           "specpicks: q3_K_M is the only all-on-GPU quant on a 3060 12 GB",
           Confidence.C, gpu_offload=1.0),
    _bench(_RTX_3060_12, "bartowski/Qwen3.6-27B-GGUF", "Qwen3.6-27B-Q4_K_M.gguf",
           "Q4_K_M", 8192, 4.5, 300.0,
           "specpicks: q4_K_M with 6-10 layers offloaded to CPU", Confidence.C,
           gpu_offload=0.7),
    # --- 4x RTX 5090 evaluation harness (same corpus for all quants) ---
    _bench(_RTX_5090, "Qwen/Qwen3.8-27B (official quant)", "AD-IQ3_S.gguf",
           "IQ3_S", 4096, 59.5, 0.0,
           "Qwen discussion #65: AD-IQ3_S best size/quality at 13.8 GB",
           Confidence.B),
    _bench(_RTX_5090, "Qwen/Qwen3.8-27B (official quant)", "AD-IQ4_XS.gguf",
           "IQ4_XS", 4096, 55.0, 0.0,
           "Qwen discussion #65: AD-IQ4_XS", Confidence.B),
    _bench(_RTX_5090, "Qwen/Qwen3.8-27B (official quant)", "AD-Q4_K_M.gguf",
           "Q4_K_M", 4096, 41.0, 0.0,
           "Qwen discussion #65 / benchmark book: Q4_K_M baseline", Confidence.B),
    # --- MTP evidence (spec: MTP must not be assumed a win) ---
    _bench(_V100, "Qwen3.6-27B", "Qwen3.6-27B-Q4_K_M.gguf", "Q4_K_M", 8192,
           31.2, 0.0, "V100 32 GB, dense 27B, MTP OFF", Confidence.C),
    _bench(_V100, "Qwen3.6-27B", "Qwen3.6-27B-Q4_K_M-mtp.gguf", "Q4_K_M", 8192,
           40.3, 0.0, "V100 32 GB, MTP n=3 (~73% acceptance)", Confidence.C,
           mtp=True),
    _bench(_RTX_5060_TI, "Qwen3.6-35B-A3B", "Qwen3.6-35B-A3B-Q4_K_M.gguf",
           "Q4_K_M", 8192, 25.5, 100.0,
           "RTX 3060 12 GB: Q4_K_M 24-25 tok/s, MoE experts on CPU", Confidence.C,
           gpu_offload=0.99, mtp=False),
    _bench(_RTX_5060_TI, "Qwen3.6-35B-A3B", "Qwen3.6-35B-A3B-Q4_K_M-mtp.gguf",
           "Q4_K_M", 8192, 16.9, 100.0,
           "RTX 3060 12 GB: MTP n=3 SLOWED the MoE model to 16.9 tok/s",
           Confidence.C, gpu_offload=0.99, mtp=True),
    # --- Swift MIX quants (RX 9070 XT, Vulkan, 32K ctx, MTP on) ---
    _bench(_RX_9070_XT, "vmarcelo/Swift-Qwen3.8-27B-MIX", "Swift-Qwen3.8-27B-IQ3-MIX.gguf",
           "IQ3_MIX", 32768, 24.0, 0.0,
           "vmarcelo measured MIX quants on RX 9070 XT with MTP on", Confidence.C,
           mtp=True, kv_quant="q8_0"),
    _bench(_RX_9070_XT, "vmarcelo/Swift-Qwen3.8-27B-MIX", "Swift-Qwen3.8-27B-IQ4-MIX.gguf",
           "IQ4_MIX", 32768, 22.0, 0.0,
           "vmarcelo measured MIX quants on RX 9070 XT with MTP on", Confidence.C,
           mtp=True, kv_quant="q8_0"),
    # --- 6 GB class community claims (lowest confidence) ---
    _bench(HardwareFingerprint(
        gpu_name="NVIDIA GeForce RTX 3060 Laptop GPU", vram_gb=6.0,
        gpu_bandwidth_gbps=192.0, cpu_name="unknown", ram_gb=32.0, ram_channels=1,
        disk_type="SSD", os_name="Windows"),
        "lmstudio-community/bonsai-27b", "bonsai-27b-q1_0.gguf", "Q1_0", 8192,
        20.5, 0.0, "benchmark book: 27B Q1_0 on a 6 GB card", Confidence.D),
]
