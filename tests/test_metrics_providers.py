"""Cross-platform metrics provider tests (spec §17)."""
import pytest

from lm_optimizer.services.metrics_providers import (
    GpuMetrics,
    LinuxMetricsProvider,
    MacOSMetricsProvider,
    MemoryMetrics,
    PowerMetrics,
    SystemMetrics,
    WindowsMetricsProvider,
    _float_or_none,
    _parse_nvidia_smi,
    collect_metrics,
    get_provider,
)


def test_provider_selection_by_platform():
    assert isinstance(get_provider("Windows"), WindowsMetricsProvider)
    assert isinstance(get_provider("Linux"), LinuxMetricsProvider)
    assert isinstance(get_provider("Darwin"), MacOSMetricsProvider)
    assert isinstance(get_provider("macOS"), MacOSMetricsProvider)
    # Unknown platforms fall back to the Linux provider (no crash).
    assert isinstance(get_provider("FreeBSD"), LinuxMetricsProvider)


def test_parse_nvidia_smi_real_line():
    output = "NVIDIA GeForce RTX 3060 Laptop GPU, 5900, 6144, 68, 45, 85.3"
    gpus = _parse_nvidia_smi(output)
    assert len(gpus) == 1
    gpu = gpus[0]
    assert gpu.name.startswith("NVIDIA")
    assert gpu.vram_free_mb == 5900
    assert gpu.vram_total_mb == 6144
    assert gpu.temperature_c == 68
    assert gpu.utilization_pct == 45
    assert gpu.power_w == pytest.approx(85.3)


def test_parse_nvidia_smi_handles_units_and_na():
    output = "GPU A, 1024 MiB, 8192 MiB, [N/A], 10 %, [Not Supported]"
    gpus = _parse_nvidia_smi(output)
    assert len(gpus) == 1
    gpu = gpus[0]
    assert gpu.vram_free_mb == 1024
    assert gpu.vram_total_mb == 8192
    assert gpu.temperature_c is None
    assert gpu.utilization_pct == 10
    assert gpu.power_w is None


def test_parse_nvidia_smi_empty():
    assert _parse_nvidia_smi(None) == []
    assert _parse_nvidia_smi("") == []
    assert _parse_nvidia_smi("garbage") == []


def test_float_or_none():
    assert _float_or_none("123") == 123
    assert _float_or_none("85.3 W") == pytest.approx(85.3)
    assert _float_or_none("[N/A]") is None
    assert _float_or_none("") is None
    assert _float_or_none(None) is None


def test_macos_never_pretends_vram():
    metrics = MacOSMetricsProvider().collect()
    assert metrics.gpus == []
    assert any("unified memory" in u for u in metrics.unavailable)


def test_system_metrics_serializes_with_none():
    metrics = SystemMetrics(
        gpus=[GpuMetrics(name="x", temperature_c=None)],
        memory=MemoryMetrics(ram_total_gb=32.0),
        power=PowerMetrics(on_ac=None),
        os_name="Test",
        unavailable=["gpu_temperature"],
    )
    payload = metrics.as_dict()
    assert payload["gpus"][0]["temperature_c"] is None
    assert payload["memory"]["ram_total_gb"] == 32.0
    assert payload["power"]["on_ac"] is None
    assert "gpu_temperature" in payload["unavailable"]


def test_collect_metrics_never_raises():
    metrics = collect_metrics()
    assert isinstance(metrics, SystemMetrics)
    assert metrics.os_name


def test_windows_provider_reports_unavailable_gpu_gracefully(monkeypatch):
    """With no nvidia-smi the provider reports unavailable, not zeros."""
    import lm_optimizer.services.metrics_providers as mod

    monkeypatch.setattr(mod, "_run", lambda *a, **k: None)
    metrics = WindowsMetricsProvider().collect()
    assert metrics.gpus == []
    assert any("gpu" in u for u in metrics.unavailable)
    # RAM still comes from psutil and must be positive on a real host.
    assert metrics.memory.ram_total_gb is None or metrics.memory.ram_total_gb > 0
