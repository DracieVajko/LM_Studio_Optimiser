"""Cross-platform metrics providers (spec §17).

One interface, platform adapters. Missing values are `None`/unavailable -
never fabricated zeros (spec §17.5, acceptance criterion 16).

Providers:
- Windows: nvidia-smi for GPU memory/temperature, psutil for RAM/battery.
- Linux: nvidia-smi (NVIDIA) or rocm-smi (AMD), /proc/meminfo via psutil.
- macOS: unified memory means there is no separate VRAM; GPU memory is
  reported as unavailable instead of pretending.

The thermal/power fields are best effort: on a laptop they decide whether a
benchmark was thermally contaminated (spec §11.4).
"""

from __future__ import annotations

import platform
import re
import subprocess
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class GpuMetrics:
    """GPU memory and thermal readings; None means unavailable."""

    name: str = ""
    vram_free_mb: float | None = None
    vram_total_mb: float | None = None
    temperature_c: float | None = None
    utilization_pct: float | None = None
    power_w: float | None = None


@dataclass
class MemoryMetrics:
    """System RAM readings."""

    ram_free_gb: float | None = None
    ram_total_gb: float | None = None
    swap_free_gb: float | None = None
    swap_total_gb: float | None = None


@dataclass
class PowerMetrics:
    """Power/battery state (§11.4, §17.2)."""

    on_ac: bool | None = None
    battery_pct: float | None = None
    power_plan: str | None = None


@dataclass
class SystemMetrics:
    """Full snapshot."""

    gpus: list[GpuMetrics] = field(default_factory=list)
    memory: MemoryMetrics = field(default_factory=MemoryMetrics)
    power: PowerMetrics = field(default_factory=PowerMetrics)
    os_name: str = ""
    unavailable: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "gpus": [g.__dict__ for g in self.gpus],
            "memory": self.memory.__dict__,
            "power": self.power.__dict__,
            "os_name": self.os_name,
            "unavailable": list(self.unavailable),
        }


class MetricsProvider(Protocol):
    """Platform adapter interface."""

    def collect(self) -> SystemMetrics:
        ...


# --- shared helpers ---------------------------------------------------------


def _run(cmd: list[str], timeout: float = 8.0) -> str | None:
    """Run a probe command; None when unavailable (never raises)."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0:
            return None
        return proc.stdout
    except Exception:  # noqa: BLE001
        return None


def _parse_nvidia_smi(output: str | None) -> list[GpuMetrics]:
    """Parse `nvidia-smi --query-gpu=...` CSV output."""
    if not output:
        return []
    gpus: list[GpuMetrics] = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        name, free, total, temp, util, power = parts[:6]
        gpus.append(
            GpuMetrics(
                name=name,
                vram_free_mb=_float_or_none(free),
                vram_total_mb=_float_or_none(total),
                temperature_c=_float_or_none(temp),
                utilization_pct=_float_or_none(util),
                power_w=_float_or_none(power),
            )
        )
    return gpus


def _float_or_none(value: str | None) -> float | None:
    """Parse a number, tolerating units and "[N/A]"."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or "N/A" in text.upper() or text.lower() in ("not supported", "unknown"):
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


_NVIDIA_QUERY = [
    "nvidia-smi",
    "--query-gpu=name,memory.free,memory.total,temperature.gpu,utilization.gpu,power.draw",
    "--format=csv,noheader,nounits",
]


def _ram_metrics() -> MemoryMetrics:
    try:
        import psutil  # type: ignore

        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return MemoryMetrics(
            ram_free_gb=round(mem.available / 1024**3, 2),
            ram_total_gb=round(mem.total / 1024**3, 2),
            swap_free_gb=round(swap.free / 1024**3, 2),
            swap_total_gb=round(swap.total / 1024**3, 2),
        )
    except Exception:  # noqa: BLE001
        return MemoryMetrics()


def _power_metrics() -> PowerMetrics:
    try:
        import psutil  # type: ignore

        battery = psutil.sensors_battery()
        if battery is None:
            return PowerMetrics(on_ac=True) if platform.system() == "Windows" else PowerMetrics()
        return PowerMetrics(on_ac=bool(battery.power_plugged), battery_pct=float(battery.percent))
    except Exception:  # noqa: BLE001
        return PowerMetrics()


# --- providers --------------------------------------------------------------


class WindowsMetricsProvider:
    """Windows: nvidia-smi + psutil."""

    def collect(self) -> SystemMetrics:
        gpus = _parse_nvidia_smi(_run(_NVIDIA_QUERY))
        unavailable: list[str] = []
        if not gpus:
            unavailable.append("gpu (nvidia-smi unavailable or no NVIDIA GPU)")
        if not any(g.temperature_c is not None for g in gpus):
            unavailable.append("gpu_temperature")
        return SystemMetrics(
            gpus=gpus,
            memory=_ram_metrics(),
            power=_power_metrics(),
            os_name="Windows",
            unavailable=unavailable,
        )


class LinuxMetricsProvider:
    """Linux: nvidia-smi (NVIDIA) or rocm-smi (AMD)."""

    def collect(self) -> SystemMetrics:
        gpus = _parse_nvidia_smi(_run(_NVIDIA_QUERY))
        unavailable: list[str] = []
        if not gpus:
            rocm = _run(["rocm-smi", "--showmeminfo", "vram", "--json"])
            if rocm:
                unavailable.append("gpu (rocm-smi present; parsing not implemented)")
            else:
                unavailable.append("gpu (no nvidia-smi or rocm-smi)")
        if not any(g.temperature_c is not None for g in gpus):
            unavailable.append("gpu_temperature")
        return SystemMetrics(
            gpus=gpus,
            memory=_ram_metrics(),
            power=_power_metrics(),
            os_name="Linux",
            unavailable=unavailable,
        )


class MacOSMetricsProvider:
    """macOS: unified memory (no separate VRAM)."""

    def collect(self) -> SystemMetrics:
        return SystemMetrics(
            gpus=[],
            memory=_ram_metrics(),
            power=_power_metrics(),
            os_name="macOS",
            unavailable=[
                "gpu_vram (unified memory: no separate VRAM on Apple Silicon)",
                "gpu_temperature",
            ],
        )


def get_provider(system: str | None = None) -> MetricsProvider:
    """Pick the provider for the current platform."""
    name = (system or platform.system()).lower()
    if name.startswith("win"):
        return WindowsMetricsProvider()
    if name == "darwin" or name == "macos":
        return MacOSMetricsProvider()
    return LinuxMetricsProvider()


def collect_metrics(system: str | None = None) -> SystemMetrics:
    """Convenience wrapper used by the benchmark service."""
    return get_provider(system).collect()
