"""Hardware detection - authoritative implementation for GPU, CPU, memory."""

import platform
import subprocess
from dataclasses import dataclass

import psutil

try:
    import GPUtil

    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

try:
    import wmi

    WMI_AVAILABLE = True
except ImportError:
    WMI_AVAILABLE = False

from ..config import config
from ..logging_config import get_logger

logger = get_logger(__name__)


# Legacy local dataclasses kept for backward compat (deprecated)
@dataclass
class GPUInfo:
    """GPU information (legacy, use domain.GPUInfo)."""

    name: str
    vram_gb: float
    driver_version: str | None = None
    cuda_version: str | None = None
    compute_capability: str | None = None
    gpu_index: int = 0


@dataclass
class CPUInfo:
    """CPU information (legacy)."""

    name: str
    cores_physical: int
    cores_logical: int
    frequency_ghz: float | None = None


@dataclass
class MemoryInfo:
    """System memory information (legacy)."""

    total_gb: float
    available_gb: float
    used_gb: float


@dataclass
class HardwareInfo:
    """Complete hardware information (legacy, use domain.HardwareInfo)."""

    gpu: GPUInfo | None
    cpu: CPUInfo
    memory: MemoryInfo
    os: str
    platform: str
    python_version: str


def get_gpu_info() -> GPUInfo | None:
    """Detect GPU information using GPUtil and nvidia-smi."""
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            logger.warning("No GPUs detected by GPUtil")
            return None

        # Use first GPU by default
        gpu = gpus[0]

        # Try to get more details from nvidia-smi
        driver_version = None
        cuda_version = None
        compute_capability = None

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version,cuda_version,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                if len(parts) >= 3:
                    driver_version = parts[0]
                    cuda_version = parts[1]
                    compute_capability = parts[2]
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.debug("nvidia-smi not available or failed")

        return GPUInfo(
            name=gpu.name,
            vram_gb=round(gpu.memoryTotal / 1024, 2),  # GPUtil reports in MB
            driver_version=driver_version,
            cuda_version=cuda_version,
            compute_capability=compute_capability,
            gpu_index=gpu.id,
        )
    except Exception as e:
        logger.warning("Failed to detect GPU info", error=str(e))
        return None


def get_cpu_info() -> CPUInfo:
    """Detect CPU information."""
    cpu_name = platform.processor()
    if not cpu_name or cpu_name == "":
        try:
            # Try to get more detailed CPU name on Windows
            if platform.system() == "Windows":
                import wmi

                c = wmi.WMI()
                for processor in c.Win32_Processor():
                    cpu_name = processor.Name
                    break
        except Exception:
            cpu_name = "Unknown"

    return CPUInfo(
        name=cpu_name,
        cores_physical=psutil.cpu_count(logical=False) or 1,
        cores_logical=psutil.cpu_count(logical=True) or 1,
        frequency_ghz=round(psutil.cpu_freq().max / 1000, 2) if psutil.cpu_freq() else None,
    )


def get_memory_info() -> MemoryInfo:
    """Detect system memory information."""
    mem = psutil.virtual_memory()
    return MemoryInfo(
        total_gb=round(mem.total / (1024**3), 2),
        available_gb=round(mem.available / (1024**3), 2),
        used_gb=round(mem.used / (1024**3), 2),
    )


def detect_hardware() -> HardwareInfo:
    """Detect all hardware information."""
    gpu = get_gpu_info()
    cpu = get_cpu_info()
    memory = get_memory_info()

    # Override with config values if provided
    hw_config = config.hardware
    if hw_config.gpu_vram_gb is not None and gpu:
        gpu.vram_gb = hw_config.gpu_vram_gb
    if hw_config.system_ram_gb is not None:
        memory.total_gb = hw_config.system_ram_gb
    if hw_config.gpu_name is not None and gpu:
        gpu.name = hw_config.gpu_name

    return HardwareInfo(
        gpu=gpu,
        cpu=cpu,
        memory=memory,
        os=platform.system(),
        platform=platform.platform(),
        python_version=platform.python_version(),
    )


def format_hardware_info(info: HardwareInfo) -> str:
    """Format hardware info for display."""
    lines = [
        "=== Hardware Information ===",
        f"OS: {info.os} ({info.platform})",
        f"Python: {info.python_version}",
        "",
        "CPU:",
        f"  Name: {info.cpu.name}",
        f"  Cores: {info.cpu.cores_physical} physical / {info.cpu.cores_logical} logical",
    ]
    if info.cpu.frequency_ghz:
        lines.append(f"  Max Frequency: {info.cpu.frequency_ghz:.2f} GHz")

    lines.append("")
    lines.append("Memory:")
    lines.append(f"  Total: {info.memory.total_gb:.1f} GB")
    lines.append(f"  Available: {info.memory.available_gb:.1f} GB")
    lines.append(f"  Used: {info.memory.used_gb:.1f} GB")

    if info.gpu:
        lines.append("")
        lines.append("GPU:")
        lines.append(f"  Name: {info.gpu.name}")
        lines.append(f"  VRAM: {info.gpu.vram_gb:.1f} GB")
        if info.gpu.driver_version:
            lines.append(f"  Driver: {info.gpu.driver_version}")
        if info.gpu.cuda_version:
            lines.append(f"  CUDA: {info.gpu.cuda_version}")
        if info.gpu.compute_capability:
            lines.append(f"  Compute Capability: {info.gpu.compute_capability}")
    else:
        lines.append("")
        lines.append("GPU: Not detected")

    return "\n".join(lines)
