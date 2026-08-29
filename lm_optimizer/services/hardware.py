"""Hardware detection service."""

import platform
import subprocess

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

from lm_optimizer.config import config
from lm_optimizer.domain.models import GPUInfo, HardwareInfo
from lm_optimizer.logging_config import get_logger

logger = get_logger(__name__)


class HardwareDetector:
    """Detects and caches hardware information."""

    def __init__(self):
        self._cached_hardware: HardwareInfo | None = None

    def detect(self, force_refresh: bool = False) -> HardwareInfo:
        """Detect hardware information."""
        if self._cached_hardware and not force_refresh:
            return self._cached_hardware

        # CPU
        cpu_name = self._get_cpu_name()
        cpu_cores_physical = psutil.cpu_count(logical=False) or 1
        cpu_cores_logical = psutil.cpu_count(logical=True) or 1

        # Memory
        mem = psutil.virtual_memory()
        total_ram_gb = round(mem.total / (1024**3), 2)

        # GPUs
        gpus = self._detect_gpus()
        gpu_count = len(gpus)

        # CUDA
        cuda_version = self._get_cuda_version()

        # Metal (macOS)
        metal_available = platform.system() == "Darwin"

        # Vulkan
        vulkan_available = self._check_vulkan()

        hardware = HardwareInfo(
            os=platform.system(),
            cpu_name=cpu_name,
            cpu_cores_physical=cpu_cores_physical,
            cpu_cores_logical=cpu_cores_logical,
            total_ram_gb=total_ram_gb,
            gpu_count=gpu_count,
            gpus=gpus,
            cuda_version=cuda_version,
            metal_available=metal_available,
            vulkan_available=vulkan_available,
        )

        # Apply config overrides
        hw_config = config.hardware
        if hw_config.gpu_vram_gb is not None and hardware.gpus:
            hardware.gpus[0].vram_gb = hw_config.gpu_vram_gb
        if hw_config.system_ram_gb is not None:
            hardware.total_ram_gb = hw_config.system_ram_gb
        if hw_config.gpu_name is not None and hardware.gpus:
            hardware.gpus[0].name = hw_config.gpu_name

        self._cached_hardware = hardware
        logger.info("Hardware detected", gpu_count=gpu_count, ram_gb=total_ram_gb)
        return hardware

    def _get_cpu_name(self) -> str:
        """Get CPU name."""
        cpu_name = platform.processor()
        if not cpu_name or cpu_name == "":
            if platform.system() == "Windows" and WMI_AVAILABLE:
                try:
                    c = wmi.WMI()
                    for processor in c.Win32_Processor():
                        cpu_name = processor.Name
                        break
                except Exception:
                    pass
        if not cpu_name:
            cpu_name = platform.machine()
        return cpu_name

    def _detect_gpus(self) -> list[GPUInfo]:
        """Detect GPU information."""
        gpus = []

        if GPUTIL_AVAILABLE:
            try:
                gpu_list = GPUtil.getGPUs()
                for gpu in gpu_list:
                    gpus.append(
                        GPUInfo(
                            index=gpu.id,
                            name=gpu.name,
                            vram_gb=round(gpu.memoryTotal / 1024, 2),
                            vendor="NVIDIA",
                            driver_version=None,
                            compute_capability=None,
                        )
                    )
            except Exception as e:
                logger.debug("GPUtil detection failed", error=str(e))

        # Try nvidia-smi for more details
        if platform.system() == "Windows" or platform.system() == "Linux":
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,name,memory.total,driver_version,compute_cap",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 5:
                            idx = int(parts[0])
                            # Update existing or add new
                            existing = next((g for g in gpus if g.index == idx), None)
                            if existing:
                                existing.driver_version = parts[3]
                                existing.compute_capability = parts[4]
                            else:
                                gpus.append(
                                    GPUInfo(
                                        index=idx,
                                        name=parts[1],
                                        vram_gb=round(float(parts[2]) / 1024, 2),
                                        vendor="NVIDIA",
                                        driver_version=parts[3],
                                        compute_capability=parts[4],
                                    )
                                )
            except (subprocess.SubprocessError, FileNotFoundError, ValueError):
                pass

        # AMD GPU detection on Linux
        if platform.system() == "Linux":
            try:
                result = subprocess.run(
                    ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    import json

                    data = json.loads(result.stdout)
                    for idx, gpu_data in enumerate(data):
                        gpus.append(
                            GPUInfo(
                                index=idx,
                                name=gpu_data.get("Card series", "AMD GPU"),
                                vram_gb=gpu_data.get("VRAM Total Memory (B)", 0) / (1024**3),
                                vendor="AMD",
                            )
                        )
            except Exception:
                pass

        return gpus

    def _get_cuda_version(self) -> str | None:
        """Get CUDA version."""
        try:
            result = subprocess.run(
                ["nvcc", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "release" in line.lower():
                        return line.split("release")[-1].split(",")[0].strip()
        except Exception:
            pass
        return None

    def _check_vulkan(self) -> bool:
        """Check Vulkan availability."""
        try:
            result = subprocess.run(
                ["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def get_available_vram(self) -> float:
        """Get total available VRAM in GB."""
        hardware = self.detect()
        return sum(g.vram_gb for g in hardware.gpus)

    def get_primary_gpu(self) -> GPUInfo | None:
        """Get primary GPU (first one)."""
        hardware = self.detect()
        return hardware.gpus[0] if hardware.gpus else None


# Global hardware detector
hardware_detector = HardwareDetector()
