"""Cross-platform hardware detection and automatic local model tier selection (macOS + Windows + Linux)."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Literal

Accelerator = Literal["cuda", "mps", "vulkan", "cpu"]
Tier = Literal["high", "balanced", "fast"]


@dataclass(frozen=True)
class HardwareProfile:
    os_name: str
    arch: str
    accelerator: Accelerator
    gpu_name: str
    ram_gb: float
    vram_gb: float
    tier: Tier
    painter_label: str
    sculptor_repo: str
    sculptor_subfolder: str
    sculptor_label: str
    estimated_download_gb: float

    def to_dict(self) -> dict:
        return asdict(self)


def _system_ram_gb() -> float:
    try:
        if sys.platform == "darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=3).strip()
            return round(int(out) / (1024 ** 3), 1)
        if sys.platform == "win32":
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            st = _MemStatus()
            st.dwLength = ctypes.sizeof(_MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return round(st.ullTotalPhys / (1024 ** 3), 1)
        if os.path.exists("/proc/meminfo"):
            for line in open("/proc/meminfo", encoding="utf-8"):
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 ** 2), 1)
    except Exception:  # noqa: BLE001
        pass
    return 16.0


def _macos_chip() -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=3).strip()
    except Exception:  # noqa: BLE001
        return "Apple Silicon"


def _nvidia_gpu() -> tuple[str, float] | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.check_output(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip()
        if not out:
            return None
        first = out.splitlines()[0].split(",")
        name = first[0].strip()
        vram_mb = float(first[1].strip()) if len(first) > 1 else 0.0
        return name, round(vram_mb / 1024.0, 1)
    except Exception:  # noqa: BLE001
        return None


def _windows_non_nvidia_gpu() -> tuple[str, float] | None:
    if sys.platform != "win32":
        return None
    try:
        out = subprocess.check_output(
            [
                "powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object -First 1 -ExpandProperty Name",
            ],
            text=True,
            timeout=5,
        ).strip()
        if out and any(k in out.lower() for k in ("amd", "radeon", "intel", "arc")):
            return out, 0.0
    except Exception:  # noqa: BLE001
        pass
    return None


def select_tier(os_name: str, arch: str, accelerator: Accelerator, ram_gb: float, vram_gb: float) -> HardwareProfile:
    """Pure decision function mapping hardware capabilities to the optimal local 2D + 3D model tier."""
    gpu_name = "CPU"
    if accelerator == "mps":
        gpu_name = "Apple Silicon"
    elif accelerator == "cuda":
        gpu_name = "NVIDIA GPU"
    elif accelerator == "vulkan":
        gpu_name = "Vulkan GPU"

    if (accelerator == "mps" and ram_gb >= 24.0) or (accelerator == "cuda" and vram_gb >= 12.0):
        return HardwareProfile(
            os_name=os_name,
            arch=arch,
            accelerator=accelerator,
            gpu_name=gpu_name,
            ram_gb=ram_gb,
            vram_gb=vram_gb,
            tier="high",
            painter_label="FLUX.1 Kontext (Q4 GGUF)",
            sculptor_repo="tencent/Hunyuan3D-2mv",
            sculptor_subfolder="hunyuan3d-dit-v2-mv",
            sculptor_label="Hunyuan3D 2 Multi-View (High Detail)",
            estimated_download_gb=14.5,
        )
    if (accelerator == "mps" and ram_gb >= 14.0) or (accelerator == "cuda" and vram_gb >= 6.0) or (
        accelerator == "vulkan" and ram_gb >= 16.0
    ):
        return HardwareProfile(
            os_name=os_name,
            arch=arch,
            accelerator=accelerator,
            gpu_name=gpu_name,
            ram_gb=ram_gb,
            vram_gb=vram_gb,
            tier="balanced",
            painter_label="FLUX.1 Kontext (Q4 GGUF)",
            sculptor_repo="tencent/Hunyuan3D-2mini",
            sculptor_subfolder="hunyuan3d-dit-v2-mini",
            sculptor_label="Hunyuan3D 2 Mini (Balanced)",
            estimated_download_gb=10.5,
        )
    return HardwareProfile(
        os_name=os_name,
        arch=arch,
        accelerator=accelerator,
        gpu_name=gpu_name,
        ram_gb=ram_gb,
        vram_gb=vram_gb,
        tier="fast",
        painter_label="FLUX.1 Kontext (Q4 GGUF, CPU offload)",
        sculptor_repo="tencent/Hunyuan3D-2mini",
        sculptor_subfolder="hunyuan3d-dit-v2-mini",
        sculptor_label="Hunyuan3D 2 Mini (Compact / CPU)",
        estimated_download_gb=9.5,
    )


_cached: HardwareProfile | None = None


def detect_hardware(force: bool = False) -> HardwareProfile:
    """Detect OS, GPU accelerator, VRAM, and RAM; cache for the process lifetime."""
    global _cached
    if _cached is not None and not force:
        return _cached

    os_name = sys.platform
    arch = platform.machine().lower()
    ram_gb = _system_ram_gb()

    accelerator: Accelerator = "cpu"
    gpu_name = "CPU"
    vram_gb = 0.0

    if os_name == "darwin" and arch in ("arm64", "aarch64"):
        accelerator = "mps"
        gpu_name = _macos_chip()
        vram_gb = ram_gb  # Apple Silicon unified memory
    else:
        nv = _nvidia_gpu()
        if nv is not None:
            accelerator = "cuda"
            gpu_name, vram_gb = nv
        else:
            other = _windows_non_nvidia_gpu()
            if other is not None:
                accelerator = "vulkan"
                gpu_name, vram_gb = other

    prof = select_tier(os_name, arch, accelerator, ram_gb, vram_gb)
    _cached = HardwareProfile(**{**prof.to_dict(), "gpu_name": gpu_name})
    return _cached
