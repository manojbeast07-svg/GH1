"""Environment fingerprint (Section 11 spec items 8, 32): everything a
benchmark result needs to record about the machine/software stack it
ran on, since CUDA performance is hardware- and driver-specific.

Deliberately best-effort and cheap: every field is either already
exposed by the platform/xray_cuda/importlib.metadata (no new
subprocess-heavy probing beyond what scripts/diagnose.py already does
for nvcc/cmake), or falls back to `None`/"unknown" rather than raising
-- a missing optional field (e.g. no git repository) must never abort a
benchmark run.
"""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _pkg_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _gpu_fields() -> dict:
    try:
        import xray_cuda
    except ImportError:
        return {
            "gpu_available": False, "gpu_name": None, "gpu_vram_bytes": None,
            "gpu_compute_capability": None, "gpu_driver_version": None, "cuda_runtime_version": None,
        }
    if not xray_cuda.cuda_available():
        return {
            "gpu_available": False, "gpu_name": None, "gpu_vram_bytes": None,
            "gpu_compute_capability": None, "gpu_driver_version": None, "cuda_runtime_version": None,
        }
    info = xray_cuda.device_info()
    return {
        "gpu_available": True,
        "gpu_name": info.get("name"),
        "gpu_vram_bytes": info.get("total_global_mem_bytes"),
        "gpu_compute_capability": info.get("compute_capability"),
        "gpu_driver_version": info.get("driver_version"),
        "cuda_runtime_version": info.get("runtime_version"),
    }


@dataclass
class EnvironmentFingerprint:
    os_platform: str
    cpu_model: Optional[str]
    cpu_logical_cores: Optional[int]
    gpu_available: bool
    gpu_name: Optional[str]
    gpu_vram_bytes: Optional[int]
    gpu_compute_capability: Optional[str]
    gpu_driver_version: Optional[str]
    cuda_runtime_version: Optional[str]
    python_version: str
    opencv_version: Optional[str]
    numpy_version: Optional[str]
    pybind11_version: Optional[str]
    git_commit: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


def get_environment_fingerprint() -> EnvironmentFingerprint:
    import os as _os

    gpu = _gpu_fields()
    return EnvironmentFingerprint(
        os_platform=platform.platform(),
        cpu_model=platform.processor() or platform.uname().processor or None,
        cpu_logical_cores=_os.cpu_count(),
        gpu_available=gpu["gpu_available"],
        gpu_name=gpu["gpu_name"],
        gpu_vram_bytes=gpu["gpu_vram_bytes"],
        gpu_compute_capability=gpu["gpu_compute_capability"],
        gpu_driver_version=gpu["gpu_driver_version"],
        cuda_runtime_version=gpu["cuda_runtime_version"],
        python_version=sys.version.split()[0],
        opencv_version=_pkg_version("opencv-python") or _pkg_version("opencv-python-headless"),
        numpy_version=_pkg_version("numpy"),
        pybind11_version=_pkg_version("pybind11"),
        git_commit=_git_commit(),
    )
