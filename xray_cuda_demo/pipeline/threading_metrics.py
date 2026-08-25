"""Section 23: centralized CPU/GPU threading & parallelism metrics for the
Streamlit "Threading & Parallelism" tab.

This module is the single place threading data is collected (spec item 47:
"Do not collect these independently in five UI files"). Every field is
either a real, live-queried value (psutil, cv2, xray_cuda.device_info(),
xray_cuda.compute_launch_config()) or explicitly `None` -- never estimated
or fabricated (spec items 34-36). `None` fields render as "N/A" / "Not
profiled" in the UI, distinct from a real zero.

Read-only: this module never runs a filter, allocates a GPU buffer, or
launches a kernel just to report metrics -- it only queries device
properties and the launch-configuration arithmetic every production kernel
call already uses (xray_cuda.compute_launch_config(), added in this
section as a thin, read-only wrapper around the same
compute_launch_grid()/default_block_dim() functions production code
calls -- see cuda/src/bindings.cpp).
"""

from __future__ import annotations

import math
import platform
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

try:
    import cv2
except ImportError:  # pragma: no cover -- opencv-python is a hard requirement, but stay defensive
    cv2 = None

try:
    import psutil
except ImportError:  # pragma: no cover -- optional; physical-core count degrades to None without it
    psutil = None

try:
    import xray_cuda
except ImportError:  # pragma: no cover -- CUDA extension not built
    xray_cuda = None


# Matches the production block dimension every Basic and Enhanced kernel
# actually uses (cuda/gaussian.py DEFAULT_ENHANCED_BLOCK etc., all (16, 16);
# gpu_image.cuh's default_block_dim()) -- verified identical across all
# five filters and both implementations in Section 22's audit.
PRODUCTION_BLOCK = (16, 16)

# threads_per_output_element for each filter's PRODUCTION variant, at the
# representative resolution's width parity. Gaussian is 2 passes (H then
# V), each 1 thread per intermediate/output element -- NOT "one thread per
# final pixel" the way a single-pass filter is. Threshold's Vectorized
# variant processes 4 pixels/thread via uchar4 when width % 4 == 0 (the
# production dispatch's own documented fallback condition -- see
# cuda/src/threshold_enhanced.cu); otherwise it falls back to 1
# pixel/thread automatically, exactly like the real kernel does.
def _threshold_pixels_per_thread(width: int) -> int:
    return 4 if width % 4 == 0 else 1


@dataclass
class FilterThreadingInfo:
    filter_name: str
    launches_per_call: int
    pixels_per_thread: int
    note: str


@dataclass
class ThreadingMetrics:
    # -- CPU --
    cpu_model: Optional[str]
    cpu_physical_cores: Optional[int]
    cpu_logical_processors: Optional[int]
    opencv_threads: Optional[int]
    opencv_threads_source: str  # "cv2.getNumThreads() -- configured, not necessarily all active"

    # -- GPU hardware --
    gpu_available: bool
    gpu_model: Optional[str]
    gpu_sm_count: Optional[int]
    gpu_warp_size: Optional[int]
    gpu_max_threads_per_block: Optional[int]
    gpu_compute_capability: Optional[str]
    gpu_vram_bytes: Optional[int]
    gpu_free_vram_bytes: Optional[int]
    gpu_shared_mem_per_block_limit_bytes: Optional[int]
    gpu_async_engine_count: Optional[int]

    # -- launch configuration, for the representative (or caller-supplied) image size/batch --
    representative_width: int
    representative_height: int
    representative_batch_size: int
    block_dimensions: Optional[Tuple[int, int, int]]
    grid_dimensions: Optional[Tuple[int, int, int]]
    threads_per_block: Optional[int]
    warps_per_block: Optional[int]
    total_threads_launched: Optional[int]

    # -- per-filter breakdown for the representative image --
    per_filter: List[FilterThreadingInfo]

    # -- profiler-only metrics (Nsight Compute), not available in this environment (Sections 20D/20E/20F
    # all independently confirmed nsys/ncu require Administrator privileges not present here) --
    registers_per_thread: Optional[int]
    occupancy_pct: Optional[float]
    gpu_utilization_pct: Optional[float]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["per_filter"] = [asdict(f) for f in self.per_filter]
        return d


def _cpu_fields() -> Dict[str, Optional[object]]:
    logical = None
    physical = None
    if psutil is not None:
        try:
            logical = psutil.cpu_count(logical=True)
            physical = psutil.cpu_count(logical=False)
        except Exception:
            pass
    if logical is None:
        import os

        logical = os.cpu_count()
    return {
        "cpu_model": platform.processor() or platform.uname().processor or None,
        "cpu_physical_cores": physical,
        "cpu_logical_processors": logical,
    }


def _opencv_threads() -> Optional[int]:
    if cv2 is None:
        return None
    try:
        return int(cv2.getNumThreads())
    except Exception:
        return None


def _gpu_fields() -> Dict[str, Optional[object]]:
    empty = {
        "gpu_available": False, "gpu_model": None, "gpu_sm_count": None, "gpu_warp_size": None,
        "gpu_max_threads_per_block": None, "gpu_compute_capability": None, "gpu_vram_bytes": None,
        "gpu_free_vram_bytes": None, "gpu_shared_mem_per_block_limit_bytes": None,
        "gpu_async_engine_count": None,
    }
    if xray_cuda is None or not xray_cuda.cuda_available():
        return empty
    try:
        info = xray_cuda.device_info()
    except Exception:
        return empty
    free_bytes = None
    try:
        mem = xray_cuda.device_memory_info()
        free_bytes = mem.get("free_bytes")
    except Exception:
        pass
    return {
        "gpu_available": True,
        "gpu_model": info.get("name"),
        "gpu_sm_count": info.get("multiprocessor_count"),
        "gpu_warp_size": info.get("warp_size"),
        "gpu_max_threads_per_block": info.get("max_threads_per_block"),
        "gpu_compute_capability": info.get("compute_capability"),
        "gpu_vram_bytes": info.get("total_global_mem_bytes"),
        "gpu_free_vram_bytes": free_bytes,
        "gpu_shared_mem_per_block_limit_bytes": info.get("shared_mem_per_block_bytes"),
        # asyncEngineCount is not currently exposed by device_info() (added ad hoc in Section 20F's
        # research via a throwaway probe, not wired into the binding) -- reported as None here rather
        # than re-querying a value this module cannot itself obtain live.
        "gpu_async_engine_count": None,
    }


def _launch_fields(width: int, height: int, batch_size: int, block: Tuple[int, int]) -> Dict[str, Optional[object]]:
    if xray_cuda is None or not xray_cuda.cuda_available():
        return {
            "block_dimensions": None, "grid_dimensions": None, "threads_per_block": None,
            "warps_per_block": None, "total_threads_launched": None,
        }
    try:
        cfg = xray_cuda.compute_launch_config(
            width=width, height=height, batch_size=batch_size, block_x=block[0], block_y=block[1])
    except Exception:
        return {
            "block_dimensions": None, "grid_dimensions": None, "threads_per_block": None,
            "warps_per_block": None, "total_threads_launched": None,
        }
    warp_size = None
    try:
        warp_size = xray_cuda.device_info().get("warp_size")
    except Exception:
        pass
    warps_per_block = math.ceil(cfg["threads_per_block"] / warp_size) if warp_size else None
    return {
        "block_dimensions": (cfg["block_x"], cfg["block_y"], cfg["block_z"]),
        "grid_dimensions": (cfg["grid_x"], cfg["grid_y"], cfg["grid_z"]),
        "threads_per_block": cfg["threads_per_block"],
        "warps_per_block": warps_per_block,
        "total_threads_launched": cfg["total_threads_launched"],
    }


def _per_filter_breakdown(width: int) -> List[FilterThreadingInfo]:
    return [
        FilterThreadingInfo(
            filter_name="Gaussian", launches_per_call=2, pixels_per_thread=1,
            note="Separable: horizontal pass then vertical pass, each kernel one thread per element."),
        FilterThreadingInfo(
            filter_name="Median", launches_per_call=1, pixels_per_thread=1,
            note="One thread per output pixel."),
        FilterThreadingInfo(
            filter_name="Sobel", launches_per_call=1, pixels_per_thread=1,
            note="One thread per output pixel."),
        FilterThreadingInfo(
            filter_name="Laplacian", launches_per_call=1, pixels_per_thread=1,
            note="One thread per output pixel."),
        FilterThreadingInfo(
            filter_name="Threshold", launches_per_call=1, pixels_per_thread=_threshold_pixels_per_thread(width),
            note=(
                "Vectorized: each thread processes 4 pixels via uchar4 (width is a multiple of 4)."
                if _threshold_pixels_per_thread(width) == 4
                else "Vectorized dispatch falls back to 1 pixel/thread (scalar) -- width is not a multiple of 4."
            )),
    ]


def get_threading_metrics(
    width: int = 224, height: int = 224, batch_size: int = 1,
    block: Tuple[int, int] = PRODUCTION_BLOCK,
) -> ThreadingMetrics:
    """Collects all Threading & Parallelism tab data from real sources.
    `width`/`height`/`batch_size` select the representative image/batch
    the launch-configuration numbers are computed for (default: a single
    224x224 image, this project's dominant resolution)."""
    cpu = _cpu_fields()
    gpu = _gpu_fields()
    launch = _launch_fields(width, height, batch_size, block)

    return ThreadingMetrics(
        cpu_model=cpu["cpu_model"],
        cpu_physical_cores=cpu["cpu_physical_cores"],
        cpu_logical_processors=cpu["cpu_logical_processors"],
        opencv_threads=_opencv_threads(),
        opencv_threads_source="cv2.getNumThreads() -- configured maximum, not a measurement of threads actually active during any specific call.",
        gpu_available=bool(gpu["gpu_available"]),
        gpu_model=gpu["gpu_model"],
        gpu_sm_count=gpu["gpu_sm_count"],
        gpu_warp_size=gpu["gpu_warp_size"],
        gpu_max_threads_per_block=gpu["gpu_max_threads_per_block"],
        gpu_compute_capability=gpu["gpu_compute_capability"],
        gpu_vram_bytes=gpu["gpu_vram_bytes"],
        gpu_free_vram_bytes=gpu["gpu_free_vram_bytes"],
        gpu_shared_mem_per_block_limit_bytes=gpu["gpu_shared_mem_per_block_limit_bytes"],
        gpu_async_engine_count=gpu["gpu_async_engine_count"],
        representative_width=width,
        representative_height=height,
        representative_batch_size=batch_size,
        block_dimensions=launch["block_dimensions"],
        grid_dimensions=launch["grid_dimensions"],
        threads_per_block=launch["threads_per_block"],
        warps_per_block=launch["warps_per_block"],
        total_threads_launched=launch["total_threads_launched"],
        per_filter=_per_filter_breakdown(width),
        # Nsight Compute (registers/thread, occupancy) and Nsight Systems (GPU utilization) were
        # confirmed unavailable in every session that reached Sections 20D/20E/20F (no Administrator
        # privileges) -- never calculated from threads/block, only ever a real profiler measurement.
        registers_per_thread=None,
        occupancy_pct=None,
        gpu_utilization_pct=None,
    )
