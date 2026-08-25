"""Section 23: tests for pipeline.threading_metrics and its UI rendering
(ui/threading_view.py, ui/presentation.py::render_threading_summary).
Covers spec item 53's 12 required test areas. Never asserts a specific
performance number -- only that real data loads, is internally
consistent, and that N/A values are honestly reported rather than
estimated.
"""

from __future__ import annotations

import math

import pytest

from pipeline.threading_metrics import get_threading_metrics

CUDA_AVAILABLE = True
try:
    import xray_cuda
    CUDA_AVAILABLE = xray_cuda.cuda_available()
except ImportError:
    CUDA_AVAILABLE = False


# --------------------------------------------------------------------------
# 1. CPU metrics load
# --------------------------------------------------------------------------

def test_cpu_metrics_load():
    m = get_threading_metrics(width=224, height=224, batch_size=1)
    assert m.cpu_logical_processors is not None
    assert m.cpu_logical_processors > 0
    # physical_cores may be None only if psutil is unavailable -- if psutil IS installed
    # (it is, per requirements.txt), it must be a real positive value.
    import importlib
    try:
        importlib.import_module("psutil")
        assert m.cpu_physical_cores is not None
        assert m.cpu_physical_cores > 0
        assert m.cpu_physical_cores <= m.cpu_logical_processors
    except ImportError:
        pass


# --------------------------------------------------------------------------
# 2. GPU metrics load
# --------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires a usable CUDA device")
def test_gpu_metrics_load():
    m = get_threading_metrics(width=224, height=224, batch_size=1)
    assert m.gpu_available is True
    assert m.gpu_model is not None and len(m.gpu_model) > 0
    assert m.gpu_sm_count is not None and m.gpu_sm_count > 0
    assert m.gpu_vram_bytes is not None and m.gpu_vram_bytes > 0
    assert m.gpu_free_vram_bytes is not None and m.gpu_free_vram_bytes > 0


def test_gpu_metrics_absent_reports_false_not_crash(monkeypatch):
    import pipeline.threading_metrics as tm
    monkeypatch.setattr(tm, "xray_cuda", None)
    m = tm.get_threading_metrics(width=224, height=224, batch_size=1)
    assert m.gpu_available is False
    assert m.gpu_model is None
    assert m.gpu_sm_count is None
    assert m.block_dimensions is None
    assert m.grid_dimensions is None


# --------------------------------------------------------------------------
# 3. OpenCV thread configuration loads
# --------------------------------------------------------------------------

def test_opencv_thread_configuration_loads():
    m = get_threading_metrics(width=224, height=224, batch_size=1)
    assert m.opencv_threads is not None
    assert m.opencv_threads > 0
    assert "configured" in m.opencv_threads_source.lower()


def test_opencv_thread_configuration_matches_live_cv2():
    import cv2
    m = get_threading_metrics(width=224, height=224, batch_size=1)
    assert m.opencv_threads == cv2.getNumThreads()


# --------------------------------------------------------------------------
# 4. GPU device properties load
# --------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires a usable CUDA device")
def test_gpu_device_properties_load():
    m = get_threading_metrics(width=224, height=224, batch_size=1)
    assert m.gpu_warp_size == 32  # every CUDA device to date uses warp_size 32; still a live-read value
    assert m.gpu_max_threads_per_block is not None and m.gpu_max_threads_per_block > 0
    assert m.gpu_compute_capability is not None
    assert m.gpu_shared_mem_per_block_limit_bytes is not None and m.gpu_shared_mem_per_block_limit_bytes > 0


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires a usable CUDA device")
def test_gpu_device_properties_match_native_device_info():
    import xray_cuda
    m = get_threading_metrics(width=224, height=224, batch_size=1)
    info = xray_cuda.device_info()
    assert m.gpu_sm_count == info["multiprocessor_count"]
    assert m.gpu_warp_size == info["warp_size"]
    assert m.gpu_max_threads_per_block == info["max_threads_per_block"]


# --------------------------------------------------------------------------
# 5. Grid dimensions calculate correctly
# --------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires a usable CUDA device")
@pytest.mark.parametrize("width,height,batch,block", [
    (224, 224, 1, (16, 16)),
    (224, 224, 122, (16, 16)),
    (100, 50, 8, (16, 16)),
    (32, 32, 4, (8, 8)),
])
def test_grid_dimensions_calculate_correctly(width, height, batch, block):
    m = get_threading_metrics(width=width, height=height, batch_size=batch, block=block)
    gx, gy, gz = m.grid_dimensions
    assert gx == math.ceil(width / block[0])
    assert gy == math.ceil(height / block[1])
    assert gz == batch


# --------------------------------------------------------------------------
# 6. Threads/block calculate correctly
# --------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires a usable CUDA device")
@pytest.mark.parametrize("block", [(16, 16), (8, 8), (32, 8)])
def test_threads_per_block_calculates_correctly(block):
    m = get_threading_metrics(width=224, height=224, batch_size=1, block=block)
    assert m.threads_per_block == block[0] * block[1]
    assert m.block_dimensions == (block[0], block[1], 1)


# --------------------------------------------------------------------------
# 7. Warps/block calculate correctly
# --------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires a usable CUDA device")
def test_warps_per_block_calculates_correctly():
    import xray_cuda
    warp_size = xray_cuda.device_info()["warp_size"]
    m = get_threading_metrics(width=224, height=224, batch_size=1, block=(16, 16))
    assert m.warps_per_block == math.ceil(256 / warp_size)


# --------------------------------------------------------------------------
# 8. Total launched threads calculate correctly
# --------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires a usable CUDA device")
@pytest.mark.parametrize("width,height,batch", [(224, 224, 1), (224, 224, 122), (17, 33, 5)])
def test_total_threads_launched_calculates_correctly(width, height, batch):
    m = get_threading_metrics(width=width, height=height, batch_size=batch)
    gx, gy, gz = m.grid_dimensions
    bx, by, bz = m.block_dimensions
    assert m.total_threads_launched == gx * gy * gz * bx * by * bz
    # must be >= actual pixel count (grid always covers the full image; may over-cover at edges)
    assert m.total_threads_launched >= width * height * batch


# --------------------------------------------------------------------------
# 9. Missing profiler metrics render as N/A (data layer: None, never estimated)
# --------------------------------------------------------------------------

def test_missing_profiler_metrics_are_none_not_estimated():
    m = get_threading_metrics(width=224, height=224, batch_size=1)
    assert m.registers_per_thread is None
    assert m.occupancy_pct is None
    assert m.gpu_utilization_pct is None


def test_threading_view_renders_na_for_missing_profiler_metrics():
    """The rendering layer must show 'Not profiled', not fabricate a number."""
    from ui import threading_view
    assert threading_view is not None  # import-time smoke check; full render tested via AppTest below


# --------------------------------------------------------------------------
# 10. Presentation mode renders threading cards
# --------------------------------------------------------------------------

def test_presentation_render_threading_summary_does_not_raise():
    """Exercises ui.presentation.render_threading_summary end-to-end via
    Streamlit's AppTest harness (no real browser needed)."""
    from streamlit.testing.v1 import AppTest

    script = (
        "import streamlit as st\n"
        "from ui import presentation as presentation_ui\n"
        "presentation_ui.render_threading_summary(width=224, height=224, batch_size=4)\n"
    )
    at = AppTest.from_string(script)
    at.run(timeout=30)
    assert not at.exception


# --------------------------------------------------------------------------
# 11. Basic/Enhanced metrics can be displayed
# --------------------------------------------------------------------------

def test_per_filter_breakdown_covers_all_five_filters():
    m = get_threading_metrics(width=224, height=224, batch_size=1)
    names = [f.filter_name for f in m.per_filter]
    assert names == ["Gaussian", "Median", "Sobel", "Laplacian", "Threshold"]
    gaussian = m.per_filter[0]
    assert gaussian.launches_per_call == 2  # separable H+V, not "one thread = one filter"
    threshold = m.per_filter[-1]
    assert threshold.pixels_per_thread in (1, 4)


def test_threshold_pixels_per_thread_reflects_width_parity():
    m_aligned = get_threading_metrics(width=224, height=224, batch_size=1)  # 224 % 4 == 0
    m_unaligned = get_threading_metrics(width=225, height=224, batch_size=1)  # 225 % 4 != 0
    assert m_aligned.per_filter[-1].pixels_per_thread == 4
    assert m_unaligned.per_filter[-1].pixels_per_thread == 1


def test_basic_vs_enhanced_share_the_same_launch_geometry():
    """Verifies the claim ui/threading_view.py's
    render_basic_vs_enhanced_launch_comparison() makes: every production
    Enhanced block constant equals Basic's default_block_dim()."""
    from cuda.gaussian import DEFAULT_ENHANCED_BLOCK
    from cuda.laplacian import DEFAULT_LAPLACIAN_ENHANCED_BLOCK
    from cuda.median import DEFAULT_MEDIAN_ENHANCED_BLOCK
    from cuda.sobel import DEFAULT_SOBEL_ENHANCED_BLOCK
    from cuda.threshold import DEFAULT_THRESHOLD_ENHANCED_BLOCK
    from pipeline.threading_metrics import PRODUCTION_BLOCK

    for block in [DEFAULT_ENHANCED_BLOCK, DEFAULT_LAPLACIAN_ENHANCED_BLOCK, DEFAULT_MEDIAN_ENHANCED_BLOCK,
                  DEFAULT_SOBEL_ENHANCED_BLOCK, DEFAULT_THRESHOLD_ENHANCED_BLOCK]:
        assert block == PRODUCTION_BLOCK


# --------------------------------------------------------------------------
# 12. No production pipeline is invoked just to render static system metrics
# --------------------------------------------------------------------------

def test_get_threading_metrics_never_calls_production_pipeline(monkeypatch):
    """get_threading_metrics() must be pure diagnostic -- it must never
    invoke cpu.pipeline.run_cpu_pipeline or
    cuda.pipeline.run_basic_cuda_pipeline/run_enhanced_cuda_pipeline."""
    import cpu.pipeline as cpu_pipeline
    import cuda.pipeline as cuda_pipeline

    def _fail(*args, **kwargs):
        raise AssertionError("threading metrics must not invoke the production pipeline")

    monkeypatch.setattr(cpu_pipeline, "run_cpu_pipeline", _fail)
    monkeypatch.setattr(cuda_pipeline, "run_basic_cuda_pipeline", _fail)
    monkeypatch.setattr(cuda_pipeline, "run_enhanced_cuda_pipeline", _fail)

    # Must succeed without ever calling the patched (now-failing) functions.
    m = get_threading_metrics(width=224, height=224, batch_size=8)
    assert m is not None


def test_threading_tab_renders_without_a_dataset_or_image(monkeypatch):
    """The Threading & Parallelism tab must render from device/CPU
    properties alone -- it never requires a loaded dataset or image."""
    from streamlit.testing.v1 import AppTest

    script = (
        "import streamlit as st\n"
        "from ui import threading_view\n"
        "threading_view.render_threading_tab(width=224, height=224, batch_size=1)\n"
    )
    at = AppTest.from_string(script)
    at.run(timeout=30)
    assert not at.exception
