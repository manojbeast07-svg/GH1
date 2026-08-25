"""Section 5 tests: the production Basic CUDA pipeline -- a single
native call running all five filters via reused ping-pong GPU buffers,
plus resolution grouping, safe GPU batch sizing, and batch-order
correctness.

--------------------------------------------------------------------------
Correctness tolerance for the full pipeline
--------------------------------------------------------------------------
Same finding as Section 4F: max_abs_diff on the final thresholded output
can be 255 (a step function amplifying tiny upstream Gaussian rounding
differences), which is expected, not a bug. The percentage of differing
pixels is the meaningful metric (Section 4F measured 0.008%-0.038% across
30 real images); this file asserts <=1%, the same empirically-justified
bound. Individual filters keep their own established exact/tolerance
standards elsewhere (tests/test_cuda_gaussian.py etc.) -- this file only
checks the full chain and the new batching/orchestration machinery.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

xray_cuda = pytest.importorskip(
    "xray_cuda", reason="xray_cuda extension is not built; run cmake --build build first."
)

from cpu.filters import FilterConfig, apply_gaussian, apply_laplacian, apply_median, apply_sobel, apply_threshold  # noqa: E402
from cuda.pipeline import (  # noqa: E402
    compute_safe_gpu_batch_size,
    group_by_resolution,
    run_basic_cuda_pipeline,
    run_basic_cuda_pipeline_selection,
)
from pipeline.dataset import DatasetManager  # noqa: E402
from pipeline.image_loader import load_image  # noqa: E402
from tests.conftest import get_real_dataset_path  # noqa: E402
from tests.fixtures import FIXTURES  # noqa: E402

cuda_present = xray_cuda.cuda_available()
pytestmark = pytest.mark.skipif(
    not cuda_present, reason="No usable CUDA device detected on this machine."
)

CHAIN_TOLERANCE_PCT = 1.0  # empirically justified in Section 4F; see module docstring


def _cpu_full_pipeline(image, config):
    return apply_threshold(
        apply_laplacian(apply_sobel(apply_median(apply_gaussian(image, config), config), config), config), config
    )


def _assert_pipeline_matches_cpu(image, config):
    cpu_out = _cpu_full_pipeline(image, config)
    gpu_batch, timing = run_basic_cuda_pipeline([image], config)
    gpu_out = gpu_batch[0]

    assert gpu_out.dtype == np.uint8
    assert gpu_out.shape == image.shape

    diff = np.abs(cpu_out.astype(np.int16) - gpu_out.astype(np.int16))
    pct_differing = 100.0 * np.count_nonzero(diff) / diff.size
    assert pct_differing <= CHAIN_TOLERANCE_PCT, (
        f"{pct_differing:.4f}% of pixels differ (max_abs_diff={diff.max()}), "
        f"exceeds chain_tolerance={CHAIN_TOLERANCE_PCT}%"
    )
    return timing


# -- single native call architecture -------------------------------------------------------


def test_pipeline_uses_a_single_native_call():
    """The whole point of Section 5: one xray_cuda call, not five. Patch
    the individual per-filter bindings and confirm they are never
    invoked by run_basic_cuda_pipeline -- only run_basic_cuda_pipeline_gpu is."""
    import unittest.mock as mock

    image = FIXTURES["random_deterministic"]()
    config = FilterConfig()

    with mock.patch.object(xray_cuda, "gaussian_basic_gpu") as m_g, \
         mock.patch.object(xray_cuda, "median_basic_gpu") as m_m, \
         mock.patch.object(xray_cuda, "sobel_basic_gpu") as m_s, \
         mock.patch.object(xray_cuda, "laplacian_basic_gpu") as m_l, \
         mock.patch.object(xray_cuda, "threshold_basic_gpu") as m_t:
        run_basic_cuda_pipeline([image], config)
        m_g.assert_not_called()
        m_m.assert_not_called()
        m_s.assert_not_called()
        m_l.assert_not_called()
        m_t.assert_not_called()


def test_pipeline_returns_all_stage_timings():
    image = FIXTURES["constant"]()
    _output, timing = run_basic_cuda_pipeline([image], FilterConfig())
    assert timing.h2d_ms >= 0.0
    assert timing.gaussian_ms is not None and timing.gaussian_ms >= 0.0
    assert timing.median_ms is not None and timing.median_ms >= 0.0
    assert timing.sobel_ms is not None and timing.sobel_ms >= 0.0
    assert timing.laplacian_ms is not None and timing.laplacian_ms >= 0.0
    assert timing.threshold_ms is not None and timing.threshold_ms >= 0.0
    assert timing.d2h_ms >= 0.0
    assert timing.compute_ms == pytest.approx(
        timing.gaussian_ms + timing.median_ms + timing.sobel_ms + timing.laplacian_ms + timing.threshold_ms
    )
    assert timing.total_ms == pytest.approx(timing.h2d_ms + timing.compute_ms + timing.d2h_ms)


def test_pipeline_disabled_stage_timing_is_none_not_zero():
    image = FIXTURES["constant"]()
    config = FilterConfig(median_enabled=False, laplacian_enabled=False)
    _output, timing = run_basic_cuda_pipeline([image], config)
    assert timing.gaussian_ms is not None
    assert timing.median_ms is None
    assert timing.sobel_ms is not None
    assert timing.laplacian_ms is None
    assert timing.threshold_ms is not None


def test_pipeline_all_disabled_returns_original():
    image = FIXTURES["random_deterministic"]()
    config = FilterConfig(
        gaussian_enabled=False, median_enabled=False, sobel_enabled=False,
        laplacian_enabled=False, threshold_enabled=False,
    )
    output_batch, timing = run_basic_cuda_pipeline([image], config)
    np.testing.assert_array_equal(output_batch[0], image)
    assert timing.compute_ms == 0.0


# -- correctness vs CPU (single image and small batch) -------------------------------------------------------


@pytest.mark.parametrize("fixture_name", list(FIXTURES.keys()))
def test_pipeline_matches_cpu_on_synthetic_fixtures(fixture_name):
    image = FIXTURES[fixture_name]()
    _assert_pipeline_matches_cpu(image, FilterConfig())


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_pipeline_matches_cpu_on_real_batch():
    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=5, seed=42)
    config = FilterConfig()

    images = [load_image(p) for p in selection.selected_paths]
    shape = images[0].shape
    assert all(im.shape == shape for im in images), "test fixture assumes a same-resolution sample"

    gpu_batch, _timing = run_basic_cuda_pipeline(images, config)
    for i, image in enumerate(images):
        cpu_out = _cpu_full_pipeline(image, config)
        diff = np.abs(cpu_out.astype(np.int16) - gpu_batch[i].astype(np.int16))
        pct = 100.0 * np.count_nonzero(diff) / diff.size
        assert pct <= CHAIN_TOLERANCE_PCT, f"image {i}: {pct:.4f}% differ, exceeds {CHAIN_TOLERANCE_PCT}%"


# -- batch order / no-missing / no-duplicate correctness -------------------------------------------------------


def test_batch_output_order_matches_input_order():
    """Recognizable per-image patterns (a unique constant value each) so
    an order/index bug is impossible to miss."""
    images = [np.full((32, 32), value, dtype=np.uint8) for value in (10, 60, 110, 160, 210)]
    config = FilterConfig(
        gaussian_enabled=False, median_enabled=False, sobel_enabled=False,
        laplacian_enabled=False, threshold_enabled=False,
    )  # identity pipeline -- output should equal input, in the same order
    output_batch, _timing = run_basic_cuda_pipeline(images, config)

    assert len(output_batch) == len(images)
    for i, image in enumerate(images):
        np.testing.assert_array_equal(output_batch[i], image, err_msg=f"index {i} does not match input order")


@pytest.mark.parametrize("batch_size", [1, 8, 32, 64])
def test_batch_sizes_no_missing_no_duplicate(batch_size):
    rng = np.random.default_rng(batch_size)
    images = [rng.integers(0, 256, size=(16, 16), dtype=np.uint8) for _ in range(batch_size)]
    config = FilterConfig(
        gaussian_enabled=False, median_enabled=False, sobel_enabled=False,
        laplacian_enabled=False, threshold_enabled=False,
    )
    output_batch, _timing = run_basic_cuda_pipeline(images, config)
    assert len(output_batch) == batch_size
    for i in range(batch_size):
        np.testing.assert_array_equal(output_batch[i], images[i])


def test_pipeline_rejects_mismatched_resolutions():
    images = [np.zeros((16, 16), dtype=np.uint8), np.zeros((32, 32), dtype=np.uint8)]
    with pytest.raises(ValueError):
        run_basic_cuda_pipeline(images, FilterConfig())


def test_pipeline_rejects_empty_list():
    with pytest.raises(ValueError):
        run_basic_cuda_pipeline([], FilterConfig())


# -- resolution grouping -------------------------------------------------------


def test_group_by_resolution(tmp_path):
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"small_{i}.png"), np.zeros((16, 16), dtype=np.uint8))
    for i in range(2):
        cv2.imwrite(str(tmp_path / f"big_{i}.png"), np.zeros((32, 32), dtype=np.uint8))

    dm = DatasetManager(tmp_path)
    dm.scan()
    selection = dm.random_batch(batch_size=5, seed=1)

    groups = group_by_resolution(selection)
    assert set(groups.keys()) == {(16, 16), (32, 32)}
    assert len(groups[(16, 16)]) == 3
    assert len(groups[(32, 32)]) == 2


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_group_by_resolution_on_real_mixed_dataset():
    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=200, seed=42)
    groups = group_by_resolution(selection)

    total_grouped = sum(len(items) for items in groups.values())
    assert total_grouped == len(selection)
    assert (224, 224) in groups  # the dominant resolution should appear in a 200-image sample


# -- safe GPU batch size -------------------------------------------------------


def test_compute_safe_gpu_batch_size_scales_with_free_memory():
    small = compute_safe_gpu_batch_size(224, 224, free_bytes=100_000_000)
    large = compute_safe_gpu_batch_size(224, 224, free_bytes=1_000_000_000)
    assert large > small
    assert small >= 1  # never returns 0, even under tight memory


def test_compute_safe_gpu_batch_size_scales_inversely_with_resolution():
    small_res = compute_safe_gpu_batch_size(224, 224, free_bytes=500_000_000)
    large_res = compute_safe_gpu_batch_size(2000, 2000, free_bytes=500_000_000)
    assert small_res > large_res


def test_compute_safe_gpu_batch_size_rejects_bad_safety_factor():
    with pytest.raises(ValueError):
        compute_safe_gpu_batch_size(224, 224, free_bytes=1_000_000_000, safety_factor=0.0)
    with pytest.raises(ValueError):
        compute_safe_gpu_batch_size(224, 224, free_bytes=1_000_000_000, safety_factor=1.5)


def test_compute_safe_gpu_batch_size_uses_live_free_memory_by_default():
    result = compute_safe_gpu_batch_size(224, 224)  # no free_bytes given -> queries the device
    assert result >= 1


# -- requested batch size vs effective GPU chunk size -------------------------------------------------------


def test_requested_batch_larger_than_gpu_capacity_is_chunked_not_rejected(tmp_path):
    for i in range(10):
        cv2.imwrite(str(tmp_path / f"img_{i}.png"), np.full((16, 16), i * 10, dtype=np.uint8))
    dm = DatasetManager(tmp_path)
    dm.scan()
    selection = dm.random_batch(batch_size=10, seed=1)

    # Force a tiny effective chunk size (3) even though the "requested"
    # selection has 10 images -- must still process everything, in order.
    results = run_basic_cuda_pipeline_selection(selection, FilterConfig(), max_batch_size=3)
    assert len(results) == 10
    assert [item.index for item, _out in results] == [item.index for item in selection.items]


# -- full orchestrator: selection -> grouped, chunked, reassembled -------------------------------------------------------


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_run_basic_cuda_pipeline_selection_preserves_order_across_resolution_groups():
    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=50, seed=7)

    results = run_basic_cuda_pipeline_selection(selection, FilterConfig(), max_batch_size=16)

    assert len(results) == 50
    assert [item.index for item, _out in results] == [item.index for item in selection.items]
    for item, output in results:
        original = load_image(item.absolute_path)
        assert output.shape == original.shape  # native resolution preserved, no resizing
        assert output.dtype == np.uint8


# -- memory: buffer reuse / cleanup -------------------------------------------------------


def test_repeated_pipeline_calls_no_crash_and_no_leak():
    image = FIXTURES["random_deterministic"]()
    config = FilterConfig()
    expected, _timing = run_basic_cuda_pipeline([image], config)

    free_before = xray_cuda.device_memory_info()["free_bytes"]

    for _ in range(50):
        output, _timing = run_basic_cuda_pipeline([image], config)
        np.testing.assert_array_equal(output, expected)

    free_after = xray_cuda.device_memory_info()["free_bytes"]
    leaked_bytes = free_before - free_after
    assert leaked_bytes < 50 * 1024 * 1024, (
        f"Possible GPU memory leak: {leaked_bytes} bytes not reclaimed after 50 pipeline calls "
        f"(free before={free_before}, after={free_after})"
    )


def test_pipeline_batch_of_32_no_crash():
    rng = np.random.default_rng(32)
    images = [rng.integers(0, 256, size=(64, 64), dtype=np.uint8) for _ in range(32)]
    output_batch, timing = run_basic_cuda_pipeline(images, FilterConfig())
    assert len(output_batch) == 32
    assert timing.total_ms >= 0.0
