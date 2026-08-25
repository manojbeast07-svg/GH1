"""Section 4F tests: Basic CUDA binary threshold vs. the Section 3 CPU
reference (cv2.threshold + THRESH_BINARY) -- the simplest kernel in the
project (pure pointwise comparison, no neighborhood, no border).

--------------------------------------------------------------------------
Exact equality, not a tolerance
--------------------------------------------------------------------------
Thresholding is a single integer comparison -- max_abs_diff == 0 is the
only acceptable outcome anywhere in this file.

--------------------------------------------------------------------------
Equality boundary verified empirically before writing the kernel
--------------------------------------------------------------------------
cv2.threshold(..., THRESH_BINARY) is strict ">": pixel == threshold_value
maps to 0, not max_value. Confirmed with threshold-1/threshold/threshold+1
triples and the threshold_value=0 / threshold_value=255 edge cases (the
latter maps every pixel to 0 -- legitimate, not an error).
"""

from pathlib import Path

import numpy as np
import pytest

xray_cuda = pytest.importorskip(
    "xray_cuda", reason="xray_cuda extension is not built; run cmake --build build first."
)

from cpu.filters import FilterConfig, apply_threshold  # noqa: E402
from cuda.threshold import threshold_cuda, threshold_cuda_gpu  # noqa: E402
from pipeline.dataset import DatasetManager  # noqa: E402
from pipeline.image_loader import load_image  # noqa: E402
from tests.conftest import get_real_dataset_path  # noqa: E402
from tests.fixtures import FIXTURES  # noqa: E402

cuda_present = xray_cuda.cuda_available()
pytestmark = pytest.mark.skipif(
    not cuda_present, reason="No usable CUDA device detected on this machine."
)

REFERENCE_DIR = Path(__file__).resolve().parent / "reference"
THRESHOLDS = [0, 1, 127, 128, 254, 255]


def _assert_exact_match(image, threshold_value=128, max_value=255):
    config = FilterConfig(threshold_value=threshold_value, threshold_max_value=max_value)
    cpu_out = apply_threshold(image, config)
    gpu_out = threshold_cuda(image, threshold_value=threshold_value, max_value=max_value)

    assert gpu_out.dtype == np.uint8
    assert gpu_out.shape == image.shape

    diff = np.abs(cpu_out.astype(np.int16) - gpu_out.astype(np.int16))
    max_diff = int(diff.max())
    num_differing = int(np.count_nonzero(diff))
    assert max_diff == 0 and num_differing == 0, (
        f"threshold={threshold_value} max_value={max_value}: max_abs_diff={max_diff}, "
        f"differing={num_differing}/{diff.size} (expected exact match)"
    )


# -- fixtures x thresholds -------------------------------------------------------


@pytest.mark.parametrize("fixture_name", list(FIXTURES.keys()))
@pytest.mark.parametrize("threshold_value", THRESHOLDS)
def test_exact_match_on_synthetic_fixtures(fixture_name, threshold_value):
    image = FIXTURES[fixture_name]()
    _assert_exact_match(image, threshold_value=threshold_value)


@pytest.mark.parametrize("max_value", [1, 100, 200, 255])
def test_exact_match_various_max_values(max_value):
    image = FIXTURES["random_deterministic"]()
    _assert_exact_match(image, threshold_value=128, max_value=max_value)


# -- equality-boundary tests -------------------------------------------------------


@pytest.mark.parametrize("threshold_value", [0, 1, 50, 127, 128, 200, 254, 255])
def test_equality_boundary_exact(threshold_value):
    """threshold-1, threshold, threshold+1 in one tiny image -- the
    classic off-by-one trap for a threshold implementation."""
    lo = max(0, threshold_value - 1)
    hi = min(255, threshold_value + 1)
    image = np.array([[lo, threshold_value, hi]], dtype=np.uint8)
    _assert_exact_match(image, threshold_value=threshold_value, max_value=255)


def test_threshold_value_128_boundary_explicit():
    image = np.array([[126, 127, 128, 129, 130]], dtype=np.uint8)
    out = threshold_cuda(image, threshold_value=128, max_value=255)
    np.testing.assert_array_equal(out, np.array([[0, 0, 0, 255, 255]], dtype=np.uint8))


def test_threshold_value_zero_edge_case():
    image = np.array([[0, 1, 2, 254, 255]], dtype=np.uint8)
    out = threshold_cuda(image, threshold_value=0, max_value=255)
    np.testing.assert_array_equal(out, np.array([[0, 255, 255, 255, 255]], dtype=np.uint8))


def test_threshold_value_255_maps_everything_to_zero():
    image = np.array([[0, 1, 254, 255]], dtype=np.uint8)
    out = threshold_cuda(image, threshold_value=255, max_value=255)
    np.testing.assert_array_equal(out, np.array([[0, 0, 0, 0]], dtype=np.uint8))


# -- real X-rays -------------------------------------------------------


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
@pytest.mark.parametrize("threshold_value", THRESHOLDS)
def test_exact_match_on_real_224x224_xray(threshold_value):
    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=1, seed=42)
    image = load_image(selection.selected_paths[0])
    assert image.shape == (224, 224)
    _assert_exact_match(image, threshold_value=threshold_value)


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_exact_match_on_large_real_xray():
    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    large_path = None
    for path in dm.paths[:2000]:
        image = load_image(path)
        if image.shape != (224, 224) and image.shape[0] * image.shape[1] > 500_000:
            large_path = path
            break
    if large_path is None:
        pytest.skip("No sufficiently large non-224x224 image found in the first 2000 scanned files.")

    image = load_image(large_path)
    _assert_exact_match(image, threshold_value=128)


# -- dimensions (including non-block-multiple) -------------------------------------------------------


@pytest.mark.parametrize("height,width", [(7, 7), (16, 16), (31, 47), (224, 224)])
def test_exact_match_various_dimensions(height, width):
    rng = np.random.default_rng(height * 1000 + width)
    image = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    _assert_exact_match(image, threshold_value=128)


# -- reference-output regression (frozen Section 3 baselines, default config) -------------------------------------------------------


@pytest.mark.parametrize("fixture_name", list(FIXTURES.keys()))
def test_matches_frozen_cpu_reference_output(fixture_name):
    ref_path = REFERENCE_DIR / fixture_name / "threshold.npy"
    if not ref_path.exists():
        pytest.skip(f"No reference file at {ref_path}")

    image = FIXTURES[fixture_name]()
    expected = np.load(ref_path)
    gpu_out = threshold_cuda(image, threshold_value=128, max_value=255)  # frozen references use default FilterConfig
    np.testing.assert_array_equal(gpu_out, expected)


# -- determinism -------------------------------------------------------


def test_repeated_execution_is_deterministic():
    image = FIXTURES["random_deterministic"]()
    run_a = threshold_cuda(image, threshold_value=128)
    run_b = threshold_cuda(image, threshold_value=128)
    run_c = threshold_cuda(image, threshold_value=128)
    np.testing.assert_array_equal(run_a, run_b)
    np.testing.assert_array_equal(run_b, run_c)


# -- GPU-native API -------------------------------------------------------


def test_threshold_cuda_gpu_chains_without_host_round_trip():
    image = FIXTURES["random_deterministic"]()
    gpu_image = xray_cuda.upload_image(image)

    result = threshold_cuda_gpu(gpu_image, threshold_value=128, max_value=255)
    assert "output" in result and "kernel_ms" in result

    gpu_downloaded = xray_cuda.download_image(result["output"])
    host_api_output = threshold_cuda(image, threshold_value=128, max_value=255)
    np.testing.assert_array_equal(gpu_downloaded, host_api_output)


def test_threshold_basic_gpu_input_not_modified():
    image = FIXTURES["random_deterministic"]()
    gpu_image = xray_cuda.upload_image(image)
    xray_cuda.threshold_basic_gpu(gpu_image, 128, 255)
    unchanged = xray_cuda.download_image(gpu_image)
    np.testing.assert_array_equal(unchanged, image)


def test_threshold_output_is_binary():
    image = FIXTURES["random_deterministic"]()
    out = threshold_cuda(image, threshold_value=128, max_value=255)
    assert set(np.unique(out)).issubset({0, 255})


# -- input validation -------------------------------------------------------


def test_threshold_cuda_rejects_out_of_range_threshold():
    image = FIXTURES["constant"]()
    with pytest.raises(ValueError):
        threshold_cuda(image, threshold_value=-1)
    with pytest.raises(ValueError):
        threshold_cuda(image, threshold_value=256)


def test_threshold_cuda_rejects_out_of_range_max_value():
    image = FIXTURES["constant"]()
    with pytest.raises(ValueError):
        threshold_cuda(image, max_value=-1)
    with pytest.raises(ValueError):
        threshold_cuda(image, max_value=256)


def test_threshold_cuda_rejects_wrong_dtype():
    with pytest.raises(ValueError):
        threshold_cuda(np.zeros((8, 8), dtype=np.float32))


def test_threshold_cuda_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        threshold_cuda(np.zeros((8, 8, 3), dtype=np.uint8))


def test_threshold_cuda_rejects_empty_image():
    with pytest.raises(ValueError):
        threshold_cuda(np.zeros((0, 8), dtype=np.uint8))


def test_threshold_basic_gpu_rejects_null_image():
    with pytest.raises(ValueError):
        xray_cuda.threshold_basic_gpu(None, 128, 255)


def test_threshold_basic_gpu_rejects_out_of_range_params_directly():
    image = FIXTURES["constant"]()
    gpu_image = xray_cuda.upload_image(image)
    with pytest.raises(ValueError):
        xray_cuda.threshold_basic_gpu(gpu_image, -1, 255)
    with pytest.raises(ValueError):
        xray_cuda.threshold_basic_gpu(gpu_image, 128, 300)


# -- memory: repeated allocation/release -------------------------------------------------------


# -- full five-filter GPU-resident pipeline smoke test -------------------------------------------------------
#
# Threshold is the fifth and final Basic CUDA filter, so this is the
# first point in the project where all five can be assembled. This is
# explicitly a smoke test (spec section 26), not the production pipeline
# (Section 5) -- no buffer reuse, no streams, just proving the chain
# wires together entirely on GpuImage objects with zero CPU round trips
# between stages.
#
# --------------------------------------------------------------------------
# Why the final-output comparison uses "% of differing pixels", not
# max_abs_diff
# --------------------------------------------------------------------------
# Measured (not assumed) before writing this test: comparing the CPU
# five-filter chain against the GPU five-filter chain on 30 real X-rays,
# max_abs_diff on the *final thresholded output* is 255 in nearly every
# case. This is expected, not a bug: Threshold is a step function, and
# Gaussian's own documented +-1 tolerance compounds through
# Median->Sobel->Laplacian (measured stage-by-stage on one image:
# gaussian max_diff=1 (71 px), median max_diff=1 (58 px), sobel
# max_diff=4 (336 px), laplacian max_diff=32 (889 px)) until a handful of
# pixels land on the opposite side of the threshold boundary and flip
# from 0 to 255 or vice versa. A max_abs_diff-based tolerance would be
# meaningless here (the only "safe" value is 255, which asserts nothing).
# The right metric is the *fraction* of pixels that flip: measured across
# 30 real images, worst case was 0.038% of pixels, mean 0.015% -- so a
# 1% threshold is used below, comfortably above the observed worst case
# while still being a meaningful bound that would catch a real bug.


def _run_gpu_pipeline_chain(image, config):
    """Runs all five Basic CUDA filters back-to-back on GpuImage objects
    (no CPU round trip between stages), returning both the final output
    array and every intermediate stage's downloaded array (the latter is
    for diagnostic use only -- see the module docstring above; the
    production path should never download intermediates)."""
    from cuda.gaussian import gaussian_cuda_gpu
    from cuda.laplacian import laplacian_cuda_gpu
    from cuda.median import median_cuda_gpu
    from cuda.sobel import sobel_cuda_gpu

    gpu_image = xray_cuda.upload_image(image)
    g = gaussian_cuda_gpu(gpu_image, kernel_size=config.gaussian_kernel_size, sigma=config.gaussian_sigma)
    m = median_cuda_gpu(g["output"], kernel_size=config.median_kernel_size)
    s = sobel_cuda_gpu(m["output"], mode=config.sobel_mode)
    lap = laplacian_cuda_gpu(
        s["output"], kernel_size=config.laplacian_kernel_size,
        scale=config.laplacian_scale, delta=config.laplacian_delta,
    )
    t = threshold_cuda_gpu(lap["output"], threshold_value=config.threshold_value, max_value=config.threshold_max_value)

    stages = {
        "gaussian": xray_cuda.download_image(g["output"]),
        "median": xray_cuda.download_image(m["output"]),
        "sobel": xray_cuda.download_image(s["output"]),
        "laplacian": xray_cuda.download_image(lap["output"]),
        "threshold": xray_cuda.download_image(t["output"]),
    }
    return stages["threshold"], stages


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_full_basic_cuda_pipeline_gpu_resident_matches_cpu():
    from cpu.filters import apply_gaussian, apply_laplacian, apply_median, apply_sobel

    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=5, seed=42)
    config = FilterConfig()  # all five stages enabled, default parameters

    worst_pct = 0.0
    for path in selection.selected_paths:
        image = load_image(path)

        cpu_final = apply_threshold(
            apply_laplacian(apply_sobel(apply_median(apply_gaussian(image, config), config), config), config),
            config,
        )
        gpu_final, _stages = _run_gpu_pipeline_chain(image, config)

        assert gpu_final.dtype == np.uint8
        assert gpu_final.shape == image.shape
        assert set(np.unique(gpu_final)).issubset({0, 255})  # still a valid binary threshold output

        diff = np.abs(cpu_final.astype(np.int16) - gpu_final.astype(np.int16))
        num_differing = int(np.count_nonzero(diff))
        pct_differing = 100.0 * num_differing / diff.size
        mean_abs_diff = float(diff.mean())
        rmse = float(np.sqrt(np.mean(diff.astype(np.float64) ** 2)))
        worst_pct = max(worst_pct, pct_differing)

        chain_tolerance_pct = 1.0  # empirically justified above; observed worst case was 0.038%
        assert pct_differing <= chain_tolerance_pct, (
            f"{path.name}: {pct_differing:.4f}% of pixels differ (max_abs_diff={diff.max()}, "
            f"mean_abs_diff={mean_abs_diff:.4f}, rmse={rmse:.4f}, {num_differing}/{diff.size}), "
            f"exceeds chain_tolerance={chain_tolerance_pct}%"
        )


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_full_basic_cuda_pipeline_intermediate_stage_validation():
    """Diagnostic-only: downloads every intermediate stage to confirm
    each of the five GPU stages independently stays close to its CPU
    counterpart, so if the final-output test above ever fails, this
    localizes which stage's divergence grew unexpectedly. NOT the
    production path -- production must never download intermediates."""
    from cpu.filters import apply_gaussian, apply_laplacian, apply_median, apply_sobel

    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=1, seed=42)
    image = load_image(selection.selected_paths[0])
    config = FilterConfig()

    cpu_g = apply_gaussian(image, config)
    cpu_m = apply_median(cpu_g, config)
    cpu_s = apply_sobel(cpu_m, config)
    cpu_l = apply_laplacian(cpu_s, config)
    cpu_t = apply_threshold(cpu_l, config)

    _final, gpu_stages = _run_gpu_pipeline_chain(image, config)

    # Generous, documented per-stage bounds (not tight regression pins --
    # this is a smoke test): Gaussian/Median have their own tight
    # standards elsewhere (Section 4B: <=1, Section 4C: ==0 in isolation);
    # here they're chained, and Sobel/Laplacian amplify whatever came in.
    stage_bounds = {"gaussian": 1, "median": 1, "sobel": 10, "laplacian": 60}
    for name, cpu_arr in (("gaussian", cpu_g), ("median", cpu_m), ("sobel", cpu_s), ("laplacian", cpu_l)):
        gpu_arr = gpu_stages[name]
        assert gpu_arr.shape == cpu_arr.shape
        assert gpu_arr.dtype == np.uint8
        diff = np.abs(cpu_arr.astype(np.int16) - gpu_arr.astype(np.int16))
        assert diff.max() <= stage_bounds[name], (
            f"stage={name}: max_abs_diff={diff.max()} exceeds diagnostic bound={stage_bounds[name]}"
        )

    # Threshold: percentage-based, same reasoning as the main chain test.
    diff_t = np.abs(cpu_t.astype(np.int16) - gpu_stages["threshold"].astype(np.int16))
    pct_t = 100.0 * np.count_nonzero(diff_t) / diff_t.size
    assert pct_t <= 1.0


def test_repeated_threshold_no_crash_and_no_leak():
    image = FIXTURES["random_deterministic"]()
    expected = threshold_cuda(image, threshold_value=128)

    free_before = xray_cuda.device_memory_info()["free_bytes"]

    for _ in range(200):
        gpu_image = xray_cuda.upload_image(image)
        result = xray_cuda.threshold_basic_gpu(gpu_image, 128, 255)
        output = xray_cuda.download_image(result["output"])
        np.testing.assert_array_equal(output, expected)
        del gpu_image, result, output

    free_after = xray_cuda.device_memory_info()["free_bytes"]
    leaked_bytes = free_before - free_after
    assert leaked_bytes < 50 * 1024 * 1024, (
        f"Possible GPU memory leak: {leaked_bytes} bytes not reclaimed after 200 iterations "
        f"(free before={free_before}, after={free_after})"
    )
