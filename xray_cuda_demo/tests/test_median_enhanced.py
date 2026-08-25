"""Section 7 tests: Enhanced CUDA median filter -- three variants
(shared memory, branchless sorting network for 3x3, compile-time
specialization for 3/5/7), each required to match CPU and Basic CUDA
EXACTLY (max_abs_diff == 0) -- Median has no floating-point
accumulation, so unlike Gaussian (Section 6) there is no tolerance to
justify here. If any test below needed a nonzero tolerance, that would
indicate a real bug.

The highest-risk new code in this section is the 3x3 branchless sorting
network (Nicolas Devillard's public-domain opt_med9, 19 compare-
exchanges) -- verified directly against CPU/Basic on real data before
this suite was written, and exercised here across every fixture,
dimension, and border case a hand-transcription error could plausibly
have missed.
"""

from pathlib import Path

import numpy as np
import pytest

xray_cuda = pytest.importorskip(
    "xray_cuda", reason="xray_cuda extension is not built; run cmake --build build first."
)

from cpu.filters import FilterConfig, apply_median  # noqa: E402
from cuda.median import (  # noqa: E402
    MEDIAN_VARIANTS,
    median_cuda,
    median_enhanced_cuda,
    median_enhanced_cuda_gpu,
    median_variant_to_int,
)
from pipeline.dataset import DatasetManager  # noqa: E402
from pipeline.image_loader import load_image  # noqa: E402
from tests.conftest import get_real_dataset_path, stable_seed  # noqa: E402
from tests.fixtures import FIXTURES  # noqa: E402

cuda_present = xray_cuda.cuda_available()
pytestmark = pytest.mark.skipif(
    not cuda_present, reason="No usable CUDA device detected on this machine."
)

REFERENCE_DIR = Path(__file__).resolve().parent / "reference"


def _variants_for(kernel_size: int):
    """network3x3 only supports k=3; specialized supports {3,5,7}; shared supports any {3,5,7}."""
    variants = ["shared"]
    if kernel_size == 3:
        variants.append("network3x3")
    if kernel_size in (3, 5, 7):
        variants.append("specialized")
    return variants


def _assert_exact_match(image, kernel_size=3, variant="shared"):
    config = FilterConfig(median_kernel_size=kernel_size)
    cpu_out = apply_median(image, config)
    basic_out = median_cuda(image, kernel_size=kernel_size)
    enhanced_out = median_enhanced_cuda(image, kernel_size=kernel_size, variant=variant)

    assert enhanced_out.dtype == np.uint8
    assert enhanced_out.shape == image.shape

    diff_cpu = np.abs(cpu_out.astype(np.int16) - enhanced_out.astype(np.int16))
    assert diff_cpu.max() == 0, (
        f"variant={variant} k={kernel_size}: vs CPU max_abs_diff={diff_cpu.max()} (expected exact match)"
    )
    diff_basic = np.abs(basic_out.astype(np.int16) - enhanced_out.astype(np.int16))
    assert diff_basic.max() == 0, (
        f"variant={variant} k={kernel_size}: vs Basic CUDA max_abs_diff={diff_basic.max()} (expected exact match)"
    )


# -- correctness: all variants x fixtures x kernel sizes -------------------------------------------------------


@pytest.mark.parametrize("fixture_name", list(FIXTURES.keys()))
@pytest.mark.parametrize("kernel_size", [3, 5, 7])
def test_exact_match_on_synthetic_fixtures(fixture_name, kernel_size):
    image = FIXTURES[fixture_name]()
    for variant in _variants_for(kernel_size):
        _assert_exact_match(image, kernel_size=kernel_size, variant=variant)


# -- real X-rays -------------------------------------------------------


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
@pytest.mark.parametrize("kernel_size", [3, 5, 7])
def test_exact_match_on_real_224x224_xray(kernel_size):
    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=1, seed=42)
    image = load_image(selection.selected_paths[0])
    assert image.shape == (224, 224)
    for variant in _variants_for(kernel_size):
        _assert_exact_match(image, kernel_size=kernel_size, variant=variant)


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
    for variant in _variants_for(3):
        _assert_exact_match(image, kernel_size=3, variant=variant)


# -- dimensions (including non-block-multiple, smaller than kernel) -------------------------------------------------------


@pytest.mark.parametrize("height,width", [(7, 7), (16, 16), (31, 47), (224, 224)])
@pytest.mark.parametrize("kernel_size", [3, 5, 7])
def test_exact_match_various_dimensions(height, width, kernel_size):
    rng = np.random.default_rng(height * 1000 + width + kernel_size)
    image = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    for variant in _variants_for(kernel_size):
        _assert_exact_match(image, kernel_size=kernel_size, variant=variant)


# -- reference-output regression (frozen Section 3 baselines, default k=3) -------------------------------------------------------


@pytest.mark.parametrize("fixture_name", list(FIXTURES.keys()))
@pytest.mark.parametrize("variant", MEDIAN_VARIANTS)
def test_matches_frozen_cpu_reference_output(fixture_name, variant):
    ref_path = REFERENCE_DIR / fixture_name / "median.npy"
    if not ref_path.exists():
        pytest.skip(f"No reference file at {ref_path}")
    image = FIXTURES[fixture_name]()
    expected = np.load(ref_path)
    gpu_out = median_enhanced_cuda(image, kernel_size=3, variant=variant)
    np.testing.assert_array_equal(gpu_out, expected)


# -- border tests: every region, every variant -------------------------------------------------------


@pytest.mark.parametrize(
    "region", ["top_left", "top", "top_right", "left", "right", "bottom_left", "bottom", "bottom_right", "interior"]
)
@pytest.mark.parametrize("kernel_size", [3, 5, 7])
def test_border_regions_exact_match(region, kernel_size):
    size = 16
    rng = np.random.default_rng(stable_seed(region, kernel_size))
    image = rng.integers(0, 256, size=(size, size), dtype=np.uint8)
    mid = size // 2
    if region == "top_left":
        image[0, 0] = 255
    elif region == "top":
        image[0, :] = 255
    elif region == "top_right":
        image[0, -1] = 255
    elif region == "left":
        image[:, 0] = 255
    elif region == "right":
        image[:, -1] = 255
    elif region == "bottom_left":
        image[-1, 0] = 255
    elif region == "bottom":
        image[-1, :] = 255
    elif region == "bottom_right":
        image[-1, -1] = 255
    elif region == "interior":
        image[mid, mid] = 255

    for variant in _variants_for(kernel_size):
        _assert_exact_match(image, kernel_size=kernel_size, variant=variant)


# -- determinism -------------------------------------------------------


@pytest.mark.parametrize("variant", MEDIAN_VARIANTS)
def test_repeated_execution_is_deterministic(variant):
    image = FIXTURES["random_deterministic"]()
    kernel_size = 3
    run_a = median_enhanced_cuda(image, kernel_size=kernel_size, variant=variant)
    run_b = median_enhanced_cuda(image, kernel_size=kernel_size, variant=variant)
    run_c = median_enhanced_cuda(image, kernel_size=kernel_size, variant=variant)
    np.testing.assert_array_equal(run_a, run_b)
    np.testing.assert_array_equal(run_b, run_c)


# -- Basic Median regression: must remain unchanged -------------------------------------------------------


def test_basic_median_unchanged_by_this_section():
    image = FIXTURES["random_deterministic"]()
    config = FilterConfig(median_kernel_size=3)
    cpu_out = apply_median(image, config)
    basic_out = median_cuda(image, kernel_size=3)
    np.testing.assert_array_equal(cpu_out, basic_out)


# -- batched Enhanced Median -------------------------------------------------------


@pytest.mark.parametrize("variant", MEDIAN_VARIANTS)
@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_batch_matches_per_image_single_calls(variant, batch_size):
    rng = np.random.default_rng(stable_seed(variant, batch_size))
    images = [rng.integers(0, 256, size=(32, 32), dtype=np.uint8) for _ in range(batch_size)]
    batch = np.stack(images, axis=0)
    variant_int = median_variant_to_int(variant)

    batch_result = xray_cuda.median_enhanced_batch_gpu(batch, 3, variant_int, 16, 16)
    batch_output = batch_result["output"]
    assert batch_output.shape == batch.shape

    for i, image in enumerate(images):
        single = median_enhanced_cuda(image, kernel_size=3, variant=variant)
        np.testing.assert_array_equal(batch_output[i], single, err_msg=f"variant={variant} image {i} mismatch")


def test_batch_output_order_matches_input_order():
    images = [np.full((16, 16), value, dtype=np.uint8) for value in (10, 60, 110, 160, 210)]
    batch = np.stack(images, axis=0)
    result = xray_cuda.median_enhanced_batch_gpu(batch, 3, median_variant_to_int("network3x3"), 16, 16)
    output = result["output"]
    for i, image in enumerate(images):
        # median of a constant neighborhood is the constant value itself
        np.testing.assert_array_equal(output[i], image, err_msg=f"index {i} does not match input order")


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_batch_125_real_images():
    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=150, seed=42)
    images = [load_image(p) for p in selection.selected_paths if load_image(p).shape == (224, 224)][:125]
    batch = np.stack(images, axis=0)

    config = FilterConfig(median_kernel_size=3)
    for variant in MEDIAN_VARIANTS:
        result = xray_cuda.median_enhanced_batch_gpu(batch, 3, median_variant_to_int(variant), 16, 16)
        output = result["output"]
        for i, image in enumerate(images):
            expected = apply_median(image, config)
            np.testing.assert_array_equal(output[i], expected, err_msg=f"variant={variant} image {i}")


# -- GPU-native API -------------------------------------------------------


def test_median_enhanced_cuda_gpu_chains_without_host_round_trip():
    image = FIXTURES["random_deterministic"]()
    gpu_image = xray_cuda.upload_image(image)

    result = median_enhanced_cuda_gpu(gpu_image, kernel_size=3, variant="network3x3")
    assert "output" in result and "kernel_ms" in result

    gpu_downloaded = xray_cuda.download_image(result["output"])
    host_api_output = median_enhanced_cuda(image, kernel_size=3, variant="network3x3")
    np.testing.assert_array_equal(gpu_downloaded, host_api_output)


def test_median_enhanced_input_not_modified():
    image = FIXTURES["random_deterministic"]()
    gpu_image = xray_cuda.upload_image(image)
    median_enhanced_cuda_gpu(gpu_image, kernel_size=3, variant="shared")
    unchanged = xray_cuda.download_image(gpu_image)
    np.testing.assert_array_equal(unchanged, image)


@pytest.mark.parametrize("block", [(8, 8), (16, 16), (32, 8), (32, 16)])
@pytest.mark.parametrize("variant", MEDIAN_VARIANTS)
def test_block_configurations_all_correct(block, variant):
    image = FIXTURES["random_deterministic"]()
    out = median_enhanced_cuda(image, kernel_size=3, variant=variant, block=block)
    basic_out = median_cuda(image, kernel_size=3)
    np.testing.assert_array_equal(out, basic_out)


# -- input validation -------------------------------------------------------


def test_median_enhanced_rejects_invalid_variant():
    image = FIXTURES["constant"]()
    with pytest.raises(ValueError):
        median_enhanced_cuda(image, kernel_size=3, variant="not_a_variant")


def test_network3x3_rejects_kernel_size_other_than_3():
    image = FIXTURES["constant"]()
    with pytest.raises(ValueError):
        median_enhanced_cuda(image, kernel_size=5, variant="network3x3")


def test_specialized_rejects_unsupported_kernel_size():
    image = FIXTURES["constant"]()
    gpu_image = xray_cuda.upload_image(image)
    with pytest.raises(ValueError):
        xray_cuda.median_enhanced_gpu(gpu_image, 9, median_variant_to_int("specialized"), 16, 16)


def test_median_enhanced_gpu_rejects_null_image():
    with pytest.raises(ValueError):
        xray_cuda.median_enhanced_gpu(None, 3, 0, 16, 16)


def test_median_enhanced_batch_gpu_rejects_wrong_dtype():
    batch = np.zeros((4, 16, 16), dtype=np.float32)
    with pytest.raises(ValueError):
        xray_cuda.median_enhanced_batch_gpu(batch, 3, 0, 16, 16)


# -- memory: repeated allocation/release -------------------------------------------------------


@pytest.mark.parametrize("variant", MEDIAN_VARIANTS)
def test_repeated_enhanced_median_no_crash_and_no_leak(variant):
    image = FIXTURES["random_deterministic"]()
    expected = median_enhanced_cuda(image, kernel_size=3, variant=variant)

    free_before = xray_cuda.device_memory_info()["free_bytes"]

    for _ in range(100):
        out = median_enhanced_cuda(image, kernel_size=3, variant=variant)
        np.testing.assert_array_equal(out, expected)

    free_after = xray_cuda.device_memory_info()["free_bytes"]
    leaked_bytes = free_before - free_after
    assert leaked_bytes < 50 * 1024 * 1024, (
        f"variant={variant}: possible GPU memory leak: {leaked_bytes} bytes not reclaimed after 100 calls"
    )


# -- pipeline integration: Enhanced Median + Basic Gaussian/Sobel/Laplacian/Threshold -------------------------------------------------------


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_enhanced_median_pipeline_matches_cpu_and_basic_pipeline():
    """cuda.pipeline.run_basic_cuda_pipeline(use_enhanced_median=True):
    Gaussian/Sobel/Laplacian/Threshold stay Basic; only Median changes.
    Median has no floating-point accumulation (unlike Gaussian), so the
    default (network3x3) variant must match the Basic pipeline EXACTLY,
    not within a tolerance."""
    from cpu.filters import apply_gaussian, apply_laplacian, apply_median, apply_sobel, apply_threshold
    from cuda.pipeline import run_basic_cuda_pipeline

    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=10, seed=42)
    images = [load_image(p) for p in selection.selected_paths if load_image(p).shape == (224, 224)]
    if len(images) < 2:
        pytest.skip("Not enough same-resolution images in this sample.")

    config = FilterConfig()
    basic_output, _t1 = run_basic_cuda_pipeline(images, config, use_enhanced_median=False)
    enhanced_output, _t2 = run_basic_cuda_pipeline(images, config, use_enhanced_median=True)

    diff_vs_basic = np.abs(basic_output.astype(np.int16) - enhanced_output.astype(np.int16))
    assert diff_vs_basic.max() == 0, (
        f"Enhanced-Median pipeline differs from Basic pipeline "
        f"(max_abs_diff={diff_vs_basic.max()}); expected bit-exact"
    )

    def cpu_pipeline(image):
        return apply_threshold(
            apply_laplacian(apply_sobel(apply_median(apply_gaussian(image, config), config), config), config), config
        )

    cpu_outputs = np.stack([cpu_pipeline(img) for img in images])
    diff_vs_cpu = np.abs(cpu_outputs.astype(np.int16) - enhanced_output.astype(np.int16))
    pct_differing = 100.0 * np.count_nonzero(diff_vs_cpu) / diff_vs_cpu.size
    assert pct_differing <= 1.0, (
        f"Enhanced-Median pipeline vs CPU: {pct_differing:.4f}% pixels differ, exceeds the "
        f"Section 4F/5-established 1% chain tolerance"
    )


@pytest.mark.parametrize("variant", MEDIAN_VARIANTS)
def test_enhanced_median_pipeline_all_variants_run_without_crash(variant):
    from cuda.pipeline import run_basic_cuda_pipeline

    images = [FIXTURES["random_deterministic"](size=32, seed=i) for i in range(4)]
    config = FilterConfig()
    output, timing = run_basic_cuda_pipeline(images, config, use_enhanced_median=True, median_variant=variant)
    assert output.shape == (4, 32, 32)
    assert timing.median_ms is not None and timing.median_ms >= 0.0


def test_enhanced_median_pipeline_disabled_median_ignores_enhanced_flag():
    """use_enhanced_median=True with median_enabled=False must not crash
    or require the stage to run -- the flag is irrelevant when the stage
    itself is off. Gaussian/Sobel/Laplacian/Threshold remain enabled
    (FilterConfig defaults), so the output is verified against the CPU
    reference rather than assumed."""
    from cpu.filters import apply_gaussian, apply_laplacian, apply_sobel, apply_threshold
    from cuda.pipeline import run_basic_cuda_pipeline

    images = [FIXTURES["constant"](size=16)]
    config = FilterConfig(median_enabled=False)
    output, timing = run_basic_cuda_pipeline(images, config, use_enhanced_median=True)
    assert timing.median_ms is None

    cpu_expected = apply_threshold(apply_laplacian(apply_sobel(apply_gaussian(images[0], config), config), config), config)
    np.testing.assert_array_equal(output[0], cpu_expected)


def test_enhanced_gaussian_and_enhanced_median_pipeline_combination():
    """Both Section 6 and Section 7 enhancements enabled together must
    still match the all-Basic pipeline exactly at default settings
    (sigma=0.0), matching the same bit-exactness each enhancement gives
    individually."""
    from cuda.pipeline import run_basic_cuda_pipeline

    images = [FIXTURES["random_deterministic"](size=64, seed=i) for i in range(3)]
    config = FilterConfig()
    basic_output, _t1 = run_basic_cuda_pipeline(images, config)
    combined_output, _t2 = run_basic_cuda_pipeline(
        images, config, use_enhanced_gaussian=True, use_enhanced_median=True
    )

    diff = np.abs(basic_output.astype(np.int16) - combined_output.astype(np.int16))
    assert diff.max() == 0, (
        f"Enhanced-Gaussian + Enhanced-Median pipeline differs from all-Basic pipeline "
        f"(max_abs_diff={diff.max()}); expected bit-exact at sigma=0.0"
    )
