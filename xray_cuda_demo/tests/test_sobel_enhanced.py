"""Section 8 tests: Enhanced CUDA Sobel edge detection -- four variants
(shared memory, constant memory, compile-time mode specialization,
separable two-pass decomposition), each required to match CPU and Basic
CUDA EXACTLY (max_abs_diff == 0) for all four modes.

--------------------------------------------------------------------------
Correctness standard: exact match, no tolerance -- and why that's expected
--------------------------------------------------------------------------
Every term in Basic Sobel's Gx/Gy arithmetic is an exact-integer float32
value (pixel differences and a single x2 multiply, never a fractional
weight like Gaussian's sigma-derived coefficients) -- so reordering the
sums (as Separable's two-pass decomposition and SharedConst's generic
3x3 loop both do) cannot change the result: exact-integer float32
addition never needs rounding here (magnitudes stay far below 2^24).
Verified directly against CPU/Basic on real data before this suite was
written: all four variants were bit-exact for every mode on the first
try. If any test below needed a nonzero tolerance, that would indicate a
real bug.
"""

from pathlib import Path

import numpy as np
import pytest

xray_cuda = pytest.importorskip(
    "xray_cuda", reason="xray_cuda extension is not built; run cmake --build build first."
)

from cpu.filters import FilterConfig, apply_sobel  # noqa: E402
from cuda.sobel import (  # noqa: E402
    SOBEL_VARIANTS,
    sobel_cuda,
    sobel_enhanced_cuda,
    sobel_enhanced_cuda_gpu,
    sobel_variant_to_int,
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
MODES = ["x", "y", "magnitude", "abs_sum"]


def _assert_exact_match(image, mode="magnitude", variant="shared"):
    config = FilterConfig(sobel_mode=mode, sobel_kernel_size=3)
    cpu_out = apply_sobel(image, config)
    basic_out = sobel_cuda(image, mode=mode)
    enhanced_out = sobel_enhanced_cuda(image, mode=mode, variant=variant)

    assert enhanced_out.dtype == np.uint8
    assert enhanced_out.shape == image.shape

    diff_cpu = np.abs(cpu_out.astype(np.int16) - enhanced_out.astype(np.int16))
    assert diff_cpu.max() == 0, (
        f"variant={variant} mode={mode}: vs CPU max_abs_diff={diff_cpu.max()} (expected exact match)"
    )
    diff_basic = np.abs(basic_out.astype(np.int16) - enhanced_out.astype(np.int16))
    assert diff_basic.max() == 0, (
        f"variant={variant} mode={mode}: vs Basic CUDA max_abs_diff={diff_basic.max()} (expected exact match)"
    )


# -- correctness: all variants x fixtures x modes -------------------------------------------------------


@pytest.mark.parametrize("fixture_name", list(FIXTURES.keys()))
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("variant", SOBEL_VARIANTS)
def test_exact_match_on_synthetic_fixtures(fixture_name, mode, variant):
    image = FIXTURES[fixture_name]()
    _assert_exact_match(image, mode=mode, variant=variant)


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("variant", SOBEL_VARIANTS)
def test_exact_match_on_real_224x224_xray(mode, variant):
    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=1, seed=42)
    image = load_image(selection.selected_paths[0])
    assert image.shape == (224, 224)
    _assert_exact_match(image, mode=mode, variant=variant)


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("variant", SOBEL_VARIANTS)
def test_exact_match_on_large_real_xray(mode, variant):
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
    _assert_exact_match(image, mode=mode, variant=variant)


# -- dimensions (including non-block-multiple, smaller than the halo) -------------------------------------------------------


@pytest.mark.parametrize("height,width", [(3, 3), (7, 7), (16, 16), (31, 47), (224, 224)])
@pytest.mark.parametrize("variant", SOBEL_VARIANTS)
def test_exact_match_various_dimensions(height, width, variant):
    rng = np.random.default_rng(stable_seed(height, width, variant))
    image = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    _assert_exact_match(image, mode="magnitude", variant=variant)


# -- frozen reference regression -------------------------------------------------------


@pytest.mark.parametrize("fixture_name", list(FIXTURES.keys()))
@pytest.mark.parametrize("variant", SOBEL_VARIANTS)
def test_matches_frozen_cpu_reference_output(fixture_name, variant):
    ref_path = REFERENCE_DIR / fixture_name / "sobel.npy"
    if not ref_path.exists():
        pytest.skip(f"No frozen reference at {ref_path}")
    expected = np.load(ref_path)
    image = FIXTURES[fixture_name]()
    gpu_out = sobel_enhanced_cuda(image, mode="magnitude", variant=variant)
    np.testing.assert_array_equal(gpu_out, expected)


# -- border-specific tests -------------------------------------------------------


@pytest.mark.parametrize("edge", ["top", "bottom", "left", "right", "corner"])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("variant", SOBEL_VARIANTS)
def test_border_regions_exact_match(edge, mode, variant):
    size = 16
    rng = np.random.default_rng(stable_seed(edge, mode, variant))
    image = rng.integers(0, 256, size=(size, size), dtype=np.uint8)
    if edge == "top":
        image[0, :] = 255
    elif edge == "bottom":
        image[-1, :] = 255
    elif edge == "left":
        image[:, 0] = 255
    elif edge == "right":
        image[:, -1] = 255
    elif edge == "corner":
        image[0, 0] = 255
        image[0, -1] = 0
        image[-1, 0] = 0
        image[-1, -1] = 255

    _assert_exact_match(image, mode=mode, variant=variant)


# -- determinism -------------------------------------------------------


@pytest.mark.parametrize("variant", SOBEL_VARIANTS)
def test_repeated_execution_is_deterministic(variant):
    image = FIXTURES["random_deterministic"]()
    first = sobel_enhanced_cuda(image, mode="magnitude", variant=variant)
    for _ in range(5):
        again = sobel_enhanced_cuda(image, mode="magnitude", variant=variant)
        np.testing.assert_array_equal(first, again)


# -- Basic-unchanged regression -------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_sobel_basic_output_unchanged_by_enhanced_addition(mode):
    """sobel_basic must remain byte-for-byte identical to its pre-Section-8
    behavior -- Enhanced Sobel is a separate implementation family."""
    image = FIXTURES["random_deterministic"]()
    config = FilterConfig(sobel_mode=mode, sobel_kernel_size=3)
    cpu_out = apply_sobel(image, config)
    basic_out = sobel_cuda(image, mode=mode)
    diff = np.abs(cpu_out.astype(np.int16) - basic_out.astype(np.int16))
    assert diff.max() == 0


# -- batched correctness -------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 8, 32])
@pytest.mark.parametrize("variant", SOBEL_VARIANTS)
def test_batched_exact_match(batch_size, variant):
    config = FilterConfig(sobel_mode="magnitude", sobel_kernel_size=3)
    images = [FIXTURES["random_deterministic"](size=32, seed=i) for i in range(batch_size)]
    batch = np.stack(images, axis=0)

    result = xray_cuda.sobel_enhanced_batch_gpu(batch, 2, sobel_variant_to_int(variant), 16, 16)
    output = result["output"]
    assert output.shape == (batch_size, 32, 32)

    for i, image in enumerate(images):
        cpu_out = apply_sobel(image, config)
        diff = np.abs(cpu_out.astype(np.int16) - output[i].astype(np.int16))
        assert diff.max() == 0, f"batch index {i}: max_abs_diff={diff.max()}"


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
@pytest.mark.parametrize("variant", SOBEL_VARIANTS)
def test_batched_exact_match_real_125_images(variant):
    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=125, seed=42)
    images = [load_image(p) for p in selection.selected_paths if load_image(p).shape == (224, 224)]
    if len(images) < 2:
        pytest.skip("Not enough same-resolution images in this sample.")
    batch = np.stack(images, axis=0)

    config = FilterConfig(sobel_mode="magnitude", sobel_kernel_size=3)
    result = xray_cuda.sobel_enhanced_batch_gpu(batch, 2, sobel_variant_to_int(variant), 16, 16)
    output = result["output"]

    cpu_outputs = np.stack([apply_sobel(img, config) for img in images])
    diff = np.abs(cpu_outputs.astype(np.int16) - output.astype(np.int16))
    assert diff.max() == 0, f"variant={variant}: max_abs_diff={diff.max()} across {len(images)} real images"


def test_batch_ordering_no_cross_contamination():
    """A batch of visibly distinct images must come back in the same
    order, each output depending only on its own input image."""
    images = [
        np.zeros((16, 16), dtype=np.uint8),
        np.full((16, 16), 255, dtype=np.uint8),
        FIXTURES["random_deterministic"](size=16),
    ]
    batch = np.stack(images, axis=0)
    result = xray_cuda.sobel_enhanced_batch_gpu(batch, 2, sobel_variant_to_int("specialized"), 16, 16)
    output = result["output"]

    config = FilterConfig(sobel_mode="magnitude", sobel_kernel_size=3)
    for i, image in enumerate(images):
        expected = apply_sobel(image, config)
        np.testing.assert_array_equal(output[i], expected)


# -- GPU-native API -------------------------------------------------------


@pytest.mark.parametrize("variant", SOBEL_VARIANTS)
def test_sobel_enhanced_cuda_gpu_chains_without_host_round_trip(variant):
    image = FIXTURES["random_deterministic"]()
    gpu_image = xray_cuda.upload_image(image)
    result = sobel_enhanced_cuda_gpu(gpu_image, mode="magnitude", variant=variant)
    assert "output" in result and "kernel_ms" in result
    assert result["kernel_ms"] >= 0.0

    downloaded = xray_cuda.download_image(result["output"])
    expected = sobel_enhanced_cuda(image, mode="magnitude", variant=variant)
    np.testing.assert_array_equal(downloaded, expected)


def test_sobel_enhanced_gpu_input_not_modified():
    image = FIXTURES["random_deterministic"]()
    gpu_image = xray_cuda.upload_image(image)
    before = xray_cuda.download_image(gpu_image)
    xray_cuda.sobel_enhanced_gpu(gpu_image, 2, sobel_variant_to_int("shared"), 16, 16)
    after = xray_cuda.download_image(gpu_image)
    np.testing.assert_array_equal(before, after)


# -- block configuration -------------------------------------------------------


@pytest.mark.parametrize("block", [(8, 8), (16, 16), (32, 8), (32, 16)])
@pytest.mark.parametrize("variant", SOBEL_VARIANTS)
def test_block_configuration_exact_match(block, variant):
    image = FIXTURES["random_deterministic"](size=64)
    config = FilterConfig(sobel_mode="magnitude", sobel_kernel_size=3)
    cpu_out = apply_sobel(image, config)
    out = sobel_enhanced_cuda(image, mode="magnitude", variant=variant, block=block)
    diff = np.abs(cpu_out.astype(np.int16) - out.astype(np.int16))
    assert diff.max() == 0, f"variant={variant} block={block}: max_abs_diff={diff.max()}"


# -- input validation -------------------------------------------------------


def test_sobel_enhanced_cuda_rejects_invalid_mode():
    image = FIXTURES["constant"]()
    with pytest.raises(ValueError):
        sobel_enhanced_cuda(image, mode="not_a_mode", variant="shared")


def test_sobel_enhanced_cuda_rejects_invalid_variant():
    image = FIXTURES["constant"]()
    with pytest.raises(ValueError):
        sobel_enhanced_cuda(image, mode="magnitude", variant="not_a_variant")


def test_sobel_enhanced_gpu_rejects_null_image():
    with pytest.raises(ValueError):
        xray_cuda.sobel_enhanced_gpu(None, 2, 0, 16, 16)


def test_sobel_enhanced_gpu_rejects_out_of_range_variant():
    image = FIXTURES["constant"]()
    gpu_image = xray_cuda.upload_image(image)
    with pytest.raises(ValueError):
        xray_cuda.sobel_enhanced_gpu(gpu_image, 2, 99, 16, 16)


def test_sobel_enhanced_gpu_rejects_out_of_range_mode():
    image = FIXTURES["constant"]()
    gpu_image = xray_cuda.upload_image(image)
    with pytest.raises(ValueError):
        xray_cuda.sobel_enhanced_gpu(gpu_image, 99, 0, 16, 16)


def test_sobel_enhanced_batch_gpu_rejects_wrong_dtype():
    batch = np.zeros((4, 16, 16), dtype=np.float32)
    with pytest.raises(ValueError):
        xray_cuda.sobel_enhanced_batch_gpu(batch, 2, 0, 16, 16)


def test_sobel_enhanced_batch_gpu_rejects_wrong_ndim():
    batch = np.zeros((16, 16), dtype=np.uint8)  # missing batch dim
    with pytest.raises(ValueError):
        xray_cuda.sobel_enhanced_batch_gpu(batch, 2, 0, 16, 16)


# -- memory: repeated allocation/release -------------------------------------------------------


@pytest.mark.parametrize("variant", SOBEL_VARIANTS)
def test_repeated_enhanced_sobel_no_crash_and_no_leak(variant):
    image = FIXTURES["random_deterministic"]()
    expected = sobel_enhanced_cuda(image, mode="magnitude", variant=variant)

    free_before = xray_cuda.device_memory_info()["free_bytes"]

    for _ in range(100):
        out = sobel_enhanced_cuda(image, mode="magnitude", variant=variant)
        np.testing.assert_array_equal(out, expected)

    free_after = xray_cuda.device_memory_info()["free_bytes"]
    leaked_bytes = free_before - free_after
    assert leaked_bytes < 50 * 1024 * 1024, (
        f"variant={variant}: possible GPU memory leak: {leaked_bytes} bytes not reclaimed after 100 calls"
    )


# -- pipeline integration: Enhanced Sobel + Basic Gaussian/Median/Laplacian/Threshold -------------------------------------------------------


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_enhanced_sobel_pipeline_matches_cpu_and_basic_pipeline():
    """cuda.pipeline.run_basic_cuda_pipeline(use_enhanced_sobel=True):
    Gaussian/Median/Laplacian/Threshold stay Basic; only Sobel changes.
    Sobel's arithmetic is all exact-integer, so the default (specialized)
    variant must match the Basic pipeline EXACTLY, not within a
    tolerance."""
    from cpu.filters import apply_gaussian, apply_laplacian, apply_median, apply_sobel, apply_threshold
    from cuda.pipeline import run_basic_cuda_pipeline

    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=10, seed=42)
    images = [load_image(p) for p in selection.selected_paths if load_image(p).shape == (224, 224)]
    if len(images) < 2:
        pytest.skip("Not enough same-resolution images in this sample.")

    config = FilterConfig()
    basic_output, _t1 = run_basic_cuda_pipeline(images, config, use_enhanced_sobel=False)
    enhanced_output, _t2 = run_basic_cuda_pipeline(images, config, use_enhanced_sobel=True)

    diff_vs_basic = np.abs(basic_output.astype(np.int16) - enhanced_output.astype(np.int16))
    assert diff_vs_basic.max() == 0, (
        f"Enhanced-Sobel pipeline differs from Basic pipeline "
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
        f"Enhanced-Sobel pipeline vs CPU: {pct_differing:.4f}% pixels differ, exceeds the "
        f"Section 4F/5-established 1% chain tolerance"
    )


@pytest.mark.parametrize("variant", SOBEL_VARIANTS)
def test_enhanced_sobel_pipeline_all_variants_run_without_crash(variant):
    from cuda.pipeline import run_basic_cuda_pipeline

    images = [FIXTURES["random_deterministic"](size=32, seed=i) for i in range(4)]
    config = FilterConfig()
    output, timing = run_basic_cuda_pipeline(images, config, use_enhanced_sobel=True, sobel_variant=variant)
    assert output.shape == (4, 32, 32)
    assert timing.sobel_ms is not None and timing.sobel_ms >= 0.0


def test_enhanced_sobel_pipeline_disabled_sobel_ignores_enhanced_flag():
    """use_enhanced_sobel=True with sobel_enabled=False must not crash or
    require the stage to run -- the flag is irrelevant when the stage
    itself is off. Gaussian/Median/Laplacian/Threshold remain enabled
    (FilterConfig defaults), so the output is verified against the CPU
    reference rather than assumed."""
    from cpu.filters import apply_gaussian, apply_laplacian, apply_median, apply_threshold
    from cuda.pipeline import run_basic_cuda_pipeline

    images = [FIXTURES["constant"](size=16)]
    config = FilterConfig(sobel_enabled=False)
    output, timing = run_basic_cuda_pipeline(images, config, use_enhanced_sobel=True)
    assert timing.sobel_ms is None

    cpu_expected = apply_threshold(apply_laplacian(apply_median(apply_gaussian(images[0], config), config), config), config)
    np.testing.assert_array_equal(output[0], cpu_expected)


def test_all_three_enhanced_stages_pipeline_combination():
    """Enhanced Gaussian + Enhanced Median + Enhanced Sobel together must
    still match the all-Basic pipeline exactly at default settings
    (sigma=0.0), matching the same bit-exactness each enhancement gives
    individually."""
    from cuda.pipeline import run_basic_cuda_pipeline

    images = [FIXTURES["random_deterministic"](size=64, seed=i) for i in range(3)]
    config = FilterConfig()
    basic_output, _t1 = run_basic_cuda_pipeline(images, config)
    combined_output, _t2 = run_basic_cuda_pipeline(
        images, config, use_enhanced_gaussian=True, use_enhanced_median=True, use_enhanced_sobel=True
    )

    diff = np.abs(basic_output.astype(np.int16) - combined_output.astype(np.int16))
    assert diff.max() == 0, (
        f"Enhanced-Gaussian + Enhanced-Median + Enhanced-Sobel pipeline differs from all-Basic pipeline "
        f"(max_abs_diff={diff.max()}); expected bit-exact at sigma=0.0"
    )
