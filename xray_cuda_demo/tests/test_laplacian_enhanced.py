"""Section 9 tests: Enhanced CUDA Laplacian filter -- four variants
(shared memory, constant memory, compile-time matrix-size
specialization, hand-written explicit arithmetic), each required to
match CPU and Basic CUDA EXACTLY (max_abs_diff == 0) for every supported
kernel_size/scale/delta combination.

--------------------------------------------------------------------------
Correctness standard: exact match, no tolerance -- and why that's expected
--------------------------------------------------------------------------
Every Laplacian coefficient (2, 4, -8, -24, ...) is a small exact
integer -- no fractional weights like Gaussian's sigma-derived
coefficients -- so reordering the sums (SharedConst/Specialized's
constant-memory loop, Explicit's hand-grouped sums that skip
zero-coefficient terms entirely) cannot change the result: exact-integer
float32 addition never needs rounding here. Verified directly against
CPU/Basic on real data before this suite was written: all four variants
were bit-exact for every kernel_size/scale/delta combination on the
first try. If any test below needed a nonzero tolerance, that would
indicate a real bug.

--------------------------------------------------------------------------
Why kernel_size=1 gets its own dedicated regression test
--------------------------------------------------------------------------
Section 4E discovered (see cuda/laplacian.py's module docstring) that
OpenCV's kernel_size=1 is a special case: it recovers a 3x3 coefficient
matrix (the textbook [[0,1,0],[1,-4,1],[0,1,0]] Laplacian), NOT a
literal 1x1 aperture -- and it is genuinely different from kernel_size=3
(which recovers a different 3x3 matrix, [[2,0,2],[0,-8,0],[2,0,2]]).
This suite never re-derives that coefficient matrix -- it always goes
through the already-verified cuda.laplacian.laplacian_kernel_2d()
impulse-response mechanism, per Section 9 spec item 11.
"""

from pathlib import Path

import numpy as np
import pytest

xray_cuda = pytest.importorskip(
    "xray_cuda", reason="xray_cuda extension is not built; run cmake --build build first."
)

from cpu.filters import FilterConfig, apply_laplacian  # noqa: E402
from cuda.laplacian import (  # noqa: E402
    LAPLACIAN_VARIANTS,
    laplacian_cuda,
    laplacian_enhanced_cuda,
    laplacian_enhanced_cuda_gpu,
    laplacian_kernel_2d,
    laplacian_variant_to_int,
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
KERNEL_SIZES = [1, 3, 5]


def _assert_exact_match(image, kernel_size=3, scale=1.0, delta=0.0, variant="shared"):
    config = FilterConfig(laplacian_kernel_size=kernel_size, laplacian_scale=scale, laplacian_delta=delta)
    cpu_out = apply_laplacian(image, config)
    basic_out = laplacian_cuda(image, kernel_size=kernel_size, scale=scale, delta=delta)
    enhanced_out = laplacian_enhanced_cuda(image, kernel_size=kernel_size, scale=scale, delta=delta, variant=variant)

    assert enhanced_out.dtype == np.uint8
    assert enhanced_out.shape == image.shape

    diff_cpu = np.abs(cpu_out.astype(np.int16) - enhanced_out.astype(np.int16))
    assert diff_cpu.max() == 0, (
        f"variant={variant} kernel={kernel_size} scale={scale} delta={delta}: "
        f"vs CPU max_abs_diff={diff_cpu.max()} (expected exact match)"
    )
    diff_basic = np.abs(basic_out.astype(np.int16) - enhanced_out.astype(np.int16))
    assert diff_basic.max() == 0, (
        f"variant={variant} kernel={kernel_size} scale={scale} delta={delta}: "
        f"vs Basic CUDA max_abs_diff={diff_basic.max()} (expected exact match)"
    )


# -- correctness: all variants x fixtures x kernel sizes -------------------------------------------------------


@pytest.mark.parametrize("fixture_name", list(FIXTURES.keys()))
@pytest.mark.parametrize("kernel_size", KERNEL_SIZES)
@pytest.mark.parametrize("variant", LAPLACIAN_VARIANTS)
def test_exact_match_on_synthetic_fixtures(fixture_name, kernel_size, variant):
    image = FIXTURES[fixture_name]()
    _assert_exact_match(image, kernel_size=kernel_size, variant=variant)


# -- scale/delta combinations -------------------------------------------------------


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("delta", [0.0, 5.0, 10.0])
@pytest.mark.parametrize("kernel_size", KERNEL_SIZES)
@pytest.mark.parametrize("variant", LAPLACIAN_VARIANTS)
def test_exact_match_scale_delta_variants(scale, delta, kernel_size, variant):
    image = FIXTURES["random_deterministic"]()
    _assert_exact_match(image, kernel_size=kernel_size, scale=scale, delta=delta, variant=variant)


# -- dedicated kernel_size=1 regression (Section 4E's discovered special case) -------------------------------------------------------


@pytest.mark.parametrize("variant", LAPLACIAN_VARIANTS)
def test_kernel_size_1_matches_textbook_laplacian_via_verified_mechanism(variant):
    """kernel_size=1 must recover the textbook [[0,1,0],[1,-4,1],[0,1,0]]
    matrix through the SAME verified impulse-response mechanism used for
    kernel_size=3/5 -- not a hand re-derivation (Section 9 spec item 11).
    """
    coeffs = laplacian_kernel_2d(1)
    expected = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    np.testing.assert_array_equal(coeffs, expected)

    image = FIXTURES["random_deterministic"]()
    _assert_exact_match(image, kernel_size=1, variant=variant)


@pytest.mark.parametrize("variant", LAPLACIAN_VARIANTS)
def test_kernel_size_1_and_3_are_genuinely_different_matrices(variant):
    """kernel_size=1 and kernel_size=3 both recover a 3x3 matrix but with
    DIFFERENT values -- Enhanced must not conflate them."""
    out_1 = laplacian_enhanced_cuda(FIXTURES["sharp_edge_square"](), kernel_size=1, variant=variant)
    out_3 = laplacian_enhanced_cuda(FIXTURES["sharp_edge_square"](), kernel_size=3, variant=variant)
    assert not np.array_equal(out_1, out_3), (
        f"variant={variant}: kernel_size=1 and kernel_size=3 produced identical output; "
        "expected genuinely different coefficient matrices"
    )


# -- real X-rays -------------------------------------------------------


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
@pytest.mark.parametrize("kernel_size", KERNEL_SIZES)
@pytest.mark.parametrize("variant", LAPLACIAN_VARIANTS)
def test_exact_match_on_real_224x224_xray(kernel_size, variant):
    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=1, seed=42)
    image = load_image(selection.selected_paths[0])
    assert image.shape == (224, 224)
    _assert_exact_match(image, kernel_size=kernel_size, variant=variant)


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
@pytest.mark.parametrize("kernel_size", KERNEL_SIZES)
@pytest.mark.parametrize("variant", LAPLACIAN_VARIANTS)
def test_exact_match_on_large_real_xray(kernel_size, variant):
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
    _assert_exact_match(image, kernel_size=kernel_size, variant=variant)


# -- dimensions -------------------------------------------------------


@pytest.mark.parametrize("height,width", [(7, 7), (16, 16), (31, 47), (224, 224)])
@pytest.mark.parametrize("variant", LAPLACIAN_VARIANTS)
def test_exact_match_various_dimensions(height, width, variant):
    rng = np.random.default_rng(stable_seed(height, width, variant))
    image = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    _assert_exact_match(image, kernel_size=3, variant=variant)


# -- frozen reference regression -------------------------------------------------------


@pytest.mark.parametrize("fixture_name", list(FIXTURES.keys()))
@pytest.mark.parametrize("variant", LAPLACIAN_VARIANTS)
def test_matches_frozen_cpu_reference_output(fixture_name, variant):
    ref_path = REFERENCE_DIR / fixture_name / "laplacian.npy"
    if not ref_path.exists():
        pytest.skip(f"No frozen reference at {ref_path}")
    expected = np.load(ref_path)
    image = FIXTURES[fixture_name]()
    gpu_out = laplacian_enhanced_cuda(image, kernel_size=3, scale=1.0, delta=0.0, variant=variant)
    np.testing.assert_array_equal(gpu_out, expected)


# -- border-specific tests -------------------------------------------------------


@pytest.mark.parametrize("region", ["top", "bottom", "left", "right", "corner", "interior"])
@pytest.mark.parametrize("kernel_size", KERNEL_SIZES)
@pytest.mark.parametrize("variant", LAPLACIAN_VARIANTS)
def test_border_regions_exact_match(region, kernel_size, variant):
    size = 16
    rng = np.random.default_rng(stable_seed(region, kernel_size, variant))
    image = rng.integers(0, 256, size=(size, size), dtype=np.uint8)
    if region == "top":
        image[0, :] = 255
    elif region == "bottom":
        image[-1, :] = 255
    elif region == "left":
        image[:, 0] = 255
    elif region == "right":
        image[:, -1] = 255
    elif region == "corner":
        image[0, 0] = 255
        image[0, -1] = 0
        image[-1, 0] = 0
        image[-1, -1] = 255
    elif region == "interior":
        image[size // 2, size // 2] = 255

    _assert_exact_match(image, kernel_size=kernel_size, variant=variant)


# -- determinism -------------------------------------------------------


@pytest.mark.parametrize("variant", LAPLACIAN_VARIANTS)
def test_repeated_execution_is_deterministic(variant):
    image = FIXTURES["random_deterministic"]()
    first = laplacian_enhanced_cuda(image, kernel_size=3, variant=variant)
    for _ in range(5):
        again = laplacian_enhanced_cuda(image, kernel_size=3, variant=variant)
        np.testing.assert_array_equal(first, again)


# -- Basic-unchanged regression -------------------------------------------------------


@pytest.mark.parametrize("kernel_size", KERNEL_SIZES)
def test_laplacian_basic_output_unchanged_by_enhanced_addition(kernel_size):
    """laplacian_basic must remain byte-for-byte identical to its
    pre-Section-9 behavior -- Enhanced Laplacian is a separate
    implementation family."""
    image = FIXTURES["random_deterministic"]()
    config = FilterConfig(laplacian_kernel_size=kernel_size)
    cpu_out = apply_laplacian(image, config)
    basic_out = laplacian_cuda(image, kernel_size=kernel_size)
    diff = np.abs(cpu_out.astype(np.int16) - basic_out.astype(np.int16))
    assert diff.max() == 0


# -- batched correctness -------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 8, 32])
@pytest.mark.parametrize("variant", LAPLACIAN_VARIANTS)
def test_batched_exact_match(batch_size, variant):
    config = FilterConfig(laplacian_kernel_size=3)
    coeffs = laplacian_kernel_2d(3)
    images = [FIXTURES["random_deterministic"](size=32, seed=i) for i in range(batch_size)]
    batch = np.stack(images, axis=0)

    result = xray_cuda.laplacian_enhanced_batch_gpu(batch, coeffs, 1.0, 0.0, laplacian_variant_to_int(variant), 16, 16)
    output = result["output"]
    assert output.shape == (batch_size, 32, 32)

    for i, image in enumerate(images):
        cpu_out = apply_laplacian(image, config)
        diff = np.abs(cpu_out.astype(np.int16) - output[i].astype(np.int16))
        assert diff.max() == 0, f"batch index {i}: max_abs_diff={diff.max()}"


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
@pytest.mark.parametrize("variant", LAPLACIAN_VARIANTS)
def test_batched_exact_match_real_125_images(variant):
    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=125, seed=42)
    images = [load_image(p) for p in selection.selected_paths if load_image(p).shape == (224, 224)]
    if len(images) < 2:
        pytest.skip("Not enough same-resolution images in this sample.")
    batch = np.stack(images, axis=0)

    config = FilterConfig(laplacian_kernel_size=3)
    coeffs = laplacian_kernel_2d(3)
    result = xray_cuda.laplacian_enhanced_batch_gpu(batch, coeffs, 1.0, 0.0, laplacian_variant_to_int(variant), 16, 16)
    output = result["output"]

    cpu_outputs = np.stack([apply_laplacian(img, config) for img in images])
    diff = np.abs(cpu_outputs.astype(np.int16) - output.astype(np.int16))
    assert diff.max() == 0, f"variant={variant}: max_abs_diff={diff.max()} across {len(images)} real images"


@pytest.mark.parametrize("batch_size", [256])
def test_batched_exact_match_large_batch(batch_size):
    config = FilterConfig(laplacian_kernel_size=3)
    coeffs = laplacian_kernel_2d(3)
    images = [FIXTURES["random_deterministic"](size=32, seed=i) for i in range(batch_size)]
    batch = np.stack(images, axis=0)
    result = xray_cuda.laplacian_enhanced_batch_gpu(batch, coeffs, 1.0, 0.0, laplacian_variant_to_int("specialized"), 16, 16)
    output = result["output"]
    assert output.shape == (batch_size, 32, 32)
    for i in range(0, batch_size, 32):  # sample every 32nd image to keep runtime reasonable
        cpu_out = apply_laplacian(images[i], config)
        diff = np.abs(cpu_out.astype(np.int16) - output[i].astype(np.int16))
        assert diff.max() == 0, f"batch index {i}: max_abs_diff={diff.max()}"


def test_batch_ordering_no_cross_contamination():
    """A batch of visibly distinct images must come back in the same
    order, each output depending only on its own input image."""
    images = [
        np.zeros((16, 16), dtype=np.uint8),
        np.full((16, 16), 255, dtype=np.uint8),
        FIXTURES["random_deterministic"](size=16),
    ]
    batch = np.stack(images, axis=0)
    coeffs = laplacian_kernel_2d(3)
    result = xray_cuda.laplacian_enhanced_batch_gpu(batch, coeffs, 1.0, 0.0, laplacian_variant_to_int("specialized"), 16, 16)
    output = result["output"]

    config = FilterConfig(laplacian_kernel_size=3)
    for i, image in enumerate(images):
        expected = apply_laplacian(image, config)
        np.testing.assert_array_equal(output[i], expected)


# -- GPU-native API -------------------------------------------------------


@pytest.mark.parametrize("variant", LAPLACIAN_VARIANTS)
def test_laplacian_enhanced_cuda_gpu_chains_without_host_round_trip(variant):
    image = FIXTURES["random_deterministic"]()
    gpu_image = xray_cuda.upload_image(image)
    result = laplacian_enhanced_cuda_gpu(gpu_image, kernel_size=3, variant=variant)
    assert "output" in result and "kernel_ms" in result
    assert result["kernel_ms"] >= 0.0

    downloaded = xray_cuda.download_image(result["output"])
    expected = laplacian_enhanced_cuda(image, kernel_size=3, variant=variant)
    np.testing.assert_array_equal(downloaded, expected)


def test_laplacian_enhanced_gpu_input_not_modified():
    image = FIXTURES["random_deterministic"]()
    gpu_image = xray_cuda.upload_image(image)
    before = xray_cuda.download_image(gpu_image)
    coeffs = laplacian_kernel_2d(3)
    xray_cuda.laplacian_enhanced_gpu(gpu_image, coeffs, 1.0, 0.0, laplacian_variant_to_int("shared"), 16, 16)
    after = xray_cuda.download_image(gpu_image)
    np.testing.assert_array_equal(before, after)


# -- block configuration -------------------------------------------------------


@pytest.mark.parametrize("block", [(8, 8), (16, 16), (32, 8), (32, 16)])
@pytest.mark.parametrize("variant", LAPLACIAN_VARIANTS)
def test_block_configuration_exact_match(block, variant):
    image = FIXTURES["random_deterministic"](size=64)
    config = FilterConfig(laplacian_kernel_size=3)
    cpu_out = apply_laplacian(image, config)
    out = laplacian_enhanced_cuda(image, kernel_size=3, variant=variant, block=block)
    diff = np.abs(cpu_out.astype(np.int16) - out.astype(np.int16))
    assert diff.max() == 0, f"variant={variant} block={block}: max_abs_diff={diff.max()}"


# -- input validation -------------------------------------------------------


def test_laplacian_enhanced_cuda_rejects_invalid_kernel_size():
    image = FIXTURES["constant"]()
    with pytest.raises(ValueError):
        laplacian_enhanced_cuda(image, kernel_size=4, variant="shared")


def test_laplacian_enhanced_cuda_rejects_invalid_variant():
    image = FIXTURES["constant"]()
    with pytest.raises(ValueError):
        laplacian_enhanced_cuda(image, kernel_size=3, variant="not_a_variant")


def test_specialized_rejects_kernel_size_not_in_3_5():
    """A literal 1x1 coefficient array (not OpenCV's kernel_size=1, which
    always recovers a 3x3 matrix -- see module docstring) has an actual
    matrix size of 1, which Specialized does not support."""
    image = FIXTURES["constant"]()
    gpu_image = xray_cuda.upload_image(image)
    bad_coeffs = np.array([[1.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        xray_cuda.laplacian_enhanced_gpu(gpu_image, bad_coeffs, 1.0, 0.0, laplacian_variant_to_int("specialized"), 16, 16)


def test_explicit_rejects_unrecognized_coefficients():
    """Explicit must throw, not silently misbehave, when given a
    coefficient array that matches none of the three known verified
    patterns -- e.g. an arbitrary 3x3 array."""
    image = FIXTURES["constant"]()
    gpu_image = xray_cuda.upload_image(image)
    bad_coeffs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    with pytest.raises(ValueError):
        xray_cuda.laplacian_enhanced_gpu(gpu_image, bad_coeffs, 1.0, 0.0, laplacian_variant_to_int("explicit"), 16, 16)


def test_laplacian_enhanced_gpu_rejects_null_image():
    coeffs = laplacian_kernel_2d(3)
    with pytest.raises(ValueError):
        xray_cuda.laplacian_enhanced_gpu(None, coeffs, 1.0, 0.0, 0, 16, 16)


def test_laplacian_enhanced_gpu_rejects_non_square_coeffs():
    image = FIXTURES["constant"]()
    gpu_image = xray_cuda.upload_image(image)
    bad_coeffs = np.zeros((3, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        xray_cuda.laplacian_enhanced_gpu(gpu_image, bad_coeffs, 1.0, 0.0, 0, 16, 16)


def test_laplacian_enhanced_batch_gpu_rejects_wrong_dtype():
    batch = np.zeros((4, 16, 16), dtype=np.float32)
    coeffs = laplacian_kernel_2d(3)
    with pytest.raises(ValueError):
        xray_cuda.laplacian_enhanced_batch_gpu(batch, coeffs, 1.0, 0.0, 0, 16, 16)


def test_laplacian_enhanced_batch_gpu_rejects_wrong_ndim():
    batch = np.zeros((16, 16), dtype=np.uint8)  # missing batch dim
    coeffs = laplacian_kernel_2d(3)
    with pytest.raises(ValueError):
        xray_cuda.laplacian_enhanced_batch_gpu(batch, coeffs, 1.0, 0.0, 0, 16, 16)


# -- memory: repeated allocation/release -------------------------------------------------------


@pytest.mark.parametrize("variant", LAPLACIAN_VARIANTS)
def test_repeated_enhanced_laplacian_no_crash_and_no_leak(variant):
    image = FIXTURES["random_deterministic"]()
    expected = laplacian_enhanced_cuda(image, kernel_size=3, variant=variant)

    free_before = xray_cuda.device_memory_info()["free_bytes"]

    for _ in range(100):
        out = laplacian_enhanced_cuda(image, kernel_size=3, variant=variant)
        np.testing.assert_array_equal(out, expected)

    free_after = xray_cuda.device_memory_info()["free_bytes"]
    leaked_bytes = free_before - free_after
    assert leaked_bytes < 50 * 1024 * 1024, (
        f"variant={variant}: possible GPU memory leak: {leaked_bytes} bytes not reclaimed after 100 calls"
    )


# -- pipeline integration: Enhanced Laplacian + Enhanced Gaussian/Median/Sobel -------------------------------------------------------


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_enhanced_laplacian_pipeline_matches_cpu_and_basic_pipeline():
    """cuda.pipeline.run_basic_cuda_pipeline(use_enhanced_laplacian=True):
    Gaussian/Median/Sobel/Threshold stay Basic; only Laplacian changes.
    Laplacian's arithmetic is all exact-integer, so the default
    (specialized) variant must match the Basic pipeline EXACTLY, not
    within a tolerance."""
    from cpu.filters import apply_gaussian, apply_laplacian, apply_median, apply_sobel, apply_threshold
    from cuda.pipeline import run_basic_cuda_pipeline

    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=10, seed=42)
    images = [load_image(p) for p in selection.selected_paths if load_image(p).shape == (224, 224)]
    if len(images) < 2:
        pytest.skip("Not enough same-resolution images in this sample.")

    config = FilterConfig()
    basic_output, _t1 = run_basic_cuda_pipeline(images, config, use_enhanced_laplacian=False)
    enhanced_output, _t2 = run_basic_cuda_pipeline(images, config, use_enhanced_laplacian=True)

    diff_vs_basic = np.abs(basic_output.astype(np.int16) - enhanced_output.astype(np.int16))
    assert diff_vs_basic.max() == 0, (
        f"Enhanced-Laplacian pipeline differs from Basic pipeline "
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
        f"Enhanced-Laplacian pipeline vs CPU: {pct_differing:.4f}% pixels differ, exceeds the "
        f"Section 4F/5-established 1% chain tolerance"
    )


@pytest.mark.parametrize("variant", LAPLACIAN_VARIANTS)
def test_enhanced_laplacian_pipeline_all_variants_run_without_crash(variant):
    from cuda.pipeline import run_basic_cuda_pipeline

    images = [FIXTURES["random_deterministic"](size=32, seed=i) for i in range(4)]
    config = FilterConfig()
    output, timing = run_basic_cuda_pipeline(images, config, use_enhanced_laplacian=True, laplacian_variant=variant)
    assert output.shape == (4, 32, 32)
    assert timing.laplacian_ms is not None and timing.laplacian_ms >= 0.0


def test_enhanced_laplacian_pipeline_disabled_laplacian_ignores_enhanced_flag():
    """use_enhanced_laplacian=True with laplacian_enabled=False must not
    crash or require the stage to run -- the flag is irrelevant when the
    stage itself is off. Gaussian/Median/Sobel/Threshold remain enabled
    (FilterConfig defaults), so the output is verified against the CPU
    reference rather than assumed."""
    from cpu.filters import apply_gaussian, apply_median, apply_sobel, apply_threshold
    from cuda.pipeline import run_basic_cuda_pipeline

    images = [FIXTURES["constant"](size=16)]
    config = FilterConfig(laplacian_enabled=False)
    output, timing = run_basic_cuda_pipeline(images, config, use_enhanced_laplacian=True)
    assert timing.laplacian_ms is None

    cpu_expected = apply_threshold(apply_sobel(apply_median(apply_gaussian(images[0], config), config), config), config)
    np.testing.assert_array_equal(output[0], cpu_expected)


def test_all_four_enhanced_stages_pipeline_combination():
    """Enhanced Gaussian + Enhanced Median + Enhanced Sobel + Enhanced
    Laplacian together must still match the all-Basic pipeline exactly
    at default settings (sigma=0.0), matching the same bit-exactness
    each enhancement gives individually. This is the first pipeline
    where four of five filters are Enhanced."""
    from cuda.pipeline import run_basic_cuda_pipeline

    images = [FIXTURES["random_deterministic"](size=64, seed=i) for i in range(3)]
    config = FilterConfig()
    basic_output, _t1 = run_basic_cuda_pipeline(images, config)
    combined_output, _t2 = run_basic_cuda_pipeline(
        images, config,
        use_enhanced_gaussian=True, use_enhanced_median=True,
        use_enhanced_sobel=True, use_enhanced_laplacian=True,
    )

    diff = np.abs(basic_output.astype(np.int16) - combined_output.astype(np.int16))
    assert diff.max() == 0, (
        f"Enhanced-Gaussian + Enhanced-Median + Enhanced-Sobel + Enhanced-Laplacian pipeline differs "
        f"from all-Basic pipeline (max_abs_diff={diff.max()}); expected bit-exact at sigma=0.0"
    )
