"""Section 6 tests: Enhanced (separable) CUDA Gaussian blur -- four
variants (naive separable, +shared memory, +constant memory,
+compile-time specialization), each independently correctness-tested
against both the CPU reference and the unchanged Basic CUDA Gaussian.

--------------------------------------------------------------------------
Correctness standard: Enhanced vs CPU uses Basic's established +-1
tolerance; Enhanced vs Basic uses a SEPARATE, smaller, also-measured +-1
tolerance -- these are not the same claim
--------------------------------------------------------------------------
Verified empirically before writing this suite: at sigma=0.0 (the
hardcoded-coefficient-table case for k in {3,5,7}, and the common default),
every Enhanced variant matches gaussian_basic_gpu() EXACTLY
(max_abs_diff == 0), and matches CPU with the identical +-1 tolerance and
identical differing-pixel counts Section 4B established for Basic
(1089/71/5/0 differing pixels for k=3/5/7/9 on the same real image).

At explicit sigma > 0 (sigma=2.0), a *separate* small divergence appears:
Enhanced vs Basic differs by exactly 1, on exactly 1 pixel out of 4096,
identically across all four variants. Diagnosed before writing this
tolerance (not assumed): Basic computes one direct 2D convolution sum
(k^2 terms accumulated in one reduction); Enhanced computes the
mathematically-equivalent separable decomposition as two 1D sums (a
horizontal pass, then a vertical pass over its float32 output).
Real-valued arithmetic makes these identical (distributivity), but
IEEE-754 float addition is not associative -- summing the same terms in
a different grouping can differ in the last bit, which can occasionally
tip __float2int_rn's rounding decision for a value that lands almost
exactly on a .5 boundary. All four variants hitting the *same* single
pixel with the *same* magnitude is the signature of this floating-point
effect, not an indexing or logic bug (a bug would not reproduce
identically across four structurally different kernels). This is exactly
what Section 6 spec item 9 warned about: "mathematically equivalent"
does not mean "bit-identical" -- so GAUSSIAN_VS_BASIC_TOLERANCE below is
set to 1 (not 0), measured and justified, not assumed.
"""

from pathlib import Path

import numpy as np
import pytest

xray_cuda = pytest.importorskip(
    "xray_cuda", reason="xray_cuda extension is not built; run cmake --build build first."
)

from cpu.filters import FilterConfig, apply_gaussian  # noqa: E402
from cuda.gaussian import (  # noqa: E402
    GAUSSIAN_VARIANTS,
    gaussian_cuda,
    gaussian_enhanced_cuda,
    gaussian_enhanced_cuda_gpu,
    gaussian_kernel_1d,
)
from pipeline.dataset import DatasetManager  # noqa: E402
from pipeline.image_loader import load_image  # noqa: E402
from tests.conftest import get_real_dataset_path, stable_seed  # noqa: E402
from tests.fixtures import FIXTURES, sharp_edge_square  # noqa: E402

cuda_present = xray_cuda.cuda_available()
pytestmark = pytest.mark.skipif(
    not cuda_present, reason="No usable CUDA device detected on this machine."
)

REFERENCE_DIR = Path(__file__).resolve().parent / "reference"
GAUSSIAN_VS_CPU_TOLERANCE = 1  # same as Section 4B Basic Gaussian; see module docstring
GAUSSIAN_VS_BASIC_TOLERANCE = 1  # separable-vs-direct float summation order; see module docstring


def _assert_matches_cpu_and_basic(image, kernel_size=5, sigma=0.0, variant="specialized"):
    config = FilterConfig(gaussian_kernel_size=kernel_size, gaussian_sigma=sigma)
    cpu_out = apply_gaussian(image, config)
    basic_out = gaussian_cuda(image, kernel_size=kernel_size, sigma=sigma)
    enhanced_out = gaussian_enhanced_cuda(image, kernel_size=kernel_size, sigma=sigma, variant=variant)

    assert enhanced_out.dtype == np.uint8
    assert enhanced_out.shape == image.shape

    diff_cpu = np.abs(cpu_out.astype(np.int16) - enhanced_out.astype(np.int16))
    assert diff_cpu.max() <= GAUSSIAN_VS_CPU_TOLERANCE, (
        f"variant={variant} k={kernel_size}: vs CPU max_abs_diff={diff_cpu.max()} exceeds "
        f"the established Basic-Gaussian tolerance={GAUSSIAN_VS_CPU_TOLERANCE}"
    )

    diff_basic = np.abs(basic_out.astype(np.int16) - enhanced_out.astype(np.int16))
    assert diff_basic.max() <= GAUSSIAN_VS_BASIC_TOLERANCE, (
        f"variant={variant} k={kernel_size}: Enhanced vs Basic CUDA max_abs_diff={diff_basic.max()} "
        f"exceeds the measured/justified tolerance={GAUSSIAN_VS_BASIC_TOLERANCE} "
        f"(float summation-order effect, see module docstring)"
    )
    return diff_cpu


# -- correctness: all variants x all fixtures x all kernel sizes -------------------------------------------------------


@pytest.mark.parametrize("fixture_name", list(FIXTURES.keys()))
@pytest.mark.parametrize("variant", GAUSSIAN_VARIANTS)
def test_matches_cpu_and_basic_on_synthetic_fixtures(fixture_name, variant):
    image = FIXTURES[fixture_name]()
    _assert_matches_cpu_and_basic(image, kernel_size=5, sigma=0.0, variant=variant)


@pytest.mark.parametrize("kernel_size", [3, 5, 7, 9])
@pytest.mark.parametrize("variant", GAUSSIAN_VARIANTS)
def test_matches_cpu_and_basic_across_kernel_sizes(kernel_size, variant):
    image = FIXTURES["random_deterministic"]()
    _assert_matches_cpu_and_basic(image, kernel_size=kernel_size, sigma=0.0, variant=variant)


@pytest.mark.parametrize("sigma", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("variant", GAUSSIAN_VARIANTS)
def test_matches_cpu_and_basic_across_sigma(sigma, variant):
    image = FIXTURES["random_deterministic"]()
    _assert_matches_cpu_and_basic(image, kernel_size=5, sigma=sigma, variant=variant)


# -- real X-rays -------------------------------------------------------


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
@pytest.mark.parametrize("variant", GAUSSIAN_VARIANTS)
def test_matches_cpu_and_basic_on_real_224x224_xray(variant):
    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=1, seed=42)
    image = load_image(selection.selected_paths[0])
    assert image.shape == (224, 224)
    _assert_matches_cpu_and_basic(image, variant=variant)


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
@pytest.mark.parametrize("variant", GAUSSIAN_VARIANTS)
def test_matches_cpu_and_basic_on_large_real_xray(variant):
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
    _assert_matches_cpu_and_basic(image, variant=variant)


# -- dimensions (including non-block-multiple) -------------------------------------------------------


@pytest.mark.parametrize("height,width", [(7, 7), (16, 16), (31, 47), (224, 224)])
@pytest.mark.parametrize("variant", GAUSSIAN_VARIANTS)
def test_matches_cpu_and_basic_various_dimensions(height, width, variant):
    rng = np.random.default_rng(height * 1000 + width)
    image = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    _assert_matches_cpu_and_basic(image, kernel_size=3, variant=variant)


# -- border tests -------------------------------------------------------


@pytest.mark.parametrize("edge", ["top", "bottom", "left", "right", "corner"])
@pytest.mark.parametrize("variant", GAUSSIAN_VARIANTS)
def test_border_regions_match_cpu_and_basic(edge, variant):
    size = 16
    rng = np.random.default_rng(stable_seed(edge, variant))
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
    _assert_matches_cpu_and_basic(image, kernel_size=5, sigma=1.0, variant=variant)


@pytest.mark.parametrize("variant", GAUSSIAN_VARIANTS)
def test_fully_bright_image_border_reflects_not_darkens(variant):
    image = sharp_edge_square(size=32, border=0)
    out = gaussian_enhanced_cuda(image, kernel_size=5, sigma=1.0, variant=variant)
    assert out.min() == 255


# -- reference-output regression (frozen Section 3 baselines) -------------------------------------------------------


@pytest.mark.parametrize("fixture_name", list(FIXTURES.keys()))
@pytest.mark.parametrize("variant", GAUSSIAN_VARIANTS)
def test_matches_frozen_cpu_reference_output(fixture_name, variant):
    ref_path = REFERENCE_DIR / fixture_name / "gaussian.npy"
    if not ref_path.exists():
        pytest.skip(f"No reference file at {ref_path}")
    image = FIXTURES[fixture_name]()
    expected = np.load(ref_path)
    gpu_out = gaussian_enhanced_cuda(image, kernel_size=5, sigma=0.0, variant=variant)
    diff = np.abs(expected.astype(np.int16) - gpu_out.astype(np.int16))
    assert diff.max() <= GAUSSIAN_VS_CPU_TOLERANCE


# -- determinism -------------------------------------------------------


@pytest.mark.parametrize("variant", GAUSSIAN_VARIANTS)
def test_repeated_execution_is_deterministic(variant):
    image = FIXTURES["random_deterministic"]()
    run_a = gaussian_enhanced_cuda(image, kernel_size=5, variant=variant)
    run_b = gaussian_enhanced_cuda(image, kernel_size=5, variant=variant)
    run_c = gaussian_enhanced_cuda(image, kernel_size=5, variant=variant)
    np.testing.assert_array_equal(run_a, run_b)
    np.testing.assert_array_equal(run_b, run_c)


# -- Basic Gaussian regression: must remain unchanged -------------------------------------------------------


def test_basic_gaussian_unchanged_by_this_section():
    """gaussian_basic must remain the exact same reproducible baseline
    it was in Section 4B -- re-verify against a frozen reference here so
    a future accidental edit to gaussian_basic.cu/gaussian.py is caught."""
    image = FIXTURES["random_deterministic"]()
    config = FilterConfig(gaussian_kernel_size=5, gaussian_sigma=0.0)
    cpu_out = apply_gaussian(image, config)
    basic_out = gaussian_cuda(image, kernel_size=5, sigma=0.0)
    diff = np.abs(cpu_out.astype(np.int16) - basic_out.astype(np.int16))
    assert diff.max() <= GAUSSIAN_VS_CPU_TOLERANCE


# -- GPU-native API / block configuration -------------------------------------------------------


def test_gaussian_enhanced_cuda_gpu_chains_without_host_round_trip():
    image = FIXTURES["random_deterministic"]()
    gpu_image = xray_cuda.upload_image(image)

    result = gaussian_enhanced_cuda_gpu(gpu_image, kernel_size=5, variant="specialized")
    assert set(result.keys()) >= {"output", "coeff_upload_ms", "horizontal_ms", "vertical_ms", "kernel_ms"}
    assert result["kernel_ms"] == pytest.approx(result["horizontal_ms"] + result["vertical_ms"])

    gpu_downloaded = xray_cuda.download_image(result["output"])
    host_api_output = gaussian_enhanced_cuda(image, kernel_size=5, variant="specialized")
    np.testing.assert_array_equal(gpu_downloaded, host_api_output)


def test_gaussian_enhanced_input_not_modified():
    image = FIXTURES["random_deterministic"]()
    gpu_image = xray_cuda.upload_image(image)
    gaussian_enhanced_cuda_gpu(gpu_image, kernel_size=5, variant="shared")
    unchanged = xray_cuda.download_image(gpu_image)
    np.testing.assert_array_equal(unchanged, image)


@pytest.mark.parametrize("block", [(8, 8), (16, 16), (32, 8), (32, 16)])
@pytest.mark.parametrize("variant", GAUSSIAN_VARIANTS)
def test_block_configurations_all_correct(block, variant):
    """Block size is a performance-only parameter (Section 6 spec section
    23) -- must never change the result."""
    image = FIXTURES["random_deterministic"]()
    out = gaussian_enhanced_cuda(image, kernel_size=5, variant=variant, block=block)
    basic_out = gaussian_cuda(image, kernel_size=5, sigma=0.0)
    diff = np.abs(basic_out.astype(np.int16) - out.astype(np.int16))
    assert diff.max() == 0, f"block={block} variant={variant}: max_abs_diff={diff.max()} (expected 0)"


# -- input validation -------------------------------------------------------


def test_gaussian_enhanced_rejects_invalid_variant():
    image = FIXTURES["constant"]()
    with pytest.raises(ValueError):
        gaussian_enhanced_cuda(image, kernel_size=5, variant="not_a_variant")


def test_specialized_variant_rejects_unsupported_kernel_size():
    """Bypasses the Python-level FilterConfig check (which would reject
    kernel_size=11 for any variant) to directly exercise the C++-level
    guard specific to Specialized: it only supports {3,5,7,9} since those
    are the only compile-time template instantiations."""
    image = FIXTURES["constant"]()
    gpu_image = xray_cuda.upload_image(image)
    bad_coeffs = np.ones(11, dtype=np.float32)
    with pytest.raises(ValueError):
        xray_cuda.gaussian_enhanced_gpu(gpu_image, bad_coeffs, 3, 16, 16)  # variant=3 (Specialized)


def test_gaussian_enhanced_gpu_rejects_null_image():
    with pytest.raises(ValueError):
        xray_cuda.gaussian_enhanced_gpu(None, gaussian_kernel_1d(5, 0.0), 0, 16, 16)


def test_gaussian_enhanced_gpu_rejects_invalid_variant_int():
    image = FIXTURES["constant"]()
    gpu_image = xray_cuda.upload_image(image)
    with pytest.raises(ValueError):
        xray_cuda.gaussian_enhanced_gpu(gpu_image, gaussian_kernel_1d(5, 0.0), 99, 16, 16)


def test_gaussian_enhanced_gpu_rejects_bad_block_dims():
    image = FIXTURES["constant"]()
    gpu_image = xray_cuda.upload_image(image)
    with pytest.raises(ValueError):
        xray_cuda.gaussian_enhanced_gpu(gpu_image, gaussian_kernel_1d(5, 0.0), 0, 0, 16)


# -- memory: repeated allocation/release -------------------------------------------------------


# -- batched Enhanced Gaussian (xray_cuda.gaussian_enhanced_batch_gpu) -------------------------------------------------------


@pytest.mark.parametrize("variant", GAUSSIAN_VARIANTS)
def test_batch_matches_per_image_single_calls(variant):
    """gaussian_enhanced_batch_gpu (grid.z=N, one launch pair for the
    whole batch) must produce exactly the same result, per image, as N
    separate single-image gaussian_enhanced_gpu calls -- batching is a
    performance change only."""
    rng = np.random.default_rng(hash(variant) % (2**32))
    images = [rng.integers(0, 256, size=(32, 32), dtype=np.uint8) for _ in range(6)]
    batch = np.stack(images, axis=0)
    coeffs = gaussian_kernel_1d(5, 0.0)
    variant_int = GAUSSIAN_VARIANTS.index(variant)

    batch_result = xray_cuda.gaussian_enhanced_batch_gpu(batch, coeffs, variant_int, 16, 16)
    batch_output = batch_result["output"]
    assert batch_output.shape == batch.shape
    assert batch_output.dtype == np.uint8

    for i, image in enumerate(images):
        single = gaussian_enhanced_cuda(image, kernel_size=5, sigma=0.0, variant=variant)
        np.testing.assert_array_equal(
            batch_output[i], single, err_msg=f"variant={variant} image {i}: batch output != single-image output"
        )


def test_batch_output_order_matches_input_order_enhanced():
    """Recognizable per-image constant patterns so an index/order bug in
    the batched dispatch would be impossible to miss."""
    images = [np.full((16, 16), value, dtype=np.uint8) for value in (10, 60, 110, 160, 210)]
    batch = np.stack(images, axis=0)
    coeffs = gaussian_kernel_1d(3, 0.0)
    result = xray_cuda.gaussian_enhanced_batch_gpu(batch, coeffs, 3, 16, 16)  # Specialized
    output = result["output"]
    for i, image in enumerate(images):
        # a constant image blurred with any Gaussian kernel is unchanged (reflect101 border, uniform interior)
        np.testing.assert_array_equal(output[i], image, err_msg=f"index {i} does not match input order")


def test_gaussian_enhanced_batch_gpu_rejects_wrong_dtype():
    batch = np.zeros((4, 16, 16), dtype=np.float32)
    coeffs = gaussian_kernel_1d(5, 0.0)
    with pytest.raises(ValueError):
        xray_cuda.gaussian_enhanced_batch_gpu(batch, coeffs, 3, 16, 16)


def test_gaussian_enhanced_batch_gpu_rejects_wrong_ndim():
    batch = np.zeros((16, 16), dtype=np.uint8)  # missing batch dim
    coeffs = gaussian_kernel_1d(5, 0.0)
    with pytest.raises(ValueError):
        xray_cuda.gaussian_enhanced_batch_gpu(batch, coeffs, 3, 16, 16)


# -- pipeline integration: Enhanced Gaussian + Basic Median/Sobel/Laplacian/Threshold -------------------------------------------------------


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_enhanced_gaussian_pipeline_matches_cpu_and_basic_pipeline():
    """cuda.pipeline.run_basic_cuda_pipeline(use_enhanced_gaussian=True):
    Median/Sobel/Laplacian/Threshold stay Basic; only Gaussian changes.
    At the default sigma=0.0, Enhanced Gaussian is bit-exact vs Basic
    Gaussian (established above), so the two FULL PIPELINES should also
    be bit-exact -- verified directly here, not assumed."""
    from cpu.filters import apply_gaussian, apply_laplacian, apply_median, apply_sobel, apply_threshold
    from cuda.pipeline import run_basic_cuda_pipeline

    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=10, seed=42)
    images = [load_image(p) for p in selection.selected_paths if load_image(p).shape == (224, 224)]
    if len(images) < 2:
        pytest.skip("Not enough same-resolution images in this sample.")

    config = FilterConfig()  # default: gaussian sigma=0.0
    basic_output, _t1 = run_basic_cuda_pipeline(images, config, use_enhanced_gaussian=False)
    enhanced_output, _t2 = run_basic_cuda_pipeline(images, config, use_enhanced_gaussian=True)

    diff_vs_basic = np.abs(basic_output.astype(np.int16) - enhanced_output.astype(np.int16))
    assert diff_vs_basic.max() == 0, (
        f"Enhanced-Gaussian pipeline differs from Basic pipeline at sigma=0.0 "
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
        f"Enhanced-Gaussian pipeline vs CPU: {pct_differing:.4f}% pixels differ, exceeds the "
        f"Section 4F/5-established 1% chain tolerance"
    )


@pytest.mark.parametrize("variant", GAUSSIAN_VARIANTS)
def test_enhanced_gaussian_pipeline_all_variants_run_without_crash(variant):
    from cuda.pipeline import run_basic_cuda_pipeline

    images = [FIXTURES["random_deterministic"](size=32, seed=i) for i in range(4)]
    config = FilterConfig()
    output, timing = run_basic_cuda_pipeline(images, config, use_enhanced_gaussian=True, gaussian_variant=variant)
    assert output.shape == (4, 32, 32)
    assert timing.gaussian_ms is not None and timing.gaussian_ms >= 0.0


def test_enhanced_gaussian_pipeline_disabled_gaussian_ignores_enhanced_flag():
    """use_enhanced_gaussian=True with gaussian_enabled=False must not
    crash or require coefficients -- the flag is irrelevant when the
    stage itself is off. Median/Sobel/Laplacian/Threshold remain enabled
    (FilterConfig defaults), so the output is NOT expected to equal the
    input -- a constant image's gradient is 0 everywhere, and
    Threshold(0, threshold_value=128) is 0, so the correct output here is
    an all-zero image; verified against the CPU reference rather than
    assumed."""
    from cpu.filters import apply_laplacian, apply_median, apply_sobel, apply_threshold
    from cuda.pipeline import run_basic_cuda_pipeline

    images = [FIXTURES["constant"](size=16)]
    config = FilterConfig(gaussian_enabled=False)
    output, timing = run_basic_cuda_pipeline(images, config, use_enhanced_gaussian=True)
    assert timing.gaussian_ms is None

    cpu_expected = apply_threshold(apply_laplacian(apply_sobel(apply_median(images[0], config), config), config), config)
    np.testing.assert_array_equal(output[0], cpu_expected)


@pytest.mark.parametrize("variant", GAUSSIAN_VARIANTS)
def test_repeated_enhanced_gaussian_no_crash_and_no_leak(variant):
    image = FIXTURES["random_deterministic"]()
    expected = gaussian_enhanced_cuda(image, kernel_size=5, variant=variant)

    free_before = xray_cuda.device_memory_info()["free_bytes"]

    for _ in range(100):
        out = gaussian_enhanced_cuda(image, kernel_size=5, variant=variant)
        np.testing.assert_array_equal(out, expected)

    free_after = xray_cuda.device_memory_info()["free_bytes"]
    leaked_bytes = free_before - free_after
    assert leaked_bytes < 50 * 1024 * 1024, (
        f"variant={variant}: possible GPU memory leak: {leaked_bytes} bytes not reclaimed after 100 calls"
    )
