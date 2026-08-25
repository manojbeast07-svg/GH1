"""Section 4E tests: Basic CUDA Laplacian filter vs. the Section 3 CPU
reference (cv2.Laplacian + convertScaleAbs).

--------------------------------------------------------------------------
Exact equality, not a tolerance
--------------------------------------------------------------------------
Verified empirically before writing this kernel (see cuda/laplacian.py
module docstring): the exact 2-D coefficient matrix for each supported
kernel_size was recovered via impulse response (NOT the textbook
[[0,1,0],[1,-4,1],[0,1,0]] kernel for every size -- kernel_size=3
actually produces [[2,0,2],[0,-8,0],[2,0,2]]), and a hand-computed
reflect-101-border convolution with that kernel, scale, and delta
matched cv2.Laplacian's actual output bit-for-bit across kernel_size in
{1,3,5} and several scale/delta combinations. So CPU and GPU outputs are
required to match exactly here (max_abs_diff == 0), the same standard
used for Median and Sobel.
"""

from pathlib import Path

import numpy as np
import pytest

xray_cuda = pytest.importorskip(
    "xray_cuda", reason="xray_cuda extension is not built; run cmake --build build first."
)

from cpu.filters import FilterConfig, apply_laplacian  # noqa: E402
from cuda.laplacian import laplacian_cuda, laplacian_cuda_gpu  # noqa: E402
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


def _assert_exact_match(image, kernel_size=3, scale=1.0, delta=0.0):
    config = FilterConfig(laplacian_kernel_size=kernel_size, laplacian_scale=scale, laplacian_delta=delta)
    cpu_out = apply_laplacian(image, config)
    gpu_out = laplacian_cuda(image, kernel_size=kernel_size, scale=scale, delta=delta)

    assert gpu_out.dtype == np.uint8
    assert gpu_out.shape == image.shape

    diff = np.abs(cpu_out.astype(np.int16) - gpu_out.astype(np.int16))
    max_diff = int(diff.max())
    assert max_diff == 0, (
        f"kernel={kernel_size} scale={scale} delta={delta}: max_abs_diff={max_diff} (expected exact match); "
        f"{np.count_nonzero(diff)}/{diff.size} pixels differ; "
        f"first mismatch at {np.argwhere(diff)[0].tolist() if np.count_nonzero(diff) else None}"
    )


# -- kernel sizes x fixtures -------------------------------------------------------


@pytest.mark.parametrize("fixture_name", list(FIXTURES.keys()))
@pytest.mark.parametrize("kernel_size", KERNEL_SIZES)
def test_exact_match_on_synthetic_fixtures(fixture_name, kernel_size):
    image = FIXTURES[fixture_name]()
    _assert_exact_match(image, kernel_size=kernel_size)


# -- scale / delta variants -------------------------------------------------------


@pytest.mark.parametrize("scale,delta", [(1.0, 0.0), (2.0, 0.0), (1.0, 10.0), (0.5, 5.0)])
def test_exact_match_scale_delta_variants(scale, delta):
    image = FIXTURES["random_deterministic"]()
    _assert_exact_match(image, kernel_size=3, scale=scale, delta=delta)


# -- real X-rays -------------------------------------------------------


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
@pytest.mark.parametrize("kernel_size", KERNEL_SIZES)
def test_exact_match_on_real_224x224_xray(kernel_size):
    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=1, seed=42)
    image = load_image(selection.selected_paths[0])
    assert image.shape == (224, 224)
    _assert_exact_match(image, kernel_size=kernel_size)


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
@pytest.mark.parametrize("kernel_size", KERNEL_SIZES)
def test_exact_match_on_large_real_xray(kernel_size):
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
    _assert_exact_match(image, kernel_size=kernel_size)


# -- dimensions (including non-block-multiple) -------------------------------------------------------


@pytest.mark.parametrize("height,width", [(7, 7), (16, 16), (31, 47), (224, 224)])
@pytest.mark.parametrize("kernel_size", KERNEL_SIZES)
def test_exact_match_various_dimensions(height, width, kernel_size):
    rng = np.random.default_rng(height * 1000 + width + kernel_size)
    image = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    _assert_exact_match(image, kernel_size=kernel_size)


# -- reference-output regression (frozen Section 3 baselines, default config) -------------------------------------------------------


@pytest.mark.parametrize("fixture_name", list(FIXTURES.keys()))
def test_matches_frozen_cpu_reference_output(fixture_name):
    ref_path = REFERENCE_DIR / fixture_name / "laplacian.npy"
    if not ref_path.exists():
        pytest.skip(f"No reference file at {ref_path}")

    image = FIXTURES[fixture_name]()
    expected = np.load(ref_path)
    gpu_out = laplacian_cuda(image, kernel_size=3, scale=1.0, delta=0.0)  # frozen references use default FilterConfig
    np.testing.assert_array_equal(gpu_out, expected)


# -- border-specific tests (corners, edges, interior) -------------------------------------------------------


@pytest.mark.parametrize("region", ["top", "bottom", "left", "right", "corner", "interior"])
@pytest.mark.parametrize("kernel_size", KERNEL_SIZES)
def test_border_regions_exact_match(region, kernel_size):
    size = 16
    rng = np.random.default_rng(stable_seed(region, kernel_size))
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

    _assert_exact_match(image, kernel_size=kernel_size)


# -- determinism -------------------------------------------------------


@pytest.mark.parametrize("kernel_size", KERNEL_SIZES)
def test_repeated_execution_is_deterministic(kernel_size):
    image = FIXTURES["random_deterministic"]()
    run_a = laplacian_cuda(image, kernel_size=kernel_size)
    run_b = laplacian_cuda(image, kernel_size=kernel_size)
    run_c = laplacian_cuda(image, kernel_size=kernel_size)
    np.testing.assert_array_equal(run_a, run_b)
    np.testing.assert_array_equal(run_b, run_c)


# -- GPU-native API -------------------------------------------------------


def test_laplacian_cuda_gpu_chains_without_host_round_trip():
    image = FIXTURES["random_deterministic"]()
    gpu_image = xray_cuda.upload_image(image)

    result = laplacian_cuda_gpu(gpu_image, kernel_size=3, scale=1.0, delta=0.0)
    assert "output" in result and "kernel_ms" in result and "coeff_upload_ms" in result

    gpu_downloaded = xray_cuda.download_image(result["output"])
    host_api_output = laplacian_cuda(image, kernel_size=3, scale=1.0, delta=0.0)
    np.testing.assert_array_equal(gpu_downloaded, host_api_output)


def test_laplacian_basic_gpu_input_not_modified():
    from cuda.laplacian import laplacian_kernel_2d

    image = FIXTURES["random_deterministic"]()
    gpu_image = xray_cuda.upload_image(image)
    xray_cuda.laplacian_basic_gpu(gpu_image, laplacian_kernel_2d(3), 1.0, 0.0)
    unchanged = xray_cuda.download_image(gpu_image)
    np.testing.assert_array_equal(unchanged, image)


def test_laplacian_consumes_gpu_image_from_earlier_pipeline_stages():
    """Manual integration smoke test (spec section 30): Gaussian -> Median
    -> Sobel -> Laplacian, entirely on GPU, no CPU round trip -- proving
    GPU Laplacian can consume a GpuImage produced by the earlier Basic
    CUDA stages. NOT the full five-filter pipeline (no Threshold yet).

    Tolerance note: Laplacian itself is exact (see test_exact_match_*
    above). This chain includes Gaussian's own documented +-1 tolerance,
    which compounds through Median/Sobel/Laplacian -- same situation as
    Section 4D's Sobel chain-integration test. A generous, documented
    bound is used here, not a guess: the point of this test is "does the
    chain wire together correctly", not "is Laplacian exact" (already
    proven above).
    """
    from cuda.gaussian import gaussian_cuda_gpu
    from cuda.median import median_cuda_gpu
    from cuda.sobel import sobel_cuda_gpu

    image = FIXTURES["random_deterministic"]()
    gpu_image = xray_cuda.upload_image(image)

    gaussian_result = gaussian_cuda_gpu(gpu_image, kernel_size=5, sigma=0.0)
    median_result = median_cuda_gpu(gaussian_result["output"], kernel_size=3)
    sobel_result = sobel_cuda_gpu(median_result["output"], mode="magnitude")
    laplacian_result = laplacian_cuda_gpu(sobel_result["output"], kernel_size=3, scale=1.0, delta=0.0)

    gpu_chain_output = xray_cuda.download_image(laplacian_result["output"])
    assert gpu_chain_output.dtype == np.uint8
    assert gpu_chain_output.shape == image.shape

    from cpu.filters import apply_gaussian, apply_median, apply_sobel

    config = FilterConfig(
        gaussian_kernel_size=5, gaussian_sigma=0.0, median_kernel_size=3,
        sobel_mode="magnitude", sobel_kernel_size=3,
        laplacian_kernel_size=3, laplacian_scale=1.0, laplacian_delta=0.0,
        threshold_enabled=False,
    )
    cpu_chain = apply_laplacian(
        apply_sobel(apply_median(apply_gaussian(image, config), config), config), config
    )
    chain_diff = np.abs(gpu_chain_output.astype(np.int16) - cpu_chain.astype(np.int16))
    chain_tolerance = 40  # generous, documented: chained tolerance amplification through 4 stages
    assert chain_diff.max() <= chain_tolerance, (
        f"GPU vs CPU full-chain max_abs_diff={chain_diff.max()} exceeds chain_tolerance={chain_tolerance}"
    )


# -- input validation -------------------------------------------------------


def test_laplacian_cuda_rejects_invalid_kernel_size():
    image = FIXTURES["constant"]()
    with pytest.raises(ValueError):
        laplacian_cuda(image, kernel_size=2)  # even
    with pytest.raises(ValueError):
        laplacian_cuda(image, kernel_size=7)  # not in ALLOWED_LAPLACIAN_KERNELS ({1,3,5})


def test_laplacian_cuda_rejects_wrong_dtype():
    with pytest.raises(ValueError):
        laplacian_cuda(np.zeros((8, 8), dtype=np.float32), kernel_size=3)


def test_laplacian_cuda_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        laplacian_cuda(np.zeros((8, 8, 3), dtype=np.uint8), kernel_size=3)


def test_laplacian_cuda_rejects_empty_image():
    with pytest.raises(ValueError):
        laplacian_cuda(np.zeros((0, 8), dtype=np.uint8), kernel_size=3)


def test_laplacian_basic_gpu_rejects_null_image():
    from cuda.laplacian import laplacian_kernel_2d

    with pytest.raises(ValueError):
        xray_cuda.laplacian_basic_gpu(None, laplacian_kernel_2d(3), 1.0, 0.0)


def test_laplacian_basic_gpu_rejects_non_square_coeffs():
    image = FIXTURES["constant"]()
    gpu_image = xray_cuda.upload_image(image)
    bad_coeffs = np.ones((5, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        xray_cuda.laplacian_basic_gpu(gpu_image, bad_coeffs, 1.0, 0.0)


# -- memory: repeated allocation/release -------------------------------------------------------


def test_repeated_laplacian_no_crash_and_no_leak():
    from cuda.laplacian import laplacian_kernel_2d

    image = FIXTURES["random_deterministic"]()
    coeffs = laplacian_kernel_2d(3)
    expected = laplacian_cuda(image, kernel_size=3)

    free_before = xray_cuda.device_memory_info()["free_bytes"]

    for _ in range(200):
        gpu_image = xray_cuda.upload_image(image)
        result = xray_cuda.laplacian_basic_gpu(gpu_image, coeffs, 1.0, 0.0)
        output = xray_cuda.download_image(result["output"])
        np.testing.assert_array_equal(output, expected)
        del gpu_image, result, output

    free_after = xray_cuda.device_memory_info()["free_bytes"]
    leaked_bytes = free_before - free_after
    assert leaked_bytes < 50 * 1024 * 1024, (
        f"Possible GPU memory leak: {leaked_bytes} bytes not reclaimed after 200 iterations "
        f"(free before={free_before}, after={free_after})"
    )
