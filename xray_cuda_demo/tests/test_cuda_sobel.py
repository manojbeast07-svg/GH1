"""Section 4D tests: Basic CUDA Sobel edge detection vs. the Section 3
CPU reference (cv2.Sobel + convertScaleAbs).

--------------------------------------------------------------------------
Exact equality, not a tolerance
--------------------------------------------------------------------------
Verified empirically before writing this kernel (see cuda/sobel.py
module docstring): a hand-computed 3x3 Gx/Gy convolution with
reflect-101 borders, converted via round(|raw|) clamped to [0,255],
matched cv2.Sobel's actual output bit-for-bit for all four modes (x, y,
magnitude, abs_sum) on a real X-ray, in float32 precision. So CPU and
GPU outputs are required to match exactly here (max_abs_diff == 0), the
same standard used for median (Section 4C) and stricter than Gaussian's
documented +-1 tolerance (Section 4B) -- if any test below needed a
nonzero tolerance, that would indicate a real bug, not acceptable
rounding noise.
"""

from pathlib import Path

import numpy as np
import pytest

xray_cuda = pytest.importorskip(
    "xray_cuda", reason="xray_cuda extension is not built; run cmake --build build first."
)

from cpu.filters import FilterConfig, apply_sobel  # noqa: E402
from cuda.sobel import sobel_cuda, sobel_cuda_gpu  # noqa: E402
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


def _assert_exact_match(image, mode="magnitude"):
    config = FilterConfig(sobel_mode=mode, sobel_kernel_size=3)
    cpu_out = apply_sobel(image, config)
    gpu_out = sobel_cuda(image, mode=mode)

    assert gpu_out.dtype == np.uint8
    assert gpu_out.shape == image.shape

    diff = np.abs(cpu_out.astype(np.int16) - gpu_out.astype(np.int16))
    max_diff = int(diff.max())
    assert max_diff == 0, (
        f"mode={mode}: max_abs_diff={max_diff} (expected exact match); "
        f"{np.count_nonzero(diff)}/{diff.size} pixels differ; "
        f"first mismatch at {np.argwhere(diff)[0].tolist() if np.count_nonzero(diff) else None}"
    )


# -- modes x fixtures (6 fixtures x 4 modes) -------------------------------------------------------


@pytest.mark.parametrize("fixture_name", list(FIXTURES.keys()))
@pytest.mark.parametrize("mode", MODES)
def test_exact_match_on_synthetic_fixtures(fixture_name, mode):
    image = FIXTURES[fixture_name]()
    _assert_exact_match(image, mode=mode)


# -- real X-rays -------------------------------------------------------


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
@pytest.mark.parametrize("mode", MODES)
def test_exact_match_on_real_224x224_xray(mode):
    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=1, seed=42)
    image = load_image(selection.selected_paths[0])
    assert image.shape == (224, 224)
    _assert_exact_match(image, mode=mode)


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
@pytest.mark.parametrize("mode", MODES)
def test_exact_match_on_large_real_xray(mode):
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
    _assert_exact_match(image, mode=mode)


# -- dimensions (including non-block-multiple, odd width/height) -------------------------------------------------------


@pytest.mark.parametrize("height,width", [(7, 7), (16, 16), (31, 47), (224, 224)])
@pytest.mark.parametrize("mode", MODES)
def test_exact_match_various_dimensions(height, width, mode):
    rng = np.random.default_rng(height * 1000 + width)
    image = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    _assert_exact_match(image, mode=mode)


# -- reference-output regression (frozen Section 3 baselines, default magnitude mode) -------------------------------------------------------


@pytest.mark.parametrize("fixture_name", list(FIXTURES.keys()))
def test_matches_frozen_cpu_reference_output(fixture_name):
    ref_path = REFERENCE_DIR / fixture_name / "sobel.npy"
    if not ref_path.exists():
        pytest.skip(f"No reference file at {ref_path}")

    image = FIXTURES[fixture_name]()
    expected = np.load(ref_path)
    gpu_out = sobel_cuda(image, mode="magnitude")  # frozen references use default FilterConfig (magnitude)
    np.testing.assert_array_equal(gpu_out, expected)


# -- border-specific tests -------------------------------------------------------


@pytest.mark.parametrize("edge", ["top", "bottom", "left", "right", "corner"])
@pytest.mark.parametrize("mode", MODES)
def test_border_regions_exact_match(edge, mode):
    size = 16
    rng = np.random.default_rng(stable_seed(edge, mode))
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

    _assert_exact_match(image, mode=mode)


# -- signed-gradient semantics -------------------------------------------------------


@pytest.mark.parametrize("mode", ["x", "y"])
def test_signed_gradient_relationship_preserved_through_gpu(mode):
    """gx(255-image) == -gx(image) mathematically (Sobel is linear and
    the constant 255 offset contributes no gradient), so after
    convertScaleAbs (|.|) the uint8 outputs for `image` and `255-image`
    must be identical -- verifying the GPU kernel preserves the same
    sign-loss semantics as the CPU reference (spec section 4/23), not
    just that final pixel values happen to match some other way."""
    image = FIXTURES["sharp_edge_square"]()
    inverted = 255 - image

    out_original = sobel_cuda(image, mode=mode)
    out_inverted = sobel_cuda(inverted, mode=mode)
    np.testing.assert_array_equal(out_original, out_inverted)

    # cross-check against the CPU reference's own version of this property
    config = FilterConfig(sobel_mode=mode, sobel_kernel_size=3)
    cpu_original = apply_sobel(image, config)
    cpu_inverted = apply_sobel(inverted, config)
    np.testing.assert_array_equal(cpu_original, cpu_inverted)


# -- determinism -------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_repeated_execution_is_deterministic(mode):
    image = FIXTURES["random_deterministic"]()
    run_a = sobel_cuda(image, mode=mode)
    run_b = sobel_cuda(image, mode=mode)
    run_c = sobel_cuda(image, mode=mode)
    np.testing.assert_array_equal(run_a, run_b)
    np.testing.assert_array_equal(run_b, run_c)


# -- GPU-native API / pipeline integration -------------------------------------------------------


def test_sobel_cuda_gpu_chains_without_host_round_trip():
    image = FIXTURES["random_deterministic"]()
    gpu_image = xray_cuda.upload_image(image)

    result = sobel_cuda_gpu(gpu_image, mode="magnitude")
    assert "output" in result and "kernel_ms" in result

    gpu_downloaded = xray_cuda.download_image(result["output"])
    host_api_output = sobel_cuda(image, mode="magnitude")
    np.testing.assert_array_equal(gpu_downloaded, host_api_output)


def test_sobel_basic_gpu_input_not_modified():
    image = FIXTURES["random_deterministic"]()
    gpu_image = xray_cuda.upload_image(image)
    xray_cuda.sobel_basic_gpu(gpu_image, 2)  # magnitude
    unchanged = xray_cuda.download_image(gpu_image)
    np.testing.assert_array_equal(unchanged, image)


def test_sobel_consumes_gpu_image_from_earlier_pipeline_stages():
    """Small manual integration (spec section 36): chain Basic CUDA
    Gaussian -> Median -> Sobel entirely on the GPU, proving the GpuImage
    produced by gaussian_basic_gpu/median_basic_gpu is directly usable by
    Sobel with no intermediate host round trip. NOT the full five-filter
    pipeline.

    Tolerance note: Sobel itself is exact (verified separately, see
    test_exact_match_* above and cuda/sobel.py's docstring) -- but this
    is a *chain*, and Gaussian carries its own documented +-1 tolerance
    (Section 4B) relative to the CPU reference. That small per-pixel
    input difference gets amplified by Sobel's gradient coefficients
    (magnitude up to 2 per neighbor). Empirically isolated before
    picking a tolerance (not guessed): feeding CPU-identical input to
    both CPU and GPU Sobel gives max_abs_diff=0; the full GPU chain vs.
    CPU chain here showed the Gaussian+Median stage differing by <=1 on
    7/4096 pixels, which Sobel then amplified to a max of 4. This test
    checks the whole chain wired together (no crash, right shape/dtype,
    a bounded difference), not Sobel's own exactness a second time.
    """
    from cuda.gaussian import gaussian_cuda_gpu
    from cuda.median import median_cuda_gpu

    image = FIXTURES["random_deterministic"]()
    gpu_image = xray_cuda.upload_image(image)

    gaussian_result = gaussian_cuda_gpu(gpu_image, kernel_size=5, sigma=0.0)
    median_result = median_cuda_gpu(gaussian_result["output"], kernel_size=3)
    sobel_result = sobel_cuda_gpu(median_result["output"], mode="magnitude")

    gpu_chain_output = xray_cuda.download_image(sobel_result["output"])
    assert gpu_chain_output.dtype == np.uint8
    assert gpu_chain_output.shape == image.shape

    # cross-check against the equivalent CPU chain, with the empirically
    # justified chain tolerance explained above (not Sobel's own
    # tolerance, which is 0 -- see test_exact_match_* elsewhere in this file).
    config = FilterConfig(
        gaussian_kernel_size=5, gaussian_sigma=0.0, median_kernel_size=3,
        sobel_mode="magnitude", sobel_kernel_size=3,
        laplacian_enabled=False, threshold_enabled=False,
    )
    from cpu.filters import apply_gaussian, apply_median

    cpu_chain = apply_sobel(apply_median(apply_gaussian(image, config), config), config)
    chain_diff = np.abs(gpu_chain_output.astype(np.int16) - cpu_chain.astype(np.int16))
    chain_tolerance = 8  # empirically observed max was 4; documented margin, not a guess
    assert chain_diff.max() <= chain_tolerance, (
        f"GPU vs CPU full-chain max_abs_diff={chain_diff.max()} exceeds the empirically "
        f"justified chain tolerance={chain_tolerance} (Gaussian's own +-1 tolerance amplified by Sobel)"
    )


# -- input validation -------------------------------------------------------


def test_sobel_cuda_rejects_invalid_mode():
    image = FIXTURES["constant"]()
    with pytest.raises(ValueError):
        sobel_cuda(image, mode="not_a_mode")


def test_sobel_cuda_rejects_wrong_dtype():
    with pytest.raises(ValueError):
        sobel_cuda(np.zeros((8, 8), dtype=np.float32), mode="magnitude")


def test_sobel_cuda_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        sobel_cuda(np.zeros((8, 8, 3), dtype=np.uint8), mode="magnitude")


def test_sobel_cuda_rejects_empty_image():
    with pytest.raises(ValueError):
        sobel_cuda(np.zeros((0, 8), dtype=np.uint8), mode="magnitude")


def test_sobel_basic_gpu_rejects_null_image():
    with pytest.raises(ValueError):
        xray_cuda.sobel_basic_gpu(None, 2)


def test_sobel_basic_gpu_rejects_out_of_range_mode():
    image = FIXTURES["constant"]()
    gpu_image = xray_cuda.upload_image(image)
    with pytest.raises(ValueError):
        xray_cuda.sobel_basic_gpu(gpu_image, 99)
    with pytest.raises(ValueError):
        xray_cuda.sobel_basic_gpu(gpu_image, -1)


# -- memory: repeated allocation/release -------------------------------------------------------


def test_repeated_sobel_no_crash_and_no_leak():
    image = FIXTURES["random_deterministic"]()
    expected = sobel_cuda(image, mode="magnitude")

    free_before = xray_cuda.device_memory_info()["free_bytes"]

    for _ in range(200):
        gpu_image = xray_cuda.upload_image(image)
        result = xray_cuda.sobel_basic_gpu(gpu_image, 2)
        output = xray_cuda.download_image(result["output"])
        np.testing.assert_array_equal(output, expected)
        del gpu_image, result, output

    free_after = xray_cuda.device_memory_info()["free_bytes"]
    leaked_bytes = free_before - free_after
    assert leaked_bytes < 50 * 1024 * 1024, (
        f"Possible GPU memory leak: {leaked_bytes} bytes not reclaimed after 200 iterations "
        f"(free before={free_before}, after={free_after})"
    )
