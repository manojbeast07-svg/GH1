"""Section 4B tests: Basic CUDA Gaussian blur vs. the Section 3 CPU
reference (cv2.GaussianBlur).

--------------------------------------------------------------------------
Justified tolerance: max_abs_diff <= 1
--------------------------------------------------------------------------
Empirically measured (not assumed) before writing these tests: comparing
a 224x224 real X-ray across kernel_size in {3,5,7,9} and sigma in
{0,1,2}, the maximum pixel difference is always <=1, and it's exactly 0
for kernel_size=9 in every case tried. The nonzero cases affect a small
fraction of pixels (e.g. ~2% for kernel_size=3, dropping to ~0.01% for
kernel_size=7). This matches the expected source of divergence: OpenCV's
actual GaussianBlur implementation is a separable two-pass filter that
may round/represent intermediates differently (partially fixed-point for
its small hardcoded kernels) than this Basic CUDA kernel's single-pass
float32 accumulation with one final round. Both are internally
consistent and correctly reproduce the *mathematical* Gaussian; the
difference is a rounding-mode artifact, not a border, coefficient, or
algorithm mismatch -- confirmed by kernel_size=9 (which does NOT use
OpenCV's hardcoded table) being bit-exact.
"""

from pathlib import Path

import numpy as np
import pytest

xray_cuda = pytest.importorskip(
    "xray_cuda", reason="xray_cuda extension is not built; run cmake --build build first."
)

from cpu.filters import FilterConfig, apply_gaussian  # noqa: E402
from cuda.gaussian import gaussian_cuda, gaussian_cuda_gpu, gaussian_kernel_2d  # noqa: E402
from pipeline.dataset import DatasetManager  # noqa: E402
from pipeline.image_loader import load_image  # noqa: E402
from tests.conftest import get_real_dataset_path  # noqa: E402
from tests.fixtures import FIXTURES, sharp_edge_square  # noqa: E402

cuda_present = xray_cuda.cuda_available()
pytestmark = pytest.mark.skipif(
    not cuda_present, reason="No usable CUDA device detected on this machine."
)

REFERENCE_DIR = Path(__file__).resolve().parent / "reference"
TOLERANCE = 1  # justified above


def _assert_matches_cpu(image, kernel_size=5, sigma=0.0, tolerance=TOLERANCE):
    config = FilterConfig(gaussian_kernel_size=kernel_size, gaussian_sigma=sigma)
    cpu_out = apply_gaussian(image, config)
    gpu_out = gaussian_cuda(image, kernel_size=kernel_size, sigma=sigma)

    assert gpu_out.dtype == np.uint8
    assert gpu_out.shape == image.shape

    diff = np.abs(cpu_out.astype(np.int16) - gpu_out.astype(np.int16))
    max_diff = int(diff.max())
    mean_diff = float(diff.mean())
    rmse = float(np.sqrt(np.mean(diff.astype(np.float64) ** 2)))
    num_differing = int(np.count_nonzero(diff))

    assert max_diff <= tolerance, (
        f"kernel={kernel_size} sigma={sigma}: max_abs_diff={max_diff} exceeds tolerance={tolerance} "
        f"(mean_diff={mean_diff:.6f}, rmse={rmse:.6f}, differing={num_differing}/{diff.size})"
    )
    return {"max_diff": max_diff, "mean_diff": mean_diff, "rmse": rmse, "num_differing": num_differing}


# -- synthetic fixtures -------------------------------------------------------


@pytest.mark.parametrize("fixture_name", list(FIXTURES.keys()))
def test_matches_cpu_on_synthetic_fixtures(fixture_name):
    image = FIXTURES[fixture_name]()
    _assert_matches_cpu(image)


# -- real X-rays -------------------------------------------------------


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_matches_cpu_on_real_224x224_xray():
    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=1, seed=42)
    image = load_image(selection.selected_paths[0])
    assert image.shape == (224, 224)
    stats = _assert_matches_cpu(image)
    assert stats["num_differing"] < image.size  # sanity: not comparing garbage


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_matches_cpu_on_large_real_xray():
    """Find a non-224x224 image (the dataset has ~190 of them) so large
    resolutions are exercised too, not just the common case."""
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
    _assert_matches_cpu(image)


# -- dimensions (including non-multiple-of-block-size) -------------------------------------------------------


@pytest.mark.parametrize("height,width", [(16, 16), (31, 47), (224, 224), (512, 512)])
def test_matches_cpu_various_dimensions(height, width):
    rng = np.random.default_rng(height * 1000 + width)
    image = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    _assert_matches_cpu(image)


# -- kernel size / sigma parameter sweep -------------------------------------------------------


@pytest.mark.parametrize("kernel_size", [3, 5, 7, 9])
def test_matches_cpu_across_kernel_sizes(kernel_size):
    image = FIXTURES["random_deterministic"]()
    _assert_matches_cpu(image, kernel_size=kernel_size, sigma=0.0)


@pytest.mark.parametrize("sigma", [0.0, 1.0, 2.0])
def test_matches_cpu_across_sigma_values(sigma):
    image = FIXTURES["random_deterministic"]()
    _assert_matches_cpu(image, kernel_size=5, sigma=sigma)


def test_kernel_size_9_is_bit_exact():
    """kernel_size=9 does not use OpenCV's hardcoded small-kernel table
    (only sizes 3/5/7 do), so this is a stronger check than the general
    tolerance=1 case -- confirms the rounding divergence really is
    table-related, not a border/coefficient bug."""
    image = FIXTURES["random_deterministic"]()
    stats = _assert_matches_cpu(image, kernel_size=9, sigma=0.0, tolerance=1)
    assert stats["max_diff"] == 0


# -- reference-output regression (frozen Section 3 baselines) -------------------------------------------------------


@pytest.mark.parametrize("fixture_name", list(FIXTURES.keys()))
def test_matches_frozen_cpu_reference_output(fixture_name):
    """Default FilterConfig (kernel=5, sigma=0.0) -- the only config the
    frozen references were generated with (Section 3)."""
    ref_path = REFERENCE_DIR / fixture_name / "gaussian.npy"
    if not ref_path.exists():
        pytest.skip(f"No reference file at {ref_path}")

    image = FIXTURES[fixture_name]()
    expected = np.load(ref_path)
    gpu_out = gaussian_cuda(image, kernel_size=5, sigma=0.0)

    diff = np.abs(expected.astype(np.int16) - gpu_out.astype(np.int16))
    assert diff.max() <= TOLERANCE


# -- border-specific tests -------------------------------------------------------


@pytest.mark.parametrize("edge", ["top", "bottom", "left", "right", "corner"])
def test_border_regions_match_cpu(edge):
    size = 32
    image = np.zeros((size, size), dtype=np.uint8)
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
        image[0, -1] = 255
        image[-1, 0] = 255
        image[-1, -1] = 255

    _assert_matches_cpu(image, kernel_size=5, sigma=1.0)


def test_fully_bright_image_border_reflects_not_darkens():
    """A BORDER_CONSTANT(0) mistake would darken this image toward 0 at
    the edges; BORDER_REFLECT_101 must not (same check used for the CPU
    reference in test_cpu_filters.py, mirrored here for the GPU path)."""
    image = sharp_edge_square(size=32, border=0)  # solid white
    gpu_out = gaussian_cuda(image, kernel_size=5, sigma=1.0)
    assert gpu_out.min() == 255


# -- GPU-native API -------------------------------------------------------


def test_gaussian_cuda_gpu_chains_without_host_round_trip():
    image = FIXTURES["random_deterministic"]()
    gpu_image = xray_cuda.upload_image(image)

    result = gaussian_cuda_gpu(gpu_image, kernel_size=5, sigma=0.0)
    assert "output" in result and "kernel_ms" in result and "coeff_upload_ms" in result

    gpu_downloaded = xray_cuda.download_image(result["output"])
    host_api_output = gaussian_cuda(image, kernel_size=5, sigma=0.0)
    np.testing.assert_array_equal(gpu_downloaded, host_api_output)  # same underlying kernel


def test_gaussian_basic_gpu_input_not_modified():
    image = FIXTURES["random_deterministic"]()
    gpu_image = xray_cuda.upload_image(image)
    xray_cuda.gaussian_basic_gpu(gpu_image, gaussian_kernel_2d(5, 0.0))
    unchanged = xray_cuda.download_image(gpu_image)
    np.testing.assert_array_equal(unchanged, image)


# -- error propagation -------------------------------------------------------


def test_gaussian_cuda_rejects_invalid_kernel_size():
    image = FIXTURES["constant"]()
    with pytest.raises(ValueError):
        gaussian_cuda(image, kernel_size=4, sigma=0.0)  # even
    with pytest.raises(ValueError):
        gaussian_cuda(image, kernel_size=11, sigma=0.0)  # not in ALLOWED_GAUSSIAN_KERNELS


def test_gaussian_basic_gpu_rejects_null_image():
    with pytest.raises(ValueError):
        xray_cuda.gaussian_basic_gpu(None, gaussian_kernel_2d(5, 0.0))


def test_gaussian_basic_gpu_rejects_non_square_coeffs():
    image = FIXTURES["constant"]()
    gpu_image = xray_cuda.upload_image(image)
    bad_coeffs = np.ones((5, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        xray_cuda.gaussian_basic_gpu(gpu_image, bad_coeffs)


# -- coefficient generation sanity -------------------------------------------------------


def test_gaussian_kernel_2d_matches_opencv_outer_product():
    import cv2

    k1d = cv2.getGaussianKernel(5, 0.0)
    expected = (k1d @ k1d.T).astype(np.float32)
    actual = gaussian_kernel_2d(5, 0.0)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(actual.sum(), 1.0, atol=1e-5)  # normalized
