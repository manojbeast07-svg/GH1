"""Section 4C tests: Basic CUDA median filter vs. the Section 3 CPU
reference (cv2.medianBlur).

--------------------------------------------------------------------------
Exact equality, not a tolerance
--------------------------------------------------------------------------
Unlike Gaussian (float accumulation -> a documented +-1 rounding
tolerance), median selects an existing integer pixel value -- there is
no floating-point accumulation anywhere in this kernel, so CPU and GPU
outputs are required to match exactly (max_abs_diff == 0). If any test
here needed a nonzero tolerance to pass, that would indicate a real bug
(wrong border rule, wrong neighborhood, wrong selection), not an
acceptable rounding difference -- per the Section 4C spec, correctness
criteria are not to be loosened to make tests pass.

--------------------------------------------------------------------------
Border rule: BORDER_REPLICATE (verified empirically, not assumed)
--------------------------------------------------------------------------
See cuda/median.py module docstring for the verification: a 5x5
all-distinct-values image run through cv2.medianBlur matched a
hand-computed BORDER_REPLICATE (clamp-to-edge) median exactly, and did
NOT match a BORDER_REFLECT_101 hypothesis. This is genuinely different
from Gaussian/Sobel/Laplacian's BORDER_DEFAULT (reflect-101).
"""

from pathlib import Path

import numpy as np
import pytest

xray_cuda = pytest.importorskip(
    "xray_cuda", reason="xray_cuda extension is not built; run cmake --build build first."
)

from cpu.filters import FilterConfig, apply_median  # noqa: E402
from cuda.median import median_cuda, median_cuda_gpu  # noqa: E402
from pipeline.dataset import DatasetManager  # noqa: E402
from pipeline.image_loader import load_image  # noqa: E402
from tests.conftest import get_real_dataset_path, stable_seed  # noqa: E402
from tests.fixtures import FIXTURES  # noqa: E402

cuda_present = xray_cuda.cuda_available()
pytestmark = pytest.mark.skipif(
    not cuda_present, reason="No usable CUDA device detected on this machine."
)

REFERENCE_DIR = Path(__file__).resolve().parent / "reference"


def _assert_exact_match(image, kernel_size=3):
    config = FilterConfig(median_kernel_size=kernel_size)
    cpu_out = apply_median(image, config)
    gpu_out = median_cuda(image, kernel_size=kernel_size)

    assert gpu_out.dtype == np.uint8
    assert gpu_out.shape == image.shape

    diff = np.abs(cpu_out.astype(np.int16) - gpu_out.astype(np.int16))
    max_diff = int(diff.max())
    assert max_diff == 0, (
        f"kernel={kernel_size}: max_abs_diff={max_diff} (expected exact match); "
        f"{np.count_nonzero(diff)}/{diff.size} pixels differ"
    )


# -- correctness: synthetic fixtures, all supported kernel sizes -------------------------------------------------------


@pytest.mark.parametrize("fixture_name", list(FIXTURES.keys()))
@pytest.mark.parametrize("kernel_size", [3, 5, 7])
def test_exact_match_on_synthetic_fixtures(fixture_name, kernel_size):
    image = FIXTURES[fixture_name]()
    _assert_exact_match(image, kernel_size=kernel_size)


# -- real X-rays -------------------------------------------------------


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
@pytest.mark.parametrize("kernel_size", [3, 5, 7])
def test_exact_match_on_real_224x224_xray(kernel_size):
    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=1, seed=42)
    image = load_image(selection.selected_paths[0])
    assert image.shape == (224, 224)
    _assert_exact_match(image, kernel_size=kernel_size)


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
    _assert_exact_match(image, kernel_size=5)


# -- dimensions (including non-multiple-of-block-size and smaller than kernel) -------------------------------------------------------


@pytest.mark.parametrize("height,width", [(7, 7), (16, 16), (31, 47), (224, 224)])
def test_exact_match_various_dimensions(height, width):
    rng = np.random.default_rng(height * 1000 + width)
    image = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    _assert_exact_match(image, kernel_size=3)


# -- reference-output regression (frozen Section 3 baselines, kernel=3 default) -------------------------------------------------------


@pytest.mark.parametrize("fixture_name", list(FIXTURES.keys()))
def test_matches_frozen_cpu_reference_output(fixture_name):
    ref_path = REFERENCE_DIR / fixture_name / "median.npy"
    if not ref_path.exists():
        pytest.skip(f"No reference file at {ref_path}")

    image = FIXTURES[fixture_name]()
    expected = np.load(ref_path)
    gpu_out = median_cuda(image, kernel_size=3)  # frozen references use default FilterConfig (kernel=3)
    np.testing.assert_array_equal(gpu_out, expected)


# -- dedicated border fixture (distinct values at every position) -------------------------------------------------------


@pytest.mark.parametrize("edge", ["top", "bottom", "left", "right", "corner"])
@pytest.mark.parametrize("kernel_size", [3, 5])
def test_border_regions_exact_match(edge, kernel_size):
    size = 16
    rng = np.random.default_rng(stable_seed(edge, kernel_size))
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

    _assert_exact_match(image, kernel_size=kernel_size)


def test_5x5_all_distinct_values_border_matches_replicate_not_reflect():
    """The exact fixture used to empirically determine the border rule
    (see cuda/median.py docstring), re-verified here through the actual
    GPU kernel rather than just the hand-computed hypothesis."""
    image = np.array([
        [10, 20, 30, 40, 50],
        [60, 70, 80, 90, 100],
        [110, 120, 130, 140, 150],
        [160, 170, 180, 190, 200],
        [210, 220, 230, 240, 250],
    ], dtype=np.uint8)
    _assert_exact_match(image, kernel_size=3)


# -- determinism / repeated execution -------------------------------------------------------


def test_repeated_execution_is_deterministic():
    image = FIXTURES["random_deterministic"]()
    run_a = median_cuda(image, kernel_size=5)
    run_b = median_cuda(image, kernel_size=5)
    run_c = median_cuda(image, kernel_size=5)
    np.testing.assert_array_equal(run_a, run_b)
    np.testing.assert_array_equal(run_b, run_c)


# -- GPU-native API -------------------------------------------------------


def test_median_cuda_gpu_chains_without_host_round_trip():
    image = FIXTURES["random_deterministic"]()
    gpu_image = xray_cuda.upload_image(image)

    result = median_cuda_gpu(gpu_image, kernel_size=5)
    assert "output" in result and "kernel_ms" in result

    gpu_downloaded = xray_cuda.download_image(result["output"])
    host_api_output = median_cuda(image, kernel_size=5)
    np.testing.assert_array_equal(gpu_downloaded, host_api_output)


def test_median_basic_gpu_input_not_modified():
    image = FIXTURES["random_deterministic"]()
    gpu_image = xray_cuda.upload_image(image)
    xray_cuda.median_basic_gpu(gpu_image, 5)
    unchanged = xray_cuda.download_image(gpu_image)
    np.testing.assert_array_equal(unchanged, image)


# -- input validation -------------------------------------------------------


def test_median_cuda_rejects_invalid_kernel_size():
    image = FIXTURES["constant"]()
    with pytest.raises(ValueError):
        median_cuda(image, kernel_size=4)  # even
    with pytest.raises(ValueError):
        median_cuda(image, kernel_size=9)  # not in ALLOWED_MEDIAN_KERNELS ({3,5,7})
    with pytest.raises(ValueError):
        median_cuda(image, kernel_size=0)


def test_median_cuda_rejects_wrong_dtype():
    with pytest.raises(ValueError):
        median_cuda(np.zeros((8, 8), dtype=np.float32), kernel_size=3)


def test_median_cuda_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        median_cuda(np.zeros((8, 8, 3), dtype=np.uint8), kernel_size=3)


def test_median_basic_gpu_rejects_null_image():
    with pytest.raises(ValueError):
        xray_cuda.median_basic_gpu(None, 3)


def test_median_basic_gpu_rejects_kernel_size_above_max():
    image = FIXTURES["constant"]()
    gpu_image = xray_cuda.upload_image(image)
    with pytest.raises(ValueError):
        xray_cuda.median_basic_gpu(gpu_image, 9)  # exceeds kMedianBasicMaxKernelSize=7


def test_median_basic_gpu_rejects_even_kernel_size():
    image = FIXTURES["constant"]()
    gpu_image = xray_cuda.upload_image(image)
    with pytest.raises(ValueError):
        xray_cuda.median_basic_gpu(gpu_image, 4)


# -- memory: repeated allocation/release -------------------------------------------------------


def test_repeated_median_no_crash_and_no_leak():
    image = FIXTURES["random_deterministic"]()
    expected = median_cuda(image, kernel_size=5)

    free_before = xray_cuda.device_memory_info()["free_bytes"]

    for _ in range(200):
        gpu_image = xray_cuda.upload_image(image)
        result = xray_cuda.median_basic_gpu(gpu_image, 5)
        output = xray_cuda.download_image(result["output"])
        np.testing.assert_array_equal(output, expected)
        del gpu_image, result, output

    free_after = xray_cuda.device_memory_info()["free_bytes"]
    leaked_bytes = free_before - free_after
    assert leaked_bytes < 50 * 1024 * 1024, (
        f"Possible GPU memory leak: {leaked_bytes} bytes not reclaimed after 200 iterations "
        f"(free before={free_before}, after={free_after})"
    )
