"""Section 4A tests: Python <-> CUDA image transfer, GpuImage lifecycle,
the trivial boundary-safe +1 kernel, and invalid-input rejection.

Skipped (not silently passed) if no usable CUDA device is present, same
pattern as tests/test_cuda_smoke.py.
"""

import numpy as np
import pytest

xray_cuda = pytest.importorskip(
    "xray_cuda", reason="xray_cuda extension is not built; run cmake --build build first."
)

from pipeline.dataset import DatasetManager  # noqa: E402
from pipeline.image_loader import load_image  # noqa: E402
from tests.conftest import get_real_dataset_path  # noqa: E402

cuda_present = xray_cuda.cuda_available()
pytestmark = pytest.mark.skipif(
    not cuda_present, reason="No usable CUDA device detected on this machine."
)


def _random_image(height, width, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width), dtype=np.uint8)


# -- Test 1: upload/download synthetic uint8 image, several shapes -------------------------------------------------------


@pytest.mark.parametrize("height,width", [(16, 16), (31, 47), (224, 224), (512, 512)])
def test_upload_download_round_trip_various_shapes(height, width):
    image = _random_image(height, width, seed=height * 1000 + width)
    gpu_image = xray_cuda.upload_image(image)

    assert gpu_image.height == height
    assert gpu_image.width == width
    assert gpu_image.shape == (height, width)
    assert gpu_image.dtype == "uint8"
    assert gpu_image.nbytes == height * width

    downloaded = xray_cuda.download_image(gpu_image)
    np.testing.assert_array_equal(downloaded, image)
    assert downloaded.dtype == np.uint8
    assert downloaded.shape == (height, width)
    assert downloaded.flags["C_CONTIGUOUS"]


# -- Test 2: real 224x224 X-ray -------------------------------------------------------


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_upload_download_real_xray():
    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=1, seed=42)
    image = load_image(selection.selected_paths[0])

    gpu_image = xray_cuda.upload_image(image)
    assert gpu_image.shape == image.shape

    downloaded = xray_cuda.download_image(gpu_image)
    np.testing.assert_array_equal(downloaded, image)  # bit-for-bit round trip


# -- Test 3: GPU +1 kernel correctness -------------------------------------------------------


def test_image_add_one_correctness():
    image = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
    gpu_image = xray_cuda.upload_image(image)

    result = xray_cuda.image_add_one(gpu_image)
    output = xray_cuda.download_image(result["output"])

    expected = (image.astype(np.uint16) + 1).astype(np.uint8)  # documented wraparound
    np.testing.assert_array_equal(output, expected)

    # input GpuImage must be unmodified
    unchanged = xray_cuda.download_image(gpu_image)
    np.testing.assert_array_equal(unchanged, image)


def test_image_add_one_uint8_wraparound():
    image = np.array([[254, 255, 0]], dtype=np.uint8)
    gpu_image = xray_cuda.upload_image(image)
    result = xray_cuda.image_add_one(gpu_image)
    output = xray_cuda.download_image(result["output"])
    np.testing.assert_array_equal(output, np.array([[255, 0, 1]], dtype=np.uint8))


def test_image_add_one_on_real_xray_matches_cpu():
    """CPU expected: (pixel + 1) mod 256, computed independently of the
    GPU kernel, then compared against the actual GPU output."""
    image = _random_image(224, 224, seed=7)
    gpu_image = xray_cuda.upload_image(image)
    result = xray_cuda.image_add_one(gpu_image)
    gpu_output = xray_cuda.download_image(result["output"])

    cpu_expected = (image.astype(np.uint16) + 1).astype(np.uint8)
    np.testing.assert_array_equal(gpu_output, cpu_expected)


# -- Test 4: non-multiple-of-block-size dimensions -------------------------------------------------------


@pytest.mark.parametrize("height,width", [(1, 1), (15, 17), (31, 47), (17, 256), (256, 17)])
def test_add_one_handles_non_block_multiple_dimensions(height, width):
    """Block size is 16x16 (see gpu_image.cuh); these shapes deliberately
    don't divide evenly, exercising the x<width/y<height bounds check in
    image_add_one_kernel."""
    image = _random_image(height, width, seed=height * 100 + width)
    gpu_image = xray_cuda.upload_image(image)
    result = xray_cuda.image_add_one(gpu_image)
    output = xray_cuda.download_image(result["output"])

    expected = (image.astype(np.uint16) + 1).astype(np.uint8)
    np.testing.assert_array_equal(output, expected)


# -- Test 5: empty/invalid shape rejection -------------------------------------------------------


def test_upload_rejects_empty_dimension():
    with pytest.raises(ValueError):
        xray_cuda.upload_image(np.zeros((0, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        xray_cuda.upload_image(np.zeros((10, 0), dtype=np.uint8))


def test_upload_rejects_1d_array():
    with pytest.raises(ValueError):
        xray_cuda.upload_image(np.zeros((100,), dtype=np.uint8))


# -- Test 6: wrong dtype rejection -------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16, np.int32, np.uint16, np.bool_])
def test_upload_rejects_wrong_dtype(dtype):
    with pytest.raises(ValueError):
        xray_cuda.upload_image(np.zeros((8, 8), dtype=dtype))


# -- Test 7: wrong channel count rejection -------------------------------------------------------


def test_upload_rejects_rgb():
    with pytest.raises(ValueError):
        xray_cuda.upload_image(np.zeros((8, 8, 3), dtype=np.uint8))


def test_upload_rejects_rgba():
    with pytest.raises(ValueError):
        xray_cuda.upload_image(np.zeros((8, 8, 4), dtype=np.uint8))


def test_upload_rejects_3d_batch():
    with pytest.raises(ValueError):
        xray_cuda.upload_image(np.zeros((5, 8, 8), dtype=np.uint8))  # a [N,H,W] batch, not supported here


def test_download_rejects_none():
    with pytest.raises(ValueError):
        xray_cuda.download_image(None)


def test_image_add_one_rejects_none():
    with pytest.raises(ValueError):
        xray_cuda.image_add_one(None)


# -- Test 8 + 9: repeated allocation/release, memory cleanup -------------------------------------------------------


def test_repeated_upload_kernel_download_release_no_crash_and_no_leak():
    image = _random_image(224, 224, seed=99)
    expected = (image.astype(np.uint16) + 1).astype(np.uint8)

    free_before = xray_cuda.device_memory_info()["free_bytes"]

    for _ in range(200):
        gpu_image = xray_cuda.upload_image(image)
        result = xray_cuda.image_add_one(gpu_image)
        output = xray_cuda.download_image(result["output"])
        np.testing.assert_array_equal(output, expected)
        del gpu_image, result, output  # explicit release; RAII frees device memory here

    free_after = xray_cuda.device_memory_info()["free_bytes"]

    # CUDA runtime memory reporting can vary run-to-run (fragmentation,
    # driver-side caching); we only check for *obvious* persistent
    # leakage, not exact equality, per the Section 4A spec.
    leaked_bytes = free_before - free_after
    assert leaked_bytes < 50 * 1024 * 1024, (
        f"Possible GPU memory leak: {leaked_bytes} bytes not reclaimed after 200 iterations "
        f"(free before={free_before}, after={free_after})"
    )


# -- Test 10: CUDA event timing -------------------------------------------------------


def test_kernel_timing_is_non_negative():
    image = _random_image(224, 224, seed=1)
    gpu_image = xray_cuda.upload_image(image)
    result = xray_cuda.image_add_one(gpu_image)
    assert isinstance(result["kernel_ms"], float)
    assert result["kernel_ms"] >= 0.0


# -- device memory reporting -------------------------------------------------------


def test_device_memory_info_sane_values():
    info = xray_cuda.device_memory_info()
    assert info["total_bytes"] > 0
    assert 0 <= info["free_bytes"] <= info["total_bytes"]
