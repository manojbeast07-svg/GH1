"""Section 1 foundation tests: prove Python -> pybind11 -> CUDA -> GPU works.

If no CUDA device is available, GPU-dependent tests are skipped explicitly
(not silently passed) via the module-level pytestmark.
"""

import pytest

xray_cuda = pytest.importorskip(
    "xray_cuda", reason="xray_cuda extension is not built; run cmake --build build first."
)

cuda_present = xray_cuda.cuda_available()
pytestmark = pytest.mark.skipif(
    not cuda_present, reason="No usable CUDA device detected on this machine."
)


def test_module_imports():
    """Test 1: the compiled extension module imports without error."""
    assert xray_cuda is not None
    assert hasattr(xray_cuda, "add_ints")
    assert xray_cuda.add_ints(2, 3) == 5


def test_cuda_availability_detected():
    """Test 2: CUDA availability is detected and reported as a bool."""
    result = xray_cuda.cuda_available()
    assert isinstance(result, bool)
    assert result is True


def test_device_info_present():
    """Test 3: device information is returned as a populated dict."""
    info = xray_cuda.device_info()
    assert isinstance(info, dict)
    assert info["available"] is True
    assert info["name"]
    assert info["max_threads_per_block"] > 0
    assert info["multiprocessor_count"] > 0
    assert info["warp_size"] == 32


def test_smoke_test_computation_is_correct():
    """Test 4: the CUDA kernel computes input[i] + 1 correctly."""
    result = xray_cuda.smoke_test()
    expected = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert result["output"] == pytest.approx(expected)
    assert result["kernel_ms"] >= 0.0


def test_smoke_test_repeated_execution_is_stable():
    """Test 5: repeated execution does not leak or crash."""
    expected = [1.0, 2.0, 3.0, 4.0, 5.0]
    for _ in range(20):
        result = xray_cuda.smoke_test()
        assert result["output"] == pytest.approx(expected)
