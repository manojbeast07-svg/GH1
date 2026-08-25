"""Section 3 tests: individual filter correctness, parameter validation,
determinism, and border-handling behavior.

Each apply_* function is compared directly against an independently
constructed cv2 call using the same parameters (not against the chained
pipeline), and against the frozen tests/reference/*.npy baselines.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from cpu.filters import (
    ALLOWED_GAUSSIAN_KERNELS,
    ALLOWED_LAPLACIAN_KERNELS,
    ALLOWED_MEDIAN_KERNELS,
    ALLOWED_SOBEL_KERNELS,
    BORDER_POLICY,
    FilterConfig,
    apply_gaussian,
    apply_laplacian,
    apply_median,
    apply_sobel,
    apply_threshold,
    validate_input_contract,
)
from tests.fixtures import FIXTURES, sharp_edge_square

REFERENCE_DIR = Path(__file__).resolve().parent / "reference"


# -- input contract -------------------------------------------------------


def test_validate_input_contract_rejects_wrong_dtype():
    with pytest.raises(ValueError):
        validate_input_contract(np.zeros((8, 8), dtype=np.float32))


def test_validate_input_contract_rejects_multi_channel():
    with pytest.raises(ValueError):
        validate_input_contract(np.zeros((8, 8, 3), dtype=np.uint8))


def test_validate_input_contract_rejects_non_array():
    with pytest.raises(TypeError):
        validate_input_contract([[1, 2], [3, 4]])


def test_validate_input_contract_accepts_valid_input():
    validate_input_contract(np.zeros((8, 8), dtype=np.uint8))  # should not raise


@pytest.mark.parametrize(
    "apply_fn",
    [apply_gaussian, apply_median, apply_sobel, apply_laplacian, apply_threshold],
)
def test_each_filter_enforces_input_contract(apply_fn):
    config = FilterConfig()
    with pytest.raises(ValueError):
        apply_fn(np.zeros((8, 8), dtype=np.float32), config)


# -- parameter validation -------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gaussian_kernel_size": 4},   # even
        {"gaussian_kernel_size": 0},
        {"median_kernel_size": 2},     # even
        {"threshold_value": -1},
        {"threshold_value": 256},
        {"threshold_max_value": 300},
        {"sobel_mode": "not_a_mode"},
        {"laplacian_kernel_size": 2},  # even / unsupported
        {"sobel_kernel_size": 2},      # even / unsupported
    ],
)
def test_invalid_config_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        FilterConfig(**kwargs)


def test_allowed_kernel_sets_are_all_odd_and_positive():
    for allowed in (ALLOWED_GAUSSIAN_KERNELS, ALLOWED_MEDIAN_KERNELS, ALLOWED_SOBEL_KERNELS, ALLOWED_LAPLACIAN_KERNELS):
        for k in allowed:
            assert k > 0 and k % 2 == 1


# -- per-filter correctness (compared against an independent cv2 call) -------------------------------------------------------


def test_gaussian_matches_direct_opencv_call():
    image = FIXTURES["random_deterministic"]()
    config = FilterConfig(gaussian_kernel_size=5, gaussian_sigma=0.0)
    expected = cv2.GaussianBlur(image, (5, 5), sigmaX=0.0, borderType=BORDER_POLICY)
    actual = apply_gaussian(image, config)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.uint8
    assert actual.shape == image.shape


def test_median_matches_direct_opencv_call():
    image = FIXTURES["impulse_noise"]()
    config = FilterConfig(median_kernel_size=3)
    expected = cv2.medianBlur(image, 3)
    actual = apply_median(image, config)
    np.testing.assert_array_equal(actual, expected)


def test_sobel_x_matches_direct_computation():
    image = FIXTURES["horizontal_gradient"]()
    config = FilterConfig(sobel_mode="x", sobel_kernel_size=3)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3, borderType=BORDER_POLICY)
    expected = cv2.convertScaleAbs(gx)
    actual = apply_sobel(image, config)
    np.testing.assert_array_equal(actual, expected)


def test_sobel_y_matches_direct_computation():
    image = FIXTURES["vertical_gradient"]()
    config = FilterConfig(sobel_mode="y", sobel_kernel_size=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3, borderType=BORDER_POLICY)
    expected = cv2.convertScaleAbs(gy)
    actual = apply_sobel(image, config)
    np.testing.assert_array_equal(actual, expected)


def test_sobel_magnitude_matches_sqrt_gx2_gy2():
    image = FIXTURES["sharp_edge_square"]()
    config = FilterConfig(sobel_mode="magnitude", sobel_kernel_size=3)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3, borderType=BORDER_POLICY)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3, borderType=BORDER_POLICY)
    expected = cv2.convertScaleAbs(np.sqrt(gx.astype(np.float64) ** 2 + gy.astype(np.float64) ** 2))
    actual = apply_sobel(image, config)
    # float32 sqrt vs float64 sqrt can differ by rounding at the uint8
    # boundary; allow a tolerance of 1 pixel value.
    np.testing.assert_allclose(actual.astype(np.int16), expected.astype(np.int16), atol=1)


def test_sobel_abs_sum_matches_direct_computation():
    image = FIXTURES["sharp_edge_square"]()
    config = FilterConfig(sobel_mode="abs_sum", sobel_kernel_size=3)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3, borderType=BORDER_POLICY)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3, borderType=BORDER_POLICY)
    expected = cv2.convertScaleAbs(np.abs(gx) + np.abs(gy))
    actual = apply_sobel(image, config)
    np.testing.assert_array_equal(actual, expected)


def test_sobel_x_and_y_lose_sign_as_documented():
    """A +gradient and a -gradient of equal magnitude must map to the
    same uint8 output (documented consequence of convertScaleAbs).

    Sobel is linear, so for image2 = 255 - image, gx(image2) == -gx(image)
    at every pixel (the constant 255 offset contributes zero gradient).
    After convertScaleAbs (|.|), the two outputs must therefore be
    identical everywhere, not just near the edge.
    """
    image = np.zeros((16, 16), dtype=np.uint8)
    image[:, 8:] = 255  # rising edge -> +gx at the boundary
    config = FilterConfig(sobel_mode="x", sobel_kernel_size=3)
    rising = apply_sobel(image, config)

    image2 = 255 - image  # falling edge -> -gx at the boundary, everywhere the exact negation
    falling = apply_sobel(image2, config)

    np.testing.assert_array_equal(rising, falling)


def test_laplacian_matches_direct_opencv_call():
    image = FIXTURES["sharp_edge_square"]()
    config = FilterConfig(laplacian_kernel_size=3, laplacian_scale=1.0, laplacian_delta=0.0)
    raw = cv2.Laplacian(image, cv2.CV_32F, ksize=3, scale=1.0, delta=0.0, borderType=BORDER_POLICY)
    expected = cv2.convertScaleAbs(raw)
    actual = apply_laplacian(image, config)
    np.testing.assert_array_equal(actual, expected)


def test_threshold_exact_binary_behavior():
    image = np.array([[0, 127, 128, 129, 255]], dtype=np.uint8)
    config = FilterConfig(threshold_value=128, threshold_max_value=255)
    actual = apply_threshold(image, config)
    # THRESH_BINARY: pixel > threshold -> max_value, else 0
    expected = np.array([[0, 0, 0, 255, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(actual, expected)


def test_threshold_custom_max_value():
    image = np.array([[0, 200, 255]], dtype=np.uint8)
    config = FilterConfig(threshold_value=100, threshold_max_value=200)
    actual = apply_threshold(image, config)
    np.testing.assert_array_equal(actual, np.array([[0, 200, 200]], dtype=np.uint8))


# -- reference-output regression tests -------------------------------------------------------


FILTER_APPLY_FNS = {
    "gaussian": apply_gaussian,
    "median": apply_median,
    "sobel": apply_sobel,
    "laplacian": apply_laplacian,
    "threshold": apply_threshold,
}


@pytest.mark.parametrize("fixture_name", list(FIXTURES.keys()))
@pytest.mark.parametrize("filter_name", list(FILTER_APPLY_FNS.keys()))
def test_matches_frozen_reference_output(fixture_name, filter_name):
    ref_path = REFERENCE_DIR / fixture_name / f"{filter_name}.npy"
    if not ref_path.exists():
        pytest.skip(f"No reference file at {ref_path}; run scripts/generate_cpu_reference_outputs.py")

    image = FIXTURES[fixture_name]()
    expected = np.load(ref_path)
    actual = FILTER_APPLY_FNS[filter_name](image, FilterConfig())
    np.testing.assert_array_equal(actual, expected)


# -- determinism -------------------------------------------------------


@pytest.mark.parametrize("filter_name,apply_fn", list(FILTER_APPLY_FNS.items()))
def test_filter_is_deterministic(filter_name, apply_fn):
    image = FIXTURES["random_deterministic"]()
    config = FilterConfig()
    a = apply_fn(image, config)
    b = apply_fn(image, config)
    np.testing.assert_array_equal(a, b)


# -- border handling -------------------------------------------------------


def test_gaussian_border_is_not_zero_padded():
    """A BORDER_CONSTANT(0) implementation would darken a bright edge
    pixel toward 0; BORDER_DEFAULT (reflect) should not, since it
    reflects the bright interior back across the border instead of
    blending in zeros."""
    image = sharp_edge_square(size=32, border=0)  # solid white square filling the whole image
    config = FilterConfig(gaussian_kernel_size=5, gaussian_sigma=1.0)
    blurred = apply_gaussian(image, config)
    # With reflect-101 border handling, a fully-white image stays fully
    # white after Gaussian blur (no darkening from imaginary zero pixels
    # outside the border).
    assert blurred.min() == 255


def test_gaussian_border_uses_border_default_explicitly():
    # border=0: bright region touches the image edge, so reflect-101 vs.
    # zero-padding actually disagree at the border pixels (a fixture
    # whose bright square has margin, like the default sharp_edge_square,
    # is too far from the edge for a 5x5 kernel to see any difference).
    image = sharp_edge_square(size=32, border=0)
    config = FilterConfig(gaussian_kernel_size=5, gaussian_sigma=1.0)
    expected_reflect = cv2.GaussianBlur(image, (5, 5), sigmaX=1.0, borderType=cv2.BORDER_DEFAULT)
    expected_constant = cv2.GaussianBlur(image, (5, 5), sigmaX=1.0, borderType=cv2.BORDER_CONSTANT)
    actual = apply_gaussian(image, config)
    np.testing.assert_array_equal(actual, expected_reflect)
    assert not np.array_equal(actual, expected_constant)


def test_output_shape_matches_input_shape_at_borders():
    for fixture_name, generator in FIXTURES.items():
        image = generator()
        config = FilterConfig()
        for apply_fn in FILTER_APPLY_FNS.values():
            out = apply_fn(image, config)
            assert out.shape == image.shape, f"{fixture_name} changed shape"
            assert out.dtype == np.uint8
