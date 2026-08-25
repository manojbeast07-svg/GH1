"""CPU reference filter implementations (OpenCV-backed).

This module is the authoritative baseline every future CUDA
implementation (Basic and Enhanced) must reproduce. It is intentionally
NOT optimized -- see the module docstring in cpu/pipeline.py for why.

--------------------------------------------------------------------------
Input/output contract (applies to every apply_* function in this module)
--------------------------------------------------------------------------
Input:  uint8, single-channel, shape (H, W), range 0-255. Violating this
        raises ValueError/TypeError immediately -- no silent resize,
        normalization, or dtype coercion.
Output: uint8, single-channel, shape (H, W), same H/W as input.

--------------------------------------------------------------------------
Border policy (fixed across every filter in this module)
--------------------------------------------------------------------------
cv2.BORDER_DEFAULT (== cv2.BORDER_REFLECT_101). Chosen once, deliberately,
and used for every OpenCV call that accepts a borderType argument, so a
future CUDA kernel only has to reproduce one border behavior. The one
exception is cv2.medianBlur, which does not expose a borderType
parameter at all and, verified empirically (Section 4C), actually uses
BORDER_REPLICATE (clamp-to-edge) internally, not reflection -- documented
below in apply_median so a mismatch is never mistaken for a bug.

--------------------------------------------------------------------------
Sobel / Laplacian output conversion (documented precisely, see Section 3
spec items 10-11 -- this is the part a CUDA reimplementation is most
likely to get subtly wrong)
--------------------------------------------------------------------------
Both filters compute their gradient in CV_32F (never CV_8U directly --
that would silently clip/saturate mid-computation and make the result
depend on intermediate rounding rather than the true gradient). The
float32 result is then converted to uint8 with cv2.convertScaleAbs,
i.e. output = saturate_cast<uint8>(|value| * 1.0 + 0.0). This means:
  - Sobel "x" and "y" modes lose gradient sign in the uint8 output (a
    pixel with gradient -40 and one with +40 both become 40). This is a
    deliberate, documented consequence of forcing a uint8 pipeline output
    contract, not an oversight.
  - Sobel "magnitude" (sqrt(gx^2+gy^2)) and "abs_sum" (|gx|+|gy|) are
    already non-negative, so convertScaleAbs only clips large values to
    255; no sign information is lost there.
  - Laplacian is signed (can be positive or negative depending on
    curvature direction); convertScaleAbs folds both signs together the
    same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet

import cv2
import numpy as np

# Single border policy used by every filter below that accepts one.
BORDER_POLICY = cv2.BORDER_DEFAULT

ALLOWED_GAUSSIAN_KERNELS: FrozenSet[int] = frozenset({3, 5, 7, 9})
ALLOWED_MEDIAN_KERNELS: FrozenSet[int] = frozenset({3, 5, 7})
ALLOWED_SOBEL_KERNELS: FrozenSet[int] = frozenset({1, 3, 5, 7})
ALLOWED_LAPLACIAN_KERNELS: FrozenSet[int] = frozenset({1, 3, 5})
SOBEL_MODES: FrozenSet[str] = frozenset({"x", "y", "magnitude", "abs_sum"})


# -- centralized validation (spec: "do not duplicate parameter validation") ------


def _validate_choice(name: str, value, allowed) -> None:
    if value not in allowed:
        raise ValueError(f"{name}={value!r} is not supported; expected one of {sorted(allowed, key=str)}")


def _validate_range(name: str, value: int, low: int, high: int) -> None:
    if not (low <= value <= high):
        raise ValueError(f"{name}={value} out of allowed range [{low}, {high}]")


def validate_input_contract(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected a numpy.ndarray, got {type(image)!r}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 input, got dtype={image.dtype}")
    if image.ndim != 2:
        raise ValueError(f"Expected a single-channel [H, W] grayscale image, got shape {image.shape}")


# -- configuration -------------------------------------------------------


@dataclass
class FilterConfig:
    """Central, validated configuration for every pipeline stage.

    Validation happens once, here, in __post_init__ -- individual apply_*
    functions trust a FilterConfig instance to already be valid rather
    than re-checking its fields.
    """

    gaussian_enabled: bool = True
    gaussian_kernel_size: int = 5
    gaussian_sigma: float = 0.0  # 0.0 => OpenCV derives sigma from kernel size

    median_enabled: bool = True
    median_kernel_size: int = 3

    sobel_enabled: bool = True
    sobel_kernel_size: int = 3
    sobel_mode: str = "magnitude"  # recommended default (spec section 9)

    laplacian_enabled: bool = True
    laplacian_kernel_size: int = 3
    laplacian_scale: float = 1.0
    laplacian_delta: float = 0.0

    threshold_enabled: bool = True
    threshold_value: int = 128
    threshold_max_value: int = 255

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _validate_choice("gaussian_kernel_size", self.gaussian_kernel_size, ALLOWED_GAUSSIAN_KERNELS)
        if self.gaussian_sigma < 0:
            raise ValueError(f"gaussian_sigma must be >= 0, got {self.gaussian_sigma}")

        _validate_choice("median_kernel_size", self.median_kernel_size, ALLOWED_MEDIAN_KERNELS)

        _validate_choice("sobel_kernel_size", self.sobel_kernel_size, ALLOWED_SOBEL_KERNELS)
        _validate_choice("sobel_mode", self.sobel_mode, SOBEL_MODES)

        _validate_choice("laplacian_kernel_size", self.laplacian_kernel_size, ALLOWED_LAPLACIAN_KERNELS)

        _validate_range("threshold_value", self.threshold_value, 0, 255)
        _validate_range("threshold_max_value", self.threshold_max_value, 0, 255)


# -- filters -------------------------------------------------------


def apply_gaussian(image: np.ndarray, config: FilterConfig) -> np.ndarray:
    """cv2.GaussianBlur, uint8 in -> uint8 out (OpenCV rounds/saturates
    internally; no explicit intermediate float stage is needed here since
    Gaussian blur is a weighted average that stays within [0, 255])."""
    validate_input_contract(image)
    k = config.gaussian_kernel_size
    return cv2.GaussianBlur(image, (k, k), sigmaX=config.gaussian_sigma, borderType=BORDER_POLICY)


def apply_median(image: np.ndarray, config: FilterConfig) -> np.ndarray:
    """cv2.medianBlur, uint8 in -> uint8 out.

    Nonlinear (the output is a selected input pixel, not a weighted
    combination), which is exactly why its future CUDA implementation
    needs a different optimization strategy (e.g. a selection network)
    from the linear Gaussian/Sobel/Laplacian filters -- this CPU
    reference does not attempt to approximate the true median.

    Note: cv2.medianBlur has no borderType parameter. Verified
    empirically (Section 4C, by comparing against a hand-computed
    replicate-border median on a small all-distinct-values test image):
    OpenCV uses BORDER_REPLICATE (clamp to nearest edge pixel)
    internally for this op, not reflection. This is the one filter in
    this module that does not go through BORDER_POLICY explicitly --
    documented here so it isn't mistaken for an inconsistency later.
    """
    validate_input_contract(image)
    return cv2.medianBlur(image, config.median_kernel_size)


def apply_sobel(image: np.ndarray, config: FilterConfig) -> np.ndarray:
    """cv2.Sobel computed in CV_32F, combined per config.sobel_mode, then
    converted to uint8 via cv2.convertScaleAbs. See the module docstring
    for the exact, documented conversion semantics."""
    validate_input_contract(image)
    ksize = config.sobel_kernel_size
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=ksize, borderType=BORDER_POLICY)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=ksize, borderType=BORDER_POLICY)

    mode = config.sobel_mode
    if mode == "x":
        raw = gx
    elif mode == "y":
        raw = gy
    elif mode == "magnitude":
        raw = cv2.magnitude(gx, gy)
    elif mode == "abs_sum":
        raw = cv2.absdiff(gx, np.zeros_like(gx)) + cv2.absdiff(gy, np.zeros_like(gy))
    else:  # unreachable if constructed via a validated FilterConfig
        raise ValueError(f"Unsupported sobel_mode: {mode!r}")

    return cv2.convertScaleAbs(raw)


def apply_laplacian(image: np.ndarray, config: FilterConfig) -> np.ndarray:
    """cv2.Laplacian computed in CV_32F, then converted to uint8 via
    cv2.convertScaleAbs. See the module docstring for exact semantics."""
    validate_input_contract(image)
    raw = cv2.Laplacian(
        image,
        cv2.CV_32F,
        ksize=config.laplacian_kernel_size,
        scale=config.laplacian_scale,
        delta=config.laplacian_delta,
        borderType=BORDER_POLICY,
    )
    return cv2.convertScaleAbs(raw)


def apply_threshold(image: np.ndarray, config: FilterConfig) -> np.ndarray:
    """cv2.threshold with THRESH_BINARY:
        pixel > threshold_value -> threshold_max_value
        otherwise               -> 0
    """
    validate_input_contract(image)
    _, output = cv2.threshold(image, config.threshold_value, config.threshold_max_value, cv2.THRESH_BINARY)
    return output


# name -> apply function, in canonical pipeline order (spec section 13).
# Used by cpu/pipeline.py to avoid hard-coding the stage list twice, and
# by the reference-output generator script.
FILTER_STAGES = (
    ("gaussian", apply_gaussian),
    ("median", apply_median),
    ("sobel", apply_sobel),
    ("laplacian", apply_laplacian),
    ("threshold", apply_threshold),
)
