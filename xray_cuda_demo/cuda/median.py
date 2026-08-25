"""Python-level Basic CUDA median filter API.

Border verification (border behavior is NOT assumed -- see Section 4C
spec section 9): cv2.medianBlur exposes no borderType parameter, so its
actual border rule was inferred empirically rather than guessed. A 5x5
all-distinct-values image run through cv2.medianBlur(k=3) was compared
against a hand-computed BORDER_REPLICATE (clamp-to-edge) median and a
hand-computed BORDER_REFLECT_101 median:

    >>> cv2.medianBlur(image, 3)[0]        # top row of the real output
    [20, 30, 40, 50, 50]
    replicate hypothesis top row:  [20, 30, 40, 50, 50]  <- matches
    reflect101 hypothesis top row: [60, 60, 70, 80, 90]  <- does not match

Confirmed: cv2.medianBlur uses BORDER_REPLICATE. cuda/src/median_basic.cu
reproduces that (clamp_index), not the reflect-101 rule Gaussian uses.
"""

from __future__ import annotations

import numpy as np

import xray_cuda
from cpu.filters import FilterConfig  # reuses existing kernel_size validation


def median_cuda(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Upload `image`, run the Basic CUDA median filter, download and
    return the uint8 [H, W] result. Validates kernel_size with the same
    rule as the Section 3 CPU reference (FilterConfig).
    """
    FilterConfig(median_kernel_size=kernel_size)  # raises ValueError if invalid

    gpu_image = xray_cuda.upload_image(image)
    result = xray_cuda.median_basic_gpu(gpu_image, kernel_size)
    return xray_cuda.download_image(result["output"])


def median_cuda_gpu(gpu_image, kernel_size: int = 3) -> dict:
    """GPU-native variant: GpuImage in, {'output': GpuImage, 'kernel_ms':
    float} out -- never touches NumPy, so a future multi-stage GPU
    pipeline can chain stages without a host round trip between them.
    """
    FilterConfig(median_kernel_size=kernel_size)
    return xray_cuda.median_basic_gpu(gpu_image, kernel_size)


# -- Section 7: Enhanced CUDA median -----------------------
#
# median_basic (Section 4C) is unchanged. Three variants -- "shared"
# (shared-memory tiling, runtime kernel_size), "network3x3" (+branchless
# sorting network, kernel_size must be 3), "specialized" (+compile-time
# kernel_size, unrolled; kernel_size in {3,5,7}) -- see
# cuda/include/median_enhanced.cuh for why Median needed a different
# optimization strategy from Gaussian (nonlinear, non-separable) and why
# a full sorting network was only hand-derived for 3x3.

MEDIAN_VARIANTS = ("shared", "network3x3", "specialized")
_MEDIAN_VARIANT_TO_INT = {name: i for i, name in enumerate(MEDIAN_VARIANTS)}

DEFAULT_MEDIAN_ENHANCED_BLOCK = (16, 16)  # Section 7 Optimization 5 tuning result; see README


def median_variant_to_int(variant: str) -> int:
    """Validate `variant` and return the int the compiled extension
    expects. Public (not underscore-prefixed) since cuda/pipeline.py also
    needs this mapping and should not duplicate it."""
    if variant not in _MEDIAN_VARIANT_TO_INT:
        raise ValueError(f"variant={variant!r} is not supported; expected one of {MEDIAN_VARIANTS}")
    return _MEDIAN_VARIANT_TO_INT[variant]


def median_enhanced_cuda(
    image: np.ndarray, kernel_size: int = 3, variant: str = "network3x3", block: tuple = DEFAULT_MEDIAN_ENHANCED_BLOCK
) -> np.ndarray:
    """Upload `image`, run the requested Enhanced CUDA median variant,
    download and return the uint8 [H, W] result. `variant` is one of
    MEDIAN_VARIANTS; "network3x3" requires kernel_size=3, "specialized"
    requires kernel_size in {3,5,7}. Exact equality vs CPU and vs Basic
    CUDA is the standard here -- no tolerance (Median has no floating-
    point accumulation).
    """
    variant_int = median_variant_to_int(variant)
    FilterConfig(median_kernel_size=kernel_size)  # raises ValueError if invalid

    gpu_image = xray_cuda.upload_image(image)
    result = xray_cuda.median_enhanced_gpu(gpu_image, kernel_size, variant_int, block[0], block[1])
    return xray_cuda.download_image(result["output"])


def median_enhanced_cuda_gpu(
    gpu_image, kernel_size: int = 3, variant: str = "network3x3", block: tuple = DEFAULT_MEDIAN_ENHANCED_BLOCK
) -> dict:
    """GPU-native variant: GpuImage in, {'output': GpuImage, 'kernel_ms':
    float} out -- no host round trip.
    """
    variant_int = median_variant_to_int(variant)
    FilterConfig(median_kernel_size=kernel_size)
    return xray_cuda.median_enhanced_gpu(gpu_image, kernel_size, variant_int, block[0], block[1])
