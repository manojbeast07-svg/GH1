"""Python-level Basic CUDA binary threshold API.

Semantics verified empirically before writing the kernel (Section 4F,
do not assume): cv2.threshold(..., THRESH_BINARY) is a strict
`pixel > threshold_value -> max_value, else -> 0` comparison --
pixel == threshold_value maps to 0, not max_value. Confirmed with pixels
at threshold-1/threshold/threshold+1 around threshold_value=128, and
edge cases threshold_value=0 and threshold_value=255 (the latter maps
every pixel to 0, since nothing can exceed 255 -- a legitimate result,
not an error).
"""

from __future__ import annotations

import numpy as np

import xray_cuda
from cpu.filters import FilterConfig  # reuses existing threshold_value/max_value range validation


def threshold_cuda(image: np.ndarray, threshold_value: int = 128, max_value: int = 255) -> np.ndarray:
    """Upload `image`, run the Basic CUDA threshold kernel, download and
    return the uint8 [H, W] result. Validates threshold_value/max_value
    with the same [0,255] range rule as the Section 3 CPU reference
    (FilterConfig).
    """
    FilterConfig(threshold_value=threshold_value, threshold_max_value=max_value)  # raises ValueError if out of range

    gpu_image = xray_cuda.upload_image(image)
    result = xray_cuda.threshold_basic_gpu(gpu_image, threshold_value, max_value)
    return xray_cuda.download_image(result["output"])


def threshold_cuda_gpu(gpu_image, threshold_value: int = 128, max_value: int = 255) -> dict:
    """GPU-native variant: GpuImage in, {'output': GpuImage, 'kernel_ms':
    float} out -- never touches NumPy, so the full GPU-resident pipeline
    (Gaussian -> Median -> Sobel -> Laplacian -> Threshold) never leaves
    the device between stages.
    """
    FilterConfig(threshold_value=threshold_value, threshold_max_value=max_value)
    return xray_cuda.threshold_basic_gpu(gpu_image, threshold_value, max_value)


# -- Section 10: Enhanced CUDA binary threshold -----------------------

THRESHOLD_VARIANTS = ("vectorized", "multi_pixel")
_THRESHOLD_VARIANT_TO_INT = {name: i for i, name in enumerate(THRESHOLD_VARIANTS)}
DEFAULT_THRESHOLD_ENHANCED_BLOCK = (16, 16)


def threshold_variant_to_int(variant: str) -> int:
    if variant not in _THRESHOLD_VARIANT_TO_INT:
        raise ValueError(f"variant={variant!r} is not supported; expected one of {THRESHOLD_VARIANTS}")
    return _THRESHOLD_VARIANT_TO_INT[variant]


def threshold_enhanced_cuda(image: np.ndarray, threshold_value: int = 128, max_value: int = 255,
                             variant: str = "vectorized", block: tuple = DEFAULT_THRESHOLD_ENHANCED_BLOCK) -> np.ndarray:
    """Upload `image`, run an Enhanced CUDA threshold variant, download
    and return the uint8 [H, W] result. Same comparison semantics as
    threshold_cuda(); `variant` selects Vectorized/MultiPixel (see
    cuda/include/threshold_enhanced.cuh)."""
    FilterConfig(threshold_value=threshold_value, threshold_max_value=max_value)
    variant_int = threshold_variant_to_int(variant)

    gpu_image = xray_cuda.upload_image(image)
    result = xray_cuda.threshold_enhanced_gpu(gpu_image, threshold_value, max_value, variant_int, block[0], block[1])
    return xray_cuda.download_image(result["output"])


def threshold_enhanced_cuda_gpu(gpu_image, threshold_value: int = 128, max_value: int = 255,
                                 variant: str = "vectorized", block: tuple = DEFAULT_THRESHOLD_ENHANCED_BLOCK) -> dict:
    """GPU-native variant of threshold_enhanced_cuda(): GpuImage in,
    {'output': GpuImage, 'kernel_ms': float} out."""
    FilterConfig(threshold_value=threshold_value, threshold_max_value=max_value)
    variant_int = threshold_variant_to_int(variant)
    return xray_cuda.threshold_enhanced_gpu(gpu_image, threshold_value, max_value, variant_int, block[0], block[1])
