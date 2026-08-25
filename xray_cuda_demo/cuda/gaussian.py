"""Python-level Basic CUDA Gaussian blur API.

Coefficient generation happens here, in Python, via cv2.getGaussianKernel
-- the exact same OpenCV function cpu/filters.py's cv2.GaussianBlur uses
internally. OpenCV special-cases kernel_size in {3,5,7} with sigma<=0 to
a hardcoded coefficient table rather than the sigma formula (verified by
inspection: cv2.getGaussianKernel(5, 0.0) == [0.0625, 0.25, 0.375, 0.25,
0.0625], not what the 0.3*((ksize-1)*0.5-1)+0.8 formula alone would
produce for every case). Calling cv2.getGaussianKernel directly avoids
having to reimplement that split in CUDA and guarantees the GPU kernel
convolves with bit-identical coefficients to the CPU reference.
"""

from __future__ import annotations

import cv2
import numpy as np

import xray_cuda
from cpu.filters import FilterConfig  # reuses existing kernel_size/sigma validation


def gaussian_kernel_1d(kernel_size: int, sigma: float) -> np.ndarray:
    """1-D Gaussian coefficients (cv2.getGaussianKernel's raw output),
    for the Section 6 separable Enhanced kernels -- gaussian_basic
    (Section 4B) uses the 2-D outer product instead (see
    gaussian_kernel_2d); Enhanced applies this 1-D vector once
    horizontally and once vertically, so no outer product is needed.
    float32, shape (kernel_size,).
    """
    kernel_1d = cv2.getGaussianKernel(kernel_size, sigma)  # (k, 1) float64
    return np.ascontiguousarray(kernel_1d.ravel(), dtype=np.float32)


def gaussian_kernel_2d(kernel_size: int, sigma: float) -> np.ndarray:
    """2-D Gaussian coefficients, bit-identical to what cv2.GaussianBlur
    uses internally for a square kernel (sigmaY defaults to sigmaX -- see
    cpu/filters.py apply_gaussian). float32, shape (kernel_size, kernel_size).
    """
    kernel_1d = cv2.getGaussianKernel(kernel_size, sigma)  # (k, 1) float64
    kernel_2d = kernel_1d @ kernel_1d.T
    return np.ascontiguousarray(kernel_2d, dtype=np.float32)


def gaussian_cuda(image: np.ndarray, kernel_size: int = 5, sigma: float = 0.0) -> np.ndarray:
    """Upload `image`, run the Basic CUDA Gaussian kernel, download and
    return the uint8 [H, W] result. Validates kernel_size/sigma with the
    same rules as the Section 3 CPU reference (FilterConfig), so an
    invalid combination fails the same way on both implementations.
    """
    FilterConfig(gaussian_kernel_size=kernel_size, gaussian_sigma=sigma)  # raises ValueError if invalid

    coeffs = gaussian_kernel_2d(kernel_size, sigma)
    gpu_image = xray_cuda.upload_image(image)
    result = xray_cuda.gaussian_basic_gpu(gpu_image, coeffs)
    return xray_cuda.download_image(result["output"])


def gaussian_cuda_gpu(gpu_image, kernel_size: int = 5, sigma: float = 0.0) -> dict:
    """GPU-native variant: GpuImage in, {'output': GpuImage, ...} out --
    never touches NumPy, so a future multi-stage GPU pipeline can chain
    stages without a host round trip between them. Returns the same dict
    shape as xray_cuda.gaussian_basic_gpu (output/kernel_ms/coeff_upload_ms).
    """
    FilterConfig(gaussian_kernel_size=kernel_size, gaussian_sigma=sigma)
    coeffs = gaussian_kernel_2d(kernel_size, sigma)
    return xray_cuda.gaussian_basic_gpu(gpu_image, coeffs)


# -- Section 6: Enhanced (separable) CUDA Gaussian -----------------------

GAUSSIAN_VARIANTS = ("naive", "shared", "shared_const", "specialized")
_VARIANT_TO_INT = {name: i for i, name in enumerate(GAUSSIAN_VARIANTS)}

DEFAULT_ENHANCED_BLOCK = (16, 16)  # matches gaussian_basic's default_block_dim(); tuned in Section 6 Optimization 5


def variant_to_int(variant: str) -> int:
    """Validate `variant` and return the int the compiled extension
    expects. Public (not underscore-prefixed) since cuda/pipeline.py also
    needs this mapping and should not duplicate it."""
    if variant not in _VARIANT_TO_INT:
        raise ValueError(f"variant={variant!r} is not supported; expected one of {GAUSSIAN_VARIANTS}")
    return _VARIANT_TO_INT[variant]




def gaussian_enhanced_cuda(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 0.0,
    variant: str = "specialized",
    block: tuple = DEFAULT_ENHANCED_BLOCK,
) -> np.ndarray:
    """Upload `image`, run the requested Enhanced CUDA Gaussian variant,
    download and return the uint8 [H, W] result. `variant` is one of
    GAUSSIAN_VARIANTS ("naive"=separable only, "shared"=+shared memory,
    "shared_const"=+constant memory, "specialized"=+compile-time
    kernel_size unrolling; the last requires kernel_size in {3,5,7,9}).
    """
    variant_int = variant_to_int(variant)
    FilterConfig(gaussian_kernel_size=kernel_size, gaussian_sigma=sigma)

    coeffs = gaussian_kernel_1d(kernel_size, sigma)
    gpu_image = xray_cuda.upload_image(image)
    result = xray_cuda.gaussian_enhanced_gpu(gpu_image, coeffs, variant_int, block[0], block[1])
    return xray_cuda.download_image(result["output"])


def gaussian_enhanced_cuda_gpu(
    gpu_image,
    kernel_size: int = 5,
    sigma: float = 0.0,
    variant: str = "specialized",
    block: tuple = DEFAULT_ENHANCED_BLOCK,
) -> dict:
    """GPU-native variant: GpuImage in, {'output': GpuImage, 'coeff_upload_ms',
    'horizontal_ms', 'vertical_ms', 'kernel_ms'} out -- no host round trip.
    """
    variant_int = variant_to_int(variant)
    FilterConfig(gaussian_kernel_size=kernel_size, gaussian_sigma=sigma)
    coeffs = gaussian_kernel_1d(kernel_size, sigma)
    return xray_cuda.gaussian_enhanced_gpu(gpu_image, coeffs, variant_int, block[0], block[1])
