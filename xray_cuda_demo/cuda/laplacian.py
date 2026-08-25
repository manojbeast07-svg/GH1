"""Python-level Basic CUDA Laplacian filter API.

Coefficient generation happens here, in Python, via cv2.Laplacian's
*impulse response* -- not the textbook [[0,1,0],[1,-4,1],[0,1,0]]
kernel. Verified empirically (Section 4E, do not assume): only
kernel_size=1 produces that classic kernel. kernel_size=3 actually
produces [[2,0,2],[0,-8,0],[2,0,2]] (cv2.Laplacian(ksize=3) is
internally Sobel(dx=2,ksize=3) + Sobel(dy=2,ksize=3), not the simple
discrete Laplacian scaled up), and kernel_size=5 a still-different 5x5
kernel. Rather than hand-deriving and combining the separable
getDerivKernels() outputs (error-prone for asymmetric intermediate
shapes -- confirmed by getting it wrong on the first attempt for
kernel_size=1 during verification), the exact effective 2D kernel is
recovered by convolving a single impulse pixel and reading off the
response, which is correct by construction for any kernel_size.
"""

from __future__ import annotations

import cv2
import numpy as np

import xray_cuda
from cpu.filters import FilterConfig  # reuses existing kernel_size validation


def laplacian_kernel_2d(kernel_size: int) -> np.ndarray:
    """Exact 2-D Laplacian coefficients for `kernel_size`, bit-identical
    to what cv2.Laplacian uses internally, recovered via impulse
    response. float32, square.

    Note: the returned array's shape is NOT necessarily
    (kernel_size, kernel_size) -- `kernel_size` is OpenCV's aperture-size
    parameter, not a literal coefficient-matrix dimension. In particular
    kernel_size=1 (a special case, not a formula-derived aperture) still
    has a 3x3 coefficient support ([[0,1,0],[1,-4,1],[0,1,0]]). The
    caller (and the CUDA binding) use this array's actual shape, not the
    `kernel_size` argument, to determine the convolution radius.
    """
    size = kernel_size + 6  # margin so the response's support never touches the array edge
    impulse = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    impulse[center, center] = 1.0
    response = cv2.Laplacian(impulse, cv2.CV_32F, ksize=kernel_size, scale=1.0, delta=0.0, borderType=cv2.BORDER_CONSTANT)

    nonzero = np.argwhere(np.abs(response) > 1e-6)
    y0, x0 = nonzero.min(axis=0)
    y1, x1 = nonzero.max(axis=0)
    kernel = response[y0 : y1 + 1, x0 : x1 + 1]
    if kernel.shape[0] != kernel.shape[1] or kernel.shape[0] % 2 == 0:
        raise RuntimeError(
            f"Recovered Laplacian kernel has unexpected shape {kernel.shape} for kernel_size={kernel_size} "
            "(expected square, odd-sized support)"
        )
    return np.ascontiguousarray(kernel, dtype=np.float32)


def laplacian_cuda(image: np.ndarray, kernel_size: int = 3, scale: float = 1.0, delta: float = 0.0) -> np.ndarray:
    """Upload `image`, run the Basic CUDA Laplacian kernel, download and
    return the uint8 [H, W] result. Validates kernel_size with the same
    rule as the Section 3 CPU reference (FilterConfig); scale/delta are
    unrestricted floats, matching the CPU reference (which does not
    validate them either).
    """
    FilterConfig(laplacian_kernel_size=kernel_size)  # raises ValueError if kernel_size invalid

    coeffs = laplacian_kernel_2d(kernel_size)
    gpu_image = xray_cuda.upload_image(image)
    result = xray_cuda.laplacian_basic_gpu(gpu_image, coeffs, scale, delta)
    return xray_cuda.download_image(result["output"])


def laplacian_cuda_gpu(gpu_image, kernel_size: int = 3, scale: float = 1.0, delta: float = 0.0) -> dict:
    """GPU-native variant: GpuImage in, {'output': GpuImage, ...} out --
    never touches NumPy, so a future multi-stage GPU pipeline can chain
    stages without a host round trip between them.
    """
    FilterConfig(laplacian_kernel_size=kernel_size)
    coeffs = laplacian_kernel_2d(kernel_size)
    return xray_cuda.laplacian_basic_gpu(gpu_image, coeffs, scale, delta)


# -- Section 9: Enhanced CUDA Laplacian filter -----------------------

LAPLACIAN_VARIANTS = ("shared", "shared_const", "specialized", "explicit")
_LAPLACIAN_VARIANT_TO_INT = {name: i for i, name in enumerate(LAPLACIAN_VARIANTS)}
DEFAULT_LAPLACIAN_ENHANCED_BLOCK = (16, 16)


def laplacian_variant_to_int(variant: str) -> int:
    if variant not in _LAPLACIAN_VARIANT_TO_INT:
        raise ValueError(f"variant={variant!r} is not supported; expected one of {LAPLACIAN_VARIANTS}")
    return _LAPLACIAN_VARIANT_TO_INT[variant]


def laplacian_enhanced_cuda(image: np.ndarray, kernel_size: int = 3, scale: float = 1.0, delta: float = 0.0,
                             variant: str = "specialized", block: tuple = DEFAULT_LAPLACIAN_ENHANCED_BLOCK) -> np.ndarray:
    """Upload `image`, run an Enhanced CUDA Laplacian filter variant,
    download and return the uint8 [H, W] result. Same kernel_size/scale/
    delta contract as laplacian_cuda(); `variant` selects Shared/
    SharedConst/Specialized/Explicit (see cuda/include/laplacian_enhanced.cuh).
    """
    FilterConfig(laplacian_kernel_size=kernel_size)
    variant_int = laplacian_variant_to_int(variant)

    coeffs = laplacian_kernel_2d(kernel_size)
    gpu_image = xray_cuda.upload_image(image)
    result = xray_cuda.laplacian_enhanced_gpu(gpu_image, coeffs, scale, delta, variant_int, block[0], block[1])
    return xray_cuda.download_image(result["output"])


def laplacian_enhanced_cuda_gpu(gpu_image, kernel_size: int = 3, scale: float = 1.0, delta: float = 0.0,
                                 variant: str = "specialized", block: tuple = DEFAULT_LAPLACIAN_ENHANCED_BLOCK) -> dict:
    """GPU-native variant of laplacian_enhanced_cuda(): GpuImage in,
    {'output': GpuImage, 'kernel_ms': float, 'coeff_upload_ms': float} out."""
    FilterConfig(laplacian_kernel_size=kernel_size)
    variant_int = laplacian_variant_to_int(variant)
    coeffs = laplacian_kernel_2d(kernel_size)
    return xray_cuda.laplacian_enhanced_gpu(gpu_image, coeffs, scale, delta, variant_int, block[0], block[1])
