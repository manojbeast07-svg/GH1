"""Python-level Basic CUDA Sobel edge detection API.

Coefficient/border/conversion verification (Section 4D spec: do not
assume): before writing any CUDA code, the exact 3x3 Gx/Gy kernels were
confirmed via cv2.getDerivKernels(1, 0, 3) and cv2.getDerivKernels(0, 1,
3) (they multiply out to the classic [[-1,0,1],[-2,0,2],[-1,0,1]] and
[[-1,-2,-1],[0,0,0],[1,2,1]] matrices), and a hand-computed
reflect-101-border 3x3 convolution matched cv2.Sobel's actual output
bit-for-bit on a real X-ray (max_abs_diff == 0.0). The final
round(|raw|) clamped to [0,255] conversion was verified the same way
against cpu/filters.py::apply_sobel for all four modes, in float32
precision -- also bit-exact.

Only kernel_size=3 (the FilterConfig default) is supported by this
Basic implementation -- see cuda/include/sobel_basic.cuh for why.
"""

from __future__ import annotations

import numpy as np

import xray_cuda
from cpu.filters import FilterConfig, SOBEL_MODES  # reuses existing mode validation

_MODE_TO_INT = {"x": 0, "y": 1, "magnitude": 2, "abs_sum": 3}  # must match SobelMode in sobel_basic.cuh


def mode_to_int(mode: str) -> int:
    """Validate `mode` and return the int the compiled extension expects.
    Public (not underscore-prefixed) since cuda/benchmark.py also needs
    this mapping and should not duplicate it."""
    if mode not in _MODE_TO_INT:
        raise ValueError(f"mode={mode!r} is not supported; expected one of {sorted(SOBEL_MODES)}")
    return _MODE_TO_INT[mode]


def sobel_cuda(image: np.ndarray, mode: str = "magnitude") -> np.ndarray:
    """Upload `image`, run the Basic CUDA Sobel kernel, download and
    return the uint8 [H, W] result. `mode` is one of "x", "y",
    "magnitude", "abs_sum" -- validated the same way FilterConfig
    validates sobel_mode for the CPU reference.
    """
    mode_int = mode_to_int(mode)
    FilterConfig(sobel_mode=mode, sobel_kernel_size=3)  # raises ValueError if mode is otherwise invalid

    gpu_image = xray_cuda.upload_image(image)
    result = xray_cuda.sobel_basic_gpu(gpu_image, mode_int)
    return xray_cuda.download_image(result["output"])


def sobel_cuda_gpu(gpu_image, mode: str = "magnitude") -> dict:
    """GPU-native variant: GpuImage in, {'output': GpuImage, 'kernel_ms':
    float} out -- never touches NumPy, so a future multi-stage GPU
    pipeline can chain stages without a host round trip between them.
    """
    mode_int = mode_to_int(mode)
    FilterConfig(sobel_mode=mode, sobel_kernel_size=3)
    return xray_cuda.sobel_basic_gpu(gpu_image, mode_int)


# -- Section 8: Enhanced CUDA Sobel edge detection -----------------------

SOBEL_VARIANTS = ("shared", "shared_const", "specialized", "separable")
_SOBEL_VARIANT_TO_INT = {name: i for i, name in enumerate(SOBEL_VARIANTS)}
DEFAULT_SOBEL_ENHANCED_BLOCK = (16, 16)


def sobel_variant_to_int(variant: str) -> int:
    if variant not in _SOBEL_VARIANT_TO_INT:
        raise ValueError(f"variant={variant!r} is not supported; expected one of {SOBEL_VARIANTS}")
    return _SOBEL_VARIANT_TO_INT[variant]


def sobel_enhanced_cuda(image: np.ndarray, mode: str = "magnitude", variant: str = "specialized",
                         block: tuple = DEFAULT_SOBEL_ENHANCED_BLOCK) -> np.ndarray:
    """Upload `image`, run an Enhanced CUDA Sobel filter variant, download
    and return the uint8 [H, W] result. Same mode/validation contract as
    sobel_cuda(); `variant` selects Shared/SharedConst/Specialized/Separable
    (see cuda/include/sobel_enhanced.cuh)."""
    mode_int = mode_to_int(mode)
    variant_int = sobel_variant_to_int(variant)
    FilterConfig(sobel_mode=mode, sobel_kernel_size=3)

    gpu_image = xray_cuda.upload_image(image)
    result = xray_cuda.sobel_enhanced_gpu(gpu_image, mode_int, variant_int, block[0], block[1])
    return xray_cuda.download_image(result["output"])


def sobel_enhanced_cuda_gpu(gpu_image, mode: str = "magnitude", variant: str = "specialized",
                             block: tuple = DEFAULT_SOBEL_ENHANCED_BLOCK) -> dict:
    """GPU-native variant of sobel_enhanced_cuda(): GpuImage in,
    {'output': GpuImage, 'kernel_ms': float} out."""
    mode_int = mode_to_int(mode)
    variant_int = sobel_variant_to_int(variant)
    FilterConfig(sobel_mode=mode, sobel_kernel_size=3)
    return xray_cuda.sobel_enhanced_gpu(gpu_image, mode_int, variant_int, block[0], block[1])
