#pragma once

#include "gpu_image.cuh"
#include <utility>

namespace xray_cuda {

// --------------------------------------------------------------------------
// Basic (naive) CUDA Sobel edge detection.
//
// Mirrors cpu/filters.py::apply_sobel exactly for kernel_size=3 (the
// FilterConfig default): gx/gy = cv2.Sobel(image, CV_32F, ..., ksize=3,
// borderType=cv2.BORDER_DEFAULT), combined per mode, then
// cv2.convertScaleAbs. Verified empirically (Section 4D) before writing
// any CUDA code: a hand-computed 3x3-kernel + reflect-101-border + round
// +clamp implementation matched apply_sobel's actual output bit-for-bit
// (max_abs_diff == 0) across all four modes on a real X-ray, in both
// float64 and float32 precision.
//
// Only kernel_size=3 is supported. cv2.Sobel's ksize in {1,5,7} uses
// larger, non-trivially-derived separable kernels (via
// cv2.getDerivKernels, not a simple extension of the 3x3 pattern); the
// Section 4D spec frames the Basic implementation entirely around the
// classic 3x3 Gx/Gy coefficient matrices, so generalizing to other
// kernel sizes is out of scope here (median_basic_gpu's kernel_size
// argument is dynamic because the CPU reference genuinely varies its
// *neighborhood size*; Sobel's kernel_size instead changes its
// *coefficients* in a way this Basic baseline doesn't attempt to
// reproduce). Requesting a non-3 kernel_size raises a clear error rather
// than silently producing wrong output.
//
// Coefficients (verified via cv2.getDerivKernels(dx,dy,3), see
// cuda/sobel.py module docstring):
//   Gx = [[-1,0,1],[-2,0,2],[-1,0,1]]
//   Gy = [[-1,-2,-1],[0,0,0],[1,2,1]]
//
// Border: BORDER_REFLECT_101 / BORDER_DEFAULT, via the shared
// reflect101() in gpu_image.cuh -- same as Gaussian, and later
// Laplacian.
//
// Output conversion (matches cv2.convertScaleAbs exactly, verified
// empirically, not assumed): out = clamp(round(|raw|), 0, 255), where
// `raw` is gx, gy, sqrt(gx^2+gy^2), or |gx|+|gy| depending on mode.
// Sign is therefore lost for x/y modes in the uint8 output -- this is
// the CPU reference's own documented behavior (see
// cpu/filters.py::apply_sobel's docstring), not something this CUDA
// kernel introduces or is free to change.
//
// Algorithm: one thread per output pixel, computes gx AND gy together
// from a single 3x3 neighborhood read (avoids two kernel launches for
// what is the same neighborhood), no shared memory, no separable
// implementation -- intentionally the naive baseline (Section 4D spec).
// --------------------------------------------------------------------------

enum class SobelMode : int {
    X = 0,
    Y = 1,
    Magnitude = 2,
    AbsSum = 3,
};

struct SobelTiming {
    float kernel_ms;
};

__global__ void sobel_basic_kernel(
    const uint8_t* input, uint8_t* output,
    int width, int height, int mode);

// `mode` must be one of the SobelMode values (as an int, since pybind11
// passes it through as a plain integer -- see cuda/sobel.py).
std::pair<std::unique_ptr<GpuImage>, SobelTiming> sobel_basic(
    const GpuImage& input, int mode);

}  // namespace xray_cuda
