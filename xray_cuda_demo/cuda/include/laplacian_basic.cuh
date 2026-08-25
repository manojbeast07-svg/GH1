#pragma once

#include "gpu_image.cuh"
#include <utility>

namespace xray_cuda {

// --------------------------------------------------------------------------
// Basic (naive) CUDA Laplacian filter.
//
// Mirrors cpu/filters.py::apply_laplacian exactly:
//   raw = cv2.Laplacian(image, CV_32F, ksize, scale, delta, borderType=cv2.BORDER_DEFAULT)
//   return cv2.convertScaleAbs(raw)
//
// Coefficient generation: NOT reimplemented here, and deliberately NOT
// the textbook [[0,1,0],[1,-4,1],[0,1,0]] kernel for every kernel_size --
// verified empirically (Section 4E) that only kernel_size=1 produces
// that kernel; kernel_size=3 actually produces
// [[2,0,2],[0,-8,0],[2,0,2]] (cv2.Laplacian(ksize=3) is internally
// Sobel(dx=2,ksize=3) + Sobel(dy=2,ksize=3), not the simple discrete
// Laplacian), and kernel_size=5 a still-different 5x5 kernel. Python
// recovers the exact effective kernel via impulse response (see
// cuda/laplacian.py::laplacian_kernel_2d) and passes it down here,
// exactly like Gaussian's coefficient-passing design -- this
// guarantees bit-identical coefficients to whatever cv2.Laplacian uses
// internally, for any of the three supported kernel sizes, without
// having to reverse-engineer OpenCV's derivative-kernel combination
// logic in C++.
//
// Border: BORDER_REFLECT_101 / BORDER_DEFAULT, via the shared
// reflect101() in gpu_image.cuh -- verified empirically (not assumed)
// against a real X-ray, bit-exact.
//
// Output conversion (matches cv2.convertScaleAbs, verified empirically
// bit-exact including with non-default scale/delta):
//   raw = sum(coeffs * neighborhood)
//   scaled = raw * scale + delta
//   out = clamp(round(|scaled|), 0, 255)
//
// Algorithm: one thread per output pixel, full non-separable 2D
// convolution read directly from global memory (coefficients live in a
// small uploaded device buffer, not shared memory), float32
// accumulation -- intentionally the naive baseline (Section 4E spec).
// --------------------------------------------------------------------------

struct LaplacianTiming {
    float coeff_upload_ms;
    float kernel_ms;
};

__global__ void laplacian_basic_kernel(
    const uint8_t* input, uint8_t* output,
    int width, int height,
    const float* coeffs, int kernel_size,
    float scale, float delta);

// `host_coeffs` must point to kernel_size*kernel_size row-major float32
// values (see cuda/laplacian.py::laplacian_kernel_2d). Allocates a new
// output GpuImage; `input` is not modified.
std::pair<std::unique_ptr<GpuImage>, LaplacianTiming> laplacian_basic(
    const GpuImage& input, const float* host_coeffs, int kernel_size, float scale, float delta);

}  // namespace xray_cuda
