#pragma once

#include "gpu_image.cuh"
#include <utility>

namespace xray_cuda {

// --------------------------------------------------------------------------
// Basic (naive) CUDA Gaussian blur.
//
// This mirrors cpu/filters.py::apply_gaussian exactly:
//   cv2.GaussianBlur(image, (k, k), sigmaX=sigma, borderType=cv2.BORDER_DEFAULT)
//
// Coefficient generation: deliberately NOT reimplemented here. OpenCV's
// getGaussianKernel(ksize, sigma) uses a *hardcoded coefficient table*
// for ksize in {3,5,7} when sigma<=0 (not the sigma formula!), and only
// falls back to the 0.3*((ksize-1)*0.5-1)+0.8 formula for other sizes
// (e.g. ksize=9) or explicit sigma>0. Reimplementing that split in CUDA
// would risk silently diverging from the CPU reference. Instead, Python
// calls cv2.getGaussianKernel directly (see cuda/gaussian.py) and passes
// the resulting kernel_size x kernel_size float32 coefficients down to
// this launcher, which only uploads them and applies the convolution --
// guaranteeing bit-identical coefficients to whatever cv2.GaussianBlur
// uses internally.
//
// Border policy: BORDER_REFLECT_101 (== cv2.BORDER_DEFAULT), implemented
// explicitly via reflect101() below -- see cpu/filters.py's BORDER_POLICY
// comment for why this one policy is used everywhere.
//
// Algorithm: one thread per output pixel, full (non-separable) 2D
// convolution read directly from global memory, float32 accumulation,
// single round+clamp to uint8 at the end (__float2int_rn matches
// OpenCV's round-half-to-even via cvRound more closely than roundf's
// round-half-away-from-zero). No shared memory, no separability -- this
// is intentionally the naive baseline; see Section 4B spec section 32.
// --------------------------------------------------------------------------

struct GaussianTiming {
    float coeff_upload_ms;
    float kernel_ms;
};

__global__ void gaussian_basic_kernel(
    const uint8_t* input, uint8_t* output,
    int width, int height,
    const float* coeffs, int kernel_size);

// `host_coeffs` must point to kernel_size*kernel_size row-major float32
// values (see cuda/gaussian.py::gaussian_kernel_2d). Allocates a new
// output GpuImage; `input` is not modified.
std::pair<std::unique_ptr<GpuImage>, GaussianTiming> gaussian_basic(
    const GpuImage& input, const float* host_coeffs, int kernel_size);

}  // namespace xray_cuda
