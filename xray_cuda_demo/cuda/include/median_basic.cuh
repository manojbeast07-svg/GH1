#pragma once

#include "gpu_image.cuh"
#include <utility>

namespace xray_cuda {

// --------------------------------------------------------------------------
// Basic (naive) CUDA median filter.
//
// Mirrors cpu/filters.py::apply_median exactly: cv2.medianBlur(image, k).
//
// Border handling: BORDER_REPLICATE (clamp to nearest edge pixel), NOT
// reflect-101 like Gaussian/Sobel/Laplacian. Determined by empirical
// inspection, not assumed: a 5x5 all-distinct-values test image run
// through cv2.medianBlur(k=3) matched a hand-computed replicate-border
// median exactly, and did not match a reflect-101 hypothesis (see
// cpu/filters.py::apply_median's docstring for the verified conclusion).
//
// Algorithm: one thread per output pixel. Collects the k*k neighborhood
// into a small per-thread local array (max 7x7=49 uint8, since
// ALLOWED_MEDIAN_KERNELS = {3,5,7} in cpu/filters.py), sorts it with a
// plain insertion sort, takes the middle element. No shared memory, no
// sorting network, no float anywhere -- intentionally the naive
// baseline (Section 4C spec). Median is well-defined here without a
// tie-breaking rule: kernel_size is always odd, so k*k is always odd,
// so there's always a unique middle element after sorting.
// --------------------------------------------------------------------------

constexpr int kMedianBasicMaxKernelSize = 7;  // matches cpu/filters.py ALLOWED_MEDIAN_KERNELS max

struct MedianTiming {
    float kernel_ms;
};

__global__ void median_basic_kernel(
    const uint8_t* input, uint8_t* output,
    int width, int height, int kernel_size);

std::pair<std::unique_ptr<GpuImage>, MedianTiming> median_basic(
    const GpuImage& input, int kernel_size);

}  // namespace xray_cuda
