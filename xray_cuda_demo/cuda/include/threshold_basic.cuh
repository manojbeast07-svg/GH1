#pragma once

#include "gpu_image.cuh"
#include <utility>

namespace xray_cuda {

// --------------------------------------------------------------------------
// Basic CUDA binary threshold -- the simplest kernel in this project.
//
// Mirrors cpu/filters.py::apply_threshold exactly:
//   cv2.threshold(image, threshold_value, max_value, cv2.THRESH_BINARY)
//   i.e. pixel > threshold_value -> max_value, else -> 0
//
// Verified empirically (not assumed) before writing this kernel: the
// comparison is strict ">" (pixel == threshold_value maps to 0, not
// max_value), confirmed with pixels at threshold-1/threshold/threshold+1
// around threshold_value=128, and edge cases threshold_value=0 (pixel=0
// maps to 0) and threshold_value=255 (every pixel maps to 0, since
// nothing can exceed 255) -- both legitimate, not errors.
//
// No neighborhood, no border handling -- this is a pure pointwise
// operation, unlike every other Basic CUDA filter in this project. No
// floating-point arithmetic either: pixel and threshold are compared as
// plain uint8 values.
//
// Algorithm: one thread per output pixel, single comparison, single
// write. No shared memory, no multi-pixel-per-thread, no vectorized
// loads -- intentionally the naive baseline (Section 4F spec).
// --------------------------------------------------------------------------

struct ThresholdTiming {
    float kernel_ms;
};

__global__ void threshold_basic_kernel(
    const uint8_t* input, uint8_t* output,
    int width, int height,
    uint8_t threshold_value, uint8_t max_value);

std::pair<std::unique_ptr<GpuImage>, ThresholdTiming> threshold_basic(
    const GpuImage& input, uint8_t threshold_value, uint8_t max_value);

}  // namespace xray_cuda
