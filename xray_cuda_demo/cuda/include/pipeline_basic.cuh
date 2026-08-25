#pragma once

#include "gaussian_enhanced.cuh"
#include "gpu_image.cuh"
#include "laplacian_enhanced.cuh"
#include "median_enhanced.cuh"
#include "sobel_enhanced.cuh"
#include "threshold_enhanced.cuh"
#include <cstdint>
#include <memory>
#include <vector>

namespace xray_cuda {

// --------------------------------------------------------------------------
// Production Basic CUDA pipeline (Section 5).
//
// This is the single native entry point that replaces the earlier
// pattern of five separate Python calls (Sections 4B-4F), each of which
// allocated its own GpuImage and returned to Python before the next
// stage. Section 4F measured that five-call overhead directly: ~0.325ms
// of actual CUDA work vs. ~1.035ms of wall time for one 224x224 image
// through all five stages -- ~0.71ms of Python/pybind11 call overhead.
//
// run_basic_cuda_pipeline_batch() runs all five (optionally-disabled)
// stages back to back, entirely inside this one C++ function, using
// exactly two persistent, reused GpuImageBatch buffers (ping-pong) for
// the whole batch:
//
//   H2D -> A
//   A -> Gaussian  -> B   (skip: A stays "current")
//   current -> Median    -> other, swap
//   current -> Sobel     -> other, swap
//   current -> Laplacian -> other, swap
//   current -> Threshold -> other, swap
//   current -> D2H
//
// No buffer is allocated or freed per stage -- only the two batch
// buffers, sized once for this call's batch_size/height/width. No
// filter algorithm is changed or optimized here (still the naive
// per-pixel kernels from Sections 4B-4F); the only new efficiency is
// architectural: fewer Python transitions and reused device memory.
// --------------------------------------------------------------------------

// Mirrors cpu.filters.FilterConfig's fields relevant to kernel dispatch.
// Coefficient arrays (Gaussian, Laplacian) are computed in Python (same
// reasoning as Sections 4B/4E: bit-identical to what OpenCV would use)
// and passed in as plain pointers here.
struct BasicPipelineConfig {
    bool gaussian_enabled = true;
    const float* gaussian_coeffs = nullptr;  // kernel_size*kernel_size, row-major (Basic, 2-D)
    int gaussian_kernel_size = 0;

    // Section 6: when true, the Gaussian stage uses gaussian_enhanced_dispatch()
    // (separable, on this call's own ping-pong buffers -- no extra
    // allocation beyond one intermediate float buffer) instead of
    // gaussian_basic_kernel. gaussian_coeffs_1d (length gaussian_kernel_size,
    // NOT the k*k 2-D array gaussian_coeffs holds) is required when this is set.
    bool gaussian_use_enhanced = false;
    const float* gaussian_coeffs_1d = nullptr;
    GaussianVariant gaussian_variant = GaussianVariant::Specialized;
    int gaussian_block_x = 16;
    int gaussian_block_y = 16;

    bool median_enabled = true;
    int median_kernel_size = 0;

    // Section 7: when true, the Median stage uses median_enhanced_dispatch()
    // (on this call's own ping-pong buffers, no extra allocation --
    // Median needs no intermediate buffer, unlike Gaussian) instead of
    // median_basic_kernel.
    bool median_use_enhanced = false;
    MedianVariant median_variant = MedianVariant::Network3x3;
    int median_block_x = 16;
    int median_block_y = 16;

    bool sobel_enabled = true;
    int sobel_mode = 0;  // SobelMode as int

    // Section 8: when true, the Sobel stage uses sobel_enhanced_dispatch()
    // instead of sobel_basic_kernel. Separable (unlike Shared/SharedConst/
    // Specialized) needs its own intermediate float buffers -- managed
    // internally by sobel_enhanced_dispatch(), not by this pipeline's
    // ping-pong buffers (see sobel_enhanced.cuh for why).
    bool sobel_use_enhanced = false;
    SobelVariant sobel_variant = SobelVariant::Specialized;
    int sobel_block_x = 16;
    int sobel_block_y = 16;

    bool laplacian_enabled = true;
    const float* laplacian_coeffs = nullptr;
    int laplacian_kernel_size = 0;
    float laplacian_scale = 1.0f;
    float laplacian_delta = 0.0f;

    // Section 9: when true, the Laplacian stage uses
    // laplacian_enhanced_dispatch() instead of laplacian_basic_kernel.
    bool laplacian_use_enhanced = false;
    LaplacianVariant laplacian_variant = LaplacianVariant::Specialized;
    int laplacian_block_x = 16;
    int laplacian_block_y = 16;

    bool threshold_enabled = true;
    uint8_t threshold_value = 128;
    uint8_t threshold_max_value = 255;

    // Section 10: when true, the Threshold stage uses
    // threshold_enhanced_dispatch() instead of threshold_basic_kernel.
    bool threshold_use_enhanced = false;
    ThresholdVariant threshold_variant = ThresholdVariant::Vectorized;
    int threshold_block_x = 16;
    int threshold_block_y = 16;
};

// Per-stage timings in milliseconds. A disabled stage's field is -1.0f
// (sentinel meaning "did not run" -- mirrors cpu.pipeline.TimingResult's
// use of None for the same purpose; the Python binding translates -1.0f
// to None so a caller never mistakes "skipped" for "ran instantly").
struct BasicPipelineTiming {
    float h2d_ms = 0.0f;
    float gaussian_ms = -1.0f;
    float median_ms = -1.0f;
    float sobel_ms = -1.0f;
    float laplacian_ms = -1.0f;
    float threshold_ms = -1.0f;
    float d2h_ms = 0.0f;

    float compute_ms() const {
        float total = 0.0f;
        for (float v : {gaussian_ms, median_ms, sobel_ms, laplacian_ms, threshold_ms}) {
            if (v >= 0.0f) total += v;
        }
        return total;
    }

    float total_ms() const { return h2d_ms + compute_ms() + d2h_ms; }
};

// Runs the full (sub-)pipeline on a batch of `batch_size` images, each
// `height`x`width`, read from `host_batch` (row-major
// [batch_size, height, width], contiguous, uint8). Returns the final
// output batch (same shape) as a host-side byte vector, plus timing.
//
// Disabled stages are skipped entirely (no kernel launch, no buffer
// swap) -- matches cpu.pipeline.run_cpu_pipeline's "skip, don't
// recompute" semantics. If every stage is disabled, the output is a
// copy of the input (still goes through H2D+D2H, since this function's
// contract is "upload once, process, download once", not a no-op
// shortcut).
std::pair<std::vector<uint8_t>, BasicPipelineTiming> run_basic_cuda_pipeline_batch(
    const uint8_t* host_batch, int batch_size, int height, int width, const BasicPipelineConfig& config);

}  // namespace xray_cuda
