#pragma once

#include "gpu_image.cuh"
#include <utility>

namespace xray_cuda {

// --------------------------------------------------------------------------
// Experimental fused Laplacian+Threshold kernel (Section 10, spec items
// 10-11, 31).
//
// A single kernel that computes Specialized Enhanced Laplacian (Section
// 9's winning variant: shared-memory tile + constant-memory coefficients
// + compile-time matrix-size unrolling) AND applies the Threshold
// comparison, in one launch, without writing the intermediate Laplacian
// output to global memory at all.
//
// This is NOT wired into the default production pipeline
// (BasicPipelineConfig has no fusion flag) -- it exists solely to
// measure whether skipping one kernel launch + one global-memory
// round-trip of the intermediate buffer is worth the loss of a
// separately-inspectable Laplacian output (needed for visualization/
// debugging/correctness verification per spec item 10). See the
// README's Section 10 "Threshold fusion experiment" for the measured
// keep/reject decision.
//
// Only supports kernel_size in {3, 5} (the same restriction Section 9's
// Specialized Laplacian variant has) and requires the caller's
// coefficient array to be one of the three known, verified sets (same
// validation as Section 9's Explicit Laplacian variant -- never
// re-derived here).
// --------------------------------------------------------------------------

struct FusedLaplacianThresholdTiming {
    float kernel_ms;
};

// `host_coeffs` is the kernel_size*kernel_size actual coefficient matrix
// (see cuda/laplacian.py::laplacian_kernel_2d). Throws if kernel_size is
// not in {3,5}.
void laplacian_threshold_fused_dispatch(
    const uint8_t* d_input, uint8_t* d_output,
    int batch_size, int height, int width,
    const float* host_coeffs, int kernel_size, float scale, float delta,
    uint8_t threshold_value, uint8_t max_value,
    int block_x, int block_y,
    FusedLaplacianThresholdTiming& timing);

}  // namespace xray_cuda
