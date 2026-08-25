#pragma once

#include "gpu_image.cuh"
#include "median_basic.cuh"
#include <utility>

namespace xray_cuda {

// --------------------------------------------------------------------------
// Enhanced CUDA median filter (Section 7).
//
// median_basic (Section 4C) is UNCHANGED and remains the reference
// "Basic CUDA" baseline. Median is fundamentally different from
// Gaussian/Sobel/Laplacian (Section 6): it is nonlinear, non-separable,
// and selection- rather than accumulation-based, so the Gaussian
// optimization strategy (separable passes) does not apply here. Three
// genuinely distinct, independently measurable variants:
//
//   Shared      -- Optimization 1. Same runtime-kernel_size insertion
//                  sort as Basic, but the k x k neighborhood is read
//                  from a shared-memory tile (2-D halo, since median is
//                  not separable -- unlike Gaussian's 1-D-per-pass halo)
//                  instead of global memory. Isolates the gain from
//                  reduced redundant global memory traffic alone.
//   Network3x3  -- Optimization 2 for kernel_size=3 only. Shared-memory
//                  tile (same as Shared), but the median of the 9
//                  values is computed via a fixed branchless
//                  compare-exchange sorting network (Nicolas Devillard's
//                  public-domain opt_med9, 19 compare-exchanges using
//                  min/max -- no data-dependent branching) instead of
//                  insertion sort. This is also this project's answer to
//                  "branch reduction" (Optimization 4) for 3x3: the
//                  network has none of insertion sort's data-dependent
//                  branches.
//   Specialized -- Optimization 2 for kernel_size in {3,5,7}. Shared
//                  memory (same tile), kernel_size as a compile-time
//                  template parameter, both the neighborhood gather and
//                  the insertion sort fully unrolled (#pragma unroll).
//                  A full hand-derived sorting network was only built
//                  for 3x3 (9 elements, a well-established, easily
//                  independently-verified public algorithm); deriving
//                  and verifying a correct network for 25 or 49 elements
//                  by hand was judged too error-prone for the benefit
//                  relative to unrolled+specialized insertion sort --
//                  a deliberate, documented scope decision (Section 7
//                  spec item 17 explicitly permits this: "if the
//                  implementation becomes excessively complex... do not
//                  assume 7x7 requires the same strategy as 3x3").
//
// Border: BORDER_REPLICATE, via the shared clamp_index() in gpu_image.cuh
// -- identical to median_basic, verified by the exact-match test suite
// (Section 7 requires max_abs_diff == 0, no tolerance, for every variant).
// --------------------------------------------------------------------------

enum class MedianVariant : int {
    Shared = 0,
    Network3x3 = 1,   // kernel_size must be 3
    Specialized = 2,  // kernel_size must be in {3,5,7}
};

struct MedianEnhancedTiming {
    float kernel_ms;
};

// Low-level dispatch: launches the requested variant directly on
// caller-provided device buffers (input/output uint8, sized
// batch_size*height*width) -- no allocation. Used by the production
// pipeline (pipeline_basic.cu) to apply Enhanced Median on its own
// existing ping-pong buffers, exactly as it already does for Enhanced
// Gaussian (Section 6); median_enhanced()/median_enhanced_batch() below
// call this internally after allocating their own buffers.
void median_enhanced_dispatch(
    const uint8_t* d_input, uint8_t* d_output,
    int batch_size, int height, int width, int kernel_size,
    MedianVariant variant, int block_x, int block_y,
    MedianEnhancedTiming& timing);

std::pair<std::unique_ptr<GpuImage>, MedianEnhancedTiming> median_enhanced(
    const GpuImage& input, int kernel_size, MedianVariant variant, int block_x, int block_y);

std::pair<std::unique_ptr<GpuImageBatch>, MedianEnhancedTiming> median_enhanced_batch(
    const GpuImageBatch& input, int kernel_size, MedianVariant variant, int block_x, int block_y);

}  // namespace xray_cuda
