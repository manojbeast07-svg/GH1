#pragma once

#include "gpu_image.cuh"
#include <utility>
#include <vector>

namespace xray_cuda {

// --------------------------------------------------------------------------
// Enhanced CUDA Gaussian blur (Section 6).
//
// gaussian_basic (Section 4B) is UNCHANGED and remains the reference
// "Basic CUDA" baseline -- this file adds a second, separate
// implementation family so the two remain independently benchmarkable
// at any time, per the Section 6 spec's explicit requirement.
//
// Four genuinely distinct, independently measurable kernel variants,
// applied one optimization at a time (spec: "do not implement all
// optimizations in one change"):
//
//   Naive        -- separable (1D horizontal pass, 1D vertical pass)
//                   instead of Basic's full O(k^2) 2D convolution, but
//                   still reads coefficients from global memory and does
//                   no shared-memory tiling. Isolates the gain from
//                   separability alone.
//   Shared       -- Naive + shared-memory tiling (halo-loaded 1D tiles,
//                   one dimension of halo per pass, not two -- a 2D tile
//                   is unnecessary once the convolution is separable).
//                   Isolates the gain from reduced redundant global
//                   memory traffic.
//   SharedConst  -- Shared + coefficients moved from a global-memory
//                   pointer into __constant__ memory. Isolates the gain
//                   from constant-memory's broadcast/cache behavior.
//   Specialized  -- SharedConst with kernel_size as a compile-time
//                   template parameter (radius loop unrolled via
//                   #pragma unroll) instead of a runtime int, for each
//                   of the 4 CPU-supported sizes {3,5,7,9}. Isolates the
//                   gain from eliminating runtime loop bounds/indexing.
//
// --------------------------------------------------------------------------
// Correctness: why the intermediate pass stays float32, not uint8
// --------------------------------------------------------------------------
// A naive separable implementation that rounds/clamps to uint8 after the
// horizontal pass and again after the vertical pass introduces an EXTRA
// rounding step that gaussian_basic's single 2D convolution (one round,
// at the very end) does not have. That would silently widen the
// CPU-vs-GPU difference beyond the already-documented +-1 Basic Gaussian
// tolerance for reasons that have nothing to do with the optimization
// being tested. To keep the enhanced variants comparable to Basic on
// equal footing, the horizontal pass writes float32 (no rounding), and
// only the vertical pass converts to uint8 -- one round, at the end,
// same as Basic.
//
// --------------------------------------------------------------------------
// Border: same reflect101() as Basic (gpu_image.cuh), applied
// independently in each 1D pass's own direction.
// --------------------------------------------------------------------------

enum class GaussianVariant : int {
    Naive = 0,
    Shared = 1,
    SharedConst = 2,
    Specialized = 3,
};

// Coefficients live in constant memory for Shared/SharedConst/Specialized;
// this bound must cover every kernel_size in cpu.filters.ALLOWED_GAUSSIAN_KERNELS.
constexpr int kGaussianEnhancedMaxKernelSize = 9;

struct GaussianEnhancedTiming {
    float coeff_upload_ms;
    float horizontal_ms;
    float vertical_ms;
    float kernel_ms() const { return horizontal_ms + vertical_ms; }
};

// Low-level dispatch: launches the horizontal+vertical kernel pair for
// `variant` directly on caller-provided device buffers (input uint8,
// intermediate float, output uint8; all sized
// batch_size*height*width) -- no allocation. Used by the production
// pipeline (pipeline_basic.cu) to apply Enhanced Gaussian on its own
// existing ping-pong buffers, exactly like the pipeline already does for
// the five Basic kernels; gaussian_enhanced()/gaussian_enhanced_batch()
// below call this internally after allocating their own buffers, so the
// variant-dispatch switch statement exists in exactly one place.
void gaussian_enhanced_dispatch(
    const uint8_t* d_input, float* d_intermediate, uint8_t* d_output,
    int batch_size, int height, int width,
    const float* host_coeffs_1d, int kernel_size,
    GaussianVariant variant, int block_x, int block_y,
    GaussianEnhancedTiming& timing);

// `host_coeffs_1d` is the kernel_size-length 1D Gaussian coefficient
// vector (see cuda/gaussian.py::gaussian_kernel_1d) -- NOT the k*k 2D
// matrix gaussian_basic uses, since these kernels are separable.
// `block_x`/`block_y` select the launch configuration (Section 6
// Optimization 5: block/tile tuning) -- a pure performance parameter,
// does not affect the result.
std::pair<std::unique_ptr<GpuImage>, GaussianEnhancedTiming> gaussian_enhanced(
    const GpuImage& input, const float* host_coeffs_1d, int kernel_size,
    GaussianVariant variant, int block_x, int block_y);

// Batched form -- same kernels (already blockIdx.z-aware), applied to
// every image in `input` via one pair of kernel launches (grid.z =
// batch_size) instead of one pair per image. Added specifically because
// single-image timing showed the two-kernel-launch structure of every
// separable variant losing to Basic's one-launch 2D convolution at the
// dataset's dominant 224x224 scale (fixed per-launch overhead dominates
// at that problem size) -- exactly the risk the Section 6 spec warned
// about. Batching amortizes that fixed overhead across many images per
// launch, the same architectural lever Section 5 used for the Basic
// pipeline; this measures whether it rescues Enhanced Gaussian at
// 224x224 too, rather than assuming it does.
std::pair<std::unique_ptr<GpuImageBatch>, GaussianEnhancedTiming> gaussian_enhanced_batch(
    const GpuImageBatch& input, const float* host_coeffs_1d, int kernel_size,
    GaussianVariant variant, int block_x, int block_y);

}  // namespace xray_cuda
