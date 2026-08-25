#pragma once

#include "gpu_image.cuh"
#include "sobel_basic.cuh"
#include <utility>

namespace xray_cuda {

// --------------------------------------------------------------------------
// Enhanced CUDA Sobel edge detection (Section 8).
//
// sobel_basic (Section 4D) is UNCHANGED and remains the reference
// "Basic CUDA" baseline -- this file adds a second, separate
// implementation family so the two remain independently benchmarkable
// at any time.
//
// -- Why Sobel needs a different strategy than both Gaussian and Median --
// Basic Sobel already computes gx AND gy from a single 3x3 neighborhood
// read inside ONE kernel launch (Section 4D's own design) -- unlike
// Basic Gaussian's separable-but-unexploited O(k^2) 2D convolution, there
// is no "do less arithmetic" lever sitting unused. The real levers here
// are (a) REUSE ACROSS THREADS in a block (adjacent output pixels share
// most of their 3x3 neighborhood -- shared-memory tiling attacks this,
// the same lever that helped Median's Shared variant, though Sobel's
// per-pixel arithmetic is cheap enough that the win is expected to be
// smaller than Gaussian's), and (b) MODE-SPECIFIC WORK -- X only needs
// gx, Y only needs gy, so a mode that discards half the neighborhood's
// contribution can skip computing it entirely (unlike Basic, which
// always computes both regardless of `mode`).
//
// Four variants, one optimization at a time:
//
//   Shared       -- 2-D shared-memory tile with a 1-pixel halo (Sobel is
//                   always a 3x3 neighborhood op -- kernel_size=3 only,
//                   same restriction as Basic), runtime `mode` dispatch,
//                   identical hand-optimized arithmetic to Basic (same
//                   term grouping) but reading from shared memory
//                   instead of 8 independent global loads per thread.
//                   Isolates the gain from reduced redundant global
//                   memory traffic.
//   SharedConst  -- Shared's tiling, but the Gx/Gy coefficients move
//                   from Basic's hand-unrolled arithmetic into
//                   `__constant__` 3x3 arrays read via a generic
//                   #pragma-unrolled 3x3 loop. Isolates constant
//                   memory's broadcast/cache behavior AND tests whether
//                   a generic loop can match Basic's hand-optimized
//                   (zero-skipping) arithmetic.
//   Specialized  -- Shared's tiling + `mode` as a compile-time template
//                   parameter: X only reads/computes gx (skips the 2
//                   neighbor reads and the arithmetic gy alone needs),
//                   Y only gx's complement, Magnitude/AbsSum compute
//                   both (same as Basic). Isolates the gain from
//                   mode-specific work reduction.
//   Separable    -- A genuine two-kernel decomposition: pass 1 computes,
//                   per pixel, the horizontal derivative (feeds gx) and
//                   horizontal smoothing (feeds gy) into two global
//                   float intermediate buffers; pass 2 applies the
//                   complementary vertical filter and combines per
//                   mode. No shared memory. This exists specifically to
//                   test the Section 6 lesson (multi-launch overhead can
//                   dominate at this project's 224x224 scale) against
//                   Sobel's already-single-launch Basic baseline --
//                   expected, but not assumed, to lose.
//
// --------------------------------------------------------------------------
// Correctness: every term in Gx/Gy is an exact-integer float32 value
// (pixel differences and a single x2 multiply, never a fractional
// weight) -- unlike Gaussian's sigma-weighted sums, reordering these
// sums cannot change the result: exact-integer float32 addition is
// associative in practice because no intermediate ever needs rounding
// (magnitudes stay far below 2^24). All four variants are therefore
// predicted -- and verified in tests/test_sobel_enhanced.py, not just
// assumed -- to be bit-exact vs. Basic, including Separable despite its
// different summation grouping.
// --------------------------------------------------------------------------
// Border: same reflect101() as Basic (gpu_image.cuh).
// --------------------------------------------------------------------------

enum class SobelVariant : int {
    Shared = 0,
    SharedConst = 1,
    Specialized = 2,
    Separable = 3,
};

// horizontal_ms/vertical_ms mirror GaussianEnhancedTiming's field names
// for consistency across Enhanced families; for the three single-kernel
// variants (Shared/SharedConst/Specialized) horizontal_ms is always 0
// and vertical_ms holds that one kernel's time. Only Separable actually
// uses both passes.
struct SobelEnhancedTiming {
    float horizontal_ms = 0.0f;
    float vertical_ms = 0.0f;
    float kernel_ms() const { return horizontal_ms + vertical_ms; }
};

// Low-level dispatch: launches `variant` directly on caller-provided
// device buffers (input/output uint8, sized batch_size*height*width) --
// no allocation for Shared/SharedConst/Specialized. Separable needs two
// float intermediate planes; it manages that allocation internally
// (self-contained DeviceBuffer<float>, freed at the end of this call)
// rather than requiring every caller to provide one, since -- unlike
// Gaussian, where every variant needs an intermediate -- only this one
// experimental Sobel variant does, and (per the header comment above)
// it is not expected to be the one selected for production use.
void sobel_enhanced_dispatch(
    const uint8_t* d_input, uint8_t* d_output,
    int batch_size, int height, int width, int mode,
    SobelVariant variant, int block_x, int block_y,
    SobelEnhancedTiming& timing);

// `mode` must be one of the SobelMode values (as an int, matching
// sobel_basic_gpu's convention -- see cuda/sobel.py).
// `block_x`/`block_y` select the launch configuration (block/tile
// tuning) -- a pure performance parameter, does not affect the result.
std::pair<std::unique_ptr<GpuImage>, SobelEnhancedTiming> sobel_enhanced(
    const GpuImage& input, int mode, SobelVariant variant, int block_x, int block_y);

// Batched form -- same kernels (already blockIdx.z-aware), applied to
// every image in `input` via one launch (or one launch pair, for
// Separable) instead of one per image, matching the production
// architecture Sections 5-7 established as the authoritative benchmark
// (never single-image timing as the primary decision metric).
std::pair<std::unique_ptr<GpuImageBatch>, SobelEnhancedTiming> sobel_enhanced_batch(
    const GpuImageBatch& input, int mode, SobelVariant variant, int block_x, int block_y);

}  // namespace xray_cuda
