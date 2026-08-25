#pragma once

#include "gpu_image.cuh"
#include <utility>

namespace xray_cuda {

// --------------------------------------------------------------------------
// Enhanced CUDA Laplacian filter (Section 9).
//
// laplacian_basic (Section 4E) is UNCHANGED and remains the reference
// "Basic CUDA" baseline -- this file adds a second, separate
// implementation family so the two remain independently benchmarkable
// at any time.
//
// -- Why Laplacian has more headroom than Sobel (Section 8) --
// Unlike Basic Sobel (which was already hand-optimized: hardcoded
// coefficients, one launch, zero-skipping arithmetic), Basic Laplacian
// is a genuinely generic, non-separable, runtime-sized 2-D convolution
// -- a double for-loop reading BOTH the neighborhood AND the
// coefficients from global memory, with no zero-skipping (see
// laplacian_basic.cu). This is architecturally the same shape as Basic
// Gaussian's O(k^2) 2D convolution (Section 6's big win) and Basic
// Median's per-pixel neighborhood gather -- so real gains from shared-
// memory reuse, constant memory, and specialization are plausible here
// in a way they were NOT for Sobel. Measured, not assumed (see
// README's Section 9 benchmark numbers).
//
// -- The three actual coefficient sets --
// cpu.filters.ALLOWED_LAPLACIAN_KERNELS = {1, 3, 5} is OpenCV's
// aperture-size PARAMETER, not a literal matrix dimension:
// kernel_size=1 and kernel_size=3 both recover a 3x3 coefficient
// matrix (via cuda/laplacian.py::laplacian_kernel_2d's impulse-response
// method) but with DIFFERENT values -- [[0,1,0],[1,-4,1],[0,1,0]] for
// kernel_size=1 (the textbook Laplacian, a special case) vs.
// [[2,0,2],[0,-8,0],[2,0,2]] for kernel_size=3 (cv2.Laplacian(ksize=3)
// is internally two Sobel(dx=2)/Sobel(dy=2) passes summed, not the
// simple discrete Laplacian); kernel_size=5 recovers a distinct 5x5
// matrix. The C++ layer only ever sees the coefficient array's actual
// shape (3x3 or 5x5), never the Python-level kernel_size=1 vs 3
// distinction -- see laplacian_basic.cuh's identical note.
//
// Four variants, one optimization at a time:
//
//   Shared       -- 2-D shared-memory tile with a halo sized to the
//                   runtime radius (radius = actual_matrix_size / 2),
//                   same generic double-loop arithmetic as Basic but
//                   reading the neighborhood from shared memory.
//                   Isolates the gain from reduced redundant global
//                   memory traffic.
//   SharedConst  -- Shared's tiling + coefficients moved from a
//                   global-memory pointer into a fixed-capacity
//                   __constant__ array (capacity
//                   kLaplacianEnhancedMaxKernelSize^2, covers up to
//                   5x5), still a runtime-sized double loop. Isolates
//                   constant memory's broadcast/cache behavior --
//                   Section 8 found a NAIVE generic constant-memory
//                   loop can regress vs. hand-optimized arithmetic, so
//                   this is measured on its own before combining with
//                   anything else, exactly as the Section 9 spec warns.
//   Specialized  -- SharedConst + the coefficient matrix's actual size
//                   as a compile-time template parameter (3 or 5),
//                   #pragma-unrolled double loop. Isolates the gain
//                   from eliminating runtime loop bounds/indexing on
//                   top of constant memory -- still reads every
//                   coefficient (including zeros) from constant memory,
//                   unlike Explicit below.
//   Explicit     -- Hand-written arithmetic for each of the three known,
//                   VERIFIED coefficient sets (never re-derived --
//                   dispatch validates the caller's coefficient array
//                   against the expected exact values before using this
//                   path, matching Section 9 spec item 11's explicit
//                   requirement to reuse, not reimplement, the verified
//                   coefficient mechanism), skipping every
//                   zero-coefficient term entirely (5/9, 5/9, and 21/25
//                   nonzero respectively -- see module comment above).
//                   Isolates the gain from exploiting the specific known
//                   sparsity pattern, the same lever Basic Sobel already
//                   used and Section 8's SharedConst experiment showed a
//                   generic loop cannot recover on its own.
//
// --------------------------------------------------------------------------
// Correctness: like Sobel (Section 8), every coefficient here is a
// small exact integer (2, 4, -8, -24, ...) and inputs are uint8 -- all
// exact-integer float32 arithmetic, so reordering terms (Shared's tiled
// reads, SharedConst/Specialized's constant-memory loop, Explicit's
// hand-grouped sums) cannot change the result. All four variants are
// predicted -- and verified in tests/test_laplacian_enhanced.py, not
// just assumed -- to be bit-exact vs. Basic, for every supported
// kernel_size/scale/delta combination.
// --------------------------------------------------------------------------
// Border: same reflect101() as Basic (gpu_image.cuh).
// --------------------------------------------------------------------------

enum class LaplacianVariant : int {
    Shared = 0,
    SharedConst = 1,
    Specialized = 2,
    Explicit = 3,
};

// Coefficients live in constant memory for SharedConst/Specialized; this
// bound covers the largest actual coefficient matrix any supported
// kernel_size recovers (5x5, from kernel_size=5 -- see module comment).
constexpr int kLaplacianEnhancedMaxKernelSize = 5;

struct LaplacianEnhancedTiming {
    float coeff_upload_ms;
    float kernel_ms;
};

// Low-level dispatch: launches `variant` directly on caller-provided
// device buffers (input/output uint8, sized batch_size*height*width) --
// no allocation. `host_coeffs` is the kernel_size*kernel_size row-major
// actual coefficient matrix (see cuda/laplacian.py::laplacian_kernel_2d
// -- kernel_size here is that array's actual shape, 3 or 5, NOT
// OpenCV's aperture-size parameter). Explicit additionally validates
// `host_coeffs` against its three known exact patterns and throws if
// none match (see header comment above) -- this is a deliberate scope
// restriction, not a silent fallback.
void laplacian_enhanced_dispatch(
    const uint8_t* d_input, uint8_t* d_output,
    int batch_size, int height, int width,
    const float* host_coeffs, int kernel_size, float scale, float delta,
    LaplacianVariant variant, int block_x, int block_y,
    LaplacianEnhancedTiming& timing);

std::pair<std::unique_ptr<GpuImage>, LaplacianEnhancedTiming> laplacian_enhanced(
    const GpuImage& input, const float* host_coeffs, int kernel_size, float scale, float delta,
    LaplacianVariant variant, int block_x, int block_y);

// Batched form -- same kernels (already blockIdx.z-aware), applied to
// every image in `input` via one launch instead of one per image,
// matching the production architecture Sections 5-8 established as the
// authoritative benchmark (never single-image timing as the primary
// decision metric).
std::pair<std::unique_ptr<GpuImageBatch>, LaplacianEnhancedTiming> laplacian_enhanced_batch(
    const GpuImageBatch& input, const float* host_coeffs, int kernel_size, float scale, float delta,
    LaplacianVariant variant, int block_x, int block_y);

}  // namespace xray_cuda
