#pragma once

#include "gpu_image.cuh"
#include <utility>

namespace xray_cuda {

// --------------------------------------------------------------------------
// Enhanced CUDA binary threshold (Section 10).
//
// threshold_basic (Section 4F) is UNCHANGED and remains the reference
// "Basic CUDA" baseline -- this file adds a second, separate
// implementation family so the two remain independently benchmarkable
// at any time.
//
// -- Why Threshold's optimization ceiling is tiny, and that's expected --
// Threshold is a pure pointwise operation: one comparison, one write,
// no neighborhood, no border handling, no floating-point arithmetic.
// Basic Threshold ALREADY has adjacent-thread-to-adjacent-pixel
// coalesced access (idx = y*width+x, x varies fastest with threadIdx.x)
// -- there is no "reduce redundant reads" lever here at all (every
// filter through Section 9 had one; Threshold doesn't). The only
// plausible levers are (a) fewer, wider memory transactions
// (vectorized uchar4 loads/stores) and (b) fewer threads/blocks
// launched for the same work (multiple pixels per thread, amortizing
// per-thread/per-block scheduling overhead). Per the Section 10 spec's
// explicit framing: it is fully acceptable for Enhanced Threshold to be
// close to Basic if profiling shows Threshold is already efficient --
// this is measured, not assumed, in the README.
//
// Two variants:
//
//   Vectorized  -- uchar4 load/compare/store, 4 contiguous pixels per
//                  thread. SAFE ONLY when width % 4 == 0 (guarantees
//                  every row -- and therefore every image plane in a
//                  batch -- starts 4-byte aligned in the flat
//                  [N,H,W] buffer); dispatch automatically and silently
//                  falls back to an equivalent scalar kernel otherwise
//                  (the real dataset has widths like 1733 that are NOT
//                  4-aligned -- Section 10 spec item 7's explicit
//                  requirement). Isolates the gain from wider memory
//                  transactions.
//   MultiPixel  -- 4 contiguous pixels per thread via a small unrolled
//                  scalar loop (no vector types, no alignment
//                  requirement at all -- safe for every width).
//                  Isolates the gain from fewer threads/blocks (reduced
//                  scheduling/launch overhead) independently of
//                  vectorized memory access.
//
// --------------------------------------------------------------------------
// Correctness: identical uint8 comparison semantics to Basic
// (pixel > threshold_value -> max_value, else 0), applied per-pixel
// regardless of grouping -- both variants are predicted, and verified in
// tests/test_threshold_enhanced.py, to be bit-exact vs. Basic for every
// threshold_value/max_value and image width (aligned or not).
// --------------------------------------------------------------------------

enum class ThresholdVariant : int {
    Vectorized = 0,
    MultiPixel = 1,
};

struct ThresholdEnhancedTiming {
    float kernel_ms;
};

void threshold_enhanced_dispatch(
    const uint8_t* d_input, uint8_t* d_output,
    int batch_size, int height, int width,
    uint8_t threshold_value, uint8_t max_value,
    ThresholdVariant variant, int block_x, int block_y,
    ThresholdEnhancedTiming& timing);

std::pair<std::unique_ptr<GpuImage>, ThresholdEnhancedTiming> threshold_enhanced(
    const GpuImage& input, uint8_t threshold_value, uint8_t max_value,
    ThresholdVariant variant, int block_x, int block_y);

std::pair<std::unique_ptr<GpuImageBatch>, ThresholdEnhancedTiming> threshold_enhanced_batch(
    const GpuImageBatch& input, uint8_t threshold_value, uint8_t max_value,
    ThresholdVariant variant, int block_x, int block_y);

}  // namespace xray_cuda
