#include "threshold_enhanced.cuh"
#include <stdexcept>
#include <string>

namespace xray_cuda {

namespace {

// Identical math to threshold_basic_kernel -- used both as its own
// dispatch path is unnecessary (Basic already exists for that) and as
// the automatic safe fallback for Vectorized when width % 4 != 0.
__global__ void threshold_scalar_kernel(
    const uint8_t* input, uint8_t* output, int width, int height,
    uint8_t threshold_value, uint8_t max_value) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    size_t plane = static_cast<size_t>(height) * width;
    int idx = y * width + x;
    const uint8_t* in = input + static_cast<size_t>(blockIdx.z) * plane;
    uint8_t* out = output + static_cast<size_t>(blockIdx.z) * plane;
    out[idx] = (in[idx] > threshold_value) ? max_value : 0;
}

// -- Optimization 1: uchar4 vectorized load/compare/store -----------------------
//
// Only launched when width % 4 == 0 (checked in dispatch) -- guarantees
// every row, and therefore every image plane in a [N,H,W] batch, starts
// 4-byte aligned, so reinterpret_cast<const uchar4*> is safe.

__global__ void threshold_vectorized_kernel(
    const uint8_t* input, uint8_t* output, int width, int height,
    uint8_t threshold_value, uint8_t max_value) {
    int width4 = width / 4;
    int vx = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (vx >= width4 || y >= height) return;

    size_t plane = static_cast<size_t>(height) * width;
    const uchar4* in4 = reinterpret_cast<const uchar4*>(input + static_cast<size_t>(blockIdx.z) * plane);
    uchar4* out4 = reinterpret_cast<uchar4*>(output + static_cast<size_t>(blockIdx.z) * plane);

    int idx = y * width4 + vx;
    uchar4 v = in4[idx];
    uchar4 r;
    r.x = (v.x > threshold_value) ? max_value : 0;
    r.y = (v.y > threshold_value) ? max_value : 0;
    r.z = (v.z > threshold_value) ? max_value : 0;
    r.w = (v.w > threshold_value) ? max_value : 0;
    out4[idx] = r;
}

// -- Optimization 2: multiple pixels per thread, scalar (no vector types, any width) -----------------------

constexpr int kThresholdMultiPixelCount = 4;

__global__ void threshold_multipixel_kernel(
    const uint8_t* input, uint8_t* output, int width, int height,
    uint8_t threshold_value, uint8_t max_value) {
    int x_start = (blockIdx.x * blockDim.x + threadIdx.x) * kThresholdMultiPixelCount;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x_start >= width || y >= height) return;

    size_t plane = static_cast<size_t>(height) * width;
    const uint8_t* in = input + static_cast<size_t>(blockIdx.z) * plane + static_cast<size_t>(y) * width;
    uint8_t* out = output + static_cast<size_t>(blockIdx.z) * plane + static_cast<size_t>(y) * width;

#pragma unroll
    for (int i = 0; i < kThresholdMultiPixelCount; ++i) {
        int x = x_start + i;
        if (x < width) {
            out[x] = (in[x] > threshold_value) ? max_value : 0;
        }
    }
}

}  // namespace

// -- shared dispatch: the ONE place the variant switch statement exists -----------------------

void threshold_enhanced_dispatch(
    const uint8_t* d_input, uint8_t* d_output,
    int batch_size, int height, int width,
    uint8_t threshold_value, uint8_t max_value,
    ThresholdVariant variant, int block_x, int block_y,
    ThresholdEnhancedTiming& timing) {
    if (block_x <= 0 || block_y <= 0) {
        throw std::invalid_argument("threshold_enhanced_dispatch: block_x/block_y must be positive.");
    }

    dim3 block(static_cast<unsigned int>(block_x), static_cast<unsigned int>(block_y));

    CudaTimer timer;
    timer.start();
    switch (variant) {
        case ThresholdVariant::Vectorized: {
            if (width % 4 == 0) {
                int width4 = width / 4;
                dim3 grid2d(
                    (static_cast<unsigned int>(width4) + block.x - 1) / block.x,
                    (static_cast<unsigned int>(height) + block.y - 1) / block.y);
                dim3 grid(grid2d.x, grid2d.y, static_cast<unsigned int>(batch_size));
                threshold_vectorized_kernel<<<grid, block>>>(
                    d_input, d_output, width, height, threshold_value, max_value);
            } else {
                // Automatic safe fallback -- width isn't 4-aligned (the
                // real dataset has widths like 1733), so an equivalent
                // scalar kernel runs instead of an unsafe vector load.
                dim3 grid2d = compute_launch_grid(width, height, block);
                dim3 grid(grid2d.x, grid2d.y, static_cast<unsigned int>(batch_size));
                threshold_scalar_kernel<<<grid, block>>>(
                    d_input, d_output, width, height, threshold_value, max_value);
            }
            break;
        }
        case ThresholdVariant::MultiPixel: {
            int groups_x = (width + kThresholdMultiPixelCount - 1) / kThresholdMultiPixelCount;
            dim3 grid2d(
                (static_cast<unsigned int>(groups_x) + block.x - 1) / block.x,
                (static_cast<unsigned int>(height) + block.y - 1) / block.y);
            dim3 grid(grid2d.x, grid2d.y, static_cast<unsigned int>(batch_size));
            threshold_multipixel_kernel<<<grid, block>>>(
                d_input, d_output, width, height, threshold_value, max_value);
            break;
        }
    }
    CUDA_CHECK_LAST_ERROR();
    timer.stop();
    CUDA_CHECK(cudaDeviceSynchronize());
    timing.kernel_ms = timer.elapsed_ms();
}

// -- thin single-image / batch wrappers -----------------------

std::pair<std::unique_ptr<GpuImage>, ThresholdEnhancedTiming> threshold_enhanced(
    const GpuImage& input, uint8_t threshold_value, uint8_t max_value,
    ThresholdVariant variant, int block_x, int block_y) {
    auto output = std::make_unique<GpuImage>(input.height(), input.width());
    ThresholdEnhancedTiming timing{};
    threshold_enhanced_dispatch(
        input.data(), output->data(), /*batch_size=*/1, input.height(), input.width(),
        threshold_value, max_value, variant, block_x, block_y, timing);
    return {std::move(output), timing};
}

std::pair<std::unique_ptr<GpuImageBatch>, ThresholdEnhancedTiming> threshold_enhanced_batch(
    const GpuImageBatch& input, uint8_t threshold_value, uint8_t max_value,
    ThresholdVariant variant, int block_x, int block_y) {
    auto output = std::make_unique<GpuImageBatch>(input.batch_size(), input.height(), input.width());
    ThresholdEnhancedTiming timing{};
    threshold_enhanced_dispatch(
        input.data(), output->data(), input.batch_size(), input.height(), input.width(),
        threshold_value, max_value, variant, block_x, block_y, timing);
    return {std::move(output), timing};
}

}  // namespace xray_cuda
