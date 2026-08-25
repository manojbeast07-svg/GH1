#include "laplacian_threshold_fused.cuh"
#include <stdexcept>
#include <string>

namespace xray_cuda {

// Self-contained constant-memory coefficient storage -- deliberately
// NOT shared with laplacian_enhanced.cu's own __constant__ symbol, to
// keep this experimental, non-production-wired module independent.
__constant__ float d_fused_laplacian_coeffs[25];

namespace {

__device__ __forceinline__ void load_fused_tile(
    uint8_t* tile, const uint8_t* in, int width, int height, int radius, int tile_w) {
    int block_start_x = blockIdx.x * blockDim.x;
    int block_start_y = blockIdx.y * blockDim.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = static_cast<int>(blockDim.x * blockDim.y);
    int tile_h = static_cast<int>(blockDim.y) + 2 * radius;
    int tile_size = tile_w * tile_h;
    for (int idx = tid; idx < tile_size; idx += num_threads) {
        int ty = idx / tile_w;
        int tx = idx % tile_w;
        int sy = reflect101(block_start_y + ty - radius, height);
        int sx = reflect101(block_start_x + tx - radius, width);
        tile[idx] = in[sy * width + sx];
    }
}

// Laplacian (Specialized-equivalent: compile-time K, unrolled, constant-
// memory coefficients) immediately followed by Threshold, on the value
// still held in a register -- no intermediate write to global memory.
template <int K>
__global__ void laplacian_threshold_fused_kernel(
    const uint8_t* input, uint8_t* output, int width, int height,
    float scale, float delta, uint8_t threshold_value, uint8_t max_value) {
    constexpr int radius = K / 2;
    extern __shared__ uint8_t s_tile_fused[];
    int tile_w = static_cast<int>(blockDim.x) + 2 * radius;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t plane = static_cast<size_t>(height) * width;
    const uint8_t* in = input + static_cast<size_t>(blockIdx.z) * plane;
    uint8_t* out = output + static_cast<size_t>(blockIdx.z) * plane;

    load_fused_tile(s_tile_fused, in, width, height, radius, tile_w);
    __syncthreads();

    if (x >= width || y >= height) return;

    int lx = threadIdx.x, ly = threadIdx.y;
    float sum = 0.0f;
#pragma unroll
    for (int ky = -radius; ky <= radius; ++ky) {
#pragma unroll
        for (int kx = -radius; kx <= radius; ++kx) {
            float v = static_cast<float>(s_tile_fused[(ly + radius + ky) * tile_w + (lx + radius + kx)]);
            sum += d_fused_laplacian_coeffs[(ky + radius) * K + (kx + radius)] * v;
        }
    }

    float scaled = sum * scale + delta;
    int rounded = __float2int_rn(fabsf(scaled));
    rounded = max(0, min(255, rounded));
    uint8_t laplacian_out = static_cast<uint8_t>(rounded);

    out[y * width + x] = (laplacian_out > threshold_value) ? max_value : 0;
}

}  // namespace

void laplacian_threshold_fused_dispatch(
    const uint8_t* d_input, uint8_t* d_output,
    int batch_size, int height, int width,
    const float* host_coeffs, int kernel_size, float scale, float delta,
    uint8_t threshold_value, uint8_t max_value,
    int block_x, int block_y,
    FusedLaplacianThresholdTiming& timing) {
    if (kernel_size != 3 && kernel_size != 5) {
        throw std::invalid_argument(
            "laplacian_threshold_fused_dispatch: only kernel_size in {3,5} is supported, got " +
            std::to_string(kernel_size));
    }
    if (block_x <= 0 || block_y <= 0) {
        throw std::invalid_argument("laplacian_threshold_fused_dispatch: block_x/block_y must be positive.");
    }

    CUDA_CHECK(cudaMemcpyToSymbol(d_fused_laplacian_coeffs, host_coeffs, kernel_size * kernel_size * sizeof(float)));

    int radius = kernel_size / 2;
    dim3 block(static_cast<unsigned int>(block_x), static_cast<unsigned int>(block_y));
    dim3 grid2d = compute_launch_grid(width, height, block);
    dim3 grid(grid2d.x, grid2d.y, static_cast<unsigned int>(batch_size));
    size_t shared_bytes = static_cast<size_t>(block_x + 2 * radius) * (block_y + 2 * radius) * sizeof(uint8_t);

    CudaTimer timer;
    timer.start();
    if (kernel_size == 3) {
        laplacian_threshold_fused_kernel<3><<<grid, block, shared_bytes>>>(
            d_input, d_output, width, height, scale, delta, threshold_value, max_value);
    } else {
        laplacian_threshold_fused_kernel<5><<<grid, block, shared_bytes>>>(
            d_input, d_output, width, height, scale, delta, threshold_value, max_value);
    }
    CUDA_CHECK_LAST_ERROR();
    timer.stop();
    CUDA_CHECK(cudaDeviceSynchronize());
    timing.kernel_ms = timer.elapsed_ms();
}

}  // namespace xray_cuda
