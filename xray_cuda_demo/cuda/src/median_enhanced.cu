#include "median_enhanced.cuh"
#include <stdexcept>
#include <string>

namespace xray_cuda {

namespace {

constexpr int kMedianEnhancedMaxCount = kMedianBasicMaxKernelSize * kMedianBasicMaxKernelSize;

// Branchless compare-exchange: min/max on integers compile to predicated
// select instructions on GPU, not a data-dependent branch -- this is
// what makes the sorting network an actual "branch reduction" versus
// insertion sort's data-dependent while loop.
__device__ __forceinline__ void pix_sort(uint8_t& a, uint8_t& b) {
    uint8_t lo = min(a, b);
    uint8_t hi = max(a, b);
    a = lo;
    b = hi;
}

// Loads a (blockDim.x+2*radius) x (blockDim.y+2*radius) BORDER_REPLICATE
// halo tile for this block into `tile` (row-major, tile_w per row).
// Shared by every Enhanced Median kernel below -- one load routine, not
// three copies.
__device__ __forceinline__ void load_median_tile(
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
        int sy = clamp_index(block_start_y + ty - radius, height);
        int sx = clamp_index(block_start_x + tx - radius, width);
        tile[idx] = in[sy * width + sx];
    }
}

// -- Optimization 1: shared-memory tiling, same runtime insertion sort as Basic -----------------------

__global__ void median_shared_kernel(
    const uint8_t* input, uint8_t* output, int width, int height, int kernel_size) {
    extern __shared__ uint8_t s_tile_median[];
    int radius = kernel_size / 2;
    int tile_w = static_cast<int>(blockDim.x) + 2 * radius;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t plane = static_cast<size_t>(height) * width;
    const uint8_t* in = input + static_cast<size_t>(blockIdx.z) * plane;
    uint8_t* out = output + static_cast<size_t>(blockIdx.z) * plane;

    load_median_tile(s_tile_median, in, width, height, radius, tile_w);
    __syncthreads();

    if (x >= width || y >= height) return;

    int lx = threadIdx.x, ly = threadIdx.y;
    uint8_t values[kMedianEnhancedMaxCount];
    int count = 0;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            values[count++] = s_tile_median[(ly + radius + dy) * tile_w + (lx + radius + dx)];
        }
    }

    for (int i = 1; i < count; ++i) {
        uint8_t key = values[i];
        int j = i - 1;
        while (j >= 0 && values[j] > key) {
            values[j + 1] = values[j];
            --j;
        }
        values[j + 1] = key;
    }

    out[y * width + x] = values[count / 2];
}

// -- Optimization 2 (k=3): shared memory + branchless sorting network -----------------------
//
// Nicolas Devillard's opt_med9 (public domain; this exact 19-compare-
// exchange sequence is widely used/verified in image-processing code).
// Median is order-independent of *which* array index holds which
// spatial neighbor, so no particular row-major assignment is required
// for correctness -- only that all 9 neighborhood values are present.

__global__ void median_network9_kernel(const uint8_t* input, uint8_t* output, int width, int height) {
    constexpr int radius = 1;
    extern __shared__ uint8_t s_tile_net9[];
    int tile_w = static_cast<int>(blockDim.x) + 2 * radius;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t plane = static_cast<size_t>(height) * width;
    const uint8_t* in = input + static_cast<size_t>(blockIdx.z) * plane;
    uint8_t* out = output + static_cast<size_t>(blockIdx.z) * plane;

    load_median_tile(s_tile_net9, in, width, height, radius, tile_w);
    __syncthreads();

    if (x >= width || y >= height) return;

    int lx = threadIdx.x, ly = threadIdx.y;
    uint8_t p[9];
    int idx = 0;
#pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            p[idx++] = s_tile_net9[(ly + radius + dy) * tile_w + (lx + radius + dx)];
        }
    }

    pix_sort(p[1], p[2]); pix_sort(p[4], p[5]); pix_sort(p[7], p[8]);
    pix_sort(p[0], p[1]); pix_sort(p[3], p[4]); pix_sort(p[6], p[7]);
    pix_sort(p[1], p[2]); pix_sort(p[4], p[5]); pix_sort(p[7], p[8]);
    pix_sort(p[0], p[3]); pix_sort(p[5], p[8]); pix_sort(p[4], p[7]);
    pix_sort(p[3], p[6]); pix_sort(p[1], p[4]); pix_sort(p[2], p[5]);
    pix_sort(p[4], p[7]); pix_sort(p[4], p[2]); pix_sort(p[6], p[4]);
    pix_sort(p[4], p[2]);

    out[y * width + x] = p[4];
}

// -- Optimization 2 (k=5,7): shared memory + compile-time-specialized, unrolled insertion sort -----------------------
//
// A full hand-derived sorting network was only built for k=3 (see
// module docstring in median_enhanced.cuh for why 5x5/7x7 use unrolled
// insertion sort instead -- a deliberate scope decision, not an
// oversight). The outer gather and outer sort loop bound are compile-
// time constants here (#pragma unroll), unlike Shared's runtime
// kernel_size; the insertion sort's inner `while` remains data-dependent
// regardless of unrolling, since its trip count depends on the data.

template <int K>
__global__ void median_specialized_kernel(const uint8_t* input, uint8_t* output, int width, int height) {
    constexpr int radius = K / 2;
    constexpr int COUNT = K * K;
    extern __shared__ uint8_t s_tile_spec_median[];
    int tile_w = static_cast<int>(blockDim.x) + 2 * radius;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t plane = static_cast<size_t>(height) * width;
    const uint8_t* in = input + static_cast<size_t>(blockIdx.z) * plane;
    uint8_t* out = output + static_cast<size_t>(blockIdx.z) * plane;

    load_median_tile(s_tile_spec_median, in, width, height, radius, tile_w);
    __syncthreads();

    if (x >= width || y >= height) return;

    int lx = threadIdx.x, ly = threadIdx.y;
    uint8_t values[COUNT];
    int count = 0;
#pragma unroll
    for (int dy = -radius; dy <= radius; ++dy) {
#pragma unroll
        for (int dx = -radius; dx <= radius; ++dx) {
            values[count++] = s_tile_spec_median[(ly + radius + dy) * tile_w + (lx + radius + dx)];
        }
    }

#pragma unroll
    for (int i = 1; i < COUNT; ++i) {
        uint8_t key = values[i];
        int j = i - 1;
        while (j >= 0 && values[j] > key) {
            values[j + 1] = values[j];
            --j;
        }
        values[j + 1] = key;
    }

    out[y * width + x] = values[COUNT / 2];
}

}  // namespace

// -- shared dispatch: the ONE place the variant switch statement exists -----------------------

void median_enhanced_dispatch(
    const uint8_t* d_input, uint8_t* d_output,
    int batch_size, int height, int width, int kernel_size,
    MedianVariant variant, int block_x, int block_y,
    MedianEnhancedTiming& timing) {
    if (kernel_size <= 0 || kernel_size % 2 == 0) {
        throw std::invalid_argument(
            "median_enhanced_dispatch: kernel_size must be a positive odd integer, got " + std::to_string(kernel_size));
    }
    if (kernel_size > kMedianBasicMaxKernelSize) {
        throw std::invalid_argument(
            "median_enhanced_dispatch: kernel_size " + std::to_string(kernel_size) +
            " exceeds the maximum supported size " + std::to_string(kMedianBasicMaxKernelSize));
    }
    if (variant == MedianVariant::Network3x3 && kernel_size != 3) {
        throw std::invalid_argument("median_enhanced_dispatch: Network3x3 only supports kernel_size=3, got " +
                                     std::to_string(kernel_size));
    }
    if (variant == MedianVariant::Specialized && kernel_size != 3 && kernel_size != 5 && kernel_size != 7) {
        throw std::invalid_argument(
            "median_enhanced_dispatch: Specialized only supports kernel_size in {3,5,7}, got " +
            std::to_string(kernel_size));
    }
    if (block_x <= 0 || block_y <= 0) {
        throw std::invalid_argument("median_enhanced_dispatch: block_x/block_y must be positive.");
    }

    int radius = kernel_size / 2;
    dim3 block(static_cast<unsigned int>(block_x), static_cast<unsigned int>(block_y));
    dim3 grid2d = compute_launch_grid(width, height, block);
    dim3 grid(grid2d.x, grid2d.y, static_cast<unsigned int>(batch_size));

    size_t shared_bytes = static_cast<size_t>(block_x + 2 * radius) * (block_y + 2 * radius) * sizeof(uint8_t);

    CudaTimer timer;
    timer.start();
    switch (variant) {
        case MedianVariant::Shared:
            median_shared_kernel<<<grid, block, shared_bytes>>>(d_input, d_output, width, height, kernel_size);
            break;
        case MedianVariant::Network3x3:
            median_network9_kernel<<<grid, block, shared_bytes>>>(d_input, d_output, width, height);
            break;
        case MedianVariant::Specialized:
            switch (kernel_size) {
                case 3: median_specialized_kernel<3><<<grid, block, shared_bytes>>>(d_input, d_output, width, height); break;
                case 5: median_specialized_kernel<5><<<grid, block, shared_bytes>>>(d_input, d_output, width, height); break;
                case 7: median_specialized_kernel<7><<<grid, block, shared_bytes>>>(d_input, d_output, width, height); break;
            }
            break;
    }
    CUDA_CHECK_LAST_ERROR();
    timer.stop();
    CUDA_CHECK(cudaDeviceSynchronize());
    timing.kernel_ms = timer.elapsed_ms();
}

// -- thin single-image / batch wrappers -----------------------

std::pair<std::unique_ptr<GpuImage>, MedianEnhancedTiming> median_enhanced(
    const GpuImage& input, int kernel_size, MedianVariant variant, int block_x, int block_y) {
    auto output = std::make_unique<GpuImage>(input.height(), input.width());
    MedianEnhancedTiming timing{};
    median_enhanced_dispatch(
        input.data(), output->data(), /*batch_size=*/1, input.height(), input.width(), kernel_size,
        variant, block_x, block_y, timing);
    return {std::move(output), timing};
}

std::pair<std::unique_ptr<GpuImageBatch>, MedianEnhancedTiming> median_enhanced_batch(
    const GpuImageBatch& input, int kernel_size, MedianVariant variant, int block_x, int block_y) {
    auto output = std::make_unique<GpuImageBatch>(input.batch_size(), input.height(), input.width());
    MedianEnhancedTiming timing{};
    median_enhanced_dispatch(
        input.data(), output->data(), input.batch_size(), input.height(), input.width(), kernel_size,
        variant, block_x, block_y, timing);
    return {std::move(output), timing};
}

}  // namespace xray_cuda
