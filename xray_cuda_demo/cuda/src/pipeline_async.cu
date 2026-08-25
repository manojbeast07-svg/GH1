// Section 20F: EXPERIMENTAL multi-stream, double/triple-buffered async
// pipeline. See pipeline_async.cuh's file header for the architecture
// summary and research/async_pipeline_research.md for the CUDA overlap
// requirements this design is built around.
//
// KERNEL PROVENANCE:
//   Basic kernels (gaussian_basic_kernel, median_basic_kernel,
//   sobel_basic_kernel, laplacian_basic_kernel, threshold_basic_kernel)
//   are the REAL production kernel symbols, called directly via their
//   declarations in cuda/include/*_basic.cuh -- no duplication, exactly
//   as Section 20D's CudaGraphPipeline already does for Basic.
//
//   Enhanced kernels are isolated, verified-identical copies -- SAME
//   bodies as Section 20E's cuda/src/pipeline_cuda_graph_enhanced.cu
//   (itself copied, with cited provenance, from gaussian_enhanced.cu/
//   median_enhanced.cu/sobel_enhanced.cu/laplacian_enhanced.cu/
//   threshold_enhanced.cu -- see that file's own top-of-file comment for
//   the original line-by-line citations). Duplicated a second time here
//   (rather than sharing Section 20E's copy via a new header) to avoid
//   modifying Section 20E's already-tested, frozen file; both copies are
//   independently verified bit-exact against production in this
//   section's own tests, exactly mirroring Section 20E's own equivalence
//   methodology. No kernel math differs from production in any way.
//
// BUFFER-SET REUSE IS NOT "PERSISTENT BUFFERS" (spec item 14): the N
// buffer sets below ARE reused across chunks *within one run() call* --
// that reuse is the structural mechanism that makes pipelining possible
// at all (there is no other way to keep multiple chunks in flight
// simultaneously). They are allocated fresh at the start of every
// run() call and freed at the end -- nothing persists FROM one run()
// call TO the next, unlike Section 20B/20C's PersistentCudaPipeline,
// which this file never imports or references.

#include "pipeline_async.cuh"
#include "gaussian_basic.cuh"
#include "laplacian_basic.cuh"
#include "median_basic.cuh"
#include "sobel_basic.cuh"
#include "threshold_basic.cuh"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace xray_cuda {

namespace {

class HostTimer {
public:
    void start() { start_ = std::chrono::steady_clock::now(); }
    void stop() { stop_ = std::chrono::steady_clock::now(); }
    float elapsed_ms() const { return std::chrono::duration<float, std::milli>(stop_ - start_).count(); }

private:
    std::chrono::steady_clock::time_point start_, stop_;
};

// --------------------------------------------------------------------------
// Isolated, verified-identical Enhanced kernel copies (production
// Specialized/Network3x3/Vectorized variants only). See this file's
// top-of-file provenance comment.
// --------------------------------------------------------------------------

constexpr int kAsyncGaussianMaxKernelSize = 9;
__constant__ float a_gaussian_coeffs_1d[kAsyncGaussianMaxKernelSize];

template <int K>
__global__ void async_gaussian_specialized_h_kernel(const uint8_t* input, float* output, int width, int height) {
    constexpr int radius = K / 2;
    extern __shared__ uint8_t s_tile_u8_spec[];
    int tile_w = blockDim.x + 2 * radius;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t plane = static_cast<size_t>(height) * width;
    const uint8_t* in = input + static_cast<size_t>(blockIdx.z) * plane;
    float* out = output + static_cast<size_t>(blockIdx.z) * plane;
    int tx = threadIdx.x, ty = threadIdx.y;
    int block_start_x = blockIdx.x * blockDim.x;
    if (y < height) {
        for (int i = tx; i < tile_w; i += blockDim.x) {
            int sx = reflect101(block_start_x + i - radius, width);
            s_tile_u8_spec[ty * tile_w + i] = in[y * width + sx];
        }
    }
    __syncthreads();
    if (x >= width || y >= height) return;
    float sum = 0.0f;
#pragma unroll
    for (int k = -radius; k <= radius; ++k) {
        sum += a_gaussian_coeffs_1d[k + radius] * static_cast<float>(s_tile_u8_spec[ty * tile_w + (tx + radius + k)]);
    }
    out[y * width + x] = sum;
}

template <int K>
__global__ void async_gaussian_specialized_v_kernel(const float* input, uint8_t* output, int width, int height) {
    constexpr int radius = K / 2;
    extern __shared__ float s_tile_f32_spec[];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t plane = static_cast<size_t>(height) * width;
    const float* in = input + static_cast<size_t>(blockIdx.z) * plane;
    uint8_t* out = output + static_cast<size_t>(blockIdx.z) * plane;
    int tx = threadIdx.x, ty = threadIdx.y;
    int block_start_y = blockIdx.y * blockDim.y;
    if (x < width) {
        for (int i = ty; i < static_cast<int>(blockDim.y) + 2 * radius; i += blockDim.y) {
            int sy = reflect101(block_start_y + i - radius, height);
            s_tile_f32_spec[i * blockDim.x + tx] = in[sy * width + x];
        }
    }
    __syncthreads();
    if (x >= width || y >= height) return;
    float sum = 0.0f;
#pragma unroll
    for (int k = -radius; k <= radius; ++k) {
        sum += a_gaussian_coeffs_1d[k + radius] * s_tile_f32_spec[(ty + radius + k) * blockDim.x + tx];
    }
    int rounded = __float2int_rn(sum);
    rounded = max(0, min(255, rounded));
    out[y * width + x] = static_cast<uint8_t>(rounded);
}

__device__ __forceinline__ void async_pix_sort(uint8_t& a, uint8_t& b) {
    uint8_t lo = min(a, b);
    uint8_t hi = max(a, b);
    a = lo;
    b = hi;
}

__device__ __forceinline__ void async_load_median_tile(
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

__global__ void async_median_network9_kernel(const uint8_t* input, uint8_t* output, int width, int height) {
    constexpr int radius = 1;
    extern __shared__ uint8_t s_tile_net9[];
    int tile_w = static_cast<int>(blockDim.x) + 2 * radius;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t plane = static_cast<size_t>(height) * width;
    const uint8_t* in = input + static_cast<size_t>(blockIdx.z) * plane;
    uint8_t* out = output + static_cast<size_t>(blockIdx.z) * plane;
    async_load_median_tile(s_tile_net9, in, width, height, radius, tile_w);
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
    async_pix_sort(p[1], p[2]); async_pix_sort(p[4], p[5]); async_pix_sort(p[7], p[8]);
    async_pix_sort(p[0], p[1]); async_pix_sort(p[3], p[4]); async_pix_sort(p[6], p[7]);
    async_pix_sort(p[1], p[2]); async_pix_sort(p[4], p[5]); async_pix_sort(p[7], p[8]);
    async_pix_sort(p[0], p[3]); async_pix_sort(p[5], p[8]); async_pix_sort(p[4], p[7]);
    async_pix_sort(p[3], p[6]); async_pix_sort(p[1], p[4]); async_pix_sort(p[2], p[5]);
    async_pix_sort(p[4], p[7]); async_pix_sort(p[4], p[2]); async_pix_sort(p[6], p[4]);
    async_pix_sort(p[4], p[2]);
    out[y * width + x] = p[4];
}

__device__ __forceinline__ uint8_t async_sobel_finalize(float raw) {
    int rounded = __float2int_rn(fabsf(raw));
    rounded = max(0, min(255, rounded));
    return static_cast<uint8_t>(rounded);
}

__device__ __forceinline__ void async_load_sobel_tile(
    uint8_t* tile, const uint8_t* in, int width, int height, int tile_w) {
    int block_start_x = blockIdx.x * blockDim.x;
    int block_start_y = blockIdx.y * blockDim.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = static_cast<int>(blockDim.x * blockDim.y);
    int tile_h = static_cast<int>(blockDim.y) + 2;
    int tile_size = tile_w * tile_h;
    for (int idx = tid; idx < tile_size; idx += num_threads) {
        int ty = idx / tile_w;
        int tx = idx % tile_w;
        int sy = reflect101(block_start_y + ty - 1, height);
        int sx = reflect101(block_start_x + tx - 1, width);
        tile[idx] = in[sy * width + sx];
    }
}

template <int MODE>
__global__ void async_sobel_specialized_kernel(const uint8_t* input, uint8_t* output, int width, int height) {
    extern __shared__ uint8_t s_tile_sobel_spec[];
    int tile_w = static_cast<int>(blockDim.x) + 2;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t plane = static_cast<size_t>(height) * width;
    const uint8_t* in = input + static_cast<size_t>(blockIdx.z) * plane;
    uint8_t* out = output + static_cast<size_t>(blockIdx.z) * plane;
    async_load_sobel_tile(s_tile_sobel_spec, in, width, height, tile_w);
    __syncthreads();
    if (x >= width || y >= height) return;
    int lx = threadIdx.x, ly = threadIdx.y;
    auto tp = [&](int dy, int dx) -> float {
        return static_cast<float>(s_tile_sobel_spec[(ly + 1 + dy) * tile_w + (lx + 1 + dx)]);
    };
    constexpr bool need_gx = (MODE != 1);
    constexpr bool need_gy = (MODE != 0);
    float p00 = tp(-1, -1), p02 = tp(-1, 1), p20 = tp(1, -1), p22 = tp(1, 1);
    float gx = 0.0f, gy = 0.0f;
    if constexpr (need_gx) {
        float p10 = tp(0, -1), p12 = tp(0, 1);
        gx = (p02 - p00) + 2.0f * (p12 - p10) + (p22 - p20);
    }
    if constexpr (need_gy) {
        float p01 = tp(-1, 0), p21 = tp(1, 0);
        gy = (p20 - p00) + 2.0f * (p21 - p01) + (p22 - p02);
    }
    (void)gx;
    (void)gy;
    float raw;
    if constexpr (MODE == 0) raw = gx;
    else if constexpr (MODE == 1) raw = gy;
    else if constexpr (MODE == 2) raw = sqrtf(gx * gx + gy * gy);
    else raw = fabsf(gx) + fabsf(gy);
    out[y * width + x] = async_sobel_finalize(raw);
}

constexpr int kAsyncLaplacianMaxKernelSize = 5;
__constant__ float a_laplacian_coeffs[kAsyncLaplacianMaxKernelSize * kAsyncLaplacianMaxKernelSize];

__device__ __forceinline__ uint8_t async_laplacian_finalize(float sum, float scale, float delta) {
    float scaled = sum * scale + delta;
    int rounded = __float2int_rn(fabsf(scaled));
    rounded = max(0, min(255, rounded));
    return static_cast<uint8_t>(rounded);
}

__device__ __forceinline__ void async_load_laplacian_tile(
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

template <int K>
__global__ void async_laplacian_specialized_kernel(
    const uint8_t* input, uint8_t* output, int width, int height, float scale, float delta) {
    constexpr int radius = K / 2;
    extern __shared__ uint8_t s_tile_lap_spec[];
    int tile_w = static_cast<int>(blockDim.x) + 2 * radius;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t plane = static_cast<size_t>(height) * width;
    const uint8_t* in = input + static_cast<size_t>(blockIdx.z) * plane;
    uint8_t* out = output + static_cast<size_t>(blockIdx.z) * plane;
    async_load_laplacian_tile(s_tile_lap_spec, in, width, height, radius, tile_w);
    __syncthreads();
    if (x >= width || y >= height) return;
    int lx = threadIdx.x, ly = threadIdx.y;
    float sum = 0.0f;
#pragma unroll
    for (int ky = -radius; ky <= radius; ++ky) {
#pragma unroll
        for (int kx = -radius; kx <= radius; ++kx) {
            float v = static_cast<float>(s_tile_lap_spec[(ly + radius + ky) * tile_w + (lx + radius + kx)]);
            sum += a_laplacian_coeffs[(ky + radius) * K + (kx + radius)] * v;
        }
    }
    out[y * width + x] = async_laplacian_finalize(sum, scale, delta);
}

__global__ void async_threshold_scalar_kernel(
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

__global__ void async_threshold_vectorized_kernel(
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

void launch_enhanced_threshold(
    const uint8_t* input, uint8_t* output, int width, int height, int batch_size,
    uint8_t threshold_value, uint8_t max_value, dim3 block, cudaStream_t stream) {
    if (width % 4 == 0) {
        int width4 = width / 4;
        dim3 grid2d((static_cast<unsigned int>(width4) + block.x - 1) / block.x,
                    (static_cast<unsigned int>(height) + block.y - 1) / block.y);
        dim3 grid(grid2d.x, grid2d.y, static_cast<unsigned int>(batch_size));
        async_threshold_vectorized_kernel<<<grid, block, 0, stream>>>(input, output, width, height, threshold_value, max_value);
    } else {
        dim3 grid2d = compute_launch_grid(width, height, block);
        dim3 grid(grid2d.x, grid2d.y, static_cast<unsigned int>(batch_size));
        async_threshold_scalar_kernel<<<grid, block, 0, stream>>>(input, output, width, height, threshold_value, max_value);
    }
}

// Launches the 5 (or fewer, per enabled flags) stages for one chunk on
// `stream`, ping-ponging between buf_a/buf_b (raw device pointers -- sized
// to hold chunk_batch_size*height*width bytes each), and, for Enhanced
// Gaussian, `intermediate` (a float buffer of the same element count,
// non-null only when config.use_enhanced && config.gaussian_enabled).
// Returns true if the final result ended up in buf_a (false if buf_b). No
// cudaDeviceSynchronize/cudaStreamSynchronize anywhere in this function --
// every launch stays on `stream`, letting the caller's event-based
// dependency chain (not this function) control ordering against other
// streams.
bool launch_stages(
    uint8_t* buf_a, uint8_t* buf_b, float* intermediate,
    DeviceBuffer<float>* gaussian_coeffs_dev, DeviceBuffer<float>* laplacian_coeffs_dev,
    int chunk_batch_size, int height, int width,
    const AsyncPipelineConfig& config, cudaStream_t stream) {
    dim3 block = default_block_dim();
    uint8_t* current = buf_a;
    uint8_t* other = buf_b;
    bool current_is_a = true;

    dim3 grid2d = compute_launch_grid(width, height, block);
    dim3 grid(grid2d.x, grid2d.y, static_cast<unsigned int>(chunk_batch_size));

    if (config.gaussian_enabled) {
        if (config.use_enhanced) {
            int K = config.gaussian_kernel_size;
            int radius = K / 2;
            size_t h_shared_bytes = static_cast<size_t>(block.x + 2 * radius) * block.y * sizeof(uint8_t);
            size_t v_shared_bytes = static_cast<size_t>(block.x) * (block.y + 2 * radius) * sizeof(float);
            CUDA_CHECK(cudaMemcpyToSymbolAsync(a_gaussian_coeffs_1d, config.gaussian_coeffs_1d,
                                                K * sizeof(float), 0, cudaMemcpyHostToDevice, stream));
            switch (K) {
                case 3: async_gaussian_specialized_h_kernel<3><<<grid, block, h_shared_bytes, stream>>>(current, intermediate, width, height); break;
                case 5: async_gaussian_specialized_h_kernel<5><<<grid, block, h_shared_bytes, stream>>>(current, intermediate, width, height); break;
                case 7: async_gaussian_specialized_h_kernel<7><<<grid, block, h_shared_bytes, stream>>>(current, intermediate, width, height); break;
                case 9: async_gaussian_specialized_h_kernel<9><<<grid, block, h_shared_bytes, stream>>>(current, intermediate, width, height); break;
                default: throw std::invalid_argument("AsyncCudaPipeline: gaussian_kernel_size must be one of {3,5,7,9} for Enhanced, got " + std::to_string(K));
            }
            CUDA_CHECK_LAST_ERROR();
            switch (K) {
                case 3: async_gaussian_specialized_v_kernel<3><<<grid, block, v_shared_bytes, stream>>>(intermediate, other, width, height); break;
                case 5: async_gaussian_specialized_v_kernel<5><<<grid, block, v_shared_bytes, stream>>>(intermediate, other, width, height); break;
                case 7: async_gaussian_specialized_v_kernel<7><<<grid, block, v_shared_bytes, stream>>>(intermediate, other, width, height); break;
                case 9: async_gaussian_specialized_v_kernel<9><<<grid, block, v_shared_bytes, stream>>>(intermediate, other, width, height); break;
            }
            CUDA_CHECK_LAST_ERROR();
        } else {
            if (config.gaussian_kernel_size <= 0 || config.gaussian_kernel_size % 2 == 0) {
                throw std::invalid_argument("AsyncCudaPipeline: invalid gaussian_kernel_size " + std::to_string(config.gaussian_kernel_size));
            }
            size_t coeff_count = static_cast<size_t>(config.gaussian_kernel_size) * config.gaussian_kernel_size;
            gaussian_coeffs_dev->upload(config.gaussian_coeffs, coeff_count, stream);
            gaussian_basic_kernel<<<grid, block, 0, stream>>>(current, other, width, height, gaussian_coeffs_dev->get(), config.gaussian_kernel_size);
            CUDA_CHECK_LAST_ERROR();
        }
        std::swap(current, other);
        current_is_a = !current_is_a;
    }
    if (config.median_enabled) {
        if (config.use_enhanced) {
            size_t shared_bytes = static_cast<size_t>(block.x + 2) * (block.y + 2) * sizeof(uint8_t);
            async_median_network9_kernel<<<grid, block, shared_bytes, stream>>>(current, other, width, height);
        } else {
            if (config.median_kernel_size <= 0 || config.median_kernel_size % 2 == 0 || config.median_kernel_size > kMedianBasicMaxKernelSize) {
                throw std::invalid_argument("AsyncCudaPipeline: invalid median_kernel_size " + std::to_string(config.median_kernel_size));
            }
            median_basic_kernel<<<grid, block, 0, stream>>>(current, other, width, height, config.median_kernel_size);
        }
        CUDA_CHECK_LAST_ERROR();
        std::swap(current, other);
        current_is_a = !current_is_a;
    }
    if (config.sobel_enabled) {
        if (config.sobel_mode < 0 || config.sobel_mode > 3) {
            throw std::invalid_argument("AsyncCudaPipeline: invalid sobel_mode " + std::to_string(config.sobel_mode));
        }
        if (config.use_enhanced) {
            size_t shared_bytes = static_cast<size_t>(block.x + 2) * (block.y + 2) * sizeof(uint8_t);
            switch (config.sobel_mode) {
                case 0: async_sobel_specialized_kernel<0><<<grid, block, shared_bytes, stream>>>(current, other, width, height); break;
                case 1: async_sobel_specialized_kernel<1><<<grid, block, shared_bytes, stream>>>(current, other, width, height); break;
                case 2: async_sobel_specialized_kernel<2><<<grid, block, shared_bytes, stream>>>(current, other, width, height); break;
                case 3: async_sobel_specialized_kernel<3><<<grid, block, shared_bytes, stream>>>(current, other, width, height); break;
            }
        } else {
            sobel_basic_kernel<<<grid, block, 0, stream>>>(current, other, width, height, config.sobel_mode);
        }
        CUDA_CHECK_LAST_ERROR();
        std::swap(current, other);
        current_is_a = !current_is_a;
    }
    if (config.laplacian_enabled) {
        if (config.laplacian_kernel_size <= 0 || config.laplacian_kernel_size % 2 == 0) {
            throw std::invalid_argument("AsyncCudaPipeline: invalid laplacian_kernel_size " + std::to_string(config.laplacian_kernel_size));
        }
        if (config.use_enhanced) {
            int K = config.laplacian_kernel_size;
            int radius = K / 2;
            size_t shared_bytes = static_cast<size_t>(block.x + 2 * radius) * (block.y + 2 * radius) * sizeof(uint8_t);
            CUDA_CHECK(cudaMemcpyToSymbolAsync(a_laplacian_coeffs, config.laplacian_coeffs,
                                                static_cast<size_t>(K) * K * sizeof(float), 0, cudaMemcpyHostToDevice, stream));
            switch (K) {
                case 3: async_laplacian_specialized_kernel<3><<<grid, block, shared_bytes, stream>>>(current, other, width, height, config.laplacian_scale, config.laplacian_delta); break;
                case 5: async_laplacian_specialized_kernel<5><<<grid, block, shared_bytes, stream>>>(current, other, width, height, config.laplacian_scale, config.laplacian_delta); break;
                default: throw std::invalid_argument("AsyncCudaPipeline: laplacian_kernel_size must be one of {3,5} for Enhanced, got " + std::to_string(K));
            }
        } else {
            size_t coeff_count = static_cast<size_t>(config.laplacian_kernel_size) * config.laplacian_kernel_size;
            laplacian_coeffs_dev->upload(config.laplacian_coeffs, coeff_count, stream);
            laplacian_basic_kernel<<<grid, block, 0, stream>>>(current, other, width, height, laplacian_coeffs_dev->get(), config.laplacian_kernel_size, config.laplacian_scale, config.laplacian_delta);
            CUDA_CHECK_LAST_ERROR();
        }
        CUDA_CHECK_LAST_ERROR();
        std::swap(current, other);
        current_is_a = !current_is_a;
    }
    if (config.threshold_enabled) {
        if (config.use_enhanced) {
            launch_enhanced_threshold(current, other, width, height, chunk_batch_size,
                                       config.threshold_value, config.threshold_max_value, block, stream);
        } else {
            threshold_basic_kernel<<<grid, block, 0, stream>>>(current, other, width, height, config.threshold_value, config.threshold_max_value);
        }
        CUDA_CHECK_LAST_ERROR();
        std::swap(current, other);
        current_is_a = !current_is_a;
    }
    return current_is_a;
}

}  // namespace

AsyncCudaPipeline::BufferSet::~BufferSet() {
    if (h2d_done) cudaEventDestroy(h2d_done);
    if (compute_done) cudaEventDestroy(compute_done);
    if (d2h_done) cudaEventDestroy(d2h_done);
    if (pinned_input) cudaFreeHost(pinned_input);
    if (pinned_output) cudaFreeHost(pinned_output);
}

AsyncCudaPipeline::AsyncCudaPipeline() {
    CUDA_CHECK(cudaStreamCreate(&h2d_stream_));
    CUDA_CHECK(cudaStreamCreate(&compute_stream_));
    CUDA_CHECK(cudaStreamCreate(&d2h_stream_));
}

AsyncCudaPipeline::~AsyncCudaPipeline() {
    release();
}

void AsyncCudaPipeline::release() {
    if (released_) return;
    if (h2d_stream_) { cudaStreamDestroy(h2d_stream_); h2d_stream_ = nullptr; }
    if (compute_stream_) { cudaStreamDestroy(compute_stream_); compute_stream_ = nullptr; }
    if (d2h_stream_) { cudaStreamDestroy(d2h_stream_); d2h_stream_ = nullptr; }
    released_ = true;
}

std::pair<std::vector<uint8_t>, AsyncPipelineTiming> AsyncCudaPipeline::run(
    const uint8_t* host_batch, int batch_size, int height, int width,
    const AsyncPipelineConfig& config, int chunk_size, int num_buffers, bool use_pinned_staging) {
    if (released_) {
        throw std::runtime_error("AsyncCudaPipeline::run() called after release().");
    }
    if (batch_size <= 0 || height <= 0 || width <= 0) {
        throw std::invalid_argument(
            "AsyncCudaPipeline::run(): batch_size/height/width must be positive, got batch_size=" +
            std::to_string(batch_size) + " height=" + std::to_string(height) + " width=" + std::to_string(width));
    }
    if (chunk_size <= 0) {
        throw std::invalid_argument("AsyncCudaPipeline::run(): chunk_size must be positive, got " + std::to_string(chunk_size));
    }
    if (num_buffers < 1 || num_buffers > 4) {
        throw std::invalid_argument("AsyncCudaPipeline::run(): num_buffers must be in [1,4], got " + std::to_string(num_buffers));
    }

    int effective_chunk_size = std::min(chunk_size, batch_size);
    int num_chunks = (batch_size + effective_chunk_size - 1) / effective_chunk_size;
    int actual_num_buffers = std::min(num_buffers, num_chunks);

    AsyncPipelineTiming timing;
    timing.chunk_size_used = effective_chunk_size;
    timing.num_chunks = num_chunks;
    timing.num_buffers_used = actual_num_buffers;
    timing.used_pinned_staging = use_pinned_staging;

    HostTimer total_timer;
    total_timer.start();

    HostTimer alloc_timer;
    alloc_timer.start();
    std::vector<std::unique_ptr<BufferSet>> buffer_sets(actual_num_buffers);
    size_t chunk_bytes_max = static_cast<size_t>(effective_chunk_size) * height * width;
    for (int i = 0; i < actual_num_buffers; ++i) {
        auto bs = std::make_unique<BufferSet>();
        bs->buf_a = std::make_unique<GpuImageBatch>(effective_chunk_size, height, width);
        bs->buf_b = std::make_unique<GpuImageBatch>(effective_chunk_size, height, width);
        if (config.gaussian_enabled) {
            if (config.use_enhanced) {
                bs->intermediate = std::make_unique<DeviceBuffer<float>>(
                    static_cast<size_t>(effective_chunk_size) * height * width);
            } else {
                bs->gaussian_coeffs_dev = std::make_unique<DeviceBuffer<float>>(
                    static_cast<size_t>(config.gaussian_kernel_size) * config.gaussian_kernel_size);
            }
        }
        if (config.laplacian_enabled && !config.use_enhanced) {
            bs->laplacian_coeffs_dev = std::make_unique<DeviceBuffer<float>>(
                static_cast<size_t>(config.laplacian_kernel_size) * config.laplacian_kernel_size);
        }
        CUDA_CHECK(cudaEventCreate(&bs->h2d_done));
        CUDA_CHECK(cudaEventCreate(&bs->compute_done));
        CUDA_CHECK(cudaEventCreate(&bs->d2h_done));
        if (use_pinned_staging) {
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&bs->pinned_input), chunk_bytes_max));
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&bs->pinned_output), chunk_bytes_max));
            bs->pinned_capacity_bytes = chunk_bytes_max;
        }
        buffer_sets[i] = std::move(bs);
    }
    alloc_timer.stop();
    timing.alloc_ms = alloc_timer.elapsed_ms();

    std::vector<uint8_t> host_output(static_cast<size_t>(batch_size) * height * width);
    const size_t plane_bytes = static_cast<size_t>(height) * width;
    std::vector<int> chunk_in_slot(actual_num_buffers, -1);

    cudaEvent_t origin, h2d_first, h2d_last, compute_first, compute_last, d2h_first, d2h_last;
    CUDA_CHECK(cudaEventCreate(&origin));
    CUDA_CHECK(cudaEventCreate(&h2d_first));
    CUDA_CHECK(cudaEventCreate(&h2d_last));
    CUDA_CHECK(cudaEventCreate(&compute_first));
    CUDA_CHECK(cudaEventCreate(&compute_last));
    CUDA_CHECK(cudaEventCreate(&d2h_first));
    CUDA_CHECK(cudaEventCreate(&d2h_last));
    CUDA_CHECK(cudaEventRecord(origin, h2d_stream_));
    CUDA_CHECK(cudaEventSynchronize(origin));
    bool h2d_first_recorded = false, compute_first_recorded = false, d2h_first_recorded = false;

    float host_stage_ms_total = 0.0f;
    HostTimer stage_timer;

    try {
        for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            int slot = chunk_idx % actual_num_buffers;
            BufferSet& bs = *buffer_sets[slot];
            size_t chunk_offset = static_cast<size_t>(chunk_idx) * effective_chunk_size;
            int this_chunk_size = std::min(effective_chunk_size, batch_size - static_cast<int>(chunk_offset));
            size_t this_chunk_bytes = static_cast<size_t>(this_chunk_size) * plane_bytes;
            const uint8_t* chunk_src = host_batch + chunk_offset * plane_bytes;
            uint8_t* chunk_dst = host_output.data() + chunk_offset * plane_bytes;

            if (chunk_in_slot[slot] >= 0) {
                CUDA_CHECK(cudaEventSynchronize(bs.d2h_done));
                if (use_pinned_staging) {
                    int prev_chunk = chunk_in_slot[slot];
                    size_t prev_offset = static_cast<size_t>(prev_chunk) * effective_chunk_size;
                    int prev_size = std::min(effective_chunk_size, batch_size - static_cast<int>(prev_offset));
                    size_t prev_bytes = static_cast<size_t>(prev_size) * plane_bytes;
                    stage_timer.start();
                    std::memcpy(host_output.data() + prev_offset * plane_bytes, bs.pinned_output, prev_bytes);
                    stage_timer.stop();
                    host_stage_ms_total += stage_timer.elapsed_ms();
                }
            }
            chunk_in_slot[slot] = chunk_idx;

            const void* h2d_src = chunk_src;
            if (use_pinned_staging) {
                stage_timer.start();
                std::memcpy(bs.pinned_input, chunk_src, this_chunk_bytes);
                stage_timer.stop();
                host_stage_ms_total += stage_timer.elapsed_ms();
                h2d_src = bs.pinned_input;
            }
            if (!h2d_first_recorded) {
                CUDA_CHECK(cudaEventRecord(h2d_first, h2d_stream_));
                h2d_first_recorded = true;
            }
            CUDA_CHECK(cudaMemcpyAsync(bs.buf_a->data(), h2d_src, this_chunk_bytes, cudaMemcpyHostToDevice, h2d_stream_));
            CUDA_CHECK(cudaEventRecord(bs.h2d_done, h2d_stream_));
            CUDA_CHECK(cudaEventRecord(h2d_last, h2d_stream_));

            CUDA_CHECK(cudaStreamWaitEvent(compute_stream_, bs.h2d_done, 0));
            if (!compute_first_recorded) {
                CUDA_CHECK(cudaEventRecord(compute_first, compute_stream_));
                compute_first_recorded = true;
            }
            bool final_is_a = launch_stages(
                bs.buf_a->data(), bs.buf_b->data(), bs.intermediate ? bs.intermediate->get() : nullptr,
                bs.gaussian_coeffs_dev.get(), bs.laplacian_coeffs_dev.get(),
                this_chunk_size, height, width, config, compute_stream_);
            CUDA_CHECK(cudaEventRecord(bs.compute_done, compute_stream_));
            CUDA_CHECK(cudaEventRecord(compute_last, compute_stream_));

            CUDA_CHECK(cudaStreamWaitEvent(d2h_stream_, bs.compute_done, 0));
            uint8_t* final_buf = final_is_a ? bs.buf_a->data() : bs.buf_b->data();
            void* d2h_dst = use_pinned_staging ? static_cast<void*>(bs.pinned_output) : static_cast<void*>(chunk_dst);
            if (!d2h_first_recorded) {
                CUDA_CHECK(cudaEventRecord(d2h_first, d2h_stream_));
                d2h_first_recorded = true;
            }
            CUDA_CHECK(cudaMemcpyAsync(d2h_dst, final_buf, this_chunk_bytes, cudaMemcpyDeviceToHost, d2h_stream_));
            CUDA_CHECK(cudaEventRecord(bs.d2h_done, d2h_stream_));
            CUDA_CHECK(cudaEventRecord(d2h_last, d2h_stream_));
        }

        for (int slot = 0; slot < actual_num_buffers; ++slot) {
            if (chunk_in_slot[slot] < 0) continue;
            BufferSet& bs = *buffer_sets[slot];
            CUDA_CHECK(cudaEventSynchronize(bs.d2h_done));
            if (use_pinned_staging) {
                int c = chunk_in_slot[slot];
                size_t off = static_cast<size_t>(c) * effective_chunk_size;
                int sz = std::min(effective_chunk_size, batch_size - static_cast<int>(off));
                size_t bytes = static_cast<size_t>(sz) * plane_bytes;
                stage_timer.start();
                std::memcpy(host_output.data() + off * plane_bytes, bs.pinned_output, bytes);
                stage_timer.stop();
                host_stage_ms_total += stage_timer.elapsed_ms();
            }
        }
        // Per-buffer-set drain above only guarantees each slot's OWN last d2h_done has
        // completed; h2d_last/compute_last/d2h_last are recorded immediately AFTER those
        // per-slot events on their respective streams, so a later stream position -- querying
        // cudaEventElapsedTime against them before the streams have fully drained returns
        // cudaErrorNotReady. Explicit full-stream syncs make them safely queryable.
        CUDA_CHECK(cudaStreamSynchronize(h2d_stream_));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
        CUDA_CHECK(cudaStreamSynchronize(d2h_stream_));
    } catch (...) {
        cudaEventDestroy(origin); cudaEventDestroy(h2d_first); cudaEventDestroy(h2d_last);
        cudaEventDestroy(compute_first); cudaEventDestroy(compute_last);
        cudaEventDestroy(d2h_first); cudaEventDestroy(d2h_last);
        throw;
    }

    timing.host_stage_ms = host_stage_ms_total;

    float h2d_start_ms = 0, h2d_end_ms = 0, compute_start_ms = 0, compute_end_ms = 0, d2h_start_ms = 0, d2h_end_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_start_ms, origin, h2d_first));
    CUDA_CHECK(cudaEventElapsedTime(&h2d_end_ms, origin, h2d_last));
    CUDA_CHECK(cudaEventElapsedTime(&compute_start_ms, origin, compute_first));
    CUDA_CHECK(cudaEventElapsedTime(&compute_end_ms, origin, compute_last));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_start_ms, origin, d2h_first));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_end_ms, origin, d2h_last));
    timing.h2d_stream_span_ms = h2d_end_ms - h2d_start_ms;
    timing.compute_stream_span_ms = compute_end_ms - compute_start_ms;
    timing.d2h_stream_span_ms = d2h_end_ms - d2h_start_ms;
    timing.h2d_compute_overlap_ms = std::max(0.0f, std::min(h2d_end_ms, compute_end_ms) - std::max(h2d_start_ms, compute_start_ms));
    timing.compute_d2h_overlap_ms = std::max(0.0f, std::min(compute_end_ms, d2h_end_ms) - std::max(compute_start_ms, d2h_start_ms));

    cudaEventDestroy(origin); cudaEventDestroy(h2d_first); cudaEventDestroy(h2d_last);
    cudaEventDestroy(compute_first); cudaEventDestroy(compute_last);
    cudaEventDestroy(d2h_first); cudaEventDestroy(d2h_last);

    total_timer.stop();
    timing.total_ms = total_timer.elapsed_ms();

    return {std::move(host_output), timing};
}

}  // namespace xray_cuda
