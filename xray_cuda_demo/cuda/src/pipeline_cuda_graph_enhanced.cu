// Section 20E: EXPERIMENTAL CUDA Graph pipeline for the ENHANCED filter
// kernels. See pipeline_cuda_graph_enhanced.cuh's file header for why this
// file duplicates kernel bodies rather than calling the existing
// *_enhanced_dispatch() kernels directly (they have internal/anonymous-
// namespace linkage, unlike Section 20D's Basic kernels).
//
// KERNEL EQUIVALENCE PROVENANCE (spec item 8's "kernel equivalence
// proof" -- exact source of every duplicated __global__ kernel and
// __device__ helper below, verified by direct line-by-line comparison
// against the file/line cited, at the time of this section's work):
//
//   gaussian_specialized_h_kernel<K>  <- cuda/src/gaussian_enhanced.cu:193-224
//   gaussian_specialized_v_kernel<K>  <- cuda/src/gaussian_enhanced.cu:226-258
//   d_gaussian_coeffs_1d (constant)   <- cuda/src/gaussian_enhanced.cu:11 (own copy, same size)
//   median_network9_kernel            <- cuda/src/median_enhanced.cu:93-127
//   pix_sort                          <- cuda/src/median_enhanced.cu:15-19
//   load_median_tile                  <- cuda/src/median_enhanced.cu:26-41
//   sobel_specialized_kernel<MODE>    <- cuda/src/sobel_enhanced.cu:136-180
//   sobel_finalize                    <- cuda/src/sobel_enhanced.cu:22-25
//   load_sobel_tile                   <- cuda/src/sobel_enhanced.cu:45-59
//   laplacian_specialized_kernel<K>   <- cuda/src/laplacian_enhanced.cu:145-172
//   laplacian_finalize                <- cuda/src/laplacian_enhanced.cu:49-54
//   load_laplacian_tile               <- cuda/src/laplacian_enhanced.cu:61-76
//   d_laplacian_coeffs (constant)     <- cuda/src/laplacian_enhanced.cu:9 (own copy, same size)
//   threshold_scalar_kernel           <- cuda/src/threshold_enhanced.cu:12-22
//   threshold_vectorized_kernel       <- cuda/src/threshold_enhanced.cu:31-49
//
// reflect101()/clamp_index() are NOT duplicated -- both are declared in
// the shared cuda/include/gpu_image.cuh header (external linkage) and
// are used here exactly as included, unmodified, same as every other
// file in this project.
//
// Every duplicated kernel is exercised by tests/test_enhanced_cuda_graph.py
// against the real *_enhanced_dispatch() production functions for
// bit-exact output equality -- the strongest form of the equivalence
// proof, on top of the textual provenance above.

#include "pipeline_cuda_graph_enhanced.cuh"
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

size_t hash_combine(size_t seed, size_t v) {
    return seed ^ (v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

uint32_t float_bits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return bits;
}

// --------------------------------------------------------------------------
// Duplicated Enhanced kernel bodies (production, Specialized/Network3x3/
// Vectorized variants only) -- see this file's top-of-file provenance
// comment. Own constant-memory symbols (distinct from production's
// d_gaussian_coeffs_1d/d_laplacian_coeffs -- this file never touches
// those, avoiding any cross-symbol interaction with the production
// dispatch path).
// --------------------------------------------------------------------------

constexpr int kGraphGaussianMaxKernelSize = 9;
__constant__ float g_gaussian_coeffs_1d[kGraphGaussianMaxKernelSize];

template <int K>
__global__ void graph_gaussian_specialized_h_kernel(const uint8_t* input, float* output, int width, int height) {
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
        sum += g_gaussian_coeffs_1d[k + radius] * static_cast<float>(s_tile_u8_spec[ty * tile_w + (tx + radius + k)]);
    }
    out[y * width + x] = sum;
}

template <int K>
__global__ void graph_gaussian_specialized_v_kernel(const float* input, uint8_t* output, int width, int height) {
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
        sum += g_gaussian_coeffs_1d[k + radius] * s_tile_f32_spec[(ty + radius + k) * blockDim.x + tx];
    }
    int rounded = __float2int_rn(sum);
    rounded = max(0, min(255, rounded));
    out[y * width + x] = static_cast<uint8_t>(rounded);
}

__device__ __forceinline__ void graph_pix_sort(uint8_t& a, uint8_t& b) {
    uint8_t lo = min(a, b);
    uint8_t hi = max(a, b);
    a = lo;
    b = hi;
}

__device__ __forceinline__ void graph_load_median_tile(
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

__global__ void graph_median_network9_kernel(const uint8_t* input, uint8_t* output, int width, int height) {
    constexpr int radius = 1;
    extern __shared__ uint8_t s_tile_net9[];
    int tile_w = static_cast<int>(blockDim.x) + 2 * radius;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t plane = static_cast<size_t>(height) * width;
    const uint8_t* in = input + static_cast<size_t>(blockIdx.z) * plane;
    uint8_t* out = output + static_cast<size_t>(blockIdx.z) * plane;

    graph_load_median_tile(s_tile_net9, in, width, height, radius, tile_w);
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

    graph_pix_sort(p[1], p[2]); graph_pix_sort(p[4], p[5]); graph_pix_sort(p[7], p[8]);
    graph_pix_sort(p[0], p[1]); graph_pix_sort(p[3], p[4]); graph_pix_sort(p[6], p[7]);
    graph_pix_sort(p[1], p[2]); graph_pix_sort(p[4], p[5]); graph_pix_sort(p[7], p[8]);
    graph_pix_sort(p[0], p[3]); graph_pix_sort(p[5], p[8]); graph_pix_sort(p[4], p[7]);
    graph_pix_sort(p[3], p[6]); graph_pix_sort(p[1], p[4]); graph_pix_sort(p[2], p[5]);
    graph_pix_sort(p[4], p[7]); graph_pix_sort(p[4], p[2]); graph_pix_sort(p[6], p[4]);
    graph_pix_sort(p[4], p[2]);

    out[y * width + x] = p[4];
}

__device__ __forceinline__ uint8_t graph_sobel_finalize(float raw) {
    int rounded = __float2int_rn(fabsf(raw));
    rounded = max(0, min(255, rounded));
    return static_cast<uint8_t>(rounded);
}

__device__ __forceinline__ void graph_load_sobel_tile(
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
__global__ void graph_sobel_specialized_kernel(const uint8_t* input, uint8_t* output, int width, int height) {
    extern __shared__ uint8_t s_tile_sobel_spec[];
    int tile_w = static_cast<int>(blockDim.x) + 2;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t plane = static_cast<size_t>(height) * width;
    const uint8_t* in = input + static_cast<size_t>(blockIdx.z) * plane;
    uint8_t* out = output + static_cast<size_t>(blockIdx.z) * plane;

    graph_load_sobel_tile(s_tile_sobel_spec, in, width, height, tile_w);
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

    out[y * width + x] = graph_sobel_finalize(raw);
}

constexpr int kGraphLaplacianMaxKernelSize = 5;
__constant__ float g_laplacian_coeffs[kGraphLaplacianMaxKernelSize * kGraphLaplacianMaxKernelSize];

__device__ __forceinline__ uint8_t graph_laplacian_finalize(float sum, float scale, float delta) {
    float scaled = sum * scale + delta;
    int rounded = __float2int_rn(fabsf(scaled));
    rounded = max(0, min(255, rounded));
    return static_cast<uint8_t>(rounded);
}

__device__ __forceinline__ void graph_load_laplacian_tile(
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
__global__ void graph_laplacian_specialized_kernel(
    const uint8_t* input, uint8_t* output, int width, int height, float scale, float delta) {
    constexpr int radius = K / 2;
    extern __shared__ uint8_t s_tile_lap_spec[];
    int tile_w = static_cast<int>(blockDim.x) + 2 * radius;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t plane = static_cast<size_t>(height) * width;
    const uint8_t* in = input + static_cast<size_t>(blockIdx.z) * plane;
    uint8_t* out = output + static_cast<size_t>(blockIdx.z) * plane;

    graph_load_laplacian_tile(s_tile_lap_spec, in, width, height, radius, tile_w);
    __syncthreads();

    if (x >= width || y >= height) return;

    int lx = threadIdx.x, ly = threadIdx.y;
    float sum = 0.0f;
#pragma unroll
    for (int ky = -radius; ky <= radius; ++ky) {
#pragma unroll
        for (int kx = -radius; kx <= radius; ++kx) {
            float v = static_cast<float>(s_tile_lap_spec[(ly + radius + ky) * tile_w + (lx + radius + kx)]);
            sum += g_laplacian_coeffs[(ky + radius) * K + (kx + radius)] * v;
        }
    }
    out[y * width + x] = graph_laplacian_finalize(sum, scale, delta);
}

__global__ void graph_threshold_scalar_kernel(
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

__global__ void graph_threshold_vectorized_kernel(
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

void launch_threshold(
    const uint8_t* input, uint8_t* output, int width, int height, int batch_size,
    uint8_t threshold_value, uint8_t max_value, dim3 block, cudaStream_t stream) {
    if (width % 4 == 0) {
        int width4 = width / 4;
        dim3 grid2d((static_cast<unsigned int>(width4) + block.x - 1) / block.x,
                    (static_cast<unsigned int>(height) + block.y - 1) / block.y);
        dim3 grid(grid2d.x, grid2d.y, static_cast<unsigned int>(batch_size));
        graph_threshold_vectorized_kernel<<<grid, block, 0, stream>>>(input, output, width, height, threshold_value, max_value);
    } else {
        dim3 grid2d = compute_launch_grid(width, height, block);
        dim3 grid(grid2d.x, grid2d.y, static_cast<unsigned int>(batch_size));
        graph_threshold_scalar_kernel<<<grid, block, 0, stream>>>(input, output, width, height, threshold_value, max_value);
    }
}

}  // namespace

// --------------------------------------------------------------------------
// CudaGraphEnhancedPipeline
// --------------------------------------------------------------------------

bool CudaGraphEnhancedPipeline::CacheKey::operator==(const CacheKey& o) const {
    return batch_size == o.batch_size && height == o.height && width == o.width && scope == o.scope &&
           gaussian_enabled == o.gaussian_enabled && gaussian_kernel_size == o.gaussian_kernel_size &&
           median_enabled == o.median_enabled &&
           sobel_enabled == o.sobel_enabled && sobel_mode == o.sobel_mode &&
           laplacian_enabled == o.laplacian_enabled && laplacian_kernel_size == o.laplacian_kernel_size &&
           laplacian_scale_bits == o.laplacian_scale_bits && laplacian_delta_bits == o.laplacian_delta_bits &&
           threshold_enabled == o.threshold_enabled && threshold_value == o.threshold_value &&
           threshold_max_value == o.threshold_max_value;
}

size_t CudaGraphEnhancedPipeline::CacheKeyHash::operator()(const CacheKey& k) const {
    size_t h = 0;
    for (int v : {k.batch_size, k.height, k.width, k.scope, static_cast<int>(k.gaussian_enabled),
                   k.gaussian_kernel_size, static_cast<int>(k.median_enabled),
                   static_cast<int>(k.sobel_enabled), k.sobel_mode, static_cast<int>(k.laplacian_enabled),
                   k.laplacian_kernel_size, static_cast<int>(k.threshold_enabled),
                   static_cast<int>(k.threshold_value), static_cast<int>(k.threshold_max_value)}) {
        h = hash_combine(h, static_cast<size_t>(v));
    }
    h = hash_combine(h, k.laplacian_scale_bits);
    h = hash_combine(h, k.laplacian_delta_bits);
    return h;
}

CudaGraphEnhancedPipeline::GraphEntry::~GraphEntry() {
    if (exec) cudaGraphExecDestroy(exec);
    if (graph) cudaGraphDestroy(graph);
}

CudaGraphEnhancedPipeline::CudaGraphEnhancedPipeline() {
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

CudaGraphEnhancedPipeline::~CudaGraphEnhancedPipeline() {
    release();
}

void CudaGraphEnhancedPipeline::release() {
    if (released_) return;
    cache_.clear();
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    released_ = true;
}

void CudaGraphEnhancedPipeline::clear_cache() {
    cache_.clear();
}

CudaGraphEnhancedPipeline::GraphEntry* CudaGraphEnhancedPipeline::build_entry(
    const CacheKey& key, const EnhancedGraphConfig& config,
    int batch_size, int height, int width, EnhancedGraphScope scope,
    float& capture_ms, float& instantiate_ms, std::string& fallback_reason) {
    auto entry = std::make_unique<GraphEntry>();
    entry->buf_a = std::make_unique<GpuImageBatch>(batch_size, height, width);
    entry->buf_b = std::make_unique<GpuImageBatch>(batch_size, height, width);
    entry->total_bytes = entry->buf_a->nbytes();
    entry->capture_scratch.resize(entry->total_bytes);

    if (config.gaussian_enabled) {
        entry->intermediate = std::make_unique<DeviceBuffer<float>>(
            static_cast<size_t>(batch_size) * static_cast<size_t>(height) * static_cast<size_t>(width));
    }

    int enabled_count = (config.gaussian_enabled ? 1 : 0) + (config.median_enabled ? 1 : 0) +
                         (config.sobel_enabled ? 1 : 0) + (config.laplacian_enabled ? 1 : 0) +
                         (config.threshold_enabled ? 1 : 0);
    entry->final_is_a = (enabled_count % 2 == 0);

    dim3 block = default_block_dim();

    HostTimer capture_timer;
    capture_timer.start();

    cudaError_t begin_err = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal);
    if (begin_err != cudaSuccess) {
        fallback_reason = std::string("cudaStreamBeginCapture failed: ") + cudaGetErrorString(begin_err);
        return nullptr;
    }

    bool capture_body_ok = true;
    std::string capture_body_error;
    try {
        GpuImageBatch* current = entry->buf_a.get();
        GpuImageBatch* other = entry->buf_b.get();

        if (scope == EnhancedGraphScope::FullPipeline) {
            CUDA_CHECK(cudaMemcpyAsync(current->data(), host_batch_for_capture_,
                                        entry->total_bytes, cudaMemcpyHostToDevice, stream_));
        }

        if (config.gaussian_enabled) {
            int K = config.gaussian_kernel_size;
            int radius = K / 2;
            dim3 grid2d = compute_launch_grid(width, height, block);
            dim3 grid(grid2d.x, grid2d.y, static_cast<unsigned int>(batch_size));
            size_t h_shared_bytes = static_cast<size_t>(block.x + 2 * radius) * block.y * sizeof(uint8_t);
            size_t v_shared_bytes = static_cast<size_t>(block.x) * (block.y + 2 * radius) * sizeof(float);

            CUDA_CHECK(cudaMemcpyToSymbolAsync(
                g_gaussian_coeffs_1d, config.gaussian_coeffs_1d, K * sizeof(float), 0,
                cudaMemcpyHostToDevice, stream_));

            float* intermediate = entry->intermediate->get();
            switch (K) {
                case 3: graph_gaussian_specialized_h_kernel<3><<<grid, block, h_shared_bytes, stream_>>>(current->data(), intermediate, width, height); break;
                case 5: graph_gaussian_specialized_h_kernel<5><<<grid, block, h_shared_bytes, stream_>>>(current->data(), intermediate, width, height); break;
                case 7: graph_gaussian_specialized_h_kernel<7><<<grid, block, h_shared_bytes, stream_>>>(current->data(), intermediate, width, height); break;
                case 9: graph_gaussian_specialized_h_kernel<9><<<grid, block, h_shared_bytes, stream_>>>(current->data(), intermediate, width, height); break;
                default: throw std::invalid_argument("CudaGraphEnhancedPipeline: gaussian_kernel_size must be one of {3,5,7,9} for Specialized, got " + std::to_string(K));
            }
            CUDA_CHECK_LAST_ERROR();
            switch (K) {
                case 3: graph_gaussian_specialized_v_kernel<3><<<grid, block, v_shared_bytes, stream_>>>(intermediate, other->data(), width, height); break;
                case 5: graph_gaussian_specialized_v_kernel<5><<<grid, block, v_shared_bytes, stream_>>>(intermediate, other->data(), width, height); break;
                case 7: graph_gaussian_specialized_v_kernel<7><<<grid, block, v_shared_bytes, stream_>>>(intermediate, other->data(), width, height); break;
                case 9: graph_gaussian_specialized_v_kernel<9><<<grid, block, v_shared_bytes, stream_>>>(intermediate, other->data(), width, height); break;
            }
            CUDA_CHECK_LAST_ERROR();
            std::swap(current, other);
        }
        if (config.median_enabled) {
            dim3 grid2d = compute_launch_grid(width, height, block);
            dim3 grid(grid2d.x, grid2d.y, static_cast<unsigned int>(batch_size));
            size_t shared_bytes = static_cast<size_t>(block.x + 2) * (block.y + 2) * sizeof(uint8_t);
            graph_median_network9_kernel<<<grid, block, shared_bytes, stream_>>>(current->data(), other->data(), width, height);
            CUDA_CHECK_LAST_ERROR();
            std::swap(current, other);
        }
        if (config.sobel_enabled) {
            dim3 grid2d = compute_launch_grid(width, height, block);
            dim3 grid(grid2d.x, grid2d.y, static_cast<unsigned int>(batch_size));
            size_t shared_bytes = static_cast<size_t>(block.x + 2) * (block.y + 2) * sizeof(uint8_t);
            switch (config.sobel_mode) {
                case 0: graph_sobel_specialized_kernel<0><<<grid, block, shared_bytes, stream_>>>(current->data(), other->data(), width, height); break;
                case 1: graph_sobel_specialized_kernel<1><<<grid, block, shared_bytes, stream_>>>(current->data(), other->data(), width, height); break;
                case 2: graph_sobel_specialized_kernel<2><<<grid, block, shared_bytes, stream_>>>(current->data(), other->data(), width, height); break;
                case 3: graph_sobel_specialized_kernel<3><<<grid, block, shared_bytes, stream_>>>(current->data(), other->data(), width, height); break;
                default: throw std::invalid_argument("CudaGraphEnhancedPipeline: sobel_mode must be in [0,3], got " + std::to_string(config.sobel_mode));
            }
            CUDA_CHECK_LAST_ERROR();
            std::swap(current, other);
        }
        if (config.laplacian_enabled) {
            int K = config.laplacian_kernel_size;
            int radius = K / 2;
            dim3 grid2d = compute_launch_grid(width, height, block);
            dim3 grid(grid2d.x, grid2d.y, static_cast<unsigned int>(batch_size));
            size_t shared_bytes = static_cast<size_t>(block.x + 2 * radius) * (block.y + 2 * radius) * sizeof(uint8_t);

            CUDA_CHECK(cudaMemcpyToSymbolAsync(
                g_laplacian_coeffs, config.laplacian_coeffs, static_cast<size_t>(K) * K * sizeof(float), 0,
                cudaMemcpyHostToDevice, stream_));

            switch (K) {
                case 3: graph_laplacian_specialized_kernel<3><<<grid, block, shared_bytes, stream_>>>(current->data(), other->data(), width, height, config.laplacian_scale, config.laplacian_delta); break;
                case 5: graph_laplacian_specialized_kernel<5><<<grid, block, shared_bytes, stream_>>>(current->data(), other->data(), width, height, config.laplacian_scale, config.laplacian_delta); break;
                default: throw std::invalid_argument("CudaGraphEnhancedPipeline: laplacian_kernel_size must be one of {3,5} for Specialized, got " + std::to_string(K));
            }
            CUDA_CHECK_LAST_ERROR();
            std::swap(current, other);
        }
        if (config.threshold_enabled) {
            launch_threshold(current->data(), other->data(), width, height, batch_size,
                              config.threshold_value, config.threshold_max_value, block, stream_);
            CUDA_CHECK_LAST_ERROR();
            std::swap(current, other);
        }

        if (scope == EnhancedGraphScope::FullPipeline) {
            CUDA_CHECK(cudaMemcpyAsync(entry->capture_scratch.data(), current->data(),
                                        entry->total_bytes, cudaMemcpyDeviceToHost, stream_));
        }
    } catch (const std::exception& e) {
        capture_body_ok = false;
        capture_body_error = e.what();
    }

    cudaError_t end_err = cudaStreamEndCapture(stream_, &entry->graph);
    if (!capture_body_ok) {
        if (entry->graph) { cudaGraphDestroy(entry->graph); entry->graph = nullptr; }
        fallback_reason = "capture body raised: " + capture_body_error;
        return nullptr;
    }
    if (end_err != cudaSuccess) {
        fallback_reason = std::string("cudaStreamEndCapture failed: ") + cudaGetErrorString(end_err);
        return nullptr;
    }
    capture_timer.stop();
    capture_ms = capture_timer.elapsed_ms();

    size_t num_nodes = 0;
    CUDA_CHECK(cudaGraphGetNodes(entry->graph, nullptr, &num_nodes));
    std::vector<cudaGraphNode_t> nodes(num_nodes);
    CUDA_CHECK(cudaGraphGetNodes(entry->graph, nodes.data(), &num_nodes));

    if (config.gaussian_enabled) CUDA_CHECK(cudaGetSymbolAddress(&entry->gaussian_symbol_addr, g_gaussian_coeffs_1d));
    if (config.laplacian_enabled) CUDA_CHECK(cudaGetSymbolAddress(&entry->laplacian_symbol_addr, g_laplacian_coeffs));

    for (auto node : nodes) {
        cudaGraphNodeType type;
        CUDA_CHECK(cudaGraphNodeGetType(node, &type));
        if (type != cudaGraphNodeTypeMemcpy) continue;
        cudaMemcpy3DParms params = {};
        CUDA_CHECK(cudaGraphMemcpyNodeGetParams(node, &params));
        void* dst_ptr = params.dstPtr.ptr;
        if (scope == EnhancedGraphScope::FullPipeline && dst_ptr == entry->buf_a->data()) {
            entry->h2d_node = node;
        } else if (scope == EnhancedGraphScope::FullPipeline && dst_ptr == entry->capture_scratch.data()) {
            entry->d2h_node = node;
        } else if (entry->gaussian_symbol_addr && dst_ptr == entry->gaussian_symbol_addr) {
            entry->gaussian_coeff_node = node;
        } else if (entry->laplacian_symbol_addr && dst_ptr == entry->laplacian_symbol_addr) {
            entry->laplacian_coeff_node = node;
        }
    }

    HostTimer inst_timer;
    inst_timer.start();
    cudaError_t inst_err = cudaGraphInstantiate(&entry->exec, entry->graph, 0);
    inst_timer.stop();
    if (inst_err != cudaSuccess) {
        fallback_reason = std::string("cudaGraphInstantiate failed: ") + cudaGetErrorString(inst_err);
        cudaGraphDestroy(entry->graph);
        entry->graph = nullptr;
        return nullptr;
    }
    instantiate_ms = inst_timer.elapsed_ms();

    GraphEntry* raw = entry.get();
    cache_.emplace(key, std::move(entry));
    return raw;
}

std::pair<std::vector<uint8_t>, EnhancedGraphTiming> CudaGraphEnhancedPipeline::run(
    const uint8_t* host_batch, int batch_size, int height, int width,
    const EnhancedGraphConfig& config, bool use_graph, EnhancedGraphScope scope) {
    if (released_) {
        throw std::runtime_error("CudaGraphEnhancedPipeline::run() called after release().");
    }

    EnhancedGraphTiming timing;
    const size_t total_bytes =
        static_cast<size_t>(batch_size) * static_cast<size_t>(height) * static_cast<size_t>(width);
    std::vector<uint8_t> host_output(total_bytes);
    dim3 block = default_block_dim();

    if (!use_graph) {
        HostTimer alloc_timer;
        alloc_timer.start();
        GpuImageBatch buf_a(batch_size, height, width);
        GpuImageBatch buf_b(batch_size, height, width);
        std::unique_ptr<DeviceBuffer<float>> intermediate;
        if (config.gaussian_enabled) {
            intermediate = std::make_unique<DeviceBuffer<float>>(
                static_cast<size_t>(batch_size) * static_cast<size_t>(height) * static_cast<size_t>(width));
        }
        alloc_timer.stop();
        timing.alloc_ms = alloc_timer.elapsed_ms();

        CudaTimer h2d_timer;
        h2d_timer.start(stream_);
        CUDA_CHECK(cudaMemcpyAsync(buf_a.data(), host_batch, total_bytes, cudaMemcpyHostToDevice, stream_));
        h2d_timer.stop(stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        timing.h2d_ms = h2d_timer.elapsed_ms();

        GpuImageBatch* current = &buf_a;
        GpuImageBatch* other = &buf_b;

        CudaTimer compute_timer;
        compute_timer.start(stream_);
        if (config.gaussian_enabled) {
            int K = config.gaussian_kernel_size;
            int radius = K / 2;
            dim3 grid2d = compute_launch_grid(width, height, block);
            dim3 grid(grid2d.x, grid2d.y, static_cast<unsigned int>(batch_size));
            size_t h_shared_bytes = static_cast<size_t>(block.x + 2 * radius) * block.y * sizeof(uint8_t);
            size_t v_shared_bytes = static_cast<size_t>(block.x) * (block.y + 2 * radius) * sizeof(float);
            CUDA_CHECK(cudaMemcpyToSymbolAsync(g_gaussian_coeffs_1d, config.gaussian_coeffs_1d, K * sizeof(float), 0, cudaMemcpyHostToDevice, stream_));
            float* interm = intermediate->get();
            switch (K) {
                case 3: graph_gaussian_specialized_h_kernel<3><<<grid, block, h_shared_bytes, stream_>>>(current->data(), interm, width, height); break;
                case 5: graph_gaussian_specialized_h_kernel<5><<<grid, block, h_shared_bytes, stream_>>>(current->data(), interm, width, height); break;
                case 7: graph_gaussian_specialized_h_kernel<7><<<grid, block, h_shared_bytes, stream_>>>(current->data(), interm, width, height); break;
                case 9: graph_gaussian_specialized_h_kernel<9><<<grid, block, h_shared_bytes, stream_>>>(current->data(), interm, width, height); break;
                default: throw std::invalid_argument("CudaGraphEnhancedPipeline: invalid gaussian_kernel_size " + std::to_string(K));
            }
            CUDA_CHECK_LAST_ERROR();
            switch (K) {
                case 3: graph_gaussian_specialized_v_kernel<3><<<grid, block, v_shared_bytes, stream_>>>(interm, other->data(), width, height); break;
                case 5: graph_gaussian_specialized_v_kernel<5><<<grid, block, v_shared_bytes, stream_>>>(interm, other->data(), width, height); break;
                case 7: graph_gaussian_specialized_v_kernel<7><<<grid, block, v_shared_bytes, stream_>>>(interm, other->data(), width, height); break;
                case 9: graph_gaussian_specialized_v_kernel<9><<<grid, block, v_shared_bytes, stream_>>>(interm, other->data(), width, height); break;
            }
            CUDA_CHECK_LAST_ERROR();
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            std::swap(current, other);
        }
        if (config.median_enabled) {
            dim3 grid2d = compute_launch_grid(width, height, block);
            dim3 grid(grid2d.x, grid2d.y, static_cast<unsigned int>(batch_size));
            size_t shared_bytes = static_cast<size_t>(block.x + 2) * (block.y + 2) * sizeof(uint8_t);
            graph_median_network9_kernel<<<grid, block, shared_bytes, stream_>>>(current->data(), other->data(), width, height);
            CUDA_CHECK_LAST_ERROR();
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            std::swap(current, other);
        }
        if (config.sobel_enabled) {
            dim3 grid2d = compute_launch_grid(width, height, block);
            dim3 grid(grid2d.x, grid2d.y, static_cast<unsigned int>(batch_size));
            size_t shared_bytes = static_cast<size_t>(block.x + 2) * (block.y + 2) * sizeof(uint8_t);
            switch (config.sobel_mode) {
                case 0: graph_sobel_specialized_kernel<0><<<grid, block, shared_bytes, stream_>>>(current->data(), other->data(), width, height); break;
                case 1: graph_sobel_specialized_kernel<1><<<grid, block, shared_bytes, stream_>>>(current->data(), other->data(), width, height); break;
                case 2: graph_sobel_specialized_kernel<2><<<grid, block, shared_bytes, stream_>>>(current->data(), other->data(), width, height); break;
                case 3: graph_sobel_specialized_kernel<3><<<grid, block, shared_bytes, stream_>>>(current->data(), other->data(), width, height); break;
                default: throw std::invalid_argument("CudaGraphEnhancedPipeline: invalid sobel_mode " + std::to_string(config.sobel_mode));
            }
            CUDA_CHECK_LAST_ERROR();
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            std::swap(current, other);
        }
        if (config.laplacian_enabled) {
            int K = config.laplacian_kernel_size;
            int radius = K / 2;
            dim3 grid2d = compute_launch_grid(width, height, block);
            dim3 grid(grid2d.x, grid2d.y, static_cast<unsigned int>(batch_size));
            size_t shared_bytes = static_cast<size_t>(block.x + 2 * radius) * (block.y + 2 * radius) * sizeof(uint8_t);
            CUDA_CHECK(cudaMemcpyToSymbolAsync(g_laplacian_coeffs, config.laplacian_coeffs, static_cast<size_t>(K) * K * sizeof(float), 0, cudaMemcpyHostToDevice, stream_));
            switch (K) {
                case 3: graph_laplacian_specialized_kernel<3><<<grid, block, shared_bytes, stream_>>>(current->data(), other->data(), width, height, config.laplacian_scale, config.laplacian_delta); break;
                case 5: graph_laplacian_specialized_kernel<5><<<grid, block, shared_bytes, stream_>>>(current->data(), other->data(), width, height, config.laplacian_scale, config.laplacian_delta); break;
                default: throw std::invalid_argument("CudaGraphEnhancedPipeline: invalid laplacian_kernel_size " + std::to_string(K));
            }
            CUDA_CHECK_LAST_ERROR();
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            std::swap(current, other);
        }
        if (config.threshold_enabled) {
            launch_threshold(current->data(), other->data(), width, height, batch_size,
                              config.threshold_value, config.threshold_max_value, block, stream_);
            CUDA_CHECK_LAST_ERROR();
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            std::swap(current, other);
        }
        compute_timer.stop(stream_);
        timing.compute_ms = compute_timer.elapsed_ms();

        CudaTimer d2h_timer;
        d2h_timer.start(stream_);
        CUDA_CHECK(cudaMemcpyAsync(host_output.data(), current->data(), total_bytes, cudaMemcpyDeviceToHost, stream_));
        d2h_timer.stop(stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        timing.d2h_ms = d2h_timer.elapsed_ms();

        timing.used_graph = false;
        return {std::move(host_output), timing};
    }

    CacheKey key{};
    key.batch_size = batch_size;
    key.height = height;
    key.width = width;
    key.scope = static_cast<int>(scope);
    key.gaussian_enabled = config.gaussian_enabled;
    key.gaussian_kernel_size = config.gaussian_enabled ? config.gaussian_kernel_size : 0;
    key.median_enabled = config.median_enabled;
    key.sobel_enabled = config.sobel_enabled;
    key.sobel_mode = config.sobel_enabled ? config.sobel_mode : 0;
    key.laplacian_enabled = config.laplacian_enabled;
    key.laplacian_kernel_size = config.laplacian_enabled ? config.laplacian_kernel_size : 0;
    key.laplacian_scale_bits = config.laplacian_enabled ? float_bits(config.laplacian_scale) : 0;
    key.laplacian_delta_bits = config.laplacian_enabled ? float_bits(config.laplacian_delta) : 0;
    key.threshold_enabled = config.threshold_enabled;
    key.threshold_value = config.threshold_enabled ? config.threshold_value : 0;
    key.threshold_max_value = config.threshold_enabled ? config.threshold_max_value : 0;

    auto it = cache_.find(key);
    GraphEntry* entry = nullptr;
    if (it != cache_.end()) {
        entry = it->second.get();
        timing.graph_cache_hit = true;
    } else {
        host_batch_for_capture_ = host_batch;
        float capture_ms = 0.0f, instantiate_ms = 0.0f;
        std::string fallback_reason;
        entry = build_entry(key, config, batch_size, height, width, scope, capture_ms, instantiate_ms, fallback_reason);
        if (entry == nullptr) {
            auto result = run(host_batch, batch_size, height, width, config, /*use_graph=*/false, scope);
            result.second.used_graph = false;
            result.second.fallback_reason = fallback_reason;
            return result;
        }
        timing.graph_cache_hit = false;
        timing.capture_ms = capture_ms;
        timing.instantiate_ms = instantiate_ms;
    }

    HostTimer update_timer;
    update_timer.start();
    if (scope == EnhancedGraphScope::FullPipeline) {
        CUDA_CHECK(cudaGraphExecMemcpyNodeSetParams1D(
            entry->exec, entry->h2d_node, entry->buf_a->data(), host_batch, entry->total_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaGraphExecMemcpyNodeSetParams1D(
            entry->exec, entry->d2h_node, host_output.data(),
            entry->final_is_a ? entry->buf_a->data() : entry->buf_b->data(),
            entry->total_bytes, cudaMemcpyDeviceToHost));
    }
    if (entry->gaussian_coeff_node) {
        CUDA_CHECK(cudaGraphExecMemcpyNodeSetParams1D(
            entry->exec, entry->gaussian_coeff_node, entry->gaussian_symbol_addr,
            config.gaussian_coeffs_1d, static_cast<size_t>(config.gaussian_kernel_size) * sizeof(float),
            cudaMemcpyHostToDevice));
    }
    if (entry->laplacian_coeff_node) {
        CUDA_CHECK(cudaGraphExecMemcpyNodeSetParams1D(
            entry->exec, entry->laplacian_coeff_node, entry->laplacian_symbol_addr,
            config.laplacian_coeffs,
            static_cast<size_t>(config.laplacian_kernel_size) * config.laplacian_kernel_size * sizeof(float),
            cudaMemcpyHostToDevice));
    }
    update_timer.stop();
    timing.node_update_ms = update_timer.elapsed_ms();

    if (scope == EnhancedGraphScope::KernelsOnly) {
        CudaTimer h2d_timer;
        h2d_timer.start(stream_);
        CUDA_CHECK(cudaMemcpyAsync(entry->buf_a->data(), host_batch, entry->total_bytes, cudaMemcpyHostToDevice, stream_));
        h2d_timer.stop(stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        timing.h2d_ms = h2d_timer.elapsed_ms();
    }

    CudaTimer launch_timer;
    launch_timer.start(stream_);
    CUDA_CHECK(cudaGraphLaunch(entry->exec, stream_));
    launch_timer.stop(stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    timing.compute_ms = launch_timer.elapsed_ms();

    if (scope == EnhancedGraphScope::KernelsOnly) {
        CudaTimer d2h_timer;
        d2h_timer.start(stream_);
        GpuImageBatch& final_buf = entry->final_is_a ? *entry->buf_a : *entry->buf_b;
        CUDA_CHECK(cudaMemcpyAsync(host_output.data(), final_buf.data(), entry->total_bytes, cudaMemcpyDeviceToHost, stream_));
        d2h_timer.stop(stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        timing.d2h_ms = d2h_timer.elapsed_ms();
    }

    timing.used_graph = true;
    return {std::move(host_output), timing};
}

}  // namespace xray_cuda
