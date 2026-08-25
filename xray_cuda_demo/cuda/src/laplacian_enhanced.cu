#include "laplacian_enhanced.cuh"
#include <stdexcept>
#include <string>

namespace xray_cuda {

// Constant-memory coefficients for SharedConst/Specialized. Flat,
// row-major, sized for the largest supported matrix (5x5); a smaller
// kernel_size (3) simply uses the first 9 entries with a 3-wide stride.
__constant__ float d_laplacian_coeffs[kLaplacianEnhancedMaxKernelSize * kLaplacianEnhancedMaxKernelSize];

namespace {

// Which of the three known, VERIFIED coefficient sets (see
// laplacian_enhanced.cuh's module comment -- all three recovered via
// cuda/laplacian.py::laplacian_kernel_2d's impulse-response method, not
// re-derived here) `host_coeffs` matches. Every value in all three sets
// is a small exact integer, so exact float equality is the correct
// (not merely convenient) comparison -- no epsilon needed.
enum class KnownLaplacianPattern { Classic3x3, CvKsize3, CvKsize5, Unknown };

KnownLaplacianPattern classify_laplacian_coeffs(const float* coeffs, int kernel_size) {
    if (kernel_size == 3) {
        static constexpr float kClassic[9] = {0, 1, 0, 1, -4, 1, 0, 1, 0};
        static constexpr float kCv3[9] = {2, 0, 2, 0, -8, 0, 2, 0, 2};
        bool is_classic = true, is_cv3 = true;
        for (int i = 0; i < 9; ++i) {
            if (coeffs[i] != kClassic[i]) is_classic = false;
            if (coeffs[i] != kCv3[i]) is_cv3 = false;
        }
        if (is_classic) return KnownLaplacianPattern::Classic3x3;
        if (is_cv3) return KnownLaplacianPattern::CvKsize3;
        return KnownLaplacianPattern::Unknown;
    }
    if (kernel_size == 5) {
        static constexpr float kCv5[25] = {
            2, 4, 4, 4, 2,
            4, 0, -8, 0, 4,
            4, -8, -24, -8, 4,
            4, 0, -8, 0, 4,
            2, 4, 4, 4, 2,
        };
        for (int i = 0; i < 25; ++i) {
            if (coeffs[i] != kCv5[i]) return KnownLaplacianPattern::Unknown;
        }
        return KnownLaplacianPattern::CvKsize5;
    }
    return KnownLaplacianPattern::Unknown;
}

__device__ __forceinline__ uint8_t laplacian_finalize(float sum, float scale, float delta) {
    float scaled = sum * scale + delta;
    int rounded = __float2int_rn(fabsf(scaled));
    rounded = max(0, min(255, rounded));
    return static_cast<uint8_t>(rounded);
}

// Loads a (blockDim.x+2*radius) x (blockDim.y+2*radius)
// BORDER_REFLECT_101 halo tile -- same border rule as Basic
// (reflect101() in gpu_image.cuh). Shared by every kernel below.
__device__ __forceinline__ void load_laplacian_tile(
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

// -- Optimization 1: shared-memory tiling, generic runtime loop, global-memory coefficients -----------------------

__global__ void laplacian_shared_kernel(
    const uint8_t* input, uint8_t* output, int width, int height,
    const float* coeffs, int kernel_size, float scale, float delta) {
    extern __shared__ uint8_t s_tile_lap[];
    int radius = kernel_size / 2;
    int tile_w = static_cast<int>(blockDim.x) + 2 * radius;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t plane = static_cast<size_t>(height) * width;
    const uint8_t* in = input + static_cast<size_t>(blockIdx.z) * plane;
    uint8_t* out = output + static_cast<size_t>(blockIdx.z) * plane;

    load_laplacian_tile(s_tile_lap, in, width, height, radius, tile_w);
    __syncthreads();

    if (x >= width || y >= height) return;

    int lx = threadIdx.x, ly = threadIdx.y;
    float sum = 0.0f;
    for (int ky = -radius; ky <= radius; ++ky) {
        const float* coeff_row = coeffs + (ky + radius) * kernel_size;
        for (int kx = -radius; kx <= radius; ++kx) {
            float v = static_cast<float>(s_tile_lap[(ly + radius + ky) * tile_w + (lx + radius + kx)]);
            sum += coeff_row[kx + radius] * v;
        }
    }
    out[y * width + x] = laplacian_finalize(sum, scale, delta);
}

// -- Optimization 2: + constant-memory coefficients, still a runtime-sized loop -----------------------

__global__ void laplacian_shared_const_kernel(
    const uint8_t* input, uint8_t* output, int width, int height,
    int kernel_size, float scale, float delta) {
    extern __shared__ uint8_t s_tile_lap_const[];
    int radius = kernel_size / 2;
    int tile_w = static_cast<int>(blockDim.x) + 2 * radius;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t plane = static_cast<size_t>(height) * width;
    const uint8_t* in = input + static_cast<size_t>(blockIdx.z) * plane;
    uint8_t* out = output + static_cast<size_t>(blockIdx.z) * plane;

    load_laplacian_tile(s_tile_lap_const, in, width, height, radius, tile_w);
    __syncthreads();

    if (x >= width || y >= height) return;

    int lx = threadIdx.x, ly = threadIdx.y;
    float sum = 0.0f;
    for (int ky = -radius; ky <= radius; ++ky) {
        const float* coeff_row = d_laplacian_coeffs + (ky + radius) * kernel_size;
        for (int kx = -radius; kx <= radius; ++kx) {
            float v = static_cast<float>(s_tile_lap_const[(ly + radius + ky) * tile_w + (lx + radius + kx)]);
            sum += coeff_row[kx + radius] * v;
        }
    }
    out[y * width + x] = laplacian_finalize(sum, scale, delta);
}

// -- Optimization 3: + compile-time matrix-size specialization, unrolled -----------------------

template <int K>
__global__ void laplacian_specialized_kernel(
    const uint8_t* input, uint8_t* output, int width, int height, float scale, float delta) {
    constexpr int radius = K / 2;
    extern __shared__ uint8_t s_tile_lap_spec[];
    int tile_w = static_cast<int>(blockDim.x) + 2 * radius;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t plane = static_cast<size_t>(height) * width;
    const uint8_t* in = input + static_cast<size_t>(blockIdx.z) * plane;
    uint8_t* out = output + static_cast<size_t>(blockIdx.z) * plane;

    load_laplacian_tile(s_tile_lap_spec, in, width, height, radius, tile_w);
    __syncthreads();

    if (x >= width || y >= height) return;

    int lx = threadIdx.x, ly = threadIdx.y;
    float sum = 0.0f;
#pragma unroll
    for (int ky = -radius; ky <= radius; ++ky) {
#pragma unroll
        for (int kx = -radius; kx <= radius; ++kx) {
            float v = static_cast<float>(s_tile_lap_spec[(ly + radius + ky) * tile_w + (lx + radius + kx)]);
            sum += d_laplacian_coeffs[(ky + radius) * K + (kx + radius)] * v;
        }
    }
    out[y * width + x] = laplacian_finalize(sum, scale, delta);
}

// -- Optimization 4: explicit hand-written arithmetic for each known coefficient set -----------------------
//
// Skips every zero-coefficient term entirely (Classic3x3/CvKsize3: 4 of
// 9 terms are zero; CvKsize5: 4 of 25) and groups equal-coefficient
// terms into one multiply, unlike Specialized which still reads/
// multiplies every position (including zeros) from constant memory.

__global__ void laplacian_explicit_classic3x3_kernel(
    const uint8_t* input, uint8_t* output, int width, int height, float scale, float delta) {
    constexpr int radius = 1;
    extern __shared__ uint8_t s_tile_exp_c3[];
    int tile_w = static_cast<int>(blockDim.x) + 2 * radius;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t plane = static_cast<size_t>(height) * width;
    const uint8_t* in = input + static_cast<size_t>(blockIdx.z) * plane;
    uint8_t* out = output + static_cast<size_t>(blockIdx.z) * plane;

    load_laplacian_tile(s_tile_exp_c3, in, width, height, radius, tile_w);
    __syncthreads();

    if (x >= width || y >= height) return;

    int lx = threadIdx.x, ly = threadIdx.y;
    auto tp = [&](int dy, int dx) -> float {
        return static_cast<float>(s_tile_exp_c3[(ly + radius + dy) * tile_w + (lx + radius + dx)]);
    };
    // [[0,1,0],[1,-4,1],[0,1,0]]
    float sum = (tp(-1, 0) + tp(1, 0) + tp(0, -1) + tp(0, 1)) - 4.0f * tp(0, 0);
    out[y * width + x] = laplacian_finalize(sum, scale, delta);
}

__global__ void laplacian_explicit_cvksize3_kernel(
    const uint8_t* input, uint8_t* output, int width, int height, float scale, float delta) {
    constexpr int radius = 1;
    extern __shared__ uint8_t s_tile_exp_cv3[];
    int tile_w = static_cast<int>(blockDim.x) + 2 * radius;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t plane = static_cast<size_t>(height) * width;
    const uint8_t* in = input + static_cast<size_t>(blockIdx.z) * plane;
    uint8_t* out = output + static_cast<size_t>(blockIdx.z) * plane;

    load_laplacian_tile(s_tile_exp_cv3, in, width, height, radius, tile_w);
    __syncthreads();

    if (x >= width || y >= height) return;

    int lx = threadIdx.x, ly = threadIdx.y;
    auto tp = [&](int dy, int dx) -> float {
        return static_cast<float>(s_tile_exp_cv3[(ly + radius + dy) * tile_w + (lx + radius + dx)]);
    };
    // [[2,0,2],[0,-8,0],[2,0,2]]
    float corners = tp(-1, -1) + tp(-1, 1) + tp(1, -1) + tp(1, 1);
    float sum = 2.0f * corners - 8.0f * tp(0, 0);
    out[y * width + x] = laplacian_finalize(sum, scale, delta);
}

__global__ void laplacian_explicit_cvksize5_kernel(
    const uint8_t* input, uint8_t* output, int width, int height, float scale, float delta) {
    constexpr int radius = 2;
    extern __shared__ uint8_t s_tile_exp_cv5[];
    int tile_w = static_cast<int>(blockDim.x) + 2 * radius;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t plane = static_cast<size_t>(height) * width;
    const uint8_t* in = input + static_cast<size_t>(blockIdx.z) * plane;
    uint8_t* out = output + static_cast<size_t>(blockIdx.z) * plane;

    load_laplacian_tile(s_tile_exp_cv5, in, width, height, radius, tile_w);
    __syncthreads();

    if (x >= width || y >= height) return;

    int lx = threadIdx.x, ly = threadIdx.y;
    auto tp = [&](int dy, int dx) -> float {
        return static_cast<float>(s_tile_exp_cv5[(ly + radius + dy) * tile_w + (lx + radius + dx)]);
    };
    // [[2,4,4,4,2],[4,0,-8,0,4],[4,-8,-24,-8,4],[4,0,-8,0,4],[2,4,4,4,2]]
    // (-1,-1)/(-1,1)/(1,-1)/(1,1) are the 4 zero-coefficient diagonal
    // neighbors of the center -- skipped entirely, not read.
    float corners = tp(-2, -2) + tp(-2, 2) + tp(2, -2) + tp(2, 2);              // coefficient 2
    float twelves = tp(-2, -1) + tp(-2, 0) + tp(-2, 1)                          // coefficient 4
                   + tp(2, -1) + tp(2, 0) + tp(2, 1)
                   + tp(-1, -2) + tp(0, -2) + tp(1, -2)
                   + tp(-1, 2) + tp(0, 2) + tp(1, 2);
    float neighbors4 = tp(-1, 0) + tp(0, -1) + tp(0, 1) + tp(1, 0);             // coefficient -8
    float sum = 2.0f * corners + 4.0f * twelves - 8.0f * neighbors4 - 24.0f * tp(0, 0);
    out[y * width + x] = laplacian_finalize(sum, scale, delta);
}

}  // namespace

// -- shared dispatch: the ONE place the variant switch statement exists -----------------------

void laplacian_enhanced_dispatch(
    const uint8_t* d_input, uint8_t* d_output,
    int batch_size, int height, int width,
    const float* host_coeffs, int kernel_size, float scale, float delta,
    LaplacianVariant variant, int block_x, int block_y,
    LaplacianEnhancedTiming& timing) {
    if (kernel_size <= 0 || kernel_size % 2 == 0) {
        throw std::invalid_argument(
            "laplacian_enhanced_dispatch: kernel_size must be a positive odd integer, got " +
            std::to_string(kernel_size));
    }
    if (kernel_size > kLaplacianEnhancedMaxKernelSize) {
        throw std::invalid_argument(
            "laplacian_enhanced_dispatch: kernel_size " + std::to_string(kernel_size) +
            " exceeds the maximum supported size " + std::to_string(kLaplacianEnhancedMaxKernelSize));
    }
    if (variant == LaplacianVariant::Specialized && kernel_size != 3 && kernel_size != 5) {
        throw std::invalid_argument(
            "laplacian_enhanced_dispatch: Specialized variant only supports kernel_size in {3,5}, got " +
            std::to_string(kernel_size));
    }
    if (block_x <= 0 || block_y <= 0) {
        throw std::invalid_argument("laplacian_enhanced_dispatch: block_x/block_y must be positive.");
    }

    KnownLaplacianPattern pattern = KnownLaplacianPattern::Unknown;
    if (variant == LaplacianVariant::Explicit) {
        pattern = classify_laplacian_coeffs(host_coeffs, kernel_size);
        if (pattern == KnownLaplacianPattern::Unknown) {
            throw std::invalid_argument(
                "laplacian_enhanced_dispatch: Explicit variant only supports the three known, verified "
                "Laplacian coefficient sets (kernel_size=3 classic/cv2, kernel_size=5 cv2); got an "
                "unrecognized coefficient array for kernel_size=" + std::to_string(kernel_size));
        }
    }

    int radius = kernel_size / 2;
    dim3 block(static_cast<unsigned int>(block_x), static_cast<unsigned int>(block_y));
    dim3 grid2d = compute_launch_grid(width, height, block);
    dim3 grid(grid2d.x, grid2d.y, static_cast<unsigned int>(batch_size));
    size_t shared_bytes = static_cast<size_t>(block_x + 2 * radius) * (block_y + 2 * radius) * sizeof(uint8_t);

    DeviceBuffer<float> device_coeffs;
    CudaTimer coeff_timer;
    coeff_timer.start();
    if (variant == LaplacianVariant::Shared) {
        device_coeffs = DeviceBuffer<float>(static_cast<size_t>(kernel_size) * kernel_size);
        device_coeffs.upload(host_coeffs, static_cast<size_t>(kernel_size) * kernel_size);
    } else if (variant == LaplacianVariant::SharedConst || variant == LaplacianVariant::Specialized) {
        CUDA_CHECK(cudaMemcpyToSymbol(d_laplacian_coeffs, host_coeffs, kernel_size * kernel_size * sizeof(float)));
    }
    coeff_timer.stop();
    timing.coeff_upload_ms = coeff_timer.elapsed_ms();

    CudaTimer timer;
    timer.start();
    switch (variant) {
        case LaplacianVariant::Shared:
            laplacian_shared_kernel<<<grid, block, shared_bytes>>>(
                d_input, d_output, width, height, device_coeffs.get(), kernel_size, scale, delta);
            break;
        case LaplacianVariant::SharedConst:
            laplacian_shared_const_kernel<<<grid, block, shared_bytes>>>(
                d_input, d_output, width, height, kernel_size, scale, delta);
            break;
        case LaplacianVariant::Specialized:
            switch (kernel_size) {
                case 3:
                    laplacian_specialized_kernel<3><<<grid, block, shared_bytes>>>(
                        d_input, d_output, width, height, scale, delta);
                    break;
                case 5:
                    laplacian_specialized_kernel<5><<<grid, block, shared_bytes>>>(
                        d_input, d_output, width, height, scale, delta);
                    break;
            }
            break;
        case LaplacianVariant::Explicit:
            switch (pattern) {
                case KnownLaplacianPattern::Classic3x3:
                    laplacian_explicit_classic3x3_kernel<<<grid, block, shared_bytes>>>(
                        d_input, d_output, width, height, scale, delta);
                    break;
                case KnownLaplacianPattern::CvKsize3:
                    laplacian_explicit_cvksize3_kernel<<<grid, block, shared_bytes>>>(
                        d_input, d_output, width, height, scale, delta);
                    break;
                case KnownLaplacianPattern::CvKsize5:
                    laplacian_explicit_cvksize5_kernel<<<grid, block, shared_bytes>>>(
                        d_input, d_output, width, height, scale, delta);
                    break;
                case KnownLaplacianPattern::Unknown:
                    break;  // unreachable -- validated above
            }
            break;
    }
    CUDA_CHECK_LAST_ERROR();
    timer.stop();
    CUDA_CHECK(cudaDeviceSynchronize());
    timing.kernel_ms = timer.elapsed_ms();
}

// -- thin single-image / batch wrappers -----------------------

std::pair<std::unique_ptr<GpuImage>, LaplacianEnhancedTiming> laplacian_enhanced(
    const GpuImage& input, const float* host_coeffs, int kernel_size, float scale, float delta,
    LaplacianVariant variant, int block_x, int block_y) {
    auto output = std::make_unique<GpuImage>(input.height(), input.width());
    LaplacianEnhancedTiming timing{};
    laplacian_enhanced_dispatch(
        input.data(), output->data(), /*batch_size=*/1, input.height(), input.width(),
        host_coeffs, kernel_size, scale, delta, variant, block_x, block_y, timing);
    return {std::move(output), timing};
}

std::pair<std::unique_ptr<GpuImageBatch>, LaplacianEnhancedTiming> laplacian_enhanced_batch(
    const GpuImageBatch& input, const float* host_coeffs, int kernel_size, float scale, float delta,
    LaplacianVariant variant, int block_x, int block_y) {
    auto output = std::make_unique<GpuImageBatch>(input.batch_size(), input.height(), input.width());
    LaplacianEnhancedTiming timing{};
    laplacian_enhanced_dispatch(
        input.data(), output->data(), input.batch_size(), input.height(), input.width(),
        host_coeffs, kernel_size, scale, delta, variant, block_x, block_y, timing);
    return {std::move(output), timing};
}

}  // namespace xray_cuda
