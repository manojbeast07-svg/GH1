#include "gaussian_enhanced.cuh"
#include <stdexcept>
#include <string>

namespace xray_cuda {

// Coefficients for the SharedConst and Specialized variants. Populated
// via cudaMemcpyToSymbol before each launch that uses it (Naive/Shared
// read a plain global-memory pointer parameter instead -- that's the
// whole point of this experiment).
__constant__ float d_gaussian_coeffs_1d[kGaussianEnhancedMaxKernelSize];

namespace {

// -- Optimization 1: separable, global-memory coefficients, no shared memory -----------------------

__global__ void gaussian_naive_h_kernel(
    const uint8_t* input, float* output, int width, int height, const float* coeffs, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    size_t plane = static_cast<size_t>(height) * width;
    const uint8_t* in = input + static_cast<size_t>(blockIdx.z) * plane;
    float* out = output + static_cast<size_t>(blockIdx.z) * plane;

    int radius = kernel_size / 2;
    float sum = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
        int sx = reflect101(x + k, width);
        sum += coeffs[k + radius] * static_cast<float>(in[y * width + sx]);
    }
    out[y * width + x] = sum;
}

__global__ void gaussian_naive_v_kernel(
    const float* input, uint8_t* output, int width, int height, const float* coeffs, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    size_t plane = static_cast<size_t>(height) * width;
    const float* in = input + static_cast<size_t>(blockIdx.z) * plane;
    uint8_t* out = output + static_cast<size_t>(blockIdx.z) * plane;

    int radius = kernel_size / 2;
    float sum = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
        int sy = reflect101(y + k, height);
        sum += coeffs[k + radius] * in[sy * width + x];
    }
    int rounded = __float2int_rn(sum);
    rounded = max(0, min(255, rounded));
    out[y * width + x] = static_cast<uint8_t>(rounded);
}

// -- Optimization 2: + shared-memory tiling (halo in one dimension only per pass) -----------------------

__global__ void gaussian_shared_h_kernel(
    const uint8_t* input, float* output, int width, int height, const float* coeffs, int kernel_size) {
    extern __shared__ uint8_t s_tile_u8[];
    int radius = kernel_size / 2;
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
            s_tile_u8[ty * tile_w + i] = in[y * width + sx];
        }
    }
    __syncthreads();

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
        sum += coeffs[k + radius] * static_cast<float>(s_tile_u8[ty * tile_w + (tx + radius + k)]);
    }
    out[y * width + x] = sum;
}

__global__ void gaussian_shared_v_kernel(
    const float* input, uint8_t* output, int width, int height, const float* coeffs, int kernel_size) {
    extern __shared__ float s_tile_f32[];
    int radius = kernel_size / 2;

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
            s_tile_f32[i * blockDim.x + tx] = in[sy * width + x];
        }
    }
    __syncthreads();

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
        sum += coeffs[k + radius] * s_tile_f32[(ty + radius + k) * blockDim.x + tx];
    }
    int rounded = __float2int_rn(sum);
    rounded = max(0, min(255, rounded));
    out[y * width + x] = static_cast<uint8_t>(rounded);
}

// -- Optimization 3: + constant-memory coefficients (same tiling as above) -----------------------

__global__ void gaussian_shared_const_h_kernel(
    const uint8_t* input, float* output, int width, int height, int kernel_size) {
    extern __shared__ uint8_t s_tile_u8[];
    int radius = kernel_size / 2;
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
            s_tile_u8[ty * tile_w + i] = in[y * width + sx];
        }
    }
    __syncthreads();

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
        sum += d_gaussian_coeffs_1d[k + radius] * static_cast<float>(s_tile_u8[ty * tile_w + (tx + radius + k)]);
    }
    out[y * width + x] = sum;
}

__global__ void gaussian_shared_const_v_kernel(
    const float* input, uint8_t* output, int width, int height, int kernel_size) {
    extern __shared__ float s_tile_f32[];
    int radius = kernel_size / 2;

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
            s_tile_f32[i * blockDim.x + tx] = in[sy * width + x];
        }
    }
    __syncthreads();

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
        sum += d_gaussian_coeffs_1d[k + radius] * s_tile_f32[(ty + radius + k) * blockDim.x + tx];
    }
    int rounded = __float2int_rn(sum);
    rounded = max(0, min(255, rounded));
    out[y * width + x] = static_cast<uint8_t>(rounded);
}

// -- Optimization 4: + compile-time kernel_size specialization, unrolled -----------------------

template <int K>
__global__ void gaussian_specialized_h_kernel(const uint8_t* input, float* output, int width, int height) {
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
        sum += d_gaussian_coeffs_1d[k + radius] * static_cast<float>(s_tile_u8_spec[ty * tile_w + (tx + radius + k)]);
    }
    out[y * width + x] = sum;
}

template <int K>
__global__ void gaussian_specialized_v_kernel(const float* input, uint8_t* output, int width, int height) {
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
        sum += d_gaussian_coeffs_1d[k + radius] * s_tile_f32_spec[(ty + radius + k) * blockDim.x + tx];
    }
    int rounded = __float2int_rn(sum);
    rounded = max(0, min(255, rounded));
    out[y * width + x] = static_cast<uint8_t>(rounded);
}

}  // namespace

// -- shared dispatch: the ONE place the variant switch statement exists -----------------------

void gaussian_enhanced_dispatch(
    const uint8_t* d_input, float* d_intermediate, uint8_t* d_output,
    int batch_size, int height, int width,
    const float* host_coeffs_1d, int kernel_size,
    GaussianVariant variant, int block_x, int block_y,
    GaussianEnhancedTiming& timing) {
    if (kernel_size <= 0 || kernel_size % 2 == 0) {
        throw std::invalid_argument(
            "gaussian_enhanced_dispatch: kernel_size must be a positive odd integer, got " +
            std::to_string(kernel_size));
    }
    if (kernel_size > kGaussianEnhancedMaxKernelSize) {
        throw std::invalid_argument(
            "gaussian_enhanced_dispatch: kernel_size " + std::to_string(kernel_size) +
            " exceeds the maximum supported size " + std::to_string(kGaussianEnhancedMaxKernelSize));
    }
    if (variant == GaussianVariant::Specialized && kernel_size != 3 && kernel_size != 5 &&
        kernel_size != 7 && kernel_size != 9) {
        throw std::invalid_argument(
            "gaussian_enhanced_dispatch: Specialized variant only supports kernel_size in {3,5,7,9}, got " +
            std::to_string(kernel_size));
    }
    if (block_x <= 0 || block_y <= 0) {
        throw std::invalid_argument("gaussian_enhanced_dispatch: block_x/block_y must be positive.");
    }

    int radius = kernel_size / 2;
    dim3 block(static_cast<unsigned int>(block_x), static_cast<unsigned int>(block_y));
    dim3 grid2d = compute_launch_grid(width, height, block);
    dim3 grid(grid2d.x, grid2d.y, static_cast<unsigned int>(batch_size));

    DeviceBuffer<float> device_coeffs;
    CudaTimer coeff_timer;
    coeff_timer.start();
    if (variant == GaussianVariant::Naive || variant == GaussianVariant::Shared) {
        device_coeffs = DeviceBuffer<float>(static_cast<size_t>(kernel_size));
        device_coeffs.upload(host_coeffs_1d, static_cast<size_t>(kernel_size));
    } else {
        CUDA_CHECK(cudaMemcpyToSymbol(d_gaussian_coeffs_1d, host_coeffs_1d, kernel_size * sizeof(float)));
    }
    coeff_timer.stop();
    timing.coeff_upload_ms = coeff_timer.elapsed_ms();

    size_t h_shared_bytes = static_cast<size_t>(block_x + 2 * radius) * block_y * sizeof(uint8_t);
    size_t v_shared_bytes = static_cast<size_t>(block_x) * (block_y + 2 * radius) * sizeof(float);

    CudaTimer h_timer;
    h_timer.start();
    switch (variant) {
        case GaussianVariant::Naive:
            gaussian_naive_h_kernel<<<grid, block>>>(d_input, d_intermediate, width, height, device_coeffs.get(), kernel_size);
            break;
        case GaussianVariant::Shared:
            gaussian_shared_h_kernel<<<grid, block, h_shared_bytes>>>(d_input, d_intermediate, width, height, device_coeffs.get(), kernel_size);
            break;
        case GaussianVariant::SharedConst:
            gaussian_shared_const_h_kernel<<<grid, block, h_shared_bytes>>>(d_input, d_intermediate, width, height, kernel_size);
            break;
        case GaussianVariant::Specialized:
            switch (kernel_size) {
                case 3: gaussian_specialized_h_kernel<3><<<grid, block, h_shared_bytes>>>(d_input, d_intermediate, width, height); break;
                case 5: gaussian_specialized_h_kernel<5><<<grid, block, h_shared_bytes>>>(d_input, d_intermediate, width, height); break;
                case 7: gaussian_specialized_h_kernel<7><<<grid, block, h_shared_bytes>>>(d_input, d_intermediate, width, height); break;
                case 9: gaussian_specialized_h_kernel<9><<<grid, block, h_shared_bytes>>>(d_input, d_intermediate, width, height); break;
            }
            break;
    }
    CUDA_CHECK_LAST_ERROR();
    h_timer.stop();
    CUDA_CHECK(cudaDeviceSynchronize());
    timing.horizontal_ms = h_timer.elapsed_ms();

    CudaTimer v_timer;
    v_timer.start();
    switch (variant) {
        case GaussianVariant::Naive:
            gaussian_naive_v_kernel<<<grid, block>>>(d_intermediate, d_output, width, height, device_coeffs.get(), kernel_size);
            break;
        case GaussianVariant::Shared:
            gaussian_shared_v_kernel<<<grid, block, v_shared_bytes>>>(d_intermediate, d_output, width, height, device_coeffs.get(), kernel_size);
            break;
        case GaussianVariant::SharedConst:
            gaussian_shared_const_v_kernel<<<grid, block, v_shared_bytes>>>(d_intermediate, d_output, width, height, kernel_size);
            break;
        case GaussianVariant::Specialized:
            switch (kernel_size) {
                case 3: gaussian_specialized_v_kernel<3><<<grid, block, v_shared_bytes>>>(d_intermediate, d_output, width, height); break;
                case 5: gaussian_specialized_v_kernel<5><<<grid, block, v_shared_bytes>>>(d_intermediate, d_output, width, height); break;
                case 7: gaussian_specialized_v_kernel<7><<<grid, block, v_shared_bytes>>>(d_intermediate, d_output, width, height); break;
                case 9: gaussian_specialized_v_kernel<9><<<grid, block, v_shared_bytes>>>(d_intermediate, d_output, width, height); break;
            }
            break;
    }
    CUDA_CHECK_LAST_ERROR();
    v_timer.stop();
    CUDA_CHECK(cudaDeviceSynchronize());
    timing.vertical_ms = v_timer.elapsed_ms();
}

// -- thin single-image / batch wrappers: allocate, delegate to gaussian_enhanced_dispatch -----------------------

std::pair<std::unique_ptr<GpuImage>, GaussianEnhancedTiming> gaussian_enhanced(
    const GpuImage& input, const float* host_coeffs_1d, int kernel_size,
    GaussianVariant variant, int block_x, int block_y) {
    int height = input.height();
    int width = input.width();

    auto output = std::make_unique<GpuImage>(height, width);
    DeviceBuffer<float> intermediate(static_cast<size_t>(height) * static_cast<size_t>(width));

    GaussianEnhancedTiming timing{};
    gaussian_enhanced_dispatch(
        input.data(), intermediate.get(), output->data(), /*batch_size=*/1, height, width,
        host_coeffs_1d, kernel_size, variant, block_x, block_y, timing);

    return {std::move(output), timing};
}

std::pair<std::unique_ptr<GpuImageBatch>, GaussianEnhancedTiming> gaussian_enhanced_batch(
    const GpuImageBatch& input, const float* host_coeffs_1d, int kernel_size,
    GaussianVariant variant, int block_x, int block_y) {
    int batch_size = input.batch_size();
    int height = input.height();
    int width = input.width();

    auto output = std::make_unique<GpuImageBatch>(batch_size, height, width);
    DeviceBuffer<float> intermediate(
        static_cast<size_t>(batch_size) * static_cast<size_t>(height) * static_cast<size_t>(width));

    GaussianEnhancedTiming timing{};
    gaussian_enhanced_dispatch(
        input.data(), intermediate.get(), output->data(), batch_size, height, width,
        host_coeffs_1d, kernel_size, variant, block_x, block_y, timing);

    return {std::move(output), timing};
}

}  // namespace xray_cuda
