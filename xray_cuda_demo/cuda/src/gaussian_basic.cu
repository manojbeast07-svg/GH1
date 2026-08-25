#include "gaussian_basic.cuh"
#include <stdexcept>
#include <string>

namespace xray_cuda {

// reflect101() (BORDER_REFLECT_101 / BORDER_DEFAULT coordinate mapping)
// now lives in gpu_image.cuh, shared with sobel_basic.cu and
// laplacian_basic.cu -- previously duplicated here.

__global__ void gaussian_basic_kernel(
    const uint8_t* input, uint8_t* output,
    int width, int height,
    const float* coeffs, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    // blockIdx.z selects the image within a batch (grid.z == 1, the dim3
    // default, for every pre-Section-5 single-image call site -- this
    // extension is backward compatible without touching those callers).
    size_t plane = static_cast<size_t>(height) * static_cast<size_t>(width);
    const uint8_t* in = input + static_cast<size_t>(blockIdx.z) * plane;
    uint8_t* out = output + static_cast<size_t>(blockIdx.z) * plane;

    int radius = kernel_size / 2;
    float sum = 0.0f;
    for (int ky = -radius; ky <= radius; ++ky) {
        int sy = reflect101(y + ky, height);
        const uint8_t* row = in + sy * width;
        const float* coeff_row = coeffs + (ky + radius) * kernel_size;
        for (int kx = -radius; kx <= radius; ++kx) {
            int sx = reflect101(x + kx, width);
            sum += coeff_row[kx + radius] * static_cast<float>(row[sx]);
        }
    }

    int rounded = __float2int_rn(sum);
    rounded = max(0, min(255, rounded));
    out[y * width + x] = static_cast<uint8_t>(rounded);
}

std::pair<std::unique_ptr<GpuImage>, GaussianTiming> gaussian_basic(
    const GpuImage& input, const float* host_coeffs, int kernel_size) {
    if (kernel_size <= 0 || kernel_size % 2 == 0) {
        throw std::invalid_argument(
            "gaussian_basic: kernel_size must be a positive odd integer, got " + std::to_string(kernel_size));
    }

    auto output = std::make_unique<GpuImage>(input.height(), input.width());

    size_t coeff_count = static_cast<size_t>(kernel_size) * static_cast<size_t>(kernel_size);
    DeviceBuffer<float> device_coeffs(coeff_count);

    CudaTimer upload_timer;
    upload_timer.start();
    device_coeffs.upload(host_coeffs, coeff_count);
    upload_timer.stop();
    float coeff_upload_ms = upload_timer.elapsed_ms();

    dim3 block = default_block_dim();
    dim3 grid = compute_launch_grid(input.width(), input.height(), block);

    CudaTimer kernel_timer;
    kernel_timer.start();
    gaussian_basic_kernel<<<grid, block>>>(
        input.data(), output->data(), input.width(), input.height(), device_coeffs.get(), kernel_size);
    CUDA_CHECK_LAST_ERROR();
    kernel_timer.stop();

    CUDA_CHECK(cudaDeviceSynchronize());
    float kernel_ms = kernel_timer.elapsed_ms();

    return {std::move(output), GaussianTiming{coeff_upload_ms, kernel_ms}};
}

}  // namespace xray_cuda
