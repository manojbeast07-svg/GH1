#include "median_basic.cuh"
#include <stdexcept>
#include <string>

namespace xray_cuda {

namespace {

constexpr int kMedianBasicMaxCount = kMedianBasicMaxKernelSize * kMedianBasicMaxKernelSize;

}  // namespace

// clamp_index() (BORDER_REPLICATE coordinate mapping) now lives in
// gpu_image.cuh, shared with median_enhanced.cu (Section 7) -- previously
// duplicated here.

__global__ void median_basic_kernel(
    const uint8_t* input, uint8_t* output,
    int width, int height, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    size_t plane = static_cast<size_t>(height) * static_cast<size_t>(width);
    const uint8_t* in = input + static_cast<size_t>(blockIdx.z) * plane;
    uint8_t* out = output + static_cast<size_t>(blockIdx.z) * plane;

    int radius = kernel_size / 2;
    uint8_t values[kMedianBasicMaxCount];
    int count = 0;

    for (int dy = -radius; dy <= radius; ++dy) {
        int sy = clamp_index(y + dy, height);
        const uint8_t* row = in + sy * width;
        for (int dx = -radius; dx <= radius; ++dx) {
            int sx = clamp_index(x + dx, width);
            values[count++] = row[sx];
        }
    }

    // Plain insertion sort -- intentionally not a sorting network; this
    // is the naive Basic baseline (Section 4C spec). count is always
    // odd (kernel_size is validated odd, so kernel_size^2 is odd), so
    // there's always a unique middle element, no tie-breaking needed.
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

std::pair<std::unique_ptr<GpuImage>, MedianTiming> median_basic(const GpuImage& input, int kernel_size) {
    if (kernel_size <= 0 || kernel_size % 2 == 0) {
        throw std::invalid_argument(
            "median_basic: kernel_size must be a positive odd integer, got " + std::to_string(kernel_size));
    }
    if (kernel_size > kMedianBasicMaxKernelSize) {
        throw std::invalid_argument(
            "median_basic: kernel_size " + std::to_string(kernel_size) +
            " exceeds the maximum supported size " + std::to_string(kMedianBasicMaxKernelSize));
    }

    auto output = std::make_unique<GpuImage>(input.height(), input.width());

    dim3 block = default_block_dim();
    dim3 grid = compute_launch_grid(input.width(), input.height(), block);

    CudaTimer timer;
    timer.start();
    median_basic_kernel<<<grid, block>>>(input.data(), output->data(), input.width(), input.height(), kernel_size);
    CUDA_CHECK_LAST_ERROR();
    timer.stop();

    CUDA_CHECK(cudaDeviceSynchronize());
    float kernel_ms = timer.elapsed_ms();

    return {std::move(output), MedianTiming{kernel_ms}};
}

}  // namespace xray_cuda
