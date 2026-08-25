#include "threshold_basic.cuh"

namespace xray_cuda {

__global__ void threshold_basic_kernel(
    const uint8_t* input, uint8_t* output,
    int width, int height,
    uint8_t threshold_value, uint8_t max_value) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    size_t plane = static_cast<size_t>(height) * static_cast<size_t>(width);
    int idx = y * width + x;
    const uint8_t* in = input + static_cast<size_t>(blockIdx.z) * plane;
    uint8_t* out = output + static_cast<size_t>(blockIdx.z) * plane;
    out[idx] = (in[idx] > threshold_value) ? max_value : 0;
}

std::pair<std::unique_ptr<GpuImage>, ThresholdTiming> threshold_basic(
    const GpuImage& input, uint8_t threshold_value, uint8_t max_value) {
    auto output = std::make_unique<GpuImage>(input.height(), input.width());

    dim3 block = default_block_dim();
    dim3 grid = compute_launch_grid(input.width(), input.height(), block);

    CudaTimer timer;
    timer.start();
    threshold_basic_kernel<<<grid, block>>>(
        input.data(), output->data(), input.width(), input.height(), threshold_value, max_value);
    CUDA_CHECK_LAST_ERROR();
    timer.stop();

    CUDA_CHECK(cudaDeviceSynchronize());
    float kernel_ms = timer.elapsed_ms();

    return {std::move(output), ThresholdTiming{kernel_ms}};
}

}  // namespace xray_cuda
