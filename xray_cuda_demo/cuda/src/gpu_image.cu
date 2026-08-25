#include "gpu_image.cuh"

namespace xray_cuda {

__global__ void image_add_one_kernel(const uint8_t* input, uint8_t* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        output[idx] = static_cast<uint8_t>(input[idx] + 1);  // wraps at 255 -> 0
    }
}

std::pair<std::unique_ptr<GpuImage>, float> image_add_one(const GpuImage& input) {
    auto output = std::make_unique<GpuImage>(input.height(), input.width());

    dim3 block = default_block_dim();
    dim3 grid = compute_launch_grid(input.width(), input.height(), block);

    CudaTimer timer;
    timer.start();
    image_add_one_kernel<<<grid, block>>>(input.data(), output->data(), input.width(), input.height());
    CUDA_CHECK_LAST_ERROR();
    timer.stop();

    // Synchronization policy for Section 4A: upload -> kernel ->
    // synchronize -> download. Correctness first; no overlap/async claims.
    CUDA_CHECK(cudaDeviceSynchronize());
    float kernel_ms = timer.elapsed_ms();

    return {std::move(output), kernel_ms};
}

}  // namespace xray_cuda
