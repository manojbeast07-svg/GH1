#include "cuda_common.cuh"
#include "smoke_test.hpp"
#include <vector>

namespace xray_cuda {

namespace {

__global__ void add_one_kernel(const float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = input[i] + 1.0f;
    }
}

}  // namespace

// Runs input[i] + 1 on the GPU for a small fixed-size vector, using the
// DeviceBuffer/CudaTimer foundations. Proves H2D -> kernel -> D2H works.
SmokeTestResult run_smoke_test() {
    const std::vector<float> input = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    const int n = static_cast<int>(input.size());

    DeviceBuffer<float> d_input(n);
    DeviceBuffer<float> d_output(n);

    d_input.upload(input.data(), n);

    const int threads_per_block = 256;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;

    CudaTimer timer;
    timer.start();
    add_one_kernel<<<blocks, threads_per_block>>>(d_input.get(), d_output.get(), n);
    CUDA_CHECK_LAST_ERROR();
    timer.stop();

    SmokeTestResult result;
    result.output.resize(n);
    d_output.download(result.output.data(), n);
    CUDA_CHECK(cudaDeviceSynchronize());
    result.kernel_ms = timer.elapsed_ms();

    return result;
}

}  // namespace xray_cuda
