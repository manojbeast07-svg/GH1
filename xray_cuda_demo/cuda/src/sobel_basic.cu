#include "sobel_basic.cuh"
#include <cmath>
#include <stdexcept>
#include <string>

namespace xray_cuda {

__global__ void sobel_basic_kernel(
    const uint8_t* input, uint8_t* output,
    int width, int height, int mode) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    size_t plane = static_cast<size_t>(height) * static_cast<size_t>(width);
    const uint8_t* in = input + static_cast<size_t>(blockIdx.z) * plane;
    uint8_t* out = output + static_cast<size_t>(blockIdx.z) * plane;

    // Read the 3x3 neighborhood once; both gx and gy are derived from it
    // (avoids a second kernel launch / second neighborhood read for the
    // same pixels -- see header comment).
    float p00 = static_cast<float>(in[reflect101(y - 1, height) * width + reflect101(x - 1, width)]);
    float p01 = static_cast<float>(in[reflect101(y - 1, height) * width + reflect101(x, width)]);
    float p02 = static_cast<float>(in[reflect101(y - 1, height) * width + reflect101(x + 1, width)]);
    float p10 = static_cast<float>(in[reflect101(y, height) * width + reflect101(x - 1, width)]);
    float p12 = static_cast<float>(in[reflect101(y, height) * width + reflect101(x + 1, width)]);
    float p20 = static_cast<float>(in[reflect101(y + 1, height) * width + reflect101(x - 1, width)]);
    float p21 = static_cast<float>(in[reflect101(y + 1, height) * width + reflect101(x, width)]);
    float p22 = static_cast<float>(in[reflect101(y + 1, height) * width + reflect101(x + 1, width)]);

    // Gx = [[-1,0,1],[-2,0,2],[-1,0,1]]
    float gx = (p02 - p00) + 2.0f * (p12 - p10) + (p22 - p20);
    // Gy = [[-1,-2,-1],[0,0,0],[1,2,1]]
    float gy = (p20 - p00) + 2.0f * (p21 - p01) + (p22 - p02);

    float raw;
    switch (static_cast<SobelMode>(mode)) {
        case SobelMode::X:
            raw = gx;
            break;
        case SobelMode::Y:
            raw = gy;
            break;
        case SobelMode::Magnitude:
            raw = sqrtf(gx * gx + gy * gy);
            break;
        case SobelMode::AbsSum:
        default:
            raw = fabsf(gx) + fabsf(gy);
            break;
    }

    // Matches cv2.convertScaleAbs: saturate_cast<uchar>(|raw|), rounded
    // to nearest (verified empirically -- see header comment).
    int rounded = __float2int_rn(fabsf(raw));
    rounded = max(0, min(255, rounded));
    out[y * width + x] = static_cast<uint8_t>(rounded);
}

std::pair<std::unique_ptr<GpuImage>, SobelTiming> sobel_basic(const GpuImage& input, int mode) {
    if (mode < 0 || mode > static_cast<int>(SobelMode::AbsSum)) {
        throw std::invalid_argument("sobel_basic: invalid mode " + std::to_string(mode));
    }

    auto output = std::make_unique<GpuImage>(input.height(), input.width());

    dim3 block = default_block_dim();
    dim3 grid = compute_launch_grid(input.width(), input.height(), block);

    CudaTimer timer;
    timer.start();
    sobel_basic_kernel<<<grid, block>>>(input.data(), output->data(), input.width(), input.height(), mode);
    CUDA_CHECK_LAST_ERROR();
    timer.stop();

    CUDA_CHECK(cudaDeviceSynchronize());
    float kernel_ms = timer.elapsed_ms();

    return {std::move(output), SobelTiming{kernel_ms}};
}

}  // namespace xray_cuda
