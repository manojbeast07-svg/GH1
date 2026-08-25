#include "sobel_enhanced.cuh"
#include <cmath>
#include <stdexcept>
#include <string>

namespace xray_cuda {

// Constant-memory coefficients for the SharedConst variant. Row-major,
// dy in {-1,0,1} outer, dx in {-1,0,1} inner -- matches sobel_basic.cuh's
// documented Gx/Gy matrices exactly. Populated via cudaMemcpyToSymbol
// once per dispatch call (the values never change, but re-uploading is
// cheap and keeps the dispatch function self-contained, same convention
// gaussian_enhanced.cu uses even though Gaussian's coefficients DO vary).
__constant__ float d_sobel_gx_coeffs[9];
__constant__ float d_sobel_gy_coeffs[9];

namespace {

constexpr float kHostSobelGxCoeffs[9] = {-1.0f, 0.0f, 1.0f, -2.0f, 0.0f, 2.0f, -1.0f, 0.0f, 1.0f};
constexpr float kHostSobelGyCoeffs[9] = {-1.0f, -2.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 1.0f};

// Converts a raw gx/gy-derived value to the final uint8 output, exactly
// matching cv2.convertScaleAbs (see sobel_basic.cu): round(|raw|),
// clamped to [0,255]. Shared by every kernel below.
__device__ __forceinline__ uint8_t sobel_finalize(float raw) {
    int rounded = __float2int_rn(fabsf(raw));
    rounded = max(0, min(255, rounded));
    return static_cast<uint8_t>(rounded);
}

__device__ __forceinline__ float sobel_combine(float gx, float gy, int mode) {
    switch (static_cast<SobelMode>(mode)) {
        case SobelMode::X: return gx;
        case SobelMode::Y: return gy;
        case SobelMode::Magnitude: return sqrtf(gx * gx + gy * gy);
        case SobelMode::AbsSum:
        default: return fabsf(gx) + fabsf(gy);
    }
}

// Loads a (blockDim.x+2) x (blockDim.y+2) BORDER_REFLECT_101 halo tile
// (radius=1, the only radius Sobel supports) into `tile`. Shared by
// Shared/SharedConst/Specialized below -- one load routine, not three
// copies (same pattern as median_enhanced.cu's load_median_tile).
__device__ __forceinline__ void load_sobel_tile(
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

// -- Optimization 1: shared-memory tiling, same hand-optimized arithmetic as Basic -----------------------

__global__ void sobel_shared_kernel(
    const uint8_t* input, uint8_t* output, int width, int height, int mode) {
    extern __shared__ uint8_t s_tile_sobel[];
    int tile_w = static_cast<int>(blockDim.x) + 2;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t plane = static_cast<size_t>(height) * width;
    const uint8_t* in = input + static_cast<size_t>(blockIdx.z) * plane;
    uint8_t* out = output + static_cast<size_t>(blockIdx.z) * plane;

    load_sobel_tile(s_tile_sobel, in, width, height, tile_w);
    __syncthreads();

    if (x >= width || y >= height) return;

    int lx = threadIdx.x, ly = threadIdx.y;
    auto tp = [&](int dy, int dx) -> float {
        return static_cast<float>(s_tile_sobel[(ly + 1 + dy) * tile_w + (lx + 1 + dx)]);
    };
    float p00 = tp(-1, -1), p01 = tp(-1, 0), p02 = tp(-1, 1);
    float p10 = tp(0, -1), p12 = tp(0, 1);
    float p20 = tp(1, -1), p21 = tp(1, 0), p22 = tp(1, 1);

    float gx = (p02 - p00) + 2.0f * (p12 - p10) + (p22 - p20);
    float gy = (p20 - p00) + 2.0f * (p21 - p01) + (p22 - p02);

    out[y * width + x] = sobel_finalize(sobel_combine(gx, gy, mode));
}

// -- Optimization 2: + constant-memory coefficients, generic 3x3 loop -----------------------

__global__ void sobel_shared_const_kernel(
    const uint8_t* input, uint8_t* output, int width, int height, int mode) {
    extern __shared__ uint8_t s_tile_sobel_const[];
    int tile_w = static_cast<int>(blockDim.x) + 2;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t plane = static_cast<size_t>(height) * width;
    const uint8_t* in = input + static_cast<size_t>(blockIdx.z) * plane;
    uint8_t* out = output + static_cast<size_t>(blockIdx.z) * plane;

    load_sobel_tile(s_tile_sobel_const, in, width, height, tile_w);
    __syncthreads();

    if (x >= width || y >= height) return;

    int lx = threadIdx.x, ly = threadIdx.y;
    float gx = 0.0f, gy = 0.0f;
#pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            float v = static_cast<float>(s_tile_sobel_const[(ly + 1 + dy) * tile_w + (lx + 1 + dx)]);
            int ci = (dy + 1) * 3 + (dx + 1);
            gx += d_sobel_gx_coeffs[ci] * v;
            gy += d_sobel_gy_coeffs[ci] * v;
        }
    }

    out[y * width + x] = sobel_finalize(sobel_combine(gx, gy, mode));
}

// -- Optimization 3: + compile-time mode specialization (skip unused derivative) -----------------------
//
// MODE matches SobelMode's int values (0=X,1=Y,2=Magnitude,3=AbsSum).
// X only needs gx (skips reading p01/p21 and all of gy's arithmetic);
// Y only needs gy (skips p10/p12 and gx's arithmetic); Magnitude/AbsSum
// need both, same as Basic/Shared.

template <int MODE>
__global__ void sobel_specialized_kernel(const uint8_t* input, uint8_t* output, int width, int height) {
    extern __shared__ uint8_t s_tile_sobel_spec[];
    int tile_w = static_cast<int>(blockDim.x) + 2;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t plane = static_cast<size_t>(height) * width;
    const uint8_t* in = input + static_cast<size_t>(blockIdx.z) * plane;
    uint8_t* out = output + static_cast<size_t>(blockIdx.z) * plane;

    load_sobel_tile(s_tile_sobel_spec, in, width, height, tile_w);
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

    out[y * width + x] = sobel_finalize(raw);
}

// -- Optimization 4: separable two-pass decomposition (no shared memory) -----------------------
//
// Gx = [1,2,1]_vertical (x) [-1,0,1]_horizontal -- pass 1 computes the
// horizontal derivative dx[y,x] = in[y,x+1]-in[y,x-1]; pass 2 applies
// the vertical [1,2,1] smoothing to dx to finish gx.
// Gy = [-1,0,1]_vertical (x) [1,2,1]_horizontal -- pass 1 computes the
// horizontal smoothing sx[y,x] = in[y,x-1]+2*in[y,x]+in[y,x+1]; pass 2
// applies the vertical [-1,0,1] derivative to sx to finish gy.
// Both pass-1 outputs are produced by ONE kernel (single global read of
// `in` per pixel), same architectural principle as Basic's "read once,
// derive both gx and gy" -- only the *vertical* work is split out.

__global__ void sobel_separable_h_kernel(
    const uint8_t* input, float* out_dx, float* out_sx, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    size_t plane = static_cast<size_t>(height) * width;
    const uint8_t* in = input + static_cast<size_t>(blockIdx.z) * plane;
    float* dxp = out_dx + static_cast<size_t>(blockIdx.z) * plane;
    float* sxp = out_sx + static_cast<size_t>(blockIdx.z) * plane;

    float left = static_cast<float>(in[y * width + reflect101(x - 1, width)]);
    float center = static_cast<float>(in[y * width + x]);
    float right = static_cast<float>(in[y * width + reflect101(x + 1, width)]);

    dxp[y * width + x] = right - left;
    sxp[y * width + x] = left + 2.0f * center + right;
}

__global__ void sobel_separable_v_kernel(
    const float* in_dx, const float* in_sx, uint8_t* output, int width, int height, int mode) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    size_t plane = static_cast<size_t>(height) * width;
    const float* dxp = in_dx + static_cast<size_t>(blockIdx.z) * plane;
    const float* sxp = in_sx + static_cast<size_t>(blockIdx.z) * plane;
    uint8_t* out = output + static_cast<size_t>(blockIdx.z) * plane;

    int y_top = reflect101(y - 1, height);
    int y_bot = reflect101(y + 1, height);

    float gx = dxp[y_top * width + x] + 2.0f * dxp[y * width + x] + dxp[y_bot * width + x];
    float gy = sxp[y_bot * width + x] - sxp[y_top * width + x];

    out[y * width + x] = sobel_finalize(sobel_combine(gx, gy, mode));
}

}  // namespace

// -- shared dispatch: the ONE place the variant switch statement exists -----------------------

void sobel_enhanced_dispatch(
    const uint8_t* d_input, uint8_t* d_output,
    int batch_size, int height, int width, int mode,
    SobelVariant variant, int block_x, int block_y,
    SobelEnhancedTiming& timing) {
    if (mode < 0 || mode > static_cast<int>(SobelMode::AbsSum)) {
        throw std::invalid_argument("sobel_enhanced_dispatch: invalid mode " + std::to_string(mode));
    }
    if (block_x <= 0 || block_y <= 0) {
        throw std::invalid_argument("sobel_enhanced_dispatch: block_x/block_y must be positive.");
    }

    dim3 block(static_cast<unsigned int>(block_x), static_cast<unsigned int>(block_y));
    dim3 grid2d = compute_launch_grid(width, height, block);
    dim3 grid(grid2d.x, grid2d.y, static_cast<unsigned int>(batch_size));

    if (variant == SobelVariant::Separable) {
        size_t plane_count = static_cast<size_t>(batch_size) * static_cast<size_t>(height) * static_cast<size_t>(width);
        DeviceBuffer<float> d_dx(plane_count);
        DeviceBuffer<float> d_sx(plane_count);

        CudaTimer h_timer;
        h_timer.start();
        sobel_separable_h_kernel<<<grid, block>>>(d_input, d_dx.get(), d_sx.get(), width, height);
        CUDA_CHECK_LAST_ERROR();
        h_timer.stop();
        CUDA_CHECK(cudaDeviceSynchronize());
        timing.horizontal_ms = h_timer.elapsed_ms();

        CudaTimer v_timer;
        v_timer.start();
        sobel_separable_v_kernel<<<grid, block>>>(d_dx.get(), d_sx.get(), d_output, width, height, mode);
        CUDA_CHECK_LAST_ERROR();
        v_timer.stop();
        CUDA_CHECK(cudaDeviceSynchronize());
        timing.vertical_ms = v_timer.elapsed_ms();
        return;
    }

    size_t shared_bytes = static_cast<size_t>(block_x + 2) * (block_y + 2) * sizeof(uint8_t);
    timing.horizontal_ms = 0.0f;

    CudaTimer timer;
    timer.start();
    switch (variant) {
        case SobelVariant::Shared:
            sobel_shared_kernel<<<grid, block, shared_bytes>>>(d_input, d_output, width, height, mode);
            break;
        case SobelVariant::SharedConst:
            CUDA_CHECK(cudaMemcpyToSymbol(d_sobel_gx_coeffs, kHostSobelGxCoeffs, 9 * sizeof(float)));
            CUDA_CHECK(cudaMemcpyToSymbol(d_sobel_gy_coeffs, kHostSobelGyCoeffs, 9 * sizeof(float)));
            sobel_shared_const_kernel<<<grid, block, shared_bytes>>>(d_input, d_output, width, height, mode);
            break;
        case SobelVariant::Specialized:
            switch (static_cast<SobelMode>(mode)) {
                case SobelMode::X:
                    sobel_specialized_kernel<0><<<grid, block, shared_bytes>>>(d_input, d_output, width, height);
                    break;
                case SobelMode::Y:
                    sobel_specialized_kernel<1><<<grid, block, shared_bytes>>>(d_input, d_output, width, height);
                    break;
                case SobelMode::Magnitude:
                    sobel_specialized_kernel<2><<<grid, block, shared_bytes>>>(d_input, d_output, width, height);
                    break;
                case SobelMode::AbsSum:
                    sobel_specialized_kernel<3><<<grid, block, shared_bytes>>>(d_input, d_output, width, height);
                    break;
            }
            break;
        case SobelVariant::Separable:
            break;  // handled above
    }
    CUDA_CHECK_LAST_ERROR();
    timer.stop();
    CUDA_CHECK(cudaDeviceSynchronize());
    timing.vertical_ms = timer.elapsed_ms();
}

// -- thin single-image / batch wrappers -----------------------

std::pair<std::unique_ptr<GpuImage>, SobelEnhancedTiming> sobel_enhanced(
    const GpuImage& input, int mode, SobelVariant variant, int block_x, int block_y) {
    auto output = std::make_unique<GpuImage>(input.height(), input.width());
    SobelEnhancedTiming timing{};
    sobel_enhanced_dispatch(
        input.data(), output->data(), /*batch_size=*/1, input.height(), input.width(), mode,
        variant, block_x, block_y, timing);
    return {std::move(output), timing};
}

std::pair<std::unique_ptr<GpuImageBatch>, SobelEnhancedTiming> sobel_enhanced_batch(
    const GpuImageBatch& input, int mode, SobelVariant variant, int block_x, int block_y) {
    auto output = std::make_unique<GpuImageBatch>(input.batch_size(), input.height(), input.width());
    SobelEnhancedTiming timing{};
    sobel_enhanced_dispatch(
        input.data(), output->data(), input.batch_size(), input.height(), input.width(), mode,
        variant, block_x, block_y, timing);
    return {std::move(output), timing};
}

}  // namespace xray_cuda
