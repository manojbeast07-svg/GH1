#include "pipeline_basic.cuh"
#include "gaussian_basic.cuh"
#include "laplacian_basic.cuh"
#include "laplacian_enhanced.cuh"
#include "median_basic.cuh"
#include "sobel_basic.cuh"
#include "sobel_enhanced.cuh"
#include "threshold_basic.cuh"
#include "threshold_enhanced.cuh"
#include <stdexcept>
#include <string>
#include <utility>

namespace xray_cuda {

std::pair<std::vector<uint8_t>, BasicPipelineTiming> run_basic_cuda_pipeline_batch(
    const uint8_t* host_batch, int batch_size, int height, int width, const BasicPipelineConfig& config) {
    BasicPipelineTiming timing;

    // Two persistent buffers for the whole call, reused (ping-pong)
    // across every enabled stage -- the core architectural change this
    // section makes: no per-stage cudaMalloc/cudaFree.
    GpuImageBatch buffer_a(batch_size, height, width);
    GpuImageBatch buffer_b(batch_size, height, width);
    const size_t total_bytes = buffer_a.nbytes();

    CudaTimer h2d_timer;
    h2d_timer.start();
    buffer_a.upload_from_host(host_batch, total_bytes);
    h2d_timer.stop();
    timing.h2d_ms = h2d_timer.elapsed_ms();

    GpuImageBatch* current = &buffer_a;
    GpuImageBatch* other = &buffer_b;

    dim3 block = default_block_dim();
    dim3 grid2d = compute_launch_grid(width, height, block);
    dim3 grid(grid2d.x, grid2d.y, static_cast<unsigned int>(batch_size));

    if (config.gaussian_enabled) {
        if (config.gaussian_use_enhanced) {
            // Section 6: separable Enhanced Gaussian, on this call's own
            // ping-pong buffers -- only the intermediate float plane is
            // extra allocation, no per-stage GpuImage/GpuImageBatch.
            DeviceBuffer<float> intermediate(
                static_cast<size_t>(batch_size) * static_cast<size_t>(height) * static_cast<size_t>(width));
            GaussianEnhancedTiming gtiming{};
            gaussian_enhanced_dispatch(
                current->data(), intermediate.get(), other->data(),
                batch_size, height, width,
                config.gaussian_coeffs_1d, config.gaussian_kernel_size,
                config.gaussian_variant, config.gaussian_block_x, config.gaussian_block_y,
                gtiming);
            timing.gaussian_ms = gtiming.kernel_ms();
        } else {
            if (config.gaussian_kernel_size <= 0 || config.gaussian_kernel_size % 2 == 0) {
                throw std::invalid_argument(
                    "run_basic_cuda_pipeline_batch: invalid gaussian_kernel_size " +
                    std::to_string(config.gaussian_kernel_size));
            }
            size_t coeff_count = static_cast<size_t>(config.gaussian_kernel_size) * config.gaussian_kernel_size;
            DeviceBuffer<float> device_coeffs(coeff_count);

            CudaTimer timer;
            timer.start();
            device_coeffs.upload(config.gaussian_coeffs, coeff_count);  // bundled into this stage's time, not reported separately
            gaussian_basic_kernel<<<grid, block>>>(
                current->data(), other->data(), width, height, device_coeffs.get(), config.gaussian_kernel_size);
            CUDA_CHECK_LAST_ERROR();
            timer.stop();
            CUDA_CHECK(cudaDeviceSynchronize());
            timing.gaussian_ms = timer.elapsed_ms();
        }
        std::swap(current, other);
    }

    if (config.median_enabled) {
        if (config.median_use_enhanced) {
            // Section 7: Enhanced Median, on this call's own ping-pong
            // buffers -- no extra allocation needed (unlike Enhanced
            // Gaussian, Median has no intermediate float buffer).
            MedianEnhancedTiming mtiming{};
            median_enhanced_dispatch(
                current->data(), other->data(), batch_size, height, width, config.median_kernel_size,
                config.median_variant, config.median_block_x, config.median_block_y, mtiming);
            timing.median_ms = mtiming.kernel_ms;
        } else {
            if (config.median_kernel_size <= 0 || config.median_kernel_size % 2 == 0 ||
                config.median_kernel_size > kMedianBasicMaxKernelSize) {
                throw std::invalid_argument(
                    "run_basic_cuda_pipeline_batch: invalid median_kernel_size " +
                    std::to_string(config.median_kernel_size));
            }
            CudaTimer timer;
            timer.start();
            median_basic_kernel<<<grid, block>>>(current->data(), other->data(), width, height, config.median_kernel_size);
            CUDA_CHECK_LAST_ERROR();
            timer.stop();
            CUDA_CHECK(cudaDeviceSynchronize());
            timing.median_ms = timer.elapsed_ms();
        }
        std::swap(current, other);
    }

    if (config.sobel_enabled) {
        if (config.sobel_mode < 0 || config.sobel_mode > static_cast<int>(SobelMode::AbsSum)) {
            throw std::invalid_argument(
                "run_basic_cuda_pipeline_batch: invalid sobel_mode " + std::to_string(config.sobel_mode));
        }
        if (config.sobel_use_enhanced) {
            // Section 8: Enhanced Sobel, on this call's own ping-pong
            // buffers -- Shared/SharedConst/Specialized need no extra
            // allocation; Separable self-manages its own intermediate
            // buffers inside sobel_enhanced_dispatch() (see
            // sobel_enhanced.cuh for why).
            SobelEnhancedTiming stiming{};
            sobel_enhanced_dispatch(
                current->data(), other->data(), batch_size, height, width, config.sobel_mode,
                config.sobel_variant, config.sobel_block_x, config.sobel_block_y, stiming);
            timing.sobel_ms = stiming.kernel_ms();
        } else {
            CudaTimer timer;
            timer.start();
            sobel_basic_kernel<<<grid, block>>>(current->data(), other->data(), width, height, config.sobel_mode);
            CUDA_CHECK_LAST_ERROR();
            timer.stop();
            CUDA_CHECK(cudaDeviceSynchronize());
            timing.sobel_ms = timer.elapsed_ms();
        }
        std::swap(current, other);
    }

    if (config.laplacian_enabled) {
        if (config.laplacian_kernel_size <= 0 || config.laplacian_kernel_size % 2 == 0) {
            throw std::invalid_argument(
                "run_basic_cuda_pipeline_batch: invalid laplacian_kernel_size " +
                std::to_string(config.laplacian_kernel_size));
        }
        if (config.laplacian_use_enhanced) {
            // Section 9: Enhanced Laplacian, on this call's own ping-pong
            // buffers -- no extra allocation needed (like Median, unlike
            // Gaussian's intermediate float buffer).
            LaplacianEnhancedTiming ltiming{};
            laplacian_enhanced_dispatch(
                current->data(), other->data(), batch_size, height, width,
                config.laplacian_coeffs, config.laplacian_kernel_size, config.laplacian_scale, config.laplacian_delta,
                config.laplacian_variant, config.laplacian_block_x, config.laplacian_block_y, ltiming);
            timing.laplacian_ms = ltiming.kernel_ms;
        } else {
            size_t coeff_count = static_cast<size_t>(config.laplacian_kernel_size) * config.laplacian_kernel_size;
            DeviceBuffer<float> device_coeffs(coeff_count);

            CudaTimer timer;
            timer.start();
            device_coeffs.upload(config.laplacian_coeffs, coeff_count);
            laplacian_basic_kernel<<<grid, block>>>(
                current->data(), other->data(), width, height,
                device_coeffs.get(), config.laplacian_kernel_size, config.laplacian_scale, config.laplacian_delta);
            CUDA_CHECK_LAST_ERROR();
            timer.stop();
            CUDA_CHECK(cudaDeviceSynchronize());
            timing.laplacian_ms = timer.elapsed_ms();
        }
        std::swap(current, other);
    }

    if (config.threshold_enabled) {
        if (config.threshold_use_enhanced) {
            // Section 10: Enhanced Threshold, on this call's own ping-pong
            // buffers -- no extra allocation needed.
            ThresholdEnhancedTiming ttiming{};
            threshold_enhanced_dispatch(
                current->data(), other->data(), batch_size, height, width,
                config.threshold_value, config.threshold_max_value,
                config.threshold_variant, config.threshold_block_x, config.threshold_block_y, ttiming);
            timing.threshold_ms = ttiming.kernel_ms;
        } else {
            CudaTimer timer;
            timer.start();
            threshold_basic_kernel<<<grid, block>>>(
                current->data(), other->data(), width, height, config.threshold_value, config.threshold_max_value);
            CUDA_CHECK_LAST_ERROR();
            timer.stop();
            CUDA_CHECK(cudaDeviceSynchronize());
            timing.threshold_ms = timer.elapsed_ms();
        }
        std::swap(current, other);
    }

    std::vector<uint8_t> host_output(total_bytes);
    CudaTimer d2h_timer;
    d2h_timer.start();
    current->download_to_host(host_output.data(), total_bytes);
    d2h_timer.stop();
    timing.d2h_ms = d2h_timer.elapsed_ms();

    return {std::move(host_output), timing};
}

}  // namespace xray_cuda
