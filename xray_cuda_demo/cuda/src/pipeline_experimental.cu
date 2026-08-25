#include "pipeline_experimental.cuh"
#include "gaussian_basic.cuh"
#include "laplacian_basic.cuh"
#include "laplacian_enhanced.cuh"
#include "median_basic.cuh"
#include "sobel_basic.cuh"
#include "sobel_enhanced.cuh"
#include "threshold_basic.cuh"
#include "threshold_enhanced.cuh"
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

namespace xray_cuda {

namespace {
// CPU-side high-resolution timer for host-only work (allocation, staging
// copies) -- spec item 16: CUDA events for H2D/D2H/kernels, CPU timers
// for native API overhead/allocation/host staging, never mixed silently.
class HostTimer {
public:
    HostTimer() : start_(std::chrono::steady_clock::now()) {}
    float elapsed_ms() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<float, std::milli>(now - start_).count();
    }
private:
    std::chrono::steady_clock::time_point start_;
};
}  // namespace

void PersistentCudaPipeline::ensure_gpu_capacity(int batch_size, int height, int width, bool& grew) {
    grew = false;
    if (persistent_buffer_a_ && gpu_capacity_batch_ >= batch_size &&
        gpu_capacity_height_ == height && gpu_capacity_width_ == width) {
        return;  // reuse -- spec item 8: never shrink/reallocate for a smaller-or-equal request
    }
    // (Re)allocate at the new capacity. A resolution change (height/width) must not reuse an
    // incompatible layout (spec item 27); a larger batch_size at the same resolution grows in
    // place (spec item 7); a smaller-or-equal request at the same resolution is a no-op above.
    persistent_buffer_a_ = std::make_unique<GpuImageBatch>(batch_size, height, width);
    persistent_buffer_b_ = std::make_unique<GpuImageBatch>(batch_size, height, width);
    persistent_intermediate_ = std::make_unique<DeviceBuffer<float>>(
        static_cast<size_t>(batch_size) * static_cast<size_t>(height) * static_cast<size_t>(width));
    gpu_capacity_batch_ = batch_size;
    gpu_capacity_height_ = height;
    gpu_capacity_width_ = width;
    grew = true;
}

void PersistentCudaPipeline::ensure_pinned_capacity(size_t bytes, bool& grew) {
    grew = false;
    if (pinned_input_ != nullptr && pinned_capacity_bytes_ >= bytes) {
        return;
    }
    if (pinned_input_ != nullptr) {
        cudaFreeHost(pinned_input_);
        cudaFreeHost(pinned_output_);
    }
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&pinned_input_), bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&pinned_output_), bytes, cudaHostAllocDefault));
    pinned_capacity_bytes_ = bytes;
    grew = true;
}

void PersistentCudaPipeline::release() {
    persistent_buffer_a_.reset();
    persistent_buffer_b_.reset();
    persistent_intermediate_.reset();
    gpu_capacity_batch_ = gpu_capacity_height_ = gpu_capacity_width_ = 0;
    if (pinned_input_ != nullptr) {
        cudaFreeHost(pinned_input_);
        pinned_input_ = nullptr;
    }
    if (pinned_output_ != nullptr) {
        cudaFreeHost(pinned_output_);
        pinned_output_ = nullptr;
    }
    pinned_capacity_bytes_ = 0;
    released_ = true;
}

std::pair<std::vector<uint8_t>, PersistentCudaPipeline::RunTiming> PersistentCudaPipeline::run(
    const uint8_t* host_batch, int batch_size, int height, int width, const BasicPipelineConfig& config,
    bool use_persistent_buffers, bool use_pinned_memory) {
    if (released_) {
        throw std::runtime_error(
            "PersistentCudaPipeline::run() called after release() -- this instance is permanently "
            "unusable once released. Construct a new PersistentCudaPipeline instead of reusing a "
            "released one.");
    }
    RunTiming timing;
    timing.used_persistent_buffers = use_persistent_buffers;
    timing.used_pinned_memory = use_pinned_memory;
    const size_t total_bytes =
        static_cast<size_t>(batch_size) * static_cast<size_t>(height) * static_cast<size_t>(width);

    // -- GPU buffers: persistent (reused/grown across calls) or fresh (matches production exactly) --
    GpuImageBatch* current;
    GpuImageBatch* other;
    DeviceBuffer<float>* intermediate;
    std::unique_ptr<GpuImageBatch> fresh_a, fresh_b;
    std::unique_ptr<DeviceBuffer<float>> fresh_intermediate;

    if (use_persistent_buffers) {
        HostTimer alloc_timer;
        bool grew = false;
        ensure_gpu_capacity(batch_size, height, width, grew);
        timing.grew_gpu_buffers = grew;
        timing.alloc_ms = alloc_timer.elapsed_ms();  // ~0 when reused, real cost only when it grows
        current = persistent_buffer_a_.get();
        other = persistent_buffer_b_.get();
        intermediate = persistent_intermediate_.get();
    } else {
        HostTimer alloc_timer;
        fresh_a = std::make_unique<GpuImageBatch>(batch_size, height, width);
        fresh_b = std::make_unique<GpuImageBatch>(batch_size, height, width);
        fresh_intermediate = std::make_unique<DeviceBuffer<float>>(total_bytes);
        timing.alloc_ms = alloc_timer.elapsed_ms();
        current = fresh_a.get();
        other = fresh_b.get();
        intermediate = fresh_intermediate.get();
    }

    // -- host input staging: pageable (direct) or pinned (staged copy first) --
    const uint8_t* h2d_source = host_batch;
    if (use_pinned_memory) {
        HostTimer stage_timer;
        bool grew = false;
        ensure_pinned_capacity(total_bytes, grew);
        timing.grew_pinned_buffers = grew;
        std::memcpy(pinned_input_, host_batch, total_bytes);
        timing.host_stage_ms = stage_timer.elapsed_ms();
        h2d_source = pinned_input_;
    }

    // Raw cudaMemcpy (not GpuImageBatch::upload_from_host()'s strict
    // byte-count check) deliberately: a persistent buffer can be LARGER
    // than the current request (spec item 8 -- reuse the 512-capacity
    // buffers for a batch=32 request rather than reallocating), so only
    // the front total_bytes of a possibly-larger allocation is used.
    CudaTimer h2d_timer;
    h2d_timer.start();
    CUDA_CHECK(cudaMemcpy(current->data(), h2d_source, total_bytes, cudaMemcpyHostToDevice));
    h2d_timer.stop();
    timing.h2d_ms = h2d_timer.elapsed_ms();

    dim3 block = default_block_dim();
    dim3 grid2d = compute_launch_grid(width, height, block);
    dim3 grid(grid2d.x, grid2d.y, static_cast<unsigned int>(batch_size));

    if (config.gaussian_enabled) {
        if (config.gaussian_use_enhanced) {
            GaussianEnhancedTiming gtiming{};
            gaussian_enhanced_dispatch(
                current->data(), intermediate->get(), other->data(), batch_size, height, width,
                config.gaussian_coeffs_1d, config.gaussian_kernel_size, config.gaussian_variant,
                config.gaussian_block_x, config.gaussian_block_y, gtiming);
            timing.gaussian_ms = gtiming.kernel_ms();
        } else {
            size_t coeff_count = static_cast<size_t>(config.gaussian_kernel_size) * config.gaussian_kernel_size;
            DeviceBuffer<float> device_coeffs(coeff_count);
            CudaTimer timer;
            timer.start();
            device_coeffs.upload(config.gaussian_coeffs, coeff_count);
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
            MedianEnhancedTiming mtiming{};
            median_enhanced_dispatch(
                current->data(), other->data(), batch_size, height, width, config.median_kernel_size,
                config.median_variant, config.median_block_x, config.median_block_y, mtiming);
            timing.median_ms = mtiming.kernel_ms;
        } else {
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
        if (config.sobel_use_enhanced) {
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
        if (config.laplacian_use_enhanced) {
            LaplacianEnhancedTiming ltiming{};
            laplacian_enhanced_dispatch(
                current->data(), other->data(), batch_size, height, width, config.laplacian_coeffs,
                config.laplacian_kernel_size, config.laplacian_scale, config.laplacian_delta,
                config.laplacian_variant, config.laplacian_block_x, config.laplacian_block_y, ltiming);
            timing.laplacian_ms = ltiming.kernel_ms;
        } else {
            size_t coeff_count = static_cast<size_t>(config.laplacian_kernel_size) * config.laplacian_kernel_size;
            DeviceBuffer<float> device_coeffs(coeff_count);
            CudaTimer timer;
            timer.start();
            device_coeffs.upload(config.laplacian_coeffs, coeff_count);
            laplacian_basic_kernel<<<grid, block>>>(
                current->data(), other->data(), width, height, device_coeffs.get(),
                config.laplacian_kernel_size, config.laplacian_scale, config.laplacian_delta);
            CUDA_CHECK_LAST_ERROR();
            timer.stop();
            CUDA_CHECK(cudaDeviceSynchronize());
            timing.laplacian_ms = timer.elapsed_ms();
        }
        std::swap(current, other);
    }

    if (config.threshold_enabled) {
        if (config.threshold_use_enhanced) {
            ThresholdEnhancedTiming ttiming{};
            threshold_enhanced_dispatch(
                current->data(), other->data(), batch_size, height, width, config.threshold_value,
                config.threshold_max_value, config.threshold_variant, config.threshold_block_x,
                config.threshold_block_y, ttiming);
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
    if (use_pinned_memory) {
        CudaTimer d2h_timer;
        d2h_timer.start();
        CUDA_CHECK(cudaMemcpy(pinned_output_, current->data(), total_bytes, cudaMemcpyDeviceToHost));
        d2h_timer.stop();
        timing.d2h_ms = d2h_timer.elapsed_ms();

        HostTimer stage_timer;
        std::memcpy(host_output.data(), pinned_output_, total_bytes);
        timing.host_stage_ms += stage_timer.elapsed_ms();
    } else {
        CudaTimer d2h_timer;
        d2h_timer.start();
        CUDA_CHECK(cudaMemcpy(host_output.data(), current->data(), total_bytes, cudaMemcpyDeviceToHost));
        d2h_timer.stop();
        timing.d2h_ms = d2h_timer.elapsed_ms();
    }

    return {std::move(host_output), timing};
}

}  // namespace xray_cuda
