#pragma once

#include "gpu_image.cuh"
#include "pipeline_basic.cuh"
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace xray_cuda {

// --------------------------------------------------------------------------
// Section 20B: EXPERIMENTAL ONLY.
//
// Isolates the two pipeline-level memory-strategy variables Section 20A's
// research identified (GPU buffer persistence across repeated calls,
// pinned host staging memory) from everything else. Never wired into the
// production run_basic_cuda_pipeline()/run_enhanced_cuda_pipeline() Python
// API (cuda/pipeline.py) -- reachable only through its own
// run_experimental_persistent_pinned_gpu() pybind11 binding.
//
// Uses the EXACT SAME per-stage kernel dispatch calls
// pipeline_basic.cu's run_basic_cuda_pipeline_batch() uses (same
// gaussian_basic_kernel/gaussian_enhanced_dispatch/etc., same
// BasicPipelineConfig) -- no kernel source is duplicated or modified,
// only buffer/host-memory lifetime and dispatch-loop plumbing, which is
// what this experiment is measuring.
// --------------------------------------------------------------------------
class PersistentCudaPipeline {
public:
    PersistentCudaPipeline() = default;
    ~PersistentCudaPipeline() { release(); }

    PersistentCudaPipeline(const PersistentCudaPipeline&) = delete;
    PersistentCudaPipeline& operator=(const PersistentCudaPipeline&) = delete;

    struct RunTiming {
        float h2d_ms = 0.0f;
        float gaussian_ms = -1.0f;
        float median_ms = -1.0f;
        float sobel_ms = -1.0f;
        float laplacian_ms = -1.0f;
        float threshold_ms = -1.0f;
        float d2h_ms = 0.0f;
        float alloc_ms = 0.0f;        // GPU buffer allocation this call paid (0 if reused)
        float host_stage_ms = 0.0f;   // NumPy/pageable <-> pinned staging copy (0 if pageable path)
        bool used_persistent_buffers = false;
        bool used_pinned_memory = false;
        bool grew_gpu_buffers = false;    // true if this call had to allocate/grow the persistent GPU buffers
        bool grew_pinned_buffers = false; // true if this call had to allocate/grow the persistent pinned buffers

        float compute_ms() const {
            float total = 0.0f;
            for (float v : {gaussian_ms, median_ms, sobel_ms, laplacian_ms, threshold_ms}) {
                if (v >= 0.0f) total += v;
            }
            return total;
        }
        float total_ms() const { return alloc_ms + host_stage_ms + h2d_ms + compute_ms() + d2h_ms; }
    };

    // Mirrors run_basic_cuda_pipeline_batch()'s config contract exactly
    // (same BasicPipelineConfig, same semantics) -- the only new
    // parameters are the two memory-strategy flags this experiment
    // isolates.
    //
    // Section 20C spec item 17: explicit terminal lifecycle -- once
    // release() has been called, this object is permanently unusable.
    // Calling run() afterward throws std::runtime_error (mapped to a
    // Python RuntimeError by pybind11) rather than silently
    // self-healing by re-allocating; this matches item 33's "prefer
    // explicit lifecycle" recommendation -- a caller who releases early
    // must construct a fresh instance, not accidentally revive a
    // released one.
    std::pair<std::vector<uint8_t>, RunTiming> run(
        const uint8_t* host_batch, int batch_size, int height, int width, const BasicPipelineConfig& config,
        bool use_persistent_buffers, bool use_pinned_memory);

    // Explicit teardown of any persistent GPU/pinned-host allocations
    // (also invoked by the destructor). Idempotent -- safe to call
    // multiple times. After this call, run() throws std::runtime_error.
    void release();

    bool is_released() const { return released_; }

private:
    void ensure_gpu_capacity(int batch_size, int height, int width, bool& grew);
    void ensure_pinned_capacity(size_t bytes, bool& grew);

    std::unique_ptr<GpuImageBatch> persistent_buffer_a_;
    std::unique_ptr<GpuImageBatch> persistent_buffer_b_;
    std::unique_ptr<DeviceBuffer<float>> persistent_intermediate_;
    int gpu_capacity_batch_ = 0, gpu_capacity_height_ = 0, gpu_capacity_width_ = 0;

    uint8_t* pinned_input_ = nullptr;
    uint8_t* pinned_output_ = nullptr;
    size_t pinned_capacity_bytes_ = 0;

    bool released_ = false;
};

}  // namespace xray_cuda
