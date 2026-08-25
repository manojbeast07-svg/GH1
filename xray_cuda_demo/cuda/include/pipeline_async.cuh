#pragma once

#include "cuda_common.cuh"
#include "gpu_image.cuh"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace xray_cuda {

// --------------------------------------------------------------------------
// Section 20F: EXPERIMENTAL multi-stream, double/triple-buffered async
// pipeline. Splits one call's batch into fixed-size chunks and pipelines
// H2D / compute / D2H across N buffer sets (N=2 double, N=3 triple) using
// three explicit non-default streams and CUDA events for cross-stream
// dependencies -- see research/async_pipeline_research.md for why this
// requires pinned host memory to achieve genuine overlap (measured, not
// assumed: this machine's GPU has asyncEngineCount=1, so H2D and D2H
// cannot be simultaneously in flight regardless of buffering depth --
// only each direction's overlap with compute is physically possible here).
//
// use_enhanced selects between the SAME production kernels used by
// run_basic_cuda_pipeline()/run_enhanced_cuda_pipeline(): Basic kernels
// are called directly (declared in their own headers, exactly as Section
// 20D's CudaGraphPipeline already does); Enhanced kernels are isolated,
// verified-identical copies (same technique and same source-line
// provenance as Section 20E's CudaGraphEnhancedPipeline -- Enhanced
// kernels have anonymous-namespace/internal linkage in their production
// .cu files and cannot be referenced from another translation unit).
// No kernel math is duplicated with any change; see
// pipeline_async.cu's top-of-file comment for exact provenance.
// --------------------------------------------------------------------------

struct AsyncPipelineConfig {
    bool use_enhanced = false;

    bool gaussian_enabled = true;
    const float* gaussian_coeffs = nullptr;     // Basic: k*k row-major 2-D
    const float* gaussian_coeffs_1d = nullptr;  // Enhanced: length k, 1-D separable
    int gaussian_kernel_size = 0;

    bool median_enabled = true;
    int median_kernel_size = 0;  // Basic only; Enhanced always uses Network3x3 (k=3, no parameter)

    bool sobel_enabled = true;
    int sobel_mode = 0;

    bool laplacian_enabled = true;
    const float* laplacian_coeffs = nullptr;
    int laplacian_kernel_size = 0;
    float laplacian_scale = 1.0f;
    float laplacian_delta = 0.0f;

    bool threshold_enabled = true;
    uint8_t threshold_value = 128;
    uint8_t threshold_max_value = 255;
};

struct AsyncPipelineTiming {
    float alloc_ms = 0.0f;
    float host_stage_ms = 0.0f;  // total pageable->pinned staging time across all chunks (0 if not used)
    float total_ms = 0.0f;       // wall-clock HostTimer around the whole call -- the ONLY trustworthy total
                                   // when stages overlap (spec item 28: never sum per-stage times and call
                                   // it the wall-clock time)
    // Diagnostic-only, NOT a substitute for total_ms: sums of per-chunk async-issue-to-completion
    // spans on each stream. These do NOT account for overlap and must never be added together.
    float h2d_stream_span_ms = 0.0f;
    float compute_stream_span_ms = 0.0f;
    float d2h_stream_span_ms = 0.0f;
    // Measured (not assumed) overlap evidence via CUDA events (spec item 29): how much of the
    // compute stream's active span coincides with the H2D stream's active span, and with the D2H
    // stream's active span, in milliseconds.
    float h2d_compute_overlap_ms = 0.0f;
    float compute_d2h_overlap_ms = 0.0f;

    int num_chunks = 0;
    int chunk_size_used = 0;
    int num_buffers_used = 0;
    bool used_pinned_staging = false;
};

class AsyncCudaPipeline {
public:
    AsyncCudaPipeline();
    ~AsyncCudaPipeline();
    AsyncCudaPipeline(const AsyncCudaPipeline&) = delete;
    AsyncCudaPipeline& operator=(const AsyncCudaPipeline&) = delete;

    // Runs the full batch, split into chunks of `chunk_size` (the last chunk may be smaller),
    // pipelined across `num_buffers` buffer sets (2=double, 3=triple buffering). When
    // use_pinned_staging is true, each chunk's host data is first copied (synchronously, on the
    // calling thread, timed as host_stage_ms) into a pre-allocated pinned staging buffer before
    // being handed to cudaMemcpyAsync -- required for genuine H2D/compute overlap per this
    // section's research. When false (the default -- matches the real application's actual data
    // path), H2D is issued directly from the caller's pageable host_batch pointer.
    std::pair<std::vector<uint8_t>, AsyncPipelineTiming> run(
        const uint8_t* host_batch, int batch_size, int height, int width,
        const AsyncPipelineConfig& config, int chunk_size, int num_buffers, bool use_pinned_staging);

    void release();
    bool is_released() const { return released_; }

private:
    struct BufferSet {
        std::unique_ptr<GpuImageBatch> buf_a, buf_b;
        std::unique_ptr<DeviceBuffer<float>> intermediate;  // Enhanced Gaussian H->V staging only
        // Basic-mode coefficient buffers, allocated once per buffer set (not per chunk) and
        // re-uploaded via cudaMemcpyAsync on compute_stream_ before each chunk's kernel that
        // reads them -- stream-ordering alone (same stream, upload issued before the consuming
        // kernel) guarantees correctness with no host-side synchronization, which matters here
        // because ANY sync inside the per-chunk hot path would collapse the whole pipeline back
        // to sequential execution. (Enhanced mode needs no equivalent -- its coefficients live in
        // this file's own __constant__ symbols, uploaded the same stream-ordered way.)
        std::unique_ptr<DeviceBuffer<float>> gaussian_coeffs_dev;
        std::unique_ptr<DeviceBuffer<float>> laplacian_coeffs_dev;
        uint8_t* pinned_input = nullptr;
        uint8_t* pinned_output = nullptr;
        size_t pinned_capacity_bytes = 0;
        cudaEvent_t h2d_done = nullptr;
        cudaEvent_t compute_done = nullptr;
        cudaEvent_t d2h_done = nullptr;
        bool d2h_done_valid = false;  // false until this set's first D2H has been recorded
        ~BufferSet();
    };

    cudaStream_t h2d_stream_ = nullptr;
    cudaStream_t compute_stream_ = nullptr;
    cudaStream_t d2h_stream_ = nullptr;
    bool released_ = false;
};

}  // namespace xray_cuda
