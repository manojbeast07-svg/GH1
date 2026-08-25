#pragma once

#include "cuda_common.cuh"
#include "gpu_image.cuh"
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xray_cuda {

// --------------------------------------------------------------------------
// Section 20D: EXPERIMENTAL CUDA Graph pipeline.
//
// Scope note (see cuda/src/pipeline_cuda_graph.cu's file header for the
// full research finding): this experiment captures the five BASIC
// (non-Enhanced) __global__ kernels only. The *_enhanced_dispatch()
// functions used by the Enhanced pipeline are NOT graph-capturable as
// currently written -- they launch on the implicit default stream
// (not this class's explicit capturing stream) and call synchronous
// cudaMemcpy/cudaMemcpyToSymbol internally. Making them capturable
// would require modifying those five shared dispatch functions, which
// is out of this section's safe-modification scope (they are also used
// by production run_basic_cuda_pipeline_batch()). run(..., use_graph=true)
// with any *_use_enhanced-equivalent request is therefore not offered by
// this class at all -- callers wanting Enhanced always get the documented
// non-graph fallback path (see GraphPipelineTiming::used_graph).
//
// No kernel (__global__ function) is modified or duplicated with
// different math -- this file only adds new host-side orchestration
// around the *existing* gaussian_basic_kernel/median_basic_kernel/
// sobel_basic_kernel/laplacian_basic_kernel/threshold_basic_kernel.
// --------------------------------------------------------------------------

// Mirrors BasicPipelineConfig's basic-kernel-relevant fields (no
// *_use_enhanced fields -- see scope note above).
struct GraphPipelineConfig {
    bool gaussian_enabled = true;
    const float* gaussian_coeffs = nullptr;  // kernel_size*kernel_size, row-major
    int gaussian_kernel_size = 0;

    bool median_enabled = true;
    int median_kernel_size = 0;

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

// What the captured graph spans. KernelsOnly leaves H2D/D2H as plain
// cudaMemcpyAsync calls outside the graph (spec Variant B); FullPipeline
// captures H2D + all enabled kernels + D2H in one graph (spec Variant C).
enum class GraphScope {
    KernelsOnly = 0,
    FullPipeline = 1,
};

struct GraphPipelineTiming {
    float alloc_ms = 0.0f;        // one-time GPU buffer allocation cost (0 on a cache hit)
    float h2d_ms = 0.0f;          // 0.0f when captured inside the graph (folded into compute_ms)
    float compute_ms = 0.0f;      // kernel time (KernelsOnly) or whole-graph/whole-sequence time (FullPipeline)
    float d2h_ms = 0.0f;          // 0.0f when captured inside the graph
    float capture_ms = 0.0f;      // cudaStreamBeginCapture..EndCapture cost; 0 on a cache hit
    float instantiate_ms = 0.0f;  // cudaGraphInstantiate cost; 0 on a cache hit
    float node_update_ms = 0.0f;  // cudaGraphExecMemcpyNodeSetParams1D cost for this replay
    bool used_graph = false;      // false => fell back to the non-graph path (capture unsupported or use_graph=false)
    bool graph_cache_hit = false;
    std::string fallback_reason;  // non-empty only when used_graph is false because capture failed

    float total_ms() const { return alloc_ms + h2d_ms + compute_ms + d2h_ms; }
};

class CudaGraphPipeline {
public:
    CudaGraphPipeline();
    ~CudaGraphPipeline();
    CudaGraphPipeline(const CudaGraphPipeline&) = delete;
    CudaGraphPipeline& operator=(const CudaGraphPipeline&) = delete;

    // Runs the Basic-kernel pipeline on one batch. When use_graph is
    // true, looks up (or captures+instantiates+caches) a graph for this
    // exact (shape, enabled-mask, kernel sizes, scalar parameter)
    // configuration, then replays it after patching this call's
    // host-memory addresses into the graph's memcpy nodes. When
    // use_graph is false, or when capture fails for this configuration,
    // runs the equivalent sequence as plain (uncaptured) stream
    // operations instead -- always produces a result, never throws
    // because graphs were unavailable (see GraphPipelineTiming::fallback_reason).
    std::pair<std::vector<uint8_t>, GraphPipelineTiming> run(
        const uint8_t* host_batch, int batch_size, int height, int width,
        const GraphPipelineConfig& config, bool use_graph, GraphScope scope);

    void release();
    bool is_released() const { return released_; }
    size_t cache_size() const { return cache_.size(); }
    void clear_cache();

private:
    struct CacheKey {
        int batch_size, height, width;
        int scope;
        bool gaussian_enabled; int gaussian_kernel_size;
        bool median_enabled; int median_kernel_size;
        bool sobel_enabled; int sobel_mode;
        bool laplacian_enabled; int laplacian_kernel_size;
        uint32_t laplacian_scale_bits; uint32_t laplacian_delta_bits;
        bool threshold_enabled; uint8_t threshold_value; uint8_t threshold_max_value;

        bool operator==(const CacheKey& o) const;
    };
    struct CacheKeyHash {
        size_t operator()(const CacheKey& k) const;
    };

    struct GraphEntry {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        std::unique_ptr<GpuImageBatch> buf_a, buf_b;
        std::unique_ptr<DeviceBuffer<float>> gaussian_coeffs_dev, laplacian_coeffs_dev;
        cudaGraphNode_t h2d_node = nullptr;              // FullPipeline only
        cudaGraphNode_t d2h_node = nullptr;               // FullPipeline only
        cudaGraphNode_t gaussian_coeff_node = nullptr;    // only if gaussian_enabled
        cudaGraphNode_t laplacian_coeff_node = nullptr;   // only if laplacian_enabled
        bool final_is_a = false;                          // which buffer holds the result after all enabled stages
        size_t total_bytes = 0;
        std::vector<uint8_t> capture_scratch;  // valid D2H host destination used only during capture; real
                                                // per-call output buffers are patched in before each replay
        ~GraphEntry();
        GraphEntry() = default;
        GraphEntry(const GraphEntry&) = delete;
        GraphEntry& operator=(const GraphEntry&) = delete;
    };

    GraphEntry* build_entry(
        const CacheKey& key, const GraphPipelineConfig& config,
        int batch_size, int height, int width, GraphScope scope,
        float& capture_ms, float& instantiate_ms, std::string& fallback_reason);

    cudaStream_t stream_ = nullptr;
    std::unordered_map<CacheKey, std::unique_ptr<GraphEntry>, CacheKeyHash> cache_;
    bool released_ = false;
    const uint8_t* host_batch_for_capture_ = nullptr;  // valid host pointer used only as the capture-time
                                                         // placeholder for the H2D node; patched before every replay
};

}  // namespace xray_cuda
