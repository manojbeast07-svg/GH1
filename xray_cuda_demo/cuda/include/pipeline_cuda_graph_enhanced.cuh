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
// Section 20E: EXPERIMENTAL CUDA Graph pipeline for the ENHANCED filter
// kernels (production variants only: Gaussian=Specialized,
// Median=Network3x3, Sobel=Specialized, Laplacian=Specialized,
// Threshold=Vectorized -- see research/enhanced_graph_capture_audit.md).
//
// WHY THIS FILE DUPLICATES KERNEL BODIES INSTEAD OF CALLING THE EXISTING
// *_enhanced_dispatch() KERNELS DIRECTLY (as Section 20D did for Basic):
// every Enhanced __global__ kernel (gaussian_specialized_h/v_kernel,
// median_network9_kernel, sobel_specialized_kernel,
// laplacian_specialized_kernel, threshold_vectorized_kernel,
// threshold_scalar_kernel) is defined inside an anonymous namespace in
// its own .cu file and is NOT declared in any .cuh header -- confirmed by
// direct inspection (research/enhanced_graph_capture_audit.md). Anonymous
// namespace gives these kernels internal (translation-unit-local) linkage,
// so they cannot be referenced, declared `extern`, or launched from any
// other .cu file, including this one -- unlike the Basic kernels (Section
// 20D), which ARE declared in shared headers. This is a hard C++/CUDA
// linkage constraint, not a design choice this section can route around
// without editing gaussian_enhanced.cu/median_enhanced.cu/sobel_enhanced.cu/
// laplacian_enhanced.cu/threshold_enhanced.cu themselves (forbidden by
// this section's scope).
//
// Per spec item 9's explicit fallback instruction, this file instead
// contains isolated, graph-compatible launch wrappers around VERIFIED
// textually-identical copies of the exact kernel bodies (see
// pipeline_cuda_graph_enhanced.cu's top-of-file comment for the line-by-
// line provenance of every duplicated kernel/helper, and
// research/enhanced_cuda_graph_decision.md's "Kernel equivalence proof"
// section for the verification methodology: exact source-line citation
// plus bit-exact numerical output against production, which is the
// stronger of the two proofs). No kernel's MATH is changed in any way --
// only which translation unit it is compiled into, and (for host-side
// orchestration only) the addition of an explicit stream parameter and
// removal of the enclosing dispatcher's cudaDeviceSynchronize()/
// cudaMemcpyToSymbol() calls, exactly mirroring Section 20D's treatment
// of the Basic kernels' dispatcher.
// --------------------------------------------------------------------------

struct EnhancedGraphConfig {
    bool gaussian_enabled = true;
    const float* gaussian_coeffs_1d = nullptr;  // length gaussian_kernel_size
    int gaussian_kernel_size = 0;                // one of {3,5,7,9} -- Specialized only

    bool median_enabled = true;  // Network3x3 -- kernel_size is always 3, no parameter

    bool sobel_enabled = true;
    int sobel_mode = 0;  // SobelMode as int -- selects the template instantiation (0..3)

    bool laplacian_enabled = true;
    const float* laplacian_coeffs = nullptr;  // kernel_size*kernel_size, row-major
    int laplacian_kernel_size = 0;              // one of {3,5} -- Specialized only
    float laplacian_scale = 1.0f;
    float laplacian_delta = 0.0f;

    bool threshold_enabled = true;
    uint8_t threshold_value = 128;
    uint8_t threshold_max_value = 255;
};

enum class EnhancedGraphScope {
    KernelsOnly = 0,
    FullPipeline = 1,
};

struct EnhancedGraphTiming {
    float alloc_ms = 0.0f;
    float h2d_ms = 0.0f;
    float compute_ms = 0.0f;
    float d2h_ms = 0.0f;
    float capture_ms = 0.0f;
    float instantiate_ms = 0.0f;
    float node_update_ms = 0.0f;
    bool used_graph = false;
    bool graph_cache_hit = false;
    std::string fallback_reason;

    float total_ms() const { return alloc_ms + h2d_ms + compute_ms + d2h_ms; }
};

class CudaGraphEnhancedPipeline {
public:
    CudaGraphEnhancedPipeline();
    ~CudaGraphEnhancedPipeline();
    CudaGraphEnhancedPipeline(const CudaGraphEnhancedPipeline&) = delete;
    CudaGraphEnhancedPipeline& operator=(const CudaGraphEnhancedPipeline&) = delete;

    std::pair<std::vector<uint8_t>, EnhancedGraphTiming> run(
        const uint8_t* host_batch, int batch_size, int height, int width,
        const EnhancedGraphConfig& config, bool use_graph, EnhancedGraphScope scope);

    void release();
    bool is_released() const { return released_; }
    size_t cache_size() const { return cache_.size(); }
    void clear_cache();

private:
    struct CacheKey {
        int batch_size, height, width, scope;
        bool gaussian_enabled; int gaussian_kernel_size;
        bool median_enabled;
        bool sobel_enabled; int sobel_mode;
        bool laplacian_enabled; int laplacian_kernel_size;
        uint32_t laplacian_scale_bits, laplacian_delta_bits;
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
        std::unique_ptr<DeviceBuffer<float>> intermediate;  // Gaussian H->V staging
        // Coefficients live in this file's own __constant__ symbols (g_gaussian_coeffs_1d /
        // g_laplacian_coeffs), not a per-entry device buffer -- these fields cache each symbol's
        // address (queried once via cudaGetSymbolAddress in build_entry(), a plain global constant
        // whose address never changes) so run() can patch the coefficient memcpy node's destination
        // correctly on every replay without a redundant driver query per call.
        void* gaussian_symbol_addr = nullptr;
        void* laplacian_symbol_addr = nullptr;
        cudaGraphNode_t h2d_node = nullptr;
        cudaGraphNode_t d2h_node = nullptr;
        cudaGraphNode_t gaussian_coeff_node = nullptr;
        cudaGraphNode_t laplacian_coeff_node = nullptr;
        bool final_is_a = false;
        size_t total_bytes = 0;
        std::vector<uint8_t> capture_scratch;
        ~GraphEntry();
        GraphEntry() = default;
        GraphEntry(const GraphEntry&) = delete;
        GraphEntry& operator=(const GraphEntry&) = delete;
    };

    GraphEntry* build_entry(
        const CacheKey& key, const EnhancedGraphConfig& config,
        int batch_size, int height, int width, EnhancedGraphScope scope,
        float& capture_ms, float& instantiate_ms, std::string& fallback_reason);

    cudaStream_t stream_ = nullptr;
    std::unordered_map<CacheKey, std::unique_ptr<GraphEntry>, CacheKeyHash> cache_;
    bool released_ = false;
    const uint8_t* host_batch_for_capture_ = nullptr;
};

}  // namespace xray_cuda
