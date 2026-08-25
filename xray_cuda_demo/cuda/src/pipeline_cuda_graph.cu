// Section 20D: EXPERIMENTAL CUDA Graph pipeline.
//
// RESEARCH FINDING that shaped this file's scope (spec item 3 -- "determine
// exactly what can be captured... do not rely on assumptions"): neither of
// this project's two existing dispatch paths is graph-capturable as written.
//
//   1. run_basic_cuda_pipeline_batch()'s own "basic kernel" branch (used
//      when a stage's *_use_enhanced is false) calls cudaDeviceSynchronize()
//      after every single stage (pipeline_basic.cu:71,99,127,161,184) to
//      read back that stage's CudaTimer immediately. cudaDeviceSynchronize
//      is illegal during stream capture (it is not a stream-ordered
//      operation) -- capturing that exact code path would fail immediately
//      at the first stage.
//   2. The five *_enhanced_dispatch() functions (gaussian_enhanced_dispatch,
//      median_enhanced_dispatch, sobel_enhanced_dispatch,
//      laplacian_enhanced_dispatch, threshold_enhanced_dispatch) launch
//      their kernels on the implicit default stream (they take no
//      cudaStream_t parameter at all) rather than on an explicit capturing
//      stream, and some variants call synchronous cudaMemcpy /
//      cudaMemcpyToSymbol internally for per-call coefficient upload
//      (confirmed by inspection: gaussian_enhanced.cu:302,
//      laplacian_enhanced.cu:323, sobel_enhanced.cu:286-287). A kernel
//      launched on a stream other than the one being captured is not
//      captured into the graph at all (no error -- it just silently runs
//      outside the graph), and a synchronous cudaMemcpy issued on the
//      capturing stream aborts capture outright.
//
// Making either path capturable would mean modifying pipeline_basic.cu's
// per-stage synchronization and/or all five shared *_enhanced_dispatch()
// signatures/internals -- both are used by the production pipeline and by
// Section 20B/20C's PersistentCudaPipeline, so changing them is out of this
// section's safe-modification scope (and no kernel __global__ function
// needs to change for this experiment's own hypothesis to be tested).
//
// This file therefore captures a *new*, isolated call sequence built
// directly on the five existing, unmodified basic __global__ kernels
// (gaussian_basic_kernel, median_basic_kernel, sobel_basic_kernel,
// laplacian_basic_kernel, threshold_basic_kernel) -- explicit stream,
// no per-stage sync, no cudaMemcpyToSymbol. Enhanced-pipeline graph
// support is consequently not offered by this class; run() with
// use_graph=true always executes the Basic-kernel math (see
// pipeline_cuda_graph.cuh's file header). Compared against the true
// stateless production pipeline (run_basic_cuda_pipeline_batch with every
// stage's use_enhanced=false), this isolates exactly the graph-vs-no-graph
// variable the research question is about.
//
// Buffer-address handling: a captured cudaMemcpyAsync node freezes the
// host/device pointers current at capture time. Every replay in this
// project passes a *different* NumPy array (different host address) and
// (for FullPipeline scope) needs a *different* output buffer, so the H2D
// source, D2H destination, and coefficient-upload source addresses are
// patched into the already-instantiated graph via
// cudaGraphExecMemcpyNodeSetParams1D() before every cudaGraphLaunch() --
// this is the standard, documented mechanism for "same structure, new
// pointers" graph reuse and is not a form of persistent-buffer reuse
// (device buffer addresses inside one cache entry stay fixed by
// construction, which is what makes a captured graph replayable at all;
// see cudaGraphExecMemcpyNodeSetParams1D in cuda_runtime_api.h).
// Any other configuration change (batch size, resolution, enabled-stage
// mask, kernel sizes, or scalar parameters baked directly into a kernel's
// launch arguments such as threshold_value/sobel_mode/laplacian_scale/
// laplacian_delta) looks up a different cache entry, recapturing a fresh
// graph on a miss rather than patching kernel-node arguments in place
// (cudaGraphExecKernelNodeSetParams would reduce recapture frequency
// further but was not implemented in this section -- a disclosed,
// deliberate scope limit, not an oversight).

#include "pipeline_cuda_graph.cuh"
#include "gaussian_basic.cuh"
#include "laplacian_basic.cuh"
#include "median_basic.cuh"
#include "sobel_basic.cuh"
#include "threshold_basic.cuh"
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace xray_cuda {

namespace {

// Host-side (chrono) timer for capture/instantiate/allocation costs --
// these are not GPU work, so a CudaTimer (cudaEvent-based) would not be
// the right tool; mirrors pipeline_experimental.cu's own HostTimer.
class HostTimer {
public:
    void start() { start_ = std::chrono::steady_clock::now(); }
    void stop() { stop_ = std::chrono::steady_clock::now(); }
    float elapsed_ms() const {
        return std::chrono::duration<float, std::milli>(stop_ - start_).count();
    }

private:
    std::chrono::steady_clock::time_point start_, stop_;
};

size_t hash_combine(size_t seed, size_t v) {
    return seed ^ (v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

}  // namespace

bool CudaGraphPipeline::CacheKey::operator==(const CacheKey& o) const {
    return batch_size == o.batch_size && height == o.height && width == o.width && scope == o.scope &&
           gaussian_enabled == o.gaussian_enabled && gaussian_kernel_size == o.gaussian_kernel_size &&
           median_enabled == o.median_enabled && median_kernel_size == o.median_kernel_size &&
           sobel_enabled == o.sobel_enabled && sobel_mode == o.sobel_mode &&
           laplacian_enabled == o.laplacian_enabled && laplacian_kernel_size == o.laplacian_kernel_size &&
           laplacian_scale_bits == o.laplacian_scale_bits && laplacian_delta_bits == o.laplacian_delta_bits &&
           threshold_enabled == o.threshold_enabled && threshold_value == o.threshold_value &&
           threshold_max_value == o.threshold_max_value;
}

size_t CudaGraphPipeline::CacheKeyHash::operator()(const CacheKey& k) const {
    size_t h = 0;
    for (int v : {k.batch_size, k.height, k.width, k.scope, static_cast<int>(k.gaussian_enabled),
                   k.gaussian_kernel_size, static_cast<int>(k.median_enabled), k.median_kernel_size,
                   static_cast<int>(k.sobel_enabled), k.sobel_mode, static_cast<int>(k.laplacian_enabled),
                   k.laplacian_kernel_size, static_cast<int>(k.threshold_enabled),
                   static_cast<int>(k.threshold_value), static_cast<int>(k.threshold_max_value)}) {
        h = hash_combine(h, static_cast<size_t>(v));
    }
    h = hash_combine(h, k.laplacian_scale_bits);
    h = hash_combine(h, k.laplacian_delta_bits);
    return h;
}

CudaGraphPipeline::GraphEntry::~GraphEntry() {
    if (exec) cudaGraphExecDestroy(exec);
    if (graph) cudaGraphDestroy(graph);
}

CudaGraphPipeline::CudaGraphPipeline() {
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

CudaGraphPipeline::~CudaGraphPipeline() {
    release();
}

void CudaGraphPipeline::release() {
    if (released_) return;
    cache_.clear();
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    released_ = true;
}

void CudaGraphPipeline::clear_cache() {
    cache_.clear();
}

namespace {

uint32_t float_bits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return bits;
}

}  // namespace

CudaGraphPipeline::GraphEntry* CudaGraphPipeline::build_entry(
    const CacheKey& key, const GraphPipelineConfig& config,
    int batch_size, int height, int width, GraphScope scope,
    float& capture_ms, float& instantiate_ms, std::string& fallback_reason) {
    auto entry = std::make_unique<GraphEntry>();
    entry->buf_a = std::make_unique<GpuImageBatch>(batch_size, height, width);
    entry->buf_b = std::make_unique<GpuImageBatch>(batch_size, height, width);
    entry->total_bytes = entry->buf_a->nbytes();
    entry->capture_scratch.resize(entry->total_bytes);

    if (config.gaussian_enabled) {
        size_t n = static_cast<size_t>(config.gaussian_kernel_size) * config.gaussian_kernel_size;
        entry->gaussian_coeffs_dev = std::make_unique<DeviceBuffer<float>>(n);
    }
    if (config.laplacian_enabled) {
        size_t n = static_cast<size_t>(config.laplacian_kernel_size) * config.laplacian_kernel_size;
        entry->laplacian_coeffs_dev = std::make_unique<DeviceBuffer<float>>(n);
    }

    int enabled_count = (config.gaussian_enabled ? 1 : 0) + (config.median_enabled ? 1 : 0) +
                         (config.sobel_enabled ? 1 : 0) + (config.laplacian_enabled ? 1 : 0) +
                         (config.threshold_enabled ? 1 : 0);
    entry->final_is_a = (enabled_count % 2 == 0);

    dim3 block = default_block_dim();
    dim3 grid2d = compute_launch_grid(width, height, block);
    dim3 grid(grid2d.x, grid2d.y, static_cast<unsigned int>(batch_size));

    HostTimer capture_timer;
    capture_timer.start();

    cudaError_t begin_err = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal);
    if (begin_err != cudaSuccess) {
        fallback_reason = std::string("cudaStreamBeginCapture failed: ") + cudaGetErrorString(begin_err);
        return nullptr;
    }

    bool capture_body_ok = true;
    std::string capture_body_error;
    try {
        GpuImageBatch* current = entry->buf_a.get();
        GpuImageBatch* other = entry->buf_b.get();

        if (scope == GraphScope::FullPipeline) {
            CUDA_CHECK(cudaMemcpyAsync(current->data(), host_batch_for_capture_,
                                        entry->total_bytes, cudaMemcpyHostToDevice, stream_));
        }

        if (config.gaussian_enabled) {
            size_t n = static_cast<size_t>(config.gaussian_kernel_size) * config.gaussian_kernel_size;
            CUDA_CHECK(cudaMemcpyAsync(entry->gaussian_coeffs_dev->get(), config.gaussian_coeffs,
                                        n * sizeof(float), cudaMemcpyHostToDevice, stream_));
            gaussian_basic_kernel<<<grid, block, 0, stream_>>>(
                current->data(), other->data(), width, height,
                entry->gaussian_coeffs_dev->get(), config.gaussian_kernel_size);
            CUDA_CHECK_LAST_ERROR();
            std::swap(current, other);
        }
        if (config.median_enabled) {
            median_basic_kernel<<<grid, block, 0, stream_>>>(
                current->data(), other->data(), width, height, config.median_kernel_size);
            CUDA_CHECK_LAST_ERROR();
            std::swap(current, other);
        }
        if (config.sobel_enabled) {
            sobel_basic_kernel<<<grid, block, 0, stream_>>>(
                current->data(), other->data(), width, height, config.sobel_mode);
            CUDA_CHECK_LAST_ERROR();
            std::swap(current, other);
        }
        if (config.laplacian_enabled) {
            size_t n = static_cast<size_t>(config.laplacian_kernel_size) * config.laplacian_kernel_size;
            CUDA_CHECK(cudaMemcpyAsync(entry->laplacian_coeffs_dev->get(), config.laplacian_coeffs,
                                        n * sizeof(float), cudaMemcpyHostToDevice, stream_));
            laplacian_basic_kernel<<<grid, block, 0, stream_>>>(
                current->data(), other->data(), width, height,
                entry->laplacian_coeffs_dev->get(), config.laplacian_kernel_size,
                config.laplacian_scale, config.laplacian_delta);
            CUDA_CHECK_LAST_ERROR();
            std::swap(current, other);
        }
        if (config.threshold_enabled) {
            threshold_basic_kernel<<<grid, block, 0, stream_>>>(
                current->data(), other->data(), width, height,
                config.threshold_value, config.threshold_max_value);
            CUDA_CHECK_LAST_ERROR();
            std::swap(current, other);
        }

        if (scope == GraphScope::FullPipeline) {
            CUDA_CHECK(cudaMemcpyAsync(entry->capture_scratch.data(), current->data(),
                                        entry->total_bytes, cudaMemcpyDeviceToHost, stream_));
        }
    } catch (const std::exception& e) {
        capture_body_ok = false;
        capture_body_error = e.what();
    }

    cudaError_t end_err = cudaStreamEndCapture(stream_, &entry->graph);
    if (!capture_body_ok) {
        if (entry->graph) {
            cudaGraphDestroy(entry->graph);
            entry->graph = nullptr;
        }
        fallback_reason = "capture body raised: " + capture_body_error;
        return nullptr;
    }
    if (end_err != cudaSuccess) {
        fallback_reason = std::string("cudaStreamEndCapture failed: ") + cudaGetErrorString(end_err);
        return nullptr;
    }
    capture_timer.stop();
    capture_ms = capture_timer.elapsed_ms();

    size_t num_nodes = 0;
    CUDA_CHECK(cudaGraphGetNodes(entry->graph, nullptr, &num_nodes));
    std::vector<cudaGraphNode_t> nodes(num_nodes);
    CUDA_CHECK(cudaGraphGetNodes(entry->graph, nodes.data(), &num_nodes));
    for (auto node : nodes) {
        cudaGraphNodeType type;
        CUDA_CHECK(cudaGraphNodeGetType(node, &type));
        if (type != cudaGraphNodeTypeMemcpy) continue;
        cudaMemcpy3DParms params = {};
        CUDA_CHECK(cudaGraphMemcpyNodeGetParams(node, &params));
        void* dst_ptr = params.dstPtr.ptr;
        if (scope == GraphScope::FullPipeline && dst_ptr == entry->buf_a->data()) {
            entry->h2d_node = node;
        } else if (scope == GraphScope::FullPipeline && dst_ptr == entry->capture_scratch.data()) {
            entry->d2h_node = node;
        } else if (entry->gaussian_coeffs_dev && dst_ptr == entry->gaussian_coeffs_dev->get()) {
            entry->gaussian_coeff_node = node;
        } else if (entry->laplacian_coeffs_dev && dst_ptr == entry->laplacian_coeffs_dev->get()) {
            entry->laplacian_coeff_node = node;
        }
    }

    HostTimer inst_timer;
    inst_timer.start();
    cudaError_t inst_err = cudaGraphInstantiate(&entry->exec, entry->graph, 0);
    inst_timer.stop();
    if (inst_err != cudaSuccess) {
        fallback_reason = std::string("cudaGraphInstantiate failed: ") + cudaGetErrorString(inst_err);
        cudaGraphDestroy(entry->graph);
        entry->graph = nullptr;
        return nullptr;
    }
    instantiate_ms = inst_timer.elapsed_ms();

    GraphEntry* raw = entry.get();
    cache_.emplace(key, std::move(entry));
    return raw;
}

std::pair<std::vector<uint8_t>, GraphPipelineTiming> CudaGraphPipeline::run(
    const uint8_t* host_batch, int batch_size, int height, int width,
    const GraphPipelineConfig& config, bool use_graph, GraphScope scope) {
    if (released_) {
        throw std::runtime_error("CudaGraphPipeline::run() called after release().");
    }

    GraphPipelineTiming timing;
    const size_t total_bytes =
        static_cast<size_t>(batch_size) * static_cast<size_t>(height) * static_cast<size_t>(width);
    std::vector<uint8_t> host_output(total_bytes);

    if (!use_graph) {
        // Non-graph reference path: same kernels, same explicit stream_,
        // but no capture -- lets the benchmark isolate "graph vs no graph"
        // using otherwise-identical host code (a second, independent
        // baseline alongside the real production pipeline).
        HostTimer alloc_timer;
        alloc_timer.start();
        GpuImageBatch buf_a(batch_size, height, width);
        GpuImageBatch buf_b(batch_size, height, width);
        alloc_timer.stop();
        timing.alloc_ms = alloc_timer.elapsed_ms();

        CudaTimer h2d_timer;
        h2d_timer.start(stream_);
        CUDA_CHECK(cudaMemcpyAsync(buf_a.data(), host_batch, total_bytes, cudaMemcpyHostToDevice, stream_));
        h2d_timer.stop(stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        timing.h2d_ms = h2d_timer.elapsed_ms();

        GpuImageBatch* current = &buf_a;
        GpuImageBatch* other = &buf_b;
        dim3 block = default_block_dim();
        dim3 grid2d = compute_launch_grid(width, height, block);
        dim3 grid(grid2d.x, grid2d.y, static_cast<unsigned int>(batch_size));

        CudaTimer compute_timer;
        compute_timer.start(stream_);
        if (config.gaussian_enabled) {
            size_t n = static_cast<size_t>(config.gaussian_kernel_size) * config.gaussian_kernel_size;
            DeviceBuffer<float> coeffs(n);
            coeffs.upload(config.gaussian_coeffs, n, stream_);
            gaussian_basic_kernel<<<grid, block, 0, stream_>>>(
                current->data(), other->data(), width, height, coeffs.get(), config.gaussian_kernel_size);
            CUDA_CHECK_LAST_ERROR();
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            std::swap(current, other);
        }
        if (config.median_enabled) {
            median_basic_kernel<<<grid, block, 0, stream_>>>(
                current->data(), other->data(), width, height, config.median_kernel_size);
            CUDA_CHECK_LAST_ERROR();
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            std::swap(current, other);
        }
        if (config.sobel_enabled) {
            sobel_basic_kernel<<<grid, block, 0, stream_>>>(
                current->data(), other->data(), width, height, config.sobel_mode);
            CUDA_CHECK_LAST_ERROR();
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            std::swap(current, other);
        }
        if (config.laplacian_enabled) {
            size_t n = static_cast<size_t>(config.laplacian_kernel_size) * config.laplacian_kernel_size;
            DeviceBuffer<float> coeffs(n);
            coeffs.upload(config.laplacian_coeffs, n, stream_);
            laplacian_basic_kernel<<<grid, block, 0, stream_>>>(
                current->data(), other->data(), width, height,
                coeffs.get(), config.laplacian_kernel_size, config.laplacian_scale, config.laplacian_delta);
            CUDA_CHECK_LAST_ERROR();
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            std::swap(current, other);
        }
        if (config.threshold_enabled) {
            threshold_basic_kernel<<<grid, block, 0, stream_>>>(
                current->data(), other->data(), width, height,
                config.threshold_value, config.threshold_max_value);
            CUDA_CHECK_LAST_ERROR();
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            std::swap(current, other);
        }
        compute_timer.stop(stream_);
        timing.compute_ms = compute_timer.elapsed_ms();

        CudaTimer d2h_timer;
        d2h_timer.start(stream_);
        CUDA_CHECK(cudaMemcpyAsync(host_output.data(), current->data(), total_bytes, cudaMemcpyDeviceToHost, stream_));
        d2h_timer.stop(stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        timing.d2h_ms = d2h_timer.elapsed_ms();

        timing.used_graph = false;
        timing.fallback_reason = "";  // not a failure -- caller explicitly requested use_graph=false
        return {std::move(host_output), timing};
    }

    CacheKey key{};
    key.batch_size = batch_size;
    key.height = height;
    key.width = width;
    key.scope = static_cast<int>(scope);
    key.gaussian_enabled = config.gaussian_enabled;
    key.gaussian_kernel_size = config.gaussian_enabled ? config.gaussian_kernel_size : 0;
    key.median_enabled = config.median_enabled;
    key.median_kernel_size = config.median_enabled ? config.median_kernel_size : 0;
    key.sobel_enabled = config.sobel_enabled;
    key.sobel_mode = config.sobel_enabled ? config.sobel_mode : 0;
    key.laplacian_enabled = config.laplacian_enabled;
    key.laplacian_kernel_size = config.laplacian_enabled ? config.laplacian_kernel_size : 0;
    key.laplacian_scale_bits = config.laplacian_enabled ? float_bits(config.laplacian_scale) : 0;
    key.laplacian_delta_bits = config.laplacian_enabled ? float_bits(config.laplacian_delta) : 0;
    key.threshold_enabled = config.threshold_enabled;
    key.threshold_value = config.threshold_enabled ? config.threshold_value : 0;
    key.threshold_max_value = config.threshold_enabled ? config.threshold_max_value : 0;

    auto it = cache_.find(key);
    GraphEntry* entry = nullptr;
    if (it != cache_.end()) {
        entry = it->second.get();
        timing.graph_cache_hit = true;
    } else {
        host_batch_for_capture_ = host_batch;
        float capture_ms = 0.0f, instantiate_ms = 0.0f;
        std::string fallback_reason;
        entry = build_entry(key, config, batch_size, height, width, scope, capture_ms, instantiate_ms, fallback_reason);
        if (entry == nullptr) {
            // Capture or instantiation failed for this configuration --
            // fall back to the non-graph path and report why (spec item
            // 31: "the failure must be visible in diagnostics, never
            // silently pretend a graph was used").
            auto result = run(host_batch, batch_size, height, width, config, /*use_graph=*/false, scope);
            result.second.used_graph = false;
            result.second.fallback_reason = fallback_reason;
            return result;
        }
        timing.graph_cache_hit = false;
        timing.capture_ms = capture_ms;
        timing.instantiate_ms = instantiate_ms;
    }

    HostTimer update_timer;
    update_timer.start();
    if (scope == GraphScope::FullPipeline) {
        CUDA_CHECK(cudaGraphExecMemcpyNodeSetParams1D(
            entry->exec, entry->h2d_node, entry->buf_a->data(), host_batch, entry->total_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaGraphExecMemcpyNodeSetParams1D(
            entry->exec, entry->d2h_node, host_output.data(),
            entry->final_is_a ? entry->buf_a->data() : entry->buf_b->data(),
            entry->total_bytes, cudaMemcpyDeviceToHost));
    }
    if (entry->gaussian_coeff_node) {
        size_t n = static_cast<size_t>(config.gaussian_kernel_size) * config.gaussian_kernel_size;
        CUDA_CHECK(cudaGraphExecMemcpyNodeSetParams1D(
            entry->exec, entry->gaussian_coeff_node, entry->gaussian_coeffs_dev->get(),
            config.gaussian_coeffs, n * sizeof(float), cudaMemcpyHostToDevice));
    }
    if (entry->laplacian_coeff_node) {
        size_t n = static_cast<size_t>(config.laplacian_kernel_size) * config.laplacian_kernel_size;
        CUDA_CHECK(cudaGraphExecMemcpyNodeSetParams1D(
            entry->exec, entry->laplacian_coeff_node, entry->laplacian_coeffs_dev->get(),
            config.laplacian_coeffs, n * sizeof(float), cudaMemcpyHostToDevice));
    }
    update_timer.stop();
    timing.node_update_ms = update_timer.elapsed_ms();

    if (scope == GraphScope::KernelsOnly) {
        CudaTimer h2d_timer;
        h2d_timer.start(stream_);
        CUDA_CHECK(cudaMemcpyAsync(entry->buf_a->data(), host_batch, entry->total_bytes, cudaMemcpyHostToDevice, stream_));
        h2d_timer.stop(stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        timing.h2d_ms = h2d_timer.elapsed_ms();
    }

    CudaTimer launch_timer;
    launch_timer.start(stream_);
    CUDA_CHECK(cudaGraphLaunch(entry->exec, stream_));
    launch_timer.stop(stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    timing.compute_ms = launch_timer.elapsed_ms();

    if (scope == GraphScope::KernelsOnly) {
        CudaTimer d2h_timer;
        d2h_timer.start(stream_);
        GpuImageBatch& final_buf = entry->final_is_a ? *entry->buf_a : *entry->buf_b;
        CUDA_CHECK(cudaMemcpyAsync(host_output.data(), final_buf.data(), entry->total_bytes, cudaMemcpyDeviceToHost, stream_));
        d2h_timer.stop(stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        timing.d2h_ms = d2h_timer.elapsed_ms();
    }
    // FullPipeline: the D2H node was patched above to write directly into
    // host_output.data(), so cudaGraphLaunch()+sync already filled it --
    // nothing further to copy.

    timing.used_graph = true;
    return {std::move(host_output), timing};
}

}  // namespace xray_cuda
