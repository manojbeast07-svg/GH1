#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <sstream>

// Throws a std::runtime_error with file/line/call/CUDA-error-string context
// whenever a CUDA runtime API call does not return cudaSuccess.
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _cuda_check_status = (call);                            \
        if (_cuda_check_status != cudaSuccess) {                            \
            std::ostringstream _cuda_check_oss;                            \
            _cuda_check_oss << "CUDA error: " << cudaGetErrorString(_cuda_check_status) \
                             << " (code " << static_cast<int>(_cuda_check_status) << ")" \
                             << " at " << __FILE__ << ":" << __LINE__          \
                             << " in call: " << #call;                        \
            throw std::runtime_error(_cuda_check_oss.str());                 \
        }                                                                    \
    } while (0)

// Checks for asynchronous errors raised by the most recent kernel launch.
// Call immediately after a kernel launch (and again after a sync during
// testing) to catch launch-configuration and in-kernel errors early.
#define CUDA_CHECK_LAST_ERROR()                                              \
    do {                                                                     \
        cudaError_t _cuda_last_status = cudaGetLastError();                 \
        if (_cuda_last_status != cudaSuccess) {                             \
            std::ostringstream _cuda_last_oss;                             \
            _cuda_last_oss << "CUDA kernel launch error: "                  \
                            << cudaGetErrorString(_cuda_last_status)         \
                            << " (code " << static_cast<int>(_cuda_last_status) << ")" \
                            << " at " << __FILE__ << ":" << __LINE__;         \
            throw std::runtime_error(_cuda_last_oss.str());                  \
        }                                                                    \
    } while (0)

namespace xray_cuda {

// True if at least one usable CUDA device is present on the system.
bool cuda_is_available();

// Raw device properties gathered from cudaGetDeviceProperties for device 0
// (the only device this project targets for now).
struct DeviceInfo {
    std::string name;
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    size_t total_global_mem_bytes = 0;
    size_t shared_mem_per_block_bytes = 0;
    int registers_per_block = 0;
    int max_threads_per_block = 0;
    int warp_size = 0;
    int multiprocessor_count = 0;
    // Driver/runtime versions as returned by cudaDriverGetVersion /
    // cudaRuntimeGetVersion, encoded as (major*1000 + minor*10); used
    // by benchmark result metadata (Section 5) so a saved result records
    // which driver/toolkit it was measured against.
    int driver_version = 0;
    int runtime_version = 0;
};

// Queries device 0. Throws std::runtime_error if no CUDA device is present.
DeviceInfo get_device_info();

// Minimal RAII wrapper around a single cudaMalloc'd buffer of T elements.
// Later sections (batch pipeline, ping-pong filter buffers) are expected to
// build on top of this rather than calling cudaMalloc/cudaFree directly.
// Non-copyable, movable.
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t count) : count_(count) {
        if (count_ > 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)));
        }
    }

    ~DeviceBuffer() { free(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            free();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    void upload(const T* host_src, size_t count, cudaStream_t stream = nullptr) {
        if (stream) {
            CUDA_CHECK(cudaMemcpyAsync(ptr_, host_src, count * sizeof(T),
                                       cudaMemcpyHostToDevice, stream));
        } else {
            CUDA_CHECK(cudaMemcpy(ptr_, host_src, count * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    void download(T* host_dst, size_t count, cudaStream_t stream = nullptr) const {
        if (stream) {
            CUDA_CHECK(cudaMemcpyAsync(host_dst, ptr_, count * sizeof(T),
                                       cudaMemcpyDeviceToHost, stream));
        } else {
            CUDA_CHECK(cudaMemcpy(host_dst, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }

    T* get() const { return ptr_; }
    size_t count() const { return count_; }

    // Wraps an already-allocated device pointer for RAII cleanup, without
    // performing a second allocation. Used when a caller needs to inspect
    // a cudaMalloc failure itself (e.g. to raise a more specific error)
    // before handing ownership of a successful allocation to a
    // DeviceBuffer.
    static DeviceBuffer<T> adopt(T* ptr, size_t count) {
        DeviceBuffer<T> buf;
        buf.ptr_ = ptr;
        buf.count_ = count;
        return buf;
    }

private:
    void free() {
        if (ptr_) {
            cudaFree(ptr_);  // destructor: no throw on failure
            ptr_ = nullptr;
        }
        count_ = 0;
    }

    T* ptr_ = nullptr;
    size_t count_ = 0;
};

// RAII CUDA-event timer. Records "start" at construction-time call to
// start(), "stop" on stop(), and reports elapsed milliseconds via
// elapsed_ms(). Intended for future per-stage GPU benchmarks.
class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_event_));
        CUDA_CHECK(cudaEventCreate(&stop_event_));
    }

    ~CudaTimer() {
        cudaEventDestroy(start_event_);
        cudaEventDestroy(stop_event_);
    }

    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

    void start(cudaStream_t stream = nullptr) { CUDA_CHECK(cudaEventRecord(start_event_, stream)); }

    void stop(cudaStream_t stream = nullptr) { CUDA_CHECK(cudaEventRecord(stop_event_, stream)); }

    // Blocks until the stop event completes, then returns elapsed time.
    float elapsed_ms() const {
        CUDA_CHECK(cudaEventSynchronize(stop_event_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_event_, stop_event_));
        return ms;
    }

private:
    cudaEvent_t start_event_ = nullptr;
    cudaEvent_t stop_event_ = nullptr;
};

}  // namespace xray_cuda
