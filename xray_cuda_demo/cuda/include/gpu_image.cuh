#pragma once

#include "cuda_common.cuh"
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace xray_cuda {

// Raised specifically for CUDA out-of-memory conditions so the Python
// binding layer can map it to MemoryError instead of a generic
// RuntimeError, while keeping a message detailed enough to act on
// (requested size vs. free/total device memory).
class CudaMemoryError : public std::runtime_error {
public:
    explicit CudaMemoryError(const std::string& message) : std::runtime_error(message) {}
};

// Allocates `byte_count` device bytes, adopting them into a fresh
// DeviceBuffer<uint8_t> on success, or throwing a CudaMemoryError with
// requested/free/total context on failure. Shared by GpuImage and
// GpuImageBatch so the allocation-failure message format isn't
// duplicated between them.
inline DeviceBuffer<uint8_t> allocate_uint8_buffer_or_throw(size_t byte_count, const std::string& shape_description) {
    uint8_t* raw_ptr = nullptr;
    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&raw_ptr), byte_count);
    if (status != cudaSuccess) {
        cudaGetLastError();  // clear the sticky error
        size_t free_bytes = 0, total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);  // best-effort; ignore failure here
        throw CudaMemoryError(
            "CUDA allocation failed for " + shape_description + "\n" +
            "requested bytes: " + std::to_string(byte_count) + "\n" +
            "free GPU memory: " + std::to_string(free_bytes) + " / " + std::to_string(total_bytes) + " bytes\n" +
            "CUDA error: " + cudaGetErrorString(status));
    }
    return DeviceBuffer<uint8_t>::adopt(raw_ptr, byte_count);
}

// -- launch configuration -------------------------------------------------------

// One stable default for Section 4A. Later optimization sections may
// want to sweep 8x8 / 16x16 / 32x8 / 32x16 -- compute_launch_grid()
// already takes an explicit block size so that experiment doesn't
// require touching this helper, only the caller's chosen `block`.
inline dim3 default_block_dim() { return dim3(16, 16); }

inline dim3 compute_launch_grid(int width, int height, dim3 block = default_block_dim()) {
    return dim3(
        (static_cast<unsigned int>(width) + block.x - 1) / block.x,
        (static_cast<unsigned int>(height) + block.y - 1) / block.y
    );
}

// -- border handling -------------------------------------------------------

// Maps an out-of-range coordinate to the equivalent in-range coordinate
// under BORDER_REFLECT_101 / BORDER_DEFAULT (reflect without duplicating
// the edge pixel), e.g. for len=5: ... 2,1,0,1,2,3,4,3,2,1,0,1,2 ...
// Handles arbitrarily large offsets via modulo, matching
// cv::borderInterpolate's behavior for very small images. Shared by
// every filter that uses cv2.BORDER_DEFAULT (Gaussian, Sobel, Laplacian)
// -- median uses BORDER_REPLICATE instead (see median_basic.cu), which
// is genuinely different and intentionally not this function.
__device__ __forceinline__ int reflect101(int idx, int len) {
    if (len == 1) return 0;
    int period = 2 * (len - 1);
    idx %= period;
    if (idx < 0) idx += period;
    if (idx >= len) idx = period - idx;
    return idx;
}

// BORDER_REPLICATE: out-of-range coordinates clamp to the nearest edge
// pixel. Used by median (Section 4C: verified empirically that
// cv2.medianBlur replicates, not reflects -- genuinely different from
// reflect101() above). Shared by median_basic.cu and median_enhanced.cu
// (Section 7) so the border rule is defined in exactly one place.
__device__ __forceinline__ int clamp_index(int idx, int len) {
    if (idx < 0) return 0;
    if (idx >= len) return len - 1;
    return idx;
}

// -- stream abstraction -------------------------------------------------------

// Not yet exposed to Python -- Section 4A intentionally uses the default
// stream everywhere for correctness-first synchronous execution. This
// exists so later (asynchronous, multi-stream) sections have an
// abstraction to build on instead of scattering raw cudaStream_t handles.
class CudaStream {
public:
    CudaStream() { CUDA_CHECK(cudaStreamCreate(&stream_)); }
    ~CudaStream() {
        if (stream_) {
            cudaStreamDestroy(stream_);  // destructor: no throw on failure
        }
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    void synchronize() { CUDA_CHECK(cudaStreamSynchronize(stream_)); }
    cudaStream_t handle() const { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

// -- GPU image buffer -------------------------------------------------------

// A single-channel uint8 image living in device memory. RAII: the
// underlying allocation is released automatically (via DeviceBuffer) when
// the GpuImage is destroyed, so application code never calls
// cudaMalloc/cudaFree directly.
//
// Layout: simple contiguous row-major storage, stride == width bytes (no
// padding/pitch). This is deliberately the simplest possible
// representation for Section 4A; if later profiling shows a
// cudaMallocPitch-aligned stride would help (e.g. for coalesced access
// on wide images), that can be introduced as an explicit `stride_bytes`
// field without changing this class's public upload/download contract.
class GpuImage {
public:
    GpuImage(int height, int width) : height_(height), width_(width) {
        if (height <= 0 || width <= 0) {
            throw std::invalid_argument(
                "GpuImage dimensions must be positive, got height=" + std::to_string(height) +
                " width=" + std::to_string(width));
        }
        const size_t byte_count = static_cast<size_t>(height) * static_cast<size_t>(width);
        allocate(byte_count);
    }

    ~GpuImage() = default;
    GpuImage(const GpuImage&) = delete;
    GpuImage& operator=(const GpuImage&) = delete;
    GpuImage(GpuImage&&) = default;
    GpuImage& operator=(GpuImage&&) = default;

    void upload_from_host(const uint8_t* host_ptr, size_t byte_count) {
        check_byte_count(byte_count);
        buffer_.upload(host_ptr, byte_count);
    }

    void download_to_host(uint8_t* host_ptr, size_t byte_count) const {
        check_byte_count(byte_count);
        buffer_.download(host_ptr, byte_count);
    }

    uint8_t* data() { return buffer_.get(); }
    const uint8_t* data() const { return buffer_.get(); }

    int height() const { return height_; }
    int width() const { return width_; }
    size_t nbytes() const { return buffer_.count(); }

private:
    void check_byte_count(size_t byte_count) const {
        if (byte_count != nbytes()) {
            throw std::invalid_argument(
                "Byte count mismatch: expected " + std::to_string(nbytes()) +
                " bytes for a " + std::to_string(height_) + "x" + std::to_string(width_) +
                " image, got " + std::to_string(byte_count));
        }
    }

    void allocate(size_t byte_count) {
        buffer_ = allocate_uint8_buffer_or_throw(
            byte_count, std::to_string(height_) + "x" + std::to_string(width_) + " image");
    }

    DeviceBuffer<uint8_t> buffer_;
    int height_ = 0;
    int width_ = 0;
};

// -- GPU image batch buffer (Section 5) -------------------------------------------------------

// A batch of `batch_size` single-channel uint8 images, all sharing one
// (height, width), living contiguously in device memory as
// [batch_size, height, width] (row-major, no padding between images --
// same simple-contiguous-layout philosophy as GpuImage). RAII, same as
// GpuImage.
//
// This exists so the production pipeline (see pipeline_basic.cuh) can
// allocate two of these once per GPU chunk and ping-pong between them
// across all five filter stages, instead of allocating/freeing a new
// buffer per filter per image (which was fine for the Section 4B-4F
// single-image correctness work, but is exactly the pattern Section 5
// replaces for batches).
class GpuImageBatch {
public:
    GpuImageBatch(int batch_size, int height, int width)
        : batch_size_(batch_size), height_(height), width_(width) {
        if (batch_size <= 0 || height <= 0 || width <= 0) {
            throw std::invalid_argument(
                "GpuImageBatch dimensions must be positive, got batch_size=" + std::to_string(batch_size) +
                " height=" + std::to_string(height) + " width=" + std::to_string(width));
        }
        const size_t byte_count =
            static_cast<size_t>(batch_size) * static_cast<size_t>(height) * static_cast<size_t>(width);
        buffer_ = allocate_uint8_buffer_or_throw(
            byte_count,
            std::to_string(batch_size) + "x" + std::to_string(height) + "x" + std::to_string(width) + " image batch");
    }

    ~GpuImageBatch() = default;
    GpuImageBatch(const GpuImageBatch&) = delete;
    GpuImageBatch& operator=(const GpuImageBatch&) = delete;
    GpuImageBatch(GpuImageBatch&&) = default;
    GpuImageBatch& operator=(GpuImageBatch&&) = default;

    void upload_from_host(const uint8_t* host_ptr, size_t byte_count) {
        check_byte_count(byte_count);
        buffer_.upload(host_ptr, byte_count);
    }

    void download_to_host(uint8_t* host_ptr, size_t byte_count) const {
        check_byte_count(byte_count);
        buffer_.download(host_ptr, byte_count);
    }

    uint8_t* data() { return buffer_.get(); }
    const uint8_t* data() const { return buffer_.get(); }

    int batch_size() const { return batch_size_; }
    int height() const { return height_; }
    int width() const { return width_; }
    size_t nbytes() const { return buffer_.count(); }
    size_t plane_bytes() const { return static_cast<size_t>(height_) * static_cast<size_t>(width_); }

private:
    void check_byte_count(size_t byte_count) const {
        if (byte_count != nbytes()) {
            throw std::invalid_argument(
                "Byte count mismatch: expected " + std::to_string(nbytes()) + " bytes for a " +
                std::to_string(batch_size_) + "x" + std::to_string(height_) + "x" + std::to_string(width_) +
                " batch, got " + std::to_string(byte_count));
        }
    }

    DeviceBuffer<uint8_t> buffer_;
    int batch_size_ = 0;
    int height_ = 0;
    int width_ = 0;
};

// -- trivial boundary-safe image kernel -------------------------------------------------------

// output[i] = input[i] + 1, uint8 wraparound (255 + 1 == 0) -- this is
// standard unsigned-integer overflow behavior, not a bug; documented here
// since Section 1's smoke-test kernel used float and never had to make
// this call. Every thread bounds-checks x<width, y<height so a launch
// grid larger than the image (i.e. width/height not a multiple of the
// block size) never touches out-of-bounds memory.
__global__ void image_add_one_kernel(const uint8_t* input, uint8_t* output, int width, int height);

// Launches image_add_one_kernel on `input`, returning a new GpuImage
// (input is not modified) and the kernel-only elapsed time in
// milliseconds (measured with CUDA events, not CPU wall clock).
// Synchronizes the device before returning, per Section 4A's
// correctness-first synchronization policy (upload -> kernel ->
// synchronize -> download).
std::pair<std::unique_ptr<GpuImage>, float> image_add_one(const GpuImage& input);

}  // namespace xray_cuda
