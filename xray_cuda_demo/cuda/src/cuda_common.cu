#include "cuda_common.cuh"

namespace xray_cuda {

bool cuda_is_available() {
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) {
        // A driver/runtime mismatch or absent GPU surfaces here as an error
        // code rather than an exception; clear it so it doesn't leak into
        // the next unrelated CUDA call.
        cudaGetLastError();
        return false;
    }
    return device_count > 0;
}

DeviceInfo get_device_info() {
    if (!cuda_is_available()) {
        throw std::runtime_error("No usable CUDA device found on this system.");
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    DeviceInfo info;
    info.name = props.name;
    info.compute_capability_major = props.major;
    info.compute_capability_minor = props.minor;
    info.total_global_mem_bytes = props.totalGlobalMem;
    info.shared_mem_per_block_bytes = props.sharedMemPerBlock;
    info.registers_per_block = props.regsPerBlock;
    info.max_threads_per_block = props.maxThreadsPerBlock;
    info.warp_size = props.warpSize;
    info.multiprocessor_count = props.multiProcessorCount;

    int driver_version = 0, runtime_version = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    info.driver_version = driver_version;
    info.runtime_version = runtime_version;

    return info;
}

}  // namespace xray_cuda
