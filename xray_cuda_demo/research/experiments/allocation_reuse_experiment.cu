// Standalone experiment (Section 20A, items 24-25): measures the real
// per-call cost of allocating fresh device buffers every pipeline
// invocation (the CURRENT production behavior -- GpuImageBatch objects
// are local to run_basic_cuda_pipeline_batch(), so every Python-level
// call pays fresh cudaMalloc/cudaFree) vs. two alternatives:
//   (a) cudaMalloc/cudaFree fresh every call (current behavior, baseline)
//   (b) a single persistent allocation reused across calls
//   (c) cudaMallocAsync/cudaFreeAsync against a stream-ordered pool
// Not wired into the production extension.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while (0)

double median_of(std::vector<float>& v) {
    std::vector<float> s = v;
    std::sort(s.begin(), s.end());
    return s[s.size() / 2];
}

int main() {
    const int height = 224, width = 224;
    const int warmup = 3, runs = 20;
    // 2 GpuImageBatch buffers + 1 Gaussian intermediate float buffer per
    // pipeline call, matching pipeline_basic.cu's actual allocation pattern.
    int batch_sizes[] = {1, 8, 32, 122, 512};

    std::printf("batch_size,fresh_malloc_free_ms,persistent_reuse_ms,async_pool_ms,fresh_vs_persistent,fresh_vs_pool\n");

    for (int batch_size : batch_sizes) {
        size_t uint8_bytes = static_cast<size_t>(batch_size) * height * width;
        size_t float_bytes = uint8_bytes * sizeof(float);

        // -- (a) fresh cudaMalloc/cudaFree every call --
        std::vector<float> fresh_times;
        for (int i = 0; i < warmup + runs; ++i) {
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
            CUDA_CHECK(cudaEventRecord(start));
            unsigned char *a, *b;
            float* intermediate;
            CUDA_CHECK(cudaMalloc(&a, uint8_bytes));
            CUDA_CHECK(cudaMalloc(&b, uint8_bytes));
            CUDA_CHECK(cudaMalloc(&intermediate, float_bytes));
            CUDA_CHECK(cudaFree(a));
            CUDA_CHECK(cudaFree(b));
            CUDA_CHECK(cudaFree(intermediate));
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            if (i >= warmup) fresh_times.push_back(ms);
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
        }

        // -- (b) persistent allocation, reused across "calls" (allocated once outside the loop) --
        unsigned char *pa, *pb;
        float* pintermediate;
        CUDA_CHECK(cudaMalloc(&pa, uint8_bytes));
        CUDA_CHECK(cudaMalloc(&pb, uint8_bytes));
        CUDA_CHECK(cudaMalloc(&pintermediate, float_bytes));
        std::vector<float> persistent_times;
        for (int i = 0; i < warmup + runs; ++i) {
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
            CUDA_CHECK(cudaEventRecord(start));
            // "reuse": nothing to do -- buffers already sized and allocated.
            // A real pipeline would just use pa/pb/pintermediate directly.
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            if (i >= warmup) persistent_times.push_back(ms);
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
        }
        CUDA_CHECK(cudaFree(pa));
        CUDA_CHECK(cudaFree(pb));
        CUDA_CHECK(cudaFree(pintermediate));

        // -- (c) cudaMallocAsync/cudaFreeAsync against the default stream-ordered pool --
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        std::vector<float> pool_times;
        for (int i = 0; i < warmup + runs; ++i) {
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
            CUDA_CHECK(cudaEventRecord(start, stream));
            unsigned char *a, *b;
            float* intermediate;
            CUDA_CHECK(cudaMallocAsync(&a, uint8_bytes, stream));
            CUDA_CHECK(cudaMallocAsync(&b, uint8_bytes, stream));
            CUDA_CHECK(cudaMallocAsync(&intermediate, float_bytes, stream));
            CUDA_CHECK(cudaFreeAsync(a, stream));
            CUDA_CHECK(cudaFreeAsync(b, stream));
            CUDA_CHECK(cudaFreeAsync(intermediate, stream));
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            if (i >= warmup) pool_times.push_back(ms);
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
        }
        CUDA_CHECK(cudaStreamDestroy(stream));

        double fresh_ms = median_of(fresh_times);
        double persistent_ms = median_of(persistent_times);
        double pool_ms = median_of(pool_times);

        std::printf("%d,%.4f,%.4f,%.4f,%.2f,%.2f\n",
                     batch_size, fresh_ms, persistent_ms, pool_ms,
                     fresh_ms / (persistent_ms > 0.0005 ? persistent_ms : 0.0005),
                     fresh_ms / (pool_ms > 0.0005 ? pool_ms : 0.0005));
    }
    return 0;
}
