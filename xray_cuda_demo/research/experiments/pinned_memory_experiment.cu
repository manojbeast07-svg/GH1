// Standalone experiment (Section 20A, item 22): measures whether pinned
// (page-locked) host memory improves H2D/D2H transfer time vs. pageable
// memory, at the actual batch sizes this project uses. Not wired into
// the production extension -- a throwaway measurement program, compiled
// and run directly via nvcc, deleted/kept only under research/.
//
// Compares, for each batch size, N repeated H2D+D2H round trips of a
// [batch, 224, 224] uint8 buffer:
//   (a) pageable host memory (plain new[]/malloc)
//   (b) pinned host memory (cudaHostAlloc)
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

struct Stat {
    double mean_ms;
    double median_ms;
};

Stat summarize(std::vector<float>& values_ms) {
    double sum = 0;
    for (float v : values_ms) sum += v;
    std::vector<float> sorted = values_ms;
    std::sort(sorted.begin(), sorted.end());
    return {sum / values_ms.size(), sorted[sorted.size() / 2]};
}

int main() {
    const int height = 224, width = 224;
    const int warmup = 3, runs = 10;
    int batch_sizes[] = {1, 8, 32, 122, 512};

    std::printf("batch_size,pageable_h2d_ms,pageable_d2h_ms,pinned_h2d_ms,pinned_d2h_ms,h2d_speedup,d2h_speedup\n");

    for (int batch_size : batch_sizes) {
        size_t bytes = static_cast<size_t>(batch_size) * height * width;

        // -- pageable --
        unsigned char* pageable_host = new unsigned char[bytes];
        for (size_t i = 0; i < bytes; ++i) pageable_host[i] = static_cast<unsigned char>(i % 256);
        unsigned char* device_buf = nullptr;
        CUDA_CHECK(cudaMalloc(&device_buf, bytes));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        std::vector<float> pageable_h2d, pageable_d2h;
        for (int i = 0; i < warmup + runs; ++i) {
            CUDA_CHECK(cudaEventRecord(start));
            CUDA_CHECK(cudaMemcpy(device_buf, pageable_host, bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            if (i >= warmup) pageable_h2d.push_back(ms);

            CUDA_CHECK(cudaEventRecord(start));
            CUDA_CHECK(cudaMemcpy(pageable_host, device_buf, bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            if (i >= warmup) pageable_d2h.push_back(ms);
        }
        delete[] pageable_host;

        // -- pinned --
        unsigned char* pinned_host = nullptr;
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&pinned_host), bytes, cudaHostAllocDefault));
        for (size_t i = 0; i < bytes; ++i) pinned_host[i] = static_cast<unsigned char>(i % 256);

        std::vector<float> pinned_h2d, pinned_d2h;
        for (int i = 0; i < warmup + runs; ++i) {
            CUDA_CHECK(cudaEventRecord(start));
            CUDA_CHECK(cudaMemcpy(device_buf, pinned_host, bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            if (i >= warmup) pinned_h2d.push_back(ms);

            CUDA_CHECK(cudaEventRecord(start));
            CUDA_CHECK(cudaMemcpy(pinned_host, device_buf, bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            if (i >= warmup) pinned_d2h.push_back(ms);
        }
        CUDA_CHECK(cudaFreeHost(pinned_host));
        CUDA_CHECK(cudaFree(device_buf));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        Stat pg_h2d = summarize(pageable_h2d), pg_d2h = summarize(pageable_d2h);
        Stat pn_h2d = summarize(pinned_h2d), pn_d2h = summarize(pinned_d2h);

        std::printf("%d,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f\n",
                     batch_size, pg_h2d.median_ms, pg_d2h.median_ms, pn_h2d.median_ms, pn_d2h.median_ms,
                     pg_h2d.median_ms / pn_h2d.median_ms, pg_d2h.median_ms / pn_d2h.median_ms);
    }
    return 0;
}
