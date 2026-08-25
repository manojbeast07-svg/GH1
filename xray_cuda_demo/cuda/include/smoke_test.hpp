#pragma once

#include <vector>

namespace xray_cuda {

struct SmokeTestResult {
    std::vector<float> output;
    float kernel_ms = 0.0f;
};

// Defined in smoke_test.cu. Runs input[i] + 1 on the GPU for a small fixed
// vector and returns the result plus kernel execution time.
SmokeTestResult run_smoke_test();

}  // namespace xray_cuda
