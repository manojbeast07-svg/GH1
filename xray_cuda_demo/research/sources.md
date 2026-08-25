# Sources (Section 20A)

## Primary evidence (generated in this repository, this session)

| Evidence | How obtained | File |
|---|---|---|
| Enhanced pipeline CUDA API / kernel / memcpy timeline | `nsys profile` on `run_enhanced_cuda_pipeline()`, 122-image batch, real dataset | `research/raw_evidence/nsys_enhanced_stats.txt` + `nsys_enhanced_cuda_*.csv` (the `.nsys-rep` binary trace itself was not retained after extracting these summaries) |
| Basic pipeline CUDA API / kernel / memcpy timeline | `nsys profile` on `run_basic_cuda_pipeline()`, same batch | `research/raw_evidence/nsys_basic_stats.txt` + `nsys_basic_cuda_*.csv` |
| Per-kernel register / shared-memory / spill usage | `nvcc --ptxas-options=-v -c` on every Basic/Enhanced `.cu` file, fresh compile this session | (compiled to throwaway `.obj`, output captured in `filter_optimization_matrix.md`; not persisted as a build artifact) |
| Pinned vs. pageable H2D/D2H transfer time (7 repeated runs) | Standalone experiment, real GPU, 5 batch sizes | `research/experiments/pinned_memory_experiment.cu`, results in `research/raw_evidence/pinned_memory_experiment_results.csv` and `pipeline_optimization_matrix.md` |
| Fresh-malloc vs. persistent-buffer vs. `cudaMallocAsync`/pool allocation cost | Standalone experiment, real GPU, 5 batch sizes | `research/experiments/allocation_reuse_experiment.cu`, results in `research/raw_evidence/allocation_reuse_experiment_results.csv` |

Nsight Compute (`ncu`) was attempted but returned `ERR_NVGPUCTRPERM` (GPU performance-counter access requires
Administrator privileges on this machine, which this session does not have) -- Nsight **Systems** worked without
elevation and supplied the CPU/GPU API-timeline evidence above instead. This is disclosed rather than silently
substituting a different measurement for what Nsight Compute would have shown (occupancy %, achieved memory
throughput as a fraction of peak, warp-level stall reasons were NOT obtained this session).

## External sources (fetched/searched this session)

1. **NVIDIA — "How to Optimize Data Transfers in CUDA C/C++"** (developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc).
   Real benchmark numbers: pinned-vs-pageable H2D bandwidth improvement ranged ~15% (desktop, GTX 680) to ~2.5x
   (laptop, NVS 4200M) depending on host system. Recommends batching small transfers rather than many small ones,
   and warns against over-allocating pinned memory (reduces OS-available physical memory).
2. **NVIDIA CUDA Programming Guide — "Stream-Ordered Memory Allocator"**
   (docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html) and
   **CUDA Runtime API — Memory Pools** (docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html).
   `cudaMallocAsync`/`cudaFreeAsync` return allocations to a driver-managed pool instead of the OS, avoiding
   cross-stream synchronization that plain `cudaMalloc`/`cudaFree` incurs -- the documented mechanism this
   project's own `allocation_reuse_experiment.cu` was built to test empirically (see
   `pipeline_optimization_matrix.md`: the empirical result did not match the documented expectation on this GPU
   at these buffer sizes).
3. **NVIDIA Nsight Compute — Kernel Profiling Guide** (docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
   Authoritative source for the project's occupancy-is-a-diagnostic-not-a-goal principle (spec item 35) and for
   the compute/memory/instruction/launch/occupancy category breakdown this section's methodology follows even
   where `ncu` itself was unavailable.
4. **CUDA Graphs launch-overhead literature** (search aggregate, including NVIDIA's CUDA Graph Best Practice docs
   at docs.nvidia.com/dl-cuda-graph and community benchmarks). Reported per-kernel CPU launch overhead ~5-10us;
   CUDA Graph replay eliminates most of this for workloads with many small, repeated launches. Used to size the
   *ceiling* of what CUDA Graphs could plausibly save for this project's 6-launches-per-call pipeline (see
   `pipeline_optimization_matrix.md`).
5. **Peer-reviewed / conference median-filter literature**: "Hierarchical Histogram-based Median Filter for GPUs"
   (ResearchGate 325944500) and "High-throughput median filtering for large kernel sizes on CUDA" (SPIE
   Proceedings 13517). Consistent finding across sources: histogram-based median filtering *scales better than
   sorting-network approaches as kernel size grows*, but sorting-based algorithms remain competitive (or better)
   for small (3x3) windows -- directly relevant to `filter_optimization_matrix.md`'s Median row (this project's
   existing `network3x3` sorting-network choice for k=3 is consistent with this literature; k=5/k=7 is flagged as
   the case where histogram-based methods have real literature support).

## Project-internal historical evidence (already established, Sections 6-10, 17 — re-cited, not re-derived)

- `README.md`'s per-filter "Rejected experiments" tables (Sections 6, 8, 9, 10): vectorized loads rejected for
  Gaussian/Sobel/Laplacian because the real dataset's widths (e.g. 1733px) are not 4-aligned; separable Sobel
  measured slower than Specialized; SharedConst regressed for Sobel but not for Laplacian (different headroom).
- `benchmark_results/variant_sweeps/*.json` (Section 17): real measured Basic-vs-every-Enhanced-variant kernel
  times for all five filters, at kernel sizes 3/5/7 (Median), 3/5 (Laplacian).
- `benchmark_results/variant_sweeps/fusion.json` (Section 17, from Section 10's experiment): the Laplacian+
  Threshold fusion measurement (1.30x combined-kernel gain, ~3% whole-pipeline gain, rejected for production
  because it discards the standalone Laplacian output the project's introspection tooling depends on).
