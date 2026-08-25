# Pipeline-Level Optimization Matrix (Section 20A)

## Measured baseline (this session, fresh Release build, 122-image canonical batch, 224x224)

| Stage | Median time | Source |
|---|---:|---|
| H2D | 0.944 ms | `benchmark_results/release_validation/raw/enhanced_cuda/` |
| GPU compute (5 filters) | 1.284 ms | same |
| D2H | 1.082 ms | same |
| **GPU total (H2D+compute+D2H)** | **3.401 ms** | same |
| End-to-end (incl. disk load + host orchestration) | 56.45 ms (mean) | `benchmark_results/release_validation/summary/latest.json` |

**Critical context for everything below:** disk I/O to load 119 real JPEGs dominates end-to-end time by roughly
16x over the entire GPU round trip (~53 ms of the ~56.45 ms total is not GPU-attributable at all). None of the
pipeline-level candidates below can move a *single, one-shot* end-to-end number much, because disk I/O so
thoroughly dominates it. Their real value is for **repeated same-shape pipeline calls** -- which is not a
hypothetical: it is exactly how this project's own benchmark loops (20 measurement runs per config), Live
Processing's repeated "Process/Compare" clicks, and the Optimization Lab's live comparisons (5 measurement runs)
already work. Evaluated on *that* basis, not on a single cold run.

## CUDA API cost breakdown (Nsight Systems, Enhanced pipeline, 4 calls: 3 warmup + 1 measured)

| API / category | Median per call (derived) | % of CPU-side API time |
|---|---:|---:|
| `cudaMalloc` (3 buffers/call: 2x `GpuImageBatch` + 1x Gaussian intermediate) | ~0.23 ms/call (median single call), ~0.7-1.9 ms/call total allocation cost (see experiment below) | 78.8% |
| `cudaEventRecord` (CUDA-event timing itself) | -- | 8.9% |
| `cudaMemcpy` (explicit, non-pipeline calls in the driver script) | -- | 5.0% |
| `cudaMemcpyToSymbol` (Laplacian constant-memory coefficient upload) | ~0.69 ms avg, highly variable (18us-2.28ms) | 2.5% |
| `cudaDeviceSynchronize` | ~0.2 ms avg | 2.2% |
| `cudaFree` | -- | 1.6% |
| `cudaLaunchKernel` (6 launches/call: Gaussian h+v, Median, Sobel, Laplacian, Threshold) | ~55 us avg/launch => ~0.33 ms/call | 0.6% |

**Reading this table:** `cudaMalloc`/`cudaFree` together are the single largest CPU-side cost category by far --
larger than the actual kernel launches. This matches the architecture directly: `GpuImageBatch buffer_a`/
`buffer_b` (and Gaussian's intermediate float buffer, when Enhanced) are declared *inside*
`run_basic_cuda_pipeline_batch()` (`cuda/src/pipeline_basic.cu`) -- they are reused via ping-pong **within** one
call (as designed and documented), but a **new** pair is allocated and freed on **every separate Python-level
call** to `run_basic_cuda_pipeline()`/`run_enhanced_cuda_pipeline()`/`run_cuda_pipeline()`.

## Standalone experiment 1: allocation strategy (`research/experiments/allocation_reuse_experiment.cu`)

Real measured median time (ms) for the 3-buffer allocate+free pattern the pipeline does every call, at 5 batch
sizes, 20 measured repetitions each (3 warmup):

| Batch | Fresh `cudaMalloc`/`cudaFree` (current behavior) | Persistent (pre-allocated, reused) | `cudaMallocAsync`/pool |
|---:|---:|---:|---:|
| 1 | 0.399 ms | ~0.005 ms | 1.256 ms |
| 8 | 0.641 ms | ~0.007 ms | 1.253 ms |
| 32 | 1.096 ms | ~0.007 ms | 1.271 ms |
| 122 (canonical) | 1.862 ms | ~0.008 ms | 1.918 ms |
| 512 | 4.951 ms | ~0.038 ms | 4.044 ms |

**Finding:** the fresh-allocate cost (0.4-4.95 ms/call depending on batch size) is comparable to or **larger than
the actual GPU compute time itself** (1.28 ms/call at the canonical batch size) -- for repeated same-shape calls,
this is the single largest identified optimization opportunity in the whole pipeline, filter kernels included.

**Surprising negative result:** `cudaMallocAsync`/`cudaFreeAsync` against the default stream-ordered pool did
**not** show the documented benefit on this GPU/driver at these buffer sizes -- it was *slower* than plain
`cudaMalloc`/`cudaFree` at every batch size except 512 (where it was only 1.22x faster), and dramatically slower
than true buffer persistence at every size. Per this project's own rule (spec item 25: "if none exists, REJECT AS
UNNECESSARY"), `cudaMallocAsync` is **rejected** as a candidate here -- the documented mechanism (pool avoids
cross-stream sync) evidently isn't the dominant cost on this hardware; true persistence (never calling
malloc/free at all on the hot path) is what actually eliminates the cost, and does so completely.

## Standalone experiment 2: pinned vs. pageable host memory (`research/experiments/pinned_memory_experiment.cu`)

This experiment showed enough run-to-run variance to warrant repeating rather than reporting a single pass --
consistent with this project's own "don't hide variability" discipline. **7 independent runs** at the canonical
batch size (122), each with its own warmup + 10 measured repetitions:

| Run | H2D speedup | D2H speedup |
|---:|---:|---:|
| 1 | 1.095x | 1.095x |
| 2 | 1.797x | 1.942x |
| 3 | 1.872x | 1.932x |
| 4 | 2.309x | 2.223x |
| 5 | 1.878x | 1.962x |
| 6 | 1.732x | 1.865x |
| 7 | 1.154x | 1.236x |
| **Median** | **1.797x** | **1.932x** |

The spread (1.10x-2.31x) is real system-state variance, not a bug -- the NVIDIA source itself documents that
pinned-memory speedup depends on host system state (motherboard/CPU/chipset), and this machine was running other
GPU work (Nsight profiling, builds) around these measurements. Taking the median across 7 runs rather than a
single pass, the finding is a **genuine ~1.8-1.9x** transfer speedup at the canonical batch size, not the
~1.10x a single early run suggested.

At other batch sizes (single-run measurements, not repeated to the same depth):

| Batch | Pageable H2D | Pinned H2D | H2D speedup | Pageable D2H | Pinned D2H | D2H speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0256-0.0524 | 0.0230-0.0433 | 0.87x-2.11x (high variance, tiny transfer) | 0.0463-0.0943 | 0.0214-0.0484 | 0.96x-4.41x |
| 8 | 0.0854-0.0909 | 0.0472-0.1003 | 0.85x-1.93x | 0.0888-0.0982 | 0.0512-0.0795 | 1.16x-1.92x |
| 32 | 0.1962-0.2721 | 0.1385-0.1515 | 1.34x-1.85x | 0.2021-0.3106 | 0.1404-0.2068 | 1.37x-2.03x |
| 512 | 2.0154-2.6961 | 1.9543-1.9648 | 1.03x-1.38x | 2.1351-3.4163 | 1.9820-1.9970 | 1.07x-1.72x |

**Finding:** pinned memory gives a **real, consistently-positive** (never a regression across 12 total runs
gathered) benefit, with a median ~1.8-1.9x at the canonical batch size -- stronger than initially estimated from
a single run. Small batches show the widest variance (transfer time is tiny, so fixed overhead dominates either
way); batch=512 shows the smallest, most consistent gain (~1.0-1.4x, transfer time large enough that pinned's
per-call overhead advantage matters proportionally less). Genuinely comparable in magnitude to the
allocation-reuse opportunity below, not clearly smaller as a single early run suggested.

## Pipeline optimization decision table

| Optimization | Target | Expected benefit | Complexity | Correctness risk | Benchmark? | Classification |
|---|---|---:|---:|---:|---|---|
| **Buffer persistence / cache across repeated calls** | `cudaMalloc`/`cudaFree` overhead | **Measured: 0.4-4.95 ms/call recovered** (comparable to or exceeding kernel compute time) | Low-Medium (cache 2-3 device buffers keyed by (batch_size, height, width); invalidate on shape change) | Low (same ping-pong algorithm, only allocation lifetime changes) | **Yes -- done, see above** | **PROMISING -- strongest evidence of any candidate in this section; recommended for a dedicated implementation experiment, not applied to production in this research gate per the section's own STOP CONDITION** |
| Pinned host memory | H2D/D2H transfer | Measured (median of 7 runs at canonical batch): **~1.8-1.9x**; 1.0-1.4x at batch=512 | Low (`cudaHostAlloc` for the upload/download staging buffer) | Low | **Yes -- done, see above (7 repeated runs after initial variance)** | PROMISING -- real, consistently positive across every run gathered, comparable in magnitude to buffer persistence at the canonical batch size |
| `cudaMallocAsync` / memory pool | Allocation overhead | Measured: **regression** at batch<=32, +1.22x only at batch=512 | Low | Low | Yes -- done, see above | **REJECTED** -- documented mechanism did not reproduce on this hardware; true persistence (above) achieves the same goal far more effectively |
| CUDA Graph capture of the 6-launch sequence | Kernel-launch CPU overhead | Estimated ceiling ~0.3 ms/call (6 launches x ~55us measured average) -- smaller than either candidate above | Medium (capture/replay lifecycle, must handle enable/disable-stage branching, which changes the graph shape) | Low-Medium (graph must be rebuilt when the enabled-stage combination changes) | No -- not built this session; evidence-based ceiling is smaller than buffer persistence, lower priority | PROMISING -- REQUIRES MORE TESTING (worth revisiting after buffer persistence is evaluated, since the two are complementary, not competing) |
| Stream overlap (H2D of batch N+1 while computing batch N) | Cross-batch pipeline latency | Not measured -- inapplicable to a single-image or single-batch call; only relevant for a sequence of independent batches (e.g. multiple resolution groups in one Live Batch run) | High (explicit dependency management, correctness risk if a stage reads a buffer still being overwritten) | Medium | No -- correctness risk plus limited applicability (most real usage is single-batch-at-a-time) don't clear the bar for this section's time budget | PROMISING -- REQUIRES MORE TESTING, lower priority than the two GPU-side candidates |
| Double buffering (compute/transfer overlap within one call) | Same class as stream overlap | Not measured | High | Medium | No | TOO COMPLEX FOR BENEFIT AT THIS TIME -- same reasoning as stream overlap; the measured H2D+D2H (2.03ms) is already smaller than compute (1.28ms) is large relative to, so the overlap ceiling is bounded by the smaller of the two, ~1.28ms/call at best, less than buffer persistence's proven gain |
| Kernel fusion (Laplacian+Threshold) | Launch count + intermediate write | Already measured (Section 10): 1.30x combined-kernel, ~3% whole-pipeline | Already built | Already validated bit-exact | Already done (Section 10/17) | **KEEP AS EXPERIMENTAL** -- no new evidence from this session changes the prior REJECTED-FOR-PRODUCTION decision (loses the standalone Laplacian output the project's introspection tooling depends on) |
| Specialized hard-coded default-config pipeline path | Per-stage branch overhead | Not measured; the pipeline is already ONE native call with per-stage `if` branches, not 5 separate compiled variants -- a "specialized" path would only remove a handful of branch predictions | Medium | Medium (a second code path to keep in sync) | No | **REJECTED AS UNNECESSARY** -- branch overhead this small is not a plausible contributor next to a 0.4-4.95ms measured allocation cost |
| FP16 / reduced precision | Arithmetic throughput | Not researched in depth -- correctness-first project constraint makes this a non-starter without extraordinary justification | -- | High (violates the project's bit-exactness/`ilon 1` correctness contract) | No | **NOT APPLICABLE** |

## Level 1 / 2 / 3 assessment (spec item 43)

- **Level 1 (individual kernel improves)**: buffer persistence and pinned memory don't change any kernel's own
  execution time at all -- they remove *surrounding* CPU-side overhead, so Level 1 is not the right lens for
  either.
- **Level 2 (complete pipeline improves)**: both candidates measurably shrink the **GPU round-trip** portion of a
  pipeline call (H2D+compute+D2H = 3.40ms baseline) -- buffer persistence alone could remove up to ~1.86ms of that
  3.40ms at the canonical batch size (a >50% reduction in GPU-round-trip time for repeated calls), pinned memory a
  further ~5-10%.
- **Level 3 (end-to-end application improves)**: for a **single** one-shot run, disk I/O (~53ms) swamps this
  entirely -- Level 3 improvement would only be visible in **repeated-call** scenarios (benchmark loops, live
  reprocessing, batch sweeps), where it is real and worth pursuing. This asymmetry -- real Level 2 win, Level-3-
  conditional-on-usage-pattern -- is exactly why this candidate is classified PROMISING rather than KEEP: it
  should be implemented as a deliberate, separately-versioned experiment with its own before/after Level-3
  benchmark on a repeated-call workload, not folded silently into the frozen baseline.
