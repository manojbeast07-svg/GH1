# Section 20D — CUDA Graph Pipeline Optimization Experiment

New artifacts this section: `cuda/include/pipeline_cuda_graph.cuh` / `cuda/src/pipeline_cuda_graph.cu`
(new, isolated `CudaGraphPipeline` class), `tests/test_cuda_graph_pipeline.py` (29 tests, all passing),
`scripts/benchmark_cuda_graphs.py`, `benchmark_results/research_optimization/cuda_graphs/`. No
`__global__` kernel is modified or duplicated with different math. `cuda/pipeline.py`, `ui/services.py`,
`pipeline_basic.cu`, `pipeline_experimental.cu` (Section 20B/20C) are completely unmodified — verified
directly (`test_production_pipeline_functions_unaffected`, `test_class_does_not_import_or_reference_persistent_pipeline`).
Deliberately isolated from Section 20B/20C's buffer-persistence/pinned-memory work per spec items 16-18:
`CudaGraphPipeline` never imports or references `PersistentCudaPipeline`, and its baseline comparisons use
the true stateless production pipeline plus its own uncaptured (`use_graph=False`) path, never a persistent-buffer variant.

## Research finding that shaped this section's scope (spec item 3)

Neither of the project's two existing kernel-dispatch paths is graph-capturable as written:

1. **`run_basic_cuda_pipeline_batch()`'s own basic-kernel branch** calls `cudaDeviceSynchronize()`
   after every single stage (`pipeline_basic.cu:71,99,127,161,184`) to read back that stage's
   `CudaTimer` immediately. `cudaDeviceSynchronize` is not a stream-ordered operation and is illegal
   during stream capture — capturing this exact code path would abort capture at the first stage.
2. **All five `*_enhanced_dispatch()` functions** (`gaussian_enhanced_dispatch`, `median_enhanced_dispatch`,
   `sobel_enhanced_dispatch`, `laplacian_enhanced_dispatch`, `threshold_enhanced_dispatch`) take no
   `cudaStream_t` parameter and launch on the implicit default stream rather than an explicit capturing
   stream, and some variants call synchronous `cudaMemcpy`/`cudaMemcpyToSymbol` internally for per-call
   coefficient upload (confirmed by direct inspection: `gaussian_enhanced.cu:302`, `laplacian_enhanced.cu:323`,
   `sobel_enhanced.cu:286-287`). A kernel launched on a stream other than the one being captured is not
   captured at all (silently runs outside the graph, no error); a synchronous `cudaMemcpy` on the
   capturing stream aborts capture outright.

Making either path capturable would require modifying shared production dispatch code used by
`run_basic_cuda_pipeline_batch()` and by Section 20B/20C's `PersistentCudaPipeline` — out of this
section's safe-modification scope. **Consequence: this experiment captures the five BASIC (non-Enhanced)
`__global__` kernels only** (`gaussian_basic_kernel`, `median_basic_kernel`, `sobel_basic_kernel`,
`laplacian_basic_kernel`, `threshold_basic_kernel`), reimplemented as new, isolated host-side
orchestration around the *existing, unmodified* kernel functions. **Enhanced-pipeline graph capture is
not offered by this section's `CudaGraphPipeline` class at all** — this is a disclosed scope limit, not
an oversight, and is the single largest caveat on this section's decision (see below).

## Design

- `CudaGraphPipeline` owns one explicit `cudaStream_t` and a cache (`std::unordered_map`) of captured
  `cudaGraphExec_t` entries keyed by `(batch_size, height, width, scope, enabled-stage mask, kernel
  sizes, sobel_mode, laplacian_scale/delta bit patterns, threshold_value, threshold_max_value)`.
- Two capture scopes, both implemented and tested: **KernelsOnly** (spec Variant B — H2D/D2H stay
  plain `cudaMemcpyAsync` calls outside the graph) and **FullPipeline** (spec Variant C — H2D + all
  enabled kernels + D2H captured in one graph).
- **Buffer-address handling**: a captured `cudaMemcpyAsync` node freezes the host/device pointers
  current at capture time, but every real call passes a *different* NumPy array (different host
  address) and, for FullPipeline, needs a *different* output buffer. This is handled via
  `cudaGraphExecMemcpyNodeSetParams1D()` — the documented, standard mechanism for "same graph
  structure, new pointers" reuse — applied to the H2D node, the D2H node, and the Gaussian/Laplacian
  coefficient-upload nodes before every `cudaGraphLaunch()`. This is *not* a form of persistent-buffer
  reuse: the device-side buffers inside one cache entry stay at fixed addresses by construction (that is
  what makes a captured graph replayable at all), but nothing about a *caller's* host memory needs to
  persist across calls, unlike Section 20B/20C's design.
- Any other configuration change (batch size, resolution, enabled-stage mask, kernel sizes, or a scalar
  baked directly into a kernel's launch arguments such as `threshold_value`/`sobel_mode`) looks up a
  different cache entry, recapturing on a miss rather than patching kernel-node arguments in place.
  `cudaGraphExecKernelNodeSetParams` could reduce recapture frequency further for scalar-only changes —
  **not implemented in this section**, a disclosed, deliberate scope limit favoring simplicity/safety
  over minimizing recapture count.
- `run(..., use_graph=False)` executes the identical kernel sequence on the same explicit stream without
  capture, giving a second, code-level-matched baseline alongside the true production pipeline, so the
  benchmark can isolate exactly the graph-vs-no-graph variable.
- **Fallback on capture failure** (spec item 31): `build_entry()`'s capture body is wrapped so any
  exception or a failing `cudaStreamBeginCapture`/`cudaStreamEndCapture`/`cudaGraphInstantiate` returns
  `nullptr` with a `fallback_reason` string; `run()` then transparently re-executes via the non-graph
  path and surfaces `used_graph=false` plus the non-empty `fallback_reason` — never silently pretends a
  graph was used. Under every valid Basic-kernel configuration exercised in this section (the full
  space this class supports, now bounds-checked identically to production — see below), capture never
  actually failed, so this path is verified by code review and structure rather than an observed
  real-world trigger; it exists as required defensive engineering, not because a failure was found.

## Correctness

All bit-exact vs. `cuda.pipeline.run_basic_cuda_pipeline()` (`tests/test_cuda_graph_pipeline.py`, 29 tests):
- No-graph path, first capture, and cache-hit replay — including replay with a **different NumPy array**
  (different host address) than the one that triggered capture, directly proving the memcpy-node
  pointer patch works, not just the capturing call (`test_graph_replay_bit_exact_vs_production_with_new_host_pointers`).
- 50 consecutive varied-seed replays, all bit-exact.
- Batch-size change, resolution change (including reverting to a previous resolution), enabled-stage-mask
  change, and threshold-value change all correctly create a new cache entry and recompute correctly
  different output — verified against production for each.
- Parameter validation mirrors production's own bounds checks (median/gaussian/laplacian kernel size
  parity+range, `sobel_mode` range) — added to `run_experimental_cuda_graph_binding` in this section
  after noticing the initial binding omitted them; a rejected call leaves the pipeline fully usable
  afterward (`test_pipeline_remains_usable_after_a_validation_error`).

## Memory stability and isolation

- 80 calls (20 warmup + 60 measured) across 4 varying shapes: free VRAM stays **exactly constant** after
  the cache reaches its steady 4-entry state (0 bytes drift, `test_memory_stable_across_60_varied_calls`).
- `release()` frees GPU memory (verified via `device_memory_info()`), is idempotent, and the destructor
  cleans up correctly without an explicit `release()` call.
- Two independent `CudaGraphPipeline` instances never share a cache; releasing one has no effect on the
  other's correctness or availability.
- Direct behavioral check that production is unaffected: `run_basic_cuda_pipeline()` produces identical
  output before and after exercising `CudaGraphPipeline`, and source inspection confirms
  `cuda/pipeline.py` never references `CudaGraphPipeline`.

## Nsight Systems profiling — NOT AVAILABLE this session

Attempted `nsys profile --trace=cuda` on both the graph-vs-normal comparison script and a trivial
`xray_cuda.smoke_test()` control; both produced **empty CUDA trace data** (`PROCESSED (EMPTY RESULTS)` /
`does not contain CUDA trace data`). `nsys status -e` confirms `Administrator privileges: No` and
`Sampling Environment: Fail` in this session — unlike Section 20A, even CUDA API-level tracing (not just
CPU sampling) is unavailable here without Administrator rights, which are not available in this
environment. This section's evidence therefore relies entirely on the CPU-timer-based measurements below
(`capture_ms`/`instantiate_ms`/`node_update_ms` fields plus wall-clock benchmark results), not on an
Nsight timeline. Disclosed as NOT AVAILABLE rather than fabricated, consistent with this project's
handling of Nsight Compute in earlier sections.

## Batch-size sweep (real dataset, seed=42, 224x224, 3 warmup + 7 measured runs)

`gain_vs_production` = true production median / variant median. `gain_vs_no_graph` = this class's own
uncaptured path median / variant median (the controlled, code-matched comparison).

| batch | production (ms) | graph FULL vs prod | graph FULL vs no-graph | graph KERNELS-ONLY vs prod | break-even (calls) |
|---:|---:|---:|---:|---:|---:|
| 1   | 0.199 | **1.496x** | 3.115x | 1.298x | 0.27 |
| 8   | 0.508 | **1.205x** | 1.423x | 1.242x | 0.66 |
| 32  | 1.742 | **1.175x** | 1.136x | 1.211x | 0.51 |
| 64  | 3.162 | **1.121x** | 1.532x | 1.164x | 0.10 |
| 100 (5 independent runs) | 3.905–4.869 | **0.885x, 0.889x, 0.960x, 1.073x, 1.151x** (median ≈0.96x) | 1.207x–1.479x (all runs positive) | 0.874x–1.167x (median ≈1.01x) | 0.05–0.14 |

**Batch=100 required 5 repeated runs to characterize** because it showed materially higher run-to-run
variance than the other sizes — `gain_vs_production` ranged from 0.885x to 1.151x across runs, straddling
1.0x (a wash) rather than showing the smaller sizes' consistent, clean win. Crucially, `gain_vs_no_graph`
(the code-matched, apples-to-apples comparison) stayed positive in **every single one of the 5 runs**
(1.21x–1.48x) — the graph mechanism itself reliably helps; what varies at this batch size is whether that
help is large enough to clear the noise floor of small implementation differences between this class's own
host code and production's. This matches the research's own prediction: kernel-launch-submission overhead
is a small, roughly *fixed* per-call cost (~0.33ms/call from Section 20A's measurement), so its relative
share of total time shrinks as batch size (and therefore per-call kernel compute time) grows — clear win at
small-to-medium batches, noise-level parity at the largest batch tested.

Every `gain_vs_no_graph` figure across every batch size and both scopes was measured **positive** — the
graph mechanism never once regressed relative to the identical uncaptured code path in this section's
testing, across dozens of runs.

## Realistic Streamlit workload (spec item 9) — THE DECISIVE MEASUREMENT

Reused Section 20C's fixed measurement methodology exactly (the same one that found persistent buffers'
28% regression), so this section's result is directly comparable and not an artifact of a differently-
biased method: `xray_cuda.smoke_test()` context warmup, 2 discarded full-sequence warmup passes for
**both** variants, 5 measured passes with alternating execution order. Sequence per pass: select
batch(32) → process → change threshold (same shape) → process → select batch(8) → process → select
batch(64) → process. Measured backend wall time only (`time.perf_counter()`, never Streamlit rendering).

Run twice independently (once as part of the combined batch-sweep+realistic invocation, once standalone):

| Run | Stateless (production) median | Graph (`CudaGraphPipeline`) median | Gain |
|---|---:|---:|---:|
| 1 | 25.283 ms | 17.604 ms | 1.436x |
| 2 | 20.716 ms | 13.286 ms | 1.559x |

**CUDA Graphs are consistently ~1.44x–1.56x FASTER than stateless production under this realistic,
varied workflow** — both independent runs agree, and both are unambiguous wins. This is the **opposite**
outcome from Section 20C's persistent-buffer finding under the identical style of test. The architectural
reason: a captured graph's benefit depends only on its **configuration recurring** (looked up by cache
key), not on any host-side state surviving between calls — and this realistic sequence's own varied,
back-and-forth parameter changes (batch(32)@threshold=128, then @threshold=200, then batch(8), then
batch(64)) are exactly repeated on every subsequent pass, so from the second pass onward every step is a
pure cache-hit replay. Persistent buffers needed the *literal buffer contents* to still be the right shape
call-to-call, which rarely holds in varied usage; graphs only need the *same configuration* to recur at
some point, which interactive tuning (adjusting a parameter, then reverting it; switching between a couple
of batch sizes) does far more often than a monotonically-varying workload would.

Correctness of the realistic-workload sequence was verified directly and separately (not inferred from the
timing run): both the `threshold=128` and `threshold=200` variants of the `batch(32)` step were confirmed
bit-exact against `run_basic_cuda_pipeline()` for their respective configs, and the two outputs were
confirmed to differ from each other (sanity check that the parameter change actually took effect through
the cache-miss recapture).

## The one major caveat: Enhanced pipeline is not covered

This section's `CudaGraphPipeline` supports the Basic-kernel pipeline only (see the research finding
above). The application's Enhanced pipeline — the pipeline most demonstrations and comparisons in this
project actually highlight — gets **no graph support from this section's work**. Extending capture to
Enhanced would require adding an optional `cudaStream_t` parameter to all five `*_enhanced_dispatch()`
functions and converting their internal synchronous `cudaMemcpy`/`cudaMemcpyToSymbol` calls to async
equivalents on that stream — a real, nontrivial, and disclosed piece of follow-on engineering, not
attempted here per this section's own safe-modification-scope boundary.

## Decision

```
CUDA GRAPHS (Basic pipeline) → EXPERIMENTAL
(keep as a documented, tested, isolated capability; not wired into production this section)
```

**Reasoning**: Unlike Section 20C's persistent buffers (cleanly and consistently REJECTED under the
realistic workload), CUDA Graphs pass the decisive realistic-workload test convincingly — twice,
independently, ~1.44x–1.56x. The mechanism is architecturally well-suited to genuine interactive use
(configuration-keyed caching survives exactly the kind of back-and-forth parameter tuning a real user
does), the correctness and memory-safety engineering are sound (29/29 tests passing, zero leaks, zero
cross-instance contamination), and every measured `gain_vs_no_graph` figure across the whole batch-size
sweep was positive. This is real, reproducible evidence in favor of the mechanism.

It stops short of ADOPT for two concrete, disclosed reasons rather than the workload evidence itself:

1. **Enhanced-pipeline coverage gap.** This section's implementation only accelerates the Basic pipeline.
   Wiring graphs into the application without Enhanced support would mean the optimization is invisible
   for a large share of real usage, or would require the follow-on dispatch-function work described above
   before a production integration decision could be made honestly.
2. **Large-batch parity, not dominance.** At the largest batch size tested (100, near this project's
   canonical 122), `gain_vs_production` straddled 1.0x across 5 runs (0.885x–1.151x, median ≈0.96x) —
   a wash, not a clear win, even though the code-matched `gain_vs_no_graph` comparison stayed positive
   throughout. A production adoption claim resting on "faster at every batch size" would not be honest
   at this project's own canonical scale.

**The experimental code is not deleted** — `cuda/include/pipeline_cuda_graph.cuh`/`cuda/src/pipeline_cuda_graph.cu`,
`tests/test_cuda_graph_pipeline.py`, and `scripts/benchmark_cuda_graphs.py` remain in the repository as a
documented, isolated, working capability. A future section extending `*_enhanced_dispatch()` to accept an
explicit stream (and converting their internal coefficient uploads to the async equivalents) would remove
this section's main caveat and could reasonably revisit this as an ADOPT candidate, ideally re-measured at
the canonical batch size with more repeated runs given the variance observed there.

## Production API recommendation

**No change to `cuda/pipeline.py` or `ui/services.py`.** `run_basic_cuda_pipeline()` and
`run_enhanced_cuda_pipeline()` remain the sole production entry points, exactly as before this section.
