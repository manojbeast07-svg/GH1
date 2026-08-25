# Section 20B — Buffer Persistence + Pinned Host Memory: Decision Record

Full experiment: `cuda/include/pipeline_experimental.cuh`, `cuda/src/pipeline_experimental.cu`,
`xray_cuda.PersistentCudaPipeline` (pybind11 class, `cuda/src/bindings.cpp`),
`scripts/benchmark_persistent_pinned.py`, `tests/test_persistent_pinned_pipeline.py` (21 tests),
results in `benchmark_results/research_optimization/persistence_pinned/`.

## What changed from Section 20A's read of the evidence

Section 20A measured pinned memory in isolation (a pinned buffer used directly as the `cudaMemcpy` source,
median ~1.8-1.9x speedup at the canonical batch size) and extrapolated that into the pipeline. Section 20B builds
the real integration and finds a materially different, more complete picture: because the pipeline's input is
always a NumPy array (necessarily pageable — NumPy does not allocate CUDA pinned memory), using pinned memory
requires an **extra staging copy** (pageable → pinned) before H2D, and another (pinned → pageable) after D2H.
Measuring that staging cost honestly (spec item 14's explicit instruction — "do not hide staging-copy cost")
changes the conclusion: pinned memory alone is break-even-to-regression in the full pipeline, and combining it
with buffer persistence is inconsistent (sometimes slightly better than persistence alone, sometimes
meaningfully worse) rather than additive. This is disclosed as the correction it is, not smoothed over.

## Measured results (real dataset, seed=42, 224x224, 3 warmup + 10 measured runs, median timings)

| Batch | Baseline (ms) | Persistent (ms) | Persistent gain | Pinned (ms) | Pinned gain | Persistent+Pinned (ms) | Combined gain |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.350 | 0.216 | **1.62x** | 0.374 | 0.94x | 0.193 | **1.82x** |
| 8 | 0.541 | 0.354 | **1.53x** | 0.543 | 1.00x | 0.652 | 0.83x |
| 32 | 1.241 | 0.800 | **1.55x** | 1.263 | 0.98x | 0.823 | **1.51x** |
| 64 | 2.244 | 1.958 | **1.15x** | 2.450 | 0.92x | 1.897 | **1.18x** |
| 122 (canonical) | 3.491 | 2.575 | **1.36x** | 4.276 | 0.82x | 3.724 | 0.94x |

All 4 variants verified **bit-exact** against each other and within the established differing-pixel-percentage
tolerance vs. CPU at every batch size (full pipeline, Enhanced-equivalent configuration) -- see
`tests/test_persistent_pinned_pipeline.py`.

**Allocation overhead recovered** (spec item 17): baseline `alloc_ms` at the canonical batch size was 0.645ms;
persistent variants show ~0.0003-0.0008ms (essentially eliminated) after the first (cold) call. This matches
Section 20A's standalone-experiment finding directionally, though the in-situ full-pipeline numbers are smaller
in absolute terms than the isolated micro-benchmark suggested (H2D/D2H/compute are also happening and dominate
more of the total at larger batches).

**H2D/D2H improvement**: pinned memory's raw transfer time IS consistently faster than pageable (e.g. at
batch=122: H2D 0.613ms→0.514ms, D2H 0.607ms→0.522ms) -- exactly matching Section 20A's isolated finding. The
staging copy simply costs more than that transfer saving recovers (at batch=122, pinned's total `host_stage_ms`
was ~1.3ms alone -- more than double the ~0.18ms saved on H2D+D2H combined).

**GPU compute**: unaffected by either optimization, as expected (spec item 23) -- all four variants call the
identical kernel dispatch sequence.

**Nsight Systems** (Variant A vs Variant D, `research/raw_evidence/nsys_variant_a_vs_d_cuda_api_sum.csv`):
confirms the mechanism directly -- `cudaMalloc` count matches "3 buffers × 4 calls" for the non-persistent
portion of the trace and does not grow further during the persistent portion; `cudaHostAlloc`/`cudaFreeHost`
each appear exactly twice (input + output pinned buffers, allocated once, freed once at `release()`).

## Correctness and stability (spec items 24-29)

- All 4 variants bit-exact vs. each other, every batch size tested (1-256).
- 50-200 repeated identical-input calls: bit-exact output every time, **zero measured GPU memory leak** (device
  free-memory delta after `release()`: -6.29MB, i.e. *more* free than before -- no leak).
- Capacity growth (8→32→128) and shrink (128→16→64→4) within one process: correct output at every step after
  fixing a real bug this section found (see below).
- Resolution change (224x224 → 96x96 → back to 224x224) within one process: correctly reallocates rather than
  reusing an incompatible buffer, verified bit-exact at each step.
- Error handling: empty batch, wrong dtype, wrong ndim all raise `ValueError` cleanly (no crash, no unreadable
  native error).

**Bug found and fixed this session**: shrinking the batch size after growing the persistent buffers (e.g.
128→16) initially raised `ValueError: Byte count mismatch` -- `GpuImageBatch::upload_from_host()`/
`download_to_host()` strictly validate the transfer size against the buffer's *full allocated* capacity, but a
persistent buffer can legitimately be larger than the current request (that's the whole point of not
shrinking/reallocating, spec item 8). Fixed by using a raw `cudaMemcpy` sized to the current request's
`total_bytes` against the buffer's front region, instead of the strict-checking wrapper method. Verified via the
capacity-change test above, now passing.

## Production decision (spec item 40)

### Buffer persistence: **KEEP AS EXPERIMENT**

Real, measurable, correctness-safe gain (1.15x-1.62x across every tested batch size, no regression observed at
any size in the fuller pipeline context — the batch=256 regression Section 20A's isolated allocation-only
micro-benchmark's context doesn't reproduce here once real GPU compute/transfer time dominates the denominator).
Not promoted to production in this section because: (1) it changes the pipeline's call contract (a
caller-managed, stateful `PersistentCudaPipeline` object vs. today's stateless `run_basic_cuda_pipeline()`
function) -- a real API-design decision that deserves its own deliberate review, not a silent swap; (2) this
project's benchmark methodology (Section 11) and Streamlit call sites were not built around a persistent-object
lifecycle, and retrofitting that is out of this section's scope (explicitly: "do not wire Variant D into
run_basic_cuda_pipeline()/run_enhanced_cuda_pipeline()").

### Pinned host memory: **REJECTED** (for this integration pattern)

Break-even to a measured 18% regression at the canonical batch size once the mandatory NumPy staging copy is
honestly counted. The isolated Section 20A finding was real but incomplete -- this section's fuller measurement
is the correct basis for the decision. Could plausibly still help IF the Python-side caller could hand over data
already in pinned memory (avoiding the staging copy entirely), but that would require rearchitecting the dataset
loading path (`pipeline.image_loader`) to allocate pinned buffers directly, which is a much larger change than
this section's scope and not justified by evidence gathered here.

### Combined (persistent + pinned): **REJECTED** as a combination

Never reliably better than persistence alone in this data (sometimes close, sometimes 25%+ worse) -- confirms
spec item 41's explicit warning not to assume additive speedups. If pinned memory is revisited later (e.g. via a
pinned-native data-loading path), it should be re-evaluated together with persistence again, not assumed
compatible based on this result.

## Streamlit integration (spec item 35)

**Not built.** Given the production decision is "keep as experiment, not adopt," and pinned memory (half of the
originally-proposed pair) is rejected outright, adding a UI toggle for a capability that isn't going into
production doesn't clear the bar of "add only if needed." The benchmark artifact
(`benchmark_results/research_optimization/persistence_pinned/latest.json`) is available for the Optimization Lab
or Performance Analytics to load and display later if that changes.

## What this means for Section 20A's pipeline matrix

`research/pipeline_optimization_matrix.md`'s "buffer persistence" and "pinned host memory" rows should be read
alongside this document as the more complete, in-situ measurement -- persistence remains PROMISING (now with a
real implementation and correctness proof behind it, still not adopted), pinned memory moves from PROMISING to
REJECTED now that the full integration cost is measured rather than estimated from an isolated benchmark.
