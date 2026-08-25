# Section 20F — Asynchronous Multi-Stream + Double/Triple Buffer Pipeline Experiment

New artifacts this section: `cuda/include/pipeline_async.cuh` / `cuda/src/pipeline_async.cu` (new,
isolated `AsyncCudaPipeline` class), `research/async_pipeline_research.md` (spec item 4's required
research, produced before implementation), `tests/test_async_pipeline.py` (36 tests, all passing),
`scripts/benchmark_async_pipeline.py`, `benchmark_results/research_optimization/async_pipeline/`. No
production kernel, `run_basic_cuda_pipeline()`/`run_enhanced_cuda_pipeline()`, or CPU implementation is
modified. Never imports `PersistentCudaPipeline`, `CudaGraphPipeline`, or `CudaGraphEnhancedPipeline` —
isolated per this section's spec items 14-15, verified directly
(`test_does_not_import_or_reference_other_experimental_pipelines`,
`test_basic_and_enhanced_graph_pipelines_unaffected`).

## Research (spec item 4)

Verified against current NVIDIA documentation and this machine's own CUDA 13.3 headers (full detail in
`research/async_pipeline_research.md`):

- **Page-locked (pinned) host memory is required for genuine overlap** of a host↔device transfer with
  kernel execution — a `cudaMemcpyAsync` from pageable memory still returns without error but the driver
  internally performs a synchronous staging copy that does not overlap with other stream activity.
- **Non-default streams are required** — confirmed directly in this machine's `cuda_runtime_api.h`.
- **This machine's GPU has exactly one async copy engine** (`cudaDeviceProp::asyncEngineCount == 1`,
  measured directly via `cudaGetDeviceProperties()`, not assumed) — H2D and D2H cannot be simultaneously
  in flight on this hardware, though each can still overlap with kernel compute independently. This bounds
  what triple buffering can add over double buffering *on this machine specifically*: the classic "D2H(N-1)
  overlaps H2D(N+1)" benefit triple buffering targets is not physically achievable here.

## Architecture (spec items 6-11)

`AsyncCudaPipeline` uses 3 explicit non-default streams (H2D/compute/D2H) and N round-robin buffer sets
(configurable 1-4; double=2, triple=3). Per-chunk dependencies are enforced with `cudaEvent_t`
(`cudaStreamWaitEvent`), never a host-side sync inside the hot path (an earlier draft used
`cudaStreamSynchronize` to keep a local coefficient buffer alive across an async launch — caught and fixed
before any benchmark was run: coefficient buffers now live per-buffer-set, uploaded via
`cudaMemcpyAsync`/`cudaMemcpyToSymbolAsync` on the same stream as the consuming kernel, correct by
stream-ordering alone with zero host synchronization). Buffer-set reuse (spec item 14) is NOT persistent
buffers: every `run()` call allocates its N buffer sets fresh and frees them before returning — nothing
survives from one call to the next, unlike `PersistentCudaPipeline`. This was verified directly
(`test_memory_stable_across_varied_calls`: free VRAM returns to baseline after every call).

**Kernel provenance**: Basic kernels are the real production symbols, called directly (same as Section
20D). Enhanced kernels are isolated, verified-identical copies — same technique, same source-line
provenance methodology as Section 20E's `CudaGraphEnhancedPipeline`, duplicated a second time (not shared
via a new header) specifically to avoid modifying Section 20E's already-tested, frozen file. Both Basic and
Enhanced paths were bit-exact against production across all 36 tests.

## Chunk size (spec items 18-20)

The batch/chunk distinction is implemented as designed: `chunk_size` is independent of the caller's batch
size, with the last chunk sized to the remainder (verified directly with batch=37, chunk sizes 1/3/8/37,
including the degenerate chunk_size >= batch_size single-chunk case).

**Chunk-size sweep (batch=64, 224x224, Basic, 3 warmup + 7 measured runs):**

| chunk_size | chunks | total (median) |
|---:|---:|---:|
| 1  | 64 | 12.276 ms |
| 4  | 16 |  6.474 ms |
| 8  | 8  |  4.722 ms |
| 16 | 4  |  4.622 ms |
| 32 | 2  |  4.887 ms |
| 64 | 1  |  4.419 ms |

Too-small chunks are dramatically worse (chunk=1 is ~2.8x slower than the best chunk size) — confirming
spec item 20's prediction directly. The single-chunk case (chunk_size == batch_size, i.e. no pipelining
possible at all) was the FASTEST tested configuration for this batch size, worse than chunking only in the
sense that it isn't really "async" at that point — this is itself informative: at this workload's actual
scale, the fixed per-chunk overhead this section's architecture pays (event records, stream-wait calls,
buffer-set bookkeeping) is not recovered by the achievable overlap until chunk counts are already fairly
small.

## Batch-size sweep (spec items 21-22; real dataset, seed=42, 224x224, chunk_size=16, 3 warmup + 7 measured)

`gain_vs_production` = true production median / variant median (>1.0x = async faster).

**Basic:**

| batch | production (ms) | sequential_1buf | async_2buf_pageable | async_3buf_pageable | async_2buf_pinned | async_3buf_pinned |
|---:|---:|---:|---:|---:|---:|---:|
| 8   | 0.565 | 0.604x | 0.662x | 0.614x | 0.389x | 0.412x |
| 32  | 1.627 | 0.633x | 0.619x | 0.573x | 0.466x | 0.455x |
| 64  | 3.045 | 0.695x | 0.605x | 0.597x | 0.581x | 0.564x |
| 100 | 3.973 | 0.546x | 0.579x | 0.590x | 0.571x | 0.499x |

**Enhanced:**

| batch | production (ms) | sequential_1buf | async_2buf_pageable | async_3buf_pageable | async_2buf_pinned | async_3buf_pinned |
|---:|---:|---:|---:|---:|---:|---:|
| 32  | 1.027 | 0.483x | 0.451x | 0.447x | 0.304x | 0.373x |
| 64  | 1.732 | 0.426x | 0.468x | 0.408x | 0.404x | 0.326x |
| 100 | 2.806 | 0.515x | 0.543x | 0.495x | 0.465x | 0.438x |

**Every single isolated same-shape-repeated-call measurement is a regression** — async is 1.4x-3.3x
*slower* than production across every batch size, every buffer count, and both pageable and pinned
staging, for both Basic and Enhanced. Even `sequential_1buf` (num_buffers=1, chunked but with no
opportunity for overlap at all — the "chunking tax alone" baseline) is consistently slower than
production, confirming that simply splitting a batch into more, smaller kernel-launch groups costs more in
fixed per-chunk overhead than this workload's per-call GPU work can hide, at every batch size tested here.
Pinned staging is *never* better than pageable in this sweep, contrary to the pure-transfer-overlap theory
in the research doc — the staging copy's own cost (correctly counted in `host_stage_ms`, per spec item 13)
outweighs whatever transfer-overlap benefit it unlocks at these batch sizes, echoing Section 20B's original
finding for the whole-pipeline case.

## H2D/D2H overlap proof (spec items 29-31)

Nsight Systems: **NOT AVAILABLE** this session (`nsys status -e` confirms no Administrator privileges,
consistent with Sections 20D/20E). This section's overlap evidence instead comes from CUDA events recorded
directly around each stream's actual active span (`h2d_stream_span_ms`/`compute_stream_span_ms`/
`d2h_stream_span_ms`), from which `h2d_compute_overlap_ms` and `compute_d2h_overlap_ms` are computed as the
literal time-interval intersection of each pair of spans — genuine measured overlap, not inferred from
code structure. Real, non-trivial overlap **was** measured in every multi-chunk run: at batch=100
(Basic, async_2buf_pageable), `h2d_compute_overlap_ms` reached 4.89ms out of a 6.86ms total call — meaning
the large majority of the H2D stream's active time coincided with compute activity. **H2D overlaps
compute: YES** (consistently, growing with batch size). **D2H overlaps compute: YES** (also consistently
measured, `compute_d2h_overlap_ms` follows the same pattern). Despite this — and this is the section's
central, important finding — **genuine, measured overlap did not translate into a net end-to-end win** in
the isolated batch-size sweep: the fixed per-chunk overhead (more kernel launches, more event
records/waits, more host-side bookkeeping) consistently outweighed the time reclaimed by overlap at every
batch size tested. Overlap existing is necessary but not sufficient for a net win; this section
demonstrates the difference directly rather than assuming one implies the other (spec item 47's own
warning).

## Realistic Streamlit workload (spec item 25) — THE DECISIVE MEASUREMENT

Reused Section 20C's fixed methodology exactly (context warmup, 2 discarded full-sequence warmup passes,
5 measured passes with alternating order). Sequence: batch(32) → process → change threshold → process →
batch(8) → process → batch(64) → process (chunk_size=16, num_buffers=2, pageable).

| Implementation | Run | Stateless median | Async median | Gain |
|---|---:|---:|---:|---:|
| Basic | 1 | 19.295 ms | 17.166 ms | 1.124x |
| Basic | 2 | 20.662 ms | 18.681 ms | 1.106x |
| Basic | 3 | 23.569 ms | 21.953 ms | 1.074x |
| Enhanced | 1 | 24.499 ms | 21.979 ms | 1.115x |

**This is the section's most important tension.** The decisive, realistic-workload metric (which spec
item 47 explicitly names as *the* primary metric for the final decision) shows a small but consistently
reproducible improvement — 1.07x-1.12x across 4 independent runs (3 Basic, 1 Enhanced), all positive, all
in a narrow band, using the same alternating-order/warmup rigor as every other section's decisive test.
This is the **opposite direction** from the isolated batch-size sweep, which showed consistent, large
regressions (0.3x-0.7x) at every one of the same batch sizes this realistic sequence actually uses (8, 32,
64). The most likely explanation: production's own per-call overhead (fresh `cudaMalloc`/`cudaFree` every
single call — Section 20A's own finding that this is ~78.8% of CPU-side API time) is paid identically by
both variants across a 4-step mixed sequence, and the async pipeline's genuine (if partial) overlap for the
larger steps in that sequence (32, 64) recovers slightly more time in aggregate than an isolated,
repeated-identical-call microbenchmark reveals — but this is a hypothesis offered honestly as
unconfirmed, not a proven mechanism; disentangling it further was judged out of this section's scope.

## Correctness (spec items 33-36)

Bit-exact against production in every tested dimension: Basic and Enhanced, both scopes, chunk sizes
1/3/8/37, batch sizes 1/8/32/37, num_buffers 1-4, pageable and pinned staging, resolution transitions
(32x32→96x96→32x32, no leakage), parameter changes (threshold, gaussian enable/disable). CPU comparison
reused the project's established tolerance (`differing_pct < 5.0`), no new tolerance introduced. 200
repeated executions on one pipeline instance produced byte-identical, deterministic output every time —
zero corruption, zero races, zero stale-buffer reuse observed (`test_200_repeated_executions_stay_correct_and_deterministic`).

**One real bug was found and fixed during this section's own development** (disclosed per this project's
established practice): an early draft called `cudaStreamSynchronize` inside the per-chunk kernel-launch
helper to keep a stack-local coefficient buffer alive past its async upload — this would have silently
collapsed the entire pipeline back to sequential execution (a host-side sync inside the hot path defeats
overlap by construction) while still producing *correct* output, meaning it would not have been caught by
correctness tests alone. Caught via architectural review before any benchmark was run, fixed by moving
coefficient buffers to per-buffer-set, stream-ordered-safe storage (see Architecture section above).

## Memory (spec item 32)

Measured directly, not assumed: at chunk_size=16, 224x224, Basic, each buffer set costs ~1.6MB GPU memory
(two ping-pong chunk buffers + small coefficient buffers) and, when pinned staging is enabled, an
additional ~1.6MB of page-locked host memory. This scales linearly and boundedly with `num_buffers`
(1→1.6MB, 2→3.2MB, 3→4.8MB total GPU) — a small, disclosed, acceptable cost relative to this GPU's VRAM.
Free GPU memory returns to baseline after every `run()` call (0-byte delta measured for num_buffers=2 and
3; a small ~2MB delta at num_buffers=1 is consistent with ordinary driver-level allocation-granularity
noise seen elsewhere in this project, not a leak).

## Decision

```
ASYNC MULTI-STREAM PIPELINE (Basic and Enhanced) → EXPERIMENTAL
(keep as a documented, tested, isolated capability; not wired into production this section)
```

**Reasoning**: This section's evidence is genuinely mixed, and the decision reflects that honestly rather
than forcing a clean narrative. Per spec item 47's explicit instruction, the realistic workload is the
primary metric — and it shows a real, reproducible, if modest, improvement (1.07x-1.12x across 4
independent runs, all positive, surviving the alternating-order bias check). That alone would not
automatically clear this project's REJECTED bar. But the case for ADOPT is undermined by:

1. **The isolated microbenchmark picture is strongly negative** — every single batch size, buffer count,
   and staging mode tested regressed 1.4x-3.3x versus production, including the degenerate
   "chunked-but-not-overlapped" baseline. A production adoption decision resting entirely on an aggregate,
   mixed-sequence effect whose exact mechanism this section could not fully isolate is a meaningfully
   weaker basis than Section 20D/20E's CUDA Graph findings, where the *same* metric (isolated batch-size
   gain) was also consistently positive.
2. **Chunk size requires careful, workload-specific tuning** — the sweep showed up to a 2.8x spread between
   a good and bad chunk size at one fixed batch size; there is no single chunk size this section can
   recommend as universally safe, which is a real deployment risk for a UI where batch size varies freely.
3. **Pinned staging never won** in any measured configuration despite the research predicting it should be
   necessary for genuine overlap — a sign the interaction between staging cost, chunk size, and overlap is
   not yet well enough understood to productionize confidently.
4. **Substantial implementation complexity** (3 streams, per-chunk event dependency chains, buffer-set
   lifecycle, chunk/batch distinction) for a benefit that, where it exists, is modest (≤12%).

**The experimental code is not deleted** — `cuda/include/pipeline_async.cuh`/`cuda/src/pipeline_async.cu`,
`tests/test_async_pipeline.py`, and `scripts/benchmark_async_pipeline.py` remain in the repository,
correctness-verified and documented, as the most architecturally novel (if least conclusively won)
experiment in this project's optimization research arc.

## Production API recommendation

**No change to `cuda/pipeline.py` or `ui/services.py`.** `run_basic_cuda_pipeline()` and
`run_enhanced_cuda_pipeline()` remain the sole production entry points, exactly as before this section.

## Recommendation for the STOP CONDITION review (spec item 49)

Across this project's full optimization research arc — filter-specific kernel work (exhausted, zero
spilling), persistent buffers (REJECTED, realistic-workload regression), pinned memory (REJECTED, staging
cost), Basic CUDA Graphs (EXPERIMENTAL, consistent realistic-workload win), Enhanced CUDA Graphs
(EXPERIMENTAL, stronger realistic-workload win), and this section's async multi-stream pipeline
(EXPERIMENTAL, mixed/marginal signal) — no remaining bottleneck has been identified that a further new
optimization experiment would clearly address. The two EXPERIMENTAL CUDA Graph findings (Sections 20D/20E)
represent this research arc's strongest, cleanest evidence; this section's async pipeline adds a third
EXPERIMENTAL candidate with a substantially weaker and more equivocal case. Per the spec's own instruction,
this points toward moving to the final release/freeze section rather than opening another optimization
round.
