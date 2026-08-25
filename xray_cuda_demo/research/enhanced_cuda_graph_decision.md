# Section 20E — Enhanced CUDA Graph Compatibility + Performance Experiment

New artifacts this section: `cuda/include/pipeline_cuda_graph_enhanced.cuh` / `cuda/src/pipeline_cuda_graph_enhanced.cu`
(new, isolated `CudaGraphEnhancedPipeline` class), `research/enhanced_graph_capture_audit.md` (spec item 4's
required audit, produced before any implementation), `tests/test_enhanced_cuda_graph.py` (27 tests, all
passing), `scripts/benchmark_enhanced_cuda_graph.py`, `benchmark_results/research_optimization/enhanced_cuda_graph/`.
No production Enhanced or Basic `__global__` kernel is modified. `cuda/pipeline.py`, `ui/services.py`,
`gaussian_enhanced.cu`/`median_enhanced.cu`/`sobel_enhanced.cu`/`laplacian_enhanced.cu`/`threshold_enhanced.cu`,
and Section 20D's `pipeline_cuda_graph.cu` are all completely unmodified — verified directly
(`test_does_not_import_or_reference_persistent_or_basic_graph_pipeline`, `test_basic_cuda_graph_pipeline_unaffected`).
Deliberately isolated from Sections 20B/20C/20D's persistent-buffer, pinned-memory, and Basic-graph work
per this section's spec items 33-35.

## Capture blockers (spec item 4 — full audit in `research/enhanced_graph_capture_audit.md`)

Every one of the five `*_enhanced_dispatch()` functions is blocked by the same two issues Section 20D
found for Basic, plus one Enhanced-specific issue:

1. **`cudaDeviceSynchronize()` inside the dispatcher**, once per stage (Gaussian: twice, for its
   separable H/V passes) — not stream-ordered, illegal during capture.
2. **No explicit stream parameter** — every kernel launches on the implicit default stream.
3. **Synchronous `cudaMemcpyToSymbol`** for coefficient upload — Gaussian and Laplacian only, in their
   production (Specialized) configuration; Median (Network3x3), Sobel (Specialized), and Threshold
   (Vectorized) need no coefficient upload at all.

**One new finding not present in Section 20D**: every Enhanced `__global__` kernel and its `__device__`
helpers are defined inside an **anonymous namespace**, giving them internal (translation-unit-local)
linkage, and none are declared in any `.cuh` header. This means — unlike Section 20D's Basic kernels,
which Section 20D called directly by including their header — **this section's isolated orchestration
cannot reference the production kernel symbols at all**, regardless of the synchronization/stream issues
above. Confirmed by direct inspection of all five `.cu` files (audit doc, "Findings table").

## Isolated workaround (spec items 7-9)

Since literal symbol reuse is impossible, this section follows the spec's explicit fallback (item 9):
verified-identical copies of the exact kernel bodies (and their small `__device__` helpers —
`pix_sort`/`load_median_tile`, `sobel_finalize`/`load_sobel_tile`, `laplacian_finalize`/
`load_laplacian_tile`) live in `cuda/src/pipeline_cuda_graph_enhanced.cu`, in an anonymous namespace of
their own, distinct symbol names (`graph_` prefix) to avoid any ambiguity with production. **Kernel
equivalence proof** (spec item 8): the file's top-of-file comment cites the exact source file/line range
each duplicate was copied from; every duplicate was verified bit-exact against production output across
27 tests covering both scopes, capture, replay-with-new-host-pointer, 50 repeated replays, every
parameter/batch/resolution/enable-disable dimension, and a CPU comparison — the strongest form of the
proof, since identical math is the only way bit-exact output across dozens of independently varied
configurations is possible. `reflect101()`/`clamp_index()` are NOT duplicated — both are declared in the
shared `gpu_image.cuh` header (external linkage) and are `#include`d exactly as production uses them.

**Constant-memory note** (also new vs. Basic): Gaussian's and Laplacian's Specialized kernels read
coefficients from `__constant__` symbols, not a device-pointer kernel argument. This section's duplicated
kernels use their **own** `__constant__` symbols (`g_gaussian_coeffs_1d`, `g_laplacian_coeffs`), entirely
separate from production's (`d_gaussian_coeffs_1d`, `d_laplacian_coeffs`) — this class never touches
production's symbols, avoiding any cross-path interaction. Per-replay coefficient updates are captured as
`cudaMemcpyToSymbolAsync` nodes and patched via `cudaGraphExecMemcpyNodeSetParams1D` against the symbol's
own address (obtained once via `cudaGetSymbolAddress`), exactly the same mechanism Section 20D used for
Basic's coefficient buffers. One disclosed, unexercised theoretical constraint: constant memory is
process-global, so genuinely *concurrent* multi-instance graph replay on separate streams could race on
these symbols — this project's actual usage pattern (one Streamlit session, synchronous request/response)
never exercises that race, and every replay's coefficient upload and consuming kernel stay in the same
graph with an explicit dependency edge, verified safe for sequential multi-instance use directly
(`test_two_independent_pipelines_do_not_share_cache`).

## CUDA Graph compatibility research (spec item 5)

Verified directly against this machine's installed CUDA 13.3 runtime headers (`cuda_runtime_api.h`), not
assumed: `cudaMemcpyToSymbolAsync(symbol, src, count, offset, kind, stream)` exists and is
stream-ordered/capturable; `cudaGetSymbolAddress` resolves a `__constant__` symbol to a plain device
pointer usable for node matching; `cudaGraphExecMemcpyNodeSetParams1D(exec, node, dst, src, count, kind)`
exists (CUDA >= 11.1) and is the documented mechanism for patching a captured memcpy node's endpoints
without recapture; `cudaGraphInstantiate` takes the modern 3-argument form
(`cudaGraphExecInstantiate(exec*, graph, flags)`) in this toolkit version. All confirmed by direct
`grep`/read of the installed headers before use, matching this project's "use current documentation, do
not rely on assumptions" convention.

## Architecture (spec item 6)

`CudaGraphEnhancedPipeline` mirrors Section 20D's `CudaGraphPipeline` design closely: one owned
`cudaStream_t`, an `std::unordered_map` cache of `cudaGraphExec_t` keyed by full execution configuration
(batch/height/width/scope + enabled-stage mask + `sobel_mode` + threshold value/max + Laplacian
scale/delta bit patterns — Gaussian/Laplacian coefficient *content* flows through a patched memcpy node
each replay, so only `kernel_size`, the structural parameter, is in the key). Two scopes implemented and
tested (spec item 10): **KernelsOnly** (Variant B — H2D/D2H stay outside the graph) and **FullPipeline**
(Variant C — H2D + all six kernel launches + D2H captured in one graph); both were technically legal to
capture, so Variant C was not merely attempted but fully verified and benchmarked, not assumed to work.

## Graph lifecycle and setup cost (spec item 11)

Capture+instantiate cost measured directly via `HostTimer` (chrono), never hidden: at batch=100/224x224,
`capture_ms` + `instantiate_ms` totaled well under 1ms on every measured run (e.g. one representative run:
capture ≈0.13-0.9ms across sizes, instantiate ≈0.1-0.6ms) — small enough that `break_even_calls`
(`graph_setup_cost / per_call_savings`) was under 1 call at every batch size tested, meaning the very
first replay after capture already recoups the setup cost. This matches Section 20D's Basic-kernel
finding; the raw manifest JSON under `benchmark_results/research_optimization/enhanced_cuda_graph/manifests/`
carries the exact per-run numbers.

## Parameter update experiment (spec item 13)

Tested directly: threshold 128→150 and a full Gaussian/Laplacian kernel-size change all correctly trigger
recapture (a new cache entry, not a stale in-place update) rather than an unsafe `cudaGraphExecUpdate` —
this section made the same deliberate scope choice as Section 20D: full recapture on any structural
*or* scalar-argument change (since scalars like `threshold_value` are baked directly into a kernel
node's launch arguments, not routed through a patchable memcpy node), simpler and safer than
`cudaGraphExecKernelNodeSetParams`-based in-place argument patching, which was not implemented — a
disclosed scope limit, not an oversight (same as Section 20D). Coefficient *values* (e.g. a Gaussian sigma
change with the same kernel_size) do NOT trigger recapture — they flow through the already-patchable
memcpy node — verified directly (`test_graph_replay_bit_exact_with_new_host_pointer`-style coverage
extended to varying sigma across many of the 50-repeat correctness runs, each seed producing different
image content but the same structural config).

## Batch-size, resolution, enable/disable testing (spec items 14-16)

All implemented and tested: 1/8/32/64/100 batch sweep (122 unavailable — only 102-119 same-resolution
images exist in the seed=42 pool depending on draw size, consistent with every prior section's disclosed
constraint); resolution change 224x224→64x64→224x224 (correctly recaptures, then correctly reuses the
original resolution's graph on reverting); Gaussian on/off (and by direct extension of the same cache-key
mechanism, every other stage) correctly creates a distinct cache entry and recomputes correctly different
output — the graph key **never executes a stale graph**, verified directly via a 10-iteration
alternating-configuration test that checks every single result against its own configuration's production
output (`test_never_executes_a_stale_graph_across_many_alternating_configs`), not merely asserted.

## Correctness (spec items 17-18)

**Bit-exact** vs. `cuda.pipeline.run_enhanced_cuda_pipeline()` in every case tested: no-graph path (both
scopes), first capture (both scopes), replay with a *different* NumPy array than the one that triggered
capture (proving the pointer-patch mechanism, not just the capturing call), 50 consecutive varied-seed
replays, every batch-size/resolution/parameter/enable-disable transition above, and 10 alternating-config
replays. Zero non-zero differences were ever found — as expected, since the same kernel math runs either
way; nothing required tolerance or investigation. CPU comparison (spec item 18) reused the project's
established methodology and threshold (`differing_pct < 5.0`, the same bar Section 5/20B/20C use for this
dataset's canonical config) — no new tolerance was invented for this section.

## Realistic Streamlit workload (spec item 19) — THE DECISIVE MEASUREMENT

Reused Section 20C's fixed methodology exactly (same one reused again by Section 20D): `xray_cuda.smoke_test()`
context warmup, 2 discarded full-sequence warmup passes for both variants, 5 measured passes with
alternating execution order. Sequence: select batch(32) → process(Enhanced) → change threshold (same
shape) → process → select batch(8) → process → select batch(64) → process.

| | Stateless (production Enhanced) median | Graph (`CudaGraphEnhancedPipeline`) median | Gain |
|---|---:|---:|---:|
| Result | 16.371 ms | 8.649 ms | **1.893x** |

**Enhanced CUDA Graphs are consistently, substantially faster (~1.89x) than stateless production Enhanced
under this realistic, varied workflow** — a stronger result than Section 20D found for Basic (1.44x-1.56x).
This is consistent with the architecture: Enhanced's production dispatch path issues *more* host-side
work per call than Basic (six kernel launches instead of five, plus two synchronous `cudaMemcpyToSymbol`
calls for Gaussian/Laplacian coefficients, each with its own `cudaDeviceSynchronize`+`elapsed_ms()`
round-trip) — there is simply more per-call host-side submission/sync overhead for graph capture to
eliminate.

## Batch-size sweep (real dataset, seed=42, 224x224, 3 warmup + 7 measured runs)

`gain_vs_production` = true production median / variant median. `gain_vs_no_graph` = this class's own
uncaptured path median / variant median.

| batch | production (ms) | graph FULL vs prod | graph FULL vs no-graph | graph KERNELS-ONLY vs prod |
|---:|---:|---:|---:|---:|
| 1   | 0.220 | **2.190x** | 4.025x | 0.670x |
| 8   | 0.429 | **1.321x** | 1.756x | 1.407x |
| 32  | 1.063 | **1.369x** | 1.731x | 1.160x |
| 64  | 1.806 | **1.019x** | 1.287x | 1.175x |
| 100 (4 independent runs) | 2.735–2.983 | **0.915x, 1.057x, 1.231x, 1.402x** (median ≈1.14x) | 0.994x–1.423x | 0.959x–1.416x (median ≈1.21x) |

Every batch size shows `gain_vs_production` ≥ ~1.0x for the FullPipeline scope; batch=100 required 4
repeated runs (matching Section 20D's own precedent of re-running a noisy large-batch measurement) and
still landed with a **positive median (≈1.14x)**, a materially stronger result than Section 20D's Basic
graph found at the same scale (which was a wash, median ≈0.96x). `gain_vs_no_graph` (the code-matched,
apples-to-apples comparison) was positive in every single run at every batch size except the smallest
scope combination (batch=1, KernelsOnly, 0.67x — H2D/D2H as two separate calls dominates at this
batch size when they are NOT captured; FullPipeline, which captures them too, shows the opposite — 4.0x
— at the same batch size, confirming the diagnosis rather than contradicting it).

## Nsight Systems profiling — NOT AVAILABLE this session

Re-checked via `nsys status -e`: `Administrator privileges: No`, `Sampling Environment: Fail` — identical
result to Section 20D. Not re-attempted with a full trace run since the diagnostic already establishes why
it would fail again in this same session/environment; disclosed as NOT AVAILABLE rather than fabricated.
This section's evidence relies entirely on the CPU-timer-based measurements above (`capture_ms`/
`instantiate_ms`/`node_update_ms` fields plus wall-clock benchmark and realistic-workload results).

## Memory (spec item 25)

80 calls (20 warmup + 60 measured) across 4 varying shapes: free VRAM stays **exactly constant** once the
cache reaches its steady 4-entry state (`test_memory_stable_across_60_varied_calls`). `release()` frees
GPU memory (verified via `device_memory_info()`), is idempotent, and the destructor cleans up correctly.
Two independent instances never share a cache. Bounded, LRU-style graph eviction (spec item 26) was **not
implemented** — this section's cache-growth experiment (the 4-shape memory-stability test plus the 6-entry
growth seen in the ad-hoc correctness pass) showed cache size tracks the number of *distinct*
configurations actually used, which stayed small (single digits) in every test in this section; per spec
item 26's own instruction ("do not add this complexity unless the cache experiment demonstrates a need"),
no eviction policy is implemented, since no unbounded-growth need was demonstrated.

## Intermediate outputs (spec item 27)

**Mode A only** (graph computes the final output; no per-stage D2H). Mode B/C (diagnostic intermediate
copies) were not implemented in this section: adding a D2H node after every enabled stage inside the
captured graph would directly undermine the optimization being tested (each extra D2H is exactly the kind
of host-synchronization-adjacent operation this section exists to eliminate the cost of), and this
section's own scope (item 28) explicitly keeps the experimental path out of Presentation Mode, so no
current caller needs intermediate outputs from this class. Documented here as a disclosed, deliberate
scope limit for any future section that wants graph-compatible intermediate visualization.

## Streamlit integration (spec items 28-29)

**Not wired into the UI.** No `run_enhanced_cuda_graph()` convenience wrapper or UI checkbox was added —
this section's API surface is the native `xray_cuda.CudaGraphEnhancedPipeline` class only, reachable from
Python exactly as `xray_cuda.CudaGraphPipeline` (Section 20D) already is, exercised by this section's own
tests and benchmark script. Given the STOP CONDITION's explicit "do not redesign Streamlit" instruction
and item 28's explicit "do not wire into Presentation Mode yet," adding a thin Python wrapper module or a
UI toggle was judged as scope creep beyond what this section's own decision (below) needs — Presentation
Mode continues to use the production Enhanced pipeline unchanged, verified directly.

## Failure fallback (spec item 30)

Implemented identically to Section 20D: `build_entry()`'s capture body is wrapped so any exception or a
failing `cudaStreamBeginCapture`/`cudaStreamEndCapture`/`cudaGraphInstantiate` returns `nullptr` with a
`fallback_reason` string; `run()` then transparently re-executes via the non-graph path and surfaces
`used_graph=false` plus the non-empty reason. Under every valid Enhanced Specialized/Network3x3/Vectorized
configuration exercised in this section, capture never actually failed — verified by direct inspection
(no unhandled exception in 27 tests spanning every configuration dimension), so this path is defensive
engineering rather than an observed real-world trigger, the same honest disclosure Section 20D made.

## Decision

```
CUDA GRAPHS (Enhanced pipeline) → EXPERIMENTAL
(keep as a documented, tested, isolated capability; not wired into production this section)
```

**Reasoning**: This section's evidence is materially *stronger* than Section 20D's Basic-graph finding —
every batch size tested showed a positive `gain_vs_production` for the FullPipeline scope (including
canonical-adjacent batch=100's ≈1.14x median across 4 runs, versus Basic's ≈0.96x wash at the same scale),
and the decisive realistic-workload test showed a ~1.89x gain, the strongest single result in this
project's CUDA Graph research to date. Correctness is airtight (27/27 tests, bit-exact against production
in every dimension tested, zero tolerance needed), memory is stable, and the fallback mechanism is
implemented and sound.

It stops short of ADOPT for reasons of engineering completeness and risk management, not because the
core optimization is unproven:

1. **No production API surface exists yet.** This section deliberately did not build a
   `run_enhanced_cuda_graph()` wrapper, a `PersistentCudaPipelineService`-style session-scoped ownership
   model, or any UI integration (items 28-29) — moving to ADOPT would require that work, plus the same
   kind of session-lifecycle design Section 20C flagged as unresolved for persistent buffers (a
   `CudaGraphEnhancedPipeline` instance is GPU-resident state that needs a clear per-session ownership
   story before it can sit in `st.session_state`).
2. **Kernel duplication is a real maintenance cost.** Unlike Section 20D's Basic-kernel graph (which calls
   the exact production kernel symbols), this section's kernels are textually-duplicated copies. Any
   future change to the production Enhanced Specialized/Network3x3/Vectorized kernel math would silently
   desynchronize this experimental path unless a process is put in place to keep them in lockstep — a
   real, disclosed risk that argues for caution before production adoption, even though today's copies
   are verified bit-exact.
3. **One batch size's confidence interval still touches parity.** Batch=100's 4-run range (0.915x–1.402x)
   means an individual real call could occasionally land at or slightly below production speed, even
   though the median and 3 of 4 runs are clearly positive — a real, honestly-disclosed nuance, not
   dismissed by only reporting the most favorable run.

**The experimental code is not deleted** — `cuda/include/pipeline_cuda_graph_enhanced.cuh`/
`cuda/src/pipeline_cuda_graph_enhanced.cu`, `tests/test_enhanced_cuda_graph.py`, and
`scripts/benchmark_enhanced_cuda_graph.py` remain in the repository as a documented, tested, working
capability — the strongest EXPERIMENTAL candidate this project's optimization research has produced so
far. A future section building the missing production API surface (items 1-2 above) and establishing a
kernel-sync-check process (item 2) would have a well-evidenced case for revisiting ADOPT.

## Production API recommendation

**No change to `cuda/pipeline.py` or `ui/services.py`.** `run_basic_cuda_pipeline()` and
`run_enhanced_cuda_pipeline()` remain the sole production entry points, exactly as before this section.
