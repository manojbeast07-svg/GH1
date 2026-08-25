# CUDA Optimization Research Report (Section 20A)

**Rule followed throughout: MEASURE, DON'T FABRICATE.** Every number in this report was measured in this
session (Nsight Systems profiling, fresh `nvcc --ptxas-options=-v` register reports, two standalone C++/CUDA
experiments, and the existing real `benchmark_results/variant_sweeps/` data from Section 17) or is an explicitly
cited external source (`sources.md`). No production kernel, CPU algorithm, or benchmark methodology was modified
to produce this report.

## 1. Methodology

1. **Profile first** (spec item 5): ran Nsight Systems (`nsys`) on both the Basic and Enhanced production
   pipelines (real 122-image, 224x224 canonical batch) to get a CPU/GPU API timeline. Nsight **Compute** (`ncu`)
   was attempted but requires Administrator privileges on this machine (`ERR_NVGPUCTRPERM`) that were not
   available in this session -- disclosed rather than silently omitted or faked.
2. **Static resource analysis**: compiled every Basic and Enhanced `.cu` kernel file fresh this session with
   `--ptxas-options=-v` to get real register/shared-memory/spill counts (no runtime permission needed for this).
3. **Literature research** (spec item 4): NVIDIA's own Best Practices Guide, Programming Guide (Stream-Ordered
   Memory Allocator), Nsight Compute Profiling Guide, CUDA Graphs documentation, and peer-reviewed/conference
   median-filter papers -- see `sources.md` for the full list with what each source was used to support.
4. **Standalone experiments**: two throwaway C++/CUDA programs (`research/experiments/`), compiled and run
   directly with `nvcc` (not wired into the production pybind11 extension), to get real measured numbers for the
   two most evidence-backed pipeline-level candidates (buffer allocation strategy, pinned host memory).
5. **Decision**: every candidate is classified against spec item 44's taxonomy (KEEP / PROMISING -- REQUIRES MORE
   TESTING / NO MEASURABLE BENEFIT / REGRESSION / TOO COMPLEX FOR BENEFIT / NOT APPLICABLE) using Level 1/2/3
   from spec item 43. **Nothing was implemented in production as a result of this report** -- per this section's
   own STOP CONDITION, that decision is left for a follow-up, separately-versioned experiment.

## 2. Filter-specific findings (full detail: `filter_optimization_matrix.md`)

### Gaussian
- **Current bottleneck**: none identified -- separable two-pass, 21-23 registers/pass, zero spilling. Already
  algorithmically minimal (O(2k) vs Basic's O(k^2)).
- **Research opportunities**: kernel fusion of the h/v passes (fusion principle, general) and vectorized loads
  (NVIDIA Best Practices coalescing guidance) were both considered.
- **Recommended**: **KEEP.** Vectorization was already investigated and rejected in Section 6 for a real,
  specific reason (dataset widths like 1733px aren't 4-aligned); fusion's expected saving (one intermediate
  buffer's global traffic, already reused not re-allocated) is small relative to its register-pressure risk.

### Median
- **Current bottleneck (k=3)**: none identified -- branchless sorting network, zero spilling, already the
  dominant win in the whole project (5.82x measured).
- **Current bottleneck (k=5/k=7)**: the *opportunity*, not a resource limit -- `shared`/`specialized` variants
  only reach 1.00-1.04x vs Basic at these sizes, the weakest measured gains of any Enhanced variant in the
  project.
- **Research opportunities**: peer-reviewed literature (`sources.md` #5) shows histogram-based rank selection
  scales better than sorting networks as window size grows, while sorting networks remain the right choice at
  3x3 -- directly consistent with this project's existing k=3 choice and with k=5/k=7's weak gains.
- **Recommended**: **KEEP** for k=3. **PROMISING -- flagged for a dedicated future experiment** for k=5/k=7 (a
  histogram-based kernel is a genuinely new algorithm, not a parameter tweak -- too large a scope change for this
  research gate itself, per spec item 40's complexity criterion).

### Sobel
- **Current bottleneck**: none identified -- 18-39 registers depending on variant, zero spilling, no `sqrt` in
  the magnitude path (confirmed by the low register counts).
- **Research opportunities**: Sections 8 and 17 already measured Shared (1.02x), SharedConst (**0.90x --
  regression**), Specialized (1.04x, current default), Separable (**0.67x -- regression**). Two of four
  already-tried variants are measured regressions.
- **Recommended**: **KEEP CURRENT IMPLEMENTATION.** This is the textbook case spec item 11 describes: profiler
  evidence (here, register data + two independently-confirmed regressions) supports explicitly stopping rather
  than inventing another variant. Sobel's now-large *share* of Enhanced's own remaining compute time (28.2% per
  Nsight Systems -- the single largest of any filter) is a consequence of its small achievable headroom, not
  evidence of an untried technique.

### Laplacian
- **Current bottleneck (k=3)**: measurement noise, not a resource limit -- `specialized` (production default,
  1.02x) and `shared_const` (1.28x) are close enough at k=3 that CUDA-event timing jitter is a meaningful
  fraction of the signal (both kernels run in well under 1ms per batch).
- **Current bottleneck (k=5)**: none -- `specialized` clearly wins (2.69x), consistent across two independent
  measurement sessions.
- **Recommended**: **KEEP** at k=5. **PROMISING -- worth a longer, higher-run-count re-measurement** at k=3
  before considering a dispatch change (switching the k=3 default from `specialized` to `shared_const`) -- this
  is explicitly not recommended for implementation from this session's data alone, because the margin is within
  plausible measurement noise.

### Threshold
- **Current bottleneck**: memory bandwidth -- 10-11 registers (the lowest in the whole project), zero shared
  memory, zero spilling, purely pointwise.
- **Research opportunities**: further vectorization considered; `multi_pixel` (process more pixels/thread at the
  same access width) was already measured as a **regression** (0.92x) in Section 17.
- **Recommended**: **KEEP CURRENT IMPLEMENTATION.** At 10-11 registers and no neighborhood dependency, this
  kernel is about as close to a practical memory-bandwidth floor as this project's kernels get -- shared-memory
  tiling would add synchronization overhead for zero algorithmic benefit (Threshold has no data reuse across
  threads to exploit), matching spec item 14's own explicit caution.

## 3. Pipeline-specific findings (full detail: `pipeline_optimization_matrix.md`)

**The most important finding in this section is here, not in the per-filter analysis.** Nsight Systems showed
`cudaMalloc`/`cudaFree` consuming more CPU-side API time (78.8%) than kernel launches, kernel execution, and
error-checking combined. The reason: `GpuImageBatch buffer_a`/`buffer_b` (and Gaussian's intermediate float
buffer, when Enhanced) are allocated fresh **every separate Python-level call** to
`run_basic_cuda_pipeline()`/`run_enhanced_cuda_pipeline()` -- they are reused via ping-pong **within** one call
(as designed, Section 5), but never persist **across** calls.

A standalone experiment isolating exactly this cost (`research/experiments/allocation_reuse_experiment.cu`)
measured **0.4-4.95 ms per pipeline call** of pure allocate+free overhead depending on batch size -- at the
canonical 122-image batch size, **1.86 ms**, larger than the entire measured GPU compute time for that batch
(1.28 ms). This overhead is invisible in a single one-shot run (disk I/O, ~53ms, dominates everything), but real
and repeated in every benchmark loop, every Live Processing re-click, and every Optimization Lab comparison this
project already runs.

A second experiment tested `cudaMallocAsync`/`cudaFreeAsync` against CUDA's own documented stream-ordered pool
(`sources.md` #2) as the "correct" fix for this class of problem -- and found it did **not** reproduce the
documented benefit on this GPU/driver at these buffer sizes (it was measurably *slower* than plain
`cudaMalloc`/`cudaFree` below batch=512). Per this project's own rule (spec item 25), this is reported honestly
as a **rejected** candidate rather than assumed to work because the documentation says it should. True buffer
**persistence** (never calling `cudaMalloc` on the hot path at all) is what actually eliminates the cost.

Pinned host memory was also measured directly. The first pass suggested a modest ~1.10x gain at the canonical
batch size, but that number showed enough run-to-run spread to warrant repeating -- 7 independent runs gave a
median of **~1.8-1.9x** (range 1.10x-2.31x, matching the NVIDIA blog's own point that transfer speedup depends on
host system state). Repeating the measurement rather than trusting the first pass changed the conclusion from
"smaller than the allocation finding" to "comparable in magnitude to it" -- a direct demonstration of why this
project measures variability rather than reporting a single run.

CUDA Graphs, stream overlap, and double buffering were evaluated by literature + measured-launch-overhead
reasoning (not built and benchmarked this session, given the time budget and that their evidence-based ceilings
are smaller than the two candidates above) -- see the pipeline matrix for the specific numbers behind each
"PROMISING -- REQUIRES MORE TESTING" classification.

## 4. What was explicitly rejected, and why

- **`cudaMallocAsync`/memory pools** -- measured regression at the project's typical batch sizes (see above).
- **Sobel SharedConst, Separable** -- already-measured regressions (Sections 8, 17).
- **Threshold `multi_pixel`** -- already-measured regression (Section 17).
- **Vectorized loads for Gaussian/Sobel/Laplacian** -- already rejected (Sections 6, 8, 9) because the real
  dataset's widths are not 4-aligned, forcing a scalar fallback that erodes the benefit.
- **FP16/reduced precision** -- not applicable; violates this project's bit-exactness correctness contract
  without an extraordinary justification that nothing in this research surfaced.
- **A specialized hard-coded default pipeline path** -- rejected as unnecessary; the pipeline is already one
  native call with per-stage branches, and branch-prediction overhead this small is implausible next to a
  measured 0.4-4.95ms allocation cost.
- **Kernel mega-fusion beyond the existing Laplacian+Threshold experiment** -- not investigated further; the
  existing fusion experiment (Section 10) already measured a real but modest whole-pipeline gain (~3%) and was
  correctly rejected for production because it discards the standalone Laplacian output the project's
  introspection tooling depends on. No new evidence in this session changes that decision.

## 5. Final decision table (spec item 45)

| Filter/Pipeline | Technique | Research support | Measured result | Production decision |
|---|---|---|---|---|
| Gaussian | Fused h+v kernel | General fusion principle | Not benchmarked (expected saving smaller than register-pressure risk) | KEEP current (separable 2-pass) |
| Gaussian | Vectorized loads | NVIDIA Best Practices (coalescing) | Already rejected, Section 6 (dataset widths not 4-aligned) | KEEP current |
| Median (k=3) | Histogram-based rank selection | Peer-reviewed literature favors sorting networks at 3x3 | Not benchmarked (literature argues against it here) | KEEP current (`network3x3`) |
| Median (k=5/k=7) | Histogram-based rank selection | Peer-reviewed literature favors histograms as window grows | Not benchmarked (new algorithm, out of scope for this gate) | **PROMISING -- future dedicated experiment** |
| Sobel | Any further variant | 2 of 4 already-tried variants measured as regressions (Sections 8, 17) | `shared_const`=0.90x, `separable`=0.67x (both regressions) | **KEEP CURRENT IMPLEMENTATION** |
| Laplacian (k=3) | Switch default to `shared_const` | Section 17's fresh sweep: `shared_const`=1.28x vs `specialized`=1.02x | Measured, but within plausible timing-jitter margin at this kernel size | PROMISING -- needs higher-run-count re-measurement before any change |
| Laplacian (k=5) | -- | `specialized`=2.69x, clear and consistent | Confirmed across 2 sessions | KEEP current |
| Threshold | `multi_pixel` / further vectorization | NVIDIA Best Practices (coalescing) | Already measured as a regression (0.92x), Section 17 | **KEEP CURRENT IMPLEMENTATION** |
| Pipeline | Buffer persistence across repeated calls | `cudaMalloc`/`cudaFree` measured as 78.8% of CPU-side API time (Nsight Systems) | 0.4-4.95ms/call recovered depending on batch size (standalone experiment) | **PROMISING -- strongest evidence in this report; recommend a dedicated follow-up implementation experiment** |
| Pipeline | Pinned host memory | NVIDIA blog, documented 15%-2.5x range | Median 1.8-1.9x at canonical batch (7 repeated runs) | **PROMISING -- comparable magnitude to buffer persistence** |
| Pipeline | `cudaMallocAsync` / memory pool | CUDA Programming Guide (stream-ordered allocator) | **Regression** at batch<=32; only 1.07-1.22x at batch=512 (standalone experiment) | **REJECTED** |
| Pipeline | CUDA Graph capture | Literature: ~5-10us/launch overhead; this pipeline has 6 launches/call | Not benchmarked; evidence-based ceiling ~0.3ms/call, smaller than the two candidates above | PROMISING -- requires more testing, lower priority |
| Pipeline | Stream overlap / double buffering | CUDA Programming Guide (streams) | Not benchmarked; correctness complexity, limited applicability to this project's mostly-single-batch-at-a-time usage | PROMISING -- requires more testing, lower priority |
| Pipeline | Kernel fusion (Laplacian+Threshold) | Already measured, Section 10 | 1.30x combined-kernel, ~3% whole-pipeline | KEEP AS EXPERIMENTAL (no new evidence changes the prior rejection) |
| Pipeline | Specialized hard-coded pipeline path | -- | Not benchmarked; implausible next to the measured allocation cost | REJECTED AS UNNECESSARY |
| Pipeline | FP16 / reduced precision | -- | Not researched in depth; violates correctness contract | NOT APPLICABLE |

**Note on `benchmark_results/research_optimization/`** (spec item 47): not created. Both experiments this
session measured OS/driver-level behavior (allocation cost, transfer cost) using standalone timing harnesses,
not new candidate image-processing kernels -- there is no correctness dimension (no image output) to record
alongside the timing, so the full raw/aggregated/correctness/manifest structure that directory implies does not
apply. If the buffer-persistence or pinned-memory candidates are implemented as a real follow-up experiment
(producing actual pipeline output to validate), that IS the point at which a `benchmark_results/` (or a
dedicated `outputs/`) namespace with correctness validation would be created -- not before.

## 6. Overall conclusion

Per-filter kernel optimization is close to exhausted for this project's default configuration: five filters, four
of which show zero register spilling and a documented history of correctly rejecting further attempted variants
(two of them measured regressions). The one filter-level candidate with real, new literature support (histogram-
based Median for k=5/k=7) is a genuinely new algorithm, not a tuning change, and is flagged for a dedicated
future experiment rather than attempted in this research gate.

The pipeline level is where this session found real, new, measured headroom: **buffer allocation strategy**
(recovering up to ~1.86ms of a ~3.4ms GPU round-trip at the canonical batch size, for repeated-call workloads)
and **pinned host memory** (median ~1.8-1.9x transfer speedup at the canonical batch size across 7 repeated
runs, comparable in magnitude, not the smaller effect a single early measurement suggested). Both are
classified PROMISING and recommended for a dedicated, separately-versioned follow-up implementation experiment --
not applied to the frozen production baseline in this research gate, per the section's own STOP CONDITION.
