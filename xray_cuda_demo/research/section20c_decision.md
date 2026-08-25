# Section 20C — Persistent CUDA Pipeline API Design + Production Adoption Decision

New artifacts this section: `cuda/include/pipeline_experimental.cuh`/`cuda/src/pipeline_experimental.cu`
(added an explicit terminal-lifecycle `released_` flag + `is_released` property — a real fix, see below),
`cuda/persistent_pipeline_experimental.py` (Model B + Model C Python wrappers,
`PersistentCudaPipelineService` prototype), `tests/test_persistent_api_design.py` (19 tests, all passing).
Production `cuda/pipeline.py` / `ui/services.py` are completely unmodified — verified directly
(`test_production_pipeline_functions_unaffected` in Section 20B's test file, still passing).

## API models evaluated

### A. Stateless (existing production API)

`run_basic_cuda_pipeline()` / `run_enhanced_cuda_pipeline()`. Reference/baseline throughout. Unmodified.

### B. Caller-owned persistent object (explicit lifecycle)

```python
pipeline = PersistentCudaPipeline()
try:
    out, timing = pipeline.run_basic(images, config)
finally:
    pipeline.release()
```

Implemented as a thin Python class (`cuda/persistent_pipeline_experimental.py`) wrapping the native
`xray_cuda.PersistentCudaPipeline` (Section 20B), with the same image-processing call contract as
`run_basic_cuda_pipeline()` (list of same-shape images + `FilterConfig` + per-stage enhanced flags) so it can be
substituted directly wherever the stateless functions are called today.

### C. Context-managed

```python
with PersistentCudaPipeline() as pipeline:
    out, timing = pipeline.run_enhanced(images, config)
```

Same object as Model B — `__enter__`/`__exit__` call the identical `release()`. Verified releases correctly
even when an exception is raised inside the `with` block (`test_context_manager_releases_on_exit_including_on_exception`).
**Recommendation if this were adopted: prefer Model C** — it removes the one realistic misuse mode (forgetting to
call `.release()`) entirely, at zero additional API surface cost over Model B.

## Correctness

All bit-exact:
- Every one of Models B/C's outputs vs. the stateless production functions, Basic and Enhanced (`test_basic_pipeline_bit_exact_vs_production`, `test_enhanced_pipeline_bit_exact_vs_production`).
- Basic and Enhanced interleaved on ONE shared persistent object never drift (`test_basic_and_enhanced_independent_on_same_service`).
- Configuration changes (threshold value, Gaussian kernel size) produce correctly *different* output, matching the stateless reference exactly for each configuration (`test_configuration_change_does_not_reuse_stale_output`, `test_gaussian_kernel_size_change_recomputes_coefficients_correctly`).
- Resolution changes correctly reallocate rather than reuse an incompatible buffer, verified at every transition including reverting to a previous resolution (`test_resolution_change_reallocates_and_stays_correct`).

## Memory stability

- 150 repeated calls, same shape: bounded free-VRAM delta (<16MB), zero drift in output.
- 60 calls with randomly varying batch size (4-64) and resolution (16x16-64x64): free VRAM grows once to the
  largest combination's requirement, then stays **exactly constant** for the remaining calls (measured: 0 bytes
  drop between any two post-warmup readings) — no fragmentation, no unbounded growth.
- `release()` frees GPU memory (verified via `device_memory_info()` before/after), is idempotent, and the
  destructor cleans up correctly even without an explicit `release()` call (`gc.collect()`-verified).

## Batch-size / resolution / configuration-change behavior

All correct (see Correctness above). Capacity policy confirmed: growing 4→16→32 allocates each time (capacity
insufficient); shrinking to 8/24/2 afterward reuses the 32-capacity buffers without reallocating
(`grew_gpu_buffers=False`), exactly matching spec item 9's policy.

## Streamlit/session safety

Two independent `PersistentCudaPipelineService` instances (simulating two browser sessions) never share state:
releasing one has zero effect on the other's correctness or availability (`test_two_independent_services_do_not_cross_contaminate`).
No global/module-level mutable GPU object exists anywhere in this design — every native buffer lives inside an
instance a caller explicitly owns, matching this project's established `st.session_state`-based convention (never
a module-level global) for every other piece of mutable state. If adopted, the natural integration is one
`PersistentCudaPipelineService` instance stored in `st.session_state`, created on first use, released when the
session ends (Streamlit does not currently expose a reliable "session end" hook — this is a real, disclosed
limitation, not solved in this section: a `PersistentCudaPipeline` left in `st.session_state` when a browser tab
is simply closed will hold its GPU buffers until the Streamlit server process itself restarts or explicitly
tracks and reaps idle sessions, which is outside this section's scope).

**Multi-session GPU memory cost**: each session's service would hold up to 2x a full canonical-batch buffer set
(Basic + Enhanced pipelines). At the canonical 224x224/batch=122 configuration this is a few MB per session — not
a concern at small session counts, but a real capacity-planning question for many concurrent sessions that this
section flags rather than solves.

## First-call vs. steady-state performance (canonical batch=122, 224x224, from Section 20B's measurements, reused per spec item 26's "for this section, pinned memory remains rejected" and item 39's explicit instruction not to re-litigate it)

| | Stateless | Persistent (steady-state, same shape repeated) |
|---|---:|---:|
| Total | 3.491 ms | 2.575 ms |
| Allocation | 0.645 ms | ~0.0004 ms (after first call) |
| **Gain** | 1x | **1.36x** |

This is real and reproduced (Section 20B: 1.15x-1.62x across batch sizes 1-122, zero regressions in this
narrow same-shape-repeated-call pattern).

## Realistic Streamlit workload (spec item 31) — THE DECISIVE MEASUREMENT

Simulated the exact sequence spec item 31 describes: select batch(32) → process(Enhanced) → change a filter
parameter → process(Enhanced, same shape) → change batch size to 8 → process(Enhanced) → select a new batch(64)
→ process(Enhanced) → switch to Basic → process(Basic). Measured backend `total_ms` only (never Streamlit
rendering time), properly warmed up (2 discarded warmup passes of the full sequence for both variants before
measuring), 5 measured passes with alternating execution order to cancel any residual ordering bias:

| Pass | Stateless total (ms) | Persistent total (ms) |
|---:|---:|---:|
| 1 | 7.564 | 10.240 |
| 2 | 7.868 | 10.545 |
| 3 | 7.873 | 9.772 |
| 4 | 7.593 | 10.884 |
| 5 | 7.393 | 10.649 |
| **Median** | **7.593** | **10.545** |

**Persistent is consistently ~28% SLOWER (0.72x) than stateless for this realistic, varied workflow** — every
single one of 5 passes, no overlap. This is the opposite of the steady-state benchmark-loop result above, and it
is the correct basis for the production decision: the actual Streamlit application's usage pattern is
overwhelmingly the *varied* case (a user tunes parameters, changes batch size, or picks a different image between
"Process" clicks far more often than they click "Process" repeatedly on an unchanged configuration). Under that
realistic pattern, persistence rarely gets to reuse a buffer, and the capacity-check bookkeeping + extra object
indirection the persistent design requires costs measurably more than simply allocating fresh every time.

This directly demonstrates spec item 46's warning: *"A persistent pipeline may look faster in repeated backend
calls while being awkward for a real Streamlit application... No single number decides the outcome."* The
steady-state number alone would have pointed toward ADOPT; the realistic-workload number reverses that.

## Decision

```
PERSISTENT BUFFERS → REJECTED (for Streamlit/production adoption)
```

**Reason**: The lifecycle, correctness, memory-safety, and session-isolation engineering are all sound — every
test passes, no leaks, no fragmentation, no cross-session contamination. But the realistic-workload measurement
(spec item 31, the specific test this section was built to answer) shows a consistent ~28% *regression* against
the actual usage pattern of this project's Streamlit application, where filter parameters, batch size, and image
selection change between calls far more often than they stay fixed. Section 20B's steady-state gain (1.15x-1.62x)
is real but describes a narrower pattern (tight, same-shape, repeated calls) that matches internal benchmark
loops (Section 11's 20-measurement-run methodology) much better than it matches interactive UI use.

**The experimental code is not deleted** — `cuda/persistent_pipeline_experimental.py`,
`cuda/include/pipeline_experimental.cuh`/`cuda/src/pipeline_experimental.cu`, and their test files remain in the
repository as a documented, isolated capability, available if a future benchmark-tooling use case (not the
Streamlit UI) can genuinely guarantee the same-shape-repeated-call pattern where it measurably wins.

## Production API recommendation

**No change to `cuda/pipeline.py` or `ui/services.py`.** `run_basic_cuda_pipeline()` and
`run_enhanced_cuda_pipeline()` remain the sole production entry points, exactly as before this section.

If a future section revisits this (e.g. specifically for Section 11's own repeated-measurement benchmark loop,
which genuinely is the same-shape-repeated-call pattern persistence wins at), the recommended integration would
be Model C (context manager) scoped narrowly to that one benchmark loop -- not a session-wide Streamlit object,
given the unresolved "session end" cleanup gap noted above.
