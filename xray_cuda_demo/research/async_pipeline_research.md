# Async Multi-Stream Pipeline Research (Section 20F, spec item 4)

## CUDA requirements for genuine transfer/compute overlap

Verified against current NVIDIA documentation (CUDA C++ Programming Guide, "Asynchronous Concurrent
Execution" chapter, release 13.3/13.4 as of this section's work — accessed via WebSearch/WebFetch, since
the guide's full HTML page was too large for direct single-page fetch; corroborated by NVIDIA's own
"How to Overlap Data Transfers in CUDA C/C++" developer blog and multiple current NVIDIA Developer Forums
threads, all cited below) and against this machine's own installed CUDA 13.3 runtime headers directly:

1. **Page-locked (pinned) host memory is required for genuine overlap of a host↔device transfer with
   kernel execution.** From the current guide (as summarized via WebSearch against docs.nvidia.com): "the
   asynchronous transfer version requires pinned host memory... and if you don't use a pinned allocator,
   the operation won't fail or return an error, but it generally will not overlap with the subsequent
   kernel call." A `cudaMemcpyAsync()` call from **pageable** memory still returns without error and is
   still associated with a stream, but the CUDA driver internally performs a synchronous staging copy
   through its own pinned bounce buffer before the transfer can proceed — this staging step does not
   overlap with other stream activity the way a true pinned-source async copy does.
2. **Non-default streams are required.** Confirmed directly in this machine's own `cuda_runtime_api.h`
   (`cudaMemcpyAsync`'s doc comment, verified by direct read): "If kind is cudaMemcpyHostToDevice or
   cudaMemcpyDeviceToHost and the stream is non-zero, the copy may overlap with operations in other
   streams." Operations on the default/null stream are implicitly synchronizing with every other stream
   on the device (documented CUDA behavior), so any overlap design must use explicitly created,
   non-default streams for every stage that should run concurrently with another.
3. **The number of concurrent copy engines (`cudaDeviceProp::asyncEngineCount`) determines whether H2D and
   D2H can be simultaneously in flight**, not just whether either can overlap with compute. Two async
   engines allow duplex (simultaneous bidirectional) PCIe transfer; one async engine means H2D and D2H
   transfers serialize against each other even though each can still individually overlap with kernel
   compute on the SM (a separate hardware unit from the copy engine).

## This machine's actual hardware capability — measured, not assumed

Queried directly via a throwaway `cudaGetDeviceProperties()` probe (compiled/run once, not part of any
shipped artifact):

```
name=NVIDIA GeForce RTX 4060 Laptop GPU
asyncEngineCount=1
concurrentKernels=1
```

**This GPU has exactly one async copy engine.** This is a concrete, hardware-specific constraint on this
section's own experiment: H2D and D2H cannot be simultaneously in flight on this machine, regardless of
how many streams or buffer sets the software uses — only one direction of transfer can be active on the
PCIe link at any instant. Each transfer direction *can* still overlap with kernel compute (a separate
engine), so the achievable overlap on this hardware is:

```
H2D(chunk N)  overlaps  compute(chunk N-1)     -- possible (different engines)
D2H(chunk N)  overlaps  compute(chunk N+1)     -- possible (different engines)
H2D(chunk N)  overlaps  D2H(chunk M)           -- NOT possible on this GPU (one copy engine, shared)
```

This directly bounds what triple buffering can add over double buffering *on this specific machine*:
double buffering already captures the two overlaps that are physically possible here (transfer-with-
compute in both directions); triple buffering's classic benefit (letting D2H(N-1) and H2D(N+1) both be
in flight at once) is **not something this GPU's hardware can actually do**, so this section
predicts — to be confirmed by measurement, not assumed, per spec item 17 — that triple buffering will
show little to no improvement over double buffering here, and may even regress slightly due to the extra
bookkeeping and buffer-set memory overhead. A machine with `asyncEngineCount=2` would be expected to show
a larger gap between double and triple buffering; this section's numbers are specific to this GPU and are
disclosed as such.

## Why the technique might apply to this application

- Section 20A's research already established H2D+D2H (~2.03ms at canonical batch=122) is comparable to or
  exceeds kernel compute time (~1.28ms) — a transfer-bound-adjacent workload is exactly the kind that
  benefits most from overlap when overlap is achievable.
- The production pipeline currently does H2D, then all compute, then D2H — fully serial. If even one
  direction of transfer can overlap with compute (established as physically possible above), there is
  real idle time to reclaim.

## Why it might not apply

- **The real application's data source is a pageable NumPy array.** Per the pinned-memory requirement
  above, `cudaMemcpyAsync` from pageable memory does not achieve genuine overlap — this is the single
  largest risk to this experiment's premise, and is why spec item 12 explicitly requires testing this
  directly rather than assuming pinned-source async copies (Section 20B already measured, and rejected,
  the NumPy→pinned staging cost for the *whole pipeline* case; this section must re-confirm that finding
  holds for the *async, chunked* case specifically rather than reuse the old number verbatim).
- Only one async copy engine on this machine (measured above) caps how much overlap benefit triple
  buffering specifically can add beyond double buffering.
- Chunking a batch into smaller pieces to enable pipelining adds per-chunk fixed overhead (more kernel
  launches, more stream synchronization points) that Section 20A/20D/20E's research already showed is not
  free (~55μs/launch average) — too-small chunks could lose more to overhead than they gain from overlap.

## Expected bottleneck

Given the pageable-memory constraint, the most likely outcome is that genuine H2D/compute overlap is
**not** achievable through the real application's data path without paying the pinned-staging cost
Section 20B already found to be a net negative — meaning this section's real, decisive question is
whether staged-pinned + async multi-stream can beat that staging cost by enough to still win end-to-end,
which is a materially harder bar to clear than Section 20D/20E's graph experiments (which never needed to
change how host memory is prepared).

## Expected memory cost

Each additional buffer set requires its own full ping-pong buffer pair (or more, matching whatever
intermediate buffers a stage needs) sized to one *chunk* (not the full batch) — since chunks are smaller
than the user-requested batch by design (spec item 18), N buffer sets at a small chunk size should cost
well under one full canonical-batch buffer pair, but this must be measured per chunk size, not assumed
constant.

## Sources

- [How to Overlap Data Transfers in CUDA C/C++ — NVIDIA Technical Blog](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)
- [CUDA C++ Programming Guide — NVIDIA (release 13.3/13.4, docs.nvidia.com)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Overlapping computation and data transfers must use pinned memory or UVA? — NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/overlapping-computation-and-data-transfers-must-use-pinned-memory-or-uva/64080)
- [concurrent D2H+H2D transfers? — NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/concurrent-d2h-h2d-transfers/42364)
- This machine's own `cuda_runtime_api.h` (CUDA 13.3 toolkit, installed at
  `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\include\cuda_runtime_api.h`), read directly.
- This machine's own `cudaGetDeviceProperties()` output, queried directly (see above).
