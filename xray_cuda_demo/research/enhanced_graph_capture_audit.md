# Enhanced CUDA Graph Capture Audit (Section 20E, spec item 4)

Direct inspection of all five `*_enhanced_dispatch()` functions as they exist in production today
(`cuda/src/gaussian_enhanced.cu`, `median_enhanced.cu`, `sobel_enhanced.cu`, `laplacian_enhanced.cu`,
`threshold_enhanced.cu`), read in full, not guessed. Production variant/config used for each (per this
section's spec item 3): Gaussian=Specialized, Median=Network3x3, Sobel=Specialized, Laplacian=Specialized,
Threshold=Vectorized, block=16x16.

## Findings table

| Stage | Kernel launch API | Stream used | Sync in dispatcher | cudaMemcpy | cudaMemcpyToSymbol | Host callbacks | Allocation | Event usage | Param update mechanism | Can isolated orchestration avoid it? |
|---|---|---|---|---|---|---|---|---|---|---|
| **Gaussian** (Specialized) | `gaussian_specialized_h_kernel<K><<<grid,block,shared>>>` then `gaussian_specialized_v_kernel<K><<<...>>>` (`gaussian_enhanced.cu:322-329,349-356`) | implicit default stream (no `cudaStream_t` param anywhere in the function signature) | `cudaDeviceSynchronize()` **twice** — after H pass (`:331`) and after V pass (`:358`) | none directly (coeff path uses ToSymbol for Specialized) | **yes** — `cudaMemcpyToSymbol(d_gaussian_coeffs_1d, host_coeffs_1d, kernel_size*4)` (`:302`), synchronous, runs on the default stream implicitly before the H kernel | none | one `DeviceBuffer<float> intermediate` per call, owned by the caller (`pipeline_basic.cu`/`pipeline_experimental.cu`), not by the dispatch function itself | `CudaTimer` around each pass; `.elapsed_ms()` calls `cudaEventSynchronize` (not invoked until after `cudaDeviceSynchronize` already returned, so it is a no-op wait in practice, but the *call itself* would still be illegal mid-capture) | none — full recapture only | **Yes.** Bypass the dispatcher; call `gaussian_specialized_h_kernel<K>`/`_v_kernel<K>` directly on an explicit stream with no synchronize; replace `cudaMemcpyToSymbol` with `cudaMemcpyAsync` to the symbol's address (via `cudaGetSymbolAddress`) on the same stream, capturable and patchable exactly like any other H2D node. |
| **Median** (Network3x3) | `median_network9_kernel<<<grid,block,shared>>>` (`median_enhanced.cu:229`) | implicit default stream | `cudaDeviceSynchronize()` once (`:241`) | none | none (Network3x3 is a fixed 9-element sorting network, no coefficients at all) | none | none beyond the caller's buffers | `CudaTimer`/`elapsed_ms()`, same note as above | none — full recapture only | **Yes.** Simplest of the five: direct kernel launch on an explicit stream, no synchronize, no coefficient upload at all. |
| **Sobel** (Specialized) | `sobel_specialized_kernel<mode><<<grid,block,shared>>>` (`sobel_enhanced.cu:292-303`, mode is a template parameter, resolved on the host from the `SobelMode` enum before launch) | implicit default stream | `cudaDeviceSynchronize()` once (`:311`) | none | none for Specialized (only `SharedConst` variant uses ToSymbol — not the production variant) | none | none | `CudaTimer`/`elapsed_ms()`, same note | none — full recapture only | **Yes.** The `mode`-driven template selection happens entirely on the host before any GPU call, so it is capture-safe already (same pattern as `threshold`'s width%4 branch below); just needs the explicit-stream/no-sync treatment. |
| **Laplacian** (Specialized) | `laplacian_specialized_kernel<K><<<grid,block,shared>>>` (`laplacian_enhanced.cu:339-347`) | implicit default stream | `cudaDeviceSynchronize()` once (`:368`, after the switch) | none directly | **yes** — `cudaMemcpyToSymbol(d_laplacian_coeffs, host_coeffs, kernel_size*kernel_size*4)` (`:323`), synchronous. Unlike Sobel's Specialized (which hardcodes its 3x3 matrix in the kernel template and needs no coefficients at all), Laplacian's Specialized kernel still reads coefficient *values* from constant memory — only the kernel_size-driven loop bounds/shared-mem layout are templated, not the coefficient values themselves. | none | none | `CudaTimer`/`elapsed_ms()`, same note | none — full recapture only | **Yes**, same technique as Gaussian: explicit stream, no sync, `cudaMemcpyAsync` to the symbol address instead of `cudaMemcpyToSymbol`. |
| **Threshold** (Vectorized) | `threshold_vectorized_kernel<<<grid,block>>>` when `width % 4 == 0`, else automatic fallback to `threshold_scalar_kernel<<<grid,block>>>` (`threshold_enhanced.cu:103,111`) | implicit default stream | `cudaDeviceSynchronize()` once (`:129`) | none | none | none | none | `CudaTimer`/`elapsed_ms()`, same note | none — full recapture only | **Yes.** The `width % 4` branch is a plain host-side `int` comparison, resolved before any GPU call — already capture-safe; only needs the explicit-stream/no-sync treatment like the others. |

## Summary of blockers (all five stages, uniformly)

1. **`cudaDeviceSynchronize()` inside the dispatcher** (all five; Gaussian has it twice). This is the
   single blocker present in every stage — a non-stream-ordered call, illegal during stream capture.
   Called immediately after each kernel launch specifically to let the dispatcher read back that stage's
   `CudaTimer` before returning.
2. **No explicit stream parameter** (all five). Every kernel launches on the implicit default/legacy
   stream (`<<<grid, block>>>` or `<<<grid, block, shared_bytes>>>`, never a 4th stream argument). A
   kernel launched on a stream other than the one being captured is not captured into the graph — it
   silently executes outside it, producing an incomplete graph with no error.
3. **Synchronous `cudaMemcpyToSymbol` for coefficient upload — Gaussian and Laplacian only**, in their
   production (Specialized) configuration. Median, Sobel (Specialized), and Threshold (Vectorized) need
   no coefficient upload at all in production configuration, so this specific blocker does not apply to
   them. `cudaMemcpyToSymbol` aborts stream capture if issued on the capturing stream; its async
   counterpart (`cudaMemcpyToSymbolAsync`, or an equivalent `cudaMemcpyAsync` to the symbol's own address
   obtained via `cudaGetSymbolAddress`) is capture-legal.

None of these three blockers live in a `__global__` kernel — all three are in the *dispatcher's*
host-side orchestration code (the synchronize-per-stage pattern and the implicit-stream launches), which
is exactly the situation Section 20D found for the Basic-kernel dispatch path too. **Every blocker found
here can be avoided by isolated orchestration that calls the same `__global__` kernels directly**, with
no dispatcher involved and no kernel modified — confirmed per-stage in the table above.

## One additional finding specific to Enhanced (not present in Section 20D's Basic-kernel work)

Gaussian's and Laplacian's Specialized kernels read their coefficients from **`__constant__` memory
symbols** (`d_gaussian_coeffs_1d`, `d_laplacian_coeffs`) declared once, globally, for the whole process —
unlike the Basic kernels, which take a coefficient **device pointer** as a plain kernel argument (memory
this project's own code allocates and owns per call). This has one real, disclosed consequence: two
`CudaGraphEnhancedPipeline` instances performing truly *concurrent* graph replay on separate streams could
race on the same constant-memory symbol if their replays' coefficient-upload node and consuming kernel
node ever interleaved across instances. This project's actual usage pattern (one Streamlit session,
synchronous request/response, no concurrent multi-thread GPU submission) does not exercise that race, and
this section's own design keeps each replay's coefficient upload and its consuming kernel in the same
graph with an explicit dependency edge (upload happens immediately before the kernel that reads it, every
single replay) — so single-instance and even multiple *sequential* instances are safe, verified directly
in this section's own tests. Genuine concurrent multi-instance replay was not attempted (this project has
no such usage pattern) and is disclosed here as an unexercised, theoretical constraint of reusing the
existing Specialized kernels' constant-memory design, not something this section's orchestration layer
fixes.

## Conclusion

All five Enhanced dispatch functions are, in their current production form, not graph-capturable for
exactly the reasons Section 20D already found for the Basic-kernel dispatch path (per-stage
`cudaDeviceSynchronize`, implicit default stream) plus one Enhanced-specific case (synchronous
`cudaMemcpyToSymbol`, Gaussian/Laplacian only). Every blocker is host-side orchestration, not kernel
math, and every blocker has a direct, capture-legal replacement that calls the exact same `__global__`
kernel with the exact same parameters. This confirms the section's premise: an isolated orchestration
layer can wrap the existing Enhanced kernels for graph capture without modifying them.
