# Filter-Specific Optimization Matrix (Section 20A)

Methodology: for each filter, the CURRENT bottleneck is assessed from (a) real register/shared-memory data
freshly compiled this session (`nvcc --ptxas-options=-v`, zero spilling confirmed on every kernel below), (b)
real per-variant kernel-time measurements already stored in `benchmark_results/variant_sweeps/` (Section 17), and
(c) this project's own documented rejected-experiment history (README, Sections 6-10). No candidate below was
implemented without at least one of these three forms of real evidence.

| Filter | Current bottleneck (measured) | Current best (measured) | Candidate | Research evidence | Expected benefit | Cost | Benchmark? |
|---|---|---|---|---|---|---|---|
| Gaussian | Compute (separable 2-pass, 21-23 registers/pass, 0 spill) — already algorithmically minimal work per pixel | `specialized`, 3.39x vs Basic (measured) | Fused single-kernel h+v | General fusion principle (reduces one intermediate write+read) | Small: intermediate buffer is already reused, not re-allocated per pass; saving is one global r/w pair per pixel | Medium (register pressure risk for fused kernel) | No — no register-pressure evidence justifies the added complexity; intermediate traffic is already the smallest share of pipeline memory traffic |
| Gaussian | (as above) | (as above) | Vectorized (`uchar4`) loads | NVIDIA Best Practices: coalescing | Zero — dataset widths (e.g. 1733px) are not 4-aligned, forcing a scalar fallback path anyway (already investigated and rejected, Section 6) | N/A | No — already rejected with real reasoning on file |
| Median (k=3) | Compute (branchless sorting network, 36-55 registers depending on variant, 0 spill, 16-56B stack for insertion-sort variants) | `network3x3`, 5.82x vs Basic (measured) | Histogram-based rank selection | Peer-reviewed (ResearchGate 325944500; SPIE 13517): histogram methods scale better than sorting for **larger** windows; sorting networks remain competitive for 3x3 | None expected at k=3 — literature explicitly favors sorting networks at this window size, matching the project's own choice | High (new algorithm class) | No — literature argues against it for k=3 |
| Median (k=5/k=7) | Compute (`shared`/`specialized` only reach 1.00-1.04x vs Basic — much smaller gain than k=3's sorting network) | `specialized`, 1.03-1.04x vs Basic (measured) | Histogram-based rank selection | Same literature as above, now favorable: histogram methods' relative advantage grows with window size (9 vs 25 vs 49 compare-exchanges) | Plausible meaningful gain (current k=5/k=7 gains are the weakest in the whole project) | High (new kernel, new correctness verification, per-channel histogram bookkeeping in shared memory) | **PROMISING — flagged for a dedicated future experiment, not built in this research gate** (see item 40's complexity criterion) |
| Sobel | Compute/instruction (26 registers Basic, 18-39 across Enhanced variants, 0 spill) — `sqrt`-free (magnitude uses `abs_sum`/comparison, confirmed by low register counts) | `specialized`, 1.04x vs Basic (measured) | Any further restructuring | Sections 8 + 17 already measured Shared (1.02x), SharedConst (0.90x — regression), Separable (0.67x — regression) | None — two of four already-tried variants **regressed**; the two-launch separable decomposition's extra global read/write outweighs its reduced arithmetic for a 3x3 kernel this small | N/A | **No — KEEP CURRENT IMPLEMENTATION.** Nsight Systems shows Sobel now consumes the largest single share (28.2%) of Enhanced's own remaining per-call compute time, but that is a symptom of its *modest* achievable gain, not evidence of an untried technique |
| Laplacian (k=3) | Compute (39-40 registers across variants, 0 spill) — margin between variants is small | `specialized`, 1.02x vs Basic at k=3 (measured; `shared_const` was actually best at 1.28x in this session's fresh sweep — see note) | Re-evaluate default variant at k=3 | Section 17's own fresh measurement this session showed `shared_const` (1.28x) and `shared` (1.24x) both beating `specialized` (1.02x) at k=3 specifically, though `specialized` wins decisively at k=5 (2.69x) | Possible small win *at k=3 only* by conditionally selecting `shared_const` when `laplacian_kernel_size == 3` | Low (dispatch-only change, no new kernel) | **PROMISING — worth a dedicated re-run with more measurement runs before any production change**, because Sections 9/17's k=3 measurements have shown run-to-run variance large enough to flip the ranking (see `cuda_optimization_research.md` "Measurement variance" note) |
| Laplacian (k=5) | (as above) | `specialized`, 2.69x vs Basic (measured) | — | Already the clear winner, consistent across two independent measurement sessions | — | — | No — KEEP |
| Threshold | Memory bandwidth (10-11 registers, 0 spill, no shared memory, no neighborhood — confirmed pure pointwise/bandwidth-bound) | `vectorized`, 1.62x vs Basic (measured) | Further vectorization (`uchar8`-equivalent) | NVIDIA Best Practices: coalescing | None expected — already memory-bandwidth-bound at the smallest practical register footprint in the whole project; `multi_pixel` (more work per thread, same access width) already measured as a **regression** (0.92x) | N/A | No — KEEP CURRENT IMPLEMENTATION, at the practical limit for this operation class |

## Notes

**Measurement variance (Laplacian k=3).** Two Section 20A measurement passes on this machine (Section 17's
original sweep and a repeat during this section's own investigation) put `specialized` and `shared_const` within
noise of each other at k=3 (both under ~0.3ms per 122-image batch, where CUDA-event timing jitter is a larger
fraction of the signal than at k=5's ~0.7-2.6ms range). This is disclosed rather than picking whichever number
looks better — per this project's own rule (measure, don't fabricate), a ranking this close needs more
measurement runs before it can justify a production dispatch change, so it is marked PROMISING rather than KEEP
or a firm recommendation to switch.

**Why "KEEP" dominates this table.** Sections 6-10 and 17 already ran a genuinely thorough per-filter search
(4-5 variants per filter, block/tile sweeps, kernel-size-specific sweeps) and this session's fresh register/
shared-memory data shows **zero spilling on every kernel** — the textbook first sign that further register-level
tuning would help. Combined with the peer-reviewed literature's own guidance that sorting networks (not
histograms) are the right choice at small kernel sizes, and Sobel/Threshold's own history of *regressing* under
further attempted optimization, the evidence-based conclusion is that per-filter kernel optimization is close to
exhausted for this project's default configuration. The genuinely new opportunity this section's research
surfaced is at the **pipeline** level — see `pipeline_optimization_matrix.md`.
