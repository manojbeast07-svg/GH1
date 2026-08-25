# Final Validation Report — Section 22 (Release Audit, Cleanup, Documentation, Freeze)

Generated 2026-08-25, on the local development machine (LOCAL ENVIRONMENT — no Brev/cloud session was
available; Section 19's clean-environment verification remains not-yet-executed, unchanged from Section 20).

## 1. Architecture

Confirmed to match exactly the three-tier diagram (verified by direct code audit, not assumed):

```
Python + OpenCV CPU  --  cpu/pipeline.py:run_cpu_pipeline() -> cpu/filters.py (cv2.*, no CUDA coupling)
        |
C++ Basic CUDA (.cu)  --  cuda/pipeline.py:run_basic_cuda_pipeline() -> xray_cuda.run_basic_cuda_pipeline_gpu()
        |                 -> cuda/src/pipeline_basic.cu
C++ Enhanced CUDA (.cu) -- cuda/pipeline.py:run_enhanced_cuda_pipeline() -> same native binding, all-Enhanced
                            variant flags -> cuda/src/*_enhanced.cu dispatch functions
```

Production call chain, traced end-to-end: `app.py` -> `ui/services.py` (`run_cpu_batch`/`run_gpu_batch`) ->
`cpu/pipeline.py` or `cuda/pipeline.py` -> pybind11 (`xray_cuda`, for GPU) -> native `.cu` code. No fourth path
exists; `cuda.pipeline.run_cuda_pipeline()` is a per-stage Basic/Enhanced generalization used by the
Optimization Lab, not a distinct production path — it still bottoms out in the same native binding.

## 2. Environment

| | |
|---|---|
| OS | Windows-11-10.0.26200-SP0 |
| CPU | AMD64 Family 25 Model 116 Stepping 1, AuthenticAMD, 16 logical cores |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8,585,216,000 bytes VRAM, compute capability 8.9, 1 async copy engine |
| Driver / CUDA runtime | 13.2 |
| NVCC | 13.3.73 |
| CMake | 4.4.2 |
| Python | 3.13.9 |
| NumPy | 2.3.5 |
| OpenCV | 5.0.0.93 |
| pybind11 | 3.1.0 |
| Environment label | LOCAL ENVIRONMENT |

## 3. Build

Performed a full clean build (deleted `build/` and the `.pyd`, reconfigured from scratch):

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="C:/Users/manoj/anaconda3/python.exe"
cmake --build build --config Release
```

Result: CMake configure succeeded (MSVC 19.51.36252.0, CUDA 13.3.73 detected), all 22 build targets compiled and
linked successfully (19 `.cu` files + `bindings.cpp`), `xray_cuda.cp313-win_amd64.pyd` produced. `import
xray_cuda` succeeds; `xray_cuda.cuda_available()` returns `True`. Only benign, pre-existing unused-variable
compiler warnings (Sobel Enhanced template instantiations, `MODE=0`/`MODE=1`) — no errors.

## 4. Tests

Full suite, three ways:

| Run | Result |
|---|---|
| Default seed | 2006 / 2006 passed |
| `PYTHONHASHSEED=1` | 2006 / 2006 passed |
| `PYTHONHASHSEED=99999` | 2006 / 2006 passed |

2006 matches the Section 20F baseline exactly — no regression introduced by this section's own changes (a
`.gitignore` addition and README edits, neither of which touch code the suite exercises).

## 5. Determinism

PASS — all three runs above produced identical pass counts with zero flakes across three separate full-suite
invocations (run sequentially, not concurrently, avoiding the GPU-contention flakiness observed and documented
in Section 20D).

## 6. Correctness

Sourced from the canonical stored benchmark (`benchmark_results/summary/20260824_055212_seed42_batch122_224x224.json`,
`correctness.detailed.filter_level`), cross-checked against a fresh single-image and batch smoke test on the
current dataset (this section, see §16-17 below):

| Filter | Tolerance | Result |
|---|---|---|
| Gaussian | ±1 (documented floating-point summation-order tolerance) | PASS |
| Median | exact | PASS |
| Sobel | exact | PASS |
| Laplacian | exact | PASS |
| Threshold | exact | PASS |
| Basic vs Enhanced (GPU-to-GPU) | bit-exact required | PASS (0 differing pixels, canonical benchmark; bit-exact in this section's own fresh single-image and batch smoke tests) |
| CPU vs GPU | differing-pixel-percentage methodology | 0.0178% differing (canonical); 0.008%-0.018% in this section's fresh spot-checks |

No tolerance was weakened or introduced.

## 7. Canonical benchmark

Dataset fingerprint check (spec item 23): current dataset fingerprint
`f4691591df89705db97946a0c935918a89129e6d64a8dc37209abea686864aab` **matches** the canonical benchmark's
recorded fingerprint exactly — **no DATASET MISMATCH**. The stored canonical benchmark
(`20260824_055212_seed42_batch122_224x224`, seed=42, requested batch=125, selected=122 images, 224x224) remains
valid and was used as-is for the final benchmark table below, per spec item 24's "explicitly selected stored
benchmark" option (re-running the full 20-run canonical suite was not necessary since the dataset is unchanged
and Sections 20A-20F made zero production code changes that could affect it).

## 8. Performance

**Final benchmark table** (source: canonical stored benchmark `20260824_055212_seed42_batch122_224x224`, 122
images, 224x224, 20 measurement runs; all values reused from that stored run, not re-measured, since the
dataset fingerprint check above confirms it is still valid):

| Metric | CPU/OpenCV | Basic CUDA | Enhanced CUDA |
|---|---:|---:|---:|
| Gaussian (median, ms) | -- | 1.929 | 0.556 |
| Median (median, ms) | -- | 2.579 | 0.451 |
| Sobel (median, ms) | -- | 0.640 | 0.704 |
| Laplacian (median, ms) | -- | 0.859 | 0.665 |
| Threshold (median, ms) | -- | 0.140 | 0.088 |
| GPU compute (mode2/3, median, ms) | N/A | 5.496 | 3.278 |
| End-to-end (mode4, median, ms) | 89.446 | 60.066 | 57.848 |
| Speedup vs CPU (end-to-end) | 1x | 1.483x | 1.535x |
| Speedup Enhanced vs Basic (compute-only) | -- | -- | 2.734x |
| Speedup Enhanced vs Basic (end-to-end) | -- | -- | 1.035x |

Per-filter kernel speedups (Basic -> Enhanced, median): Gaussian 3.45x, Median 5.70x, Sobel 0.90x (a documented,
expected regression — Sobel's Basic kernel is already near-optimal for this small a stencil; see
`research/filter_optimization_matrix.md`), Laplacian 1.36x, Threshold 1.64x.

**Real single-image spot check** (this section, fresh measurement, seed=42, one 224x224 image):
CPU 6.69ms, Basic CUDA 3.62ms (H2D 0.041ms, D2H 0.072ms), Enhanced CUDA 0.47ms (H2D 0.043ms, D2H 0.036ms).

**Real batch spot check** (this section, fresh measurement, seed=42; batch sizes below reflect the actual count
of same-resolution images available in the seed=42 selection, never silently padded — consistent with every
prior section's disclosed convention):

| Requested | Actual (same-res) | Basic total (ms) | Enhanced total (ms) | Basic==Enhanced bit-exact |
|---:|---:|---:|---:|---|
| 8 | 7 | 4.310 | 1.097 | Yes |
| 32 | 31 | 2.109 | 1.041 | Yes |
| 122 | 119 | 5.281 | 2.754 | Yes |

## 9. Streamlit verification

`streamlit run app.py --server.headless true` launched cleanly; `curl http://localhost:8502` returned HTTP 200;
server log showed zero Python errors/tracebacks (only a benign, unrelated Streamlit CORS/XSRF informational
warning). Tab wiring confirmed present in `app.py`: Live Processing, CPU vs GPU, Performance Analytics,
Optimization Lab, Correctness, System, Presentation Mode (all 6 tabs named in the spec are present, plus one
additional "CPU vs GPU" tab). Server was stopped cleanly after verification; no process left running.

## 10. Optimization decisions

Frozen, per Section 22's spec item 4 (all verified not referenced by `app.py`/`ui/*.py`/`cuda/pipeline.py`):

| Feature | Status | Section |
|---|---|---|
| Persistent GPU buffers | REJECTED for production | 20B/20C |
| Pinned host memory | REJECTED for production | 20B |
| CUDA Graph — Basic | EXPERIMENTAL | 20D |
| CUDA Graph — Enhanced | EXPERIMENTAL | 20E |
| Async multi-stream pipeline | EXPERIMENTAL | 20F |
| Laplacian+Threshold fusion | EXPERIMENTAL | 10 |

## 11. Experimental features

All five experimental/rejected pipeline classes (`PersistentCudaPipeline`, `CudaGraphPipeline`,
`CudaGraphEnhancedPipeline`, `AsyncCudaPipeline`, plus the Section 10 fusion kernel) exist only in the compiled
native extension (via `cuda/src/bindings.cpp`) and their own isolated Python wrapper
(`cuda/persistent_pipeline_experimental.py`, confirmed inert — zero inbound references from `app.py`/`ui/`/
`cuda/pipeline.py`). Left in place per spec item 35's explicit instruction not to relocate already-tested
experimental code for cosmetic reasons. All research evidence, decision documents, and raw benchmark results are
preserved (`research/` — 11 markdown documents; `benchmark_results/research_optimization/` — 5 experiment
namespaces: `persistence_pinned/`, `persistent_api/`, `cuda_graphs/`, `enhanced_cuda_graph/`, `async_pipeline/`).

## 12. Repository hygiene

**This is the most significant finding of this audit and is reported in full rather than summarized away.**

The actual git repository root is `C:\Users\manoj\Desktop\dev\GH1` (one level above `xray_cuda_demo/`), not
`xray_cuda_demo/` itself. That repository's entire history is **one commit**: `c3763d3 Create README.md` (an
essentially empty, 2-byte root `README.md`, unrelated to the project's real README). **The entire
`xray_cuda_demo/` project tree — every file from Section 1 through Section 20F, including all source, tests,
research, and documentation — is untracked, uncommitted local state.** `git ls-files` returns 0 entries.

This was verified directly (`git status --short` from the repo root: `?? data/` and `?? xray_cuda_demo/`
before this section's fix, matching the exact two-line output seen at the very start of this entire multi-
section conversation).

**Dataset protection — gap found and fixed.** The X-ray dataset (`GH1/data/`, 9,463 images, ~198MB) is
deliberately placed one directory level outside `xray_cuda_demo/` (per `config.yaml`'s own documented comment:
"the actual X-ray dataset... was placed one level up, at the workspace root"). `xray_cuda_demo/.gitignore`
already correctly protects the placeholder `xray_cuda_demo/data/.gitkeep` directory, but — because the real
dataset lives *outside* that directory, at the actual repo root — **no `.gitignore` rule anywhere in the
repository actually covered `GH1/data/`** before this section. `git check-ignore -v data/` returned no match.
This meant a future `git add -A` from the repo root would have staged all 9,463 dataset images. **Fixed in this
section**: added `GH1/.gitignore` containing `data/` — verified via `git check-ignore -v data/` now correctly
reporting the new rule. No dataset file was ever staged or committed at any point.

**Other hygiene checks, all clean:**
- Secrets audit: no API keys, tokens, passwords, private keys, or `.env` files found (pattern search across
  `.py`/`.yaml`/`.cfg`/`.ini`/`.json`, excluding benchmark/cache directories).
- Machine-specific path audit: zero occurrences of `C:\Users\manoj`, `/home/manoj`, or `Users/manoj` in any
  production source file (`.py`/`.cu`/`.cuh`/`.cpp`/`.hpp`/`.yaml`/`.txt`/`.cfg`/build config); `config.yaml`
  uses a portable relative path (`../data`).
- Generated-file audit: `build/`, `__pycache__/`, `.pytest_cache/`, `*.pyd` are present (expected, from this
  section's own clean build and test run) and are all correctly covered by `xray_cuda_demo/.gitignore`.
- `git status --ignored --short` confirms `outputs/*.json` benchmark scratch files, `build/`, `__pycache__/`,
  and `.pytest_cache/` are all properly ignored, not accidentally trackable.

**Not fixed, and deliberately left for the user's decision**: whether/when to actually run `git add` +
`git commit` to bring `xray_cuda_demo/` under version control. Per this section's own explicit instruction
("Do not create a commit unless explicitly requested" / "Do NOT commit anything unless explicitly instructed"),
no commit was made. This is flagged here as the clearest actionable follow-up from this entire audit.

## 13. Limitations

Unchanged from Section 20, now also covering Sections 20A-20F (see README's "Known limitations" section,
updated in this section): technical demonstration only, not a diagnostic tool; GPU performance is hardware-
dependent; end-to-end speedup is smaller than compute-only speedup due to host-side/transfer overhead; small
CPU/GPU numerical differences are expected and documented (Gaussian ±1, threshold amplification); the canonical
benchmark reflects one specific local machine (Brev/clean-environment verification remains not executed); the
Section 20A-20F experimental optimizations are not production defaults and are not selected by any UI default.

## 14. Release decision

```
READY FOR RELEASE
```

with one explicitly flagged, non-blocking follow-up: the project has never been committed to git (see §12).
This does not block release readiness by the criteria this audit was asked to verify (correctness,
reproducibility, cleanliness, documentation, buildability, testability, presentation-readiness, safety) — all
of which passed — but it is the single most important fact for the user to act on next, since "release" in the
version-control sense has not yet happened. No dataset, secret, or machine-specific path is at risk of being
committed accidentally now that the `.gitignore` gap is closed.
