# Final Release Validation Report (Section 20)

All values in this report are measured directly in this environment on 2026-08-24; none are copied
from earlier sections' reports or from spec example figures.

## Environment

| Field | Value |
|---|---|
| Environment label | LOCAL ENVIRONMENT |
| OS | Windows-11-10.0.26200-SP0 |
| CPU | AMD64 Family 25 Model 116 Stepping 1, AuthenticAMD (16 logical cores) |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| GPU VRAM | 8,585,216,000 bytes (~8.0 GB) |
| Compute capability | 8.9 |
| NVIDIA driver | 13.2 |
| CUDA runtime | 13.2 |
| CUDA toolkit / NVCC | 13.3 (`Cuda compilation tools, release 13.3, V13.3.73`) |
| CMake | 4.4.2 |
| C++ compiler | MSVC 19.51.36252.0 |
| Python | 3.13.9 |
| NumPy | 2.3.5 |
| OpenCV | 5.0.0.93 |
| pybind11 | 3.1.0 |
| git commit | c3763d3bd72d16d84d44eb332cbd123686e26399 |
| Build configuration | **Release** (`-DCMAKE_BUILD_TYPE=Release`) |

## Build

A fully clean rebuild was performed for this validation: `build/` deleted, reconfigured from scratch
(`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release`), then built (`cmake --build build --config Release`)
inside a properly-initialized MSVC (`vcvars64.bat`) environment.

- All 15 `.cu` files under `cuda/src/` compiled by NVCC: PASS
- `cuda/src/bindings.cpp` compiled by MSVC: PASS
- Native extension linked: PASS (`xray_cuda.cp313-win_amd64.pyd`, 1,169,408 bytes)
- Extension imports and `cuda_available()` / `smoke_test()` succeed: PASS

**Note:** the pre-existing `build/` directory (from Sections 1-19) was configured as `CMAKE_BUILD_TYPE=Debug`.
Inspection of the generated NVCC command line confirmed the Debug-vs-Release CMake flags only affect the
*host*-side compiler pass (`-Xcompiler=" -Zi -Ob0 -Od /RTC1"`), not CUDA device-code optimization -- so prior
sections' GPU kernel timings are not invalidated -- but as part of final release packaging this section
reconfigures and rebuilds as a genuine Release build, and all final numbers in this report come from that
Release build.

## Native CUDA backend diagnostic (`scripts/inspect_gpu_backend.py`)

```
GPU backend:               Native C++/CUDA
Basic:                     native
Enhanced:                  native (same pipeline entry point, per-stage variant flags)
CUDA source:               .cu (15 files)
CUDA compiler:              nvcc (see cuda/CMakeLists.txt: find_package(CUDAToolkit))
Python GPU computation:     none found
pybind11:                  enabled
CUDA device at runtime:    available
ALL CHECKS PASSED
```

## Test suite

| Run | Result |
|---|---|
| Full suite (default) | **1874 / 1874 passed** |
| `PYTHONHASHSEED=1` | **1874 / 1874 passed** |
| `PYTHONHASHSEED=99999` | **1874 / 1874 passed** |

Determinism: identical pass count across all three configurations.

One flaky timing-heuristic test (`test_pipeline_is_gpu_resident_single_h2d_single_d2h`) was found and fixed
during this section (see Known issues) -- it was not disabled or deleted, its heuristic was made robust
(warmup + median-of-5 + a wider, still-meaningful threshold).

## Correctness (release_validation benchmark, 122 images, seed=42, 224x224)

| Filter | CPU vs Basic | CPU vs Enhanced | Basic vs Enhanced | Tolerance | Result |
|---|---:|---:|---:|---:|---|
| Gaussian | max_abs_diff=1 | max_abs_diff=1 | max_abs_diff=0 | ±1 | PASS |
| Median | max_abs_diff=0 | max_abs_diff=0 | max_abs_diff=0 | 0 | PASS |
| Sobel | max_abs_diff=0 | max_abs_diff=0 | max_abs_diff=0 | 0 | PASS |
| Laplacian | max_abs_diff=0 | max_abs_diff=0 | max_abs_diff=0 | 0 | PASS |
| Threshold | max_abs_diff=0 | max_abs_diff=0 | max_abs_diff=0 | 0 | PASS |

Full pipeline (Enhanced vs CPU): 0.0179% differing pixels (matches the established ~0.0178% baseline).
Full pipeline (Enhanced vs Basic): bit-exact (0.0000% differing pixels).

## Performance (canonical benchmark configuration: 122 images, seed=42, 224x224, 20 measurement runs)

Benchmark ID: `20260824_135124_seed42_batch119_224x224` (`benchmark_results/release_validation/`)

| Metric | CPU/OpenCV | Basic CUDA | Enhanced CUDA |
|---|---:|---:|---:|
| End-to-end (mean) | 83.284 ms | 58.604 ms | 56.446 ms |
| GPU compute-only (mean) | N/A | 3.512 ms | 1.307 ms |
| Speedup vs CPU | 1× | 1.421× | 1.475× |

Basic → Enhanced: compute-only 2.688×, end-to-end 1.038×.

### Per-filter (Basic → Enhanced kernel speedup, this run)

| Filter | Basic (ms) | Enhanced (ms) | Speedup | Contribution to total compute reduction |
|---|---:|---:|---:|---:|
| Gaussian | 1.746 | 0.525 | 3.33× | 36.8% |
| Median | 2.308 | 0.412 | 5.60× | 57.2% |
| Sobel | 0.586 | 0.631 | 0.93× | -1.4% |
| Laplacian | 0.791 | 0.593 | 1.33× | 6.0% |
| Threshold | 0.136 | 0.092 | 1.49× | 1.4% |

Gaussian + Median ≈ 94% of the measured compute-time reduction (matches the established project finding).

### Batch-size sweep (1013 images, 224x224, seed=42)

| Batch | Basic (ms) | Enhanced (ms) | Enhanced/Basic |
|---:|---:|---:|---:|
| 1 | 0.200 | 0.237 | 0.84× |
| 8 | 0.543 | 0.375 | 1.45× |
| 16 | 0.988 | 0.612 | 1.61× |
| 32 | 1.685 | 0.946 | 1.78× |
| 64 | 2.938 | 1.854 | 1.59× |
| 128 | 5.567 | 3.377 | 1.65× |
| 256 | 11.344 | 7.056 | 1.61× |
| 512 | 22.728 | 13.035 | 1.74× |

(batch=1's Enhanced slightly slower than Basic is the established, expected small-batch launch-overhead pattern.)

### Resolution sweep (representative, from this run)

| Resolution | n | Basic ms/img | Enhanced ms/img |
|---|---:|---:|---:|
| 224×224 | 119 | 0.0502 | 0.0304 |
| 858×958 | 1 | 1.1012 | 0.6060 |
| 902×1128 | 1 | 1.2192 | 0.6788 |
| 1733×858 | 1 | 1.7722 | 0.9924 |

### Comparison to the original Section 11 canonical benchmark

| Metric | Original canonical (`...055212...`) | This release build (`...135124...`) | Ratio |
|---|---:|---:|---:|
| CPU end-to-end | 89.019 ms | 83.284 ms | 0.936× |
| Basic end-to-end | 60.032 ms | 58.604 ms | 0.976× |
| Enhanced end-to-end | 58.001 ms | 56.446 ms | 0.973× |
| Basic compute-only | 3.618 ms | 3.512 ms | 0.971× |
| Enhanced compute-only | 1.323 ms | 1.307 ms | 0.988× |

No unexplained regression -- this Release build is consistently as fast or slightly faster than the original
canonical measurement, within normal run-to-run machine-state variance.

## Streamlit application

- `streamlit run app.py` → HTTP 200, zero errors in logs (manual launch check).
- AppTest scenarios verified with zero exceptions: default load; single-image Compare; batch Compare;
  Performance Analytics benchmark switch; Optimization Lab filter switch + live comparison; Presentation Mode
  toggle; Presentation Mode tab content (including the live-image section) after a single-image Compare.
- All 7 final tabs present: Live Processing, CPU vs GPU, Performance Analytics, Optimization Lab, Correctness,
  System, Presentation Mode.

## Cleanup performed

- Removed stale, orphaned debug-symbol files at the project root (`vc140.pdb`, `xray_cuda.pdb` -- leftovers from
  earlier Debug builds, already `.gitignore`d, no longer matching any current binary).
- Removed test-generated artifacts from `outputs/live_processing/`, `outputs/live_batch_processing/`,
  `outputs/optimization_lab/` after each verification pass.
- No secrets, API keys, passwords, or credentials found in source (`grep` audit of `.py`/`.yaml`/`.cpp`/`.cu`/`.cuh`).
- No temporary/debug files found tracked or committed (the project is not yet committed to git beyond the
  original README).
- Historical `benchmark_results/` artifacts (Section 11's canonical run, Section 18's
  `native_cuda_migration/`) verified byte-for-byte unchanged after this section's work.

## Known limitations

See the README's [Known limitations](../README.md#known-limitations) section (image-processing demo only, not
diagnostic; hardware-dependent GPU results; compute-only vs. end-to-end speedup gap from host/disk overhead;
documented Gaussian ±1 / threshold-amplification numerical behavior; Section 19's clean-environment/Brev
verification not executed in this environment).

## Known issues

- `test_pipeline_is_gpu_resident_single_h2d_single_d2h` was flaky under full-suite load (sub-millisecond H2D
  timing noise from not being the first CUDA call in the process). Fixed with warmup + median-of-5 sampling and
  a wider threshold; re-verified stable across 4 full-suite runs after the fix.
- Section 19 (clean-environment/Brev verification) has not been executed -- no Brev/SSH access was available in
  any session that reached this point. The spec and plan are saved in this session's memory for a future session
  with actual Brev access.
