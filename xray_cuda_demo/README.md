# xray_cuda_demo

A CUDA-accelerated X-ray image-processing performance demonstration
(CPU/OpenCV vs. Basic CUDA vs. Enhanced CUDA). This is a technical
performance demo, not a medical diagnostic tool.

> **Status: Section 1 (CUDA foundation) + Section 2 (dataset handling) +
> Section 3 (CPU reference pipeline) + Section 4A (CUDA image
> infrastructure) + Section 4B (Basic CUDA Gaussian blur) + Section 4C
> (Basic CUDA median filter) + Section 4D (Basic CUDA Sobel edge
> detection) + Section 4E (Basic CUDA Laplacian filter) + Section 4F
> (Basic CUDA binary threshold + full 5-filter GPU-resident smoke test)
> + Section 5 (production Basic CUDA pipeline + batching + CPU-vs-GPU
> baseline benchmark) + Section 6 (Enhanced CUDA Gaussian: separable +
> shared memory + constant memory + specialization, 3.34× measured
> kernel speedup, 1.20× full-pipeline speedup) + Section 7 (Enhanced CUDA
> Median: shared-memory tiling + branchless 3×3 sorting network +
> compile-time specialization, 5.86× measured kernel speedup at k=3,
> 1.49× full-pipeline speedup) + Section 8 (Enhanced CUDA Sobel:
> shared-memory tiling + constant memory + compile-time mode
> specialization + separable decomposition, 1.03-1.07× measured kernel
> speedup) + Section 9 (Enhanced CUDA Laplacian: shared-memory tiling +
> constant memory + compile-time specialization + hand-written explicit
> arithmetic, 1.5-2.9× measured kernel speedup, four of five filters now
> Enhanced) + Section 10 (Enhanced CUDA Threshold: uchar4 vectorization,
> ~1.4-1.8× measured kernel speedup; final canonical
> `run_basic_cuda_pipeline()` / `run_enhanced_cuda_pipeline()` all-Basic
> vs. all-Enhanced pipelines, all five filters now Enhanced; ~2.7×
> measured GPU-compute-only speedup, CPU/Basic/Enhanced three-way
> benchmark infrastructure) + Section 11 (final reproducible benchmark &
> experiment framework: dataset/environment/configuration fingerprints,
> deterministic benchmark IDs, four separated timing modes, raw +
> aggregated results, batch-size + resolution sweeps, per-filter
> Amdahl-style contribution analysis, 3-level correctness benchmark,
> on-disk manifests + reproduction command, loader APIs for Section 12)
> + Section 12 (final Streamlit interactive CUDA X-ray lab) + Section 14
> (live single-image CPU vs Basic CUDA vs Enhanced CUDA processing) +
> Section 15 (live real-batch processing, aggregate correctness, batch-
> size sweep) + Section 16 (historical Performance Analytics: benchmark
> selector, canonical badge, GPU compute breakdown, benchmark comparison,
> CSV/JSON export) + Section 17 (Interactive CUDA Optimization Lab:
> per-filter historical variant sweeps, live variant comparison, pipeline
> impact) + Section 18 (native C++/CUDA architecture audit -- confirmed
> already fully native, no migration needed) + Section 20 (Presentation
> Mode, README finalization, final release validation) + Sections
> 20A-20F (research-driven optimization audit: filter-specific and
> pipeline-level research; buffer persistence and pinned host memory
> experiments, both REJECTED for production after measurement against
> the realistic Streamlit workload; CUDA Graph capture for Basic and
> Enhanced pipelines, both EXPERIMENTAL; multi-stream async
> double/triple buffering, EXPERIMENTAL -- none of these changed the
> production pipeline) + Section 22 (final release audit, repository
> hygiene, documentation, freeze) **RELEASED / FROZEN.**
> See [Streamlit Application](#streamlit-application-section-12) below
> for Sections 14-20's own subsections,
> [Optimization Research](#optimization-research-sections-20a-20f) for
> the post-release experiment arc, [Project status](#project-status)
> for the release/freeze statement, and
> [Known limitations](#known-limitations) for what this demo does not
> claim. (Section 19, clean-environment/Brev verification, is planned
> but was not executed in this environment -- see
> [Project status](#project-status).)
> Any further optimization would be a separately measured experiment,
> not a silent change to the production baseline.

See [environment_report.md](environment_report.md) for the full detected
hardware/software inventory.

## Requirements

- NVIDIA GPU (CUDA-capable; this project was validated on an RTX 4060
  Laptop GPU, compute capability 8.9)
- NVIDIA driver with CUDA support
- CUDA Toolkit (nvcc) — validated with 13.3
- C++ compiler — MSVC on Windows (Visual Studio Build Tools/Community),
  or g++ on Linux
- CMake >= 3.24 (for `CMAKE_CUDA_ARCHITECTURES=native` GPU auto-detection)
- Python 3.9+

## Installation

From the `xray_cuda_demo/` directory:

```bash
pip install -r requirements.txt
```

This installs `numpy`, `opencv-python`, `streamlit`, `pybind11`, `cmake`,
`pytest`, and `pyyaml`.

If more than one Python interpreter is present on your machine, make sure
the one on `PATH` (`python -c "import sys; print(sys.executable)"`) is the
one you installed the requirements into — CMake will otherwise pick
whichever Python it finds first, which may not have pybind11 installed.

## Build

**Windows (MSVC):** run from a "Developer Command Prompt for VS" (or run
`vcvars64.bat` first) so `cl.exe` and `nvcc` are both on `PATH`:

```bash
cmake -S . -B build -G Ninja -DPython3_EXECUTABLE="<path to your python.exe>"
cmake --build build --config Release
```

**Linux:**

```bash
cmake -S . -B build
cmake --build build -j
```

`CMAKE_CUDA_ARCHITECTURES` defaults to `native`, so CMake targets whatever
GPU is physically present on the build machine rather than a hard-coded
compute capability. Override with `-DCMAKE_CUDA_ARCHITECTURES=<arch>` if
cross-compiling for a different GPU.

The build produces `xray_cuda*.pyd` (Windows) / `xray_cuda*.so` (Linux) in
the project root, so `import xray_cuda` works from there without extra
path setup.

## Smoke test

```bash
python -c "import xray_cuda; print(xray_cuda.add_ints(2, 3))"
python -c "import xray_cuda; print(xray_cuda.cuda_available())"
python -c "import xray_cuda; print(xray_cuda.device_info())"
python -c "import xray_cuda; print(xray_cuda.smoke_test())"

pytest -q

python scripts/diagnose.py
```

## Expected result

```text
$ python -c "import xray_cuda; print(xray_cuda.add_ints(2, 3))"
5

$ python -c "import xray_cuda; print(xray_cuda.cuda_available())"
True

$ python -c "import xray_cuda; print(xray_cuda.device_info())"
{'available': True, 'name': 'NVIDIA GeForce RTX 4060 Laptop GPU', 'compute_capability': '8.9', ...}

$ python -c "import xray_cuda; print(xray_cuda.smoke_test())"
{'output': [1.0, 2.0, 3.0, 4.0, 5.0], 'kernel_ms': ...}

$ pytest -q
5 passed in ...s
```

If no CUDA device is available on the machine, the pytest smoke tests are
skipped explicitly (not silently reported as passed).

## GPU Architecture

The production GPU implementation is native C++/CUDA. Python is used for
UI, orchestration, and bindings only.

```text
Python / Streamlit
       │
    pybind11
       │
   Native C++            (cuda/src/bindings.cpp -- validates dtype/shape,
       │                  builds a native config struct, calls the
       ▼                  pipeline, marshals results back; no filter
   CUDA `.cu`             algorithm logic lives here)
       │
      GPU
```

Every filter (Gaussian, Median, Sobel, Laplacian, Threshold) is
implemented as a `.cu` kernel under `cuda/src/`, declared in a matching
`cuda/include/*.cuh` header, and compiled by NVCC via `cuda/CMakeLists.txt`
(`find_package(CUDAToolkit)`, `CMAKE_CUDA_ARCHITECTURES=native`). Basic
and Enhanced share the same production pipeline entry point,
`run_basic_cuda_pipeline_batch()` (`cuda/src/pipeline_basic.cu`): ONE
native call performs H2D → Gaussian → Median → Sobel → Laplacian →
Threshold → D2H entirely on the GPU, using two persistent, reused
`GpuImageBatch` ping-pong buffers for the whole call -- no
GPU→CPU→GPU round trip between stages, and no per-stage
`cudaMalloc`/`cudaFree`. A per-stage `use_enhanced`/`variant` flag in
`BasicPipelineConfig` selects Basic or a named Enhanced variant
(`specialized`, `network3x3`, `vectorized`, ...) for each filter
independently; "the Enhanced pipeline" is this same native function
called with all five flags set to the production defaults, not a
separate code path.

GPU memory is RAII-owned (`GpuImage`/`GpuImageBatch`/`DeviceBuffer<T>`
in `cuda/include/gpu_image.cuh` / `cuda_common.cuh`) -- no raw unmanaged
`cudaMalloc`/`cudaFree` pairs. All kernel/H2D/D2H timing is measured
with CUDA events (`CudaTimer`, `cudaEventRecord`/`cudaEventElapsedTime`),
never Python wall-clock. Native errors become specific Python
exceptions via pybind11 (`std::invalid_argument` → `ValueError`,
`std::runtime_error` → `RuntimeError`, `CudaMemoryError` →
`GpuMemoryError`, a `MemoryError` subclass).

`cuda/*.py` (`gaussian.py`, `median.py`, ..., `pipeline.py`) are thin
wrappers around the compiled `xray_cuda` extension -- they validate
arguments and call into native code, they never compute a filter
result in Python. The Streamlit production path
(`ui/services.py::_gpu_batch()`) calls only `run_basic_cuda_pipeline()`
/ `run_enhanced_cuda_pipeline()` / `run_cuda_pipeline()`, each a thin
wrapper around the single native pipeline call above -- never a
Python-side chain of five separate filter calls. Individual per-filter
Python bindings (`gaussian_basic_gpu`, `median_enhanced_gpu`, ...) still
exist and remain in use for unit tests, diagnostics, and the
Optimization Lab's live variant comparison (Section 17) -- never for
production batch/pipeline execution.

Run `python scripts/inspect_gpu_backend.py` to verify these claims
against the currently-built extension rather than taking this
description on faith.

## Project layout

```text
xray_cuda_demo/
├── app.py                  # Streamlit entry point: thin orchestrator over ui/ (Section 12)
├── CMakeLists.txt          # Top-level build: finds CUDA, adds cuda/ subdir
├── requirements.txt
├── config.yaml              # dataset.path, cuda.enabled, benchmark run counts
├── logging_config.py       # Shared INFO/WARNING/ERROR/DEBUG console logging
├── environment_report.md
├── cpu/
│   ├── filters.py            # FilterConfig + apply_gaussian/median/sobel/laplacian/threshold (Section 3)
│   ├── pipeline.py            # run_cpu_pipeline/run_cpu_image/run_cpu_selection (Section 3)
│   └── benchmark.py           # benchmark_cpu_pipeline, run_selection_benchmark (Section 3)
├── cuda/
│   ├── include/
│   │   ├── cuda_common.cuh # CUDA_CHECK macros, DeviceInfo, DeviceBuffer<T>, CudaTimer
│   │   ├── smoke_test.hpp
│   │   ├── gpu_image.cuh    # GpuImage (RAII), CudaStream, CudaMemoryError, launch-grid helper,
│   │   │                    #   reflect101() border helper shared by Gaussian/Sobel (Section 4A, extended 4D);
│   │   │                    #   clamp_index() BORDER_REPLICATE helper shared by Median (Section 7, moved from median_basic.cu)
│   │   ├── gaussian_basic.cuh # naive Gaussian kernel + launcher declaration (Section 4B)
│   │   ├── median_basic.cuh  # naive median kernel + launcher declaration (Section 4C)
│   │   ├── sobel_basic.cuh   # naive Sobel kernel + SobelMode enum + launcher declaration (Section 4D)
│   │   ├── laplacian_basic.cuh # naive Laplacian kernel + launcher declaration (Section 4E)
│   │   ├── threshold_basic.cuh # simplest kernel: pointwise comparison, launcher declaration (Section 4F)
│   │   ├── pipeline_basic.cuh  # BasicPipelineConfig/Timing, GpuImageBatch-based launcher declaration (Section 5)
│   │   ├── gaussian_enhanced.cuh # 4 separable variants + shared dispatch declaration (Section 6)
│   │   ├── median_enhanced.cuh # 3 variants (Shared/Network3x3/Specialized) + shared dispatch declaration (Section 7)
│   │   ├── sobel_enhanced.cuh # 4 variants (Shared/SharedConst/Specialized/Separable) + shared dispatch declaration (Section 8)
│   │   ├── laplacian_enhanced.cuh # 4 variants (Shared/SharedConst/Specialized/Explicit) + shared dispatch declaration (Section 9)
│   │   ├── threshold_enhanced.cuh # 2 variants (Vectorized/MultiPixel) + shared dispatch declaration (Section 10)
│   │   └── laplacian_threshold_fused.cuh # EXPERIMENTAL fused Laplacian+Threshold kernel (Section 10, not production-wired)
│   ├── src/
│   │   ├── cuda_common.cu  # cuda_is_available(), get_device_info()
│   │   ├── smoke_test.cu   # trivial input[i]+1 kernel (1-D, Section 1)
│   │   ├── gpu_image.cu     # boundary-safe 2-D image +1 kernel + launcher (Section 4A)
│   │   ├── gaussian_basic.cu # naive 2-D Gaussian convolution, reflect101 border (Section 4B)
│   │   ├── median_basic.cu   # naive median filter, insertion sort, replicate border (Section 4C)
│   │   ├── sobel_basic.cu    # naive Sobel (4 modes), reflect101 border, one kernel computes gx+gy (Section 4D)
│   │   ├── laplacian_basic.cu # naive Laplacian, impulse-derived coefficients, scale/delta (Section 4E)
│   │   ├── threshold_basic.cu # simplest kernel: pointwise binary threshold, no neighborhood (Section 4F)
│   │   ├── pipeline_basic.cu  # single native call: all 5 kernels via 2 reused ping-pong batch buffers (Section 5);
│   │   │                      #   gaussian_use_enhanced option added (Section 6); median_use_enhanced added (Section 7);
│   │   │                      #   sobel_use_enhanced added (Section 8); laplacian_use_enhanced added (Section 9);
│   │   │                      #   threshold_use_enhanced added (Section 10)
│   │   ├── gaussian_enhanced.cu # Naive/Shared/SharedConst/Specialized kernels + shared dispatch (Section 6)
│   │   ├── median_enhanced.cu # Shared-tile/branchless-sorting-network/Specialized kernels + shared dispatch (Section 7)
│   │   ├── sobel_enhanced.cu # Shared/SharedConst/Specialized/Separable kernels + shared dispatch (Section 8)
│   │   ├── laplacian_enhanced.cu # Shared/SharedConst/Specialized/Explicit kernels + shared dispatch (Section 9)
│   │   ├── threshold_enhanced.cu # Vectorized/MultiPixel kernels + shared dispatch (Section 10)
│   │   ├── laplacian_threshold_fused.cu # EXPERIMENTAL fused kernel, standalone, not pipeline-wired (Section 10)
│   │   └── bindings.cpp    # pybind11 module: add_ints, cuda_available, device_info, smoke_test,
│   │                        #   upload_image, download_image, image_add_one, device_memory_info,
│   │                        #   gaussian_basic_gpu, median_basic_gpu, sobel_basic_gpu,
│   │                        #   laplacian_basic_gpu, threshold_basic_gpu,
│   │                        #   gaussian_enhanced_gpu(+_batch) (Section 6), median_enhanced_gpu(+_batch) (Section 7),
│   │                        #   sobel_enhanced_gpu(+_batch) (Section 8), laplacian_enhanced_gpu(+_batch) (Section 9),
│   │                        #   threshold_enhanced_gpu(+_batch), laplacian_threshold_fused_batch_gpu (Section 10),
│   │                        #   run_basic_cuda_pipeline_gpu, GpuImage
│   ├── gaussian.py          # Python API: gaussian_cuda(), gaussian_cuda_gpu(), gaussian_kernel_2d() (Section 4B);
│   │                        #   gaussian_enhanced_cuda(), gaussian_kernel_1d(), 4 variants (Section 6)
│   ├── median.py             # Python API: median_cuda(), median_cuda_gpu() (Section 4C);
│   │                          #   median_enhanced_cuda(), median_variant_to_int(), 3 variants (Section 7)
│   ├── sobel.py               # Python API: sobel_cuda(), sobel_cuda_gpu(), mode_to_int() (Section 4D);
│   │                           #   sobel_enhanced_cuda(), sobel_variant_to_int(), 4 variants (Section 8)
│   ├── laplacian.py           # Python API: laplacian_cuda(), laplacian_cuda_gpu(), laplacian_kernel_2d() (Section 4E);
│   │                           #   laplacian_enhanced_cuda(), laplacian_variant_to_int(), 4 variants (Section 9)
│   ├── threshold.py           # Python API: threshold_cuda(), threshold_cuda_gpu() (Section 4F);
│   │                           #   threshold_enhanced_cuda(), threshold_variant_to_int(), 2 variants (Section 10)
│   ├── pipeline.py            # run_basic_cuda_pipeline(), group_by_resolution(), compute_safe_gpu_batch_size(),
│   │                          #   run_basic_cuda_pipeline_selection() (Section 5); use_enhanced_gaussian (Section 6),
│   │                          #   use_enhanced_median (Section 7), use_enhanced_sobel (Section 8),
│   │                          #   use_enhanced_laplacian (Section 9), use_enhanced_threshold (Section 10) options added;
│   │                          #   run_enhanced_cuda_pipeline(), PipelineConfig, run_cuda_pipeline() (Section 10)
│   ├── benchmark.py          # shared per-filter benchmarking (4B-4F) + benchmark_basic_cuda_pipeline() (Section 5);
│   │                          #   benchmark_cpu_basic_enhanced_pipelines() 3-way CPU/Basic/Enhanced (Section 10)
│   ├── final_benchmark.py    # reproducible benchmark framework: canonical/batch-sweep/resolution-sweep/
│   │                          #   per-filter/correctness benchmarks, manifests, loader APIs (Section 11)
│   └── CMakeLists.txt      # builds the `xray_cuda` pybind11 extension
├── pipeline/
│   ├── dataset.py           # DatasetManager, ImageSelection, DatasetStats (Section 2)
│   ├── batch.py              # iterate_batches, LoadedBatch, memory estimation (Section 2)
│   ├── image_loader.py      # load_image() — grayscale uint8 contract (Section 2)
│   └── environment.py       # get_environment_fingerprint(): OS/CPU/GPU/driver/CUDA/Python/OpenCV/git (Section 11)
├── ui/                      # Streamlit UI package (Section 12) -- rendering only, see ui/services.py
│   ├── state.py              # st.session_state helpers -- single source of truth, no module-level globals
│   ├── services.py            # THE ONLY module that calls the backend (DatasetManager/FilterConfig/
│   │                          #   cpu.pipeline/cuda.pipeline/cuda.final_benchmark/xray_cuda); ServiceError wrapper
│   ├── controls.py            # sidebar: dataset/selection/filter/implementation/compare/Optimization Lab variant controls
│   ├── images.py              # pipeline-stage grid, batch thumbnails, side-by-side + difference visualization
│   ├── performance.py         # metric cards, compute-vs-end-to-end, per-filter charts, live-vs-canonical labeling
│   ├── analytics.py           # batch-sweep/resolution-sweep charts, benchmark-history table
│   ├── correctness.py         # 3-level correctness dashboard + details panel
│   └── system.py              # GPU/CPU/environment info, GPU memory
├── benchmark/                # Placeholder for the benchmarking framework (Section 5+) -- unused; the real
│                              #   framework lives in cuda/benchmark.py and cuda/final_benchmark.py
├── benchmark_results/        # Section 11 output: raw/, aggregated/, batch_sweeps/, resolution_sweeps/,
│                              #   per_filter/, correctness/, manifests/, summary/, experiment_registry.json
├── tests/
│   ├── test_cuda_smoke.py    # 5 foundation tests (import, availability, device info, correctness, stability)
│   ├── test_dataset.py        # 31 dataset/selection/batching tests (Section 2)
│   ├── test_cpu_filters.py    # 67 per-filter correctness/validation/border/determinism tests (Section 3)
│   ├── test_cpu_pipeline.py   # 13 pipeline ordering/enable-disable/mixed-resolution tests (Section 3)
│   ├── test_cpu_benchmark.py  # 10 benchmark aggregation tests (Section 3)
│   ├── test_cuda_image.py     # 29 GPU image transfer/kernel/leak tests (Section 4A)
│   ├── test_cuda_gaussian.py  # 38 Basic CUDA Gaussian correctness/border/param tests (Section 4B)
│   ├── test_cuda_median.py    # 53 Basic CUDA median exact-match/border/param tests (Section 4C)
│   ├── test_cuda_sobel.py     # 90 Basic CUDA Sobel (4 modes) exact-match/border/pipeline-integration tests (Section 4D)
│   ├── test_cuda_laplacian.py # 77 Basic CUDA Laplacian exact-match/scale-delta/pipeline-integration tests (Section 4E)
│   ├── test_cuda_threshold.py # 82 Threshold exact-match/boundary tests + full 5-filter GPU pipeline smoke test (Section 4F)
│   ├── test_cuda_pipeline.py  # 28 production pipeline tests: single-call architecture, batching, order, memory (Section 5)
│   ├── test_gaussian_enhanced.py # 169 tests: 4 variants x correctness/border/batch/pipeline-integration (Section 6)
│   ├── test_median_enhanced.py # 122 tests: 3 variants x exact-match/border/batch/pipeline-integration (Section 7)
│   ├── test_sobel_enhanced.py # 316 tests: 4 variants x 4 modes x exact-match/border/batch/pipeline-integration (Section 8)
│   ├── test_laplacian_enhanced.py # 389 tests: 4 variants x 3 kernel sizes x scale/delta/exact-match/border/batch/pipeline-integration (Section 9)
│   ├── test_threshold_enhanced.py # 181 tests: 2 variants x width-alignment/exact-match/boundary/batch/pipeline-integration + fused-kernel tests (Section 10)
│   ├── test_final_benchmark.py # 36 tests: serialization/round-trip, reproducibility, validation, Amdahl analysis, loaders (Section 11)
│   ├── test_streamlit_app.py # 27 tests: component imports, session state, services integration, AppTest smoke tests (Section 12)
│   ├── test_streamlit_live_processing.py # 17 tests: single-image process/compare, correctness, save, real-data integration (Section 14)
│   ├── conftest.py            # shared real_dataset_path fixture/helper; stable_seed() deterministic RNG seeding (Section 10)
│   ├── fixtures/              # synthetic deterministic test images (Section 3)
│   └── reference/             # frozen .npy per-filter reference outputs (Section 3)
├── scripts/
│   ├── diagnose.py                        # one-command environment diagnostic
│   ├── inspect_dataset.py                # one-command dataset diagnostic (Section 2)
│   ├── generate_cpu_reference_outputs.py # (re)generates tests/reference/*.npy -- run manually only (Section 3)
│   ├── run_cpu_pipeline.py               # single-image CPU pipeline demo + saved outputs (Section 3)
│   ├── benchmark_cpu.py                  # batch CPU benchmark -> JSON + CSV (Section 3)
│   ├── test_cuda_image.py                # real X-ray -> GPU -> back diagnostic (Section 4A)
│   ├── test_gaussian_cuda.py             # CPU-vs-CUDA Gaussian comparison + timing + diff image (Section 4B)
│   ├── test_median_cuda.py               # CPU-vs-CUDA median comparison + timing + diff image (Section 4C)
│   ├── test_sobel_cuda.py                # CPU-vs-CUDA Sobel comparison + timing + diff image (Section 4D)
│   ├── test_laplacian_cuda.py            # CPU-vs-CUDA Laplacian comparison + timing + diff image (Section 4E)
│   ├── test_threshold_cuda.py            # CPU-vs-CUDA threshold comparison + timing + diff image (Section 4F)
│   ├── benchmark_basic_cuda.py           # CPU vs Basic CUDA production pipeline, per resolution group -> JSON+CSV (Section 5)
│   ├── benchmark_batch_sweep.py          # CPU vs Basic CUDA throughput across batch sizes -> CSV (Section 5)
│   ├── benchmark_gaussian_optimization.py # Basic vs 4 Enhanced Gaussian variants A/B + block tuning + pipeline effect (Section 6)
│   ├── benchmark_median_optimization.py # Basic vs 3 Enhanced Median variants A/B + block tuning + pipeline effect (Section 7)
│   ├── benchmark_sobel_optimization.py # Basic vs 4 Enhanced Sobel variants A/B + per-mode table + block tuning + pipeline effect (Section 8)
│   ├── benchmark_laplacian_optimization.py # Basic vs 4 Enhanced Laplacian variants A/B + block tuning + pipeline effect (Section 9)
│   ├── benchmark_threshold_optimization.py # Basic vs 2 Enhanced Threshold variants A/B + block tuning + pipeline effect (Section 10)
│   ├── benchmark_final_pipelines.py # CPU vs Basic CUDA vs Enhanced CUDA, same images, 3-way comparison + JSON/CSV (Section 10)
│   ├── run_final_benchmark.py # canonical benchmark: 4-mode timing + batch sweep + resolution sweep +
│   │                           #   per-filter Amdahl analysis + 3-level correctness -> benchmark_results/ (Section 11)
│   └── reproduce_benchmark.py # loads a benchmark_id's manifest, checks dataset fingerprint, re-runs (Section 11)
├── data/                    # (empty placeholder here — see Dataset Setup below)
├── outputs/                  # Placeholder for processed-image outputs
└── build/                    # CMake build directory (generated)
```

## Dataset Setup

1. Put the X-ray dataset in a directory. In this project it was placed at
   the workspace root as `../data` (one level above `xray_cuda_demo/`),
   with subfolders such as `train/fractured`, `train/not fractured`,
   `val/fractured`, `val/not fractured` — the scanner recurses into any
   subdirectory structure, so the exact layout doesn't matter.
2. Configure the dataset path in [config.yaml](config.yaml)
   (`dataset.path`, resolved relative to `xray_cuda_demo/` if not
   absolute).
3. Run `python scripts/inspect_dataset.py` (fast mode by default) to
   confirm discovery works; add `--mode full` to decode every image and
   get real width/height statistics (slower — see below).
4. Review the printed statistics (file counts, resolutions, a selection
   sample) before trusting the dataset for later sections.

### Supported formats

`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff` (case-insensitive).
DICOM (`.dcm`) is not supported yet — the current dataset doesn't need it —
but the loader is structured so it can be added as one more branch in
`pipeline/image_loader.py` without touching the scanner or selection logic.

### Grayscale / dtype policy

`pipeline.image_loader.load_image()` always returns a single-channel
`uint8` array with the original 0–255 pixel range, undecoded/unmodified.
No per-image normalization happens at load time — normalizing each image
independently would make the eventual CPU vs. Basic CUDA vs. Enhanced CUDA
comparison unfair, since it would change the input rather than just how
it's processed. Any normalization the pipeline needs is deferred to an
explicit, documented pipeline stage in a later section.

### Random sampling behavior

`DatasetManager.random_batch(batch_size, seed)` samples **without
replacement** using `random.Random(seed)`, then **sorts the sampled
indices ascending** before returning them. This means:

- The same `(dataset ordering, seed, batch_size)` always yields the exact
  same file list — verified by `test_seed_reproducibility`.
- The returned order reflects the dataset's own deterministic scan order,
  not sampling order, so chunked GPU batches process files in a
  predictable sequence.
- The resulting `ImageSelection` is meant to be constructed once and
  reused unchanged by the CPU, Basic CUDA, and Enhanced CUDA
  implementations in later sections — none of them should resample the
  dataset themselves.

### Batching behavior

`ImageSelection.iter_batches(n)` / `pipeline.batch.iterate_batches(items,
n)` slice an already-selected path list into consecutive chunks of at
most `n` (the last chunk may be smaller); nothing is loaded into memory by
this step. Actual pixel loading only happens when `pipeline.batch.load_batch()`
is called on one chunk at a time — the dataset manager never loads all
images into RAM at once.

### Corrupt-file handling

`DatasetManager.validate(mode="fast")` checks existence, readability, and
extension only (no decoding) — safe to run on every startup even for
~10,000 files. `mode="full"` decodes every image to catch corrupt/truncated
files and to record true width/height; a single corrupt file is reported
in `DatasetStats.invalid_samples` and does not abort validation of the
rest of the dataset.

### Actual dataset findings (this workspace)

Running `python scripts/inspect_dataset.py --mode full` against the
9,463-file dataset at `../data` found **0 corrupt/invalid files**, and
**mixed resolutions**: 9,273 images (98%) are 224×224, with ~190 images
spread across 150+ other resolutions. No resize/padding policy has been
applied — later sections that need a fixed `[N, H, W]` tensor must decide
one explicitly (`LoadedBatch.collate()` raises a clear error rather than
guessing if a requested batch mixes resolutions).

## CPU Reference Pipeline

The CPU pipeline (`cpu/filters.py`, `cpu/pipeline.py`, `cpu/benchmark.py`)
is the **authoritative baseline** that the future Basic CUDA and Enhanced
CUDA implementations must reproduce, output-for-output. It is
deliberately not performance-optimized (no multiprocessing, no OpenMP, no
GPU-backed OpenCV) — see the module docstring in `cpu/pipeline.py`.

### Stage order and enable/disable

```
Original -> Gaussian -> Median -> Sobel -> Laplacian -> Threshold
```

Fixed order; individual stages can be disabled via `FilterConfig`, in
which case the next enabled stage receives the previous *enabled*
stage's output (stages are skipped, never reordered). With every stage
disabled, `PipelineResult.final_output` is the original image, unchanged.

### Border policy

`cv2.BORDER_DEFAULT` (== `cv2.BORDER_REFLECT_101`), used consistently for
every OpenCV call that accepts a `borderType` — one exception:
`cv2.medianBlur` has no `borderType` parameter at all and always reflects
internally. Documented in `cpu/filters.py` so it's never mistaken for an
inconsistency.

### Sobel / Laplacian output contract

Both compute their gradient in `CV_32F` (never `CV_8U` directly, which
would clip mid-computation), then convert to `uint8` via
`cv2.convertScaleAbs` — i.e. `saturate_cast<uint8>(|value|)`. Consequence:
Sobel `"x"`/`"y"` modes lose gradient **sign** in the uint8 output (a
`-40` and a `+40` gradient both become `40`); `"magnitude"` and
`"abs_sum"` are already non-negative so no sign is lost there. Default
mode is `"magnitude"`.

### Threshold contract

`cv2.threshold(..., THRESH_BINARY)`: `pixel > threshold_value ->
threshold_max_value`, else `0`. Defaults: `threshold_value=128`,
`threshold_max_value=255`.

### Input/output contract

Every stage: `uint8`, single-channel, shape `(H, W)` in and out. No
silent resize, pad, normalize, or dtype coercion anywhere in this
module — a malformed input raises immediately (`validate_input_contract`
in `cpu/filters.py`), even if every stage is disabled.

### Mixed-resolution dataset handling

`run_cpu_selection()` processes each image independently at its native
resolution (a lazy generator — never forces a `[N,H,W]` stack), so the
real dataset's mixed resolutions (9,273 images at 224×224, ~190 spread
across 150+ other resolutions) do not crash it and are never resized.

### Timing

Two modes, never mixed in one result: `"processing"` (default — every
image preloaded once, only `run_cpu_pipeline()` calls are timed) and
`"end_to_end"` (each measurement run reloads from disk too). Per-filter
and total timings use `time.perf_counter()`.

### Commands

```bash
# Single image, saves original/gaussian/median/sobel/laplacian/threshold/final + prints timing
python scripts/run_cpu_pipeline.py --image "../data/train/fractured/10-rotated1-rotated1-rotated1.jpg" --save-dir outputs/cpu_demo

# Batch benchmark over a random selection -> JSON (full detail) + CSV (summary row) under outputs/
python scripts/benchmark_cpu.py --batch-size 100 --seed 42 --runs 5

# Regenerate the frozen tests/reference/*.npy baselines (only when CPU filter behavior intentionally changes)
python scripts/generate_cpu_reference_outputs.py --force

pytest -q
```

## CUDA Image Infrastructure (Section 4A)

Connects the Section 1 CUDA foundation to real image data, without yet
implementing any of the five actual filters.

```
NumPy uint8 [H,W]  --upload_image()-->  GpuImage (device memory)
                                              |
                                       image_add_one() [trivial boundary-safe kernel]
                                              |
GpuImage  --download_image()-->  NumPy uint8 [H,W]
```

### Python API (`xray_cuda` extension)

```python
gpu_image = xray_cuda.upload_image(image)   # np.uint8 [H,W] -> GpuImage
result = xray_cuda.image_add_one(gpu_image) # {'output': GpuImage, 'kernel_ms': float}
output = xray_cuda.download_image(result["output"])  # GpuImage -> np.uint8 [H,W]
info = xray_cuda.device_memory_info()        # {'free_bytes': int, 'total_bytes': int}
```

`GpuImage` exposes `.width`, `.height`, `.shape`, `.dtype` (`"uint8"`),
`.nbytes` as read-only properties; it is never constructed directly from
Python, only returned by `upload_image`/`image_add_one`. Its underlying
CUDA memory is released automatically (RAII, `cuda/include/gpu_image.cuh`)
when the Python object is garbage-collected — no manual `free()` call.

### Input contract

Only 2-D `uint8` `[H, W]` arrays are accepted. RGB/RGBA arrays, `[N,H,W]`
batches, and non-`uint8` dtypes are rejected with `ValueError` before any
GPU work happens (`upload_image` validates ndim, dtype, and non-empty
shape explicitly rather than relying on implicit NumPy casting).

### Layout / stride

Simple contiguous row-major storage (stride == width bytes, no pitch).
Deliberately the simplest representation for this section; if profiling
later shows `cudaMallocPitch`-aligned rows would help wide-image
coalescing, that can be added without changing the upload/download API.

### Launch configuration

Fixed `16x16` thread blocks (`default_block_dim()` in `gpu_image.cuh`),
grid size computed as `ceil(width/16) x ceil(height/16)`. Every thread
bounds-checks `x<width && y<height`, so dimensions that don't divide the
block size evenly (e.g. `31x47`, `17x256`) never touch out-of-bounds
memory — exercised directly by
`test_add_one_handles_non_block_multiple_dimensions`.

### Synchronization policy

`upload -> kernel -> cudaDeviceSynchronize() -> download`, always. No
asynchronous overlap or multi-stream execution is implemented or claimed
in this section — `CudaStream` exists in `gpu_image.cuh` as an
abstraction for later sections to build on, but every Section 4A
operation uses the default stream.

### Error handling

- Invalid shape/dtype -> Python `ValueError` (via `std::invalid_argument`,
  translated automatically by pybind11).
- CUDA allocation failure -> Python `GpuMemoryError` (a `MemoryError`
  subclass, registered via `py::register_exception`), with a message
  naming the requested size and free/total device memory.
- Any other CUDA runtime error -> Python `RuntimeError` with file/line
  context (`CUDA_CHECK`, unchanged from Section 1).

Note: on this Windows/WDDM machine, CUDA allocations can transparently
spill into shared system memory well beyond the 8 GB card's physical
VRAM (observed: 200 MB x 60 held allocations = ~12 GB succeeded without
error), so `GpuMemoryError` is verified by code path/registration rather
than by a live out-of-memory reproduction — forcing a real one would
require exceeding this machine's ~15 GB total RAM, which isn't safe to
do in this environment.

### uint8 arithmetic

The trivial kernel computes `output[i] = input[i] + 1` with standard
uint8 wraparound (`255 + 1 == 0`) — not clamped. Documented in
`gpu_image.cuh`/`gpu_image.cu` since Section 1's smoke-test kernel used
`float` and never had to make this call.

### Commands

```bash
python scripts/test_cuda_image.py                          # uses one real X-ray (seed 42)
python scripts/test_cuda_image.py --image "../data/val/fractured/1.jpg"
pytest tests/test_cuda_image.py -v
```

## Basic CUDA Gaussian Blur (Section 4B)

The first actual image-processing CUDA kernel — intentionally naive (no
shared memory, no separability, no constant memory), establishing the
baseline that a later Enhanced CUDA implementation will improve on.

```
NumPy uint8 [H,W]
      |
   H2D (upload_image)
      |
Basic Gaussian CUDA kernel  <- [k,k] float32 coefficients (cv2.getGaussianKernel outer product)
      |
   D2H (download_image)
      |
NumPy uint8 [H,W]
```

### CPU reference mirrored exactly

`cpu/filters.py::apply_gaussian` is `cv2.GaussianBlur(image, (k,k),
sigmaX=sigma, borderType=cv2.BORDER_DEFAULT)`. This kernel reproduces
that, not a reinvented Gaussian:

- **Coefficients**: generated in Python via `cv2.getGaussianKernel(k,
  sigma)` (`cuda/gaussian.py::gaussian_kernel_2d`), *not* recomputed in
  CUDA. This matters because OpenCV uses a **hardcoded coefficient
  table** for `k` in `{3,5,7}` when `sigma<=0` — verified by inspection
  (`cv2.getGaussianKernel(5, 0.0)` → `[0.0625, 0.25, 0.375, 0.25,
  0.0625]`, not what the sigma-formula alone would give) — and only
  falls back to the `0.3*((k-1)*0.5-1)+0.8` formula otherwise (e.g.
  `k=9`, or explicit `sigma>0`). Calling OpenCV's own function
  sidesteps having to replicate that split correctly in C++.
- **Border**: `BORDER_REFLECT_101` (== `BORDER_DEFAULT`), implemented as
  an explicit `reflect101()` device function (`cuda/src/gaussian_basic.cu`)
  handling arbitrary offsets via modulo, not just clamped to one radius.
- **Output conversion**: float32 accumulation, single round via
  `__float2int_rn` (round-half-to-even, matching OpenCV's `cvRound`
  more closely than `roundf`'s round-half-away-from-zero), clamped to
  `[0,255]`, cast to `uint8`.

### Algorithm

One CUDA thread computes one output pixel: reads the full `k×k`
neighborhood directly from global memory (no shared memory), multiplies
by the matching coefficient, sums in `float32`, converts once. `16×16`
thread blocks (reused from Section 4A's `default_block_dim()`), every
thread bounds-checked (`x<width && y<height`) so non-block-multiple
dimensions (`31×47`, `17×256`, ...) never read/write out of bounds.

### Measured CPU-vs-GPU numerical difference — justified tolerance: `max_abs_diff <= 1`

Empirically measured, not assumed, before writing tests: on a real
224×224 X-ray, `max_abs_diff` is always ≤1 across `kernel_size ∈
{3,5,7,9}` and `sigma ∈ {0,1,2}`, and **exactly 0** for `kernel_size=9`
in every case tried. The nonzero cases affect a small minority of pixels
(≈2% at `k=3`, dropping to ≈0.01% at `k=7`). This is a rounding-mode
artifact — OpenCV's actual `GaussianBlur` is a two-pass separable filter
with its own internal intermediate representation, not a single-pass
float 2D convolution — not a border, coefficient, or algorithm bug;
confirmed by `k=9` (which bypasses OpenCV's hardcoded table) being
bit-exact.

### Python API

```python
from cuda.gaussian import gaussian_cuda, gaussian_cuda_gpu

output = gaussian_cuda(image, kernel_size=5, sigma=1.0)      # np.uint8[H,W] -> np.uint8[H,W]
result = gaussian_cuda_gpu(gpu_image, kernel_size=5, sigma=1.0)  # GpuImage -> {'output': GpuImage, ...} (no host round trip)
```

Low-level primitive (`xray_cuda.gaussian_basic_gpu(gpu_image, coeffs) ->
{'output': GpuImage, 'kernel_ms': float, 'coeff_upload_ms': float}`)
takes precomputed coefficients directly — `gaussian_cuda`/`gaussian_cuda_gpu`
are thin Python wrappers that compute those coefficients and validate
`kernel_size`/`sigma` via the same `FilterConfig` the CPU reference uses,
so an invalid parameter fails identically on both implementations.

### Benchmark methodology

Two separate CUDA-events-for-kernel / `time.perf_counter()`-for-host
timing utilities in `cuda/benchmark.py`:

- `benchmark_gaussian_single_image` — one **cold** sample (first GPU
  call; only meaningful if genuinely first in a fresh process) plus
  **warm** mean/median/min/max/std over `measurement_runs` (after
  `warmup_runs` discarded), for H2D, kernel, D2H, and end-to-end,
  separately — never blended into one number.
- `benchmark_gaussian_selection` — loops the *single-image* GPU call
  over many different real images; explicitly labeled `"single-image
  GPU invocation baseline (not batch-optimized)"` since no batching/
  streaming optimization exists yet.

**Real measurements from this machine** (RTX 4060 Laptop GPU), kernel
size 5, sigma 1.0:

| | CPU (mean) | GPU warm end-to-end (mean) | GPU kernel only (mean) |
|---|---|---|---|
| 224×224 | 0.075 ms | 0.581 ms | 0.035 ms |
| 512×512 | 0.061 ms | 0.580 ms | 0.074 ms |
| 1024×1024 | 0.453 ms | 1.772 ms | 0.232 ms |
| large real X-ray (2232×3282) | 2.256 ms | 7.106 ms | 1.464 ms |
| **Process-cold** GPU end-to-end (224×224) | — | **97.2 ms** | — |
| 100-image batch, single-invocation loop | 9.5 ms total (10,530 img/s) | 130.0 ms total (769 img/s) | — |

**CPU is faster end-to-end at every size tested here** — H2D+D2H
transfer overhead dominates a single-image invocation, and `cv2`'s CPU
Gaussian is already fast. This is the expected, accepted outcome for a
*basic, unbatched* baseline (per the Section 4B spec) — it is the number
a later batched/streamed Enhanced CUDA implementation should improve on,
not a claim that GPU acceleration doesn't help here.

### Commands

```bash
python scripts/test_gaussian_cuda.py --kernel-size 5 --sigma 1.0
pytest tests/test_cuda_gaussian.py -v
```

## Basic CUDA Median Filter (Section 4C)

The second Basic CUDA filter. Median filtering is **nonlinear** (the
output is a selected input pixel, not a weighted sum), so it cannot be
made separable the way Gaussian can — this kernel collects the full
neighborhood and sorts it, per pixel.

### CPU reference mirrored exactly

`cpu/filters.py::apply_median` is `cv2.medianBlur(image, k)`.

- **Border**: `cv2.medianBlur` exposes no `borderType` parameter, so its
  actual rule was **verified empirically, not assumed** (per the Section
  4C spec's explicit warning not to reuse Gaussian's border mapping
  blindly). A 5×5 all-distinct-values test image run through
  `cv2.medianBlur(k=3)` was compared against a hand-computed
  `BORDER_REPLICATE` (clamp-to-edge) result and a hand-computed
  `BORDER_REFLECT_101` result:

  ```
  input row 0:              [10, 20, 30, 40, 50]
  cv2.medianBlur output:    [20, 30, 40, 50, 50]
  replicate hypothesis:     [20, 30, 40, 50, 50]  <- exact match
  reflect101 hypothesis:    [60, 60, 70, 80, 90]  <- does not match
  ```

  Confirmed: **`BORDER_REPLICATE`**, genuinely different from
  Gaussian/Sobel/Laplacian's `BORDER_DEFAULT` (reflect-101). (This also
  corrected an inaccurate comment left in `cpu/filters.py` from Section 3,
  which had said medianBlur "always reflects" — fixed to say replicate,
  now that it's actually been checked.)
- **Output**: pure integer selection, no floating-point accumulation
  anywhere in the kernel.
- **Supported kernel sizes**: `{3, 5, 7}`, mirrored exactly from
  `FilterConfig.ALLOWED_MEDIAN_KERNELS`.

### Algorithm

One thread per output pixel: gathers the `k×k` neighborhood (clamped at
borders) into a small per-thread local array (max 7×7=49 `uint8`
values), sorts it with a plain insertion sort, takes the middle element.
No shared memory, no sorting network — intentionally naive. `16×16`
thread blocks (same `default_block_dim()` as Gaussian), bounds-checked
per thread.

### Correctness methodology: exact equality, not a tolerance

Median has no floating-point accumulation, so CPU and GPU outputs are
required to match **exactly** (`max_abs_diff == 0`) — not the ±1
tolerance Gaussian needed. Verified across all 6 synthetic fixtures ×
all 3 kernel sizes, real 224×224 and large real X-rays, 5 border
scenarios × 2 kernel sizes, and 4 non-block-multiple dimensions
(including `7×7`, smaller than the largest supported kernel) — **0
pixel differences in every case**.

### Python API

```python
from cuda.median import median_cuda, median_cuda_gpu

output = median_cuda(image, kernel_size=3)                 # np.uint8[H,W] -> np.uint8[H,W]
result = median_cuda_gpu(gpu_image, kernel_size=3)          # GpuImage -> {'output': GpuImage, 'kernel_ms': float}
```

`xray_cuda.median_basic_gpu(gpu_image, kernel_size)` is the low-level
primitive; the Python wrappers validate `kernel_size` via the same
`FilterConfig` the CPU reference uses.

### Benchmark methodology

Reuses the same shared `cuda/benchmark.py` machinery as Gaussian
(refactored in this section specifically to avoid duplicating the
cold/warm/CPU timing loop a second time) — CUDA events for kernel time,
`time.perf_counter()` for H2D/D2H/end-to-end/CPU, never blended.

**Real measurements from this machine** (RTX 4060 Laptop GPU), kernel
size 3:

| | CPU (mean) | GPU warm end-to-end (mean) | GPU kernel only (mean) |
|---|---|---|---|
| 224×224 | 0.018 ms | 0.527 ms | 0.035 ms |
| large real X-ray (2232×3282) | 2.457 ms | 6.938 ms | **2.253 ms** |
| **Process-cold** GPU end-to-end (224×224) | — | **98.7 ms** | — |
| 100-image batch, single-invocation loop | 3.0 ms total (33,035 img/s) | 21.3 ms total (4,688 img/s) | — |

Same pattern as Gaussian: **CPU is faster end-to-end** at every size
tested — H2D+D2H overhead dominates a single-image invocation. One
genuinely interesting (unforced) data point: at the large real X-ray
size, GPU **kernel-only** time (2.25 ms) is already roughly on par with
total CPU time (2.46 ms) — median's higher per-pixel compute cost (sort
up to 49 values vs. Gaussian's 9-81 multiply-adds) means the GPU's
parallelism starts to pay for itself in raw compute sooner than it did
for Gaussian, even though transfer overhead still dominates the
end-to-end total today. This is exactly the kind of gap a later batched/
streamed Enhanced CUDA implementation is meant to close.

### Commands

```bash
python scripts/test_median_cuda.py --kernel-size 3
pytest tests/test_cuda_median.py -v
```

## Basic CUDA Sobel Edge Detection (Section 4D)

The third Basic CUDA filter, and the first with multiple output modes.

### CPU reference mirrored exactly

`cpu/filters.py::apply_sobel` computes `gx`/`gy` via `cv2.Sobel(image,
CV_32F, ..., ksize=3, borderType=cv2.BORDER_DEFAULT)`, combines them per
`sobel_mode`, then applies `cv2.convertScaleAbs`. Verified empirically
before writing any CUDA code (per the spec's explicit "do not assume"
warning):

- **Coefficients**: `cv2.getDerivKernels(1,0,3)` / `(0,1,3)` confirmed
  the classic `Gx=[[-1,0,1],[-2,0,2],[-1,0,1]]` and
  `Gy=[[-1,-2,-1],[0,0,0],[1,2,1]]` matrices.
- **Border**: a hand-computed reflect-101 3×3 convolution matched
  `cv2.Sobel`'s actual `gx`/`gy` output **bit-for-bit** on a real X-ray
  (`max_abs_diff == 0.0`) — same `BORDER_DEFAULT`/reflect-101 rule as
  Gaussian (the shared `reflect101()` helper, now factored into
  `gpu_image.cuh` so it isn't duplicated a second time).
- **Output conversion**: `round(|raw|)` clamped to `[0,255]` matched
  `apply_sobel`'s actual output **bit-for-bit**, for all four modes, in
  both float64 and float32 precision.
- **Only `kernel_size=3`** is supported (the `FilterConfig` default and
  the only size the spec frames this Basic implementation around —
  `cv2.Sobel`'s other sizes use non-trivial separable kernels via
  `getDerivKernels`, out of scope here; requesting anything else from
  the CPU-side `FilterConfig` comparison still works, it's only the
  *Basic CUDA kernel itself* that's fixed at 3).

### Four modes

`"x"`, `"y"` (signed gradient, **sign lost** in the uint8 output via
`convertScaleAbs`'s `|.|` — this is the CPU reference's own documented
behavior, not something this kernel introduces), `"magnitude"`
(`sqrt(gx²+gy²)`, the default), `"abs_sum"` (`|gx|+|gy|`).

### Algorithm

One thread per output pixel reads the 3×3 neighborhood **once** and
computes both `gx` and `gy` from it in the same kernel invocation
(avoiding a second neighborhood read for what would otherwise be a
second kernel launch) — still the naive baseline: no shared memory, no
separable implementation. `16×16` thread blocks, bounds-checked. Mode is
a plain `int` (`SobelMode` enum in C++), not a string comparison inside
the kernel.

### Correctness methodology: exact equality, verified achievable

Like median, Sobel involves no ambiguous floating-point accumulation
that survives to the output (the final `round(|raw|)` is deterministic),
so `max_abs_diff == 0` was targeted and achieved — verified across 6
fixtures × 4 modes, real 224×224 and large real X-rays × 4 modes, 5
border scenarios × 4 modes, and dimensions including `7×7` and
`31×47`. A dedicated test also verifies the *mathematical* sign
relationship the CPU reference relies on (`sobel_cuda(image, "x") ==
sobel_cuda(255-image, "x")`, since Sobel is linear and a constant offset
contributes no gradient), not just final-value equality.

### Pipeline integration (not the full 5-stage pipeline yet)

`gaussian_cuda_gpu -> median_cuda_gpu -> sobel_cuda_gpu` was chained
directly on `GpuImage` objects (no host round trip) as a small manual
integration test, proving Sobel can consume a `GpuImage` produced by an
earlier Basic CUDA stage. This chain's output is compared against the
equivalent CPU chain with an **empirically-derived tolerance of 8**
(observed max was 4) — not because Sobel itself is inexact (it isn't:
`max_abs_diff==0` in every dedicated Sobel test), but because Gaussian's
own documented ±1 tolerance compounds through Median and gets amplified
by Sobel's gradient coefficients. This is expected and explained in the
test, not papered over.

### Python API

```python
from cuda.sobel import sobel_cuda, sobel_cuda_gpu

output = sobel_cuda(image, mode="magnitude")           # np.uint8[H,W] -> np.uint8[H,W]
result = sobel_cuda_gpu(gpu_image, mode="x")            # GpuImage -> {'output': GpuImage, 'kernel_ms': float}
```

### Benchmark methodology

Same shared `cuda/benchmark.py` machinery as Gaussian/Median, extended
with Sobel wrappers — all four modes benchmarked separately, never
assumed identical in cost.

**Real measurements from this machine** (RTX 4060 Laptop GPU):

| 224×224 | CPU (mean) | GPU warm end-to-end (mean) | GPU kernel only (mean) |
|---|---|---|---|
| x | 0.051 ms | 0.478 ms | 0.016 ms |
| y | 0.053 ms | 0.406 ms | 0.017 ms |
| magnitude | 0.094 ms | 0.507 ms | 0.027 ms |
| abs_sum | 0.093 ms | 0.441 ms | 0.018 ms |
| **Process-cold** GPU end-to-end | — | **99.2 ms** | — |

| large real X-ray (2232×3282) | CPU (mean) | GPU warm end-to-end (mean) |
|---|---|---|
| x | 15.97 ms | 4.77 ms |
| y | 13.42 ms | 4.59 ms |
| magnitude | 24.18 ms | 4.12 ms |
| abs_sum | 42.77 ms | **4.96 ms** |

| 100-image batch, single-invocation loop (magnitude) | total | throughput |
|---|---|---|
| CPU | 22.5 ms | 4,445 img/s |
| GPU | 18.7 ms | 5,358 img/s |

**This is the first Basic (unbatched) filter where GPU wins outright** —
at the large real X-ray size, GPU end-to-end beats CPU on every mode,
up to **~8.6× for `abs_sum`** (CPU's multiple full-array passes —
two `cv2.Sobel` calls plus an `absdiff`/`add` combination — cost more on
a ~7.3-million-pixel image than the GPU's single fused kernel pass, even
after paying full H2D+D2H transfer). The 100-image batch also edges out
CPU, unlike Gaussian/Median at the same scale. This is measured, not
assumed — Sobel's higher per-pixel compute cost (two derivatives + mode
combination vs. Gaussian's single weighted sum or Median's single sort)
is exactly the kind of workload where GPU parallelism pays for itself
sooner, even in an unoptimized single-image-invocation baseline.

### Commands

```bash
python scripts/test_sobel_cuda.py --mode magnitude
pytest tests/test_cuda_sobel.py -v
```

## Basic CUDA Laplacian Filter (Section 4E)

The fourth Basic CUDA filter — and the one where "verify, don't assume"
mattered most: the naive assumption about its coefficients was wrong.

### CPU reference mirrored exactly

`cpu/filters.py::apply_laplacian` is `cv2.Laplacian(image, CV_32F,
ksize, scale, delta, borderType=cv2.BORDER_DEFAULT)` then
`cv2.convertScaleAbs`.

- **Coefficients are NOT the textbook `[[0,1,0],[1,-4,1],[0,1,0]]`
  kernel for every `kernel_size`.** Verified empirically, and it's a
  good thing this was checked: `kernel_size=1` does produce that
  classic kernel, but **`kernel_size=3` actually produces
  `[[2,0,2],[0,-8,0],[2,0,2]]`** — `cv2.Laplacian(ksize=3)` is
  internally `Sobel(dx=2,ksize=3) + Sobel(dy=2,ksize=3)`, not a scaled
  version of the simple discrete Laplacian. `kernel_size=5` produces a
  still-different 5×5 kernel. A first hand-derivation attempt (combining
  `cv2.getDerivKernels`' separable x/y outputs via outer products) got
  the `kernel_size=1` case wrong (mismatched shapes, 1×3 vs 3×1, need
  careful zero-padding to combine). The robust fix: recover the exact
  effective kernel via **impulse response** — apply `cv2.Laplacian` to a
  single 1.0 pixel on a zero background and read off the result. This
  is correct by construction for any `kernel_size`, sidestepping the
  padding logic entirely (`cuda/laplacian.py::laplacian_kernel_2d`).
- **Border**: `BORDER_REFLECT_101`/`BORDER_DEFAULT` — verified
  empirically, not assumed, via the shared `reflect101()` helper (same
  one Gaussian and Sobel use).
- **Scale/delta**: applied as `scaled = raw*scale + delta` *after* the
  convolution sum (not folded into the coefficients beforehand) —
  verified bit-exact against `apply_laplacian` for `scale∈{0.5,1.0,2.0}`
  and `delta∈{0,5,10}`.
- **Output conversion**: `clamp(round(|scaled|), 0, 255)`, same
  `convertScaleAbs`-equivalent pattern as Sobel.
- **Supported kernel sizes**: `{1, 3, 5}`, mirrored exactly from
  `FilterConfig.ALLOWED_LAPLACIAN_KERNELS`.

### Algorithm

Same coefficient-passing design as Gaussian (kernel size varies, so
coefficients are computed in Python and uploaded to a small device
buffer) combined with Sobel's abs+round+clamp output conversion: one
thread per output pixel, full non-separable 2-D convolution read
directly from global memory, `float32` accumulation, `16×16` thread
blocks, bounds-checked. No shared memory, no separable filtering.

### Correctness methodology: exact equality, verified achievable

`max_abs_diff == 0` was targeted (per the spec, unlike Gaussian) and
achieved — verified across 6 fixtures × 3 kernel sizes, 4 scale/delta
combinations, real 224×224 and large real X-rays × 3 kernel sizes, 6
border/interior scenarios × 3 kernel sizes, and dimensions including
`7×7` and `31×47`.

### Pipeline integration

`gaussian_cuda_gpu -> median_cuda_gpu -> sobel_cuda_gpu ->
laplacian_cuda_gpu` chained entirely on `GpuImage` objects, no CPU round
trip, no Threshold yet. Cross-checked against the equivalent CPU chain
with a documented tolerance of 40 (Laplacian itself is exact — this
bound only accounts for Gaussian's own ±1 tolerance compounding through
three more stages, same reasoning as Section 4D's Sobel chain test).

### Python API

```python
from cuda.laplacian import laplacian_cuda, laplacian_cuda_gpu

output = laplacian_cuda(image, kernel_size=3, scale=1.0, delta=0.0)
result = laplacian_cuda_gpu(gpu_image, kernel_size=1)   # GpuImage -> {'output': GpuImage, 'kernel_ms': float, 'coeff_upload_ms': float}
```

### Benchmark methodology

Same shared `cuda/benchmark.py` machinery, extended with Laplacian
wrappers.

**Real measurements from this machine** (RTX 4060 Laptop GPU), kernel
size 3:

| | CPU (mean) | GPU warm end-to-end (mean) | GPU kernel only (mean) |
|---|---|---|---|
| 224×224 | 0.031 ms | 0.564 ms | 0.030 ms |
| large real X-ray (2232×3282) | **30.44 ms** | **5.75 ms** | 0.625 ms |
| **Process-cold** GPU end-to-end (224×224) | — | **89.9 ms** | — |
| 100-image batch, single-invocation loop | 13.4 ms total (7,480 img/s) | 30.8 ms total (3,242 img/s) | — |

Same small-image pattern as Gaussian/Median/100-image-batch (CPU wins —
transfer overhead dominates a single small invocation), but at the
**large real X-ray size, GPU end-to-end is ~5.3× faster than CPU**
(kernel-only is ~49× faster) — `cv2.Laplacian`'s internal two-Sobel-call
computation gets expensive on a ~7.3-million-pixel image on CPU, the
same effect observed for Sobel's more expensive modes in Section 4D,
now even more pronounced. Measured, not assumed.

### Commands

```bash
python scripts/test_laplacian_cuda.py --kernel-size 3 --scale 1.0 --delta 0.0
pytest tests/test_cuda_laplacian.py -v
```

## Basic CUDA Binary Threshold (Section 4F)

The fifth and final Basic CUDA filter — and the simplest kernel in this
project: a pure pointwise comparison, no neighborhood, no border
handling, no floating-point arithmetic at all.

### CPU reference mirrored exactly

`cpu/filters.py::apply_threshold` is `cv2.threshold(image,
threshold_value, max_value, cv2.THRESH_BINARY)`. Verified empirically
before writing the kernel (not assumed): the comparison is **strict
`>`** — a pixel exactly equal to `threshold_value` maps to `0`, not
`max_value`. Confirmed with `threshold-1`/`threshold`/`threshold+1`
triples, and two edge cases: `threshold_value=0` (pixel `0` still maps
to `0`, since `0 > 0` is false) and `threshold_value=255` (every pixel
maps to `0`, since nothing can exceed `255` — a legitimate result, not
an error).

### Algorithm

`output[i] = input[i] > threshold_value ? max_value : 0`. One thread,
one comparison, one write — `16×16` thread blocks, bounds-checked, no
optimization of any kind.

### Correctness methodology: exact equality

`max_abs_diff == 0` in every test — 6 fixtures × 6 threshold values, 4
`max_value` variants, 8 explicit equality-boundary cases, real 224×224
and large real X-rays, and non-block-multiple dimensions.

### Python API

```python
from cuda.threshold import threshold_cuda, threshold_cuda_gpu

output = threshold_cuda(image, threshold_value=128, max_value=255)
result = threshold_cuda_gpu(gpu_image, threshold_value=128, max_value=255)  # {'output': GpuImage, 'kernel_ms': float}
```

### Benchmark methodology

Same shared `cuda/benchmark.py` machinery. **Real measurements** (RTX
4060 Laptop GPU): 224×224 — CPU `0.004 ms`, GPU warm end-to-end `0.478
ms`, GPU kernel only `0.009 ms`; large real X-ray (2232×3282) — CPU
`1.99 ms`, GPU warm end-to-end `4.12 ms`. As the spec anticipated,
threshold is such a low-compute operation that transfer/launch overhead
dominates almost entirely — CPU wins even at the large-image size here
(unlike Sobel/Laplacian), because there's essentially no per-pixel
compute for the GPU's parallelism to amortize against.

### Commands

```bash
python scripts/test_threshold_cuda.py --threshold 128 --max-value 255
pytest tests/test_cuda_threshold.py -v
```

## Basic CUDA Full Pipeline (smoke test only, Section 4F)

With all five Basic CUDA filters implemented, `tests/test_cuda_threshold.py`
includes the first end-to-end assembly: Gaussian → Median → Sobel →
Laplacian → Threshold, chained entirely on `GpuImage` objects with
**zero CPU round trips between stages**. This is explicitly a smoke
test — no buffer reuse beyond each stage's own output, no streams, no
batching. The production pipeline (Section 5) builds on this.

### Why the final-output comparison uses "% of differing pixels", not max_abs_diff

Measured before writing the test (not assumed): comparing the CPU
five-filter chain against the GPU chain on real X-rays, **`max_abs_diff`
on the final thresholded output is 255 in nearly every case.** This is
expected, not a bug — Threshold is a step function. Gaussian's own
documented ±1 tolerance compounds through Median → Sobel → Laplacian
(measured stage-by-stage on one image: Gaussian max_diff=1 (71 px),
Median max_diff=1 (58 px), Sobel max_diff=4 (336 px), Laplacian
max_diff=32 (889 px)), until a small number of pixels land on the
opposite side of the threshold boundary and flip from `0` to `255` or
back. A `max_abs_diff`-based tolerance is meaningless here (255 asserts
nothing). The right metric is the *fraction* of pixels that flip:
measured across 30 real images, worst case was **0.038%**, mean 0.015%
— so the test asserts **≤1%**, comfortably above the observed worst case
while still tight enough to catch a real bug. A separate diagnostic test
(`test_full_basic_cuda_pipeline_intermediate_stage_validation`) downloads
every intermediate stage to localize *where* a divergence grows, should
this ever fail — that diagnostic path is not the production path.

### Full-pipeline timing (224×224, warm, CUDA events per kernel)

| Stage | Time |
|---|---|
| H2D | 0.064 ms |
| Gaussian kernel | 0.041 ms |
| Median kernel | 0.030 ms |
| Sobel kernel | 0.022 ms |
| Laplacian kernel | 0.049 ms |
| Threshold kernel | 0.025 ms |
| D2H | 0.094 ms |
| Sum of the above | 0.325 ms |
| **Measured total (5 separate Python/pybind11 calls)** | **1.035 ms** |

The measured total is **notably larger than the sum of its parts** —
the difference (~0.71 ms) is per-call Python/pybind11 overhead across
five separate `xray_cuda.*_basic_gpu()` calls (each allocates its own
output `GpuImage`, launches independently, syncs independently). This
is exactly the overhead a production GPU-resident pipeline (Section 5)
should reduce — measured here honestly rather than glossed over.

## Production Basic CUDA Pipeline + Batching + CPU-vs-GPU Baseline (Section 5)

This is the official **Basic GPU baseline** the future Enhanced CUDA
work is measured against. Three architectural changes turn the five
individually-validated filters (Sections 4B-4F) into a production
pipeline — no filter *algorithm* changes.

### 1. Five Python calls → one native call

Section 4F measured the cost of the old architecture directly: ~0.325ms
of actual CUDA work vs. ~1.035ms of wall time for one 224×224 image
through all five stages — **~0.71ms of Python/pybind11 call overhead**
across five separate calls. `xray_cuda.run_basic_cuda_pipeline_gpu()`
(bound from `cuda/src/pipeline_basic.cu`) runs all five (optionally
disabled) stages inside **one** C++ function call: upload once, launch
up to five kernels back to back, download once. `cuda/pipeline.py`'s
`run_basic_cuda_pipeline()` is the only Python entry point production
code (benchmarks, future Streamlit, future Enhanced comparisons) should
use; the individual `cuda/gaussian.py` etc. modules remain available for
per-filter testing and diagnostics only.

### 2. Reused ping-pong buffers, not per-stage allocation

`GpuImageBatch` (new in `gpu_image.cuh`, RAII, same philosophy as
`GpuImage`) holds `[N, H, W]` contiguous device memory for a whole
batch. The native pipeline allocates **exactly two** of these per call
and ping-pongs between them across every enabled stage:

```
H2D → A
A → Gaussian  → B   (swap)
B → Median    → A   (swap)
A → Sobel     → B   (swap)
B → Laplacian → A   (swap)
A → Threshold → B   (swap)
B → D2H
```

No `cudaMalloc`/`cudaFree` per filter, per image, or per batch element.

### 3. Batch-aware kernels (backward compatible)

Each of the five existing kernels gained one line: a `blockIdx.z`-indexed
plane offset into the input/output pointers. `grid.z` is the batch
dimension; every pre-Section-5 single-image call site is unaffected
because `dim3`'s `z` defaults to `1`, so `blockIdx.z` is always `0` there
— confirmed by re-running the full Section 4B-4F test suite (336 tests)
unchanged after this edit, before writing any Section 5-specific code.

### Mixed-resolution handling: group, don't resize

The real dataset is still mixed-resolution (9,273/9,463 images at
224×224 — Section 2 finding). Images are **never** resized or padded to
force a uniform batch. `cuda/pipeline.py::group_by_resolution()` loads
each selected image (unavoidable — shape isn't known without decoding)
and groups items by `(height, width)`, preserving order within each
group; `run_basic_cuda_pipeline()` requires all images in one call to
share a resolution and raises `ValueError` otherwise.

### Safe GPU batch sizing — actual free VRAM, not WDDM overcommit

`compute_safe_gpu_batch_size(height, width, safety_factor=0.7,
num_buffers=2)` sizes chunks against **currently reported free VRAM**,
not the WDDM overcommit behavior Section 4A observed (200MB×60
allocations succeeding on an 8GB card by silently spilling into system
RAM) — that behavior is real on this machine but isn't planned around,
since it isn't guaranteed and doesn't exist on non-Windows/non-WDDM
systems. `run_basic_cuda_pipeline_selection()` is the top-level
orchestrator: group by resolution → chunk each group to
`min(requested, safe_capacity)` → run the native pipeline once per
chunk → reassemble results in the **original selection order**
(keyed by each item's unique dataset index, not object identity),
regardless of how many resolution groups or GPU chunks that required.
A requested batch larger than GPU capacity is chunked, never rejected.

### Python API

```python
from cuda.pipeline import run_basic_cuda_pipeline, run_basic_cuda_pipeline_selection

output_batch, timing = run_basic_cuda_pipeline(images, config)   # images: same-resolution list -> [N,H,W] + PipelineTiming
results = run_basic_cuda_pipeline_selection(selection, config, max_batch_size=128)  # handles grouping/chunking/reassembly
```

### Correctness (full pipeline): same finding as Section 4F, re-verified at batch scale

`max_abs_diff` on the thresholded output is still 255 in nearly every
batch (Threshold is a step function amplifying Gaussian's own ±1
tolerance) — expected, not a bug. The meaningful metric remains **% of
differing pixels**, asserted ≤1% (empirically justified in Section 4F:
observed worst case 0.038%); actual measured values across real batches
in this section ranged **0.0019%-0.0178%**, comfortably inside that
bound.

### Real measurements — CPU vs. Basic CUDA production pipeline (RTX 4060 Laptop GPU)

**Per-resolution-group** (128-image selection, seed 42, 2 warmup + 5 measurement runs):

| Resolution | N | CPU total | GPU total | CPU img/s | GPU img/s | Speedup | Differing px |
|---|---:|---:|---:|---:|---:|---:|---:|
| 224×224 | 125 | 90.2 ms | 62.7 ms | 1,386 | 1,993 | **1.44×** | 0.0178% |
| 1733×858 | 1 | 22.8 ms | 9.2 ms | 44 | 109 | **2.49×** | 0.0035% |
| 902×1128 | 1 | 8.7 ms | 4.0 ms | 115 | 248 | **2.16×** | 0.0060% |
| 858×958 | 1 | 8.3 ms | 4.3 ms | 121 | 233 | **1.93×** | 0.0019% |

**This is the first time in the project GPU wins outright at the
dominant 224×224/small-batch scale** — the old single-image, five-call
baseline (Sections 4B-4F) consistently lost to CPU there; removing the
call overhead and batching more images per native call was enough to
flip that result, exactly the architectural goal of this section.

**Batch-size sweep at 224×224** (`benchmark_batch_sweep.py`, 593 real
images available, seed 42):

| Batch size | CPU img/s | GPU img/s | Speedup |
|---:|---:|---:|---:|
| 1 | 1,096 | 1,141 | 1.04× |
| 8 | 1,426 | 1,971 | 1.38× |
| 16 | 1,390 | 1,876 | 1.35× |
| 32 | 1,460 | 2,107 | 1.44× |
| 64 | 1,367 | 2,004 | 1.47× |
| 128 | 1,363 | 2,014 | 1.48× |
| 256 | 1,351 | 2,067 | **1.53×** (peak) |
| 512 | 1,320 | 1,927 | 1.46× |

Speedup grows from ~1.04× (batch=1, transfer overhead barely amortized)
to a peak ~1.53× around batch=256, then dips slightly at 512 — reported
as measured, with no smoothing or cherry-picking of the peak value as
"the" result.

### Measured bottleneck table (224×224, batch=125, mean kernel time)

| Stage | Basic GPU time | Likely bottleneck |
|---|---:|---|
| H2D | 0.563 ms | transfer |
| Gaussian | 1.240 ms | global memory + arithmetic (25-tap 5×5 convolution/pixel — most expensive linear filter) |
| Median | 1.410 ms | memory + sorting (9-tap gather + insertion sort/pixel — most expensive stage overall) |
| Sobel | 0.388 ms | neighborhood memory (only a 3×3 read) |
| Laplacian | 0.519 ms | neighborhood memory (3×3, more nonzero coefficients than Sobel) |
| Threshold | 0.080 ms | memory throughput only (no neighborhood at all) |
| D2H | 0.807 ms | transfer |

No optimizations are prescribed here — purely measured, for the
Enhanced CUDA stage (Section 6+) to target. Median and Gaussian
dominate compute time, consistent with their per-pixel algorithmic cost
(sorting vs. a 25-tap weighted sum); this is exactly the kind of
finding Enhanced CUDA (shared-memory tiling, separable Gaussian, etc.)
is meant to address.

### Commands

```bash
python scripts/benchmark_basic_cuda.py --dataset ../data --batch-size 128 --seed 42 --runs 5
python scripts/benchmark_batch_sweep.py --resolution 224x224 --sizes 1,8,16,32,64,128,256,512
pytest tests/test_cuda_pipeline.py -v
```

## Enhanced CUDA Gaussian Optimization (Section 6)

The first Enhanced CUDA implementation — Basic Gaussian
(`gaussian_basic`) is **unchanged**; this adds a second, separately
benchmarkable family (`gaussian_enhanced`) so `Basic` and `Enhanced`
remain comparable at any time.

### Four variants, one optimization at a time

| Variant | Adds | Isolates the gain from |
|---|---|---|
| Naive | Separable (1D horizontal + 1D vertical pass) instead of Basic's O(k²) 2D convolution | Separability alone |
| Shared | + shared-memory tiling (1D halo per pass, not a 2D tile — unnecessary once separable) | Reduced redundant global memory traffic |
| SharedConst | + coefficients moved from a global pointer to `__constant__` memory | Constant memory's broadcast/cache behavior |
| Specialized | + `kernel_size` as a compile-time template parameter, `#pragma unroll` | Eliminating runtime loop bounds/indexing |

### Correctness: two separate, both measured, tolerances

- **Enhanced vs CPU**: same ±1 tolerance Basic Gaussian already has
  (Section 4B) — verified identical differing-pixel counts (1089/71/5/0
  for k=3/5/7/9) at the common `sigma=0.0` case.
- **Enhanced vs Basic**: at `sigma=0.0`, **bit-exact** (`max_abs_diff ==
  0`) across every variant and kernel size — a direct result of keeping
  the horizontal pass's output in `float32` (not rounding to `uint8`
  until the final vertical-pass write), so there's still only one round,
  at the end, same as Basic's single-pass 2D convolution. At **explicit
  `sigma>0`** (e.g. `sigma=2.0`), a genuine, diagnosed (not assumed)
  divergence appears: `max_abs_diff=1` on exactly 1 pixel out of 4096,
  **identically across all four variants**. This is the signature of
  IEEE-754 float addition's non-associativity — Basic sums k² terms in
  one direct reduction, Enhanced sums the mathematically-equivalent
  separable decomposition as two smaller sums, and a value landing
  almost exactly on a `.5` rounding boundary can tip either way
  depending on summation order. Not a bug (confirmed by all four
  structurally different kernels hitting the same single pixel); the
  test tolerance for Enhanced-vs-Basic is set to 1, measured and
  justified, not assumed — exactly what Section 6 spec item 9 warned
  "mathematically equivalent" does not guarantee.

### The central methodological finding: single-image benchmarking was misleading

Isolated single-image (batch=1) kernel timing at 224×224 showed **Basic
beating every Enhanced variant** (0.44×–0.68× "speedup") — a result
that would normally mean rejecting the whole optimization family. Before
accepting that, it was diagnosed rather than assumed:

1. A 5-run comparison first showed `SharedConst` at 0.455× (2.2× slower
   than Basic) — investigated with 20 runs instead of 5, which flipped
   the result to `SharedConst` being *faster* than `Shared`. **At these
   timescales (tens of microseconds), 5 measurement runs is not enough**
   — the project's usual `2 warmup + 5 measurement` convention was
   insufficient here and this script uses 20+ instead, documented as a
   deliberate deviation.
2. Even with 20 runs, single-image results stayed inconsistent across
   kernel sizes (0.44×–1.32×, no reliable winner) — because a separable
   implementation issues **two** kernel launches per image where Basic
   issues **one**, and at 224×224 the actual arithmetic work (tens of
   microseconds) is smaller than the fixed per-launch overhead, so the
   second launch's overhead isn't amortized.
3. Batching (the Section 5 architecture — `grid.z` = batch dimension,
   one native call covers the whole batch) was the fix: with a fair,
   apples-to-apples comparison (Basic Gaussian *also* batched via the
   existing production pipeline, Gaussian-only, vs. Enhanced batched),
   **every variant wins cleanly and monotonically, even at 224×224**:

| Variant (batched, 125×224×224) | Kernel (median) | Speedup vs Basic |
|---|---:|---:|
| Basic | 1.258 ms | 1.00× |
| Naive | 0.578 ms | 2.18× |
| Shared | 0.449 ms | 2.80× |
| SharedConst | 0.424 ms | 2.97× |
| Specialized | 0.377 ms | **3.34×** |

Confirmed at large-image scale too (2232×3282 real X-ray, single image):
2.10× → 2.49× → 2.82× → **3.12×**, the same monotonic pattern. The
lesson: **for this project, always benchmark the batched/production
call, not an isolated single-image loop** — the latter can produce the
opposite conclusion at small resolutions purely from fixed overhead.

### Block/tile tuning (Optimization 5) — rejected switching away from the default

Swept `{8×8, 16×16, 32×8, 32×16}` for the Specialized variant at both
224×224 and large-image scale. `8×8` and `16×16` were statistically tied
(within a few percent — noise level) at both scales; `32×16` was
consistently ~5-8% worse. **Decision: keep `16×16`**, the block size
already used by every other kernel in this project — switching to `8×8`
for a negligible, within-noise difference would add inconsistency for
no measured benefit (spec's own rejection criterion: "adds complexity
with negligible benefit").

### Optimization 6 (vectorized loads) — rejected without a full implementation

Reasoned investigation, not a built-and-discarded prototype: vectorizing
the shared-memory tile load (e.g. `uchar4`) would need a scalar
fallback for non-4-aligned widths, and the real dataset includes many
non-224×224 images with widths that aren't 4-aligned (e.g. 1733, an odd
number). Shared memory already coalesces the tile load reasonably well
across threads in a block, so the realistic additional gain is modest.
Given Specialized already delivers a measured 3.34× (batched, 224×224)
to 3.12× (large image) speedup from the other three optimizations, the
added alignment/remainder-handling complexity wasn't judged worth it at
this stage — documented as a reasoned rejection per the spec's explicit
allowance ("Do NOT force vectorization if it complicates the
implementation or produces no measurable benefit").

### Pipeline integration: Enhanced Gaussian + Basic Median/Sobel/Laplacian/Threshold

`cuda/pipeline.py::run_basic_cuda_pipeline(..., use_enhanced_gaussian=True)`
swaps only the Gaussian stage in the single native pipeline call
(Section 5's `BasicPipelineConfig` gained `gaussian_use_enhanced` and
related fields, all defaulted to preserve Section 5's exact prior
behavior — verified backward compatible by re-running the full Section
5 test suite unchanged before adding anything new). Measured on 125 real
224×224 images:

| | Basic pipeline | Enhanced-Gaussian pipeline |
|---|---:|---:|
| Total (median) | 5.346 ms | 4.465 ms |
| **Full-pipeline speedup** | | **1.197×** |

Correctness: at the default `sigma=0.0`, the Enhanced-Gaussian pipeline
is **bit-exact** vs. the Basic pipeline (`max_abs_diff=0`,
`0.0000%` differing), and matches CPU with the same `0.0178%`
differing-pixel rate the Basic pipeline already had (Section 5) — no
threshold-sensitivity regression from optimizing Gaussian, confirmed
directly rather than assumed.

### Python API

```python
from cuda.pipeline import run_basic_cuda_pipeline

output, timing = run_basic_cuda_pipeline(images, config, use_enhanced_gaussian=True, gaussian_variant="specialized")
```

```python
from cuda.gaussian import gaussian_enhanced_cuda, gaussian_enhanced_cuda_gpu

output = gaussian_enhanced_cuda(image, kernel_size=5, sigma=1.0, variant="specialized")  # single image, testing/diagnostics
```

### Commands

```bash
python scripts/benchmark_gaussian_optimization.py --batch-size 128 --kernel-size 5 --sigma 0.0 --runs 20
pytest tests/test_gaussian_enhanced.py -v
```

## Enhanced CUDA Median Optimization (Section 7)

Basic Median (`median_basic`) is **unchanged**; this adds a second,
separately benchmarkable family (`median_enhanced`) so `Basic` and
`Enhanced` remain comparable at any time. `clamp_index()` (the
`BORDER_REPLICATE` coordinate mapping Section 4C verified matches
`cv2.medianBlur`) was moved from a private copy inside
`median_basic.cu` into the shared `gpu_image.cuh`, so both
implementations use exactly one definition of the border rule.

### Why Median needs a different strategy than Gaussian

Gaussian (Section 6) is a **linear, separable** filter — its dominant
cost is redundant global-memory reads, which shared-memory tiling and
separability directly attack. Median is **nonlinear and
non-separable**: there's no way to decompose "find the middle of k²
sorted values" into two 1-D passes, and the actual per-pixel cost is
dominated by the **data-dependent branching** inside the insertion
sort (`median_basic`'s selection loop), not by memory traffic. This
predicts — and the measurements below confirm — that shared-memory
tiling alone gives Median almost nothing, while removing branches
(a sorting network) gives it the large win.

### Three variants, testing that prediction directly

| Variant | Adds | Isolates the gain from |
|---|---|---|
| Shared | 2-D shared-memory tile with halo, runtime `kernel_size`, same insertion sort as Basic | Reduced redundant global memory traffic (the Gaussian-style lever) |
| Network3x3 | Branchless sorting network for the fixed 3×3/9-element case (Nicolas Devillard's public-domain `opt_med9`, 19 `min`/`max` compare-exchanges, no data-dependent branches at all) | Eliminating branching entirely, for the one kernel size where a compact known network exists |
| Specialized | Compile-time `kernel_size` template (3/5/7), `#pragma unroll`-ed gather + outer sort loop (inner pass still data-dependent) | Eliminating runtime loop bounds/indexing, without a full sorting network |

### Correctness: exact match, no tolerance

Unlike Gaussian, Median has no floating-point accumulation — every
comparison and swap is done on integer pixel values, so there is no
summation-order non-associativity to justify a tolerance. All three
variants match **both** CPU (`cv2.medianBlur`) and Basic CUDA
**exactly** (`max_abs_diff == 0`) across every fixture, kernel size the
variant supports, border region, batch size, and a large real X-ray
image (2232×3282) — verified directly, not assumed, since the 3×3
sorting network is genuinely new, hand-transcribed code with real risk
of a transcription error.

### Measured results (batched, 125×224×224, real X-ray)

| Variant | Kernel (median) | Speedup vs Basic | Correctness |
|---|---:|---:|---|
| Basic | 1.582 ms | 1.00× | PASS (reference) |
| Shared | 1.578 ms | 1.00× | PASS |
| **Network3x3** | **0.269 ms** | **5.87×** | PASS |
| Specialized | 1.554 ms | 1.02× | PASS |

Confirms the prediction: Shared and Specialized (which still branch
inside the insertion sort) give essentially nothing; only removing the
branches entirely (Network3x3) produces a real speedup. On a large real
X-ray image (2232×3282, single image, k=3): **7.01×**. Batch-size sweep
{1, 8, 16, 32, 64, 128, 256, 512} at k=3, 224×224 shows the speedup
growing with batch size as fixed launch overhead amortizes (4.14× at
batch=1 up to 6.40× at batch≈256), consistent with the Section 6 lesson
that launch overhead dominates at small batch sizes — all measurements
here used the batched production architecture from the start.

### k=5 / k=7: why a true sorting network wasn't built (measured, not assumed)

The spec explicitly allowed skipping a true sorting network for 5×5
(25-element) and 7×7 (49-element) if the complexity/benefit tradeoff
didn't justify it, and warned not to assume 7×7 needs the same strategy
as 3×3. Measured instead of assumed:

| kernel_size | Shared speedup | Specialized speedup |
|---|---:|---:|
| k=5 | 1.006× | 1.008× |
| k=7 | 1.003× | 1.029× |

Both essentially flat — confirming the Network3x3 result wasn't a
fluke of small K, and that for k=5/k=7 the real 25-/49-input optimal
sorting networks (which exist in the literature but are far more
complex — dozens to ~130+ compare-exchanges, much higher register
pressure) would be the only way to get a comparable win. Given no
downstream stage in this pipeline currently uses kernel_size 5 or 7 by
default, and the marginal gain measured here is within noise,
building those larger networks was **not pursued** — a reasoned,
measured rejection, not corner-cutting.

### Register pressure (measured via `nvcc -Xptxas -v`, sm_89)

| Kernel | Registers | Stack frame | Spill stores/loads |
|---|---:|---:|---:|
| Network9 (3×3 sorting network) | 36 | 0 B | 0 / 0 |
| Specialized k=3 | 36 | 16 B | 0 / 0 |
| Shared (runtime k) | 40 | 56 B | 0 / 0 |
| Specialized k=5 | 39 | 32 B | 0 / 0 |
| Specialized k=7 | 55 | 56 B | 0 / 0 |

No kernel spills to local memory at any kernel size — register
pressure rises with k (as expected, since Specialized unrolls the
gather into that many live values) but never crosses into spilling on
this GPU. k=7 Specialized's 55 registers for a measured 1.03× gain is
further evidence that unrolling alone, without removing branches, isn't
where Median's cost lives.

### Block/tile tuning — rejected switching away from the default

Swept `{8×8, 16×16, 32×8, 32×16}` for the winning Network3x3 variant at
224×224: `16×16` (0.268 ms) and `32×8` (0.272 ms) were statistically
tied; `8×8` (0.296 ms) and `32×16` (0.273 ms) were each a few percent
worse. **Decision: keep `16×16`**, the block size already used by every
other kernel in this project — same reasoning and outcome as Section
6's block-tuning experiment.

### Rejected optimizations, summarized

| Optimization | Result | Decision |
|---|---|---|
| Shared-memory tiling alone | 1.00–1.01× across k=3/5/7 | Rejected as the primary lever (Median isn't memory-bound); kept as the `Shared` variant for completeness/comparison |
| Specialization (unroll) alone | 1.02–1.03× across k=3/5/7 | Rejected as the primary lever (still branches); kept as the `Specialized` variant |
| True sorting networks for k=5 (25-element) / k=7 (49-element) | Not built | Rejected: Shared/Specialized data at k=5/k=7 show no branching-removal opportunity is being left on the table proportionally to the 3×3 case's complexity cost, and no downstream stage needs k=5/k=7 by default |
| Block/tile tuning away from 16×16 | 8×8/32×8/32×16 each ≤ tied or worse | Rejected — no measured benefit |
| Register/local reuse (Optimization 3) | No spilling observed at any variant/size (measured via `-Xptxas -v`) | No action needed — nothing to reclaim |

### Pipeline integration: Enhanced Median + Basic Gaussian/Sobel/Laplacian/Threshold

`cuda/pipeline.py::run_basic_cuda_pipeline(..., use_enhanced_median=True)`
swaps only the Median stage in the single native pipeline call
(`BasicPipelineConfig` gained `median_use_enhanced` and related fields,
mirroring Section 6's `gaussian_use_enhanced` pattern, all defaulted to
preserve prior behavior). `use_enhanced_gaussian` and
`use_enhanced_median` are independent — either, both, or neither can be
set, and the underlying native call always takes both enhancement
argument groups (each individually gated by its own `_use_enhanced`
flag) rather than branching per combination in Python. Measured on 122
real 224×224 images (batch=125 sampled, 122 shared that resolution):

| | Basic pipeline | Enhanced-Median pipeline | Enhanced-Both pipeline |
|---|---:|---:|---:|
| Total (median) | 7.857 ms | 5.257 ms | 3.618 ms |
| Median stage (median) | 1.428 ms | 0.332 ms | 0.332 ms |
| **Full-pipeline speedup** | | **1.494×** | **2.171×** |

Correctness: the Enhanced-Median pipeline is **bit-exact** vs. the
Basic pipeline (`max_abs_diff=0`, `0.0000%` differing), and both it and
the combined Enhanced-Gaussian+Enhanced-Median pipeline match CPU with
the same `0.0178%` differing-pixel rate the Basic pipeline already had
(Section 5/6) — no threshold-sensitivity amplification from optimizing
Median, confirmed directly rather than assumed (the spec's specific
concern: Threshold can amplify small upstream differences, but here
there are no upstream differences to amplify).

### Python API

```python
from cuda.pipeline import run_basic_cuda_pipeline

output, timing = run_basic_cuda_pipeline(
    images, config,
    use_enhanced_gaussian=True, gaussian_variant="specialized",
    use_enhanced_median=True, median_variant="network3x3",
)
```

```python
from cuda.median import median_enhanced_cuda, median_enhanced_cuda_gpu

output = median_enhanced_cuda(image, kernel_size=3, variant="network3x3")  # single image, testing/diagnostics
```

### Commands

```bash
python scripts/benchmark_median_optimization.py --batch-size 128 --kernel-size 3 --runs 20
pytest tests/test_median_enhanced.py -v
```

## Enhanced CUDA Sobel Optimization (Section 8)

Basic Sobel (`sobel_basic`) is **unchanged**; this adds a second,
separately benchmarkable family (`sobel_enhanced`) so `Basic` and
`Enhanced` remain comparable at any time.

### Why Sobel's optimization headroom is smaller than Gaussian's or Median's

Basic Sobel (Section 4D) was already designed to compute gx AND gy from
a single 3×3 neighborhood read inside **one** kernel launch — unlike
Basic Gaussian's unexploited-but-separable O(k²) 2D convolution
(Section 6's big win) or Basic Median's branchy insertion sort (Section
7's big win), there is no "do less arithmetic" lever sitting unused in
Basic Sobel. The only remaining levers are **reuse across threads**
(shared-memory tiling, the same lever that helped Median) and
**mode-specific work** (X only needs gx, Y only needs gy — Basic always
computes both regardless of `mode`). This predicts a real but modest
gain, not a multiplier — confirmed, not assumed, below.

### Four variants, testing that prediction directly

| Variant | Adds | Isolates the gain from |
|---|---|---|
| Shared | 2-D shared-memory tile with a 1-pixel halo, runtime `mode`, identical hand-optimized arithmetic to Basic | Reduced redundant global memory traffic |
| SharedConst | Shared's tiling + coefficients in `__constant__` 3×3 arrays via a generic unrolled loop | Constant memory's broadcast behavior, and whether a generic loop can match Basic's hand-optimized (zero-skipping) arithmetic |
| Specialized | Shared's tiling + `mode` as a compile-time template: X/Y skip the unused derivative's 2 neighbor reads and all of its arithmetic | Mode-specific work reduction |
| Separable | A genuine two-kernel decomposition (horizontal pass → global float intermediates → vertical pass + combine), no shared memory | Whether Section 6's "multi-launch overhead can dominate at 224×224" lesson also applies here, against a baseline that's *already* single-launch |

### Correctness: exact match, no tolerance

Every term in Gx/Gy is an exact-integer float32 value (pixel
differences and one ×2 multiply — never a fractional weight like
Gaussian's sigma-derived coefficients), so no reordering of these sums
— not Separable's different grouping, not SharedConst's added
zero-coefficient center-pixel term — can change the result: exact-
integer float32 addition needs no rounding here (magnitudes stay far
below 2^24). Predicted before measuring, then verified: all four
variants are **bit-exact** vs. both CPU and Basic CUDA
(`max_abs_diff == 0`) for all four modes, across every fixture, kernel
size, border region, batch size, and a large real X-ray image
(2232×3282) — including Separable, despite its different arithmetic
grouping.

### Measured results (batched, 122×224×224, real X-ray, mode=magnitude)

| Variant | Kernel (median) | Speedup vs Basic | Correctness |
|---|---:|---:|---|
| Basic | 0.421 ms | 1.00× | PASS (reference) |
| Shared | 0.406 ms | 1.04× | PASS |
| SharedConst | 0.455 ms | 0.92× | PASS |
| **Specialized** | **0.395 ms** | **1.07×** | PASS |
| Separable | 0.569 ms | 0.74× | PASS |

A high-run-count re-check (60 measurement runs instead of 20, following
the Section 6 lesson that these timescales need more samples) confirmed
Specialized's edge is real but modest and noisy: median speedup settled
between 1.03× and 1.07× across repeated measurement sessions, block
sizes, and a large real X-ray image (2232×3282: 1.03×) — never dramatic,
matching the "already single-launch, already minimal arithmetic"
prediction above. Batch-size sweep {1, 8, 16, 32, 64, 128, 256, 512}
shows the effect is only reliably visible at batch≥128 — at smaller
batches Specialized was sometimes measurably *slower* than Basic
(e.g. 0.64-0.94× at batch∈{8,16,32}), consistent with Sobel's kernel
being so cheap that per-launch/measurement noise exceeds the real
effect size at small batch×resolution products. Per-mode, all four
modes showed a consistent small win (X: 1.10×, Y: 1.10×, Magnitude:
1.05×, AbsSum: 1.08×), confirming the mode-specific-work-skipping
prediction isn't an artifact of one particular mode.

### Register pressure (measured via `nvcc -Xptxas -v`, sm_89)

| Kernel | Registers | Stack frame | Spill stores/loads |
|---|---:|---:|---:|
| Shared | 39 | 0 B | 0 / 0 |
| SharedConst | 39 | 0 B | 0 / 0 |
| Specialized (all 4 modes) | 39 | 0 B | 0 / 0 |
| Separable horizontal pass | 20 | 0 B | 0 / 0 |
| Separable vertical pass | 18 | 0 B | 0 / 0 |

No spilling anywhere. Notably, **Specialized uses the identical 39
registers as Shared/SharedConst for every one of its 4 mode
instantiations** — mode specialization's win comes from fewer
instructions/shared-memory reads, not from freed-up registers or
improved occupancy; register pressure was never the bottleneck for this
kernel, measured rather than assumed (Optimization 3, "register/local
reuse," has nothing to reclaim here, same finding as Section 7's
Median).

### Rejected optimizations, summarized

| Optimization | Result | Decision |
|---|---|---|
| SharedConst (constant-memory coefficients, generic 3×3 loop) | 0.92× (a regression) | Rejected: Basic's hand-optimized arithmetic already skips the zero-coefficient center pixel and uses one multiply total; a generic 9-term loop does strictly more work despite constant memory's broadcast behavior |
| Separable (two-kernel decomposition) | 0.74× (a regression) | Rejected: confirms the Section 6 lesson (multi-launch overhead can dominate at 224×224) applies with extra force here, since Basic Sobel — unlike Basic Gaussian — was *already* a single launch; splitting it into two can only add overhead, never remove redundant work Basic wasn't already avoiding |
| Block/tile tuning away from 16×16 | 8×8 worse (~15-20%); 32×8/32×16 statistically tied with 16×16 | Kept 16×16 — no measured benefit from switching |
| Register/local reuse (Optimization 3) | 39 registers, 0 spilling at every variant (measured via `-Xptxas -v`) | No action needed — nothing to reclaim |
| Vectorized loads (`uchar4`) | Not built | Rejected by reasoning, same as Section 6's identical rejection: the real dataset has widths like 1733 that aren't 4-aligned, requiring a scalar fallback; given Specialized's already-modest ~1.03-1.07× ceiling, the added complexity wasn't judged worth prototyping |

### Full-pipeline effect: real at the kernel level, below the noise floor alone

Measured on 122 real 224×224 images, isolating Sobel only (Gaussian/
Median/Laplacian/Threshold stay Basic): Basic pipeline total = 5.631 ms
(Sobel stage: 0.378 ms) vs. Enhanced-Sobel pipeline total = 5.667 ms
(Sobel stage: 0.365 ms) — the Sobel stage itself is reliably faster
(1.04×, consistent with the kernel-only numbers above), but the
**whole-pipeline** effect (0.994×) is statistically indistinguishable
from 1.0×. This is expected, not a bug: Sobel is only ~7% of total
pipeline time, so even a genuine ~4% kernel-level win translates to a
~0.3% pipeline-level effect — smaller than the run-to-run variance in
the other four stages. Documented honestly rather than reported as a
false pipeline-level win. Combined with Sections 6-7 (Enhanced Gaussian
+ Enhanced Median + Enhanced Sobel, all three stages at once, same 122
images): **1.56× full-pipeline speedup** — driven almost entirely by
Gaussian and Median's much larger contributions, with Sobel adding a
real but small increment on top.

Correctness: the Enhanced-Sobel pipeline (alone or combined with
Enhanced Gaussian/Median) is **bit-exact** vs. the Basic pipeline
(`max_abs_diff=0`, `0.0000%` differing), and matches CPU with the same
`0.0178%` differing-pixel rate the Basic pipeline has had since Section
5 — no threshold-sensitivity amplification from optimizing Sobel,
confirmed directly rather than assumed.

### Python API

```python
from cuda.pipeline import run_basic_cuda_pipeline

output, timing = run_basic_cuda_pipeline(
    images, config,
    use_enhanced_gaussian=True, use_enhanced_median=True,
    use_enhanced_sobel=True, sobel_variant="specialized",
)
```

```python
from cuda.sobel import sobel_enhanced_cuda, sobel_enhanced_cuda_gpu

output = sobel_enhanced_cuda(image, mode="magnitude", variant="specialized")  # single image, testing/diagnostics
```

### Commands

```bash
python scripts/benchmark_sobel_optimization.py --batch-size 125 --mode magnitude --runs 20
pytest tests/test_sobel_enhanced.py -v
```

## Enhanced CUDA Laplacian Optimization (Section 9)

Basic Laplacian (`laplacian_basic`) is **unchanged**; this adds a second,
separately benchmarkable family (`laplacian_enhanced`) so `Basic` and
`Enhanced` remain comparable at any time. This is the first Enhanced
filter where **four of five** pipeline stages are now Enhanced.

### Why Laplacian has more headroom than Sobel (Section 8)

Unlike Basic Sobel (already hand-optimized: hardcoded coefficients, one
launch, zero-skipping arithmetic — Section 8 found only ~1.03-1.07×
headroom left), Basic Laplacian is a genuinely generic, non-separable,
runtime-sized 2-D convolution: a double for-loop reading BOTH the
neighborhood AND the coefficients from global memory, with no
zero-skipping. This is architecturally the same shape as Basic
Gaussian's unexploited O(k²) convolution (Section 6) — so real gains
from shared memory, constant memory, and specialization were plausible
here, and were confirmed, not assumed, by measuring first per the spec's
explicit instruction.

### The three actual coefficient sets (not what `kernel_size` suggests)

`cpu.filters.ALLOWED_LAPLACIAN_KERNELS = {1, 3, 5}` is OpenCV's
aperture-size *parameter*, not a literal matrix dimension. Recovered via
impulse response (Section 4E, `cuda/laplacian.py::laplacian_kernel_2d`):

| kernel_size | Actual matrix | Coefficients | Nonzero |
|---|---|---|---|
| 1 | 3×3 | `[[0,1,0],[1,-4,1],[0,1,0]]` (the textbook Laplacian — a special case) | 5/9 |
| 3 | 3×3 | `[[2,0,2],[0,-8,0],[2,0,2]]` (cv2.Laplacian(ksize=3) is internally two Sobel(dx=2)/Sobel(dy=2) passes summed, NOT the textbook kernel) | 5/9 |
| 5 | 5×5 | `[[2,4,4,4,2],[4,0,-8,0,4],[4,-8,-24,-8,4],[4,0,-8,0,4],[2,4,4,4,2]]` | 21/25 |

The C++ layer only ever sees the coefficient array's actual shape (3×3
or 5×5) — never Python's kernel_size=1-vs-3 distinction. Per spec item
11, this coefficient recovery is never re-derived in CUDA: the Enhanced
`Explicit` variant (below) hardcodes arithmetic for these three exact,
already-verified value sets and validates the caller's array against
them (exact float comparison — every value is a small exact integer)
before using that path, throwing rather than silently misapplying the
wrong hardcoded formula.

### Four variants, testing the prediction directly

| Variant | Adds | Isolates the gain from |
|---|---|---|
| Shared | 2-D shared-memory tile with a halo sized to the runtime radius, same generic double-loop as Basic | Reduced redundant global memory traffic |
| SharedConst | Shared's tiling + coefficients in a fixed-capacity `__constant__` array, still a runtime-sized loop | Constant memory's broadcast behavior alone (Section 8 warned a *naive* generic constant-memory loop can regress — measured here on its own before combining with anything else) |
| Specialized | SharedConst + the matrix's actual size (3 or 5) as a compile-time template, `#pragma unroll`-ed | Eliminating runtime loop bounds/indexing on top of constant memory — still reads every coefficient, including zeros |
| Explicit | Hand-written arithmetic for each of the three known, verified coefficient sets, skipping every zero-coefficient term entirely | Exploiting the specific known sparsity pattern — the lever Basic Sobel already used, that Section 8's generic SharedConst loop could not recover on its own |

### Correctness: exact match, no tolerance

Every Laplacian coefficient is a small exact integer (2, 4, -8, -24) —
no fractional weights — so reordering sums (SharedConst/Specialized's
constant-memory loop order, Explicit's hand-grouped sums) cannot change
the result. Verified across all 3 kernel sizes × 3 scale values
{0.5, 1.0, 2.0} × 3 delta values {0, 5, 10} × 4 variants (108
combinations) on a real image: **every one bit-exact** vs. both CPU and
Basic CUDA (`max_abs_diff == 0`) on the first run — matching the
project's established Laplacian correctness standard (Section 4E), not
loosened for Enhanced.

### Measured results (batched, 122×224×224, real X-ray)

| kernel_size | Basic | Shared | SharedConst | **Specialized** | Explicit |
|---|---:|---:|---:|---:|---:|
| 1 | 0.533 ms | 1.25× | 1.28× | **1.52×** | 1.48× |
| 3 | 0.531 ms | 1.24× | 1.27× | **1.51×** | 1.47× |
| 5 | 1.226 ms | 1.78× | 2.02× | **2.89×** | 2.80× |

Specialized consistently wins, with Explicit a close second — a genuine,
interesting finding worth stating plainly: hand-written zero-skipping
arithmetic (Explicit) did **not** beat the compiler-unrolled generic
constant-memory loop (Specialized) at either kernel size, despite
Explicit doing measurably less arithmetic (skipping 4/9 or 4/25 terms).
Register-pressure data below shows both use identical register counts,
so the gap isn't register pressure — most likely `nvcc`'s unroller
schedules the loop's more uniform accumulation pattern better than the
hand-grouped sums' data-dependency chains. Documented as an honest
result, not adjusted to fit an expectation. On a large real X-ray
(2232×3282, k=3): Specialized 1.53×, Explicit 1.50× — consistent with
the batched numbers. Batch-size sweep {1,8,16,32,64,128,256,512} at k=3
shows Specialized's kernel-level speedup holding steady at 1.46×-1.77×
across every batch size tested, including batch=1 — unlike Sobel
(Section 8), this gain is NOT dependent on large-batch amortization.

### Register pressure (measured via `nvcc -Xptxas -v`, sm_89)

| Kernel | Registers | Stack frame | Spill stores/loads |
|---|---:|---:|---:|
| Shared | 40 | 0 B | 0 / 0 |
| SharedConst | 40 | 0 B | 0 / 0 |
| Specialized (K=3, K=5) | 39 | 0 B | 0 / 0 |
| Explicit (all 3 patterns) | 39 | 0 B | 0 / 0 |

No spilling anywhere. Specialized and Explicit use **identical** register
counts (39) at every kernel size/pattern — confirming register pressure
is not what separates them (see above); the gap is in instruction
scheduling, not resource pressure.

### Block/tile tuning — rejected switching away from the default

Swept `{8×8, 16×16, 32×8, 32×16}` for Specialized at k=3, 224×224:
`16×16` (0.349 ms) and `32×8` (0.350 ms) were statistically tied;
`32×16` (0.344 ms) was marginally best but within noise of 16×16;
`8×8` (0.428 ms) was clearly worse. **Decision: keep `16×16`** — same
reasoning and outcome as every prior section's block-tuning experiment.

### Rejected/non-adopted experiments, summarized

| Optimization | Result | Decision |
|---|---|---|
| Shared-memory tiling alone | 1.24-1.78× | Real gain, but not the winner; kept as `Shared` variant for comparison |
| Constant memory alone (still generic loop) | 1.27-2.02× | Real gain (unlike Section 8's Sobel SharedConst, which *regressed* — Laplacian's Basic has real headroom Sobel's didn't); kept as `SharedConst` |
| Explicit hand-written arithmetic | 1.47-2.80× | A strong, real speedup — but consistently a hair behind Specialized despite skipping more arithmetic; kept as the `Explicit` variant (useful for the block/tile/register comparison and as a worked example of the technique) but NOT the recommended default |
| Block/tile tuning away from 16×16 | 8×8 worse; 32×8/32×16 tied with 16×16 | Kept 16×16 — no measured benefit |
| Register/local reuse | 39-40 registers, 0 spilling everywhere (measured via `-Xptxas -v`) | Nothing to reclaim |
| Vectorized loads (`uchar4`) | Not built | Rejected by reasoning, same as Sections 6 and 8: real dataset widths like 1733 aren't 4-aligned, requiring a scalar fallback; Specialized's already-strong 1.5-2.9× ceiling didn't justify the added complexity |

### Full-pipeline integration: Enhanced Gaussian + Median + Sobel + Laplacian + Basic Threshold

`cuda/pipeline.py::run_basic_cuda_pipeline(..., use_enhanced_laplacian=True)`
swaps only the Laplacian stage (mirrors Sections 6-8's pattern exactly);
all four Enhanced flags are independent and can be combined freely.
Measured on 122 real 224×224 images:

| | Basic pipeline | Enhanced-Laplacian only | Enhanced-All (4 stages) |
|---|---:|---:|---:|
| Total (median) | 5.489 ms | 5.299 ms | 3.206 ms |
| Laplacian stage (median) | 0.518 ms | 0.336 ms | — |
| **Full-pipeline speedup** | | **1.036×** | **1.712×** |

Laplacian-stage-only kernel speedup in this run: 1.543× (consistent with
the dedicated benchmark above). Correctness: the Enhanced-Laplacian
pipeline (alone or combined with the other three) is **bit-exact** vs.
the Basic pipeline (`max_abs_diff=0`, `0.0000%` differing), and matches
CPU with the same `0.0178%` differing-pixel rate the Basic pipeline has
had since Section 5 — no threshold-sensitivity amplification from
optimizing Laplacian, confirmed directly.

### Python API

```python
from cuda.pipeline import run_basic_cuda_pipeline

output, timing = run_basic_cuda_pipeline(
    images, config,
    use_enhanced_gaussian=True, use_enhanced_median=True,
    use_enhanced_sobel=True, use_enhanced_laplacian=True, laplacian_variant="specialized",
)
```

```python
from cuda.laplacian import laplacian_enhanced_cuda, laplacian_enhanced_cuda_gpu

output = laplacian_enhanced_cuda(image, kernel_size=3, variant="specialized")  # single image, testing/diagnostics
```

### Commands

```bash
python scripts/benchmark_laplacian_optimization.py --batch-size 125 --kernel-size 3 --scale 1.0 --delta 0 --runs 20
pytest tests/test_laplacian_enhanced.py -v
```

## Enhanced CUDA Threshold + Final Five-Filter Pipelines (Section 10)

Basic Threshold (`threshold_basic`) is **unchanged**; this adds a second,
separately benchmarkable family (`threshold_enhanced`) so `Basic` and
`Enhanced` remain comparable at any time. This is the final filter --
all five stages are now Enhanced, and this section adds the two
canonical production pipelines (`run_basic_cuda_pipeline` /
`run_enhanced_cuda_pipeline`) plus the CPU-vs-Basic-vs-Enhanced
benchmark infrastructure.

### Why Threshold's optimization ceiling is the smallest of all five filters

Threshold is a pure pointwise operation: one comparison, one write, no
neighborhood, no border handling, no floating-point arithmetic. Basic
Threshold already has adjacent-thread-to-adjacent-pixel coalesced
access (verified by inspection, not changed -- Section 10 spec item 6:
"if it already does [have coalesced access], record already optimal
rather than modifying code unnecessarily"). Unlike every other filter
in this project, there is no "reduce redundant reads" lever at all. The
only two plausible levers are wider memory transactions (vectorization)
and fewer threads/blocks (multi-pixel-per-thread) -- measured, not
assumed, below.

### Two variants

| Variant | Adds | Isolates the gain from |
|---|---|---|
| Vectorized | `uchar4` load/compare/store, 4 contiguous pixels/thread. Safe only when `width % 4 == 0`; **automatically falls back to an equivalent scalar kernel otherwise** (the real dataset has widths like 1733 that aren't 4-aligned -- spec item 7's explicit requirement, exercised directly in tests) | Wider memory transactions |
| MultiPixel | 4 contiguous pixels/thread via a scalar unrolled loop, no vector types, safe for any width | Fewer threads/blocks (reduced scheduling overhead), independent of vectorized access |

### Correctness: exact match, no tolerance

Threshold is a single uint8 comparison -- `max_abs_diff == 0` is the
only acceptable outcome. Verified across all equality-boundary cases
(`threshold_value` at 0, 1, 127, 128, 254, 255, and the classic
`threshold-1/threshold/threshold+1` triples spec item 15 requires),
every fixture, and widths spanning both the vectorized fast path and
its scalar fallback (4, 5, 15, 33, 224, and the real dataset's 1733) --
**every one bit-exact** vs. CPU and Basic CUDA.

### Measured results (batched, 122×224×224, real X-ray)

| Variant | Kernel (median) | Speedup vs Basic | Correctness |
|---|---:|---:|---|
| Basic | 0.105 ms | 1.00× | PASS (reference) |
| **Vectorized** | **0.075 ms** | **1.41×** | PASS |
| MultiPixel | 0.105 ms | 1.00× | PASS |

MultiPixel's essentially-flat result confirms the prediction: "fewer
threads" alone doesn't help a kernel this cheap, but "wider memory
transactions" (Vectorized) does. Batch-size sweep {1,8,...,512}: at
batch≤32 the ~0.01-0.05ms kernel time is dominated by measurement
noise (Vectorized even measured *slower* than Basic at some small
batches -- documented honestly, not cherry-picked away); from batch=128
upward the effect is clean and grows with scale (1.45× at batch=128 to
1.84× at batch=512), the same launch-overhead-amortization pattern
Section 6 first identified. On a large real X-ray with a **non-4-aligned
width** (2232×3282, width%4=2): Vectorized correctly used its scalar
fallback and measured 1.02× (i.e. equivalent to Basic, as expected when
the fast path can't engage, with zero correctness cost) -- confirmed,
not assumed, on real data.

### Register pressure (measured via `nvcc -Xptxas -v`, sm_89)

| Kernel | Registers | Shared memory | Spill stores/loads |
|---|---:|---:|---:|
| Scalar (Basic-equivalent) | 10 | 0 B | 0 / 0 |
| Vectorized | 11 | 0 B | 0 / 0 |
| MultiPixel | 11 | 0 B | 0 / 0 |

Lowest register usage of any kernel in the project (no shared memory,
no `__syncthreads()`, no neighborhood) -- confirms this was never a
register- or occupancy-bound kernel; the entire, small measured gain
comes from memory transaction width alone.

### Threshold fusion experiment (spec items 10-11, 31)

An experimental fused Laplacian+Threshold kernel (`laplacian_threshold_fused_batch_gpu`,
`cuda/include/laplacian_threshold_fused.cuh`) computes Specialized
Enhanced Laplacian and applies Threshold in one launch, without writing
the intermediate Laplacian output to global memory at all. Measured on
125 real 224×224 images:

| | Unfused (2 launches) | Fused (1 launch) |
|---|---:|---:|
| Combined kernel time (median) | 0.430 ms | 0.330 ms |
| **Speedup** | | **1.30×** |

Correctness: bit-exact vs. both the unfused 2-launch result and CPU.
This is a genuine, real, measured gain -- **not** rejected as
negligible. However, it is **NOT wired into the production pipeline**
(`BasicPipelineConfig` has no fusion flag; `run_basic_cuda_pipeline()`/
`run_enhanced_cuda_pipeline()` always run Laplacian and Threshold as
separate, independently-inspectable stages). Reasoning, matching the
spec's own explicit two-mode framing (item 10 "Standard mode" vs.
"Experimental fused mode"):

- The measured combined-kernel gain (1.30×, ≈0.1ms saved) translates to
  only a **modest ~3% additional whole-pipeline** improvement on top of
  the already-Enhanced pipeline -- real, but small relative to the
  architectural cost of wiring cross-filter fusion into the production
  config surface.
- Fusion permanently discards the standalone Laplacian output, which
  spec item 10 explicitly requires be available for visualization,
  debugging, benchmarking, and correctness verification -- exactly the
  per-stage introspection this project's whole benchmark suite (every
  `*_ms` timing field, every intermediate-output test) depends on.
- Kept as a fully implemented, tested (`tests/test_threshold_enhanced.py`'s
  fusion tests), benchmarked, and documented **experimental capability**
  -- available via its own standalone function for anyone who wants the
  extra ~3% and doesn't need the intermediate Laplacian output -- rather
  than the production default.

### Full-pipeline effect (Threshold alone): real at the kernel level, below the noise floor alone

Same pattern as Sobel (Section 8): isolating Threshold only (all other
stages Basic), the kernel-level gain (1.41×) doesn't produce a
measurable whole-pipeline effect (0.97-1.03× across batch sizes,
statistically flat) because Threshold is only ~2% of total pipeline
time. Documented honestly, not inflated.

### The two canonical production pipelines

```python
from cuda.pipeline import run_basic_cuda_pipeline, run_enhanced_cuda_pipeline

basic_output, basic_timing = run_basic_cuda_pipeline(images, config)        # all 5 stages Basic
enhanced_output, enhanced_timing = run_enhanced_cuda_pipeline(images, config)  # all 5 stages Enhanced, best variant per stage
```

`run_basic_cuda_pipeline()` (Section 5) remains the official all-Basic
baseline unchanged -- with no `use_enhanced_*` flags passed, it is
exactly what it always was. `run_enhanced_cuda_pipeline()` (new, spec
item 18) is a thin, explicitly-named wrapper that sets all five
`use_enhanced_*` flags True with each stage's Section 6-10 measured-best
variant (Gaussian: specialized, Median: network3x3, Sobel: specialized,
Laplacian: specialized, Threshold: vectorized) -- kept as a real function
rather than a convention callers must remember, so this exact
configuration can never be silently redefined elsewhere.

For anything in between (per spec items 19-20's explicit requirement to
distinguish CUDA *implementation* parameters from image-processing
*parameters*), `PipelineConfig` + `run_cuda_pipeline()` let any
combination of Basic/Enhanced be selected per stage independently --
this is what the future Streamlit UI's per-filter "Enhanced" toggles
will drive:

```python
from cpu.filters import FilterConfig
from cuda.pipeline import PipelineConfig, run_cuda_pipeline

filter_config = FilterConfig(gaussian_sigma=1.0, threshold_value=100)   # WHAT processing happens
pipeline_config = PipelineConfig(gaussian_variant="specialized", median_variant="basic")  # HOW it's computed on the GPU
output, timing = run_cuda_pipeline(images, filter_config, pipeline_config)
```

### CPU vs. Basic CUDA vs. Enhanced CUDA -- three-way benchmark infrastructure

`cuda.benchmark.benchmark_cpu_basic_enhanced_pipelines()` extends
Section 5's `benchmark_basic_cuda_pipeline()` with a third measurement
arm (`run_enhanced_cuda_pipeline()`), on the SAME loaded images, reusing
the same `Stats`/`CorrectnessMetrics` machinery so all three arms are
directly comparable. `scripts/benchmark_final_pipelines.py` drives this
across resolution groups and writes JSON (full detail, one file per
group) + a CSV summary row per group -- timestamp, GPU/driver/CUDA
version, dataset fingerprint, seed, resolution, batch size, all three
implementations' per-stage timings, throughput, speedups, and 3-way
correctness metrics (Basic-vs-CPU, Enhanced-vs-CPU, Enhanced-vs-Basic).
This is the dataset Section 11's Streamlit dashboard will read.

### Per-filter kernel speedup summary (all sections, batched 224×224)

| Filter | Basic kernel | Enhanced kernel | Speedup |
|---|---:|---:|---:|
| Gaussian | 1.258 ms | 0.377 ms | 3.34× |
| Median | 1.582 ms | 0.269 ms | 5.87× |
| Sobel | 0.420 ms | 0.395 ms | ~1.05× |
| Laplacian (k=3) | 0.531 ms | 0.352 ms | 1.51× |
| Threshold | 0.105 ms | 0.075 ms | 1.41× |

### Full-pipeline measurement: two honest numbers, not one

`scripts/benchmark_final_pipelines.py` measured on 122 real 224×224
images (batch=125 sampled, seed=42, 20 measurement runs):

| | CPU | Basic CUDA | Enhanced CUDA |
|---|---:|---:|---:|
| GPU compute only (H2D+kernels+D2H) | -- | 5.553 ms | 3.405 ms |
| **Full total (incl. one-time disk load, Section 5's fairness convention)** | 75.126 ms | 47.016 ms | 44.868 ms |
| Images/sec (full total) | 1624 | 2595 | 2719 |
| Speedup vs. CPU (full total) | 1.00× | 1.60× | 1.67× |

**Two different, both-valid numbers, not a contradiction:** the
**GPU-compute-only** ratio (5.553/3.405 = **1.63×**, consistent with
Section 9's dedicated combined-4-stage measurement of 1.71×, small
difference from different sample images/run counts) is what the
per-filter table above and every earlier section's "full-pipeline
speedup" number measured -- it isolates GPU processing exactly as
Sections 6-9 did. The **full total** additionally folds in one-time
disk I/O for loading 122 real JPEG files (Section 5's `load_ms`
fairness convention, needed so the CPU arm's total is comparable at
all) -- at this batch size, real disk I/O (tens of ms) dominates over
GPU processing (a few ms for the whole batch), which mechanically
compresses the *visible* Basic→Enhanced gain from 1.63× down to 1.05×
in the full-total column, even though the underlying GPU work sped up
by the full 1.63×. Batch-size sweep {1,...,512} confirms this: the
full-total "Enhanced optimization gain" stays in the 1.0-1.16× range at
every batch size (disk I/O is a roughly constant per-call cost
regardless of GPU batch size), while the GPU-compute-only ratio matches
the per-filter numbers throughout. Both numbers are reported rather
than picking the more flattering one.

### Correctness: three levels (spec item 26)

- **Level 1 (filter-level):** every one of the five filters' own test
  suites (Sections 4B-4F Basic, 6-10 Enhanced) independently verifies
  CPU vs. Basic, CPU vs. Enhanced, and Basic vs. Enhanced -- already
  exact-match (Median/Sobel/Laplacian/Threshold) or the measured,
  justified ±1 tolerance (Gaussian) established in their own sections.
- **Level 2 (intermediate pipeline):** every filter's GPU-native API
  (`*_cuda_gpu()` / `*_enhanced_cuda_gpu()`) returns
  `{'output': GpuImage, ...}`, so any stage's output can be inspected in
  isolation by chaining GPU-native calls manually -- no separate
  "diagnostic mode" flag was needed since this capability already
  exists per-filter (used throughout this project's own pipeline-
  integration tests, e.g. `test_..._pipeline_matches_cpu_and_basic_pipeline`).
- **Level 3 (final output):** `scripts/benchmark_final_pipelines.py`
  reports `max_abs_diff`, `mean_abs_diff`, `rmse`, and differing-pixel
  percentage for all three pairwise comparisons (Basic-vs-CPU,
  Enhanced-vs-CPU, Enhanced-vs-Basic) -- `max_abs_diff=255` alone is
  never treated as failure (a single flipped Threshold pixel from a
  Gaussian-tolerance-level upstream difference legitimately produces
  the maximum possible byte difference; the differing-pixel percentage
  is the meaningful metric, per spec item 26).

### Final differing-pixel percentage (threshold sensitivity, spec item 27)

Measured on 122 real 224×224 images, all five stages Enhanced vs. all
five stages Basic vs. CPU:

| Comparison | Differing pixels | max_abs_diff |
|---|---:|---:|
| Basic pipeline vs. CPU | 0.0178% | 255 |
| Enhanced pipeline vs. CPU | 0.0178% | 255 |
| **Enhanced pipeline vs. Basic pipeline** | **0.0000%** | **0** |

Identical to the baseline established in Section 5 and re-confirmed
after every subsequent Enhanced filter -- no deterioration at any point
across Sections 6-10, and the two production pipelines are **bit-exact**
against each other despite five independently-optimized filter stages.

### Test suite determinism (spec item 37)

An intermittent failure in `test_gaussian_enhanced.py`'s border-region
tests was traced to `hash((edge, variant)) % (2**32))`-style RNG
seeding across 8 test files: Python's built-in `hash()` salts
str/tuple hashes with a per-process random seed (`PYTHONHASHSEED`)
unless explicitly disabled, so these tests generated a **different**
"random" synthetic image on every interpreter invocation -- occasionally
landing on a pixel that crosses Gaussian's documented ±1 tolerance
purely by chance, not a real bug. Fixed by adding `tests.conftest.stable_seed()`
(hashlib-based, not salted) and replacing every `hash((...))
% (2**32)` call site across all 8 files with it. Verified deterministic
by running the full suite twice with different explicit
`PYTHONHASHSEED` values (1 and 99999): **1700/1700 passed identically
both times**, plus a normal (randomized-hash) run -- the flakiness is
resolved, not merely rerun-until-green.

### Rejected/non-default experiments, summarized

| Optimization | Result | Decision |
|---|---|---|
| MultiPixel (fewer threads, scalar) | 1.00× (flat) | Kept as `MultiPixel` variant for the comparison; not the winner |
| Vectorized at small batch sizes | Noisy, occasionally < 1.0× at batch≤32 | Documented honestly; real gain only visible/reliable from batch≈128+ |
| Block/tile tuning away from 16×16 | No meaningful difference at this kernel's scale (register/shared-memory-free) | Kept 16×16 |
| Laplacian+Threshold fusion | Real 1.30× combined-kernel gain, ~3% additional whole-pipeline | Kept as a tested, benchmarked, documented experimental capability; NOT wired into production (sacrifices per-stage introspection for a modest additional gain) |

### Python API

```python
from cuda.threshold import threshold_enhanced_cuda, threshold_enhanced_cuda_gpu

output = threshold_enhanced_cuda(image, threshold_value=128, variant="vectorized")  # single image, testing/diagnostics
```

### Commands

```bash
python scripts/benchmark_threshold_optimization.py --batch-size 125 --runs 20
python scripts/benchmark_final_pipelines.py --dataset ../data --batch-size 125 --seed 42 --runs 5
pytest tests/test_threshold_enhanced.py -v
pytest -q   # full suite, deterministic regardless of PYTHONHASHSEED
```

## Final Reproducible Benchmark & Experiment Framework (Section 11)

Every filter is optimized and validated (Sections 6-10); this section
builds the framework that makes those results reproducible,
machine-readable, and ready for Section 12's Streamlit dashboard to
consume directly -- without rerunning anything at startup.

### Architecture

`cuda/final_benchmark.py` is the framework; `scripts/run_final_benchmark.py`
is the canonical CLI that drives it end-to-end and writes everything
under `benchmark_results/`:

```text
benchmark_results/
├── raw/{cpu,basic_cuda,enhanced_cuda}/{benchmark_id}.json   # every individual run, never just averages
├── aggregated/{benchmark_id}.json        # full CanonicalBenchmarkResult (mean/median/min/max/std/cv)
├── batch_sweeps/{benchmark_id}.json
├── resolution_sweeps/{benchmark_id}.json
├── per_filter/{benchmark_id}.json        # + Amdahl-style contribution analysis
├── correctness/{benchmark_id}.json       # 3-level correctness benchmark
├── manifests/{benchmark_id}.json         # dataset+environment+configuration fingerprint, reproducibility metadata
├── summary/{benchmark_id}.json + latest.json   # everything combined into one document
└── experiment_registry.json              # catalogs all 8 known experiment types + their latest run
```

Benchmark IDs (`20260824_055212_seed42_batch122_224x224`) are
timestamp-unique and never overwrite earlier data (a same-second
collision appends a numeric suffix, verified in tests).

### The four timing modes (spec item 5) -- no new instrumentation needed

`run_basic_cuda_pipeline()`/`run_enhanced_cuda_pipeline()` already
return CUDA-event-measured H2D/per-kernel/D2H/compute/total timings
(Section 5), so the four modes map directly onto existing fields with
zero new kernel instrumentation:

| Mode | Definition | Source |
|---|---|---|
| 1. Kernel only | Sum of per-filter kernel times | `compute_ms` |
| 2. GPU processing | H2D + kernels + D2H | `total_ms` |
| 3. Pipeline (no disk) | GPU: same as Mode 2. CPU: processing time alone | `total_ms` / `processing_ms` |
| 4. End-to-end | Mode 3 + one-time disk load | + `load_ms` (Section 5/10 convention) |

### Fingerprints (spec items 7-9) -- everything needed to know exactly what ran, where

- **Dataset**: `DatasetManager.fingerprint()` (Section 2, already existed) -- sorted relative paths + sizes +
  mtimes, cheap on every run (never hashes pixel contents, per spec item 7).
- **Environment**: new `pipeline/environment.py::get_environment_fingerprint()` -- OS, CPU model/cores, GPU
  name/VRAM/compute-capability/driver/CUDA-runtime (via `xray_cuda.device_info()`), Python/OpenCV/NumPy/pybind11
  versions, git commit. Every field best-effort (`None` on failure, never raises -- a missing optional field
  must never abort a benchmark).
- **Configuration**: `CudaImplementationConfig` (variant + block per filter) kept strictly separate from
  `FilterConfig` (image-processing parameters) in every manifest -- the same separation `PipelineConfig`
  (Section 10) already enforces at the API level, now enforced in the *stored result* too.

### Validation and sanity checks (spec items 33-34)

Every canonical benchmark validates: selected file paths still exist, timings are non-negative, image count is
positive, and every reported speedup is *computed* from the raw aggregated means (never hand-entered -- verified
directly in tests by recomputing each speedup from its `AggregatedStat` and comparing). A run with any problem is
marked `status: "INVALID"` with `validation_notes` explaining why, in the manifest itself -- never silently
discarded.

### Measured results (canonical benchmark: 122 real 224×224 images, seed=42, 20 runs)

| | CPU | Basic CUDA | Enhanced CUDA |
|---|---:|---:|---:|
| Mode 1 (kernel only) | -- | 3.618 ms | 1.323 ms |
| Mode 4 (end-to-end) | 89.019 ms | 60.032 ms | 58.001 ms |

Basic speedup vs CPU: **1.483×** · Enhanced speedup vs CPU: **1.535×** · Enhanced vs Basic (compute-only):
**2.734×** · Enhanced vs Basic (end-to-end): **1.035×** (compressed by one-time disk I/O, exactly Section 10's
already-documented finding, reconfirmed here). Correctness: Enhanced vs Basic **bit-exact** (0.0000% differing);
Enhanced vs CPU **0.0178%** differing -- unchanged from the Section 5 baseline.

### Batch-size sweep (spec items 15-16, 1013-image selection, same seed)

| Batch | Enhanced vs Basic gain |
|---:|---:|
| 1 | 0.95× (noise-dominated at this scale, documented not hidden) |
| 8 | 1.47× |
| 32 | 1.73× |
| 128 | 1.67× |
| 512 | 1.60× |

Machine-readable rows (Plots A-D's underlying data) are written to `batch_sweeps/{id}.json`; Section 12 renders
them, this section only produces the numbers.

### Resolution sweep (spec items 17-18)

| Resolution | Images | Basic ms/image | Enhanced ms/image |
|---|---:|---:|---:|
| 1733×858 | 1 | 1.914 | 1.020 |
| 902×1128 | 1 | 1.271 | 0.775 |
| 858×958 | 1 | 1.282 | 0.634 |
| 224×224 | 122 | 0.053 | 0.031 |

### Per-filter benchmark + Amdahl-style contribution analysis (spec items 19-21)

Not a theoretical Amdahl's-Law bound -- each filter's actual measured `basic_ms - enhanced_ms` as a percentage of
the total measured compute-time reduction (verified in tests to sum to ~100% across all five filters):

| Filter | Basic | Enhanced | Speedup | % of total compute reduction |
|---|---:|---:|---:|---:|
| Gaussian | 1.943 ms | 0.563 ms | 3.46× | 37.0% |
| Median | 2.581 ms | 0.453 ms | 5.70× | 57.0% |
| Sobel | 0.643 ms | 0.715 ms | 0.90× | -1.9% |
| Laplacian | 0.911 ms | 0.672 ms | 1.36× | 6.4% |
| Threshold | 0.148 ms | 0.090 ms | 1.64× | 1.6% |

This is exactly why Section 8 (Sobel) mattered least and Section 7 (Median) mattered most for the overall
pipeline: **Gaussian + Median alone account for ~94% of the total measured compute-time reduction** -- Sobel's
occasional negative contribution here (a session's measurement noise around its already-small, previously-
documented ~1.0× effect) is reported honestly, not smoothed away.

### 3-level correctness benchmark (spec items 23-26)

- **Level 1 (filter)**: CPU-vs-Basic, CPU-vs-Enhanced, Basic-vs-Enhanced for all five filters, each against its
  documented tolerance (Gaussian ±1, all others exact). All pass.
- **Level 2 (intermediate pipeline)**: satisfied by every filter's existing GPU-native API (documented in
  Section 10's README entry) -- no separate "diagnostic mode" needed.
- **Level 3 (final pipeline)**: `max_abs_diff`/`mean_abs_diff`/`rmse`/differing-pixel % for all three pairwise
  comparisons, with an automatic baseline-comparison note (flags if the ~0.0178% established baseline is
  exceeded, rather than silently accepting drift).

### Reproducibility (spec items 29-32) -- verified, not just built

```bash
python scripts/reproduce_benchmark.py --benchmark-id 20260824_055212_seed42_batch122_224x224
```

Loads the manifest, re-checks the dataset fingerprint (reports a mismatch rather than silently proceeding --
spec item 31), and re-runs the identical selection/seed/configuration as a **new** benchmark_id. Measured
directly: original run's Enhanced-vs-Basic compute-only speedup was 2.734×; the reproduction run measured
2.754× -- a **0.7% relative difference**, i.e. genuinely reproducible given normal run-to-run GPU timing
variance.

### Determinism (spec item 42)

Ran the full suite at three different `PYTHONHASHSEED` values (default/randomized, `1`, `99999`): **1736/1736
passed identically every time** -- the Section 10 `stable_seed()` fix holds, and Section 11 introduces no new
`hash()`-seeded randomness (benchmark IDs are timestamp-based, not hash-based).

### Loader APIs for Section 12 (spec item 40)

```python
from cuda.final_benchmark import (
    load_benchmark_summary, load_batch_sweep, load_resolution_sweep,
    load_per_filter_results, load_correctness_results,
)

summary = load_benchmark_summary()  # defaults to the most recent run; returns None (not an exception) if nothing has run yet
```

Every loader accepts an explicit `benchmark_id` or defaults to the most recently written result -- the future
dashboard reads these directly and only re-runs `scripts/run_final_benchmark.py` when a user explicitly asks
for fresh numbers, per spec item 40's requirement.

### No fabricated values (spec item 35)

Every number in this section came from an actual `scripts/run_final_benchmark.py` run against the real dataset,
captured directly from its printed output and the JSON it wrote -- none hand-typed or estimated.

### Commands

```bash
python scripts/run_final_benchmark.py --dataset ../data --seed 42 --images 125 --runs 20
python scripts/reproduce_benchmark.py --benchmark-id <ID>
pytest tests/test_final_benchmark.py -v
```

## Streamlit Application (Section 12)

### Launch

```bash
streamlit run app.py
```

To reach it from another machine (e.g. viewing a remote desktop session's Streamlit app over the network
instead of only on that box via `localhost`), bind it to all interfaces and open the printed "Network URL":

```bash
streamlit run app.py --server.address 0.0.0.0
```

This has no authentication, so only do this on a trusted network. See `presentation/README.md` for the
equivalent remote-access setup for the standalone React presentation app (`presentation/`).

### Architecture

Thin orchestrator, per the project's own layering discipline (spec item 58):

```text
app.py  (tabs, layout, session-state wiring)
   ↓
ui/ components (controls, images, performance, analytics, correctness, system -- rendering only)
   ↓
ui/services.py  (the ONLY module that calls the backend)
   ↓
DatasetManager, FilterConfig, cpu.pipeline, cuda.pipeline, cuda.final_benchmark, xray_cuda
```

No filter, kernel, or benchmark algorithm is duplicated anywhere under `ui/` -- every processing/benchmark call
in `ui/services.py` is orchestration around code that already exists and is already tested (Sections 1-11).
`ui/services.ServiceError` wraps every backend exception into a user-facing message; the UI never shows a raw
C++ traceback unless Debug mode is explicitly enabled.

### UI overview

**Header**: the project's story in under 30 seconds -- CPU/Basic/Enhanced timing cards and the GPU compute
speedup, loaded directly from the Section 11 canonical benchmark summary (`benchmark_results/summary/latest.json`),
never rerun just to render the page.

**Sidebar**: dataset directory (uses `DatasetManager` directly, no reimplemented scanner) → image selection
(Single image / Random batch, using the existing Section 2 selection infrastructure, never independently sampled
per implementation) → per-filter parameter controls (values constrained to `FilterConfig`'s own allowed sets) →
processing implementation choice → Compare-mode checkboxes → Presentation/Debug toggles.

**Tabs** (Presentation mode hides Optimization Lab and System):

1. **Live Processing** -- pipeline diagram (disabled stages struck through, order never changes). **Single-image
   mode** (Section 14) shows the selected X-ray's filename/path/width/height/dtype and the original image
   (fit-to-container or actual size), then two explicit actions: **"Process Selected Image"** (runs only the
   sidebar's chosen implementation) and **"Compare CPU vs Basic CUDA vs Enhanced CUDA"** (always runs all three,
   regardless of the Compare-mode checkboxes, resilient per-implementation -- if one backend fails the others'
   results still display, each backed by `ui.services.ServiceError`). A comparison shows: three performance cards
   + the three measured speedups; H2D/Kernel/D2H breakdown for both CUDA implementations; a per-filter
   CPU/Basic/Enhanced timing table; the three-way final output; an optional "Compare intermediate stages" view
   (every enabled stage, one row of CPU/Basic/Enhanced columns) with optional per-stage difference images;
   per-stage PASS/WARNING correctness (Gaussian's documented ±1 tolerance, all others exact) plus full-pipeline
   `max_abs_diff`/`mean_abs_diff`/RMSE/differing-pixel-% computed from that exact run (never hardcoded) with an
   expandable note on threshold amplification; a labeled CUDA-implementation-vs-image-processing-parameter
   summary; and a **"Save Results"** button that writes `original.png` + per-implementation stage/final PNGs +
   difference PNGs + `metadata.json` to a fresh, timestamped directory under `outputs/live_processing/` (never
   `benchmark_results/`, never overwritten). Batch mode keeps Section 12's original thumbnail-grid + "Run
   Comparison" (checkbox-gated) behavior unchanged -- richer batch tooling is Section 15's scope.
2. **CPU vs GPU** -- mirrors whichever result is current (single-image `ComparisonResult` or batch `run_compare`
   result): side-by-side final outputs, amplified absolute-difference visualization with real (not exaggerated)
   differing-pixel statistics, and a small session-local "Benchmark Current Configuration" button (clearly
   labeled 🔴 LIVE RESULT, distinct from 📊 CANONICAL BENCHMARK -- never mixed, never written to
   `benchmark_results/`).
3. **Performance Analytics** -- loads Section 11 artifacts directly: total-time/throughput charts, compute-only
   vs. end-to-end (with the explanatory note about disk I/O), per-filter kernel-time and speedup charts, the
   measured (not theoretical) Amdahl-style contribution pie chart, batch-size sweep charts, resolution sweep
   table, and a benchmark-history selector for inspecting any past canonical run.
4. **Optimization Lab** -- per-filter CUDA variant pickers, each option explicitly labeled "production default",
   "basic (no optimization)", or "experimental" -- never silently substituting a rejected variant. Runs the
   chosen mixed configuration directly via `cuda.pipeline.PipelineConfig`/`run_cuda_pipeline()` (Section 10's own
   API, not a UI reimplementation) and shows the resulting pipeline stages. Also lists the experiment registry.
5. **Correctness** -- three levels (filter/intermediate/final pipeline), loaded from the stored correctness
   benchmark; a `max_abs_diff=255` is explained rather than treated as automatic failure; an expandable details
   panel with known expected differences, all pairwise metrics, and the exact configuration used.
6. **System** -- GPU/CUDA/driver/CPU/Python/OpenCV info (`pipeline.environment.get_environment_fingerprint()`,
   Section 11) and GPU memory (total/free/estimated-batch, explicitly not encouraging full-VRAM allocation).

### Processing modes

- **CPU / Basic CUDA / Enhanced CUDA** (sidebar radio) selects the implementation Live Processing uses when
  Compare mode isn't active.
- **Compare mode** (sidebar checkboxes) runs any combination of the three on the identical selected image(s), in
  the identical order, for a fair comparison -- never independently sampled.
- **Live benchmark** (CPU vs GPU tab) is a small, session-local, configurable (warmup/runs) benchmark for
  whatever filter parameters are currently tuned -- explicitly NOT the full Section 11 experiment matrix, and
  never overwrites `benchmark_results/`.

### Expected workflow

```text
Open Streamlit → point at the dataset → Random Batch (125 images, seed 42) → tune filters →
pick a preview image → Run Comparison (CPU + Basic + Enhanced) → compare outputs and timings →
open Optimization Lab → try an experimental variant → open Performance Analytics →
inspect per-filter speedups, batch-size scaling, Amdahl contribution → open Correctness →
confirm PASS at all three levels
```

### Testing

`tests/test_streamlit_app.py` covers component imports, `ui.state` session-state behavior,
`ui.services` dataset/selection validation and error handling, the spec-item-61 integration test
(select → CPU → Basic CUDA → Enhanced CUDA → verify results and timing, no Streamlit rendering involved), and
app-level smoke tests via Streamlit's own `AppTest` framework (verifies `app.py` runs end-to-end with zero
exceptions in several states -- default, single-image mode, Presentation mode, Debug mode, CPU-only, and after
clicking "Run Comparison" -- and that a bad dataset path produces a friendly sidebar error, not a crash).

```bash
pytest tests/test_streamlit_app.py -v
```

`tests/test_streamlit_live_processing.py` (Section 14) covers `ui.services.process_single_image()` /
`compare_single_image()` with small deterministic fixtures: single-implementation processing (CPU/Basic/Enhanced),
same `FilterConfig` and same image reaching all three, final + intermediate outputs and timing objects existing,
per-stage and full-pipeline correctness generation, per-backend CUDA-failure resilience (a Basic/Enhanced failure
never loses CPU's result), `Section 14` session-state fields, `save_comparison_results()`'s directory structure
and never-overwrite guarantee, and one real-data integration test (a real 224×224 X-ray through CPU → Basic CUDA
→ Enhanced CUDA → compare, asserting structure/timing existence, never an exact performance threshold, per spec
item 38).

```bash
pytest tests/test_streamlit_live_processing.py -v
```

### Batch Processing (Section 15)

Extends the exact same live-processing architecture to real batches, reusing `run_cpu_batch()`/`run_gpu_batch()`
(never a second CUDA/CPU processing path). Batch-size presets (1/8/16/32/64/128/256/512) + custom size + seed
reuse `DatasetManager`'s existing selection logic; mixed-resolution selections are grouped by native resolution
(never resized/padded) and each group is capped to its actual safe GPU capacity
(`cuda.pipeline.compute_safe_gpu_batch_size()`). A `BatchComparisonResult` aggregates timing/throughput/speedups
from measured values only and correctness across **every** processed image (never just the first), plus a
per-image distribution (images with zero vs. nonzero diff, max/mean per-image diff %). Only one representative
image's intermediate stages are ever computed (lazily, on navigation) -- never every image in a large batch. A
small "Quick batch-size sweep" (explicitly labeled 🔴 LIVE EXPERIMENT, distinct from Section 11's 📊 HISTORICAL
BENCHMARK) is available on demand. "Save Batch Results" writes to a fresh `outputs/live_batch_processing/{run_id}/`
directory, never `benchmark_results/`.

```bash
pytest tests/test_streamlit_batch_processing.py -v
```

### Historical Performance Analytics (Section 16)

The Performance Analytics tab loads a selectable historical benchmark (`benchmark_results/summary/*.json`, each
one a fully self-contained bundle including its own per-filter/batch-sweep/resolution-sweep/correctness data) --
never reruns Section 11's suite to render. Adds: a metadata panel + ★ CANONICAL BENCHMARK / HISTORICAL EXPERIMENT
badge (`benchmark_results/canonical.json` explicitly designates the canonical run, distinct from "most recently
run"); a GPU compute breakdown chart (H2D/per-filter/D2H, averaged from stored raw per-run measurements); a
3-bucket "measured contribution analysis" donut (Gaussian/Median/other filters); batch-size and resolution
speedup/throughput charts; run statistics + a raw-per-run-measurement expander; two-benchmark comparison (with an
explicit warning if the underlying configurations differ); and CSV/JSON export of the selected benchmark (never
modifying the source file). All loaders are `st.cache_data`-wrapped (historical JSON is static once written;
live/GPU state is never cached).

```bash
pytest tests/test_streamlit_performance_analytics.py -v
```

### Interactive CUDA Optimization Lab (Section 17)

Rebuilt around one filter at a time (Gaussian/Median/Sobel/Laplacian/Threshold radio selector) rather than
Section 12's "all 5 filters at once" picker. For the selected filter: a historical variant chart + table, loaded
from `benchmark_results/variant_sweeps/{filter}.json` (produced once by
`scripts/run_optimization_lab_experiments.py`, using the exact direct-call measurement methodology each
`scripts/benchmark_<filter>_optimization.py` script established -- never `cuda.pipeline.run_cuda_pipeline()`'s
general wrapper, whose overhead measurably distorts these sub-millisecond kernel comparisons); an optimization
progression, rejected-variant notes, and (for Laplacian) the Section 10 Laplacian+Threshold fusion experiment,
all clearly labeled 📊 HISTORICAL. A live "Run Live Variant Comparison" (2 warmup + 5 measured runs, same
input/config to both variants) reports measured kernel times, speedup, and full correctness (vs. CPU and vs. each
other) as 🔴 LIVE EXPERIMENT, plus a "Measure Pipeline Impact" action that actually runs the whole 5-filter
pipeline Basic vs. this-filter-enhanced (never inferred by multiplying the isolated per-filter speedup). Every
variant is already compiled -- this lab never invokes `nvcc`/`cmake` and never mutates the production Enhanced
configuration. "Save Experiment" writes to `outputs/optimization_lab/{run_id}/`, never `benchmark_results/`.

```bash
pytest tests/test_streamlit_optimization_lab.py -v
```

### Native CUDA Architecture Verification (Section 18)

An audit (not a rewrite) confirmed the GPU implementation was already fully native C++/CUDA: a single native
pipeline entry point (`run_basic_cuda_pipeline_batch()`, `cuda/src/pipeline_basic.cu`) performs
H2D→Gaussian→Median→Sobel→Laplacian→Threshold→D2H entirely on the GPU using two persistent, reused
`GpuImageBatch` ping-pong buffers -- zero GPU→CPU→GPU round trips between stages; RAII GPU memory
(`GpuImage`/`GpuImageBatch`/`DeviceBuffer<T>`); CUDA-event timing (`CudaTimer`) throughout; a thin pybind11 bridge
(`cuda/src/bindings.cpp`) with zero filter-algorithm logic; and zero Python-side GPU computation anywhere in the
repository (no cupy/numba/torch.cuda/cv2.cuda/pycuda). `scripts/inspect_gpu_backend.py` independently verifies
each of these claims at runtime rather than asserting them. See [GPU Architecture](#gpu-architecture) above for
the full data-flow description.

```bash
python scripts/inspect_gpu_backend.py
pytest tests/test_native_cuda_backend.py -v
```

### Presentation Mode & Final Release (Section 20)

A dedicated **Presentation Mode** tab (always available, alongside the existing sidebar "🎤 Presentation mode"
toggle that declutters the other tabs for a live demo) assembles a self-contained, ~30-second "story" screen:
title/subtitle + the "technical demonstration only, not a medical diagnostic system" notice, the
CPU/Basic-CUDA/Enhanced-CUDA architecture and 5-filter pipeline diagrams, the current live session's compared
image (if one has been run in Live Processing -- never auto-triggers a new run), the selected canonical
benchmark's performance cards / compute-vs-end-to-end / per-filter optimization cards / measured contribution
insight / correctness / a batch-size insight, and a GPU-implementation fact panel backed by the same verified
checks as `scripts/inspect_gpu_backend.py`. The System tab additionally shows NVCC/CMake versions and a
LOCAL ENVIRONMENT / BREV ENVIRONMENT badge (detected from Brev's own environment variables, defaulting to LOCAL).

## Optimization Research (Sections 20A-20F)

After Section 20's initial release, a further research-driven optimization arc investigated whether the
production pipeline had exploitable performance headroom beyond the per-filter Enhanced work (Sections 6-10).
Every experiment below lives in its own isolated files (`cuda/include/pipeline_*.cuh` / `cuda/src/pipeline_*.cu`,
`research/*.md`, `tests/test_*.py`, `scripts/benchmark_*.py`, `benchmark_results/research_optimization/`) and
**none of them modified a production kernel, `run_basic_cuda_pipeline()`, `run_enhanced_cuda_pipeline()`, or the
CPU implementation** -- verified directly in each section's own tests and re-verified in Section 22's final audit.

| Experiment | Section | Verdict | Why |
|---|---|---|---|
| Filter/pipeline research | 20A | -- | Literature + Nsight-informed research: per-filter kernels largely optimized (zero register spilling); pipeline-level memory management identified as the remaining opportunity. |
| GPU buffer persistence | 20B/20C | **REJECTED** | 1.15-1.62x faster in a same-shape-repeated-call microbenchmark, but ~28% *slower* in the realistic, varied Streamlit workload (parameters/batch/image changing between calls) -- the actual usage pattern this application has. |
| Pinned host memory | 20B | **REJECTED** | Isolated transfer-only benefit (~1.8-1.9x) did not survive integration: the real data path is always a pageable NumPy array, and the mandatory pageable->pinned staging copy cost exceeded the transfer speedup. |
| CUDA Graphs -- Basic pipeline | 20D | **EXPERIMENTAL** | Captures/replays the five Basic kernels via an isolated orchestration layer (production kernels called directly, unmodified). Realistic-workload gain ~1.44-1.56x; a wash at the largest (canonical) batch size tested. Kept as a tested, documented capability, not wired into production. |
| CUDA Graphs -- Enhanced pipeline | 20E | **EXPERIMENTAL** | Production Enhanced kernels have anonymous-namespace linkage (not header-exposed), so this experiment uses verified-identical, line-cited kernel copies in isolated files rather than editing production `.cu` files. Realistic-workload gain ~1.89x, the strongest result of the arc; withheld from ADOPT because it only covers the Basic-kernel-equivalent capture path and has no production API surface yet. |
| Multi-stream async pipeline | 20F | **EXPERIMENTAL** | Double/triple-buffered chunked pipelining across 3 explicit CUDA streams. Isolated same-shape microbenchmarks regressed 1.4-3.3x (chunking overhead exceeded the achievable overlap on this GPU's single async copy engine), while the realistic workload showed a modest, reproducible ~1.07-1.12x gain -- a genuine, disclosed tension the section did not force into a clean verdict. |

Every REJECTED/EXPERIMENTAL decision above was reached by measuring against the **same realistic, alternating-
order Streamlit workload methodology** (established in Section 20C, reused unchanged through Section 20F) --
never from an isolated best-case microbenchmark alone. Full research, measured evidence, and decision reasoning
for each experiment live in `research/*_decision.md`; nothing was deleted after being rejected -- rejected and
experimental code, tests, and raw benchmark evidence are preserved as part of the project's technical story (see
[Known limitations](#known-limitations) and the `research/`/`benchmark_results/research_optimization/`
directories).

## Project status

**Released and frozen.** Sections 1-20 plus the 20A-20F optimization research arc and the Section 22 final
release audit are complete and verified. The production CUDA kernels, the Basic CUDA baseline, the Enhanced CUDArun it 
algorithms, and the benchmark methodology are frozen -- see [Known limitations](#known-limitations) below and
[release/final_validation.md](release/final_validation.md) for the final verification record (test count,
determinism, build result, benchmark IDs, correctness/performance summary, repository hygiene audit).

Any future optimization or new CUDA variant must be introduced as a separately measured experiment (its own
benchmark, its own keep/reject decision, its own `benchmark_results/` or `outputs/` namespace) -- never a silent
change to the released production baseline. The CUDA Graph and async-pipeline experiments above are documented,
tested EXPERIMENTAL capabilities, not production defaults, and remain OFF unless a future section explicitly
re-evaluates and adopts one.

## Known limitations

- This is an image-processing **performance demonstration**, not a diagnostic tool -- it makes no claim about
  clinical validity, and its outputs must never be used for medical decision-making.
- GPU performance results are hardware-dependent: absolute timings and even some per-filter Basic→Enhanced
  speedup ratios will differ on a different GPU (compute capability, VRAM, driver, clock behavior). The
  qualitative story -- Enhanced meaningfully faster than Basic for most filters, GPU advantage scaling with batch
  size and image resolution -- is expected to hold; identical numbers are not.
- End-to-end (application-level) speedup is consistently smaller than GPU-compute-only speedup, because disk
  I/O, host-side orchestration, and memory transfers remain part of the full path regardless of how fast the
  kernels themselves run (see "Compute-only vs. end-to-end" in Performance Analytics / Presentation Mode).
- Small numerical differences between CPU/OpenCV and CUDA outputs can occur due to Gaussian floating-point
  summation-order differences (a documented ±1 per-pixel tolerance) -- Threshold can then amplify that into a
  0↔255 flip on a small percentage of pixels. This is expected, established project behavior, not a defect; the
  differing-pixel-percentage methodology (not `max_abs_diff` alone) is the correct way to interpret final-pipeline
  correctness, and every correctness display in this project uses it.
- The historical canonical benchmark (`benchmark_results/summary/`, `benchmark_results/canonical.json`) reflects
  one specific machine (see its stored `environment` field); the "clean environment" / Brev verification described
  in Section 19 is planned but was not executed in this environment (no Brev/SSH access was available in the
  session that reached Section 20) -- see `benchmark_results/native_cuda_migration/` for the most recent
  same-machine re-verification instead.
- The Section 20A-20F experimental optimizations (persistent GPU buffers, pinned host memory, CUDA Graphs for
  Basic/Enhanced, multi-stream async buffering) are **not** production defaults and are not selected by any
  Streamlit UI default. Persistent buffers and pinned memory were measured and explicitly REJECTED for
  production; CUDA Graphs and async streaming remain EXPERIMENTAL, documented, tested capabilities that a future
  section could revisit, not something silently active today. See
  [Optimization Research](#optimization-research-sections-20a-20f) above for the measured reasoning behind each
  decision.

