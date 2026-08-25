"""Basic CUDA filter benchmarking utilities (Section 4B: Gaussian,
Section 4C: Median). Shared machinery lives here once; per-filter
wrappers below only supply what differs (the GPU call, the CPU call,
and the reported parameters).

Two distinct, clearly-labeled benchmarks:

  benchmark_*_single_image -- repeated timing of ONE image: a single
      "cold" sample (first GPU call, including CUDA context init --
      only meaningful if called first in a fresh process) plus "warm"
      steady-state stats (mean/median/min/max/std over
      `measurement_runs`, after `warmup_runs` discarded runs).

  benchmark_*_selection -- loops the single-image Basic CUDA call over
      many different real images. Explicitly a "single-image GPU
      invocation baseline", NOT batch-optimized GPU processing -- it
      exists to measure where we're starting from before any
      batch/stream optimization work.

CPU timing uses time.perf_counter() (the CPU reference is pure
Python/OpenCV, no CUDA events available). GPU kernel-only timing comes
from the CUDA-event-measured 'kernel_ms' each xray_cuda.*_basic_gpu call
returns; H2D/D2H/end-to-end use time.perf_counter() around the
upload/download calls, per the Section 4B/4C spec's timing rules.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import xray_cuda
from cpu.filters import FilterConfig, apply_gaussian, apply_laplacian, apply_median, apply_sobel, apply_threshold
from cuda.gaussian import gaussian_kernel_2d
from cuda.laplacian import laplacian_kernel_2d
from cuda.pipeline import run_basic_cuda_pipeline, run_enhanced_cuda_pipeline
from cuda.sobel import mode_to_int as sobel_mode_to_int
from pipeline.dataset import ImageSelection
from pipeline.image_loader import load_image


@dataclass
class Stats:
    mean: float
    median: float
    min: float
    max: float
    std: float

    @classmethod
    def from_values(cls, values: List[float]) -> "Stats":
        return cls(
            mean=statistics.mean(values),
            median=statistics.median(values),
            min=min(values),
            max=max(values),
            std=statistics.stdev(values) if len(values) > 1 else 0.0,
        )


@dataclass
class GpuTimingSample:
    h2d_ms: float
    kernel_ms: float
    d2h_ms: float
    end_to_end_ms: float
    coeff_upload_ms: Optional[float] = None  # only Gaussian's result dict has this key


def _run_gpu_once(image: np.ndarray, gpu_filter_call: Callable[[object], dict]) -> GpuTimingSample:
    """`gpu_filter_call(gpu_image)` must return a dict with 'output'
    (GpuImage) and 'kernel_ms' (CUDA-event-measured float), optionally
    'coeff_upload_ms'."""
    end_to_end_start = time.perf_counter()

    h2d_start = time.perf_counter()
    gpu_image = xray_cuda.upload_image(image)
    h2d_ms = (time.perf_counter() - h2d_start) * 1000.0

    result = gpu_filter_call(gpu_image)

    d2h_start = time.perf_counter()
    xray_cuda.download_image(result["output"])
    d2h_ms = (time.perf_counter() - d2h_start) * 1000.0

    end_to_end_ms = (time.perf_counter() - end_to_end_start) * 1000.0

    return GpuTimingSample(
        h2d_ms=h2d_ms,
        kernel_ms=result["kernel_ms"],
        d2h_ms=d2h_ms,
        end_to_end_ms=end_to_end_ms,
        coeff_upload_ms=result.get("coeff_upload_ms"),
    )


@dataclass
class FilterBenchmarkResult:
    filter_name: str
    params: dict
    cold_end_to_end_ms: float
    warm: Dict[str, Stats]  # keys: h2d_ms, kernel_ms, d2h_ms, end_to_end_ms, (coeff_upload_ms if applicable)
    cpu_processing_ms: Stats
    warmup_runs: int
    measurement_runs: int
    image_shape: Tuple[int, int]


def benchmark_filter_single_image(
    image: np.ndarray,
    filter_name: str,
    gpu_filter_call: Callable[[object], dict],
    cpu_call: Callable[[], np.ndarray],
    params: dict,
    warmup_runs: int = 2,
    measurement_runs: int = 5,
) -> FilterBenchmarkResult:
    """Shared single-image benchmark loop. Call this first (in a fresh
    process, before any other xray_cuda GPU call) if `cold_end_to_end_ms`
    should reflect genuine CUDA context-initialization overhead."""
    if not xray_cuda.cuda_available():
        raise RuntimeError(f"benchmark_filter_single_image({filter_name}) requires a usable CUDA device.")

    cold_sample = _run_gpu_once(image, gpu_filter_call)

    for _ in range(warmup_runs):
        _run_gpu_once(image, gpu_filter_call)

    warm_samples = [_run_gpu_once(image, gpu_filter_call) for _ in range(measurement_runs)]
    warm = {
        "h2d_ms": Stats.from_values([s.h2d_ms for s in warm_samples]),
        "kernel_ms": Stats.from_values([s.kernel_ms for s in warm_samples]),
        "d2h_ms": Stats.from_values([s.d2h_ms for s in warm_samples]),
        "end_to_end_ms": Stats.from_values([s.end_to_end_ms for s in warm_samples]),
    }
    if warm_samples[0].coeff_upload_ms is not None:
        warm["coeff_upload_ms"] = Stats.from_values([s.coeff_upload_ms for s in warm_samples])

    cpu_times: List[float] = []
    for _ in range(warmup_runs + measurement_runs):
        start = time.perf_counter()
        cpu_call()
        cpu_times.append((time.perf_counter() - start) * 1000.0)
    cpu_processing_ms = Stats.from_values(cpu_times[warmup_runs:])

    return FilterBenchmarkResult(
        filter_name=filter_name,
        params=params,
        cold_end_to_end_ms=cold_sample.end_to_end_ms,
        warm=warm,
        cpu_processing_ms=cpu_processing_ms,
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs,
        image_shape=image.shape,
    )


@dataclass
class SelectionBenchmarkResult:
    label: str
    filter_name: str
    num_images: int
    cpu_total_ms: float
    cpu_throughput_images_per_sec: float
    gpu_total_ms: float
    gpu_throughput_images_per_sec: float


def benchmark_filter_selection(
    selection: ImageSelection,
    filter_name: str,
    gpu_filter_call: Callable[[object], dict],
    cpu_call: Callable[[np.ndarray], np.ndarray],
) -> SelectionBenchmarkResult:
    """Single-image GPU invocation baseline over a whole selection: calls
    upload -> <filter>_basic_gpu -> download once per image, in a loop.
    NOT a batched/optimized GPU pipeline. Images are preloaded before
    timing starts so disk I/O isn't conflated with processing.
    """
    if not xray_cuda.cuda_available():
        raise RuntimeError(f"benchmark_filter_selection({filter_name}) requires a usable CUDA device.")
    if len(selection) == 0:
        raise ValueError("Cannot benchmark an empty selection.")

    images = [load_image(item.absolute_path) for item in selection.items]

    cpu_start = time.perf_counter()
    for image in images:
        cpu_call(image)
    cpu_total_ms = (time.perf_counter() - cpu_start) * 1000.0

    gpu_start = time.perf_counter()
    for image in images:
        gpu_image = xray_cuda.upload_image(image)
        result = gpu_filter_call(gpu_image)
        xray_cuda.download_image(result["output"])
    gpu_total_ms = (time.perf_counter() - gpu_start) * 1000.0

    n = len(images)
    return SelectionBenchmarkResult(
        label="single-image GPU invocation baseline (not batch-optimized)",
        filter_name=filter_name,
        num_images=n,
        cpu_total_ms=cpu_total_ms,
        cpu_throughput_images_per_sec=(n / (cpu_total_ms / 1000.0)) if cpu_total_ms > 0 else 0.0,
        gpu_total_ms=gpu_total_ms,
        gpu_throughput_images_per_sec=(n / (gpu_total_ms / 1000.0)) if gpu_total_ms > 0 else 0.0,
    )


# -- Gaussian wrappers (Section 4B) -----------------------


def benchmark_gaussian_single_image(
    image: np.ndarray, kernel_size: int = 5, sigma: float = 0.0, warmup_runs: int = 2, measurement_runs: int = 5
) -> FilterBenchmarkResult:
    coeffs = gaussian_kernel_2d(kernel_size, sigma)
    config = FilterConfig(gaussian_kernel_size=kernel_size, gaussian_sigma=sigma)
    return benchmark_filter_single_image(
        image, "gaussian",
        gpu_filter_call=lambda gpu_image: xray_cuda.gaussian_basic_gpu(gpu_image, coeffs),
        cpu_call=lambda: apply_gaussian(image, config),
        params={"kernel_size": kernel_size, "sigma": sigma},
        warmup_runs=warmup_runs, measurement_runs=measurement_runs,
    )


def benchmark_gaussian_selection(
    selection: ImageSelection, kernel_size: int = 5, sigma: float = 0.0
) -> SelectionBenchmarkResult:
    coeffs = gaussian_kernel_2d(kernel_size, sigma)
    config = FilterConfig(gaussian_kernel_size=kernel_size, gaussian_sigma=sigma)
    return benchmark_filter_selection(
        selection, "gaussian",
        gpu_filter_call=lambda gpu_image: xray_cuda.gaussian_basic_gpu(gpu_image, coeffs),
        cpu_call=lambda image: apply_gaussian(image, config),
    )


# -- Median wrappers (Section 4C) -----------------------


def benchmark_median_single_image(
    image: np.ndarray, kernel_size: int = 3, warmup_runs: int = 2, measurement_runs: int = 5
) -> FilterBenchmarkResult:
    config = FilterConfig(median_kernel_size=kernel_size)
    return benchmark_filter_single_image(
        image, "median",
        gpu_filter_call=lambda gpu_image: xray_cuda.median_basic_gpu(gpu_image, kernel_size),
        cpu_call=lambda: apply_median(image, config),
        params={"kernel_size": kernel_size},
        warmup_runs=warmup_runs, measurement_runs=measurement_runs,
    )


def benchmark_median_selection(selection: ImageSelection, kernel_size: int = 3) -> SelectionBenchmarkResult:
    config = FilterConfig(median_kernel_size=kernel_size)
    return benchmark_filter_selection(
        selection, "median",
        gpu_filter_call=lambda gpu_image: xray_cuda.median_basic_gpu(gpu_image, kernel_size),
        cpu_call=lambda image: apply_median(image, config),
    )


# -- Sobel wrappers (Section 4D) -----------------------


def benchmark_sobel_single_image(
    image: np.ndarray, mode: str = "magnitude", warmup_runs: int = 2, measurement_runs: int = 5
) -> FilterBenchmarkResult:
    mode_int = sobel_mode_to_int(mode)
    config = FilterConfig(sobel_mode=mode, sobel_kernel_size=3)
    return benchmark_filter_single_image(
        image, "sobel",
        gpu_filter_call=lambda gpu_image: xray_cuda.sobel_basic_gpu(gpu_image, mode_int),
        cpu_call=lambda: apply_sobel(image, config),
        params={"mode": mode},
        warmup_runs=warmup_runs, measurement_runs=measurement_runs,
    )


def benchmark_sobel_selection(selection: ImageSelection, mode: str = "magnitude") -> SelectionBenchmarkResult:
    mode_int = sobel_mode_to_int(mode)
    config = FilterConfig(sobel_mode=mode, sobel_kernel_size=3)
    return benchmark_filter_selection(
        selection, "sobel",
        gpu_filter_call=lambda gpu_image: xray_cuda.sobel_basic_gpu(gpu_image, mode_int),
        cpu_call=lambda image: apply_sobel(image, config),
    )


# -- Laplacian wrappers (Section 4E) -----------------------


def benchmark_laplacian_single_image(
    image: np.ndarray, kernel_size: int = 3, scale: float = 1.0, delta: float = 0.0,
    warmup_runs: int = 2, measurement_runs: int = 5,
) -> FilterBenchmarkResult:
    coeffs = laplacian_kernel_2d(kernel_size)
    config = FilterConfig(laplacian_kernel_size=kernel_size, laplacian_scale=scale, laplacian_delta=delta)
    return benchmark_filter_single_image(
        image, "laplacian",
        gpu_filter_call=lambda gpu_image: xray_cuda.laplacian_basic_gpu(gpu_image, coeffs, scale, delta),
        cpu_call=lambda: apply_laplacian(image, config),
        params={"kernel_size": kernel_size, "scale": scale, "delta": delta},
        warmup_runs=warmup_runs, measurement_runs=measurement_runs,
    )


def benchmark_laplacian_selection(
    selection: ImageSelection, kernel_size: int = 3, scale: float = 1.0, delta: float = 0.0
) -> SelectionBenchmarkResult:
    coeffs = laplacian_kernel_2d(kernel_size)
    config = FilterConfig(laplacian_kernel_size=kernel_size, laplacian_scale=scale, laplacian_delta=delta)
    return benchmark_filter_selection(
        selection, "laplacian",
        gpu_filter_call=lambda gpu_image: xray_cuda.laplacian_basic_gpu(gpu_image, coeffs, scale, delta),
        cpu_call=lambda image: apply_laplacian(image, config),
    )


# -- Threshold wrappers (Section 4F) -----------------------


def benchmark_threshold_single_image(
    image: np.ndarray, threshold_value: int = 128, max_value: int = 255,
    warmup_runs: int = 2, measurement_runs: int = 5,
) -> FilterBenchmarkResult:
    config = FilterConfig(threshold_value=threshold_value, threshold_max_value=max_value)
    return benchmark_filter_single_image(
        image, "threshold",
        gpu_filter_call=lambda gpu_image: xray_cuda.threshold_basic_gpu(gpu_image, threshold_value, max_value),
        cpu_call=lambda: apply_threshold(image, config),
        params={"threshold_value": threshold_value, "max_value": max_value},
        warmup_runs=warmup_runs, measurement_runs=measurement_runs,
    )


def benchmark_threshold_selection(
    selection: ImageSelection, threshold_value: int = 128, max_value: int = 255
) -> SelectionBenchmarkResult:
    config = FilterConfig(threshold_value=threshold_value, threshold_max_value=max_value)
    return benchmark_filter_selection(
        selection, "threshold",
        gpu_filter_call=lambda gpu_image: xray_cuda.threshold_basic_gpu(gpu_image, threshold_value, max_value),
        cpu_call=lambda image: apply_threshold(image, config),
    )


# -- full CPU-vs-Basic-CUDA pipeline benchmark (Section 5) -----------------------
#
# Unlike benchmark_*_single_image/_selection above (which measure ONE
# filter at a time via the still-available per-filter APIs), this
# benchmarks the production run_basic_cuda_pipeline() -- one native call
# running all five stages -- against the full Section 3 CPU pipeline, on
# a single resolution group (the real dataset is mixed-resolution; see
# cuda/pipeline.py::group_by_resolution -- mixing resolutions into one
# throughput number would not be a fair or meaningful comparison, so
# this function intentionally operates on one resolution at a time).


def _cpu_full_pipeline(image: np.ndarray, config: FilterConfig) -> np.ndarray:
    output = image
    if config.gaussian_enabled:
        output = apply_gaussian(output, config)
    if config.median_enabled:
        output = apply_median(output, config)
    if config.sobel_enabled:
        output = apply_sobel(output, config)
    if config.laplacian_enabled:
        output = apply_laplacian(output, config)
    if config.threshold_enabled:
        output = apply_threshold(output, config)
    return output


@dataclass
class CorrectnessMetrics:
    max_abs_diff: int
    mean_abs_diff: float
    rmse: float
    differing_pixel_count: int
    differing_pixel_percentage: float

    @classmethod
    def compare(cls, cpu_batch: List[np.ndarray], gpu_batch: np.ndarray) -> "CorrectnessMetrics":
        diffs = [
            np.abs(cpu_img.astype(np.int16) - gpu_img.astype(np.int16))
            for cpu_img, gpu_img in zip(cpu_batch, gpu_batch)
        ]
        stacked = np.stack(diffs, axis=0)
        total_pixels = stacked.size
        differing = int(np.count_nonzero(stacked))
        return cls(
            max_abs_diff=int(stacked.max()),
            mean_abs_diff=float(stacked.mean()),
            rmse=float(np.sqrt(np.mean(stacked.astype(np.float64) ** 2))),
            differing_pixel_count=differing,
            differing_pixel_percentage=100.0 * differing / total_pixels,
        )


@dataclass
class BasicPipelineBenchmarkResult:
    timestamp_utc: str
    gpu_name: str
    driver_version: Optional[str]
    cuda_toolkit_version: Optional[str]
    dataset_fingerprint: Optional[str]
    seed: Optional[int]
    selected_image_count: int
    resolution: Tuple[int, int]
    requested_batch_size: Optional[int]
    effective_gpu_batch_size: int
    filter_config: dict
    warmup_runs: int
    measurement_runs: int

    cpu_load_ms: float
    cpu_processing_ms: Stats
    cpu_total_ms: float

    gpu_h2d_ms: Stats
    gpu_gaussian_ms: Optional[Stats]
    gpu_median_ms: Optional[Stats]
    gpu_sobel_ms: Optional[Stats]
    gpu_laplacian_ms: Optional[Stats]
    gpu_threshold_ms: Optional[Stats]
    gpu_compute_ms: Stats
    gpu_d2h_ms: Stats
    gpu_total_ms: float

    cpu_images_per_second: float
    gpu_images_per_second: float
    gpu_speedup: float

    correctness: CorrectnessMetrics

    def to_dict(self) -> dict:
        return asdict(self)


def _gpu_name() -> str:
    try:
        return xray_cuda.device_info().get("name", "unknown")
    except Exception:
        return "unknown"


def benchmark_basic_cuda_pipeline(
    images: List[np.ndarray],
    config: FilterConfig,
    warmup_runs: int = 2,
    measurement_runs: int = 5,
    requested_batch_size: Optional[int] = None,
    dataset_fingerprint: Optional[str] = None,
    seed: Optional[int] = None,
    load_ms: float = 0.0,
) -> BasicPipelineBenchmarkResult:
    """Benchmark the full CPU pipeline vs. the production Basic CUDA
    pipeline (one native call) on `images`, which must all share one
    resolution (same requirement as run_basic_cuda_pipeline).

    `load_ms` is the (already-measured, by the caller) disk-load time
    for `images` -- charged identically to both cpu_total_ms and
    gpu_total_ms so the comparison is fair (same input images, same
    base I/O cost) and matches the spec's "GPU end-to-end = disk load +
    H2D + kernels + D2H" definition. Pass 0.0 (default) to compare
    processing-only totals.
    """
    if not images:
        raise ValueError("benchmark_basic_cuda_pipeline requires at least one image.")
    shape = images[0].shape
    if any(img.shape != shape for img in images):
        raise ValueError("benchmark_basic_cuda_pipeline requires all images to share one resolution.")
    if not xray_cuda.cuda_available():
        raise RuntimeError("benchmark_basic_cuda_pipeline requires a usable CUDA device.")

    n = len(images)

    # -- CPU --
    cpu_times: List[float] = []
    for _ in range(warmup_runs + measurement_runs):
        start = time.perf_counter()
        cpu_outputs = [_cpu_full_pipeline(img, config) for img in images]
        cpu_times.append((time.perf_counter() - start) * 1000.0)
    cpu_processing_stats = Stats.from_values(cpu_times[warmup_runs:])
    cpu_total_ms = load_ms + cpu_processing_stats.mean

    # -- GPU (production pipeline, one native call per run) --
    for _ in range(warmup_runs):
        run_basic_cuda_pipeline(images, config)

    gpu_runs = [run_basic_cuda_pipeline(images, config) for _ in range(measurement_runs)]
    gpu_outputs_last, gpu_timings = gpu_runs[-1]

    def _optional_stats(attr: str) -> Optional[Stats]:
        values = [getattr(t, attr) for _out, t in gpu_runs]
        if values[0] is None:
            return None
        return Stats.from_values(values)

    gpu_h2d_stats = Stats.from_values([t.h2d_ms for _out, t in gpu_runs])
    gpu_compute_stats = Stats.from_values([t.compute_ms for _out, t in gpu_runs])
    gpu_d2h_stats = Stats.from_values([t.d2h_ms for _out, t in gpu_runs])
    gpu_total_ms = load_ms + gpu_h2d_stats.mean + gpu_compute_stats.mean + gpu_d2h_stats.mean

    # -- correctness (computed once, not per measurement run -- expensive relative to the timing itself) --
    correctness = CorrectnessMetrics.compare(cpu_outputs, gpu_outputs_last)

    cpu_images_per_second = n / (cpu_total_ms / 1000.0) if cpu_total_ms > 0 else 0.0
    gpu_images_per_second = n / (gpu_total_ms / 1000.0) if gpu_total_ms > 0 else 0.0
    gpu_speedup = cpu_total_ms / gpu_total_ms if gpu_total_ms > 0 else 0.0

    try:
        device_info = xray_cuda.device_info()
    except Exception:
        device_info = {}

    return BasicPipelineBenchmarkResult(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        gpu_name=_gpu_name(),
        driver_version=device_info.get("driver_version"),
        cuda_toolkit_version=device_info.get("runtime_version"),
        dataset_fingerprint=dataset_fingerprint,
        seed=seed,
        selected_image_count=n,
        resolution=(shape[0], shape[1]),
        requested_batch_size=requested_batch_size,
        effective_gpu_batch_size=n,
        filter_config=asdict(config),
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs,
        cpu_load_ms=load_ms,
        cpu_processing_ms=cpu_processing_stats,
        cpu_total_ms=cpu_total_ms,
        gpu_h2d_ms=gpu_h2d_stats,
        gpu_gaussian_ms=_optional_stats("gaussian_ms"),
        gpu_median_ms=_optional_stats("median_ms"),
        gpu_sobel_ms=_optional_stats("sobel_ms"),
        gpu_laplacian_ms=_optional_stats("laplacian_ms"),
        gpu_threshold_ms=_optional_stats("threshold_ms"),
        gpu_compute_ms=gpu_compute_stats,
        gpu_d2h_ms=gpu_d2h_stats,
        gpu_total_ms=gpu_total_ms,
        cpu_images_per_second=cpu_images_per_second,
        gpu_images_per_second=gpu_images_per_second,
        gpu_speedup=gpu_speedup,
        correctness=correctness,
    )


# -- CPU vs Basic CUDA vs Enhanced CUDA (Section 10) -----------------------
#
# The final three-way comparison: extends benchmark_basic_cuda_pipeline()
# above with a third measurement arm (run_enhanced_cuda_pipeline(), all
# five stages using each stage's Section 6-10 measured-best variant),
# reusing the same Stats/CorrectnessMetrics machinery so the three arms
# are directly comparable and computed from the SAME loaded images.


@dataclass
class ThreeWayPipelineBenchmarkResult:
    timestamp_utc: str
    gpu_name: str
    driver_version: Optional[str]
    cuda_toolkit_version: Optional[str]
    dataset_fingerprint: Optional[str]
    seed: Optional[int]
    selected_image_count: int
    resolution: Tuple[int, int]
    requested_batch_size: Optional[int]
    effective_gpu_batch_size: int
    filter_config: dict
    warmup_runs: int
    measurement_runs: int

    cpu_load_ms: float
    cpu_processing_ms: Stats
    cpu_total_ms: float

    basic_h2d_ms: Stats
    basic_gaussian_ms: Optional[Stats]
    basic_median_ms: Optional[Stats]
    basic_sobel_ms: Optional[Stats]
    basic_laplacian_ms: Optional[Stats]
    basic_threshold_ms: Optional[Stats]
    basic_compute_ms: Stats
    basic_d2h_ms: Stats
    basic_total_ms: float

    enhanced_h2d_ms: Stats
    enhanced_gaussian_ms: Optional[Stats]
    enhanced_median_ms: Optional[Stats]
    enhanced_sobel_ms: Optional[Stats]
    enhanced_laplacian_ms: Optional[Stats]
    enhanced_threshold_ms: Optional[Stats]
    enhanced_compute_ms: Stats
    enhanced_d2h_ms: Stats
    enhanced_total_ms: float

    cpu_images_per_second: float
    basic_images_per_second: float
    enhanced_images_per_second: float

    basic_speedup_vs_cpu: float       # cpu_total_ms / basic_total_ms
    enhanced_speedup_vs_cpu: float    # cpu_total_ms / enhanced_total_ms
    enhanced_speedup_vs_basic: float  # basic_total_ms / enhanced_total_ms ("optimization gain")

    correctness_basic_vs_cpu: CorrectnessMetrics
    correctness_enhanced_vs_cpu: CorrectnessMetrics
    correctness_enhanced_vs_basic: CorrectnessMetrics

    def to_dict(self) -> dict:
        return asdict(self)


def benchmark_cpu_basic_enhanced_pipelines(
    images: List[np.ndarray],
    config: FilterConfig,
    warmup_runs: int = 2,
    measurement_runs: int = 5,
    requested_batch_size: Optional[int] = None,
    dataset_fingerprint: Optional[str] = None,
    seed: Optional[int] = None,
    load_ms: float = 0.0,
) -> ThreeWayPipelineBenchmarkResult:
    """Benchmark CPU vs. the production Basic CUDA pipeline vs. the
    production Enhanced CUDA pipeline, all on the SAME `images` (must
    share one resolution), using each stage's Section 6-10 measured-best
    variant for the Enhanced arm (run_enhanced_cuda_pipeline()).

    Reuses benchmark_basic_cuda_pipeline()'s CPU+Basic measurement
    exactly (same warmup/measurement loop, same load_ms convention) and
    adds a third, otherwise-identical measurement loop for Enhanced --
    see that function's docstring for the load_ms fairness rationale.
    """
    if not images:
        raise ValueError("benchmark_cpu_basic_enhanced_pipelines requires at least one image.")
    shape = images[0].shape
    if any(img.shape != shape for img in images):
        raise ValueError("benchmark_cpu_basic_enhanced_pipelines requires all images to share one resolution.")
    if not xray_cuda.cuda_available():
        raise RuntimeError("benchmark_cpu_basic_enhanced_pipelines requires a usable CUDA device.")

    n = len(images)

    # -- CPU --
    cpu_times: List[float] = []
    for _ in range(warmup_runs + measurement_runs):
        start = time.perf_counter()
        cpu_outputs = [_cpu_full_pipeline(img, config) for img in images]
        cpu_times.append((time.perf_counter() - start) * 1000.0)
    cpu_processing_stats = Stats.from_values(cpu_times[warmup_runs:])
    cpu_total_ms = load_ms + cpu_processing_stats.mean

    # -- Basic CUDA --
    for _ in range(warmup_runs):
        run_basic_cuda_pipeline(images, config)
    basic_runs = [run_basic_cuda_pipeline(images, config) for _ in range(measurement_runs)]
    basic_outputs_last, _basic_timings = basic_runs[-1]

    def _optional_stats(runs, attr: str) -> Optional[Stats]:
        values = [getattr(t, attr) for _out, t in runs]
        if values[0] is None:
            return None
        return Stats.from_values(values)

    basic_h2d_stats = Stats.from_values([t.h2d_ms for _out, t in basic_runs])
    basic_compute_stats = Stats.from_values([t.compute_ms for _out, t in basic_runs])
    basic_d2h_stats = Stats.from_values([t.d2h_ms for _out, t in basic_runs])
    basic_total_ms = load_ms + basic_h2d_stats.mean + basic_compute_stats.mean + basic_d2h_stats.mean

    # -- Enhanced CUDA --
    for _ in range(warmup_runs):
        run_enhanced_cuda_pipeline(images, config)
    enhanced_runs = [run_enhanced_cuda_pipeline(images, config) for _ in range(measurement_runs)]
    enhanced_outputs_last, _enhanced_timings = enhanced_runs[-1]

    enhanced_h2d_stats = Stats.from_values([t.h2d_ms for _out, t in enhanced_runs])
    enhanced_compute_stats = Stats.from_values([t.compute_ms for _out, t in enhanced_runs])
    enhanced_d2h_stats = Stats.from_values([t.d2h_ms for _out, t in enhanced_runs])
    enhanced_total_ms = load_ms + enhanced_h2d_stats.mean + enhanced_compute_stats.mean + enhanced_d2h_stats.mean

    # -- correctness (computed once per pair, not per measurement run) --
    correctness_basic_vs_cpu = CorrectnessMetrics.compare(cpu_outputs, basic_outputs_last)
    correctness_enhanced_vs_cpu = CorrectnessMetrics.compare(cpu_outputs, enhanced_outputs_last)
    correctness_enhanced_vs_basic = CorrectnessMetrics.compare(list(basic_outputs_last), enhanced_outputs_last)

    cpu_images_per_second = n / (cpu_total_ms / 1000.0) if cpu_total_ms > 0 else 0.0
    basic_images_per_second = n / (basic_total_ms / 1000.0) if basic_total_ms > 0 else 0.0
    enhanced_images_per_second = n / (enhanced_total_ms / 1000.0) if enhanced_total_ms > 0 else 0.0

    try:
        device_info = xray_cuda.device_info()
    except Exception:
        device_info = {}

    return ThreeWayPipelineBenchmarkResult(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        gpu_name=_gpu_name(),
        driver_version=device_info.get("driver_version"),
        cuda_toolkit_version=device_info.get("runtime_version"),
        dataset_fingerprint=dataset_fingerprint,
        seed=seed,
        selected_image_count=n,
        resolution=(shape[0], shape[1]),
        requested_batch_size=requested_batch_size,
        effective_gpu_batch_size=n,
        filter_config=asdict(config),
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs,
        cpu_load_ms=load_ms,
        cpu_processing_ms=cpu_processing_stats,
        cpu_total_ms=cpu_total_ms,
        basic_h2d_ms=basic_h2d_stats,
        basic_gaussian_ms=_optional_stats(basic_runs, "gaussian_ms"),
        basic_median_ms=_optional_stats(basic_runs, "median_ms"),
        basic_sobel_ms=_optional_stats(basic_runs, "sobel_ms"),
        basic_laplacian_ms=_optional_stats(basic_runs, "laplacian_ms"),
        basic_threshold_ms=_optional_stats(basic_runs, "threshold_ms"),
        basic_compute_ms=basic_compute_stats,
        basic_d2h_ms=basic_d2h_stats,
        basic_total_ms=basic_total_ms,
        enhanced_h2d_ms=enhanced_h2d_stats,
        enhanced_gaussian_ms=_optional_stats(enhanced_runs, "gaussian_ms"),
        enhanced_median_ms=_optional_stats(enhanced_runs, "median_ms"),
        enhanced_sobel_ms=_optional_stats(enhanced_runs, "sobel_ms"),
        enhanced_laplacian_ms=_optional_stats(enhanced_runs, "laplacian_ms"),
        enhanced_threshold_ms=_optional_stats(enhanced_runs, "threshold_ms"),
        enhanced_compute_ms=enhanced_compute_stats,
        enhanced_d2h_ms=enhanced_d2h_stats,
        enhanced_total_ms=enhanced_total_ms,
        cpu_images_per_second=cpu_images_per_second,
        basic_images_per_second=basic_images_per_second,
        enhanced_images_per_second=enhanced_images_per_second,
        basic_speedup_vs_cpu=(cpu_total_ms / basic_total_ms) if basic_total_ms > 0 else 0.0,
        enhanced_speedup_vs_cpu=(cpu_total_ms / enhanced_total_ms) if enhanced_total_ms > 0 else 0.0,
        enhanced_speedup_vs_basic=(basic_total_ms / enhanced_total_ms) if enhanced_total_ms > 0 else 0.0,
        correctness_basic_vs_cpu=correctness_basic_vs_cpu,
        correctness_enhanced_vs_cpu=correctness_enhanced_vs_cpu,
        correctness_enhanced_vs_basic=correctness_enhanced_vs_basic,
    )
