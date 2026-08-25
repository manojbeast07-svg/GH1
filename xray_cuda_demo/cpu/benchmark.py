"""CPU timing/benchmark aggregation.

Two distinct, never-mixed timing modes (spec section 17):

  "processing"  -- default. Every image in the selection is loaded once,
                   up front, outside the timed region. Only
                   run_cpu_pipeline() calls are timed. This isolates
                   "image processing only", matching the per-filter
                   TimingResult semantics used everywhere else in this
                   project.
  "end_to_end"  -- each measurement run reloads every image from disk
                   inside the timed region (load_image() + pipeline).
                   Useful later for understanding disk I/O's share of
                   total latency, but its numbers are not comparable to
                   "processing" mode numbers and a single BenchmarkResult
                   only ever reports one mode (see `timing_mode`).

Neither mode uses multiprocessing/threading -- see cpu/pipeline.py for
why the CPU reference is deliberately not optimized.
"""

from __future__ import annotations

import platform
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import cv2

from cpu.filters import FilterConfig
from cpu.pipeline import run_cpu_pipeline
from pipeline.dataset import ImageSelection
from pipeline.image_loader import load_image
from logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_WARMUP_RUNS = 2
DEFAULT_MEASUREMENT_RUNS = 5


@dataclass
class AggregatedTiming:
    """Statistics over `measurement_runs` repeated executions of a single
    image through the pipeline (see benchmark_cpu_pipeline)."""

    measurement_runs: int
    total_mean_ms: float
    total_min_ms: float
    total_max_ms: float
    total_std_ms: float
    per_filter_mean_ms: Dict[str, float]


def benchmark_cpu_pipeline(
    image,
    config: FilterConfig,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    measurement_runs: int = DEFAULT_MEASUREMENT_RUNS,
) -> AggregatedTiming:
    """Run the pipeline on one already-loaded image repeatedly, discarding
    `warmup_runs` untimed passes, then aggregating `measurement_runs`
    timed passes into mean/min/max/std (total) and per-filter means.
    """
    if warmup_runs < 0:
        raise ValueError(f"warmup_runs must be >= 0, got {warmup_runs}")
    if measurement_runs <= 0:
        raise ValueError(f"measurement_runs must be positive, got {measurement_runs}")

    for _ in range(warmup_runs):
        run_cpu_pipeline(image, config)

    timings = [run_cpu_pipeline(image, config)[1] for _ in range(measurement_runs)]

    totals = [t.total_ms for t in timings]
    filter_sums: Dict[str, float] = {}
    filter_counts: Dict[str, int] = {}
    for t in timings:
        for name, value in t.as_dict().items():
            if value is not None:
                filter_sums[name] = filter_sums.get(name, 0.0) + value
                filter_counts[name] = filter_counts.get(name, 0) + 1

    return AggregatedTiming(
        measurement_runs=measurement_runs,
        total_mean_ms=statistics.mean(totals),
        total_min_ms=min(totals),
        total_max_ms=max(totals),
        total_std_ms=statistics.stdev(totals) if len(totals) > 1 else 0.0,
        per_filter_mean_ms={name: filter_sums[name] / filter_counts[name] for name in filter_sums},
    )


@dataclass
class BenchmarkMetadata:
    """Everything needed to know whether two benchmark runs are actually
    comparable (spec section 31)."""

    dataset_fingerprint: Optional[str]
    selection_mode: str
    selection_seed: Optional[int]
    selection_batch_size: Optional[int]
    selected_relative_paths: List[str]
    filter_config: dict
    cpu_model: str
    python_version: str
    opencv_version: str
    timestamp_utc: str
    warmup_runs: int
    measurement_runs: int
    timing_mode: str


@dataclass
class BenchmarkResult:
    metadata: BenchmarkMetadata
    num_images: int
    total_time_ms: float  # mean, across measurement runs, of one full-batch pass
    avg_time_per_image_ms: float
    throughput_images_per_sec: float
    per_filter_mean_ms: Dict[str, float]
    per_run_total_ms: List[float]

    def to_dict(self) -> dict:
        return asdict(self)


def _cpu_model() -> str:
    model = platform.processor()
    return model if model else f"{platform.system()} {platform.machine()}"


def run_selection_benchmark(
    selection: ImageSelection,
    config: FilterConfig,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    measurement_runs: int = DEFAULT_MEASUREMENT_RUNS,
    timing_mode: str = "processing",
    dataset_fingerprint: Optional[str] = None,
) -> BenchmarkResult:
    """Benchmark the CPU pipeline over an entire selection.

    `timing_mode="processing"` (default) preloads every image once, then
    times `measurement_runs` full passes of run_cpu_pipeline() only.
    `timing_mode="end_to_end"` times load_image() + run_cpu_pipeline()
    together, reloading from disk on every run.
    """
    if timing_mode not in ("processing", "end_to_end"):
        raise ValueError(f"Unknown timing_mode: {timing_mode!r} (expected 'processing' or 'end_to_end')")
    if len(selection) == 0:
        raise ValueError("Cannot benchmark an empty selection.")

    items = selection.items
    logger.info(
        "Starting CPU benchmark: %d images, warmup=%d, measurement=%d, mode=%s",
        len(items),
        warmup_runs,
        measurement_runs,
        timing_mode,
    )

    preloaded = None
    if timing_mode == "processing":
        preloaded = [(item, load_image(item.absolute_path)) for item in items]

    def run_once() -> list:
        run_timings = []
        if timing_mode == "processing":
            for _item, image in preloaded:
                _, timing = run_cpu_pipeline(image, config)
                run_timings.append(timing)
        else:
            for item in items:
                image = load_image(item.absolute_path)
                _, timing = run_cpu_pipeline(image, config)
                run_timings.append(timing)
        return run_timings

    for _ in range(warmup_runs):
        run_once()

    per_run_total_ms: List[float] = []
    filter_sums: Dict[str, float] = {}
    filter_counts: Dict[str, int] = {}
    for _ in range(measurement_runs):
        run_timings = run_once()
        per_run_total_ms.append(sum(t.total_ms for t in run_timings))
        for t in run_timings:
            for name, value in t.as_dict().items():
                if value is not None:
                    filter_sums[name] = filter_sums.get(name, 0.0) + value
                    filter_counts[name] = filter_counts.get(name, 0) + 1

    total_time_ms = statistics.mean(per_run_total_ms)
    avg_time_per_image_ms = total_time_ms / len(items)
    throughput = (len(items) / (total_time_ms / 1000.0)) if total_time_ms > 0 else 0.0
    per_filter_mean_ms = {name: filter_sums[name] / filter_counts[name] for name in filter_sums}

    metadata = BenchmarkMetadata(
        dataset_fingerprint=dataset_fingerprint,
        selection_mode=selection.mode,
        selection_seed=selection.seed,
        selection_batch_size=selection.requested_batch_size,
        selected_relative_paths=[item.relative_path for item in items],
        filter_config=asdict(config),
        cpu_model=_cpu_model(),
        python_version=platform.python_version(),
        opencv_version=cv2.__version__,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs,
        timing_mode=timing_mode,
    )

    logger.info(
        "CPU benchmark complete: %d images, %.3f ms/image, %.2f images/sec",
        len(items),
        avg_time_per_image_ms,
        throughput,
    )

    return BenchmarkResult(
        metadata=metadata,
        num_images=len(items),
        total_time_ms=total_time_ms,
        avg_time_per_image_ms=avg_time_per_image_ms,
        throughput_images_per_sec=throughput,
        per_filter_mean_ms=per_filter_mean_ms,
        per_run_total_ms=per_run_total_ms,
    )
