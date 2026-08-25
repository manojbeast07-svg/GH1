"""Final reproducible benchmark & experiment framework (Section 11).

Builds on Section 10's `cuda.benchmark.benchmark_cpu_basic_enhanced_pipelines`
(which remains unchanged and is still the function Section 10's own
script uses) with the stricter requirements Section 11 adds: raw
per-run measurements (not just aggregates), a dataset+environment+
configuration fingerprint on every result, deterministic benchmark IDs,
four explicitly separated timing modes, batch-size and resolution
sweeps, per-filter Amdahl-style contribution analysis, a 3-level
correctness benchmark, on-disk manifests, and loader APIs the future
Streamlit dashboard (Section 12) will read directly instead of
rerunning benchmarks at startup.

--------------------------------------------------------------------------
The four timing modes (spec item 5) and where they come from
--------------------------------------------------------------------------
run_basic_cuda_pipeline()/run_enhanced_cuda_pipeline() already return a
PipelineTiming with h2d_ms, one field per kernel, d2h_ms, compute_ms
(=sum of kernel fields), and total_ms (=h2d_ms+compute_ms+d2h_ms) -- all
measured with CUDA events, no Python-side overhead. That maps directly
onto three of the four modes with no new instrumentation needed:

  Mode 1 (Kernel only)   = compute_ms                (GPU only; CPU has no equivalent)
  Mode 2 (GPU processing)= total_ms  (= H2D+kernels+D2H)
  Mode 3 (Pipeline)      = GPU: total_ms: CPU: processing_ms alone (no disk load)
  Mode 4 (End-to-end)    = disk load_ms + Mode 3 (both CPU and GPU)

Mode 3 and Mode 2 coincide for the GPU arms because
run_basic_cuda_pipeline()/run_enhanced_cuda_pipeline() never touch disk
-- images are already loaded by the caller. Mode 3 vs Mode 4 only
differs by whether `load_ms` is added, exactly Section 10's `load_ms`
convention.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import xray_cuda
from cpu.filters import FilterConfig, apply_gaussian, apply_laplacian, apply_median, apply_sobel, apply_threshold
from cuda.benchmark import CorrectnessMetrics
from cuda.gaussian import gaussian_kernel_1d, gaussian_kernel_2d
from cuda.gaussian import variant_to_int as gaussian_variant_to_int
from cuda.laplacian import laplacian_kernel_2d, laplacian_variant_to_int
from cuda.median import median_variant_to_int
from cuda.pipeline import run_basic_cuda_pipeline, run_enhanced_cuda_pipeline
from cuda.sobel import mode_to_int as sobel_mode_to_int
from cuda.sobel import sobel_variant_to_int
from cuda.threshold import threshold_variant_to_int
from pipeline.dataset import DatasetManager, ImageSelection
from pipeline.environment import EnvironmentFingerprint, get_environment_fingerprint
from pipeline.image_loader import load_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "benchmark_results"

RESULT_SUBDIRS = (
    "raw/cpu", "raw/basic_cuda", "raw/enhanced_cuda",
    "aggregated", "batch_sweeps", "resolution_sweeps",
    "per_filter", "correctness", "manifests", "summary",
)


def ensure_results_dirs(root: Path = DEFAULT_RESULTS_ROOT) -> None:
    for sub in RESULT_SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)


# -- benchmark IDs (spec item 30) -----------------------------------------------------


def generate_benchmark_id(
    seed: Optional[int], n_images: int, width: int, height: int,
    when: Optional[datetime] = None, root: Path = DEFAULT_RESULTS_ROOT,
) -> str:
    """Deterministic-shaped, timestamp-unique ID, e.g.
    '20260824_111500_seed42_batch125_224x224'. Never overwrites earlier
    data -- if the exact same second/config collides (extremely
    unlikely, but checked), a numeric suffix is appended."""
    when = when or datetime.now(timezone.utc)
    seed_part = f"seed{seed}" if seed is not None else "seedNone"
    base = f"{when.strftime('%Y%m%d_%H%M%S')}_{seed_part}_batch{n_images}_{width}x{height}"
    candidate = base
    suffix = 2
    while (root / "manifests" / f"{candidate}.json").exists():
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


# -- aggregation (spec items 13-14, 27) -----------------------------------------------------


@dataclass
class AggregatedStat:
    """mean/median/min/max/std, plus coefficient of variation (item 13).
    Built from ALL raw values -- no outliers silently dropped (item 14)."""

    mean: float
    median: float
    min: float
    max: float
    std: float
    coefficient_of_variation: Optional[float]  # std/mean; None if mean==0
    n: int

    @classmethod
    def from_values(cls, values: List[float]) -> "AggregatedStat":
        if not values:
            raise ValueError("AggregatedStat.from_values requires at least one value.")
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return cls(
            mean=mean, median=statistics.median(values), min=min(values), max=max(values),
            std=std, coefficient_of_variation=(std / mean) if mean != 0 else None, n=len(values),
        )

    def to_dict(self) -> dict:
        return asdict(self)


# -- configuration fingerprint (spec items 9, 11, 20) -----------------------------------------------------


@dataclass
class CudaImplementationConfig:
    """CUDA *implementation* parameters -- never mixed with FilterConfig's
    image-processing parameters (spec item 11/20 -- same separation
    PipelineConfig enforces in cuda/pipeline.py)."""

    gaussian_variant: str
    gaussian_block: Tuple[int, int]
    median_variant: str
    median_block: Tuple[int, int]
    sobel_variant: str
    sobel_block: Tuple[int, int]
    laplacian_variant: str
    laplacian_block: Tuple[int, int]
    threshold_variant: str
    threshold_block: Tuple[int, int]

    @classmethod
    def enhanced_defaults(cls) -> "CudaImplementationConfig":
        return cls(
            gaussian_variant="specialized", gaussian_block=(16, 16),
            median_variant="network3x3", median_block=(16, 16),
            sobel_variant="specialized", sobel_block=(16, 16),
            laplacian_variant="specialized", laplacian_block=(16, 16),
            threshold_variant="vectorized", threshold_block=(16, 16),
        )

    def to_dict(self) -> dict:
        return asdict(self)


# -- validation / sanity checks (spec items 33-34) -----------------------------------------------------


def validate_selection_paths(selection: ImageSelection) -> List[str]:
    """Returns a list of problem descriptions (empty = all good).
    Never raises -- callers decide whether to mark a run INVALID."""
    problems = []
    for item in selection.items:
        if not item.absolute_path.exists():
            problems.append(f"selected path does not exist: {item.absolute_path}")
    return problems


def sanity_check_timing(
    n_images: int, cpu_total_ms: float, basic_total_ms: float, enhanced_total_ms: float,
) -> List[str]:
    """Checks 1-3 from spec item 34. Returns problem descriptions."""
    problems = []
    if n_images <= 0:
        problems.append(f"image count must be > 0, got {n_images}")
    for label, value in [("cpu_total_ms", cpu_total_ms), ("basic_total_ms", basic_total_ms),
                          ("enhanced_total_ms", enhanced_total_ms)]:
        if value < 0:
            problems.append(f"{label} must be >= 0, got {value}")
    return problems


# -- manifest (spec items 7-9, 29) -----------------------------------------------------


@dataclass
class BenchmarkManifest:
    benchmark_id: str
    timestamp_utc: str
    status: str  # "VALID" or "INVALID"
    validation_notes: List[str]

    dataset_fingerprint: str
    dataset_path: str
    valid_image_count: int

    environment: dict  # EnvironmentFingerprint.to_dict()

    seed: Optional[int]
    requested_batch_size: Optional[int]
    selected_image_count: int
    selected_relative_paths: List[str]

    resolution: Tuple[int, int]
    filter_config: dict  # image-processing parameters
    basic_cuda_config: dict  # CudaImplementationConfig.to_dict(), all "basic"
    enhanced_cuda_config: dict  # CudaImplementationConfig.to_dict(), measured-best

    warmup_runs: int
    measurement_runs: int

    result_file_paths: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def _basic_cuda_config() -> CudaImplementationConfig:
    return CudaImplementationConfig(
        gaussian_variant="basic", gaussian_block=(16, 16),
        median_variant="basic", median_block=(16, 16),
        sobel_variant="basic", sobel_block=(16, 16),
        laplacian_variant="basic", laplacian_block=(16, 16),
        threshold_variant="basic", threshold_block=(16, 16),
    )


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


# -- Mode 1-4 timing container -----------------------------------------------------


@dataclass
class FourModeMetrics:
    """Mode 1 (kernel-only) is None for CPU -- there is no CUDA-event
    kernel concept on the CPU arm."""

    mode1_kernel_only_ms: Optional[AggregatedStat]
    mode2_gpu_processing_ms: Optional[AggregatedStat]  # None for CPU
    mode3_pipeline_ms: AggregatedStat  # CPU: processing only; GPU: H2D+kernels+D2H (same as mode2)
    mode4_end_to_end_ms: AggregatedStat  # + one-time disk load_ms

    def to_dict(self) -> dict:
        return {
            "mode1_kernel_only_ms": self.mode1_kernel_only_ms.to_dict() if self.mode1_kernel_only_ms else None,
            "mode2_gpu_processing_ms": self.mode2_gpu_processing_ms.to_dict() if self.mode2_gpu_processing_ms else None,
            "mode3_pipeline_ms": self.mode3_pipeline_ms.to_dict(),
            "mode4_end_to_end_ms": self.mode4_end_to_end_ms.to_dict(),
        }


@dataclass
class CanonicalBenchmarkResult:
    manifest: BenchmarkManifest
    cpu: FourModeMetrics
    basic: FourModeMetrics
    enhanced: FourModeMetrics
    per_stage_basic_ms: Dict[str, AggregatedStat]     # gaussian/median/sobel/laplacian/threshold
    per_stage_enhanced_ms: Dict[str, AggregatedStat]
    cpu_images_per_second: float
    basic_images_per_second: float
    enhanced_images_per_second: float
    basic_speedup_vs_cpu: float
    enhanced_speedup_vs_cpu: float
    enhanced_speedup_vs_basic_compute_only: float
    enhanced_speedup_vs_basic_end_to_end: float
    correctness_basic_vs_cpu: dict
    correctness_enhanced_vs_cpu: dict
    correctness_enhanced_vs_basic: dict

    def to_dict(self) -> dict:
        return {
            "manifest": self.manifest.to_dict(),
            "cpu": self.cpu.to_dict(),
            "basic": self.basic.to_dict(),
            "enhanced": self.enhanced.to_dict(),
            "per_stage_basic_ms": {k: v.to_dict() for k, v in self.per_stage_basic_ms.items()},
            "per_stage_enhanced_ms": {k: v.to_dict() for k, v in self.per_stage_enhanced_ms.items()},
            "cpu_images_per_second": self.cpu_images_per_second,
            "basic_images_per_second": self.basic_images_per_second,
            "enhanced_images_per_second": self.enhanced_images_per_second,
            "basic_speedup_vs_cpu": self.basic_speedup_vs_cpu,
            "enhanced_speedup_vs_cpu": self.enhanced_speedup_vs_cpu,
            "enhanced_speedup_vs_basic_compute_only": self.enhanced_speedup_vs_basic_compute_only,
            "enhanced_speedup_vs_basic_end_to_end": self.enhanced_speedup_vs_basic_end_to_end,
            "correctness_basic_vs_cpu": self.correctness_basic_vs_cpu,
            "correctness_enhanced_vs_cpu": self.correctness_enhanced_vs_cpu,
            "correctness_enhanced_vs_basic": self.correctness_enhanced_vs_basic,
        }


def _correctness_to_dict(cpu_batch: List[np.ndarray], gpu_batch: np.ndarray) -> dict:
    return asdict(CorrectnessMetrics.compare(cpu_batch, gpu_batch))


def run_canonical_benchmark(
    images: List[np.ndarray],
    selection: ImageSelection,
    filter_config: FilterConfig,
    dataset_fingerprint: str,
    dataset_path: str,
    valid_image_count: int,
    warmup_runs: int = 2,
    measurement_runs: int = 20,
    load_ms: float = 0.0,
    results_root: Path = DEFAULT_RESULTS_ROOT,
) -> CanonicalBenchmarkResult:
    """The Section 11 canonical benchmark: CPU vs Basic CUDA vs Enhanced
    CUDA on the SAME `images` (spec item 6 -- never independently
    sampled per implementation), capturing raw per-run measurements
    (spec item 26, not just aggregates) for all four timing modes, then
    writing raw/aggregated/manifest JSON under `results_root` and
    returning the full in-memory result.
    """
    if not images:
        raise ValueError("run_canonical_benchmark requires at least one image.")
    shape = images[0].shape
    if any(img.shape != shape for img in images):
        raise ValueError("run_canonical_benchmark requires all images to share one resolution.")
    if not xray_cuda.cuda_available():
        raise RuntimeError("run_canonical_benchmark requires a usable CUDA device.")

    ensure_results_dirs(results_root)
    height, width = shape
    n = len(images)
    benchmark_id = generate_benchmark_id(selection.seed, n, width, height, root=results_root)

    validation_notes = validate_selection_paths(selection)

    # -- CPU: raw per-run capture --
    cpu_raw: List[dict] = []
    cpu_outputs = None
    for i in range(warmup_runs + measurement_runs):
        start = time.perf_counter()
        outs = [_cpu_full_pipeline(img, filter_config) for img in images]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if i >= warmup_runs:
            cpu_raw.append({"run": i - warmup_runs, "processing_ms": elapsed_ms})
        cpu_outputs = outs

    # -- Basic CUDA: raw per-run capture --
    for _ in range(warmup_runs):
        run_basic_cuda_pipeline(images, filter_config)
    basic_raw: List[dict] = []
    basic_outputs = None
    for i in range(measurement_runs):
        out, t = run_basic_cuda_pipeline(images, filter_config)
        basic_raw.append({
            "run": i, "h2d_ms": t.h2d_ms, "gaussian_ms": t.gaussian_ms, "median_ms": t.median_ms,
            "sobel_ms": t.sobel_ms, "laplacian_ms": t.laplacian_ms, "threshold_ms": t.threshold_ms,
            "d2h_ms": t.d2h_ms, "compute_ms": t.compute_ms, "total_ms": t.total_ms,
        })
        basic_outputs = out

    # -- Enhanced CUDA: raw per-run capture --
    for _ in range(warmup_runs):
        run_enhanced_cuda_pipeline(images, filter_config)
    enhanced_raw: List[dict] = []
    enhanced_outputs = None
    for i in range(measurement_runs):
        out, t = run_enhanced_cuda_pipeline(images, filter_config)
        enhanced_raw.append({
            "run": i, "h2d_ms": t.h2d_ms, "gaussian_ms": t.gaussian_ms, "median_ms": t.median_ms,
            "sobel_ms": t.sobel_ms, "laplacian_ms": t.laplacian_ms, "threshold_ms": t.threshold_ms,
            "d2h_ms": t.d2h_ms, "compute_ms": t.compute_ms, "total_ms": t.total_ms,
        })
        enhanced_outputs = out

    # -- aggregate into the four modes --
    cpu_processing_stat = AggregatedStat.from_values([r["processing_ms"] for r in cpu_raw])
    cpu_e2e_values = [r["processing_ms"] + load_ms for r in cpu_raw]
    cpu_modes = FourModeMetrics(
        mode1_kernel_only_ms=None, mode2_gpu_processing_ms=None,
        mode3_pipeline_ms=cpu_processing_stat,
        mode4_end_to_end_ms=AggregatedStat.from_values(cpu_e2e_values),
    )

    def _gpu_modes(raw: List[dict]) -> FourModeMetrics:
        compute_stat = AggregatedStat.from_values([r["compute_ms"] for r in raw])
        total_stat = AggregatedStat.from_values([r["total_ms"] for r in raw])
        e2e_stat = AggregatedStat.from_values([r["total_ms"] + load_ms for r in raw])
        return FourModeMetrics(
            mode1_kernel_only_ms=compute_stat, mode2_gpu_processing_ms=total_stat,
            mode3_pipeline_ms=total_stat, mode4_end_to_end_ms=e2e_stat,
        )

    basic_modes = _gpu_modes(basic_raw)
    enhanced_modes = _gpu_modes(enhanced_raw)

    def _per_stage(raw: List[dict]) -> Dict[str, AggregatedStat]:
        stages = {}
        for stage in ("gaussian", "median", "sobel", "laplacian", "threshold"):
            values = [r[f"{stage}_ms"] for r in raw if r[f"{stage}_ms"] is not None]
            if values:
                stages[stage] = AggregatedStat.from_values(values)
        return stages

    per_stage_basic = _per_stage(basic_raw)
    per_stage_enhanced = _per_stage(enhanced_raw)

    cpu_e2e_mean = cpu_modes.mode4_end_to_end_ms.mean
    basic_e2e_mean = basic_modes.mode4_end_to_end_ms.mean
    enhanced_e2e_mean = enhanced_modes.mode4_end_to_end_ms.mean

    cpu_ips = n / (cpu_e2e_mean / 1000.0) if cpu_e2e_mean > 0 else 0.0
    basic_ips = n / (basic_e2e_mean / 1000.0) if basic_e2e_mean > 0 else 0.0
    enhanced_ips = n / (enhanced_e2e_mean / 1000.0) if enhanced_e2e_mean > 0 else 0.0

    correctness_basic_vs_cpu = _correctness_to_dict(cpu_outputs, basic_outputs)
    correctness_enhanced_vs_cpu = _correctness_to_dict(cpu_outputs, enhanced_outputs)
    correctness_enhanced_vs_basic = _correctness_to_dict(list(basic_outputs), enhanced_outputs)

    timing_problems = sanity_check_timing(n, cpu_e2e_mean, basic_e2e_mean, enhanced_e2e_mean)
    validation_notes = validation_notes + timing_problems
    status = "VALID" if not validation_notes else "INVALID"

    manifest = BenchmarkManifest(
        benchmark_id=benchmark_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        status=status,
        validation_notes=validation_notes,
        dataset_fingerprint=dataset_fingerprint,
        dataset_path=dataset_path,
        valid_image_count=valid_image_count,
        environment=get_environment_fingerprint().to_dict(),
        seed=selection.seed,
        requested_batch_size=selection.requested_batch_size,
        selected_image_count=n,
        selected_relative_paths=[item.relative_path for item in selection.items[:n]],
        resolution=(height, width),
        filter_config=asdict(filter_config),
        basic_cuda_config=_basic_cuda_config().to_dict(),
        enhanced_cuda_config=CudaImplementationConfig.enhanced_defaults().to_dict(),
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs,
    )

    result = CanonicalBenchmarkResult(
        manifest=manifest,
        cpu=cpu_modes, basic=basic_modes, enhanced=enhanced_modes,
        per_stage_basic_ms=per_stage_basic, per_stage_enhanced_ms=per_stage_enhanced,
        cpu_images_per_second=cpu_ips, basic_images_per_second=basic_ips, enhanced_images_per_second=enhanced_ips,
        basic_speedup_vs_cpu=(cpu_e2e_mean / basic_e2e_mean) if basic_e2e_mean > 0 else 0.0,
        enhanced_speedup_vs_cpu=(cpu_e2e_mean / enhanced_e2e_mean) if enhanced_e2e_mean > 0 else 0.0,
        enhanced_speedup_vs_basic_compute_only=(
            (basic_modes.mode1_kernel_only_ms.mean / enhanced_modes.mode1_kernel_only_ms.mean)
            if enhanced_modes.mode1_kernel_only_ms.mean > 0 else 0.0
        ),
        enhanced_speedup_vs_basic_end_to_end=(basic_e2e_mean / enhanced_e2e_mean) if enhanced_e2e_mean > 0 else 0.0,
        correctness_basic_vs_cpu=correctness_basic_vs_cpu,
        correctness_enhanced_vs_cpu=correctness_enhanced_vs_cpu,
        correctness_enhanced_vs_basic=correctness_enhanced_vs_basic,
    )

    # -- write raw + aggregated + manifest --
    raw_cpu_path = results_root / "raw" / "cpu" / f"{benchmark_id}.json"
    raw_basic_path = results_root / "raw" / "basic_cuda" / f"{benchmark_id}.json"
    raw_enhanced_path = results_root / "raw" / "enhanced_cuda" / f"{benchmark_id}.json"
    aggregated_path = results_root / "aggregated" / f"{benchmark_id}.json"
    manifest_path = results_root / "manifests" / f"{benchmark_id}.json"

    _save_json(raw_cpu_path, {"benchmark_id": benchmark_id, "runs": cpu_raw})
    _save_json(raw_basic_path, {"benchmark_id": benchmark_id, "runs": basic_raw})
    _save_json(raw_enhanced_path, {"benchmark_id": benchmark_id, "runs": enhanced_raw})
    _save_json(aggregated_path, result.to_dict())

    manifest.result_file_paths = {
        "raw_cpu": str(raw_cpu_path.relative_to(results_root)),
        "raw_basic_cuda": str(raw_basic_path.relative_to(results_root)),
        "raw_enhanced_cuda": str(raw_enhanced_path.relative_to(results_root)),
        "aggregated": str(aggregated_path.relative_to(results_root)),
    }
    _save_json(manifest_path, manifest.to_dict())
    _save_json(aggregated_path, result.to_dict())  # rewrite with final manifest.result_file_paths included

    return result


# -- batch-size sweep (spec items 15-16) -----------------------------------------------------


def run_batch_sweep(
    all_images: List[np.ndarray],
    filter_config: FilterConfig,
    batch_sizes: List[int] = (1, 8, 16, 32, 64, 128, 256, 512),
    warmup_runs: int = 3,
    measurement_runs: int = 10,
    seed: Optional[int] = None,
    results_root: Path = DEFAULT_RESULTS_ROOT,
) -> dict:
    """For each batch size (capped at len(all_images)), slices the FRONT
    of `all_images` (same images reused across batch sizes -- not
    independently resampled, spec item 6) and measures CPU/Basic/
    Enhanced totals, throughput, and speedups. Produces the
    machine-readable data for Plots A-D (item 16) without rendering
    anything -- that's Section 12's job.
    """
    if not all_images:
        raise ValueError("run_batch_sweep requires at least one image.")
    shape = all_images[0].shape
    height, width = shape

    rows = []
    for batch_size in batch_sizes:
        n = min(batch_size, len(all_images))
        if n <= 0:
            continue
        images = all_images[:n]

        for _ in range(warmup_runs):
            _cpu_full_pipeline(images[0], filter_config)
            run_basic_cuda_pipeline(images, filter_config)
            run_enhanced_cuda_pipeline(images, filter_config)

        cpu_times = []
        for _ in range(measurement_runs):
            start = time.perf_counter()
            for img in images:
                _cpu_full_pipeline(img, filter_config)
            cpu_times.append((time.perf_counter() - start) * 1000.0)
        basic_times = [run_basic_cuda_pipeline(images, filter_config)[1].total_ms for _ in range(measurement_runs)]
        enhanced_times = [run_enhanced_cuda_pipeline(images, filter_config)[1].total_ms for _ in range(measurement_runs)]

        cpu_stat = AggregatedStat.from_values(cpu_times)
        basic_stat = AggregatedStat.from_values(basic_times)
        enhanced_stat = AggregatedStat.from_values(enhanced_times)

        rows.append({
            "requested_batch_size": batch_size,
            "effective_batch_size": n,
            "cpu_total_ms": cpu_stat.to_dict(),
            "basic_total_ms": basic_stat.to_dict(),
            "enhanced_total_ms": enhanced_stat.to_dict(),
            "cpu_images_per_second": n / (cpu_stat.mean / 1000.0) if cpu_stat.mean > 0 else 0.0,
            "basic_images_per_second": n / (basic_stat.mean / 1000.0) if basic_stat.mean > 0 else 0.0,
            "enhanced_images_per_second": n / (enhanced_stat.mean / 1000.0) if enhanced_stat.mean > 0 else 0.0,
            "basic_speedup_vs_cpu": (cpu_stat.mean / basic_stat.mean) if basic_stat.mean > 0 else 0.0,
            "enhanced_speedup_vs_cpu": (cpu_stat.mean / enhanced_stat.mean) if enhanced_stat.mean > 0 else 0.0,
            "enhanced_vs_basic_gain": (basic_stat.mean / enhanced_stat.mean) if enhanced_stat.mean > 0 else 0.0,
        })

    benchmark_id = generate_benchmark_id(seed, len(all_images), width, height, root=results_root)
    payload = {
        "benchmark_id": benchmark_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "resolution": [height, width],
        "seed": seed,
        "warmup_runs": warmup_runs,
        "measurement_runs": measurement_runs,
        "rows": rows,
    }
    ensure_results_dirs(results_root)
    out_path = results_root / "batch_sweeps" / f"{benchmark_id}.json"
    _save_json(out_path, payload)
    payload["_file_path"] = str(out_path.relative_to(results_root))
    return payload


# -- resolution sweep (spec items 17-18) -----------------------------------------------------


def run_resolution_sweep(
    resolution_groups: Dict[Tuple[int, int], List[np.ndarray]],
    filter_config: FilterConfig,
    warmup_runs: int = 2,
    measurement_runs: int = 10,
    seed: Optional[int] = None,
    results_root: Path = DEFAULT_RESULTS_ROOT,
) -> dict:
    """`resolution_groups` maps (height, width) -> list of already-loaded
    images sharing that resolution (e.g. from group_by_resolution()).
    Reports pixels/image, ms/image, and images/sec per implementation
    for each group, so scaling behavior across the real dataset's mixed
    resolutions is visible (item 18)."""
    rows = []
    for (height, width), images in resolution_groups.items():
        if not images:
            continue
        n = len(images)
        pixels = height * width

        for _ in range(warmup_runs):
            _cpu_full_pipeline(images[0], filter_config)
            run_basic_cuda_pipeline(images, filter_config)
            run_enhanced_cuda_pipeline(images, filter_config)

        cpu_times = []
        for _ in range(measurement_runs):
            start = time.perf_counter()
            for img in images:
                _cpu_full_pipeline(img, filter_config)
            cpu_times.append((time.perf_counter() - start) * 1000.0)
        basic_times = [run_basic_cuda_pipeline(images, filter_config)[1].total_ms for _ in range(measurement_runs)]
        enhanced_times = [run_enhanced_cuda_pipeline(images, filter_config)[1].total_ms for _ in range(measurement_runs)]

        cpu_stat = AggregatedStat.from_values(cpu_times)
        basic_stat = AggregatedStat.from_values(basic_times)
        enhanced_stat = AggregatedStat.from_values(enhanced_times)

        rows.append({
            "width": width, "height": height, "pixel_count": pixels, "image_count": n,
            "cpu_ms_per_image": cpu_stat.mean / n, "basic_ms_per_image": basic_stat.mean / n,
            "enhanced_ms_per_image": enhanced_stat.mean / n,
            "cpu_images_per_second": n / (cpu_stat.mean / 1000.0) if cpu_stat.mean > 0 else 0.0,
            "basic_images_per_second": n / (basic_stat.mean / 1000.0) if basic_stat.mean > 0 else 0.0,
            "enhanced_images_per_second": n / (enhanced_stat.mean / 1000.0) if enhanced_stat.mean > 0 else 0.0,
            "cpu_total_ms": cpu_stat.to_dict(), "basic_total_ms": basic_stat.to_dict(),
            "enhanced_total_ms": enhanced_stat.to_dict(),
        })

    total_images = sum(len(v) for v in resolution_groups.values())
    first_shape = next(iter(resolution_groups.keys()), (0, 0))
    benchmark_id = generate_benchmark_id(seed, total_images, first_shape[1], first_shape[0], root=results_root)
    payload = {
        "benchmark_id": benchmark_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "warmup_runs": warmup_runs,
        "measurement_runs": measurement_runs,
        "rows": sorted(rows, key=lambda r: -r["pixel_count"]),
    }
    ensure_results_dirs(results_root)
    out_path = results_root / "resolution_sweeps" / f"{benchmark_id}.json"
    _save_json(out_path, payload)
    payload["_file_path"] = str(out_path.relative_to(results_root))
    return payload


# -- per-filter benchmark + Amdahl-style contribution analysis (spec items 19-21) -----------------------------------------------------


def run_per_filter_benchmark(
    images: List[np.ndarray],
    filter_config: FilterConfig,
    warmup_runs: int = 5,
    measurement_runs: int = 20,
    seed: Optional[int] = None,
    results_root: Path = DEFAULT_RESULTS_ROOT,
) -> dict:
    """Isolates each filter (all others disabled) using the SAME
    batched-production-pipeline architecture Sections 6-10 established
    as authoritative (never single-image timing) to measure its Basic
    and Enhanced kernel time independently, then computes each filter's
    ACTUAL measured contribution to the total compute-time reduction
    (item 21: `basic_ms - enhanced_ms`, and that as a percentage of the
    total reduction across all five filters) -- not a theoretical
    Amdahl's-Law bound, an empirical one from real measured numbers.
    """
    batch = np.stack(images, axis=0)
    gaussian_coeffs_2d = gaussian_kernel_2d(filter_config.gaussian_kernel_size, filter_config.gaussian_sigma)
    gaussian_coeffs_1d = gaussian_kernel_1d(filter_config.gaussian_kernel_size, filter_config.gaussian_sigma)
    laplacian_coeffs = laplacian_kernel_2d(filter_config.laplacian_kernel_size)
    sobel_mode_int = sobel_mode_to_int(filter_config.sobel_mode)

    def _bench_basic(**enabled) -> AggregatedStat:
        values = []
        for i in range(warmup_runs + measurement_runs):
            r = xray_cuda.run_basic_cuda_pipeline_gpu(
                batch,
                enabled.get("gaussian", False), gaussian_coeffs_2d,
                enabled.get("median", False), filter_config.median_kernel_size,
                enabled.get("sobel", False), sobel_mode_int,
                enabled.get("laplacian", False), laplacian_coeffs, filter_config.laplacian_scale, filter_config.laplacian_delta,
                enabled.get("threshold", False), filter_config.threshold_value, filter_config.threshold_max_value,
            )
            if i >= warmup_runs:
                values.append(r["compute_ms"])
        return AggregatedStat.from_values(values)

    def _bench_enhanced(**enabled) -> AggregatedStat:
        values = []
        for i in range(warmup_runs + measurement_runs):
            r = xray_cuda.run_basic_cuda_pipeline_gpu(
                batch,
                enabled.get("gaussian", False), gaussian_coeffs_2d,
                enabled.get("median", False), filter_config.median_kernel_size,
                enabled.get("sobel", False), sobel_mode_int,
                enabled.get("laplacian", False), laplacian_coeffs, filter_config.laplacian_scale, filter_config.laplacian_delta,
                enabled.get("threshold", False), filter_config.threshold_value, filter_config.threshold_max_value,
                enabled.get("gaussian", False), gaussian_coeffs_1d, gaussian_variant_to_int("specialized"), 16, 16,
                enabled.get("median", False), median_variant_to_int("network3x3"), 16, 16,
                enabled.get("sobel", False), sobel_variant_to_int("specialized"), 16, 16,
                enabled.get("laplacian", False), laplacian_variant_to_int("specialized"), 16, 16,
                enabled.get("threshold", False), threshold_variant_to_int("vectorized"), 16, 16,
            )
            if i >= warmup_runs:
                values.append(r["compute_ms"])
        return AggregatedStat.from_values(values)

    filters = ["gaussian", "median", "sobel", "laplacian", "threshold"]
    basic_stats = {f: _bench_basic(**{f: True}) for f in filters}
    enhanced_stats = {f: _bench_enhanced(**{f: True}) for f in filters}

    total_basic_ms = sum(s.mean for s in basic_stats.values())
    total_enhanced_ms = sum(s.mean for s in enhanced_stats.values())
    total_reduction_ms = total_basic_ms - total_enhanced_ms

    rows = []
    for f in filters:
        basic_ms = basic_stats[f].mean
        enhanced_ms = enhanced_stats[f].mean
        reduction_ms = basic_ms - enhanced_ms
        rows.append({
            "filter": f,
            "basic_kernel_ms": basic_stats[f].to_dict(),
            "enhanced_kernel_ms": enhanced_stats[f].to_dict(),
            "kernel_speedup": (basic_ms / enhanced_ms) if enhanced_ms > 0 else 0.0,
            "basic_pct_of_total_basic_compute": 100.0 * basic_ms / total_basic_ms if total_basic_ms > 0 else 0.0,
            "enhanced_pct_of_total_enhanced_compute": 100.0 * enhanced_ms / total_enhanced_ms if total_enhanced_ms > 0 else 0.0,
            "absolute_reduction_ms": reduction_ms,
            "pct_of_total_compute_reduction": (100.0 * reduction_ms / total_reduction_ms) if total_reduction_ms > 0 else 0.0,
        })

    shape = images[0].shape
    benchmark_id = generate_benchmark_id(seed, len(images), shape[1], shape[0], root=results_root)
    payload = {
        "benchmark_id": benchmark_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "resolution": [shape[0], shape[1]],
        "image_count": len(images),
        "seed": seed,
        "total_basic_compute_ms": total_basic_ms,
        "total_enhanced_compute_ms": total_enhanced_ms,
        "total_reduction_ms": total_reduction_ms,
        "rows": rows,
    }
    ensure_results_dirs(results_root)
    out_path = results_root / "per_filter" / f"{benchmark_id}.json"
    _save_json(out_path, payload)
    payload["_file_path"] = str(out_path.relative_to(results_root))
    return payload


# -- 3-level correctness benchmark (spec items 23-25) -----------------------------------------------------


def run_correctness_benchmark(
    images: List[np.ndarray],
    filter_config: FilterConfig,
    seed: Optional[int] = None,
    results_root: Path = DEFAULT_RESULTS_ROOT,
) -> dict:
    """Level 1 (filter-level, CPU/Basic/Enhanced pairwise) + Level 3
    (final pipeline max_abs_diff/mean_abs_diff/RMSE/differing-pixel %).
    Level 2 (intermediate pipeline inspection) is documented, not run
    here -- it's satisfied by each filter's own GPU-native
    `*_cuda_gpu()`/`*_enhanced_cuda_gpu()` API, exercised directly by
    every Section's pipeline-integration tests (see README)."""
    known_expected = {
        "gaussian": "±1 (measured, justified float non-associativity -- Section 6)",
        "median": "exact (0)", "sobel": "exact (0)",
        "laplacian": "exact (0)", "threshold": "exact (0)",
    }

    from cuda.gaussian import gaussian_cuda, gaussian_enhanced_cuda
    from cuda.laplacian import laplacian_cuda, laplacian_enhanced_cuda
    from cuda.median import median_cuda, median_enhanced_cuda
    from cuda.sobel import sobel_cuda, sobel_enhanced_cuda
    from cuda.threshold import threshold_cuda, threshold_enhanced_cuda

    sample = images[: min(5, len(images))]

    def _filter_level(cpu_fn, basic_fn, enhanced_fn, tolerance: int) -> dict:
        max_diff_basic_cpu = max_diff_enh_cpu = max_diff_enh_basic = 0
        for img in sample:
            cpu_out = cpu_fn(img)
            basic_out = basic_fn(img)
            enhanced_out = enhanced_fn(img)
            max_diff_basic_cpu = max(max_diff_basic_cpu, int(np.abs(cpu_out.astype(np.int16) - basic_out.astype(np.int16)).max()))
            max_diff_enh_cpu = max(max_diff_enh_cpu, int(np.abs(cpu_out.astype(np.int16) - enhanced_out.astype(np.int16)).max()))
            max_diff_enh_basic = max(max_diff_enh_basic, int(np.abs(basic_out.astype(np.int16) - enhanced_out.astype(np.int16)).max()))
        return {
            "cpu_vs_basic_max_abs_diff": max_diff_basic_cpu,
            "cpu_vs_enhanced_max_abs_diff": max_diff_enh_cpu,
            "basic_vs_enhanced_max_abs_diff": max_diff_enh_basic,
            "tolerance": tolerance,
            "pass": max_diff_basic_cpu <= tolerance and max_diff_enh_cpu <= tolerance,
        }

    filter_level = {
        "gaussian": _filter_level(
            lambda img: apply_gaussian(img, filter_config), gaussian_cuda,
            lambda img: gaussian_enhanced_cuda(img, kernel_size=filter_config.gaussian_kernel_size, sigma=filter_config.gaussian_sigma),
            tolerance=1,
        ),
        "median": _filter_level(
            lambda img: apply_median(img, filter_config), median_cuda, median_enhanced_cuda, tolerance=0,
        ),
        "sobel": _filter_level(
            lambda img: apply_sobel(img, filter_config), sobel_cuda, sobel_enhanced_cuda, tolerance=0,
        ),
        "laplacian": _filter_level(
            lambda img: apply_laplacian(img, filter_config), laplacian_cuda, laplacian_enhanced_cuda, tolerance=0,
        ),
        "threshold": _filter_level(
            lambda img: apply_threshold(img, filter_config), threshold_cuda, threshold_enhanced_cuda, tolerance=0,
        ),
    }

    # -- Level 3: final pipeline --
    cpu_outputs = [_cpu_full_pipeline(img, filter_config) for img in images]
    basic_outputs, _t1 = run_basic_cuda_pipeline(images, filter_config)
    enhanced_outputs, _t2 = run_enhanced_cuda_pipeline(images, filter_config)

    pipeline_level = {
        "basic_vs_cpu": _correctness_to_dict(cpu_outputs, basic_outputs),
        "enhanced_vs_cpu": _correctness_to_dict(cpu_outputs, enhanced_outputs),
        "enhanced_vs_basic": _correctness_to_dict(list(basic_outputs), enhanced_outputs),
    }

    established_baseline_pct = 0.0178
    baseline_note = (
        "matches established Section 5 baseline (~0.0178%)"
        if abs(pipeline_level["enhanced_vs_cpu"]["differing_pixel_percentage"] - established_baseline_pct) < 0.01
        else "DIFFERS from established Section 5 baseline (~0.0178%) -- investigate"
    )

    shape = images[0].shape
    benchmark_id = generate_benchmark_id(seed, len(images), shape[1], shape[0], root=results_root)
    payload = {
        "benchmark_id": benchmark_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "resolution": [shape[0], shape[1]],
        "image_count": len(images),
        "filter_level": filter_level,
        "known_expected_differences": known_expected,
        "pipeline_level": pipeline_level,
        "baseline_comparison_note": baseline_note,
        "overall_pass": all(v["pass"] for v in filter_level.values()),
    }
    ensure_results_dirs(results_root)
    out_path = results_root / "correctness" / f"{benchmark_id}.json"
    _save_json(out_path, payload)
    payload["_file_path"] = str(out_path.relative_to(results_root))
    return payload


# -- final machine-readable summary (spec item 39) -----------------------------------------------------


def build_final_summary(
    canonical: CanonicalBenchmarkResult,
    batch_sweep: Optional[dict] = None,
    resolution_sweep: Optional[dict] = None,
    per_filter: Optional[dict] = None,
    correctness: Optional[dict] = None,
    results_root: Path = DEFAULT_RESULTS_ROOT,
) -> dict:
    """Combines every piece Section 11 produced into ONE JSON document,
    exactly the shape spec item 39 asks for -- the file Section 12's
    Streamlit dashboard loads directly at startup instead of rerunning
    anything."""
    summary = {
        "benchmark_id": canonical.manifest.benchmark_id,
        "timestamp_utc": canonical.manifest.timestamp_utc,
        "environment": canonical.manifest.environment,
        "dataset_fingerprint": canonical.manifest.dataset_fingerprint,
        "cpu": canonical.cpu.to_dict(),
        "basic_cuda": canonical.basic.to_dict(),
        "enhanced_cuda": canonical.enhanced.to_dict(),
        "cpu_images_per_second": canonical.cpu_images_per_second,
        "basic_images_per_second": canonical.basic_images_per_second,
        "enhanced_images_per_second": canonical.enhanced_images_per_second,
        "speedups": {
            "basic_vs_cpu": canonical.basic_speedup_vs_cpu,
            "enhanced_vs_cpu": canonical.enhanced_speedup_vs_cpu,
            "enhanced_vs_basic_compute_only": canonical.enhanced_speedup_vs_basic_compute_only,
            "enhanced_vs_basic_end_to_end": canonical.enhanced_speedup_vs_basic_end_to_end,
        },
        "per_filter": per_filter,
        "batch_sweep": batch_sweep,
        "resolution_sweep": resolution_sweep,
        "correctness": {
            "canonical_pipeline": {
                "basic_vs_cpu": canonical.correctness_basic_vs_cpu,
                "enhanced_vs_cpu": canonical.correctness_enhanced_vs_cpu,
                "enhanced_vs_basic": canonical.correctness_enhanced_vs_basic,
            },
            "detailed": correctness,
        },
        "manifest": canonical.manifest.to_dict(),
    }
    ensure_results_dirs(results_root)
    out_path = results_root / "summary" / f"{canonical.manifest.benchmark_id}.json"
    _save_json(out_path, summary)
    latest_path = results_root / "summary" / "latest.json"
    _save_json(latest_path, summary)
    return summary


# -- experiment registry (spec item 38) -----------------------------------------------------


KNOWN_EXPERIMENTS = [
    {"experiment": "Gaussian optimization", "script": "scripts/benchmark_gaussian_optimization.py", "section": 6},
    {"experiment": "Median optimization", "script": "scripts/benchmark_median_optimization.py", "section": 7},
    {"experiment": "Sobel optimization", "script": "scripts/benchmark_sobel_optimization.py", "section": 8},
    {"experiment": "Laplacian optimization", "script": "scripts/benchmark_laplacian_optimization.py", "section": 9},
    {"experiment": "Threshold optimization", "script": "scripts/benchmark_threshold_optimization.py", "section": 10},
    {"experiment": "Final pipeline", "script": "scripts/run_final_benchmark.py", "section": 11},
    {"experiment": "Batch sweep", "script": "scripts/run_final_benchmark.py", "section": 11},
    {"experiment": "Resolution sweep", "script": "scripts/run_final_benchmark.py", "section": 11},
]


def update_experiment_registry(
    experiment: str, benchmark_id: str, result_files: List[str], configuration: dict,
    results_root: Path = DEFAULT_RESULTS_ROOT,
) -> dict:
    """Appends/updates one entry (keyed by `experiment` name) in
    benchmark_results/experiment_registry.json, seeding the file with
    the 8 known experiment types on first use."""
    ensure_results_dirs(results_root)
    registry_path = results_root / "experiment_registry.json"
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as fh:
            registry = json.load(fh)
    else:
        registry = {"entries": [dict(e) for e in KNOWN_EXPERIMENTS]}

    entries = registry["entries"]
    updated = False
    for entry in entries:
        if entry["experiment"] == experiment:
            entry.update({
                "benchmark_id": benchmark_id, "result_files": result_files, "configuration": configuration,
                "last_run_utc": datetime.now(timezone.utc).isoformat(),
            })
            updated = True
            break
    if not updated:
        entries.append({
            "experiment": experiment, "benchmark_id": benchmark_id, "result_files": result_files,
            "configuration": configuration, "last_run_utc": datetime.now(timezone.utc).isoformat(),
        })

    _save_json(registry_path, registry)
    return registry


# -- loader APIs for Section 12 (spec item 40) -----------------------------------------------------


def _latest_file(directory: Path) -> Optional[Path]:
    if not directory.exists():
        return None
    candidates = [p for p in directory.glob("*.json") if p.name != "latest.json"]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_by_id_or_latest(directory: Path, benchmark_id: Optional[str]) -> Optional[dict]:
    if benchmark_id:
        path = directory / f"{benchmark_id}.json"
        if not path.exists():
            return None
    else:
        path = _latest_file(directory)
        if path is None:
            return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_benchmark_summary(benchmark_id: Optional[str] = None, results_root: Path = DEFAULT_RESULTS_ROOT) -> Optional[dict]:
    """Loads a summary JSON (spec item 39/40). Defaults to the most
    recently written summary (`summary/latest.json`-equivalent freshness)
    if `benchmark_id` isn't given. Returns None (not an exception) when
    nothing has been run yet -- callers (the future dashboard) should
    treat that as "no data yet", not a crash."""
    if benchmark_id is None:
        latest_path = results_root / "summary" / "latest.json"
        if latest_path.exists():
            with open(latest_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
    return _load_by_id_or_latest(results_root / "summary", benchmark_id)


def load_batch_sweep(benchmark_id: Optional[str] = None, results_root: Path = DEFAULT_RESULTS_ROOT) -> Optional[dict]:
    return _load_by_id_or_latest(results_root / "batch_sweeps", benchmark_id)


def load_resolution_sweep(benchmark_id: Optional[str] = None, results_root: Path = DEFAULT_RESULTS_ROOT) -> Optional[dict]:
    return _load_by_id_or_latest(results_root / "resolution_sweeps", benchmark_id)


def load_per_filter_results(benchmark_id: Optional[str] = None, results_root: Path = DEFAULT_RESULTS_ROOT) -> Optional[dict]:
    return _load_by_id_or_latest(results_root / "per_filter", benchmark_id)


def load_correctness_results(benchmark_id: Optional[str] = None, results_root: Path = DEFAULT_RESULTS_ROOT) -> Optional[dict]:
    return _load_by_id_or_latest(results_root / "correctness", benchmark_id)


def load_manifest(benchmark_id: str, results_root: Path = DEFAULT_RESULTS_ROOT) -> Optional[dict]:
    return _load_by_id_or_latest(results_root / "manifests", benchmark_id)


def load_aggregated(benchmark_id: Optional[str] = None, results_root: Path = DEFAULT_RESULTS_ROOT) -> Optional[dict]:
    """Loads a canonical benchmark's full aggregated result (the file
    `run_canonical_benchmark()` writes to `aggregated/{benchmark_id}.json`
    -- CanonicalBenchmarkResult.to_dict())."""
    return _load_by_id_or_latest(results_root / "aggregated", benchmark_id)


def load_raw_runs(
    benchmark_id: Optional[str], implementation: str, results_root: Path = DEFAULT_RESULTS_ROOT,
) -> Optional[dict]:
    """Loads the individual (never-averaged) per-run measurements
    `run_canonical_benchmark()` wrote to `raw/{implementation}/{benchmark_id}.json`
    -- `implementation` is one of "cpu" / "basic_cuda" / "enhanced_cuda".
    Used by Section 16's dashboard for the H2D/kernel/D2H breakdown chart
    and the "raw measurements" expander (spec items 13, 30) -- never
    recomputed or re-measured, only the already-written per-run rows."""
    if implementation not in ("cpu", "basic_cuda", "enhanced_cuda"):
        raise ValueError(f"Unknown implementation: {implementation!r}")
    return _load_by_id_or_latest(results_root / "raw" / implementation, benchmark_id)


# -- canonical benchmark designation (Section 16 spec item 32) -----------------------------------------------------


def get_canonical_benchmark_id(results_root: Path = DEFAULT_RESULTS_ROOT) -> Optional[str]:
    """The benchmark_id explicitly designated as THE canonical result,
    distinct from "most recently run" -- re-running Section 11's sweeps
    (e.g. to verify a fix) creates a new historical entry without
    silently reassigning which one is canonical. Falls back to the
    experiment registry's "Final pipeline" entry (the last run pipeline
    benchmark) when no explicit designation file exists.
    """
    designation_path = results_root / "canonical.json"
    if designation_path.exists():
        with open(designation_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("benchmark_id")

    registry_path = results_root / "experiment_registry.json"
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as fh:
            registry = json.load(fh)
        for entry in registry.get("entries", []):
            if entry.get("experiment") == "Final pipeline":
                return entry.get("benchmark_id")
    return None


# -- reproducibility (spec items 31-32) -----------------------------------------------------


def check_reproducibility(benchmark_id: str, results_root: Path = DEFAULT_RESULTS_ROOT) -> Tuple[bool, List[str]]:
    """Loads `benchmark_id`'s manifest and checks whether the CURRENT
    dataset on disk still matches the fingerprint recorded at benchmark
    time. Returns (can_reproduce, notes) -- never raises for a mismatch,
    since the point is to REPORT the mismatch (spec item 31: 'report the
    fingerprint mismatch rather than silently proceeding'), not to
    silently refuse.
    """
    manifest = load_manifest(benchmark_id, results_root)
    if manifest is None:
        return False, [f"No manifest found for benchmark_id={benchmark_id!r}."]

    notes = []
    dataset_path = Path(manifest["dataset_path"])
    if not dataset_path.exists():
        notes.append(f"Recorded dataset_path no longer exists: {dataset_path}")
        return False, notes

    dm = DatasetManager(dataset_path)
    dm.scan()
    current_fingerprint = dm.fingerprint()
    if current_fingerprint != manifest["dataset_fingerprint"]:
        notes.append(
            f"Dataset fingerprint mismatch: recorded={manifest['dataset_fingerprint']} "
            f"current={current_fingerprint} -- the on-disk dataset has changed since this "
            f"benchmark ran; results are not guaranteed reproducible."
        )
        return False, notes

    missing = [
        item_path for item_path in manifest["selected_relative_paths"]
        if not (dataset_path / item_path).exists()
    ]
    if missing:
        notes.append(f"{len(missing)} of {len(manifest['selected_relative_paths'])} selected files no longer exist.")
        return False, notes

    return True, ["Dataset fingerprint matches; all selected files present."]
