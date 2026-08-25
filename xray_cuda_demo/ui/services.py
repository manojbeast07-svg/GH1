"""Application services layer (Section 12 spec items 58-59): the ONLY
module that calls into the existing backend (DatasetManager,
FilterConfig, cpu.pipeline, cuda.pipeline, cuda.final_benchmark,
xray_cuda). No filter/kernel/benchmark algorithm is reimplemented here
-- every function below is orchestration (loading, chaining existing
calls, catching exceptions into a UI-friendly form) around code that
already exists and is already tested.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from cpu.filters import FilterConfig
from cpu.pipeline import run_cpu_pipeline
from pipeline.dataset import DatasetManager, ImageSelection, SelectedItem
from pipeline.image_loader import load_image

try:
    import xray_cuda
    from cuda.gaussian import gaussian_cuda_gpu, gaussian_enhanced_cuda_gpu
    from cuda.laplacian import laplacian_cuda_gpu, laplacian_enhanced_cuda_gpu
    from cuda.median import median_cuda_gpu, median_enhanced_cuda_gpu
    from cuda.pipeline import (
        PipelineConfig,
        compute_safe_gpu_batch_size,
        run_basic_cuda_pipeline,
        run_cuda_pipeline,
        run_enhanced_cuda_pipeline,
    )
    from cuda.optimization_lab import FILTER_VARIANTS as _OPT_LAB_FILTER_VARIANTS
    from cuda.optimization_lab import PRODUCTION_DEFAULTS as _OPT_LAB_PRODUCTION_DEFAULTS
    from cuda.optimization_lab import available_variants as _opt_lab_available_variants
    from cuda.optimization_lab import measure_variant as _opt_lab_measure_variant
    from cuda.sobel import sobel_cuda_gpu, sobel_enhanced_cuda_gpu
    from cuda.threshold import threshold_cuda_gpu, threshold_enhanced_cuda_gpu

    _CUDA_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # extension not built, or import-time failure
    _CUDA_IMPORT_ERROR = str(exc)


# Static, CUDA-import-independent so the Optimization Lab's filter/variant
# selector can still render (labels only, no live run) even when the xray_cuda
# extension isn't built at all -- matches these five filters' stable,
# already-documented variant names (cuda/gaussian.py, cuda/median.py, etc.).
OPTIMIZATION_LAB_FILTERS = ["gaussian", "median", "sobel", "laplacian", "threshold"]
_OPTIMIZATION_LAB_VARIANTS_FALLBACK = {
    "gaussian": ("naive", "shared", "shared_const", "specialized"),
    "median": ("shared", "network3x3", "specialized"),
    "sobel": ("shared", "shared_const", "specialized", "separable"),
    "laplacian": ("shared", "shared_const", "specialized", "explicit"),
    "threshold": ("vectorized", "multi_pixel"),
}
OPTIMIZATION_LAB_PRODUCTION_DEFAULTS = {
    "gaussian": "specialized", "median": "network3x3", "sobel": "specialized",
    "laplacian": "specialized", "threshold": "vectorized",
}


class ServiceError(Exception):
    """Raised for any user-facing error this layer catches (CUDA
    unavailable, corrupt image, invalid config, OOM, ...) -- app.py
    displays `str(exc)`, never a raw traceback, unless Debug mode is on
    (spec item 38)."""


def cuda_available() -> bool:
    if _CUDA_IMPORT_ERROR is not None:
        return False
    try:
        return xray_cuda.cuda_available()
    except Exception:
        return False


def cuda_unavailable_reason() -> str:
    if _CUDA_IMPORT_ERROR is not None:
        return f"xray_cuda extension not importable: {_CUDA_IMPORT_ERROR}"
    return "No usable CUDA device detected on this machine."


# -- dataset (spec item 6: use DatasetManager, don't reimplement scanning) -----------------------------------------------------


@dataclass
class DatasetInfo:
    path: str
    total_files: int
    resolution_note: str


def scan_dataset(dataset_path: str) -> Tuple[DatasetManager, DatasetInfo]:
    try:
        dm = DatasetManager(dataset_path)
        dm.scan()
    except Exception as exc:
        raise ServiceError(f"Could not scan dataset at {dataset_path!r}: {exc}") from exc
    if len(dm.paths) == 0:
        raise ServiceError(f"No supported image files found under {dataset_path!r}.")
    info = DatasetInfo(path=dataset_path, total_files=len(dm.paths), resolution_note="mixed resolution (grouped at selection time)")
    return dm, info


def select_single(dm: DatasetManager, index: int) -> ImageSelection:
    try:
        return dm.select_single(index)
    except Exception as exc:
        raise ServiceError(f"Could not select image at index {index}: {exc}") from exc


def select_random_batch(dm: DatasetManager, batch_size: int, seed: int) -> ImageSelection:
    try:
        return dm.random_batch(batch_size=batch_size, seed=seed)
    except Exception as exc:
        raise ServiceError(f"Could not select a random batch: {exc}") from exc


def load_selection_images(selection: ImageSelection) -> List[np.ndarray]:
    images = []
    for item in selection.items:
        try:
            images.append(load_image(item.absolute_path))
        except Exception as exc:
            raise ServiceError(f"Could not load {item.relative_path!r}: {exc}") from exc
    return images


def group_images_by_shape(images: List[np.ndarray]) -> Dict[Tuple[int, int], List[int]]:
    """Returns {shape: [indices into `images`]} -- used to show mixed-
    resolution selections without silently resizing anything (item 8)."""
    groups: Dict[Tuple[int, int], List[int]] = {}
    for i, img in enumerate(images):
        groups.setdefault(img.shape, []).append(i)
    return groups


# -- per-implementation processing -----------------------------------------------------


@dataclass
class ImplementationResult:
    """One implementation's result on one batch: final outputs, per-
    stage intermediate outputs (only populated for a single preview
    image -- never hundreds of images), and timing.

    `h2d_ms`/`compute_ms`/`d2h_ms` are the authoritative backend timing-
    boundary breakdown (Section 14 spec item 12): for GPU implementations
    these come directly from the production pipeline's CUDA-event-
    measured `PipelineTiming` (never Streamlit wall-clock time); for CPU
    there is no H2D/D2H concept, so both stay None and `compute_ms`
    equals `total_ms` (pure processing time, image loading excluded --
    tracked separately in ComparisonResult.load_ms, spec item 11).
    """

    label: str  # "CPU" / "Basic CUDA" / "Enhanced CUDA"
    final_outputs: List[np.ndarray]
    stage_outputs: Optional[Dict[str, np.ndarray]]  # for the preview image only
    total_ms: float
    per_stage_ms: Dict[str, Optional[float]]
    h2d_ms: Optional[float] = None
    compute_ms: Optional[float] = None
    d2h_ms: Optional[float] = None


def _require_cuda() -> None:
    if not cuda_available():
        raise ServiceError(cuda_unavailable_reason())


def run_cpu_batch(images: List[np.ndarray], config: FilterConfig, preview_index: Optional[int] = None) -> ImplementationResult:
    final_outputs = []
    per_stage_totals: Dict[str, List[float]] = {}
    stage_outputs = None
    start = time.perf_counter()
    for i, img in enumerate(images):
        result, timing = run_cpu_pipeline(img, config)
        final_outputs.append(result.final_output)
        for stage, ms in timing.as_dict().items():
            if ms is not None:
                per_stage_totals.setdefault(stage, []).append(ms)
        if preview_index is not None and i == preview_index:
            stage_outputs = {
                "original": result.original,
                "gaussian": result.gaussian_output,
                "median": result.median_output,
                "sobel": result.sobel_output,
                "laplacian": result.laplacian_output,
                "threshold": result.threshold_output,
            }
    total_ms = (time.perf_counter() - start) * 1000.0
    per_stage_ms = {stage: sum(v) for stage, v in per_stage_totals.items()}
    return ImplementationResult("CPU", final_outputs, stage_outputs, total_ms, per_stage_ms, compute_ms=total_ms)


def _gpu_batch(images: List[np.ndarray], config: FilterConfig, use_enhanced: bool,
               pipeline_config: Optional["PipelineConfig"]) -> Tuple[np.ndarray, object]:
    _require_cuda()
    try:
        if pipeline_config is not None:
            return run_cuda_pipeline(images, config, pipeline_config)
        if use_enhanced:
            return run_enhanced_cuda_pipeline(images, config)
        return run_basic_cuda_pipeline(images, config)
    except MemoryError as exc:
        raise ServiceError(f"GPU ran out of memory for this batch ({len(images)} images): {exc}") from exc
    except Exception as exc:
        raise ServiceError(f"GPU processing failed: {exc}") from exc


def run_gpu_batch(
    images: List[np.ndarray], config: FilterConfig, use_enhanced: bool,
    pipeline_config: Optional["PipelineConfig"] = None, preview_index: Optional[int] = None,
) -> ImplementationResult:
    label = "Enhanced CUDA" if (use_enhanced or pipeline_config is not None) else "Basic CUDA"
    outputs, timing = _gpu_batch(images, config, use_enhanced, pipeline_config)

    stage_outputs = None
    if preview_index is not None:
        stage_outputs = run_single_image_stages(images[preview_index], config, use_enhanced, pipeline_config)

    per_stage_ms = {
        "gaussian": timing.gaussian_ms, "median": timing.median_ms, "sobel": timing.sobel_ms,
        "laplacian": timing.laplacian_ms, "threshold": timing.threshold_ms,
    }
    return ImplementationResult(
        label, list(outputs), stage_outputs, timing.total_ms, per_stage_ms,
        h2d_ms=timing.h2d_ms, compute_ms=timing.compute_ms, d2h_ms=timing.d2h_ms,
    )


def run_single_image_stages(
    image: np.ndarray, config: FilterConfig, use_enhanced: bool,
    pipeline_config: Optional["PipelineConfig"] = None,
) -> Dict[str, np.ndarray]:
    """Chains the already-tested per-filter GPU-native (*_cuda_gpu /
    *_enhanced_cuda_gpu) calls on ONE image, with no host round trip
    between stages, to expose intermediate outputs for the pipeline
    visualization -- the batched production pipeline is one fused
    native call and has no intermediate hooks, so single-image
    visualization goes through these lower-level (already existing,
    already tested) per-filter APIs instead. No new algorithm code.
    """
    _require_cuda()

    def _variant(name: str, default: str) -> str:
        if pipeline_config is None:
            return default if use_enhanced else "basic"
        return getattr(pipeline_config, f"{name}_variant")

    def _block(name: str, default=(16, 16)):
        if pipeline_config is None:
            return default
        return getattr(pipeline_config, f"{name}_block")

    try:
        gpu_image = xray_cuda.upload_image(image)
        stages = {"original": image}

        if config.gaussian_enabled:
            v = _variant("gaussian", "specialized")
            r = (gaussian_cuda_gpu(gpu_image, config.gaussian_kernel_size, config.gaussian_sigma) if v == "basic"
                 else gaussian_enhanced_cuda_gpu(gpu_image, config.gaussian_kernel_size, config.gaussian_sigma, v, _block("gaussian")))
            gpu_image = r["output"]
            stages["gaussian"] = xray_cuda.download_image(gpu_image)

        if config.median_enabled:
            v = _variant("median", "network3x3")
            r = (median_cuda_gpu(gpu_image, config.median_kernel_size) if v == "basic"
                 else median_enhanced_cuda_gpu(gpu_image, config.median_kernel_size, v, _block("median")))
            gpu_image = r["output"]
            stages["median"] = xray_cuda.download_image(gpu_image)

        if config.sobel_enabled:
            v = _variant("sobel", "specialized")
            r = (sobel_cuda_gpu(gpu_image, config.sobel_mode) if v == "basic"
                 else sobel_enhanced_cuda_gpu(gpu_image, config.sobel_mode, v, _block("sobel")))
            gpu_image = r["output"]
            stages["sobel"] = xray_cuda.download_image(gpu_image)

        if config.laplacian_enabled:
            v = _variant("laplacian", "specialized")
            r = (laplacian_cuda_gpu(gpu_image, config.laplacian_kernel_size, config.laplacian_scale, config.laplacian_delta) if v == "basic"
                 else laplacian_enhanced_cuda_gpu(gpu_image, config.laplacian_kernel_size, config.laplacian_scale, config.laplacian_delta, v, _block("laplacian")))
            gpu_image = r["output"]
            stages["laplacian"] = xray_cuda.download_image(gpu_image)

        if config.threshold_enabled:
            v = _variant("threshold", "vectorized")
            r = (threshold_cuda_gpu(gpu_image, config.threshold_value, config.threshold_max_value) if v == "basic"
                 else threshold_enhanced_cuda_gpu(gpu_image, config.threshold_value, config.threshold_max_value, v, _block("threshold")))
            gpu_image = r["output"]
            stages["threshold"] = xray_cuda.download_image(gpu_image)

        return stages
    except Exception as exc:
        raise ServiceError(f"GPU single-image processing failed: {exc}") from exc


# -- compare mode (spec item 16: same images, same order, all enabled implementations) -----------------------------------------------------


def run_compare(
    images: List[np.ndarray], config: FilterConfig,
    run_cpu: bool, run_basic: bool, run_enhanced: bool,
    preview_index: Optional[int] = None,
    enhanced_pipeline_config: Optional["PipelineConfig"] = None,
) -> Dict[str, ImplementationResult]:
    if not images:
        raise ServiceError("No images selected.")
    results: Dict[str, ImplementationResult] = {}
    if run_cpu:
        results["CPU"] = run_cpu_batch(images, config, preview_index=preview_index)
    if run_basic:
        results["Basic CUDA"] = run_gpu_batch(images, config, use_enhanced=False, preview_index=preview_index)
    if run_enhanced:
        results["Enhanced CUDA"] = run_gpu_batch(
            images, config, use_enhanced=True, pipeline_config=enhanced_pipeline_config, preview_index=preview_index,
        )
    return results


# -- single-image processing + structured comparison (Section 14) -----------------------------------------------------
#
# Everything below drives the single-image "[Process Selected Image]"
# and "[Compare CPU vs Basic CUDA vs Enhanced CUDA]" actions. Unlike
# run_compare() above (batch-oriented, gated by the sidebar's Compare-
# mode checkboxes), compare_single_image() ALWAYS runs all three
# implementations regardless of those checkboxes (spec item 10), and
# is resilient per-implementation: if one backend fails, the others'
# results are still returned (spec items 33-34) rather than the whole
# comparison aborting.


def load_single_image_with_timing(path) -> Tuple[np.ndarray, float]:
    """Loads one image from disk, measuring load time SEPARATELY from
    any processing time (spec item 11) -- the returned `load_ms` is
    never folded into an ImplementationResult's timing."""
    start = time.perf_counter()
    try:
        image = load_image(path)
    except Exception as exc:
        raise ServiceError(f"Could not load image at {path!r}: {exc}") from exc
    load_ms = (time.perf_counter() - start) * 1000.0
    return image, load_ms


@dataclass
class ImageMetadata:
    filename: str
    relative_path: Optional[str]
    absolute_path: Optional[str]
    width: int
    height: int
    dtype: str


def build_image_metadata(image: np.ndarray, item: Optional[SelectedItem] = None) -> ImageMetadata:
    return ImageMetadata(
        filename=item.filename if item else "(unselected)",
        relative_path=item.relative_path if item else None,
        absolute_path=str(item.absolute_path) if item else None,
        width=int(image.shape[1]), height=int(image.shape[0]), dtype=str(image.dtype),
    )


def process_single_image(image: np.ndarray, config: FilterConfig, implementation: str) -> ImplementationResult:
    """Runs exactly ONE implementation on exactly ONE image (spec item
    9's "[Process Selected Image]" action). `implementation` is one of
    "CPU" / "Basic CUDA" / "Enhanced CUDA". Uses the SAME production
    APIs run_compare()/run_gpu_batch() already use (run_cpu_pipeline /
    run_basic_cuda_pipeline / run_enhanced_cuda_pipeline) -- this is a
    thin single-image special case of run_compare(), not a parallel
    implementation."""
    if implementation == "CPU":
        return run_cpu_batch([image], config, preview_index=0)
    if implementation == "Basic CUDA":
        return run_gpu_batch([image], config, use_enhanced=False, preview_index=0)
    if implementation == "Enhanced CUDA":
        return run_gpu_batch([image], config, use_enhanced=True, preview_index=0)
    raise ServiceError(f"Unknown implementation: {implementation!r}")


# Known per-stage correctness tolerances (spec item 20) -- the same
# values already established and documented in Sections 6-10 and
# reused (not re-derived) by cuda.final_benchmark.run_correctness_benchmark.
STAGE_TOLERANCES = {"gaussian": 1, "median": 0, "sobel": 0, "laplacian": 0, "threshold": 0}
STAGE_ORDER = ["gaussian", "median", "sobel", "laplacian", "threshold"]


@dataclass
class StageCorrectness:
    stage: str
    tolerance: int
    max_abs_diff_vs_cpu: Optional[int]  # None when not computable (stage disabled, or CPU/GPU result missing)
    status: str  # "PASS" / "WARNING" / "N/A"


@dataclass
class PipelineCorrectness:
    comparison: str  # "basic_vs_cpu" / "enhanced_vs_cpu" / "enhanced_vs_basic"
    max_abs_diff: int
    mean_abs_diff: float
    rmse: float
    differing_pixel_count: int
    differing_pixel_percentage: float


def _compute_stage_correctness(results: Dict[str, ImplementationResult], config: FilterConfig) -> List[StageCorrectness]:
    cpu = results.get("CPU")
    stage_flags = {
        "gaussian": config.gaussian_enabled, "median": config.median_enabled, "sobel": config.sobel_enabled,
        "laplacian": config.laplacian_enabled, "threshold": config.threshold_enabled,
    }
    out = []
    for stage in STAGE_ORDER:
        tolerance = STAGE_TOLERANCES[stage]
        if not stage_flags[stage] or cpu is None or cpu.stage_outputs is None or cpu.stage_outputs.get(stage) is None:
            out.append(StageCorrectness(stage, tolerance, None, "N/A"))
            continue
        cpu_out = cpu.stage_outputs[stage]
        max_diff = 0
        any_gpu = False
        for label in ("Basic CUDA", "Enhanced CUDA"):
            r = results.get(label)
            if r is None or r.stage_outputs is None or r.stage_outputs.get(stage) is None:
                continue
            any_gpu = True
            gpu_out = r.stage_outputs[stage]
            d = int(np.abs(cpu_out.astype(np.int16) - gpu_out.astype(np.int16)).max())
            max_diff = max(max_diff, d)
        if not any_gpu:
            out.append(StageCorrectness(stage, tolerance, None, "N/A"))
            continue
        out.append(StageCorrectness(stage, tolerance, max_diff, "PASS" if max_diff <= tolerance else "WARNING"))
    return out


def _compute_pipeline_correctness(results: Dict[str, ImplementationResult]) -> List[PipelineCorrectness]:
    """Full-pipeline correctness (spec item 21), computed from THIS
    run's actual outputs via the existing, already-tested
    cuda.benchmark.CorrectnessMetrics -- never hardcoded, never
    reimplemented."""
    from cuda.benchmark import CorrectnessMetrics

    out = []
    for name, label_a, label_b in [
        ("basic_vs_cpu", "CPU", "Basic CUDA"), ("enhanced_vs_cpu", "CPU", "Enhanced CUDA"),
        ("enhanced_vs_basic", "Basic CUDA", "Enhanced CUDA"),
    ]:
        a, b = results.get(label_a), results.get(label_b)
        if a is None or b is None:
            continue
        metrics = CorrectnessMetrics.compare(a.final_outputs, np.stack(b.final_outputs))
        out.append(PipelineCorrectness(
            name, metrics.max_abs_diff, metrics.mean_abs_diff, metrics.rmse,
            metrics.differing_pixel_count, metrics.differing_pixel_percentage,
        ))
    return out


@dataclass
class ComparisonResult:
    """Spec item 14: one structured object bundling everything a
    comparison run produced, stored as ONE session-state value rather
    than scattered across arbitrary keys."""

    image_meta: ImageMetadata
    filter_config: dict
    load_ms: float
    results: Dict[str, ImplementationResult]  # successful implementations only
    errors: Dict[str, str]  # implementation label -> error message, for failed ones (spec items 33-34)
    stage_correctness: List[StageCorrectness]
    pipeline_correctness: List[PipelineCorrectness]
    timestamp_utc: str


def compare_single_image(
    image: np.ndarray, config: FilterConfig, image_meta: ImageMetadata, load_ms: float = 0.0,
) -> ComparisonResult:
    """Spec item 10: ALWAYS runs CPU, Basic CUDA, AND Enhanced CUDA (not
    gated by the sidebar's Compare-mode checkboxes -- this is the
    dedicated "Compare CPU vs Basic CUDA vs Enhanced CUDA" action).
    Resilient per-implementation: a failure in one does not lose the
    others' results (spec item 33-34)."""
    results: Dict[str, ImplementationResult] = {}
    errors: Dict[str, str] = {}
    for label in ("CPU", "Basic CUDA", "Enhanced CUDA"):
        try:
            results[label] = process_single_image(image, config, label)
        except ServiceError as exc:
            errors[label] = str(exc)

    return ComparisonResult(
        image_meta=image_meta, filter_config=asdict(config), load_ms=load_ms,
        results=results, errors=errors,
        stage_correctness=_compute_stage_correctness(results, config),
        pipeline_correctness=_compute_pipeline_correctness(results),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


# -- saving live results (spec item 29: separate from benchmark_results/, never overwritten) -----------------------------------------------------

DEFAULT_LIVE_OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "live_processing"


def save_comparison_results(comparison: ComparisonResult, output_root: Optional[Path] = None) -> Path:
    """Saves original.png, cpu/, basic_cuda/, enhanced_cuda/ (per-stage
    + final PNGs), differences/, and metadata.json under a fresh,
    timestamped run directory -- never benchmark_results/ (spec item
    36), never overwrites a previous run (spec item 29: a new directory
    every call, `mkdir(exist_ok=False)` makes a collision fail loudly
    rather than silently clobber)."""
    import json

    import cv2

    root = output_root or DEFAULT_LIVE_OUTPUT_ROOT
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    label_to_dirname = {"CPU": "cpu", "Basic CUDA": "basic_cuda", "Enhanced CUDA": "enhanced_cuda"}
    any_result = next(iter(comparison.results.values()), None)
    if any_result is not None and any_result.stage_outputs and "original" in any_result.stage_outputs:
        cv2.imwrite(str(run_dir / "original.png"), any_result.stage_outputs["original"])

    for label, result in comparison.results.items():
        stage_dir = run_dir / label_to_dirname[label]
        stage_dir.mkdir(parents=True, exist_ok=True)
        if result.stage_outputs:
            for stage, img in result.stage_outputs.items():
                if stage == "original" or img is None:
                    continue
                cv2.imwrite(str(stage_dir / f"{stage}.png"), img)
        if result.final_outputs:
            cv2.imwrite(str(stage_dir / "final.png"), result.final_outputs[0])

    diff_dir = run_dir / "differences"
    diff_dir.mkdir(parents=True, exist_ok=True)
    for name, label_a, label_b in [("basic_vs_cpu", "CPU", "Basic CUDA"), ("enhanced_vs_cpu", "CPU", "Enhanced CUDA"),
                                    ("enhanced_vs_basic", "Basic CUDA", "Enhanced CUDA")]:
        a, b = comparison.results.get(label_a), comparison.results.get(label_b)
        if a is None or b is None or not a.final_outputs or not b.final_outputs:
            continue
        diff = np.abs(a.final_outputs[0].astype(np.int16) - b.final_outputs[0].astype(np.int16)).astype(np.uint8)
        cv2.imwrite(str(diff_dir / f"{name}.png"), diff)

    metadata = {
        "run_id": run_id,
        "timestamp_utc": comparison.timestamp_utc,
        "image_meta": asdict(comparison.image_meta),
        "filter_config": comparison.filter_config,
        "load_ms": comparison.load_ms,
        "timings_ms": {
            label: {"total_ms": r.total_ms, "h2d_ms": r.h2d_ms, "compute_ms": r.compute_ms, "d2h_ms": r.d2h_ms,
                    "per_stage_ms": r.per_stage_ms}
            for label, r in comparison.results.items()
        },
        "errors": comparison.errors,
        "stage_correctness": [asdict(s) for s in comparison.stage_correctness],
        "pipeline_correctness": [asdict(p) for p in comparison.pipeline_correctness],
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)

    return run_dir


# -- lightweight live benchmark (spec item 34: NOT the full Section 11 matrix) -----------------------------------------------------


@dataclass
class LiveBenchmarkResult:
    n_images: int
    warmup_runs: int
    measurement_runs: int
    cpu_ms: Optional[float]
    basic_ms: Optional[float]
    enhanced_ms: Optional[float]


def run_live_benchmark(
    images: List[np.ndarray], config: FilterConfig,
    run_cpu: bool, run_basic: bool, run_enhanced: bool,
    warmup_runs: int = 1, measurement_runs: int = 3,
) -> LiveBenchmarkResult:
    """A small, session-local benchmark for the CURRENT configuration --
    deliberately not cuda.final_benchmark's full matrix (spec item 34),
    and never written to benchmark_results/ (spec item 33: historical
    artifacts are never touched by live UI actions)."""
    if not images:
        raise ServiceError("No images selected.")

    cpu_ms = basic_ms = enhanced_ms = None
    if run_cpu:
        for _ in range(warmup_runs):
            run_cpu_batch(images, config)
        values = [run_cpu_batch(images, config).total_ms for _ in range(measurement_runs)]
        cpu_ms = statistics.median(values)
    if run_basic:
        _require_cuda()
        for _ in range(warmup_runs):
            run_gpu_batch(images, config, use_enhanced=False)
        values = [run_gpu_batch(images, config, use_enhanced=False).total_ms for _ in range(measurement_runs)]
        basic_ms = statistics.median(values)
    if run_enhanced:
        _require_cuda()
        for _ in range(warmup_runs):
            run_gpu_batch(images, config, use_enhanced=True)
        values = [run_gpu_batch(images, config, use_enhanced=True).total_ms for _ in range(measurement_runs)]
        enhanced_ms = statistics.median(values)

    return LiveBenchmarkResult(len(images), warmup_runs, measurement_runs, cpu_ms, basic_ms, enhanced_ms)


# -- device / environment info -----------------------------------------------------


def device_info() -> Optional[dict]:
    if not cuda_available():
        return None
    try:
        return xray_cuda.device_info()
    except Exception:
        return None


def gpu_memory_info() -> Optional[dict]:
    if not cuda_available():
        return None
    try:
        return xray_cuda.device_memory_info()
    except Exception:
        return None


def environment_fingerprint() -> dict:
    from pipeline.environment import get_environment_fingerprint

    return get_environment_fingerprint().to_dict()


def build_tool_versions() -> dict:
    """NVCC/CMake versions (Section 20 spec item 21) -- live subprocess
    probes, same approach scripts/diagnose.py already uses, never
    hardcoded. Returns "MISSING" for a tool not found on PATH rather
    than raising, since this is informational display, not a build
    step."""
    import shutil
    import subprocess

    def _tool_version(cmd: List[str], prefer_line_containing: Optional[str] = None) -> str:
        exe = shutil.which(cmd[0])
        if exe is None:
            return "MISSING"
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            lines = (out.stdout or out.stderr).strip().splitlines()
            if not lines:
                return "UNKNOWN"
            if prefer_line_containing:
                match = next((ln for ln in lines if prefer_line_containing in ln), None)
                if match:
                    return match.strip()
            return lines[0]
        except Exception as exc:  # noqa: BLE001 -- best-effort diagnostic
            return f"ERROR ({exc})"

    return {
        "nvcc": _tool_version(["nvcc", "--version"], prefer_line_containing="release"),
        "cmake": _tool_version(["cmake", "--version"]),
    }


def gpu_implementation_facts() -> dict:
    """Section 20 spec items 18/32/43: the same runtime/source checks
    scripts/inspect_gpu_backend.py performs, returned as a dict for
    Presentation Mode -- every field is independently verified here,
    never an asserted claim."""
    from pathlib import Path as _Path

    project_root = _Path(__file__).resolve().parent.parent
    facts = {
        "native_extension": False, "single_pipeline_call": False, "cu_file_count": 0,
        "no_python_gpu_libs": False, "raii_memory": False, "cuda_event_timing": False,
    }
    try:
        import xray_cuda

        ext_path = _Path(xray_cuda.__file__).resolve()
        facts["native_extension"] = ext_path.suffix in (".pyd", ".so")
        fn = getattr(xray_cuda, "run_basic_cuda_pipeline_gpu", None)
        facts["single_pipeline_call"] = fn is not None and getattr(fn, "__module__", None) == "xray_cuda"
    except Exception:
        pass

    cu_files = sorted((project_root / "cuda" / "src").glob("*.cu"))
    facts["cu_file_count"] = len(cu_files)

    import ast

    forbidden = {"cupy", "numba", "torch", "pycuda"}
    found = []
    for py_file in (project_root / "cuda").glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.extend(a.name for a in node.names if a.name.split(".")[0] in forbidden)
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in forbidden:
                found.append(node.module)
    facts["no_python_gpu_libs"] = not found

    gpu_image_header = project_root / "cuda" / "include" / "gpu_image.cuh"
    facts["raii_memory"] = gpu_image_header.exists() and "cudaMalloc" in gpu_image_header.read_text(encoding="utf-8")
    cuda_common_header = project_root / "cuda" / "include" / "cuda_common.cuh"
    facts["cuda_event_timing"] = cuda_common_header.exists() and "cudaEventRecord" in cuda_common_header.read_text(encoding="utf-8")
    return facts


def environment_label() -> str:
    """Spec item 22: LOCAL vs BREV environment badge. Detected from
    well-known Brev environment variables (never guessed from GPU
    model, since Brev instances can use many different GPUs) -- falls
    back to LOCAL when none are present."""
    import os

    brev_markers = ("BREV_WORKSPACE_ID", "BREV_INSTANCE_ID", "BREV_HOST", "NVIDIA_BREV")
    if any(os.environ.get(marker) for marker in brev_markers):
        return "BREV ENVIRONMENT"
    return "LOCAL ENVIRONMENT"


# -- Section 11 benchmark artifact loaders (spec item 41: one import point, no duplicated JSON parsing) -----------------------------------------------------


def load_benchmark_summary(benchmark_id: Optional[str] = None, results_root=None) -> Optional[dict]:
    from cuda.final_benchmark import DEFAULT_RESULTS_ROOT, load_benchmark_summary as _load

    return _load(benchmark_id, results_root or DEFAULT_RESULTS_ROOT)


def load_batch_sweep(benchmark_id: Optional[str] = None, results_root=None) -> Optional[dict]:
    from cuda.final_benchmark import DEFAULT_RESULTS_ROOT, load_batch_sweep as _load

    return _load(benchmark_id, results_root or DEFAULT_RESULTS_ROOT)


def load_resolution_sweep(benchmark_id: Optional[str] = None, results_root=None) -> Optional[dict]:
    from cuda.final_benchmark import DEFAULT_RESULTS_ROOT, load_resolution_sweep as _load

    return _load(benchmark_id, results_root or DEFAULT_RESULTS_ROOT)


def load_per_filter_results(benchmark_id: Optional[str] = None, results_root=None) -> Optional[dict]:
    from cuda.final_benchmark import DEFAULT_RESULTS_ROOT, load_per_filter_results as _load

    return _load(benchmark_id, results_root or DEFAULT_RESULTS_ROOT)


def load_correctness_results(benchmark_id: Optional[str] = None, results_root=None) -> Optional[dict]:
    from cuda.final_benchmark import DEFAULT_RESULTS_ROOT, load_correctness_results as _load

    return _load(benchmark_id, results_root or DEFAULT_RESULTS_ROOT)


def list_known_benchmark_ids(results_root=None) -> List[str]:
    from cuda.final_benchmark import DEFAULT_RESULTS_ROOT

    root = results_root or DEFAULT_RESULTS_ROOT
    manifests_dir = root / "manifests"
    if not manifests_dir.exists():
        return []
    return sorted((p.stem for p in manifests_dir.glob("*.json")), reverse=True)


def load_experiment_registry(results_root=None) -> Optional[dict]:
    import json

    from cuda.final_benchmark import DEFAULT_RESULTS_ROOT

    root = results_root or DEFAULT_RESULTS_ROOT
    path = root / "experiment_registry.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_raw_runs(benchmark_id: Optional[str], implementation: str, results_root=None) -> Optional[dict]:
    from cuda.final_benchmark import DEFAULT_RESULTS_ROOT, load_raw_runs as _load

    return _load(benchmark_id, implementation, results_root or DEFAULT_RESULTS_ROOT)


def get_canonical_benchmark_id(results_root=None) -> Optional[str]:
    from cuda.final_benchmark import DEFAULT_RESULTS_ROOT, get_canonical_benchmark_id as _get

    return _get(results_root or DEFAULT_RESULTS_ROOT)


def is_canonical_benchmark(benchmark_id: Optional[str], results_root=None) -> bool:
    if not benchmark_id:
        return False
    return benchmark_id == get_canonical_benchmark_id(results_root)


def gpu_compute_breakdown(benchmark_id: Optional[str], implementation: str, results_root=None) -> Optional[dict]:
    """Averages the already-measured (never recomputed) per-run H2D /
    per-filter / D2H timings for one GPU implementation of one
    benchmark, for Section 16's Chart 3 (spec item 13: 'where does GPU
    time go?'). `implementation` is 'basic_cuda' or 'enhanced_cuda'."""
    raw = load_raw_runs(benchmark_id, implementation, results_root)
    if raw is None or not raw.get("runs"):
        return None
    runs = raw["runs"]
    fields = ["h2d_ms", "gaussian_ms", "median_ms", "sobel_ms", "laplacian_ms", "threshold_ms", "d2h_ms",
              "compute_ms", "total_ms"]
    means = {}
    for field in fields:
        values = [r[field] for r in runs if r.get(field) is not None]
        means[field] = (sum(values) / len(values)) if values else None
    means["n_runs"] = len(runs)
    means["benchmark_id"] = raw["benchmark_id"]
    return means


def benchmark_metadata(summary: dict) -> dict:
    """Flattens the fields Section 16's metadata panel needs (spec items
    6, 29) out of an already-loaded summary dict's nested `manifest` --
    no new parsing, just field selection."""
    manifest = summary["manifest"]
    env = manifest.get("environment", {})
    return {
        "benchmark_id": summary["benchmark_id"],
        "timestamp_utc": summary["timestamp_utc"],
        "dataset_fingerprint": summary.get("dataset_fingerprint"),
        "seed": manifest.get("seed"),
        "image_count": manifest.get("selected_image_count"),
        "resolution": manifest.get("resolution"),
        "runs": manifest.get("measurement_runs"),
        "warmup_runs": manifest.get("warmup_runs"),
        "gpu_name": env.get("gpu_name"),
        "cuda_runtime_version": env.get("cuda_runtime_version"),
        "gpu_driver_version": env.get("gpu_driver_version"),
        "cpu_model": env.get("cpu_model"),
    }


BENCHMARK_COMPARABLE_FIELDS = [
    ("GPU", lambda m: m.get("environment", {}).get("gpu_name")),
    ("Dataset fingerprint", lambda m: m.get("dataset_fingerprint")),
    ("Image count", lambda m: m.get("selected_image_count")),
    ("Resolution", lambda m: tuple(m.get("resolution")) if m.get("resolution") else None),
    ("Filter config", lambda m: m.get("filter_config")),
    ("Basic CUDA config", lambda m: m.get("basic_cuda_config")),
    ("Enhanced CUDA config", lambda m: m.get("enhanced_cuda_config")),
]


def compare_benchmarks(benchmark_id_a: str, benchmark_id_b: str, results_root=None) -> dict:
    """Loads two historical summaries for side-by-side comparison (spec
    items 33-34), flagging any configuration field that differs between
    them so two non-comparable runs are never presented as directly
    comparable without a warning."""
    summary_a = load_benchmark_summary(benchmark_id_a, results_root)
    summary_b = load_benchmark_summary(benchmark_id_b, results_root)
    if summary_a is None:
        raise ServiceError(f"No benchmark found with id {benchmark_id_a!r}.")
    if summary_b is None:
        raise ServiceError(f"No benchmark found with id {benchmark_id_b!r}.")

    manifest_a, manifest_b = summary_a["manifest"], summary_b["manifest"]
    differences = []
    for label, getter in BENCHMARK_COMPARABLE_FIELDS:
        value_a, value_b = getter(manifest_a), getter(manifest_b)
        if value_a != value_b:
            differences.append(f"{label}: {value_a!r} vs {value_b!r}")

    return {"summary_a": summary_a, "summary_b": summary_b, "differences": differences}


def export_benchmark_json(summary: dict) -> str:
    """Spec item 36: exports the selected benchmark's stored data
    verbatim -- never modifies the source file on disk."""
    import json

    return json.dumps(summary, indent=2)


def export_benchmark_csv(summary: dict) -> str:
    """Spec item 36: a flat CSV of the selected benchmark's headline
    end-to-end / compute-only measurements per implementation, built
    from already-stored AggregatedStat fields (mean/median/min/max/std)
    -- no new statistics are computed here."""
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["benchmark_id", "implementation", "metric", "mean_ms", "median_ms", "min_ms", "max_ms",
                      "std_ms", "n"])
    for label, key in [("CPU", "cpu"), ("Basic CUDA", "basic_cuda"), ("Enhanced CUDA", "enhanced_cuda")]:
        impl = summary.get(key, {})
        for metric_name in ("mode1_kernel_only_ms", "mode4_end_to_end_ms"):
            stat = impl.get(metric_name)
            if stat is None:
                continue
            writer.writerow([summary["benchmark_id"], label, metric_name, stat["mean"], stat["median"],
                              stat["min"], stat["max"], stat["std"], stat["n"]])

    per_filter = summary.get("per_filter")
    if per_filter:
        for row in per_filter["rows"]:
            writer.writerow([summary["benchmark_id"], "per_filter", row["filter"],
                              row["basic_kernel_ms"]["mean"] if isinstance(row["basic_kernel_ms"], dict) else row["basic_kernel_ms"],
                              "", "", "", "", ""])
    return buffer.getvalue()


# -- Section 15: real-batch live processing -----------------------------------------------------
#
# Extends Section 14's single-image process_single_image()/
# compare_single_image() to a full selection (item 4: "reuse the same
# processing service functions for both" -- run_cpu_batch()/
# run_gpu_batch() below are the exact functions Section 14's
# ImplementationResult already comes from; this section only adds the
# resolution-grouping loop and batch-scale aggregation around them,
# never a second CUDA/CPU processing path).
#
# The real dataset is mixed-resolution (Section 2 finding), and
# run_basic_cuda_pipeline()/run_enhanced_cuda_pipeline() require one
# uniform (height, width) per call (Section 5's own constraint) -- so a
# batch selection is processed one resolution group at a time (mirrors
# cuda.final_benchmark.run_resolution_sweep()'s and
# scripts/run_final_benchmark.py's established pattern), each group
# additionally capped at the actual safe GPU capacity
# (compute_safe_gpu_batch_size(), Section 5 -- never assumes a request
# fits in VRAM), and reassembled/reported in the ORIGINAL selection
# order (item 7).


@dataclass
class BatchGroupResult:
    shape: Tuple[int, int]  # (height, width)
    original_indices: List[int]  # this group's images' positions in the flat, original-order selection
    requested_count: int
    effective_count: int  # after the safe-GPU-capacity cap (item 39: never silently drops the rest)
    results: Dict[str, ImplementationResult]  # "CPU" / "Basic CUDA" / "Enhanced CUDA" -> result for this group only
    errors: Dict[str, str]


@dataclass
class BatchCorrectnessAggregate:
    comparison: str  # "basic_vs_cpu" / "enhanced_vs_cpu" / "enhanced_vs_basic"
    max_abs_diff: int
    mean_abs_diff: float
    rmse: float
    differing_pixel_count: int
    differing_pixel_percentage: float
    images_compared: int
    images_with_zero_diff: int
    images_with_nonzero_diff: int
    max_image_diff_percentage: float
    mean_image_diff_percentage: float


@dataclass
class BatchComparisonResult:
    seed: Optional[int]
    requested_batch_size: int
    effective_count: int  # total images actually processed, across all groups, after capacity capping
    dataset_fingerprint: Optional[str]
    selected_relative_paths: List[str]  # flat, original selection order
    resolution_group_summary: Dict[str, int]  # "224x224" -> image count, for display (item 8)
    filter_config: dict
    load_ms: float
    groups: List[BatchGroupResult]
    correctness: List[BatchCorrectnessAggregate]
    timestamp_utc: str

    def implementation_totals(self) -> Dict[str, dict]:
        """Sums each implementation's timing across all groups (a batch
        spanning groups is processed group-by-group, sequentially, so
        summing is the correct "total time for this batch" -- item 16's
        Total field) and computes aggregate throughput (item 17)."""
        totals: Dict[str, dict] = {}
        for label in ("CPU", "Basic CUDA", "Enhanced CUDA"):
            total_ms = h2d_ms = compute_ms = d2h_ms = 0.0
            has_h2d = False
            n = 0
            per_stage_totals: Dict[str, float] = {}
            for group in self.groups:
                r = group.results.get(label)
                if r is None:
                    continue
                total_ms += r.total_ms
                if r.h2d_ms is not None:
                    has_h2d = True
                    h2d_ms += r.h2d_ms
                    compute_ms += r.compute_ms or 0.0
                    d2h_ms += r.d2h_ms or 0.0
                n += len(r.final_outputs)
                for stage, ms in r.per_stage_ms.items():
                    if ms is not None:
                        per_stage_totals[stage] = per_stage_totals.get(stage, 0.0) + ms
            if n == 0:
                continue
            totals[label] = {
                "n_images": n,
                "total_ms": total_ms,
                "h2d_ms": h2d_ms if has_h2d else None,
                "compute_ms": compute_ms if has_h2d else total_ms,
                "d2h_ms": d2h_ms if has_h2d else None,
                "images_per_second": n / (total_ms / 1000.0) if total_ms > 0 else 0.0,
                "ms_per_image": total_ms / n if n > 0 else 0.0,
                "per_stage_ms": per_stage_totals,
            }
        return totals


def _pairwise_group_diff_stats(cpu_list: List[np.ndarray], gpu_list: List[np.ndarray]) -> dict:
    """Per-group, per-image difference statistics (never just the first
    image, spec item 22) -- combined losslessly across groups by
    _combine_batch_correctness() below (sums/concatenations, not
    approximations)."""
    total_pixels = 0
    differing = 0
    sum_abs = 0.0
    sum_sq = 0.0
    max_diff = 0
    per_image_pct: List[float] = []
    for cpu_img, gpu_img in zip(cpu_list, gpu_list):
        d = np.abs(cpu_img.astype(np.int16) - gpu_img.astype(np.int16))
        n = d.size
        nz = int(np.count_nonzero(d))
        total_pixels += n
        differing += nz
        sum_abs += float(d.sum())
        sum_sq += float(np.sum(d.astype(np.float64) ** 2))
        max_diff = max(max_diff, int(d.max()))
        per_image_pct.append(100.0 * nz / n)
    return {
        "total_pixels": total_pixels, "differing": differing, "sum_abs": sum_abs,
        "sum_sq": sum_sq, "max_diff": max_diff, "per_image_pct": per_image_pct,
    }


def _combine_batch_correctness(comparison_name: str, per_group_stats: List[dict]) -> BatchCorrectnessAggregate:
    total_pixels = sum(s["total_pixels"] for s in per_group_stats)
    differing = sum(s["differing"] for s in per_group_stats)
    sum_abs = sum(s["sum_abs"] for s in per_group_stats)
    sum_sq = sum(s["sum_sq"] for s in per_group_stats)
    max_diff = max((s["max_diff"] for s in per_group_stats), default=0)
    all_pct = [pct for s in per_group_stats for pct in s["per_image_pct"]]

    return BatchCorrectnessAggregate(
        comparison=comparison_name,
        max_abs_diff=max_diff,
        mean_abs_diff=(sum_abs / total_pixels) if total_pixels > 0 else 0.0,
        rmse=float(np.sqrt(sum_sq / total_pixels)) if total_pixels > 0 else 0.0,
        differing_pixel_count=differing,
        differing_pixel_percentage=(100.0 * differing / total_pixels) if total_pixels > 0 else 0.0,
        images_compared=len(all_pct),
        images_with_zero_diff=sum(1 for p in all_pct if p == 0.0),
        images_with_nonzero_diff=sum(1 for p in all_pct if p > 0.0),
        max_image_diff_percentage=max(all_pct, default=0.0),
        mean_image_diff_percentage=(sum(all_pct) / len(all_pct)) if all_pct else 0.0,
    )


def compare_batch(
    selection: ImageSelection, images: List[np.ndarray], config: FilterConfig, load_ms: float = 0.0,
    run_cpu: bool = True, run_basic: bool = True, run_enhanced: bool = True,
    dataset_fingerprint: Optional[str] = None,
) -> BatchComparisonResult:
    """The Section 15 batch equivalent of compare_single_image(): the
    SAME selection/images reach every implementation, grouped by native
    resolution (never resized -- item 8), each group's requested size
    capped at its actual safe GPU capacity (item 39), processed via the
    exact same run_cpu_batch()/run_gpu_batch() Section 14 already uses
    (item 4/9 -- never a second processing path), and reassembled in
    original selection order for reporting.
    """
    if not images:
        raise ServiceError("No images selected.")
    if len(images) != len(selection.items):
        raise ServiceError("Selection and loaded images are out of sync.")

    shape_to_indices = group_images_by_shape(images)
    resolution_group_summary = {f"{shape[1]}x{shape[0]}": len(idxs) for shape, idxs in shape_to_indices.items()}

    groups: List[BatchGroupResult] = []
    total_effective = 0
    for shape, indices in shape_to_indices.items():
        height, width = shape
        requested_count = len(indices)
        if cuda_available():
            try:
                safe_capacity = compute_safe_gpu_batch_size(height, width)
            except Exception:
                safe_capacity = requested_count
        else:
            safe_capacity = requested_count
        effective_indices = indices[: max(1, min(requested_count, safe_capacity))]
        group_images = [images[i] for i in effective_indices]

        group_results: Dict[str, ImplementationResult] = {}
        group_errors: Dict[str, str] = {}
        if run_cpu:
            try:
                group_results["CPU"] = run_cpu_batch(group_images, config)
            except ServiceError as exc:
                group_errors["CPU"] = str(exc)
        if run_basic:
            try:
                group_results["Basic CUDA"] = run_gpu_batch(group_images, config, use_enhanced=False)
            except ServiceError as exc:
                group_errors["Basic CUDA"] = str(exc)
        if run_enhanced:
            try:
                group_results["Enhanced CUDA"] = run_gpu_batch(group_images, config, use_enhanced=True)
            except ServiceError as exc:
                group_errors["Enhanced CUDA"] = str(exc)

        groups.append(BatchGroupResult(
            shape=shape, original_indices=effective_indices, requested_count=requested_count,
            effective_count=len(effective_indices), results=group_results, errors=group_errors,
        ))
        total_effective += len(effective_indices)

    correctness: List[BatchCorrectnessAggregate] = []
    for name, label_a, label_b in [
        ("basic_vs_cpu", "CPU", "Basic CUDA"), ("enhanced_vs_cpu", "CPU", "Enhanced CUDA"),
        ("enhanced_vs_basic", "Basic CUDA", "Enhanced CUDA"),
    ]:
        per_group_stats = []
        for group in groups:
            a, b = group.results.get(label_a), group.results.get(label_b)
            if a is None or b is None:
                continue
            per_group_stats.append(_pairwise_group_diff_stats(a.final_outputs, b.final_outputs))
        if per_group_stats:
            correctness.append(_combine_batch_correctness(name, per_group_stats))

    return BatchComparisonResult(
        seed=selection.seed, requested_batch_size=selection.requested_batch_size or len(selection),
        effective_count=total_effective,
        dataset_fingerprint=dataset_fingerprint,
        selected_relative_paths=[item.relative_path for item in selection.items],
        resolution_group_summary=resolution_group_summary,
        filter_config=asdict(config), load_ms=load_ms, groups=groups, correctness=correctness,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


def find_group_for_flat_index(batch_result: BatchComparisonResult, flat_index: int) -> Optional[Tuple[BatchGroupResult, int]]:
    """Maps a flat (original-selection-order) image index to
    (its BatchGroupResult, its local index within that group), or None
    if that image fell outside every group's effective (capacity-
    capped) subset."""
    for group in batch_result.groups:
        if flat_index in group.original_indices:
            return group, group.original_indices.index(flat_index)
    return None


def get_preview_stage_outputs(
    batch_result: BatchComparisonResult, flat_index: int, images: List[np.ndarray], config: FilterConfig,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Computes per-stage outputs for ONE representative image on
    demand (item 13/29: never store intermediate stages for the whole
    batch) -- reuses process_single_image() per implementation, exactly
    Section 14's single-image machinery, just pointed at whichever
    image the user is currently previewing."""
    located = find_group_for_flat_index(batch_result, flat_index)
    if located is None:
        raise ServiceError(f"Image index {flat_index} is not part of the processed batch (capacity-capped out).")
    group, _local_index = located
    image = images[flat_index]

    stage_outputs_by_impl: Dict[str, Dict[str, np.ndarray]] = {}
    for label in group.results.keys():
        result = process_single_image(image, config, label)
        if result.stage_outputs:
            stage_outputs_by_impl[label] = result.stage_outputs
    return stage_outputs_by_impl


# -- Section 15: quick live batch-size sweep (item 34-35, NOT the Section 11 framework) -----------------------------------------------------


def run_quick_batch_sweep(
    images_pool: List[np.ndarray], config: FilterConfig, batch_sizes: List[int],
    run_cpu: bool = True, run_basic: bool = True, run_enhanced: bool = True,
) -> dict:
    """A small, session-local, single-resolution-group batch-size
    comparison -- explicitly NOT cuda.final_benchmark's full sweep
    (item 34), only run when the user clicks the button, never written
    to benchmark_results/. `images_pool` should already share one
    resolution (the caller's dominant group); each requested batch size
    is capped at len(images_pool) and at the actual safe GPU capacity.
    """
    if not images_pool:
        raise ServiceError("No images available for the batch-size sweep.")
    shape = images_pool[0].shape
    rows = []
    for requested in batch_sizes:
        if cuda_available():
            try:
                safe_capacity = compute_safe_gpu_batch_size(*shape)
            except Exception:
                safe_capacity = requested
        else:
            safe_capacity = requested
        n = max(1, min(requested, len(images_pool), safe_capacity))
        subset = images_pool[:n]

        row = {"requested_batch_size": requested, "effective_batch_size": n}
        if run_cpu:
            r = run_cpu_batch(subset, config)
            row["cpu_images_per_second"] = n / (r.total_ms / 1000.0) if r.total_ms > 0 else 0.0
            row["cpu_ms"] = r.total_ms
        if run_basic:
            r = run_gpu_batch(subset, config, use_enhanced=False)
            row["basic_images_per_second"] = n / (r.total_ms / 1000.0) if r.total_ms > 0 else 0.0
            row["basic_ms"] = r.total_ms
        if run_enhanced:
            r = run_gpu_batch(subset, config, use_enhanced=True)
            row["enhanced_images_per_second"] = n / (r.total_ms / 1000.0) if r.total_ms > 0 else 0.0
            row["enhanced_ms"] = r.total_ms
        rows.append(row)

    return {"resolution": shape, "rows": rows, "timestamp_utc": datetime.now(timezone.utc).isoformat()}


# -- Section 15: saving live batch results (item 30-31, never benchmark_results/) -----------------------------------------------------


def save_batch_results(
    batch_result: BatchComparisonResult,
    preview_stage_outputs: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    preview_image: Optional[np.ndarray] = None,
    output_root: Optional[Path] = None,
) -> Path:
    """Saves batch-level metadata/selection/correctness/timing as JSON,
    plus (only) the representative preview image's per-implementation
    outputs as PNGs -- NOT every processed image (item 29: batches can
    be hundreds of images; saving all of them would be wasteful and is
    not what a "representative" save is for). A fresh, timestamped
    directory every call -- never benchmark_results/, never overwritten."""
    import json

    import cv2

    root = output_root or (Path(__file__).resolve().parent.parent / "outputs" / "live_batch_processing")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    totals = batch_result.implementation_totals()
    metadata = {
        "run_id": run_id,
        "timestamp_utc": batch_result.timestamp_utc,
        "seed": batch_result.seed,
        "requested_batch_size": batch_result.requested_batch_size,
        "effective_count": batch_result.effective_count,
        "dataset_fingerprint": batch_result.dataset_fingerprint,
        "resolution_group_summary": batch_result.resolution_group_summary,
        "filter_config": batch_result.filter_config,
        "load_ms": batch_result.load_ms,
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)

    with open(run_dir / "selection.json", "w", encoding="utf-8") as fh:
        json.dump({"selected_relative_paths": batch_result.selected_relative_paths}, fh, indent=2)

    with open(run_dir / "timing.json", "w", encoding="utf-8") as fh:
        json.dump(totals, fh, indent=2, default=str)

    with open(run_dir / "correctness.json", "w", encoding="utf-8") as fh:
        json.dump([asdict(c) for c in batch_result.correctness], fh, indent=2)

    label_to_dirname = {"CPU": "cpu", "Basic CUDA": "basic_cuda", "Enhanced CUDA": "enhanced_cuda"}
    if preview_image is not None:
        cv2.imwrite(str(run_dir / "original.png"), preview_image)
    if preview_stage_outputs:
        for label, stages in preview_stage_outputs.items():
            stage_dir = run_dir / label_to_dirname[label]
            stage_dir.mkdir(parents=True, exist_ok=True)
            for stage, img in stages.items():
                if img is None:
                    continue
                cv2.imwrite(str(stage_dir / f"{stage}.png"), img)

    return run_dir


# -- Section 17: Interactive CUDA Optimization Lab -----------------------------------------------------
# Strict safety rule (spec item 3): this section only LOADS already-written historical JSON and RUNS
# already-compiled kernels via cuda.optimization_lab.measure_variant() -- it never compiles anything,
# never touches CUDA source, never overwrites benchmark_results/'s Section 11 canonical artifacts or
# the production Enhanced defaults, and never writes to benchmark_results/ itself (live experiments go
# to outputs/optimization_lab/, mirroring Sections 14-15's live/historical separation).


def optimization_lab_filters() -> List[str]:
    return list(OPTIMIZATION_LAB_FILTERS)


def optimization_lab_variants(filter_name: str) -> Tuple[str, ...]:
    """"basic" plus every already-implemented Enhanced variant for
    `filter_name` -- falls back to a static, CUDA-import-independent
    list (so the selector can still render its labels) when the
    xray_cuda extension isn't built at all; a live run still requires
    cuda_available()."""
    if _CUDA_IMPORT_ERROR is None:
        try:
            return _opt_lab_available_variants(filter_name)
        except ValueError as exc:
            raise ServiceError(str(exc)) from exc
    if filter_name not in _OPTIMIZATION_LAB_VARIANTS_FALLBACK:
        raise ServiceError(f"Unknown filter: {filter_name!r}")
    return ("basic",) + _OPTIMIZATION_LAB_VARIANTS_FALLBACK[filter_name]


def optimization_lab_production_default(filter_name: str) -> str:
    if filter_name not in OPTIMIZATION_LAB_PRODUCTION_DEFAULTS:
        raise ServiceError(f"Unknown filter: {filter_name!r}")
    return OPTIMIZATION_LAB_PRODUCTION_DEFAULTS[filter_name]


def _variant_sweeps_root(results_root=None) -> Path:
    from cuda.final_benchmark import DEFAULT_RESULTS_ROOT

    return (results_root or DEFAULT_RESULTS_ROOT) / "variant_sweeps"


def load_variant_sweep(filter_name: str, results_root=None) -> Optional[dict]:
    """Loads the historical, already-measured variant sweep for
    `filter_name` written by scripts/run_optimization_lab_experiments.py
    -- never re-measured from the UI (spec item 8)."""
    import json

    path = _variant_sweeps_root(results_root) / f"{filter_name}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_fusion_experiment(results_root=None) -> Optional[dict]:
    """Loads the historical Laplacian+Threshold fusion experiment (spec
    item 19), or None if it was never measured/backfilled -- never
    fabricated."""
    import json

    path = _variant_sweeps_root(results_root) / "fusion.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class VariantRunResult:
    variant: str
    is_production_default: bool
    kernel_ms_runs: List[float]
    kernel_ms_mean: float
    final_outputs: List[np.ndarray]


@dataclass
class VariantComparisonResult:
    filter_name: str
    variant_a: str
    variant_b: str
    filter_config: dict
    n_images: int
    warmup_runs: int
    measurement_runs: int
    result_a: VariantRunResult
    result_b: VariantRunResult
    speedup_a_over_b: float  # variant_a's mean kernel_ms / variant_b's -- >1 means B is faster
    correctness_a_vs_b: dict
    correctness_a_vs_cpu: dict
    correctness_b_vs_cpu: dict
    timestamp_utc: str


def _cpu_reference_for_filter(filter_name: str, images: List[np.ndarray], config: FilterConfig) -> List[np.ndarray]:
    from cpu.filters import apply_gaussian, apply_laplacian, apply_median, apply_sobel, apply_threshold

    apply_fn = {"gaussian": apply_gaussian, "median": apply_median, "sobel": apply_sobel,
                "laplacian": apply_laplacian, "threshold": apply_threshold}[filter_name]
    return [apply_fn(img, config) for img in images]


def compare_filter_variants(
    images: List[np.ndarray], config: FilterConfig, filter_name: str, variant_a: str, variant_b: str,
    warmup_runs: int = 2, measurement_runs: int = 5,
) -> VariantComparisonResult:
    """Live A/B comparison of two already-compiled variants of ONE
    filter (spec items 20-26): the exact same images/configuration/
    batch reach both variants (never resampled between them), using the
    project's configured warmup convention by default. Every measured
    value -- kernel timings, speedup, correctness -- comes from this
    run; nothing is hardcoded or estimated.
    """
    _require_cuda()
    if filter_name not in OPTIMIZATION_LAB_FILTERS:
        raise ServiceError(f"Unknown filter: {filter_name!r}; expected one of {OPTIMIZATION_LAB_FILTERS}")
    valid_variants = optimization_lab_variants(filter_name)
    if variant_a not in valid_variants:
        raise ServiceError(f"variant_a={variant_a!r} is not valid for {filter_name}; expected one of {valid_variants}")
    if variant_b not in valid_variants:
        raise ServiceError(f"variant_b={variant_b!r} is not valid for {filter_name}; expected one of {valid_variants}")
    if not images:
        raise ServiceError("No images selected for the live variant comparison.")
    shape = images[0].shape
    if any(img.shape != shape for img in images):
        raise ServiceError("A live variant comparison requires all selected images to share one resolution.")

    batch = np.stack(images, axis=0)
    try:
        times_a, out_a = _opt_lab_measure_variant(filter_name, batch, config, variant_a, warmup_runs, measurement_runs)
        times_b, out_b = _opt_lab_measure_variant(filter_name, batch, config, variant_b, warmup_runs, measurement_runs)
    except Exception as exc:  # noqa: BLE001 -- surfaced as a ServiceError (spec item 42: never crash the app)
        raise ServiceError(f"Live variant comparison failed for {filter_name} ({variant_a} vs {variant_b}): {exc}") from exc

    from cuda.benchmark import CorrectnessMetrics

    cpu_expected = _cpu_reference_for_filter(filter_name, images, config)
    correctness_a_vs_cpu = asdict(CorrectnessMetrics.compare(cpu_expected, out_a))
    correctness_b_vs_cpu = asdict(CorrectnessMetrics.compare(cpu_expected, out_b))
    correctness_a_vs_b = asdict(CorrectnessMetrics.compare(list(out_a), out_b))

    mean_a = statistics.mean(times_a)
    mean_b = statistics.mean(times_b)
    production_default = OPTIMIZATION_LAB_PRODUCTION_DEFAULTS[filter_name]

    return VariantComparisonResult(
        filter_name=filter_name, variant_a=variant_a, variant_b=variant_b, filter_config=asdict(config),
        n_images=len(images), warmup_runs=warmup_runs, measurement_runs=measurement_runs,
        result_a=VariantRunResult(variant_a, variant_a == production_default, times_a, mean_a, list(out_a)),
        result_b=VariantRunResult(variant_b, variant_b == production_default, times_b, mean_b, list(out_b)),
        speedup_a_over_b=(mean_a / mean_b) if mean_b > 0 else 0.0,
        correctness_a_vs_b=correctness_a_vs_b, correctness_a_vs_cpu=correctness_a_vs_cpu,
        correctness_b_vs_cpu=correctness_b_vs_cpu, timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


@dataclass
class PipelineImpactResult:
    filter_name: str
    variant: str
    basic_pipeline_ms: float
    enhanced_pipeline_ms: float
    improvement: float
    warmup_runs: int
    measurement_runs: int


def measure_pipeline_impact(
    images: List[np.ndarray], config: FilterConfig, filter_name: str, variant: str,
    warmup_runs: int = 2, measurement_runs: int = 5,
) -> PipelineImpactResult:
    """Spec item 33: whole-5-filter-pipeline effect of swapping ONLY
    `filter_name` from Basic to `variant`, all other stages left Basic
    -- measured directly (never inferred by multiplying the isolated
    per-filter speedup), reusing run_basic_cuda_pipeline() exactly as
    the production pipeline does."""
    _require_cuda()
    if not images:
        raise ServiceError("No images selected for the pipeline-impact measurement.")
    shape = images[0].shape
    if any(img.shape != shape for img in images):
        raise ServiceError("Pipeline-impact measurement requires all selected images to share one resolution.")

    kwargs = {f"use_enhanced_{filter_name}": True, f"{filter_name}_variant": variant}
    try:
        for _ in range(warmup_runs):
            run_basic_cuda_pipeline(images, config)
            run_basic_cuda_pipeline(images, config, **kwargs)
        basic_times, enhanced_times = [], []
        for _ in range(measurement_runs):
            _out_b, t_b = run_basic_cuda_pipeline(images, config)
            basic_times.append(t_b.total_ms)
            _out_e, t_e = run_basic_cuda_pipeline(images, config, **kwargs)
            enhanced_times.append(t_e.total_ms)
    except Exception as exc:  # noqa: BLE001
        raise ServiceError(f"Pipeline-impact measurement failed: {exc}") from exc

    basic_mean = statistics.mean(basic_times)
    enhanced_mean = statistics.mean(enhanced_times)
    return PipelineImpactResult(
        filter_name=filter_name, variant=variant, basic_pipeline_ms=basic_mean, enhanced_pipeline_ms=enhanced_mean,
        improvement=(basic_mean / enhanced_mean) if enhanced_mean > 0 else 0.0,
        warmup_runs=warmup_runs, measurement_runs=measurement_runs,
    )


DEFAULT_OPTIMIZATION_LAB_OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "optimization_lab"


def save_optimization_experiment(
    comparison: VariantComparisonResult, dataset_fingerprint: Optional[str] = None,
    selected_relative_paths: Optional[List[str]] = None, output_root: Optional[Path] = None,
) -> Path:
    """Spec items 36-37: saves ONE live variant-comparison experiment's
    metadata + timing + correctness (never the raw image outputs -- this
    is a timing/correctness record, not an image export) to a fresh,
    timestamped `outputs/optimization_lab/{run_id}/` directory -- never
    `benchmark_results/`, never overwritten."""
    import json

    root = output_root or DEFAULT_OPTIMIZATION_LAB_OUTPUT_ROOT
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    device = device_info() or {}
    metadata = {
        "run_id": run_id, "timestamp_utc": comparison.timestamp_utc, "filter": comparison.filter_name,
        "variant_a": comparison.variant_a, "variant_b": comparison.variant_b,
        "dataset_fingerprint": dataset_fingerprint, "selected_relative_paths": selected_relative_paths or [],
        "n_images": comparison.n_images, "filter_config": comparison.filter_config,
        "warmup_runs": comparison.warmup_runs, "measurement_runs": comparison.measurement_runs,
        "gpu_name": device.get("name"), "cuda_runtime_version": device.get("runtime_version"),
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)

    timing = {
        "variant_a": {"variant": comparison.variant_a, "kernel_ms_runs": comparison.result_a.kernel_ms_runs,
                      "kernel_ms_mean": comparison.result_a.kernel_ms_mean},
        "variant_b": {"variant": comparison.variant_b, "kernel_ms_runs": comparison.result_b.kernel_ms_runs,
                      "kernel_ms_mean": comparison.result_b.kernel_ms_mean},
        "speedup_a_over_b": comparison.speedup_a_over_b,
    }
    with open(run_dir / "timing.json", "w", encoding="utf-8") as fh:
        json.dump(timing, fh, indent=2)

    correctness = {
        "variant_a_vs_variant_b": comparison.correctness_a_vs_b,
        "variant_a_vs_cpu": comparison.correctness_a_vs_cpu,
        "variant_b_vs_cpu": comparison.correctness_b_vs_cpu,
    }
    with open(run_dir / "correctness.json", "w", encoding="utf-8") as fh:
        json.dump(correctness, fh, indent=2)

    return run_dir
