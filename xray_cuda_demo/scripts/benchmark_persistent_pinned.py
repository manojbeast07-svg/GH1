"""Section 20B: benchmarks the EXPERIMENTAL PersistentCudaPipeline
(GPU-buffer persistence + pinned host memory) against the production
baseline, on the real dataset, using a fair A/B methodology (same
images, same order, same config, same warmup/run counts for every
variant).

NEVER touches the production run_basic_cuda_pipeline()/
run_enhanced_cuda_pipeline() Python API or cuda/pipeline.py -- calls
only xray_cuda.PersistentCudaPipeline directly, and (for the baseline)
xray_cuda.run_basic_cuda_pipeline_gpu itself (the same production entry
point, used unmodified as Variant A's reference).

Usage:
    python scripts/benchmark_persistent_pinned.py --dataset ../data --seed 42 --batch-size 125 --runs 7
    python scripts/benchmark_persistent_pinned.py --dataset ../data --seed 42 --batch-sweep 1,8,32,122,512
"""

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import xray_cuda  # noqa: E402

from cpu.filters import FilterConfig, apply_gaussian, apply_laplacian, apply_median, apply_sobel, apply_threshold  # noqa: E402
from cuda.gaussian import gaussian_kernel_1d, gaussian_kernel_2d  # noqa: E402
from cuda.laplacian import laplacian_kernel_2d  # noqa: E402
from pipeline.dataset import DatasetManager  # noqa: E402
from pipeline.environment import get_environment_fingerprint  # noqa: E402
from pipeline.image_loader import load_image  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "benchmark_results" / "research_optimization" / "persistence_pinned"

VARIANT_NAMES = {
    "baseline": (False, False),           # Variant A: pageable + fresh allocate (matches production)
    "persistent": (True, False),          # Variant B: pageable + persistent GPU buffers
    "pinned": (False, True),              # Variant C: pinned + fresh allocate
    "persistent_pinned": (True, True),    # Variant D: pinned + persistent GPU buffers
}


def _load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_dataset_path(explicit, config: dict) -> Path:
    if explicit:
        return Path(explicit).resolve()
    raw_path = (config.get("dataset") or {}).get("path")
    if not raw_path:
        raise SystemExit("No --dataset given and config.yaml has no dataset.path set.")
    p = Path(raw_path)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def cpu_reference(images, config: FilterConfig) -> np.ndarray:
    return np.stack([
        apply_threshold(apply_laplacian(apply_sobel(apply_median(apply_gaussian(img, config), config), config), config), config)
        for img in images
    ])


def run_variant(batch, config: FilterConfig, coeffs2d, coeffs1d, lap_coeffs, use_persistent, use_pinned,
                 warmup_runs, measurement_runs):
    """Runs one variant with a FRESH PersistentCudaPipeline instance (so
    'persistent' truly means persisted only ACROSS this variant's own
    repeated calls, never leaking state from a previous variant) and
    returns (list_of_per_run_dicts, final_output_array)."""
    pipeline = xray_cuda.PersistentCudaPipeline()
    try:
        per_run = []
        output = None
        for i in range(warmup_runs + measurement_runs):
            result = pipeline.run(
                batch, True, coeffs2d, True, config.median_kernel_size, True, 2,
                True, lap_coeffs, config.laplacian_scale, config.laplacian_delta,
                True, config.threshold_value, config.threshold_max_value,
                gaussian_use_enhanced=True, gaussian_coeffs_1d=coeffs1d, gaussian_variant=3,
                median_use_enhanced=True, median_variant=1,
                sobel_use_enhanced=True, sobel_variant=2,
                laplacian_use_enhanced=True, laplacian_variant=2,
                threshold_use_enhanced=True, threshold_variant=0,
                use_persistent_buffers=use_persistent, use_pinned_memory=use_pinned,
            )
            output = result.pop("output")
            if i >= warmup_runs:
                per_run.append(result)
        return per_run, output
    finally:
        pipeline.release()


def summarize(per_run: list, field: str) -> dict:
    values = [r[field] for r in per_run]
    return {
        "mean": statistics.mean(values), "median": statistics.median(values),
        "min": min(values), "max": max(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0, "n": len(values),
    }


def benchmark_one_batch_size(images_pool, config, coeffs2d, coeffs1d, lap_coeffs, batch_size, warmup_runs, measurement_runs):
    batch_images = images_pool[:batch_size]
    if len(batch_images) < batch_size:
        raise ValueError(f"Requested batch_size={batch_size} but only {len(batch_images)} images of this "
                          f"resolution are available.")
    batch = np.stack(batch_images, axis=0)
    cpu_expected = cpu_reference(batch_images, config)

    variants = {}
    reference_output = None
    for name, (use_persistent, use_pinned) in VARIANT_NAMES.items():
        per_run, output = run_variant(batch, config, coeffs2d, coeffs1d, lap_coeffs, use_persistent, use_pinned,
                                       warmup_runs, measurement_runs)
        if reference_output is None:
            reference_output = output
        bit_exact_vs_baseline = bool(np.array_equal(output, reference_output))
        diff = np.abs(output.astype(np.int16) - cpu_expected.astype(np.int16))
        differing_pct = 100.0 * np.count_nonzero(diff) / diff.size

        variants[name] = {
            "h2d_ms": summarize(per_run, "h2d_ms"),
            "gaussian_ms": summarize(per_run, "gaussian_ms"),
            "median_ms": summarize(per_run, "median_ms"),
            "sobel_ms": summarize(per_run, "sobel_ms"),
            "laplacian_ms": summarize(per_run, "laplacian_ms"),
            "threshold_ms": summarize(per_run, "threshold_ms"),
            "d2h_ms": summarize(per_run, "d2h_ms"),
            "alloc_ms": summarize(per_run, "alloc_ms"),
            "host_stage_ms": summarize(per_run, "host_stage_ms"),
            "total_ms": summarize(per_run, "total_ms"),
            "images_per_second": batch_size / (statistics.median(r["total_ms"] for r in per_run) / 1000.0),
            "bit_exact_vs_baseline": bit_exact_vs_baseline,
            "differing_pixel_percentage_vs_cpu": differing_pct,
            "cold_first_run": per_run[0] if per_run else None,  # informational only, not used in the medians above
        }

    baseline_total = variants["baseline"]["total_ms"]["median"]
    for name in VARIANT_NAMES:
        variants[name]["gain_vs_baseline"] = baseline_total / variants[name]["total_ms"]["median"]

    return {"batch_size": batch_size, "variants": variants}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=125)
    parser.add_argument("--batch-sweep", type=str, default=None, help="Comma-separated batch sizes, e.g. 1,8,32,122,512")
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--runs", type=int, default=7)
    args = parser.parse_args()

    if not xray_cuda.cuda_available():
        print("No usable CUDA device detected. Cannot proceed.")
        return 1

    dataset_path = _resolve_dataset_path(args.dataset, _load_config())
    dm = DatasetManager(dataset_path)
    dm.scan()
    dataset_fingerprint = dm.fingerprint()

    batch_sizes = [int(b) for b in args.batch_sweep.split(",")] if args.batch_sweep else [args.batch_size]
    max_needed = max(batch_sizes)

    selection = dm.random_batch(batch_size=max_needed, seed=args.seed)
    images = [load_image(p) for p in selection.selected_paths]
    shape = images[0].shape
    images_pool = [img for img in images if img.shape == shape]
    if len(images_pool) < max_needed:
        print(f"Warning: only {len(images_pool)} of {max_needed} requested images share the dominant "
              f"resolution {shape}; batch sizes above {len(images_pool)} will fail.")

    config = FilterConfig()
    coeffs2d = gaussian_kernel_2d(config.gaussian_kernel_size, config.gaussian_sigma)
    coeffs1d = gaussian_kernel_1d(config.gaussian_kernel_size, config.gaussian_sigma)
    lap_coeffs = laplacian_kernel_2d(config.laplacian_kernel_size)

    print("=" * 70)
    print("SECTION 20B -- BUFFER PERSISTENCE + PINNED HOST MEMORY EXPERIMENT")
    print("=" * 70)
    print(f"Resolution: {shape[1]}x{shape[0]}   seed={args.seed}   warmup={args.warmup_runs}   runs={args.runs}")
    print(f"Batch sizes: {batch_sizes}")

    results_by_batch = []
    for batch_size in batch_sizes:
        if batch_size > len(images_pool):
            print(f"\n--- batch_size={batch_size} SKIPPED: only {len(images_pool)} images of the dominant "
                  f"resolution are available in this seed=42 selection (never resampling to fill the gap) ---")
            continue
        print(f"\n--- batch_size={batch_size} ---")
        result = benchmark_one_batch_size(images_pool, config, coeffs2d, coeffs1d, lap_coeffs, batch_size,
                                           args.warmup_runs, args.runs)
        results_by_batch.append(result)
        for name in VARIANT_NAMES:
            v = result["variants"][name]
            print(f"  {name:<20} total={v['total_ms']['median']:8.4f}ms  alloc={v['alloc_ms']['median']:7.4f}ms  "
                  f"h2d={v['h2d_ms']['median']:7.4f}ms  d2h={v['d2h_ms']['median']:7.4f}ms  "
                  f"gain_vs_baseline={v['gain_vs_baseline']:.3f}x  "
                  f"bit_exact={v['bit_exact_vs_baseline']}  differing_vs_cpu={v['differing_pixel_percentage_vs_cpu']:.4f}%")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    experiment_id = "persistence_pinned_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    env = get_environment_fingerprint().to_dict()

    manifest = {
        "experiment_id": experiment_id, "timestamp_utc": timestamp_utc, "environment": env,
        "dataset_fingerprint": dataset_fingerprint, "seed": args.seed, "resolution": [shape[0], shape[1]],
        "filter_config": {
            "gaussian_kernel_size": config.gaussian_kernel_size, "gaussian_sigma": config.gaussian_sigma,
            "median_kernel_size": config.median_kernel_size, "laplacian_kernel_size": config.laplacian_kernel_size,
            "laplacian_scale": config.laplacian_scale, "laplacian_delta": config.laplacian_delta,
            "threshold_value": config.threshold_value, "threshold_max_value": config.threshold_max_value,
        },
        "implementation_config": {"gaussian_variant": "specialized", "median_variant": "network3x3",
                                   "sobel_variant": "specialized", "laplacian_variant": "specialized",
                                   "threshold_variant": "vectorized", "block": "16x16"},
        "warmup_runs": args.warmup_runs, "measurement_runs": args.runs, "batch_sizes": batch_sizes,
        "variants": list(VARIANT_NAMES.keys()), "results": results_by_batch,
    }
    out_path = RESULTS_DIR / f"{experiment_id}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    latest_path = RESULTS_DIR / "latest.json"
    with open(latest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    print(f"\nWrote results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
