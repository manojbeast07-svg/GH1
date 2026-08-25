"""Section 17: measures every already-implemented, already-compiled
Enhanced CUDA variant for each filter (the Section 6-10 A/B sweeps),
and the Section 10 Laplacian+Threshold fusion experiment, then writes
structured JSON so the Streamlit Optimization Lab can LOAD real
historical numbers instead of re-running anything from the UI.

Methodology deliberately mirrors each of scripts/benchmark_<filter>_
optimization.py exactly (same direct xray_cuda.<filter>_enhanced_batch_gpu
/ run_basic_cuda_pipeline_gpu calls, same isolated-stage flags) rather
than going through cuda.pipeline.run_cuda_pipeline()'s general-purpose
wrapper -- an earlier draft of this script used run_cuda_pipeline() and
that extra layer's overhead was large enough, relative to these
sub-millisecond kernels, to distort small-kernel comparisons (verified:
k=3 Laplacian appeared to regress under the wrapper while the original
scripts' direct-call method reproduces the historically-documented
~1.51x gain). Using the SAME call path as the validated Section 6-10
scripts avoids repeating that mistake.

This does not compile any CUDA code -- every variant measured here is
an already-built kernel exposed by the compiled `xray_cuda` extension.
It also does not modify cuda/final_benchmark.py's Section 11 canonical
benchmark artifacts (summary/aggregated/manifests/raw/batch_sweeps/
resolution_sweeps/per_filter/correctness) -- this writes to a new,
separate `benchmark_results/variant_sweeps/` subdirectory only.

Usage:
    python scripts/run_optimization_lab_experiments.py --dataset ../data --batch-size 128 --seed 42
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import xray_cuda  # noqa: E402

from cpu.filters import FilterConfig, apply_gaussian, apply_laplacian, apply_median, apply_sobel, apply_threshold  # noqa: E402
from cuda.final_benchmark import AggregatedStat  # noqa: E402
from cuda.gaussian import GAUSSIAN_VARIANTS, gaussian_kernel_1d, gaussian_kernel_2d, variant_to_int as gaussian_variant_to_int  # noqa: E402
from cuda.laplacian import LAPLACIAN_VARIANTS, laplacian_kernel_2d, laplacian_variant_to_int  # noqa: E402
from cuda.median import MEDIAN_VARIANTS, median_variant_to_int  # noqa: E402
from cuda.sobel import SOBEL_VARIANTS, mode_to_int, sobel_variant_to_int  # noqa: E402
from cuda.threshold import THRESHOLD_VARIANTS, threshold_variant_to_int  # noqa: E402
from pipeline.dataset import DatasetManager  # noqa: E402
from pipeline.image_loader import load_image  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "benchmark_results" / "variant_sweeps"

PRODUCTION_DEFAULTS = {
    "gaussian": "specialized", "median": "network3x3", "sobel": "specialized",
    "laplacian": "specialized", "threshold": "vectorized",
}

REJECTED_VARIANT_NOTES = {
    "sobel": {
        "shared": "Shared memory: a modest gain, not the winning variant.",
        "shared_const": "Shared + constant memory: Basic Sobel was already hand-optimized (one launch, "
                         "minimal arithmetic), so there was limited headroom left for this variant to win.",
        "separable": "Separable two-pass: the intermediate pass's extra global-memory round trip outweighed "
                     "the reduced arithmetic for a kernel this small -- rejected as production default.",
    },
    "median": {
        "shared": "Shared memory alone: the real lever for Median was removing the sorting algorithm's "
                  "data-dependent branching, not memory reuse -- shared memory alone gives close to no benefit.",
        "specialized": "Compile-time specialization without the sorting network still uses branching "
                       "insertion sort -- a real but much smaller gain than network3x3's branchless network.",
    },
    "threshold": {
        "multi_pixel": "Processing multiple pixels per thread without widening the memory transaction "
                       "itself: register usage is already minimal (no shared memory, no neighborhood), so "
                       "this variant measures close to flat vs. Basic.",
    },
    "gaussian": {
        "naive": "Separable convolution alone (2 passes, no shared/constant memory): a real algorithmic win "
                 "over Basic's O(k^2) 2-D convolution, but leaves memory-access optimization on the table.",
        "shared": "+ shared memory tiling: reduces redundant global reads for the separable passes.",
        "shared_const": "+ constant memory for the 1-D coefficients: a further, smaller gain once the "
                        "passes are already tiled.",
    },
}


def _cpu_reference(filter_name: str, images, config: FilterConfig):
    apply_fn = {"gaussian": apply_gaussian, "median": apply_median, "sobel": apply_sobel,
                "laplacian": apply_laplacian, "threshold": apply_threshold}[filter_name]
    return np.stack([apply_fn(img, config) for img in images])


def _run_variant_sweep(filter_name, variants, bench_basic, bench_enhanced, cpu_expected, warmup, runs, cpu_tolerance):
    basic_times = bench_basic()
    basic_stat = AggregatedStat.from_values(basic_times)
    rows = [{
        "variant": "basic", "kernel_ms": basic_stat.to_dict(), "speedup_vs_basic": 1.0,
        "is_production_default": False, "max_abs_diff_vs_cpu": None, "correctness": None, "error": None,
    }]
    for variant in variants:
        try:
            times, out = bench_enhanced(variant)
        except Exception as exc:  # noqa: BLE001 -- an unsupported variant/kernel_size combination (e.g.
                                   # network3x3 at kernel_size=5) must be RECORDED, not crash the whole sweep.
            rows.append({
                "variant": variant, "kernel_ms": None, "speedup_vs_basic": None,
                "is_production_default": PRODUCTION_DEFAULTS.get(filter_name) == variant,
                "max_abs_diff_vs_cpu": None, "correctness": None, "error": str(exc),
            })
            continue
        stat = AggregatedStat.from_values(times)
        diff = int(np.abs(cpu_expected.astype(np.int16) - out.astype(np.int16)).max())
        rows.append({
            "variant": variant, "kernel_ms": stat.to_dict(),
            "speedup_vs_basic": (basic_stat.mean / stat.mean) if stat.mean > 0 else None,
            "is_production_default": PRODUCTION_DEFAULTS.get(filter_name) == variant,
            "max_abs_diff_vs_cpu": diff, "correctness": "PASS" if diff <= cpu_tolerance else f"FAIL(max_diff={diff})",
            "error": None,
        })
    return rows


def measure_gaussian(images, batch, warmup, runs, kernel_size=5, sigma=0.0):
    config = FilterConfig(gaussian_kernel_size=kernel_size, gaussian_sigma=sigma)
    coeffs2d = gaussian_kernel_2d(kernel_size, sigma)
    coeffs1d = gaussian_kernel_1d(kernel_size, sigma)
    cpu_expected = _cpu_reference("gaussian", images, config)

    def bench_basic():
        vals = []
        for i in range(warmup + runs):
            r = xray_cuda.run_basic_cuda_pipeline_gpu(
                batch, True, coeffs2d, False, 3, False, 2, False, coeffs2d, 1.0, 0.0, False, 128, 255)
            if i >= warmup:
                vals.append(r["gaussian_ms"])
        return vals

    def bench_enhanced(variant):
        vint = gaussian_variant_to_int(variant)
        vals, out = [], None
        for i in range(warmup + runs):
            r = xray_cuda.gaussian_enhanced_batch_gpu(batch, coeffs1d, vint, 16, 16)
            if i >= warmup:
                vals.append(r["kernel_ms"])
            out = r["output"]
        return vals, out

    return _run_variant_sweep("gaussian", GAUSSIAN_VARIANTS, bench_basic, bench_enhanced, cpu_expected, warmup, runs, 1)


def measure_median(images, batch, warmup, runs, kernel_size):
    config = FilterConfig(median_kernel_size=kernel_size)
    gaussian_placeholder = gaussian_kernel_2d(3, 0.0)
    laplacian_placeholder = laplacian_kernel_2d(3)
    cpu_expected = _cpu_reference("median", images, config)

    def bench_basic():
        vals = []
        for i in range(warmup + runs):
            r = xray_cuda.run_basic_cuda_pipeline_gpu(
                batch, False, gaussian_placeholder, True, kernel_size, False, 2,
                False, laplacian_placeholder, 1.0, 0.0, False, 128, 255)
            if i >= warmup:
                vals.append(r["median_ms"])
        return vals

    def bench_enhanced(variant):
        vint = median_variant_to_int(variant)
        vals, out = [], None
        for i in range(warmup + runs):
            r = xray_cuda.median_enhanced_batch_gpu(batch, kernel_size, vint, 16, 16)
            if i >= warmup:
                vals.append(r["kernel_ms"])
            out = r["output"]
        return vals, out

    applicable = [v for v in MEDIAN_VARIANTS if not (v == "network3x3" and kernel_size != 3)]
    return _run_variant_sweep("median", applicable, bench_basic, bench_enhanced, cpu_expected, warmup, runs, 0)


def measure_sobel(images, batch, warmup, runs, mode="magnitude"):
    config = FilterConfig(sobel_mode=mode, sobel_kernel_size=3)
    gaussian_placeholder = gaussian_kernel_2d(3, 0.0)
    laplacian_placeholder = laplacian_kernel_2d(3)
    mode_int = mode_to_int(mode)
    cpu_expected = _cpu_reference("sobel", images, config)

    def bench_basic():
        vals = []
        for i in range(warmup + runs):
            r = xray_cuda.run_basic_cuda_pipeline_gpu(
                batch, False, gaussian_placeholder, False, 3, True, mode_int,
                False, laplacian_placeholder, 1.0, 0.0, False, 128, 255)
            if i >= warmup:
                vals.append(r["sobel_ms"])
        return vals

    def bench_enhanced(variant):
        vint = sobel_variant_to_int(variant)
        vals, out = [], None
        for i in range(warmup + runs):
            r = xray_cuda.sobel_enhanced_batch_gpu(batch, mode_int, vint, 16, 16)
            if i >= warmup:
                vals.append(r["kernel_ms"])
            out = r["output"]
        return vals, out

    return _run_variant_sweep("sobel", SOBEL_VARIANTS, bench_basic, bench_enhanced, cpu_expected, warmup, runs, 0)


def measure_laplacian(images, batch, warmup, runs, kernel_size, scale=1.0, delta=0.0):
    config = FilterConfig(laplacian_kernel_size=kernel_size, laplacian_scale=scale, laplacian_delta=delta)
    gaussian_placeholder = gaussian_kernel_2d(3, 0.0)
    coeffs = laplacian_kernel_2d(kernel_size)
    cpu_expected = _cpu_reference("laplacian", images, config)

    def bench_basic():
        vals = []
        for i in range(warmup + runs):
            r = xray_cuda.run_basic_cuda_pipeline_gpu(
                batch, False, gaussian_placeholder, False, 3, False, 2,
                True, coeffs, scale, delta, False, 128, 255)
            if i >= warmup:
                vals.append(r["laplacian_ms"])
        return vals

    def bench_enhanced(variant):
        vint = laplacian_variant_to_int(variant)
        vals, out = [], None
        for i in range(warmup + runs):
            r = xray_cuda.laplacian_enhanced_batch_gpu(batch, coeffs, scale, delta, vint, 16, 16)
            if i >= warmup:
                vals.append(r["kernel_ms"])
            out = r["output"]
        return vals, out

    return _run_variant_sweep("laplacian", LAPLACIAN_VARIANTS, bench_basic, bench_enhanced, cpu_expected, warmup, runs, 0)


def measure_threshold(images, batch, warmup, runs, threshold_value=128, max_value=255):
    config = FilterConfig(threshold_value=threshold_value, threshold_max_value=max_value)
    gaussian_placeholder = gaussian_kernel_2d(3, 0.0)
    laplacian_placeholder = laplacian_kernel_2d(3)
    cpu_expected = _cpu_reference("threshold", images, config)

    def bench_basic():
        vals = []
        for i in range(warmup + runs):
            r = xray_cuda.run_basic_cuda_pipeline_gpu(
                batch, False, gaussian_placeholder, False, 3, False, 2,
                False, laplacian_placeholder, 1.0, 0.0, True, threshold_value, max_value)
            if i >= warmup:
                vals.append(r["threshold_ms"])
        return vals

    def bench_enhanced(variant):
        vint = threshold_variant_to_int(variant)
        vals, out = [], None
        for i in range(warmup + runs):
            r = xray_cuda.threshold_enhanced_batch_gpu(batch, threshold_value, max_value, vint, 16, 16)
            if i >= warmup:
                vals.append(r["kernel_ms"])
            out = r["output"]
        return vals, out

    return _run_variant_sweep("threshold", THRESHOLD_VARIANTS, bench_basic, bench_enhanced, cpu_expected, warmup, runs, 0)


def measure_fusion(batch, warmup, runs):
    """Section 10's Laplacian+Threshold fusion experiment: unfused
    (Enhanced Laplacian specialized -> Enhanced Threshold vectorized,
    2 launches) vs. the fused single-launch kernel -- same direct-call
    methodology as the other measure_*() functions above."""
    coeffs = laplacian_kernel_2d(3)
    scale, delta, threshold_value, max_value = 1.0, 0.0, 128, 255
    lap_vint = laplacian_variant_to_int("specialized")
    thr_vint = threshold_variant_to_int("vectorized")

    def bench_unfused():
        vals = []
        for i in range(warmup + runs):
            r_lap = xray_cuda.laplacian_enhanced_batch_gpu(batch, coeffs, scale, delta, lap_vint, 16, 16)
            r_thr = xray_cuda.threshold_enhanced_batch_gpu(r_lap["output"], threshold_value, max_value, thr_vint, 16, 16)
            if i >= warmup:
                vals.append(r_lap["kernel_ms"] + r_thr["kernel_ms"])
        return vals, r_thr["output"]

    def bench_fused():
        vals, out = [], None
        for i in range(warmup + runs):
            r = xray_cuda.laplacian_threshold_fused_batch_gpu(batch, coeffs, scale, delta, threshold_value, max_value, 16, 16)
            if i >= warmup:
                vals.append(r["kernel_ms"])
            out = r["output"]
        return vals, out

    unfused_times, unfused_out = bench_unfused()
    fused_times, fused_out = bench_fused()
    unfused_stat = AggregatedStat.from_values(unfused_times)
    fused_stat = AggregatedStat.from_values(fused_times)
    exact_vs_unfused = int(np.abs(unfused_out.astype(np.int16) - fused_out.astype(np.int16)).max())

    return {
        "unfused_combined_kernel_ms": unfused_stat.to_dict(),
        "fused_kernel_ms": fused_stat.to_dict(),
        "combined_kernel_speedup": (unfused_stat.mean / fused_stat.mean) if fused_stat.mean > 0 else None,
        "max_abs_diff_fused_vs_unfused": exact_vs_unfused,
        "bit_exact_vs_unfused": exact_vs_unfused == 0,
        "wired_into_production": False,
        "note": "Experimental (Section 10): kept as a tested, benchmarked capability; NOT wired into the "
                "production pipeline because it permanently discards the standalone Laplacian output that "
                "per-stage introspection (visualization/benchmarking/correctness) depends on.",
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()

    if not xray_cuda.cuda_available():
        print("No usable CUDA device detected. Cannot proceed.")
        return 1

    dataset_path = _resolve_dataset_path(args.dataset, _load_config())
    dm = DatasetManager(dataset_path)
    dm.scan()
    selection = dm.random_batch(batch_size=args.batch_size, seed=args.seed)
    images = [load_image(p) for p in selection.selected_paths]
    shape = images[0].shape
    images = [img for img in images if img.shape == shape]
    batch = np.stack(images, axis=0)
    n = len(images)
    dataset_fingerprint = dm.fingerprint()

    print(f"Resolution: {shape[1]}x{shape[0]}   Batch: {n} images   seed={args.seed}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    gpu_name = xray_cuda.device_info().get("name")

    common_meta = {
        "timestamp_utc": timestamp_utc, "resolution": [shape[0], shape[1]], "image_count": n, "seed": args.seed,
        "dataset_fingerprint": dataset_fingerprint, "warmup_runs": args.warmup_runs, "measurement_runs": args.runs,
        "gpu_name": gpu_name,
    }

    def _print_rows(rows):
        for r in rows:
            km = r["kernel_ms"]["mean"] if r["kernel_ms"] else "ERROR"
            print(f"  {r['variant']:<14} kernel_ms(mean)={km}  speedup={r['speedup_vs_basic']}  "
                  f"correctness={r['correctness']}  error={r['error']}")

    print("\n--- Gaussian (kernel_size=5) ---")
    rows = measure_gaussian(images, batch, args.warmup_runs, args.runs, kernel_size=5)
    _print_rows(rows)
    with open(RESULTS_DIR / "gaussian.json", "w", encoding="utf-8") as fh:
        json.dump({**common_meta, "filter": "gaussian", "kernel_size": 5, "rows": rows,
                   "rejected_notes": REJECTED_VARIANT_NOTES["gaussian"]}, fh, indent=2)

    print("\n--- Median (kernel_size in {3, 5, 7}) ---")
    median_by_ksize = {}
    for ksize in (3, 5, 7):
        rows = measure_median(images, batch, args.warmup_runs, args.runs, kernel_size=ksize)
        median_by_ksize[str(ksize)] = rows
        print(f" k={ksize}")
        _print_rows(rows)
    with open(RESULTS_DIR / "median.json", "w", encoding="utf-8") as fh:
        json.dump({**common_meta, "filter": "median", "by_kernel_size": median_by_ksize,
                   "rejected_notes": REJECTED_VARIANT_NOTES["median"]}, fh, indent=2)

    print("\n--- Sobel (kernel_size=3) ---")
    rows = measure_sobel(images, batch, args.warmup_runs, args.runs)
    _print_rows(rows)
    with open(RESULTS_DIR / "sobel.json", "w", encoding="utf-8") as fh:
        json.dump({**common_meta, "filter": "sobel", "kernel_size": 3, "rows": rows,
                   "rejected_notes": REJECTED_VARIANT_NOTES["sobel"]}, fh, indent=2)

    print("\n--- Laplacian (kernel_size in {3, 5}) ---")
    laplacian_by_ksize = {}
    for ksize in (3, 5):
        rows = measure_laplacian(images, batch, args.warmup_runs, args.runs, kernel_size=ksize)
        laplacian_by_ksize[str(ksize)] = rows
        print(f" k={ksize}")
        _print_rows(rows)
    with open(RESULTS_DIR / "laplacian.json", "w", encoding="utf-8") as fh:
        json.dump({**common_meta, "filter": "laplacian", "by_kernel_size": laplacian_by_ksize}, fh, indent=2)

    print("\n--- Threshold ---")
    rows = measure_threshold(images, batch, args.warmup_runs, args.runs)
    _print_rows(rows)
    with open(RESULTS_DIR / "threshold.json", "w", encoding="utf-8") as fh:
        json.dump({**common_meta, "filter": "threshold", "rows": rows,
                   "rejected_notes": REJECTED_VARIANT_NOTES["threshold"]}, fh, indent=2)

    print("\n--- Laplacian+Threshold fusion ---")
    fusion = measure_fusion(batch, args.warmup_runs, args.runs)
    print(f"  unfused={fusion['unfused_combined_kernel_ms']['mean']:.4f}ms  "
          f"fused={fusion['fused_kernel_ms']['mean']:.4f}ms  speedup={fusion['combined_kernel_speedup']}")
    with open(RESULTS_DIR / "fusion.json", "w", encoding="utf-8") as fh:
        json.dump({**common_meta, "experiment": "laplacian_threshold_fusion", **fusion}, fh, indent=2)

    print(f"\nWrote variant sweep results to {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
