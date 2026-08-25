"""CPU vs. Basic CUDA vs. Enhanced CUDA final pipeline benchmark (Section 10).

Usage:
    python scripts/benchmark_final_pipelines.py --dataset ../data --batch-size 125 --seed 42 --runs 5

Discovers the dataset, makes ONE deterministic selection, groups it by
resolution (the real dataset is mixed-resolution -- see
cuda/pipeline.py::group_by_resolution), and benchmarks the SAME images
through all three implementations: the full Section 3 CPU pipeline, the
production Basic CUDA pipeline (run_basic_cuda_pipeline, all five
filters Basic), and the production Enhanced CUDA pipeline
(run_enhanced_cuda_pipeline, all five filters using each stage's
Section 6-10 measured-best variant). Writes JSON (one file per
resolution group) and appends a CSV summary row per group -- this is
the dataset the future Streamlit dashboard (Section 11) will read.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import xray_cuda  # noqa: E402

from cpu.benchmark import DEFAULT_MEASUREMENT_RUNS, DEFAULT_WARMUP_RUNS  # noqa: E402
from cpu.filters import FilterConfig  # noqa: E402
from cuda.benchmark import benchmark_cpu_basic_enhanced_pipelines  # noqa: E402
from cuda.pipeline import compute_safe_gpu_batch_size, group_by_resolution  # noqa: E402
from pipeline.dataset import DatasetManager  # noqa: E402
from pipeline.image_loader import load_image  # noqa: E402


def _load_config() -> dict:
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_dataset_path(explicit, config: dict) -> Path:
    if explicit:
        return Path(explicit).resolve()
    raw_path = (config.get("dataset") or {}).get("path")
    if not raw_path:
        raise SystemExit("No --dataset given and config.yaml has no dataset.path set.")
    dataset_path = Path(raw_path)
    if not dataset_path.is_absolute():
        dataset_path = (PROJECT_ROOT / dataset_path).resolve()
    return dataset_path


def _print_stats(label: str, stats) -> None:
    if stats is None:
        print(f"{label}: (disabled)")
        return
    print(f"{label}: mean={stats.mean:.4f}ms median={stats.median:.4f}ms "
          f"min={stats.min:.4f}ms max={stats.max:.4f}ms std={stats.std:.4f}ms")


def _write_csv_row(csv_path: Path, row: dict) -> None:
    import csv

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    config_yaml = _load_config()
    benchmark_defaults = config_yaml.get("benchmark", {})

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=125, help="Requested selection batch size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=benchmark_defaults.get("measurement_runs", DEFAULT_MEASUREMENT_RUNS))
    parser.add_argument("--warmup-runs", type=int, default=benchmark_defaults.get("warmup_runs", DEFAULT_WARMUP_RUNS))
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    if not xray_cuda.cuda_available():
        print("No usable CUDA device detected. Cannot proceed.")
        return 1

    dataset_path = _resolve_dataset_path(args.dataset, config_yaml)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dm = DatasetManager(dataset_path)
    dm.scan()
    fingerprint = dm.fingerprint()
    selection = dm.random_batch(batch_size=args.batch_size, seed=args.seed)

    groups = group_by_resolution(selection)
    config = FilterConfig()

    print("=" * 70)
    print("CPU vs BASIC CUDA vs ENHANCED CUDA -- FINAL PIPELINE BENCHMARK")
    print("=" * 70)
    print(f"\nDataset: {dataset_path}")
    print(f"Dataset fingerprint: {fingerprint}")
    print(f"Selection: {len(selection)} images (seed={args.seed})")
    print(f"Resolution groups found: {[(res, len(items)) for res, items in groups.items()]}")

    for (height, width), items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        safe_capacity = compute_safe_gpu_batch_size(height, width)
        effective_count = min(len(items), safe_capacity)
        chosen_items = items[:effective_count]

        load_start = time.perf_counter()
        images = [load_image(item.absolute_path) for item in chosen_items]
        load_ms = (time.perf_counter() - load_start) * 1000.0

        print(f"\n{'-'*70}")
        print(f"Resolution group: {width}x{height}  "
              f"(requested {len(items)}, safe GPU capacity {safe_capacity}, using {effective_count})")

        result = benchmark_cpu_basic_enhanced_pipelines(
            images, config,
            warmup_runs=args.warmup_runs, measurement_runs=args.runs,
            requested_batch_size=len(items), dataset_fingerprint=fingerprint, seed=args.seed, load_ms=load_ms,
        )

        print(f"\nCPU load:       {result.cpu_load_ms:.3f} ms")
        _print_stats("CPU processing", result.cpu_processing_ms)
        print(f"CPU total:      {result.cpu_total_ms:.3f} ms")

        print(f"\n-- Basic CUDA --")
        _print_stats("H2D       ", result.basic_h2d_ms)
        _print_stats("Gaussian  ", result.basic_gaussian_ms)
        _print_stats("Median    ", result.basic_median_ms)
        _print_stats("Sobel     ", result.basic_sobel_ms)
        _print_stats("Laplacian ", result.basic_laplacian_ms)
        _print_stats("Threshold ", result.basic_threshold_ms)
        _print_stats("compute   ", result.basic_compute_ms)
        _print_stats("D2H       ", result.basic_d2h_ms)
        print(f"Basic total: {result.basic_total_ms:.3f} ms")

        print(f"\n-- Enhanced CUDA --")
        _print_stats("H2D       ", result.enhanced_h2d_ms)
        _print_stats("Gaussian  ", result.enhanced_gaussian_ms)
        _print_stats("Median    ", result.enhanced_median_ms)
        _print_stats("Sobel     ", result.enhanced_sobel_ms)
        _print_stats("Laplacian ", result.enhanced_laplacian_ms)
        _print_stats("Threshold ", result.enhanced_threshold_ms)
        _print_stats("compute   ", result.enhanced_compute_ms)
        _print_stats("D2H       ", result.enhanced_d2h_ms)
        print(f"Enhanced total: {result.enhanced_total_ms:.3f} ms")

        print(f"\nCPU throughput:      {result.cpu_images_per_second:.2f} images/sec")
        print(f"Basic throughput:    {result.basic_images_per_second:.2f} images/sec")
        print(f"Enhanced throughput: {result.enhanced_images_per_second:.2f} images/sec")

        print(f"\nBasic speedup vs CPU:       {result.basic_speedup_vs_cpu:.3f}x")
        print(f"Enhanced speedup vs CPU:    {result.enhanced_speedup_vs_cpu:.3f}x")
        print(f"Enhanced optimization gain (vs Basic): {result.enhanced_speedup_vs_basic:.3f}x")

        print(f"\nCorrectness (Basic vs CPU):")
        c = result.correctness_basic_vs_cpu
        print(f"  max_abs_diff={c.max_abs_diff}  mean_abs_diff={c.mean_abs_diff:.4f}  rmse={c.rmse:.4f}  "
              f"differing={c.differing_pixel_percentage:.4f}%")

        print(f"Correctness (Enhanced vs CPU):")
        c = result.correctness_enhanced_vs_cpu
        print(f"  max_abs_diff={c.max_abs_diff}  mean_abs_diff={c.mean_abs_diff:.4f}  rmse={c.rmse:.4f}  "
              f"differing={c.differing_pixel_percentage:.4f}%")

        print(f"Correctness (Enhanced vs Basic):")
        c = result.correctness_enhanced_vs_basic
        print(f"  max_abs_diff={c.max_abs_diff}  mean_abs_diff={c.mean_abs_diff:.4f}  rmse={c.rmse:.4f}  "
              f"differing={c.differing_pixel_percentage:.4f}%")

        timestamp_tag = result.timestamp_utc.replace("+00:00", "Z").replace(":", "-")
        json_path = output_dir / f"benchmark_final_pipelines_{width}x{height}_{timestamp_tag}.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2)
        print(f"\nJSON detail saved to: {json_path}")

        csv_path = output_dir / "benchmark_final_pipelines_results.csv"
        _write_csv_row(csv_path, {
            "timestamp_utc": result.timestamp_utc,
            "gpu_name": result.gpu_name,
            "driver_version": result.driver_version,
            "cuda_toolkit_version": result.cuda_toolkit_version,
            "dataset_fingerprint": result.dataset_fingerprint,
            "seed": result.seed,
            "resolution": f"{width}x{height}",
            "requested_batch_size": result.requested_batch_size,
            "effective_gpu_batch_size": result.effective_gpu_batch_size,
            "cpu_total_ms": result.cpu_total_ms,
            "basic_total_ms": result.basic_total_ms,
            "enhanced_total_ms": result.enhanced_total_ms,
            "cpu_images_per_second": result.cpu_images_per_second,
            "basic_images_per_second": result.basic_images_per_second,
            "enhanced_images_per_second": result.enhanced_images_per_second,
            "basic_speedup_vs_cpu": result.basic_speedup_vs_cpu,
            "enhanced_speedup_vs_cpu": result.enhanced_speedup_vs_cpu,
            "enhanced_speedup_vs_basic": result.enhanced_speedup_vs_basic,
            "basic_vs_cpu_differing_pixel_pct": result.correctness_basic_vs_cpu.differing_pixel_percentage,
            "enhanced_vs_cpu_differing_pixel_pct": result.correctness_enhanced_vs_cpu.differing_pixel_percentage,
            "enhanced_vs_basic_differing_pixel_pct": result.correctness_enhanced_vs_basic.differing_pixel_percentage,
            "json_detail_file": json_path.name,
        })
        print(f"CSV summary appended to: {csv_path}")

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
