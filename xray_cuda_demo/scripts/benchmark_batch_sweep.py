"""CPU vs. Basic CUDA batch-size sweep (Section 5).

Usage:
    python scripts/benchmark_batch_sweep.py --resolution 224x224 --sizes 1,8,16,32,64,128,256,512

For a single resolution (default: 224x224, the dominant resolution in
the real dataset -- 9,273/9,463 images per Section 2), runs
benchmark_basic_cuda_pipeline() at each requested batch size (skipping
any that exceed the resolution group's available image count or the
live safe GPU capacity) and reports CPU/GPU throughput and speedup as a
function of batch size. Writes one CSV with all sweep points.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import xray_cuda  # noqa: E402

from cpu.filters import FilterConfig  # noqa: E402
from cuda.benchmark import benchmark_basic_cuda_pipeline  # noqa: E402
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


def main() -> int:
    config_yaml = _load_config()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--resolution", type=str, default="224x224", help="WIDTHxHEIGHT to sweep.")
    parser.add_argument("--sizes", type=str, default="1,8,16,32,64,128,256,512")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--pool-size", type=int, default=600,
                         help="How many images to select before filtering to the target resolution "
                              "(must be large enough to contain --sizes worth of that resolution).")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    if not xray_cuda.cuda_available():
        print("No usable CUDA device detected. Cannot proceed.")
        return 1

    width_str, height_str = args.resolution.lower().split("x")
    target_width, target_height = int(width_str), int(height_str)
    requested_sizes = [int(s) for s in args.sizes.split(",")]

    dataset_path = _resolve_dataset_path(args.dataset, config_yaml)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dm = DatasetManager(dataset_path)
    dm.scan()
    selection = dm.random_batch(batch_size=args.pool_size, seed=args.seed)
    groups = group_by_resolution(selection)
    items = groups.get((target_height, target_width), [])

    print("=" * 70)
    print(f"BATCH SIZE SWEEP: {target_width}x{target_height}")
    print("=" * 70)
    print(f"\nDataset: {dataset_path}")
    print(f"Pool selection: {args.pool_size} images (seed={args.seed}); "
          f"{len(items)} match {target_width}x{target_height}")

    if not items:
        print(f"\nNo images of resolution {target_width}x{target_height} found in the pool; "
              f"try a larger --pool-size.")
        return 1

    safe_capacity = compute_safe_gpu_batch_size(target_height, target_width)
    config = FilterConfig()

    csv_path = output_dir / "benchmark_batch_sweep_results.csv"
    write_header = not csv_path.exists()
    fieldnames = [
        "timestamp_utc", "resolution", "requested_batch_size", "effective_batch_size",
        "cpu_total_ms", "gpu_total_ms", "cpu_images_per_second", "gpu_images_per_second",
        "gpu_speedup", "differing_pixel_percentage",
    ]

    print(f"\nSafe GPU capacity at this resolution: {safe_capacity}")
    print(f"\n{'batch':>8} {'available':>10} {'CPU img/s':>12} {'GPU img/s':>12} {'speedup':>9}")

    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for requested in requested_sizes:
            effective = min(requested, len(items), safe_capacity)
            if effective <= 0:
                print(f"{requested:>8} {len(items):>10}   (skipped: no images available)")
                continue

            chosen_items = items[:effective]
            load_start = time.perf_counter()
            images = [load_image(item.absolute_path) for item in chosen_items]
            load_ms = (time.perf_counter() - load_start) * 1000.0

            result = benchmark_basic_cuda_pipeline(
                images, config, warmup_runs=args.warmup_runs, measurement_runs=args.runs,
                requested_batch_size=requested, seed=args.seed, load_ms=load_ms,
            )

            print(f"{requested:>8} {len(items):>10} {result.cpu_images_per_second:>12.2f} "
                  f"{result.gpu_images_per_second:>12.2f} {result.gpu_speedup:>8.3f}x")

            writer.writerow({
                "timestamp_utc": result.timestamp_utc,
                "resolution": f"{target_width}x{target_height}",
                "requested_batch_size": requested,
                "effective_batch_size": effective,
                "cpu_total_ms": result.cpu_total_ms,
                "gpu_total_ms": result.gpu_total_ms,
                "cpu_images_per_second": result.cpu_images_per_second,
                "gpu_images_per_second": result.gpu_images_per_second,
                "gpu_speedup": result.gpu_speedup,
                "differing_pixel_percentage": result.correctness.differing_pixel_percentage,
            })
            fh.flush()

    print(f"\nCSV results saved to: {csv_path}")
    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
