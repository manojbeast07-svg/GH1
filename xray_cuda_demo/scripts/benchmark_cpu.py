"""Batch CPU benchmark CLI.

Usage:
    python scripts/benchmark_cpu.py --dataset ../data --batch-size 100 --seed 42 --runs 5

If --dataset is omitted, the path is read from config.yaml (dataset.path).
warmup/measurement run counts default to config.yaml's benchmark section
if not given explicitly. Results are written as JSON (full detail) and
appended as one row to a CSV summary, both under outputs/.

No GPU columns yet -- this is the CPU-only baseline (Section 3). Later
sections add Basic/Enhanced CUDA columns to the same result schema.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cpu.benchmark import DEFAULT_MEASUREMENT_RUNS, DEFAULT_WARMUP_RUNS, run_selection_benchmark  # noqa: E402
from cpu.filters import FilterConfig  # noqa: E402
from pipeline.dataset import DatasetManager  # noqa: E402


def _load_config() -> dict:
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_dataset_path(explicit: "str | None", config: dict) -> Path:
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
    benchmark_defaults = config_yaml.get("benchmark", {})

    parser = argparse.ArgumentParser(description="Batch CPU benchmark.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset root (default: config.yaml dataset.path).")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=benchmark_defaults.get("measurement_runs", DEFAULT_MEASUREMENT_RUNS),
                         help="Measurement runs (default: config.yaml benchmark.measurement_runs).")
    parser.add_argument("--warmup-runs", type=int, default=benchmark_defaults.get("warmup_runs", DEFAULT_WARMUP_RUNS))
    parser.add_argument("--mode", choices=["processing", "end_to_end"], default="processing")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    dataset_path = _resolve_dataset_path(args.dataset, config_yaml)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dm = DatasetManager(dataset_path)
    dm.scan()
    fingerprint = dm.fingerprint()
    selection = dm.random_batch(batch_size=args.batch_size, seed=args.seed)

    config = FilterConfig()
    result = run_selection_benchmark(
        selection,
        config,
        warmup_runs=args.warmup_runs,
        measurement_runs=args.runs,
        timing_mode=args.mode,
        dataset_fingerprint=fingerprint,
    )

    print("=" * 60)
    print("CPU BATCH BENCHMARK")
    print("=" * 60)
    print(f"\nDataset:\n{dataset_path}")
    print(f"\nImages:\n{result.num_images}")
    print(f"\nMode:\n{result.metadata.timing_mode}")
    print(f"\nWarmup / measurement runs:\n{args.warmup_runs} / {args.runs}")
    print(f"\nTotal processing time (mean per full-batch pass):\n{result.total_time_ms:.3f} ms")
    print(f"\nAverage per image:\n{result.avg_time_per_image_ms:.3f} ms")
    print(f"\nThroughput:\n{result.throughput_images_per_sec:.2f} images/sec")
    print("\nPer-filter mean:")
    for name, value in result.per_filter_mean_ms.items():
        print(f"  {name.capitalize()}:\n  {value:.3f} ms")

    timestamp_tag = result.metadata.timestamp_utc.replace("+00:00", "Z").replace(":", "-")
    json_path = output_dir / f"benchmark_cpu_{timestamp_tag}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2)

    csv_path = output_dir / "benchmark_cpu_results.csv"
    csv_row = {
        "timestamp_utc": result.metadata.timestamp_utc,
        "implementation": "cpu",
        "num_images": result.num_images,
        "seed": result.metadata.selection_seed,
        "batch_size": result.metadata.selection_batch_size,
        "timing_mode": result.metadata.timing_mode,
        "warmup_runs": result.metadata.warmup_runs,
        "measurement_runs": result.metadata.measurement_runs,
        "total_time_ms": result.total_time_ms,
        "avg_time_per_image_ms": result.avg_time_per_image_ms,
        "throughput_images_per_sec": result.throughput_images_per_sec,
        "gaussian_mean_ms": result.per_filter_mean_ms.get("gaussian"),
        "median_mean_ms": result.per_filter_mean_ms.get("median"),
        "sobel_mean_ms": result.per_filter_mean_ms.get("sobel"),
        "laplacian_mean_ms": result.per_filter_mean_ms.get("laplacian"),
        "threshold_mean_ms": result.per_filter_mean_ms.get("threshold"),
        "dataset_fingerprint": result.metadata.dataset_fingerprint,
        "json_detail_file": json_path.name,
    }
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(csv_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(csv_row)

    print(f"\nJSON detail saved to:\n{json_path}")
    print(f"\nCSV summary appended to:\n{csv_path}")
    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
