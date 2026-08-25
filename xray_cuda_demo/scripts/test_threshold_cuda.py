"""Basic CUDA binary threshold demonstration + CPU-vs-GPU comparison.

Usage:
    python scripts/test_threshold_cuda.py --image path/to/xray.jpg --threshold 128 --max-value 255

If --image is omitted, one real X-ray is selected via the Section 2
dataset manager (config.yaml dataset.path, seed 42). Loads the image,
runs the CPU reference and the Basic CUDA kernel, compares outputs
(threshold is a single integer comparison, so max_abs_diff is expected
to be exactly 0 -- verified empirically, not a tolerance), prints
cold/warm GPU timings plus CPU timing, and saves
original/CPU/CUDA/difference images.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import xray_cuda  # noqa: E402

from cpu.filters import FilterConfig, apply_threshold  # noqa: E402
from cuda.benchmark import benchmark_threshold_single_image  # noqa: E402
from cuda.threshold import threshold_cuda  # noqa: E402
from pipeline.dataset import DatasetManager  # noqa: E402
from pipeline.image_loader import load_image  # noqa: E402


def _default_image_path() -> Path:
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    raw_path = (config.get("dataset") or {}).get("path")
    if not raw_path:
        raise SystemExit("No --image given and config.yaml has no dataset.path set.")
    dataset_path = Path(raw_path)
    if not dataset_path.is_absolute():
        dataset_path = (PROJECT_ROOT / dataset_path).resolve()

    dm = DatasetManager(dataset_path)
    dm.scan()
    selection = dm.random_batch(batch_size=1, seed=42)
    return selection.selected_paths[0]


def _print_stats(label: str, stats) -> None:
    print(f"\n{label}:")
    print(f"  mean={stats.mean:.4f} ms  median={stats.median:.4f} ms  "
          f"min={stats.min:.4f} ms  max={stats.max:.4f} ms  std={stats.std:.4f} ms")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--max-value", type=int, default=255)
    parser.add_argument("--save-dir", type=str, default="outputs/threshold_cuda_demo")
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--measurement-runs", type=int, default=5)
    args = parser.parse_args()

    if not xray_cuda.cuda_available():
        print("No usable CUDA device detected. Cannot proceed.")
        return 1

    image_path = Path(args.image) if args.image else _default_image_path()
    image = load_image(image_path)

    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = (PROJECT_ROOT / save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    print("THRESHOLD CUDA TEST")
    print("-" * 40)
    h, w = image.shape
    print(f"\nImage:\n{image_path}")
    print(f"\nResolution:\n{w} x {h}")
    print(f"\nThreshold:\n{args.threshold}")
    print(f"\nMax value:\n{args.max_value}")

    config = FilterConfig(threshold_value=args.threshold, threshold_max_value=args.max_value)
    cpu_output = apply_threshold(image, config)
    cuda_output = threshold_cuda(image, threshold_value=args.threshold, max_value=args.max_value)

    diff = np.abs(cpu_output.astype(np.int16) - cuda_output.astype(np.int16))
    max_diff = int(diff.max())
    num_differing = int(np.count_nonzero(diff))
    correct = max_diff == 0

    bench = benchmark_threshold_single_image(
        image, threshold_value=args.threshold, max_value=args.max_value,
        warmup_runs=args.warmup_runs, measurement_runs=args.measurement_runs,
    )

    print(f"\nCPU:\n{bench.cpu_processing_ms.mean:.4f} ms (mean of {bench.measurement_runs} runs)")
    _print_stats("GPU H2D", bench.warm["h2d_ms"])
    _print_stats("GPU kernel", bench.warm["kernel_ms"])
    _print_stats("GPU D2H", bench.warm["d2h_ms"])
    _print_stats("GPU end-to-end", bench.warm["end_to_end_ms"])
    print(f"\nGPU cold end-to-end (first call in this run):\n{bench.cold_end_to_end_ms:.4f} ms")

    print(f"\nMax difference:\n{max_diff}")
    print(f"\nDiffering pixels:\n{num_differing}")
    print(f"\nCorrectness:\n{'PASS' if correct else 'FAIL'}")

    cv2.imwrite(str(save_dir / "original.png"), image)
    cv2.imwrite(str(save_dir / "cpu_threshold.png"), cpu_output)
    cv2.imwrite(str(save_dir / "cuda_threshold.png"), cuda_output)
    diff_vis = np.clip(diff, 0, 255).astype(np.uint8)
    cv2.imwrite(str(save_dir / "difference.png"), diff_vis)
    print(f"\nSaved outputs to:\n{save_dir}")

    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
