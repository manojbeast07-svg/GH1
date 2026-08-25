"""Basic CUDA median filter demonstration + CPU-vs-GPU comparison.

Usage:
    python scripts/test_median_cuda.py --image path/to/xray.jpg --kernel-size 3

If --image is omitted, one real X-ray is selected via the Section 2
dataset manager (config.yaml dataset.path, seed 42). Loads the image,
runs the CPU reference and the Basic CUDA kernel, compares outputs
(median filtering uses only integer pixel values, so max_abs_diff is
expected to be exactly 0 -- not a tolerance), prints cold/warm GPU
timings plus CPU timing, and saves original/CPU/CUDA/difference images.
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

from cpu.filters import FilterConfig, apply_median  # noqa: E402
from cuda.benchmark import benchmark_median_single_image  # noqa: E402
from cuda.median import median_cuda  # noqa: E402
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
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--save-dir", type=str, default="outputs/median_cuda_demo")
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

    print("=" * 60)
    print("BASIC CUDA MEDIAN FILTER")
    print("=" * 60)
    print(f"\nImage:\n{image_path}")
    h, w = image.shape
    print(f"\nResolution:\n{w} x {h}")
    print(f"\nKernel size:\n{args.kernel_size}")

    config = FilterConfig(median_kernel_size=args.kernel_size)
    cpu_output = apply_median(image, config)
    cuda_output = median_cuda(image, kernel_size=args.kernel_size)

    diff = np.abs(cpu_output.astype(np.int16) - cuda_output.astype(np.int16))
    max_diff = int(diff.max())
    num_differing = int(np.count_nonzero(diff))

    print("\nCPU vs CUDA comparison:")
    print(f"  max_abs_diff:     {max_diff}")
    print(f"  differing_pixels: {num_differing} / {diff.size}")

    correct = max_diff == 0  # exact-equality requirement (Section 4C spec item 15)
    print(f"\nCorrectness (exact match required):\n{'PASS' if correct else 'FAIL'}")

    bench = benchmark_median_single_image(
        image, kernel_size=args.kernel_size,
        warmup_runs=args.warmup_runs, measurement_runs=args.measurement_runs,
    )
    print(f"\nGPU cold end-to-end (first call in this run):\n{bench.cold_end_to_end_ms:.4f} ms")
    _print_stats("GPU warm H2D", bench.warm["h2d_ms"])
    _print_stats("GPU warm kernel (CUDA events)", bench.warm["kernel_ms"])
    _print_stats("GPU warm D2H", bench.warm["d2h_ms"])
    _print_stats("GPU warm end-to-end", bench.warm["end_to_end_ms"])
    _print_stats("CPU processing (time.perf_counter)", bench.cpu_processing_ms)

    cv2.imwrite(str(save_dir / "original.png"), image)
    cv2.imwrite(str(save_dir / "cpu_output.png"), cpu_output)
    cv2.imwrite(str(save_dir / "cuda_output.png"), cuda_output)
    diff_vis = np.clip(diff * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(str(save_dir / "difference.png"), diff_vis)
    print(f"\nSaved outputs to:\n{save_dir}")

    print("\n" + "=" * 60)
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
