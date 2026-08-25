"""Section 4A CLI diagnostic: proves a real X-ray reaches the GPU and back.

Usage:
    python scripts/test_cuda_image.py --image path/to/xray.jpg

If --image is omitted, one real image is selected via the Section 2
dataset manager (config.yaml dataset.path, seed 42).

Loads the image, uploads it, runs the trivial +1 kernel, downloads the
result, and verifies it against an independently-computed CPU expectation
-- all timings are measured, never fabricated.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import xray_cuda  # noqa: E402

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=str, default=None, help="Path to an image (default: one real X-ray).")
    args = parser.parse_args()

    print("CUDA IMAGE TEST")
    print("-" * 40)

    if not xray_cuda.cuda_available():
        print("\nNo usable CUDA device detected. Cannot proceed.")
        return 1

    image_path = Path(args.image) if args.image else _default_image_path()
    image = load_image(image_path)
    height, width = image.shape

    print(f"\nImage:\n{image_path}")
    print(f"\nDimensions:\n{width} x {height}")
    print(f"\ndtype:\n{image.dtype}")

    h2d_start = time.perf_counter()
    gpu_image = xray_cuda.upload_image(image)
    h2d_ms = (time.perf_counter() - h2d_start) * 1000.0
    print(f"\nH2D:\n{h2d_ms:.3f} ms ({gpu_image.nbytes} bytes)")

    result = xray_cuda.image_add_one(gpu_image)
    print(f"\nKernel:\n{result['kernel_ms']:.3f} ms (CUDA event timing)")

    d2h_start = time.perf_counter()
    output = xray_cuda.download_image(result["output"])
    d2h_ms = (time.perf_counter() - d2h_start) * 1000.0
    print(f"\nD2H:\n{d2h_ms:.3f} ms")

    expected = (image.astype(np.uint16) + 1).astype(np.uint8)  # documented uint8 wraparound
    correct = np.array_equal(output, expected)
    print(f"\nCorrectness:\n{'PASS' if correct else 'FAIL'}")

    round_trip = xray_cuda.download_image(gpu_image)
    round_trip_ok = np.array_equal(round_trip, image)
    print(f"\nRound-trip (upload -> download, no-op) bit-identical:\n{'PASS' if round_trip_ok else 'FAIL'}")

    mem_info = xray_cuda.device_memory_info()
    print(f"\nGPU memory free/total:\n{mem_info['free_bytes'] / 1e9:.2f} / {mem_info['total_bytes'] / 1e9:.2f} GB")

    return 0 if (correct and round_trip_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
