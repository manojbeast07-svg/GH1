"""Single-image CPU pipeline demonstration.

Usage:
    python scripts/run_cpu_pipeline.py --image path/to/xray.jpg --save-dir outputs/cpu_demo

Loads one image, runs the configured CPU reference pipeline on it, saves
every enabled intermediate output plus the original and final images, and
prints per-filter + total timing (measured with time.perf_counter(),
never fabricated).
"""

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cpu.filters import FilterConfig  # noqa: E402
from cpu.pipeline import run_cpu_image  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the CPU reference pipeline on a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--save-dir", type=str, default="outputs/cpu_demo", help="Where to save outputs.")
    parser.add_argument("--gaussian-kernel", type=int, default=5)
    parser.add_argument("--median-kernel", type=int, default=3)
    parser.add_argument("--sobel-mode", type=str, default="magnitude", choices=["x", "y", "magnitude", "abs_sum"])
    parser.add_argument("--laplacian-kernel", type=int, default=3)
    parser.add_argument("--threshold-value", type=int, default=128)
    args = parser.parse_args()

    image_path = Path(args.image)
    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = (PROJECT_ROOT / save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    config = FilterConfig(
        gaussian_kernel_size=args.gaussian_kernel,
        median_kernel_size=args.median_kernel,
        sobel_mode=args.sobel_mode,
        laplacian_kernel_size=args.laplacian_kernel,
        threshold_value=args.threshold_value,
    )

    result, timing = run_cpu_image(image_path, config)

    print("=" * 60)
    print("CPU PIPELINE")
    print("=" * 60)
    print(f"\nImage:\n{image_path}")
    h, w = result.original.shape
    print(f"\nResolution:\n{w} x {h}")

    cv2.imwrite(str(save_dir / "original.png"), result.original)
    for stage_name in ("gaussian", "median", "sobel", "laplacian", "threshold"):
        output = result.get(stage_name)
        elapsed = getattr(timing, f"{stage_name}_ms")
        label = stage_name.capitalize()
        if output is None:
            print(f"\n{label}:\ndisabled")
            continue
        cv2.imwrite(str(save_dir / f"{stage_name}.png"), output)
        print(f"\n{label}:\n{elapsed:.3f} ms")

    cv2.imwrite(str(save_dir / "final.png"), result.final_output)
    print(f"\nTotal:\n{timing.total_ms:.3f} ms")
    print(f"\nSaved outputs to:\n{save_dir}")
    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
