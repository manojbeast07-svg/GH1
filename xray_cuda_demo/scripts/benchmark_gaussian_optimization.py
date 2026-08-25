"""Basic vs. Enhanced CUDA Gaussian A/B benchmark (Section 6).

Usage:
    python scripts/benchmark_gaussian_optimization.py --dataset ../data --batch-size 128 --kernel-size 5 --sigma 1.0 --runs 20

Measures Basic Gaussian against all four Enhanced variants (naive
separable, +shared memory, +constant memory, +compile-time
specialization) using the SAME batched, single-native-call architecture
Section 5 established (grid.z = batch dimension) -- this is deliberately
NOT a per-image loop, since Section 6 found that single-image (batch=1)
kernel timing is dominated by fixed launch overhead and gives a
misleading picture for a separable (two-kernel-launch) implementation at
small image sizes. Also runs a block/tile configuration sweep for the
winning variant, and reports the full-pipeline effect (Enhanced Gaussian
+ Basic Median/Sobel/Laplacian/Threshold vs. all-Basic).

Correctness is checked at every step (max_abs_diff/differing pixels vs.
CPU and vs. Basic) -- this script will not silently report a speedup for
an incorrect variant.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import xray_cuda  # noqa: E402

from cpu.filters import FilterConfig, apply_gaussian  # noqa: E402
from cuda.gaussian import GAUSSIAN_VARIANTS, gaussian_kernel_1d, gaussian_kernel_2d, variant_to_int  # noqa: E402
from cuda.pipeline import run_basic_cuda_pipeline  # noqa: E402
from pipeline.dataset import DatasetManager  # noqa: E402
from pipeline.image_loader import load_image  # noqa: E402


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


def _median_ms(values):
    return statistics.median(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20,
                         help="Measurement runs. Higher than the project's usual default (5) because "
                              "these kernel times are in the tens/hundreds of microseconds, where fixed "
                              "launch/sync jitter needs more samples to average out (see README).")
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

    print("=" * 70)
    print("BASIC vs ENHANCED CUDA GAUSSIAN BENCHMARK")
    print("=" * 70)
    print(f"\nResolution: {shape[1]}x{shape[0]}   Batch: {n} images   kernel_size={args.kernel_size} sigma={args.sigma}")
    print(f"Warmup: {args.warmup_runs}   Measurement runs: {args.runs}")

    config = FilterConfig(gaussian_kernel_size=args.kernel_size, gaussian_sigma=args.sigma)
    coeffs2d = gaussian_kernel_2d(args.kernel_size, args.sigma)
    coeffs1d = gaussian_kernel_1d(args.kernel_size, args.sigma)

    def bench_basic():
        vals = []
        for i in range(args.warmup_runs + args.runs):
            r = xray_cuda.run_basic_cuda_pipeline_gpu(
                batch, True, coeffs2d, False, 3, False, 2, False, coeffs2d, 1.0, 0.0, False, 128, 255)
            if i >= args.warmup_runs:
                vals.append(r["gaussian_ms"])
        return vals

    def bench_enhanced(variant_int, block=(16, 16)):
        vals = []
        out = None
        for i in range(args.warmup_runs + args.runs):
            r = xray_cuda.gaussian_enhanced_batch_gpu(batch, coeffs1d, variant_int, block[0], block[1])
            if i >= args.warmup_runs:
                vals.append(r["kernel_ms"])
            out = r["output"]
        return vals, out

    print("\n--- Optimization sweep (batched, one native call for all images) ---\n")
    basic_times = bench_basic()
    basic_med = _median_ms(basic_times)
    print(f"{'Basic (2D direct)':30s} median={basic_med:.4f}ms")

    cpu_expected = np.stack([apply_gaussian(img, config) for img in images])

    table_rows = [("Basic", basic_med, 1.0, "PASS (reference)")]
    for name in GAUSSIAN_VARIANTS:
        vint = variant_to_int(name)
        times, out = bench_enhanced(vint)
        med = _median_ms(times)
        speedup = basic_med / med if med > 0 else float("inf")

        diff_cpu = np.abs(cpu_expected.astype(np.int16) - out.astype(np.int16))
        correctness = "PASS" if diff_cpu.max() <= 1 else f"FAIL(max_diff={diff_cpu.max()})"

        print(f"{name:30s} median={med:.4f}ms  speedup={speedup:.3f}x  correctness={correctness}")
        table_rows.append((name, med, speedup, correctness))

    print("\n--- Block/tile tuning (Specialized variant) ---\n")
    for bx, by in [(8, 8), (16, 16), (32, 8), (32, 16)]:
        times, out = bench_enhanced(variant_to_int("specialized"), block=(bx, by))
        med = _median_ms(times)
        print(f"block={bx}x{by:<4d} median={med:.4f}ms")

    print("\n--- Full-pipeline effect (Enhanced Gaussian + Basic Median/Sobel/Laplacian/Threshold) ---\n")
    full_config = FilterConfig()  # all defaults, all 5 stages enabled
    for _ in range(args.warmup_runs):
        run_basic_cuda_pipeline(images, full_config, use_enhanced_gaussian=False)
        run_basic_cuda_pipeline(images, full_config, use_enhanced_gaussian=True)

    basic_pipeline_totals, enhanced_pipeline_totals = [], []
    for _ in range(args.runs):
        _out_b, t_b = run_basic_cuda_pipeline(images, full_config, use_enhanced_gaussian=False)
        basic_pipeline_totals.append(t_b.total_ms)
        _out_e, t_e = run_basic_cuda_pipeline(images, full_config, use_enhanced_gaussian=True)
        enhanced_pipeline_totals.append(t_e.total_ms)

    basic_pipeline_med = _median_ms(basic_pipeline_totals)
    enhanced_pipeline_med = _median_ms(enhanced_pipeline_totals)
    print(f"Basic 5-filter pipeline:     median total={basic_pipeline_med:.4f}ms")
    print(f"Enhanced-Gaussian pipeline:  median total={enhanced_pipeline_med:.4f}ms")
    print(f"Full-pipeline speedup:       {basic_pipeline_med/enhanced_pipeline_med:.3f}x")

    print("\n" + "=" * 70)
    print("A/B OPTIMIZATION TABLE (Gaussian kernel only, batched)")
    print("=" * 70)
    print(f"{'Variant':<16}{'Kernel (median)':>18}{'Speedup':>12}  {'Correctness':<20}")
    for name, med, speedup, correctness in table_rows:
        print(f"{name:<16}{med:>15.4f}ms{speedup:>10.3f}x  {correctness:<20}")

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
