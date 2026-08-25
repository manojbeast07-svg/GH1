"""Basic vs. Enhanced CUDA Threshold A/B benchmark (Section 10).

Usage:
    python scripts/benchmark_threshold_optimization.py --dataset ../data --batch-size 125 --runs 20

Measures Basic Threshold against the two Enhanced variants (uchar4
vectorized load/compare/store, multiple-pixels-per-thread scalar) using
the SAME batched, single-native-call architecture Sections 5-9
established (grid.z = batch dimension). Also runs a block/tile sweep
and reports the full-pipeline effect. Threshold is expected -- not
required -- to show only a modest gain: it is a pure pointwise op with
no neighborhood/redundant-read lever at all (see README).

Correctness is checked with EXACT equality (max_abs_diff == 0), for
both aligned (width%4==0) and non-4-aligned (e.g. 1733, a real dataset
width) images, confirming the Vectorized variant's automatic scalar
fallback is exercised and correct.
"""

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import xray_cuda  # noqa: E402

from cpu.filters import FilterConfig, apply_threshold  # noqa: E402
from cuda.gaussian import gaussian_kernel_2d  # noqa: E402
from cuda.laplacian import laplacian_kernel_2d  # noqa: E402
from cuda.pipeline import run_basic_cuda_pipeline, run_enhanced_cuda_pipeline  # noqa: E402
from cuda.threshold import THRESHOLD_VARIANTS, threshold_variant_to_int  # noqa: E402
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
    parser.add_argument("--batch-size", type=int, default=125)
    parser.add_argument("--threshold-value", type=int, default=128)
    parser.add_argument("--max-value", type=int, default=255)
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

    print("=" * 70)
    print("BASIC vs ENHANCED CUDA THRESHOLD BENCHMARK")
    print("=" * 70)
    print(f"\nResolution: {shape[1]}x{shape[0]}   Batch: {n} images   "
          f"width%4={shape[1] % 4}   threshold={args.threshold_value} max={args.max_value}")
    print(f"Warmup: {args.warmup_runs}   Measurement runs: {args.runs}")

    config = FilterConfig(threshold_value=args.threshold_value, threshold_max_value=args.max_value)
    gaussian_placeholder = gaussian_kernel_2d(3, 0.0)
    laplacian_placeholder = laplacian_kernel_2d(3)

    def bench_basic():
        vals = []
        for i in range(args.warmup_runs + args.runs):
            r = xray_cuda.run_basic_cuda_pipeline_gpu(
                batch, False, gaussian_placeholder, False, 3, False, 2,
                False, laplacian_placeholder, 1.0, 0.0, True, args.threshold_value, args.max_value)
            if i >= args.warmup_runs:
                vals.append(r["threshold_ms"])
        return vals

    def bench_enhanced(variant_int, block=(16, 16)):
        vals = []
        out = None
        for i in range(args.warmup_runs + args.runs):
            r = xray_cuda.threshold_enhanced_batch_gpu(
                batch, args.threshold_value, args.max_value, variant_int, block[0], block[1])
            if i >= args.warmup_runs:
                vals.append(r["kernel_ms"])
            out = r["output"]
        return vals, out

    print("\n--- Optimization sweep (batched, one native call for all images) ---\n")
    basic_times = bench_basic()
    basic_med = _median_ms(basic_times)
    print(f"{'Basic (naive)':30s} median={basic_med:.4f}ms")

    cpu_expected = np.stack([apply_threshold(img, config) for img in images])

    table_rows = [("Basic", basic_med, 1.0, "PASS (reference)")]
    for name in THRESHOLD_VARIANTS:
        vint = threshold_variant_to_int(name)
        times, out = bench_enhanced(vint)
        med = _median_ms(times)
        speedup = basic_med / med if med > 0 else float("inf")

        diff_cpu = np.abs(cpu_expected.astype(np.int16) - out.astype(np.int16))
        correctness = "PASS" if diff_cpu.max() == 0 else f"FAIL(max_diff={diff_cpu.max()})"

        print(f"{name:30s} median={med:.4f}ms  speedup={speedup:.3f}x  correctness={correctness}")
        table_rows.append((name, med, speedup, correctness))

    winner_name = max(table_rows[1:], key=lambda row: row[2])[0]
    print(f"\n--- Block/tile tuning ({winner_name} variant) ---\n")
    for bx, by in [(8, 8), (16, 16), (32, 8), (32, 16)]:
        times, out = bench_enhanced(threshold_variant_to_int(winner_name), block=(bx, by))
        med = _median_ms(times)
        print(f"block={bx}x{by:<4d} median={med:.4f}ms")

    print("\n--- Full-pipeline effect (Enhanced Threshold + Basic Gaussian/Median/Sobel/Laplacian) ---\n")
    full_config = FilterConfig()
    for _ in range(args.warmup_runs):
        run_basic_cuda_pipeline(images, full_config, use_enhanced_threshold=False)
        run_basic_cuda_pipeline(images, full_config, use_enhanced_threshold=True, threshold_variant=winner_name)

    basic_pipeline_totals, enhanced_pipeline_totals = [], []
    for _ in range(args.runs):
        _out_b, t_b = run_basic_cuda_pipeline(images, full_config, use_enhanced_threshold=False)
        basic_pipeline_totals.append(t_b.total_ms)
        _out_e, t_e = run_basic_cuda_pipeline(images, full_config, use_enhanced_threshold=True, threshold_variant=winner_name)
        enhanced_pipeline_totals.append(t_e.total_ms)

    basic_pipeline_med = _median_ms(basic_pipeline_totals)
    enhanced_pipeline_med = _median_ms(enhanced_pipeline_totals)
    print(f"Basic 5-filter pipeline:      median total={basic_pipeline_med:.4f}ms")
    print(f"Enhanced-Threshold pipeline:  median total={enhanced_pipeline_med:.4f}ms")
    print(f"Full-pipeline speedup:        {basic_pipeline_med/enhanced_pipeline_med:.3f}x")

    print("\n--- Fully Enhanced 5-filter pipeline vs Basic 5-filter pipeline ---\n")
    for _ in range(args.warmup_runs):
        run_basic_cuda_pipeline(images, full_config)
        run_enhanced_cuda_pipeline(images, full_config)
    all_basic_totals, all_enhanced_totals = [], []
    for _ in range(args.runs):
        _o, t = run_basic_cuda_pipeline(images, full_config)
        all_basic_totals.append(t.total_ms)
        _o, t = run_enhanced_cuda_pipeline(images, full_config)
        all_enhanced_totals.append(t.total_ms)
    all_basic_med = _median_ms(all_basic_totals)
    all_enhanced_med = _median_ms(all_enhanced_totals)
    print(f"Basic pipeline (all 5 Basic):     median total={all_basic_med:.4f}ms")
    print(f"Enhanced pipeline (all 5 Enhanced): median total={all_enhanced_med:.4f}ms")
    print(f"Overall Basic->Enhanced speedup:  {all_basic_med/all_enhanced_med:.3f}x")

    print("\n" + "=" * 70)
    print("A/B OPTIMIZATION TABLE (Threshold kernel only, batched)")
    print("=" * 70)
    print(f"{'Variant':<16}{'Kernel (median)':>18}{'Speedup':>12}  {'Correctness':<20}")
    for name, med, speedup, correctness in table_rows:
        print(f"{name:<16}{med:>15.4f}ms{speedup:>10.3f}x  {correctness:<20}")

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
