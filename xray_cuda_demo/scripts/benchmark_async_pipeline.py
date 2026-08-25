"""Section 20F: benchmarks the EXPERIMENTAL AsyncCudaPipeline (multi-stream,
double/triple-buffered chunked pipeline) against production, on the real
dataset, for both Basic and Enhanced kernels.

Deliberately never touches xray_cuda.PersistentCudaPipeline, pinned memory
as a hidden default, CudaGraphPipeline, or CudaGraphEnhancedPipeline --
isolated per this section's spec items 14-15.

Usage:
    python scripts/benchmark_async_pipeline.py --seed 42 --impl basic --batch-sweep 8,32,64,100
    python scripts/benchmark_async_pipeline.py --seed 42 --impl enhanced --chunk-sweep 1,4,8,16,32,64
    python scripts/benchmark_async_pipeline.py --seed 42 --realistic-workload
"""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import xray_cuda  # noqa: E402

from cpu.filters import FilterConfig  # noqa: E402
from cuda.gaussian import gaussian_kernel_1d, gaussian_kernel_2d  # noqa: E402
from cuda.laplacian import laplacian_kernel_2d  # noqa: E402
from cuda.pipeline import run_basic_cuda_pipeline, run_enhanced_cuda_pipeline  # noqa: E402
from cuda.sobel import mode_to_int  # noqa: E402
from pipeline.dataset import DatasetManager  # noqa: E402
from pipeline.environment import get_environment_fingerprint  # noqa: E402
from pipeline.image_loader import load_image  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "benchmark_results" / "research_optimization" / "async_pipeline"


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


def _async_call(pipeline, batch, config, use_enhanced, chunk_size, num_buffers, use_pinned_staging):
    sm = mode_to_int(config.sobel_mode)
    lc = laplacian_kernel_2d(config.laplacian_kernel_size)
    if use_enhanced:
        gc1d = gaussian_kernel_1d(config.gaussian_kernel_size, config.gaussian_sigma)
        return pipeline.run(
            batch=batch, use_enhanced=True, gaussian_enabled=True, gaussian_coeffs_1d=gc1d,
            median_enabled=True, sobel_enabled=True, sobel_mode=sm,
            laplacian_enabled=True, laplacian_coeffs=lc,
            laplacian_scale=config.laplacian_scale, laplacian_delta=config.laplacian_delta,
            threshold_enabled=True, threshold_value=config.threshold_value, threshold_max_value=config.threshold_max_value,
            chunk_size=chunk_size, num_buffers=num_buffers, use_pinned_staging=use_pinned_staging,
        )
    gc = gaussian_kernel_2d(config.gaussian_kernel_size, config.gaussian_sigma)
    return pipeline.run(
        batch=batch, use_enhanced=False, gaussian_enabled=True, gaussian_coeffs=gc,
        median_enabled=True, median_kernel_size=config.median_kernel_size,
        sobel_enabled=True, sobel_mode=sm,
        laplacian_enabled=True, laplacian_coeffs=lc,
        laplacian_scale=config.laplacian_scale, laplacian_delta=config.laplacian_delta,
        threshold_enabled=True, threshold_value=config.threshold_value, threshold_max_value=config.threshold_max_value,
        chunk_size=chunk_size, num_buffers=num_buffers, use_pinned_staging=use_pinned_staging,
    )


def _production_call(images, config, use_enhanced):
    if use_enhanced:
        return run_enhanced_cuda_pipeline(images, config)
    return run_basic_cuda_pipeline(images, config)


def summarize(values: list) -> dict:
    return {
        "mean": statistics.mean(values), "median": statistics.median(values),
        "min": min(values), "max": max(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0, "n": len(values),
    }


def benchmark_one_batch_size(images_pool, config, use_enhanced, batch_size, chunk_size, warmup_runs, measurement_runs):
    imgs = images_pool[:batch_size]
    batch = np.stack(imgs, axis=0)
    expected, _ = _production_call(imgs, config, use_enhanced)

    prod_times = []
    for i in range(warmup_runs + measurement_runs):
        _, t = _production_call(imgs, config, use_enhanced)
        if i >= warmup_runs:
            prod_times.append(t.total_ms)

    results = {}
    for mode_name, num_buffers, pinned in [
        ("sequential_1buf", 1, False),
        ("async_2buf_pageable", 2, False),
        ("async_3buf_pageable", 3, False),
        ("async_2buf_pinned", 2, True),
        ("async_3buf_pinned", 3, True),
    ]:
        pipeline = xray_cuda.AsyncCudaPipeline()
        try:
            times, correct = [], None
            for i in range(warmup_runs + measurement_runs):
                r = _async_call(pipeline, batch, config, use_enhanced, chunk_size, num_buffers, pinned)
                if i >= warmup_runs:
                    times.append(r["total_ms"])
                correct = bool(np.array_equal(r["output"], expected))
            results[mode_name] = {
                "total_ms": summarize(times),
                "bit_exact_vs_production": correct,
                "gain_vs_production": statistics.median(prod_times) / statistics.median(times),
                "num_chunks": r["num_chunks"], "chunk_size_used": r["chunk_size_used"],
                "h2d_compute_overlap_ms": r["h2d_compute_overlap_ms"],
                "compute_d2h_overlap_ms": r["compute_d2h_overlap_ms"],
                "host_stage_ms": r["host_stage_ms"],
            }
        finally:
            pipeline.release()

    return {
        "batch_size": batch_size, "chunk_size": chunk_size,
        "production": {"total_ms": summarize(prod_times)},
        "variants": results,
    }


def chunk_size_sweep(images_pool, config, use_enhanced, batch_size, chunk_sizes, warmup_runs, measurement_runs):
    imgs = images_pool[:batch_size]
    batch = np.stack(imgs, axis=0)
    expected, _ = _production_call(imgs, config, use_enhanced)
    results = []
    for cs in chunk_sizes:
        if cs > batch_size:
            continue
        pipeline = xray_cuda.AsyncCudaPipeline()
        try:
            times = []
            for i in range(warmup_runs + measurement_runs):
                r = _async_call(pipeline, batch, config, use_enhanced, cs, 2, False)
                if i >= warmup_runs:
                    times.append(r["total_ms"])
            results.append({
                "chunk_size": cs, "num_chunks": r["num_chunks"],
                "total_ms": summarize(times),
                "bit_exact": bool(np.array_equal(r["output"], expected)),
            })
        finally:
            pipeline.release()
    return results


def realistic_workload_benchmark(images_pool, config, use_enhanced, chunk_size, num_buffers, passes: int):
    """Reuses Section 20C's fixed measurement methodology exactly."""
    xray_cuda.smoke_test()

    def sequence(pipeline):
        b32 = np.stack(images_pool[:32], axis=0)
        _async_call(pipeline, b32, config, use_enhanced, chunk_size, num_buffers, False)
        import dataclasses
        config2 = dataclasses.replace(config, threshold_value=200)
        _async_call(pipeline, b32, config2, use_enhanced, chunk_size, num_buffers, False)
        b8 = np.stack(images_pool[32:40], axis=0)
        _async_call(pipeline, b8, config, use_enhanced, chunk_size, num_buffers, False)
        b64 = np.stack(images_pool[40:104] if len(images_pool) >= 104 else images_pool[:64], axis=0)
        _async_call(pipeline, b64, config, use_enhanced, chunk_size, num_buffers, False)

    def stateless_sequence():
        b32 = images_pool[:32]
        _production_call(b32, config, use_enhanced)
        import dataclasses
        config2 = dataclasses.replace(config, threshold_value=200)
        _production_call(b32, config2, use_enhanced)
        b8 = images_pool[32:40]
        _production_call(b8, config, use_enhanced)
        b64 = images_pool[40:104] if len(images_pool) >= 104 else images_pool[:64]
        _production_call(b64, config, use_enhanced)

    def timed(fn):
        start = time.perf_counter()
        fn()
        return (time.perf_counter() - start) * 1000.0

    async_pipeline = xray_cuda.AsyncCudaPipeline()
    try:
        for _ in range(2):
            stateless_sequence()
            sequence(async_pipeline)

        stateless_times, async_times = [], []
        for i in range(passes):
            if i % 2 == 0:
                stateless_times.append(timed(stateless_sequence))
                async_times.append(timed(lambda: sequence(async_pipeline)))
            else:
                async_times.append(timed(lambda: sequence(async_pipeline)))
                stateless_times.append(timed(stateless_sequence))
    finally:
        async_pipeline.release()

    return {
        "stateless_total_ms": summarize(stateless_times),
        "async_total_ms": summarize(async_times),
        "gain_async_vs_stateless": statistics.median(stateless_times) / statistics.median(async_times),
        "passes": passes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--impl", choices=["basic", "enhanced"], default="basic")
    parser.add_argument("--batch-sweep", type=str, default=None)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--chunk-sweep", type=str, default=None)
    parser.add_argument("--chunk-sweep-batch", type=int, default=64)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--realistic-workload", action="store_true")
    parser.add_argument("--realistic-passes", type=int, default=5)
    args = parser.parse_args()

    if not xray_cuda.cuda_available():
        print("No usable CUDA device detected. Cannot proceed.")
        return 1

    dataset_path = _resolve_dataset_path(args.dataset, _load_config())
    dm = DatasetManager(dataset_path)
    dm.scan()
    dataset_fingerprint = dm.fingerprint()

    batch_sizes = [int(b) for b in args.batch_sweep.split(",")] if args.batch_sweep else []
    max_needed = max(batch_sizes + [args.chunk_sweep_batch]) if batch_sizes else max(args.chunk_sweep_batch, 104)

    selection = dm.random_batch(batch_size=max(max_needed, 104), seed=args.seed)
    images = [load_image(p) for p in selection.selected_paths]
    shape = images[0].shape
    images_pool = [img for img in images if img.shape == shape]

    config = FilterConfig()
    use_enhanced = args.impl == "enhanced"

    print("=" * 70)
    print(f"SECTION 20F -- ASYNC MULTI-STREAM PIPELINE EXPERIMENT ({args.impl.upper()})")
    print("=" * 70)
    print(f"Resolution: {shape[1]}x{shape[0]}   seed={args.seed}   pool={len(images_pool)} images")

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": get_environment_fingerprint().to_dict(),
        "dataset_fingerprint": dataset_fingerprint, "seed": args.seed, "resolution": [shape[0], shape[1]],
        "impl": args.impl, "warmup_runs": args.warmup_runs, "measurement_runs": args.runs,
    }

    if batch_sizes:
        results_by_batch = []
        for batch_size in batch_sizes:
            if batch_size > len(images_pool):
                print(f"\n--- batch_size={batch_size} SKIPPED: only {len(images_pool)} images available ---")
                continue
            print(f"\n--- batch_size={batch_size} chunk_size={args.chunk_size} ---")
            result = benchmark_one_batch_size(images_pool, config, use_enhanced, batch_size, args.chunk_size,
                                               args.warmup_runs, args.runs)
            results_by_batch.append(result)
            print(f"  production            total={result['production']['total_ms']['median']:8.4f}ms")
            for name, v in result["variants"].items():
                print(f"  {name:<22} total={v['total_ms']['median']:8.4f}ms  gain_vs_prod={v['gain_vs_production']:.3f}x  "
                      f"bit_exact={v['bit_exact_vs_production']}  h2d/compute_overlap={v['h2d_compute_overlap_ms']:.3f}ms")
        manifest["batch_sweep"] = {"batch_sizes": batch_sizes, "chunk_size": args.chunk_size, "results": results_by_batch}

    if args.chunk_sweep:
        chunk_sizes = [int(c) for c in args.chunk_sweep.split(",")]
        print(f"\n--- CHUNK SIZE SWEEP (batch={args.chunk_sweep_batch}) ---")
        chunk_results = chunk_size_sweep(images_pool, config, use_enhanced, args.chunk_sweep_batch, chunk_sizes,
                                          args.warmup_runs, args.runs)
        for r in chunk_results:
            print(f"  chunk_size={r['chunk_size']:<5} chunks={r['num_chunks']:<4} total={r['total_ms']['median']:8.4f}ms  bit_exact={r['bit_exact']}")
        manifest["chunk_sweep"] = {"batch_size": args.chunk_sweep_batch, "results": chunk_results}

    if args.realistic_workload:
        if len(images_pool) < 104:
            print(f"\nWarning: realistic workload needs 104 same-resolution images, only {len(images_pool)} available.")
        print(f"\n--- REALISTIC WORKLOAD (decisive test, {args.realistic_passes} passes, alternating order) ---")
        realistic = realistic_workload_benchmark(images_pool, config, use_enhanced, args.chunk_size, 2, args.realistic_passes)
        print(f"  stateless (production) total: median={realistic['stateless_total_ms']['median']:.4f}ms")
        print(f"  async (AsyncCudaPipeline) total: median={realistic['async_total_ms']['median']:.4f}ms")
        print(f"  gain: {realistic['gain_async_vs_stateless']:.3f}x")
        manifest["realistic_workload"] = realistic

    for sub in ("raw", "aggregated", "chunk_sweeps", "memory", "correctness", "manifests"):
        (RESULTS_DIR / sub).mkdir(parents=True, exist_ok=True)
    experiment_id = f"async_pipeline_{args.impl}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    manifest["experiment_id"] = experiment_id
    out_path = RESULTS_DIR / "manifests" / f"{experiment_id}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    with open(RESULTS_DIR / "manifests" / f"latest_{args.impl}.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"\nWrote results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
