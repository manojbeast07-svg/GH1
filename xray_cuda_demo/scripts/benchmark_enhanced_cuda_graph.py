"""Section 20E: benchmarks the EXPERIMENTAL CudaGraphEnhancedPipeline
(CUDA Graph capture/replay for the five ENHANCED filter kernels, production
variants only) against two baselines, on the real dataset:

  A. TRUE production pipeline -- cuda.pipeline.run_enhanced_cuda_pipeline()
     (all-Enhanced production defaults), completely unmodified.
  B. This experiment's own use_graph=False path -- identical host code to
     the graph path minus capture/replay, isolating exactly the
     graph-vs-no-graph variable.
  C. This experiment's use_graph=True path, both scopes (KernelsOnly and
     FullPipeline).

Deliberately never touches xray_cuda.PersistentCudaPipeline, pinned
memory, or the Basic-kernel CudaGraphPipeline (Section 20D) -- isolated
per this section's spec items 33-35.

Usage:
    python scripts/benchmark_enhanced_cuda_graph.py --seed 42 --batch-sweep 1,8,32,64,100
    python scripts/benchmark_enhanced_cuda_graph.py --seed 42 --realistic-workload
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
from cuda.gaussian import gaussian_kernel_1d  # noqa: E402
from cuda.laplacian import laplacian_kernel_2d  # noqa: E402
from cuda.pipeline import run_enhanced_cuda_pipeline  # noqa: E402
from cuda.sobel import mode_to_int  # noqa: E402
from pipeline.dataset import DatasetManager  # noqa: E402
from pipeline.environment import get_environment_fingerprint  # noqa: E402
from pipeline.image_loader import load_image  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "benchmark_results" / "research_optimization" / "enhanced_cuda_graph"


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


def _graph_call(pipeline, batch, config, gc1d, lc, sm, use_graph, full_pipeline_scope):
    return pipeline.run(
        batch, True, gc1d, True, True, sm, True, lc,
        config.laplacian_scale, config.laplacian_delta,
        True, config.threshold_value, config.threshold_max_value,
        use_graph=use_graph, full_pipeline_scope=full_pipeline_scope,
    )


def summarize(values: list) -> dict:
    return {
        "mean": statistics.mean(values), "median": statistics.median(values),
        "min": min(values), "max": max(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0, "n": len(values),
    }


def benchmark_one_batch_size(images_pool, config, gc1d, lc, sm, batch_size, warmup_runs, measurement_runs):
    imgs = images_pool[:batch_size]
    batch = np.stack(imgs, axis=0)
    expected, _ = run_enhanced_cuda_pipeline(imgs, config)

    prod_times = []
    for i in range(warmup_runs + measurement_runs):
        out, t = run_enhanced_cuda_pipeline(imgs, config)
        if i >= warmup_runs:
            prod_times.append(t.total_ms)

    pipeline = xray_cuda.CudaGraphEnhancedPipeline()
    try:
        variant_times = {"no_graph": [], "graph_kernels_only": [], "graph_full_pipeline": []}
        variant_correct = {}
        for i in range(warmup_runs + measurement_runs):
            r = _graph_call(pipeline, batch, config, gc1d, lc, sm, use_graph=False, full_pipeline_scope=True)
            if i >= warmup_runs:
                variant_times["no_graph"].append(r["total_ms"])
            variant_correct["no_graph"] = bool(np.array_equal(r["output"], expected))
        pipeline.clear_cache()
        for i in range(warmup_runs + measurement_runs):
            r = _graph_call(pipeline, batch, config, gc1d, lc, sm, use_graph=True, full_pipeline_scope=False)
            if i >= warmup_runs:
                variant_times["graph_kernels_only"].append(r["total_ms"])
            variant_correct["graph_kernels_only"] = bool(np.array_equal(r["output"], expected))
        first_capture_ms = first_instantiate_ms = None
        pipeline.clear_cache()
        for i in range(warmup_runs + measurement_runs):
            r = _graph_call(pipeline, batch, config, gc1d, lc, sm, use_graph=True, full_pipeline_scope=True)
            if i == 0:
                first_capture_ms = r["capture_ms"]
                first_instantiate_ms = r["instantiate_ms"]
            if i >= warmup_runs:
                variant_times["graph_full_pipeline"].append(r["total_ms"])
            variant_correct["graph_full_pipeline"] = bool(np.array_equal(r["output"], expected))
    finally:
        pipeline.release()

    prod_median = statistics.median(prod_times)
    result = {
        "batch_size": batch_size,
        "production": {"total_ms": summarize(prod_times)},
        "variants": {},
        "graph_setup_cost_ms": {"capture_ms": first_capture_ms, "instantiate_ms": first_instantiate_ms},
    }
    for name, times in variant_times.items():
        median = statistics.median(times)
        per_call_savings = statistics.median(variant_times["no_graph"]) - median if name != "no_graph" else 0.0
        break_even_calls = (
            (first_capture_ms + first_instantiate_ms) / per_call_savings
            if name == "graph_full_pipeline" and per_call_savings > 0 else None
        )
        result["variants"][name] = {
            "total_ms": summarize(times),
            "bit_exact_vs_production": variant_correct[name],
            "gain_vs_production": prod_median / median,
            "gain_vs_no_graph": statistics.median(variant_times["no_graph"]) / median,
            "break_even_calls": break_even_calls,
        }
    return result


def realistic_workload_benchmark(images_pool, config, gc1d, lc, sm, passes: int):
    """Reuses Section 20C's fixed measurement methodology exactly (context
    warmup, discarded full-sequence warmup passes for both variants, 5+
    measured passes with alternating execution order) -- the same one
    Section 20D reused for the Basic-kernel graph, so this result is
    directly comparable to both."""
    xray_cuda.smoke_test()

    def sequence(pipeline, use_graph):
        b32 = np.stack(images_pool[:32], axis=0)
        _graph_call(pipeline, b32, config, gc1d, lc, sm, use_graph, True)
        import dataclasses
        config2 = dataclasses.replace(config, threshold_value=200)
        _graph_call(pipeline, b32, config2, gc1d, lc, sm, use_graph, True)
        b8 = np.stack(images_pool[32:40], axis=0)
        _graph_call(pipeline, b8, config, gc1d, lc, sm, use_graph, True)
        b64 = np.stack(images_pool[40:104] if len(images_pool) >= 104 else images_pool[:64], axis=0)
        _graph_call(pipeline, b64, config, gc1d, lc, sm, use_graph, True)

    def stateless_sequence():
        b32 = images_pool[:32]
        run_enhanced_cuda_pipeline(b32, config)
        import dataclasses
        config2 = dataclasses.replace(config, threshold_value=200)
        run_enhanced_cuda_pipeline(b32, config2)
        b8 = images_pool[32:40]
        run_enhanced_cuda_pipeline(b8, config)
        b64 = images_pool[40:104] if len(images_pool) >= 104 else images_pool[:64]
        run_enhanced_cuda_pipeline(b64, config)

    def timed(fn):
        start = time.perf_counter()
        fn()
        return (time.perf_counter() - start) * 1000.0

    graph_pipeline = xray_cuda.CudaGraphEnhancedPipeline()
    try:
        for _ in range(2):
            stateless_sequence()
            sequence(graph_pipeline, True)

        stateless_times, graph_times = [], []
        for i in range(passes):
            if i % 2 == 0:
                stateless_times.append(timed(stateless_sequence))
                graph_times.append(timed(lambda: sequence(graph_pipeline, True)))
            else:
                graph_times.append(timed(lambda: sequence(graph_pipeline, True)))
                stateless_times.append(timed(stateless_sequence))
    finally:
        graph_pipeline.release()

    return {
        "stateless_total_ms": summarize(stateless_times),
        "graph_total_ms": summarize(graph_times),
        "gain_graph_vs_stateless": statistics.median(stateless_times) / statistics.median(graph_times),
        "passes": passes,
        "per_pass": {"stateless_ms": stateless_times, "graph_ms": graph_times},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-sweep", type=str, default=None)
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
    max_needed = max(batch_sizes) if batch_sizes else 104

    selection = dm.random_batch(batch_size=max(max_needed, 104), seed=args.seed)
    images = [load_image(p) for p in selection.selected_paths]
    shape = images[0].shape
    images_pool = [img for img in images if img.shape == shape]

    config = FilterConfig()
    gc1d = gaussian_kernel_1d(config.gaussian_kernel_size, config.gaussian_sigma)
    lc = laplacian_kernel_2d(config.laplacian_kernel_size)
    sm = mode_to_int(config.sobel_mode)

    print("=" * 70)
    print("SECTION 20E -- ENHANCED CUDA GRAPH PIPELINE OPTIMIZATION EXPERIMENT")
    print("=" * 70)
    print(f"Resolution: {shape[1]}x{shape[0]}   seed={args.seed}   pool={len(images_pool)} images")

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": get_environment_fingerprint().to_dict(),
        "dataset_fingerprint": dataset_fingerprint, "seed": args.seed, "resolution": [shape[0], shape[1]],
        "warmup_runs": args.warmup_runs, "measurement_runs": args.runs,
    }

    if batch_sizes:
        results_by_batch = []
        for batch_size in batch_sizes:
            if batch_size > len(images_pool):
                print(f"\n--- batch_size={batch_size} SKIPPED: only {len(images_pool)} images of the dominant "
                      f"resolution are available (never resampling to fill the gap) ---")
                continue
            print(f"\n--- batch_size={batch_size} ---")
            result = benchmark_one_batch_size(images_pool, config, gc1d, lc, sm, batch_size, args.warmup_runs, args.runs)
            results_by_batch.append(result)
            print(f"  production            total={result['production']['total_ms']['median']:8.4f}ms")
            for name, v in result["variants"].items():
                be = f"  break_even={v['break_even_calls']:.2f} calls" if v["break_even_calls"] else ""
                print(f"  {name:<22} total={v['total_ms']['median']:8.4f}ms  "
                      f"gain_vs_prod={v['gain_vs_production']:.3f}x  gain_vs_no_graph={v['gain_vs_no_graph']:.3f}x  "
                      f"bit_exact={v['bit_exact_vs_production']}{be}")
        manifest["batch_sweep"] = {"batch_sizes": batch_sizes, "results": results_by_batch}

    if args.realistic_workload:
        if len(images_pool) < 104:
            print(f"\nWarning: realistic workload needs 104 same-resolution images, only {len(images_pool)} available.")
        print(f"\n--- REALISTIC WORKLOAD (decisive test, {args.realistic_passes} passes, alternating order) ---")
        realistic = realistic_workload_benchmark(images_pool, config, gc1d, lc, sm, args.realistic_passes)
        print(f"  stateless (production) total: median={realistic['stateless_total_ms']['median']:.4f}ms")
        print(f"  graph (CudaGraphEnhancedPipeline) total: median={realistic['graph_total_ms']['median']:.4f}ms")
        print(f"  gain: {realistic['gain_graph_vs_stateless']:.3f}x")
        manifest["realistic_workload"] = realistic

    for sub in ("raw", "aggregated", "correctness", "graph_cache", "manifests"):
        (RESULTS_DIR / sub).mkdir(parents=True, exist_ok=True)
    experiment_id = "enhanced_cuda_graph_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    manifest["experiment_id"] = experiment_id
    out_path = RESULTS_DIR / "manifests" / f"{experiment_id}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    with open(RESULTS_DIR / "manifests" / "latest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"\nWrote results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
