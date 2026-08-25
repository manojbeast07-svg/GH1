"""Final canonical benchmark (Section 11).

Usage:
    python scripts/run_final_benchmark.py --dataset ../data --seed 42 --images 125 --runs 20

Runs the complete Section 11 benchmark suite -- one deterministic
dataset selection, reused unchanged across CPU/Basic CUDA/Enhanced CUDA
(spec item 6) -- and writes everything under benchmark_results/:
the canonical CPU-vs-Basic-vs-Enhanced pipeline benchmark (raw +
aggregated + manifest), a batch-size sweep, a resolution sweep, a
per-filter benchmark with Amdahl-style contribution analysis, a 3-level
correctness benchmark, the combined machine-readable summary, and an
updated experiment registry. This is the dataset Section 12's Streamlit
dashboard reads directly -- it never needs to rerun this script itself.
"""

import argparse
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import xray_cuda  # noqa: E402

import cuda.final_benchmark as fb  # noqa: E402
from cpu.filters import FilterConfig  # noqa: E402
from cuda.pipeline import group_by_resolution  # noqa: E402
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


def _print_agg(label: str, agg_dict) -> None:
    if agg_dict is None:
        print(f"{label}: (n/a)")
        return
    print(f"{label}: mean={agg_dict['mean']:.4f}ms median={agg_dict['median']:.4f}ms "
          f"min={agg_dict['min']:.4f}ms max={agg_dict['max']:.4f}ms std={agg_dict['std']:.4f}ms")


def main() -> int:
    config_yaml = _load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--images", type=int, default=125, help="Requested selection batch size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=20, help="Canonical benchmark measurement runs.")
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--skip-batch-sweep", action="store_true")
    parser.add_argument("--skip-resolution-sweep", action="store_true")
    parser.add_argument("--skip-per-filter", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--results-dir", type=str, default="benchmark_results")
    args = parser.parse_args()

    if not xray_cuda.cuda_available():
        print("No usable CUDA device detected. Cannot proceed.")
        return 1

    results_root = Path(args.results_dir)
    if not results_root.is_absolute():
        results_root = (PROJECT_ROOT / results_root).resolve()
    fb.ensure_results_dirs(results_root)

    dataset_path = _resolve_dataset_path(args.dataset, config_yaml)
    dm = DatasetManager(dataset_path)
    dm.scan()
    fingerprint = dm.fingerprint()
    valid_image_count = len(dm.paths)

    print("=" * 70)
    print("SECTION 11 -- FINAL CANONICAL BENCHMARK")
    print("=" * 70)
    print(f"\nDataset: {dataset_path}")
    print(f"Dataset fingerprint: {fingerprint}")
    print(f"Scanned (supported-extension) file count: {valid_image_count}")

    selection = dm.random_batch(batch_size=args.images, seed=args.seed)
    load_start = time.perf_counter()
    images_all = [load_image(p) for p in selection.selected_paths]
    load_ms_all = (time.perf_counter() - load_start) * 1000.0

    shape = images_all[0].shape
    images_main = [img for img in images_all if img.shape == shape]
    load_ms_main = load_ms_all * (len(images_main) / len(images_all))
    print(f"Selection: {len(selection)} images (seed={args.seed}); "
          f"{len(images_main)} share the dominant resolution {shape[1]}x{shape[0]}")

    filter_config = FilterConfig()

    # -- canonical benchmark --
    print(f"\n{'-'*70}\nCanonical benchmark ({shape[1]}x{shape[0]}, n={len(images_main)})\n{'-'*70}")
    canonical = fb.run_canonical_benchmark(
        images_main, selection, filter_config,
        dataset_fingerprint=fingerprint, dataset_path=str(dataset_path), valid_image_count=valid_image_count,
        warmup_runs=args.warmup_runs, measurement_runs=args.runs, load_ms=load_ms_main, results_root=results_root,
    )
    print(f"benchmark_id: {canonical.manifest.benchmark_id}   status: {canonical.manifest.status}")
    _print_agg("CPU   Mode4 (end-to-end)", canonical.cpu.mode4_end_to_end_ms.to_dict())
    _print_agg("Basic Mode4 (end-to-end)", canonical.basic.mode4_end_to_end_ms.to_dict())
    _print_agg("Enhanced Mode4 (end-to-end)", canonical.enhanced.mode4_end_to_end_ms.to_dict())
    _print_agg("Basic Mode1 (kernel-only)", canonical.basic.mode1_kernel_only_ms.to_dict())
    _print_agg("Enhanced Mode1 (kernel-only)", canonical.enhanced.mode1_kernel_only_ms.to_dict())
    print(f"\nBasic speedup vs CPU:                {canonical.basic_speedup_vs_cpu:.3f}x")
    print(f"Enhanced speedup vs CPU:              {canonical.enhanced_speedup_vs_cpu:.3f}x")
    print(f"Enhanced vs Basic (compute-only):     {canonical.enhanced_speedup_vs_basic_compute_only:.3f}x")
    print(f"Enhanced vs Basic (end-to-end):       {canonical.enhanced_speedup_vs_basic_end_to_end:.3f}x")
    print(f"Correctness enhanced vs basic: {canonical.correctness_enhanced_vs_basic}")
    print(f"Correctness enhanced vs cpu:   {canonical.correctness_enhanced_vs_cpu}")

    batch_sweep = None
    if not args.skip_batch_sweep:
        print(f"\n{'-'*70}\nBatch-size sweep\n{'-'*70}")
        batch_sizes = [1, 8, 16, 32, 64, 128, 256, 512]
        # The canonical selection (--images, default 125) is deliberately
        # small (fast to run); reaching the larger batch sizes needs its
        # own, separately-sized selection (same seed, same dominant
        # resolution) rather than silently repeating the same capped
        # effective batch size for every requested size above what the
        # canonical selection contains.
        if max(batch_sizes) > len(images_main):
            sweep_selection = dm.random_batch(batch_size=max(batch_sizes) * 2, seed=args.seed)
            sweep_images_all = [load_image(p) for p in sweep_selection.selected_paths]
            images_for_sweep = [img for img in sweep_images_all if img.shape == shape]
            print(f"(batch sweep uses a separately-sized selection: {len(images_for_sweep)} images "
                  f"at {shape[1]}x{shape[0]}, same seed={args.seed})")
        else:
            images_for_sweep = images_main
        batch_sweep = fb.run_batch_sweep(
            images_for_sweep, filter_config, batch_sizes=batch_sizes,
            warmup_runs=3, measurement_runs=10, seed=args.seed, results_root=results_root,
        )
        for row in batch_sweep["rows"]:
            print(f"  batch={row['effective_batch_size']:<4d} "
                  f"basic={row['basic_total_ms']['mean']:.3f}ms enhanced={row['enhanced_total_ms']['mean']:.3f}ms "
                  f"enhanced_vs_basic={row['enhanced_vs_basic_gain']:.3f}x")
        fb.update_experiment_registry(
            "Batch sweep", batch_sweep["benchmark_id"], [batch_sweep["_file_path"]],
            {"batch_sizes": [1, 8, 16, 32, 64, 128, 256, 512], "seed": args.seed}, results_root=results_root,
        )

    resolution_sweep = None
    if not args.skip_resolution_sweep:
        print(f"\n{'-'*70}\nResolution sweep\n{'-'*70}")
        groups_items = group_by_resolution(selection)
        groups = {res: [load_image(it.absolute_path) for it in items] for res, items in groups_items.items()}
        resolution_sweep = fb.run_resolution_sweep(
            groups, filter_config, warmup_runs=2, measurement_runs=10, seed=args.seed, results_root=results_root,
        )
        for row in resolution_sweep["rows"]:
            print(f"  {row['width']}x{row['height']:<5d} n={row['image_count']:<4d} "
                  f"basic_ms/img={row['basic_ms_per_image']:.4f} enhanced_ms/img={row['enhanced_ms_per_image']:.4f}")
        fb.update_experiment_registry(
            "Resolution sweep", resolution_sweep["benchmark_id"], [resolution_sweep["_file_path"]],
            {"seed": args.seed}, results_root=results_root,
        )

    per_filter = None
    if not args.skip_per_filter:
        print(f"\n{'-'*70}\nPer-filter benchmark + Amdahl-style contribution analysis\n{'-'*70}")
        per_filter = fb.run_per_filter_benchmark(
            images_main, filter_config, warmup_runs=5, measurement_runs=20, seed=args.seed, results_root=results_root,
        )
        for row in per_filter["rows"]:
            print(f"  {row['filter']:<10s} basic={row['basic_kernel_ms']['mean']:.4f}ms "
                  f"enhanced={row['enhanced_kernel_ms']['mean']:.4f}ms speedup={row['kernel_speedup']:.3f}x "
                  f"contribution={row['pct_of_total_compute_reduction']:.1f}%")
        fb.update_experiment_registry(
            "Final pipeline", canonical.manifest.benchmark_id,
            list(canonical.manifest.result_file_paths.values()), {"seed": args.seed, "images": len(images_main)},
            results_root=results_root,
        )

    correctness = None
    if not args.skip_correctness:
        print(f"\n{'-'*70}\nCorrectness benchmark (3 levels)\n{'-'*70}")
        correctness = fb.run_correctness_benchmark(
            images_main, filter_config, seed=args.seed, results_root=results_root,
        )
        print(f"  overall_pass: {correctness['overall_pass']}")
        print(f"  baseline note: {correctness['baseline_comparison_note']}")
        print(f"  final pipeline enhanced-vs-cpu differing%: "
              f"{correctness['pipeline_level']['enhanced_vs_cpu']['differing_pixel_percentage']:.4f}%")

    summary = fb.build_final_summary(
        canonical, batch_sweep=batch_sweep, resolution_sweep=resolution_sweep,
        per_filter=per_filter, correctness=correctness, results_root=results_root,
    )
    print(f"\n{'-'*70}")
    print(f"Final summary written: benchmark_results/summary/{canonical.manifest.benchmark_id}.json")
    print(f"Also available as:     benchmark_results/summary/latest.json")

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
