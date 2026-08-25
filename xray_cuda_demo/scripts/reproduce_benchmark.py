"""Reproduce a previously-run canonical benchmark (Section 11 spec item 31).

Usage:
    python scripts/reproduce_benchmark.py --benchmark-id 20260824_111500_seed42_batch125_224x224

Loads the stored manifest for `--benchmark-id`, checks whether the
dataset on disk still matches the fingerprint recorded at benchmark
time (reporting a mismatch rather than silently proceeding -- spec item
31's explicit requirement), and if it matches, re-runs the exact same
canonical benchmark (same selection, same seed, same filter config) as
a NEW benchmark run (its own new benchmark_id -- earlier results are
never overwritten, per spec item 30) so the two can be compared.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import xray_cuda  # noqa: E402

import cuda.final_benchmark as fb  # noqa: E402
from cpu.filters import FilterConfig  # noqa: E402
from pipeline.dataset import DatasetManager  # noqa: E402
from pipeline.image_loader import load_image  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-id", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="benchmark_results")
    parser.add_argument("--force", action="store_true",
                         help="Re-run even if the dataset fingerprint no longer matches "
                              "(the new run's own manifest will still record whatever the "
                              "CURRENT dataset state is -- it does not pretend to reproduce "
                              "the original exactly).")
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    if not results_root.is_absolute():
        results_root = (PROJECT_ROOT / results_root).resolve()

    print("=" * 70)
    print(f"REPRODUCING BENCHMARK: {args.benchmark_id}")
    print("=" * 70)

    manifest = fb.load_manifest(args.benchmark_id, results_root)
    if manifest is None:
        print(f"\nNo manifest found for benchmark_id={args.benchmark_id!r} under {results_root}/manifests/")
        return 1

    print(f"\nOriginal run: {manifest['timestamp_utc']}")
    print(f"Dataset path: {manifest['dataset_path']}")
    print(f"Dataset fingerprint (recorded): {manifest['dataset_fingerprint']}")
    print(f"Seed: {manifest['seed']}   Selected images: {manifest['selected_image_count']}   "
          f"Resolution: {manifest['resolution'][1]}x{manifest['resolution'][0]}")

    ok, notes = fb.check_reproducibility(args.benchmark_id, results_root)
    print(f"\nReproducibility check: {'PASS' if ok else 'MISMATCH'}")
    for note in notes:
        print(f"  {note}")

    if not ok and not args.force:
        print("\nRefusing to proceed (dataset state has changed since the original run). "
              "Pass --force to re-run anyway against the CURRENT dataset state.")
        return 1

    if not xray_cuda.cuda_available():
        print("\nNo usable CUDA device detected. Cannot proceed.")
        return 1

    dataset_path = Path(manifest["dataset_path"])
    dm = DatasetManager(dataset_path)
    dm.scan()
    fingerprint = dm.fingerprint()
    valid_image_count = len(dm.paths)

    selection = dm.random_batch(batch_size=manifest["requested_batch_size"] or manifest["selected_image_count"],
                                 seed=manifest["seed"])
    images_all = [load_image(p) for p in selection.selected_paths]
    height, width = manifest["resolution"]
    images_main = [img for img in images_all if img.shape == (height, width)]

    filter_config = FilterConfig(**manifest["filter_config"])

    print(f"\nRe-running canonical benchmark (n={len(images_main)}, "
          f"warmup={manifest['warmup_runs']}, measurement={manifest['measurement_runs']})...")
    result = fb.run_canonical_benchmark(
        images_main, selection, filter_config,
        dataset_fingerprint=fingerprint, dataset_path=str(dataset_path), valid_image_count=valid_image_count,
        warmup_runs=manifest["warmup_runs"], measurement_runs=manifest["measurement_runs"],
        results_root=results_root,
    )

    print(f"\nNew benchmark_id: {result.manifest.benchmark_id}")
    print(f"Status: {result.manifest.status}")
    print(f"\n-- Comparison --")
    orig_aggregated = fb.load_aggregated(args.benchmark_id, results_root)
    if orig_aggregated:
        orig_speedup = orig_aggregated["enhanced_speedup_vs_basic_compute_only"]
        new_speedup = result.enhanced_speedup_vs_basic_compute_only
        print(f"Original enhanced-vs-basic (compute-only): {orig_speedup:.3f}x")
        print(f"New      enhanced-vs-basic (compute-only): {new_speedup:.3f}x")
        pct_diff = 100.0 * abs(new_speedup - orig_speedup) / orig_speedup if orig_speedup else 0.0
        print(f"Relative difference: {pct_diff:.1f}%")
    else:
        print("(original aggregated result file not found -- cannot compare numerically)")

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
