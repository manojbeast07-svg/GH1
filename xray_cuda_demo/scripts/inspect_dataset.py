"""One-command dataset diagnostic.

Usage:
    python scripts/inspect_dataset.py [--path PATH] [--mode fast|full]
                                       [--seed 42] [--batch-size 10]
                                       [--limit N]

If --path is omitted, the dataset path is read from config.yaml
(dataset.path), resolved relative to the project root.
"""

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.dataset import DatasetManager  # noqa: E402


def _load_default_path() -> Path:
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    raw_path = (config.get("dataset") or {}).get("path")
    if not raw_path:
        raise SystemExit(
            "No --path given and config.yaml has no dataset.path set. "
            "Pass --path explicitly."
        )
    dataset_path = Path(raw_path)
    if not dataset_path.is_absolute():
        dataset_path = (PROJECT_ROOT / dataset_path).resolve()
    return dataset_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the X-ray dataset.")
    parser.add_argument("--path", type=str, default=None, help="Dataset root directory.")
    parser.add_argument("--mode", choices=["fast", "full"], default="fast",
                         help="Validation depth (default: fast).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the selection test.")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for the selection test.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only validate the first N discovered files (useful with --mode full).")
    parser.add_argument("--sample-count", type=int, default=5,
                         help="How many selected filenames to print (default: 5).")
    args = parser.parse_args()

    dataset_path = Path(args.path).resolve() if args.path else _load_default_path()

    print("=" * 60)
    print("X-RAY DATASET")
    print("=" * 60)
    print(f"\nDataset:\n{dataset_path}")

    manager = DatasetManager(dataset_path)
    manager.scan()

    print(f"\nFiles:\n{manager.num_images}")

    stats = manager.validate(mode=args.mode, limit=args.limit)
    print(f"\nValid:\n{stats.valid_files}")
    print(f"\nInvalid:\n{stats.invalid_files}")
    print(f"\nUnsupported extension (excluded from Files count above):\n{stats.unsupported_files}")

    if stats.invalid_samples:
        print("\nInvalid file samples:")
        for rel_path, reason in stats.invalid_samples[:10]:
            print(f"  {rel_path}: {reason}")

    if stats.unique_resolutions:
        print("\nResolutions:")
        for (width, height), count in sorted(stats.unique_resolutions.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {width}x{height}: {count}")
        if len(stats.unique_resolutions) > 10:
            print(f"  ... and {len(stats.unique_resolutions) - 10} more distinct resolutions")
    elif args.mode == "fast":
        print("\nResolutions:\n  (not measured -- run with --mode full to decode and measure dimensions)")

    print(f"\nSelection test:\nSeed {args.seed}")
    print(f"Batch size {args.batch_size}")

    try:
        selection = manager.random_batch(batch_size=args.batch_size, seed=args.seed)
        sample = selection.selected_paths[: args.sample_count]
        print("\nSelected (sample):")
        for path in sample:
            print(f"  {manager.relative_path(path)}")
        remaining = len(selection) - len(sample)
        if remaining > 0:
            print(f"  ... and {remaining} more")
    except ValueError as exc:
        print(f"\nSelection test FAILED: {exc}")

    print(f"\nDataset fingerprint:\n{manager.fingerprint()[:16]}...")

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
