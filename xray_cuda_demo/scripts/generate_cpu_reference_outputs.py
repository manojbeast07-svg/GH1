"""Generate frozen CPU reference outputs under tests/reference/.

Run manually, deliberately -- NOT part of the normal test suite. Future
CUDA implementations (and future CPU refactors) get compared against
these files, so they must only change when the CPU filter behavior is
intentionally changed, never silently as a side effect of running tests.

Usage:
    python scripts/generate_cpu_reference_outputs.py [--force]

Each reference file is the result of applying ONE filter directly to the
raw fixture image with default FilterConfig() parameters (not the
chained 5-stage pipeline) -- this is what tests/test_cpu_filters.py
compares against, since it isolates each filter's own correctness from
the others'.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cpu.filters import FilterConfig, apply_gaussian, apply_laplacian, apply_median, apply_sobel, apply_threshold  # noqa: E402
from tests.fixtures import FIXTURES  # noqa: E402

REFERENCE_DIR = PROJECT_ROOT / "tests" / "reference"

FILTERS = {
    "gaussian": apply_gaussian,
    "median": apply_median,
    "sobel": apply_sobel,
    "laplacian": apply_laplacian,
    "threshold": apply_threshold,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite existing reference files.")
    args = parser.parse_args()

    config = FilterConfig()
    written, skipped = 0, 0

    for fixture_name, generator in FIXTURES.items():
        image = generator()
        fixture_dir = REFERENCE_DIR / fixture_name
        fixture_dir.mkdir(parents=True, exist_ok=True)

        for filter_name, apply_fn in FILTERS.items():
            out_path = fixture_dir / f"{filter_name}.npy"
            if out_path.exists() and not args.force:
                print(f"skip (exists): {out_path.relative_to(PROJECT_ROOT)}")
                skipped += 1
                continue
            output = apply_fn(image, config)
            np.save(out_path, output)
            print(f"wrote: {out_path.relative_to(PROJECT_ROOT)}")
            written += 1

    print(f"\n{written} written, {skipped} skipped (use --force to overwrite).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
