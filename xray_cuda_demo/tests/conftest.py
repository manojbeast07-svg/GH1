"""Shared pytest fixtures."""

import hashlib
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def stable_seed(*parts) -> int:
    """Deterministic, process-independent replacement for
    `hash((...)) % (2**32)` (Section 10 spec item 37): Python's built-in
    `hash()` salts str/bytes/tuples-of-those with a per-process random
    seed (PYTHONHASHSEED) unless explicitly disabled, so a test that
    seeds `np.random.default_rng(hash((a, b)) % (2**32))` picks a
    DIFFERENT "random" image on every interpreter run -- this was
    confirmed as the root cause of an intermittent failure in
    test_gaussian_enhanced.py's border-region tests (occasionally
    landing on a pixel that crosses the documented +-1 Gaussian
    tolerance purely because a different image was generated that run,
    not a real bug). hashlib is not seed-randomized, so hashing the
    parts' string representation gives the same seed on every run,
    every machine, every process -- the fix is to always derive
    per-test RNG seeds through this function, never through hash().
    """
    digest = hashlib.md5(repr(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def get_real_dataset_path():
    """Path to the real X-ray dataset (config.yaml dataset.path), or None
    if it isn't present on this machine. Plain function (not a fixture)
    so it can also be called directly inside @pytest.mark.skipif
    decorators, which run at collection time before fixtures exist."""
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    raw_path = (config.get("dataset") or {}).get("path")
    if not raw_path:
        return None
    dataset_path = Path(raw_path)
    if not dataset_path.is_absolute():
        dataset_path = (PROJECT_ROOT / dataset_path).resolve()
    return dataset_path if dataset_path.exists() else None


@pytest.fixture(scope="session")
def real_dataset_path():
    """Fixture form of get_real_dataset_path(), for tests that want to
    skip from within the test body rather than via a decorator."""
    return get_real_dataset_path()
