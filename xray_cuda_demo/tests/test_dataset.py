"""Section 2 tests: dataset discovery, validation, selection, batching.

Uses small synthetic datasets under pytest's tmp_path rather than the real
~9,463-image X-ray dataset, so these tests stay fast and self-contained.
"""

import numpy as np
import cv2
import pytest

from pipeline.dataset import DatasetManager
from pipeline.batch import iterate_batches, load_batch, LoadedBatch, estimate_batch_memory
from pipeline.image_loader import load_image, ImageLoadError


def _write_image(path, size=8, value=128):
    cv2.imwrite(str(path), np.full((size, size), value, dtype=np.uint8))


@pytest.fixture
def small_dataset(tmp_path):
    """2 valid images, 1 corrupt .png, 1 unsupported extension."""
    _write_image(tmp_path / "a.png", value=10)
    _write_image(tmp_path / "b.png", value=200)
    (tmp_path / "corrupt.png").write_bytes(b"this is not a real png file")
    (tmp_path / "notes.txt").write_text("not an image")
    return tmp_path


@pytest.fixture
def dataset_20(tmp_path):
    """20 valid same-size images for selection/batching tests."""
    for i in range(20):
        _write_image(tmp_path / f"img_{i:02d}.png", value=i)
    return tmp_path


# -- discovery -------------------------------------------------------


def test_scan_finds_only_supported_extensions(small_dataset):
    dm = DatasetManager(small_dataset)
    dm.scan()
    names = {p.name for p in dm.paths}
    assert names == {"a.png", "b.png", "corrupt.png"}
    assert dm.num_images == 3


def test_scan_deterministic_ordering(small_dataset):
    dm1 = DatasetManager(small_dataset)
    dm1.scan()
    dm2 = DatasetManager(small_dataset)
    dm2.scan()
    assert dm1.paths == dm2.paths
    assert dm1.paths == sorted(dm1.paths, key=lambda p: str(p).lower())


def test_scan_missing_root_raises(tmp_path):
    dm = DatasetManager(tmp_path / "does_not_exist")
    with pytest.raises(FileNotFoundError):
        dm.scan()


def test_properties_require_scan_first(small_dataset):
    dm = DatasetManager(small_dataset)
    with pytest.raises(RuntimeError):
        _ = dm.paths


# -- validation (corrupt-file handling) -------------------------------------------------------


def test_fast_validation_does_not_decode(small_dataset):
    dm = DatasetManager(small_dataset)
    dm.scan()
    stats = dm.validate(mode="fast")
    # Fast mode only checks existence/readability/extension, not pixel
    # content, so the corrupt-but-nonempty .png still counts as "valid".
    assert stats.valid_files == 3
    assert stats.invalid_files == 0
    assert stats.unsupported_files == 1
    assert stats.unique_resolutions == {}


def test_full_validation_detects_corrupt_file_without_crashing(small_dataset):
    dm = DatasetManager(small_dataset)
    dm.scan()
    stats = dm.validate(mode="full")
    assert stats.valid_files == 2
    assert stats.invalid_files == 1
    assert stats.unsupported_files == 1
    assert any("corrupt.png" in rel for rel, _reason in stats.invalid_samples)
    assert stats.unique_resolutions == {(8, 8): 2}


def test_valid_images_remain_usable_despite_corrupt_file(small_dataset):
    dm = DatasetManager(small_dataset)
    dm.scan()
    for path in dm.paths:
        if path.name == "corrupt.png":
            continue
        image = load_image(path)
        assert image.dtype == np.uint8
        assert image.shape == (8, 8)


def test_invalid_validate_mode_rejected(small_dataset):
    dm = DatasetManager(small_dataset)
    dm.scan()
    with pytest.raises(ValueError):
        dm.validate(mode="bogus")


# -- image loader -------------------------------------------------------


def test_load_image_valid(small_dataset):
    image = load_image(small_dataset / "a.png")
    assert image.dtype == np.uint8
    assert image.shape == (8, 8)


def test_load_image_corrupt_raises(small_dataset):
    with pytest.raises(ImageLoadError):
        load_image(small_dataset / "corrupt.png")


def test_load_image_missing_raises(small_dataset):
    with pytest.raises(ImageLoadError):
        load_image(small_dataset / "missing.png")


# -- single selection -------------------------------------------------------


def test_select_single_by_index(dataset_20):
    dm = DatasetManager(dataset_20)
    dm.scan()
    selection = dm.select_single(index=3)
    assert len(selection) == 1
    assert selection.items[0].index == 3
    assert selection.mode == "single"


def test_select_single_by_path(dataset_20):
    dm = DatasetManager(dataset_20)
    dm.scan()
    target = dm.paths[5]
    selection = dm.select_single(path=target)
    assert selection.items[0].absolute_path == target


def test_select_single_requires_exactly_one_arg(dataset_20):
    dm = DatasetManager(dataset_20)
    dm.scan()
    with pytest.raises(ValueError):
        dm.select_single()
    with pytest.raises(ValueError):
        dm.select_single(index=0, path=dm.paths[0])


def test_select_single_index_out_of_range(dataset_20):
    dm = DatasetManager(dataset_20)
    dm.scan()
    with pytest.raises(IndexError):
        dm.select_single(index=999)


# -- random batch selection / reproducibility -------------------------------------------------------


def test_seed_reproducibility(dataset_20):
    """Test 1: same seed + batch size -> identical selection."""
    dm = DatasetManager(dataset_20)
    dm.scan()
    a = dm.random_batch(batch_size=10, seed=42)
    b = dm.random_batch(batch_size=10, seed=42)
    assert a.selected_paths == b.selected_paths
    assert a.seed == b.seed == 42


def test_different_seed_changes_selection(dataset_20):
    """Test 2: different seed -> different selection."""
    dm = DatasetManager(dataset_20)
    dm.scan()
    a = dm.random_batch(batch_size=10, seed=42)
    b = dm.random_batch(batch_size=10, seed=43)
    assert a.selected_paths != b.selected_paths


def test_batch_size_exceeds_dataset_size_rejected(dataset_20):
    """Test 3: batch size > dataset size -> clear error."""
    dm = DatasetManager(dataset_20)
    dm.scan()
    with pytest.raises(ValueError):
        dm.random_batch(batch_size=dm.num_images + 1, seed=1)


def test_batch_size_one(dataset_20):
    """Test 4: batch size 1 -> exactly one path."""
    dm = DatasetManager(dataset_20)
    dm.scan()
    selection = dm.random_batch(batch_size=1, seed=7)
    assert len(selection) == 1


def test_batch_size_equals_dataset_size(dataset_20):
    """Test 5: batch size == dataset size -> all valid paths selected."""
    dm = DatasetManager(dataset_20)
    dm.scan()
    n = dm.num_images
    selection = dm.random_batch(batch_size=n, seed=7)
    assert len(selection) == n
    assert sorted(selection.selected_paths) == sorted(dm.paths)


def test_random_batch_has_no_duplicates(dataset_20):
    """Test 6: no duplicates in a random batch."""
    dm = DatasetManager(dataset_20)
    dm.scan()
    selection = dm.random_batch(batch_size=15, seed=99)
    paths = selection.selected_paths
    assert len(paths) == len(set(paths))


def test_random_batch_zero_or_negative_rejected(dataset_20):
    dm = DatasetManager(dataset_20)
    dm.scan()
    with pytest.raises(ValueError):
        dm.random_batch(batch_size=0, seed=1)
    with pytest.raises(ValueError):
        dm.random_batch(batch_size=-5, seed=1)


# -- batch iteration -------------------------------------------------------


def test_iterate_batches_partial_last_chunk():
    """100 items, batch_size 32 -> chunks of 32,32,32,4; no dup/missing."""
    items = list(range(100))
    chunks = list(iterate_batches(items, 32))
    assert [len(c) for c in chunks] == [32, 32, 32, 4]
    flattened = [x for chunk in chunks for x in chunk]
    assert flattened == items  # ordering preserved, nothing missing or duplicated


def test_iterate_batches_exact_multiple():
    items = list(range(64))
    chunks = list(iterate_batches(items, 32))
    assert [len(c) for c in chunks] == [32, 32]


def test_iterate_batches_rejects_non_positive_size():
    with pytest.raises(ValueError):
        list(iterate_batches([1, 2, 3], 0))


def test_selection_iter_batches(dataset_20):
    dm = DatasetManager(dataset_20)
    dm.scan()
    selection = dm.random_batch(batch_size=20, seed=1)
    chunks = list(selection.iter_batches(6))
    assert [len(c) for c in chunks] == [6, 6, 6, 2]


# -- loading / collation -------------------------------------------------------


def test_load_batch_and_collate_same_shape(dataset_20):
    dm = DatasetManager(dataset_20)
    dm.scan()
    selection = dm.random_batch(batch_size=5, seed=3)
    loaded = load_batch(selection.items)
    assert len(loaded) == 5
    stacked = loaded.collate()
    assert stacked.shape == (5, 8, 8)
    assert stacked.dtype == np.uint8


def test_collate_raises_on_mismatched_shapes():
    small = np.zeros((8, 8), dtype=np.uint8)
    big = np.zeros((16, 16), dtype=np.uint8)
    loaded = LoadedBatch(items=[], images=[small, big])
    with pytest.raises(ValueError):
        loaded.collate()


# -- memory estimation -------------------------------------------------------


def test_estimate_batch_memory():
    estimate = estimate_batch_memory(count=100, height=512, width=512)
    assert estimate.host_bytes == 100 * 512 * 512
    assert estimate.gpu_estimate_bytes == 512 * 512 * 2  # default ping-pong pair


# -- fingerprint -------------------------------------------------------


def test_fingerprint_stable_across_rescans(small_dataset):
    dm1 = DatasetManager(small_dataset)
    dm1.scan()
    dm2 = DatasetManager(small_dataset)
    dm2.scan()
    assert dm1.fingerprint() == dm2.fingerprint()


def test_fingerprint_changes_when_dataset_changes(small_dataset):
    dm1 = DatasetManager(small_dataset)
    dm1.scan()
    fp1 = dm1.fingerprint()

    _write_image(small_dataset / "new_file.png", value=42)

    dm2 = DatasetManager(small_dataset)
    dm2.scan()
    fp2 = dm2.fingerprint()

    assert fp1 != fp2
