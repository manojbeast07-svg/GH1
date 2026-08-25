"""Dataset discovery, validation, and reproducible selection.

This module deliberately separates three concerns that are easy to
conflate:

1. Discovery  - which files exist (filesystem metadata only).
2. Validation - are those files readable/decodable (opt-in, two speeds).
3. Selection  - which subset of paths to hand to a downstream pipeline.

Nothing here decodes pixel data during discovery, and random_batch()/
select_single() only ever produce file paths, never pixel arrays. This
keeps the same DatasetManager usable, unmodified, by the CPU, Basic CUDA,
and Enhanced CUDA implementations added in later sections -- they all
receive the same ImageSelection and are responsible for their own loading.
"""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2

from logging_config import get_logger

logger = get_logger(__name__)

# Formats supported initially. DICOM is intentionally not included here --
# it needs a dedicated dependency (pydicom) and different pixel-range
# handling, and the current dataset does not contain any .dcm files. Adding
# it later means adding one more suffix to this set plus a branch in
# image_loader.load_image(); nothing else in this module assumes raster
# formats.
DEFAULT_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})


@dataclass
class SelectedItem:
    """A single selected dataset entry, identified independently of list
    position so that benchmark result files stay traceable back to the
    original file even if the selection list is reordered or sliced.
    """

    index: int
    absolute_path: Path
    relative_path: str
    filename: str


@dataclass
class ImageSelection:
    """A selection batch: file paths only, no pixel data.

    This object is intended to be constructed once by DatasetManager and
    then passed unchanged to the CPU, Basic CUDA, and Enhanced CUDA
    implementations, so that all three operate on identical inputs.
    """

    mode: str  # "single" | "random_batch"
    seed: Optional[int]
    requested_batch_size: Optional[int]
    dataset_size: int
    items: List[SelectedItem]

    @property
    def selected_paths(self) -> List[Path]:
        return [item.absolute_path for item in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def iter_batches(self, batch_size: int):
        """Yield consecutive chunks of `batch_size` SelectedItems.

        This only slices the already-selected list of paths; it does not
        load any image data and does not introduce any CUDA/device
        concept. GPU chunk sizing is decided by later sections.
        """
        from pipeline.batch import iterate_batches  # local import avoids a cycle

        yield from iterate_batches(self.items, batch_size)


@dataclass
class FileValidationResult:
    path: Path
    valid: bool
    reason: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass
class DatasetStats:
    total_files: int = 0
    unsupported_files: int = 0
    valid_files: int = 0
    invalid_files: int = 0
    min_width: Optional[int] = None
    max_width: Optional[int] = None
    min_height: Optional[int] = None
    max_height: Optional[int] = None
    unique_resolutions: Dict[Tuple[int, int], int] = field(default_factory=dict)
    # Capped list of (relative_path, reason) for invalid files, so a report
    # doesn't have to print thousands of entries.
    invalid_samples: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def uniform_resolution(self) -> bool:
        """True only once dimensions are known (i.e. after full validation)
        and every valid image shares one resolution."""
        return len(self.unique_resolutions) == 1


class DatasetManager:
    """Scans a directory tree for supported X-ray images and produces
    deterministic, reproducible selections of file paths.
    """

    def __init__(self, root: "str | Path", extensions: Optional[Iterable[str]] = None):
        self.root = Path(root)
        self.extensions = frozenset(e.lower() for e in (extensions or DEFAULT_EXTENSIONS))
        self._paths: List[Path] = []
        self._unsupported_count = 0
        self._scanned = False

    # -- discovery ---------------------------------------------------

    def scan(self) -> None:
        """Recursively discover supported image files under self.root.

        Operates purely on filenames (no pixel decoding). Results are
        sorted by relative path so that discovery order is deterministic
        across runs and across machines.
        """
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"Dataset path is not a directory: {self.root}")

        logger.info("Dataset scan starting: %s", self.root)

        supported: List[Path] = []
        unsupported_count = 0
        for entry in self.root.rglob("*"):
            if not entry.is_file():
                continue
            if entry.suffix.lower() in self.extensions:
                supported.append(entry)
            else:
                unsupported_count += 1

        supported.sort(key=lambda p: str(p.relative_to(self.root)).lower())

        self._paths = supported
        self._unsupported_count = unsupported_count
        self._scanned = True

        logger.info(
            "Dataset scan complete: %d supported files, %d unsupported files",
            len(self._paths),
            self._unsupported_count,
        )

    def _ensure_scanned(self) -> None:
        if not self._scanned:
            raise RuntimeError("DatasetManager.scan() must be called before this operation.")

    @property
    def paths(self) -> List[Path]:
        self._ensure_scanned()
        return list(self._paths)

    @property
    def num_images(self) -> int:
        self._ensure_scanned()
        return len(self._paths)

    def relative_path(self, path: Path) -> str:
        return str(Path(path).relative_to(self.root))

    # -- validation ----------------------------------------------------

    def validate(self, mode: str = "fast", limit: Optional[int] = None) -> DatasetStats:
        """Validate discovered files.

        mode="fast": checks existence, readability, and extension only.
                     Does not decode pixel data. Safe to run on every
                     startup even with ~10,000 files.
        mode="full": additionally decodes each image (grayscale) to catch
                     corrupt/truncated files and to record true
                     width/height. Explicitly opt-in since decoding
                     thousands of images takes noticeably longer.

        `limit` restricts validation to the first N discovered files
        (after the deterministic sort), useful for a quick spot-check
        without paying for the full dataset.
        """
        if mode not in ("fast", "full"):
            raise ValueError(f"Unknown validation mode: {mode!r} (expected 'fast' or 'full')")

        self._ensure_scanned()
        paths = self._paths[:limit] if limit is not None else self._paths

        stats = DatasetStats(
            total_files=len(self._paths) + self._unsupported_count,
            unsupported_files=self._unsupported_count,
        )

        for path in paths:
            result = self._validate_one(path, mode)
            if result.valid:
                stats.valid_files += 1
                if result.width is not None and result.height is not None:
                    res = (result.width, result.height)
                    stats.unique_resolutions[res] = stats.unique_resolutions.get(res, 0) + 1
                    stats.min_width = result.width if stats.min_width is None else min(stats.min_width, result.width)
                    stats.max_width = result.width if stats.max_width is None else max(stats.max_width, result.width)
                    stats.min_height = result.height if stats.min_height is None else min(stats.min_height, result.height)
                    stats.max_height = result.height if stats.max_height is None else max(stats.max_height, result.height)
            else:
                stats.invalid_files += 1
                if len(stats.invalid_samples) < 20:
                    stats.invalid_samples.append((self.relative_path(path), result.reason or "unknown"))
                logger.warning("Invalid file (%s): %s", result.reason, path)

        logger.info(
            "Dataset validation (%s) complete: %d valid, %d invalid, %d unsupported",
            mode,
            stats.valid_files,
            stats.invalid_files,
            stats.unsupported_files,
        )
        return stats

    def _validate_one(self, path: Path, mode: str) -> FileValidationResult:
        if not path.exists():
            return FileValidationResult(path, False, reason="missing")
        if not os.access(path, os.R_OK):
            return FileValidationResult(path, False, reason="unreadable (permission denied)")
        if path.suffix.lower() not in self.extensions:
            return FileValidationResult(path, False, reason="unsupported extension")

        if mode == "fast":
            try:
                if path.stat().st_size == 0:
                    return FileValidationResult(path, False, reason="empty file")
            except OSError as exc:
                return FileValidationResult(path, False, reason=f"stat failed: {exc}")
            return FileValidationResult(path, True)

        # mode == "full": actually decode.
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return FileValidationResult(path, False, reason="failed to decode (corrupt or unsupported content)")
        height, width = image.shape[:2]
        return FileValidationResult(path, True, width=int(width), height=int(height))

    # -- selection -------------------------------------------------------

    def select_single(self, index: Optional[int] = None, path: Optional["str | Path"] = None) -> ImageSelection:
        """Select exactly one image by dataset index or by path.

        Does not decode the image. Exactly one of `index`/`path` must be
        given.
        """
        self._ensure_scanned()
        if (index is None) == (path is None):
            raise ValueError("select_single requires exactly one of `index` or `path`.")

        if index is not None:
            if not (0 <= index < len(self._paths)):
                raise IndexError(f"index {index} out of range for dataset of size {len(self._paths)}")
            resolved_index = index
        else:
            target = Path(path).resolve()
            resolved_index = None
            for i, p in enumerate(self._paths):
                if p.resolve() == target:
                    resolved_index = i
                    break
            if resolved_index is None:
                raise FileNotFoundError(f"Path not found in scanned dataset: {path}")

        selected_path = self._paths[resolved_index]
        item = SelectedItem(
            index=resolved_index,
            absolute_path=selected_path,
            relative_path=self.relative_path(selected_path),
            filename=selected_path.name,
        )
        logger.info("Single image selected: index=%d path=%s", resolved_index, item.relative_path)
        return ImageSelection(
            mode="single",
            seed=None,
            requested_batch_size=1,
            dataset_size=len(self._paths),
            items=[item],
        )

    def random_batch(self, batch_size: int, seed: int) -> ImageSelection:
        """Select `batch_size` images without replacement, deterministically.

        Given the same scanned dataset ordering, the same `seed`, and the
        same `batch_size`, this always returns the same set of files. The
        sampled indices are sorted ascending before being returned so the
        selection preserves the dataset's own ordering rather than the
        (irrelevant) order random.sample happened to draw them in -- this
        keeps chunked GPU processing order predictable and keeps repeated
        calls trivially comparable.
        """
        self._ensure_scanned()
        n = len(self._paths)

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_size > n:
            raise ValueError(f"batch_size {batch_size} exceeds dataset size {n}")

        rng = random.Random(seed)
        indices = sorted(rng.sample(range(n), batch_size))

        items = [
            SelectedItem(
                index=i,
                absolute_path=self._paths[i],
                relative_path=self.relative_path(self._paths[i]),
                filename=self._paths[i].name,
            )
            for i in indices
        ]
        logger.info(
            "Random batch selected: seed=%d batch_size=%d dataset_size=%d",
            seed,
            batch_size,
            n,
        )
        return ImageSelection(
            mode="random_batch",
            seed=seed,
            requested_batch_size=batch_size,
            dataset_size=n,
            items=items,
        )

    # -- fingerprint -------------------------------------------------------

    def fingerprint(self) -> str:
        """Lightweight dataset fingerprint based on sorted relative paths,
        file sizes, and modification times -- not file contents. Cheap
        enough to compute on every benchmark run to detect whether the
        on-disk dataset changed between runs.
        """
        self._ensure_scanned()
        hasher = hashlib.sha256()
        for path in self._paths:
            stat = path.stat()
            hasher.update(self.relative_path(path).encode("utf-8"))
            hasher.update(str(stat.st_size).encode("utf-8"))
            hasher.update(str(int(stat.st_mtime)).encode("utf-8"))
        return hasher.hexdigest()
