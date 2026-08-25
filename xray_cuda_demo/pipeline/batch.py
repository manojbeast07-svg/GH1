"""Batch iteration, lazy loading, collation, and memory estimation.

Keeps two distinct concepts explicitly separate:

    Selection batch: a list of SelectedItem (file paths + identifiers).
                      See pipeline.dataset.ImageSelection.
    Loaded batch:     actual decoded NumPy arrays for a chunk of that
                      selection. See LoadedBatch below.

Nothing in this module ever loads an entire ~10,000-image dataset into
memory: iterate_batches() only slices an already-selected list of paths,
and load_batch() only decodes the items handed to it (typically one GPU
chunk at a time).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, List, Sequence, TypeVar

import numpy as np

from pipeline.dataset import SelectedItem
from pipeline.image_loader import load_image
from logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def iterate_batches(items: Sequence[T], batch_size: int) -> Iterator[List[T]]:
    """Yield consecutive chunks of `items`, each of length <= batch_size.

    The final chunk may be smaller than batch_size. No item is skipped,
    duplicated, or reordered.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    n = len(items)
    for start in range(0, n, batch_size):
        yield list(items[start : start + batch_size])


@dataclass
class LoadedBatch:
    """Decoded images for a chunk of SelectedItems, in matching order."""

    items: List[SelectedItem]
    images: List[np.ndarray]

    def __len__(self) -> int:
        return len(self.items)

    def collate(self) -> np.ndarray:
        """Stack images into a single [N, H, W] uint8 array.

        Raises ValueError with the offending resolutions listed if the
        images do not share a common shape -- this deliberately does not
        resize or pad, since Section 2 has no approved resize policy yet.
        """
        if not self.images:
            raise ValueError("Cannot collate an empty batch.")

        shapes = {img.shape for img in self.images}
        if len(shapes) > 1:
            resolutions = sorted(f"{w}x{h}" for (h, w) in shapes)
            raise ValueError(
                "Cannot construct fixed-shape batch: batch contains images "
                f"with resolutions: {', '.join(resolutions)}"
            )
        return np.stack(self.images, axis=0)


def load_batch(items: Sequence[SelectedItem], loader: Callable[[object], np.ndarray] = load_image) -> LoadedBatch:
    """Decode exactly the given items (typically one GPU-sized chunk)."""
    images = [loader(item.absolute_path) for item in items]
    return LoadedBatch(items=list(items), images=images)


@dataclass
class MemoryEstimate:
    count: int
    height: int
    width: int
    bytes_per_pixel: int
    host_bytes: int
    intermediate_buffers: int
    gpu_estimate_bytes: int

    @property
    def host_mb(self) -> float:
        return self.host_bytes / (1024 * 1024)

    @property
    def gpu_estimate_mb(self) -> float:
        return self.gpu_estimate_bytes / (1024 * 1024)


def estimate_batch_memory(
    count: int,
    height: int,
    width: int,
    bytes_per_pixel: int = 1,
    intermediate_buffers: int = 2,
) -> MemoryEstimate:
    """Estimate host/GPU memory for a batch of `count` images of a known
    (height, width), without allocating anything.

    `intermediate_buffers` defaults to 2 (a ping-pong pair), matching the
    "GPU stays GPU" design in the project spec: the five-stage pipeline is
    expected to alternate between two device buffers rather than keep one
    buffer per stage, since each stage only needs the previous stage's
    output. This default is a documented planning assumption for later
    sections, not a value used to allocate memory here.
    """
    if count < 0 or height <= 0 or width <= 0 or bytes_per_pixel <= 0:
        raise ValueError("count, height, width, and bytes_per_pixel must be positive.")

    bytes_per_image = height * width * bytes_per_pixel
    host_bytes = count * bytes_per_image
    gpu_estimate_bytes = bytes_per_image * intermediate_buffers

    return MemoryEstimate(
        count=count,
        height=height,
        width=width,
        bytes_per_pixel=bytes_per_pixel,
        host_bytes=host_bytes,
        intermediate_buffers=intermediate_buffers,
        gpu_estimate_bytes=gpu_estimate_bytes,
    )
