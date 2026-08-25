"""Single-image loading.

Input contract (documented, not incidental):
    channels = 1 (grayscale)
    dtype    = uint8
    range    = 0-255, unmodified from the source file

Grayscale + uint8 was chosen because the eventual five-stage pipeline
(Gaussian, median, Sobel, Laplacian, threshold) is designed around
single-channel X-ray processing, and every implementation (CPU, Basic
CUDA, Enhanced CUDA) must start from identical pixel values for the
comparison to be fair. No per-image normalization is applied here --
scaling each image by its own min/max would make the three
implementations' inputs diverge in a way that has nothing to do with
their filters, so any normalization is deferred to a later, explicitly
documented pipeline stage rather than done silently at load time.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ImageLoadError(RuntimeError):
    """Raised when an image cannot be loaded as grayscale uint8."""


def load_image(path: "str | Path") -> np.ndarray:
    """Load a single image as grayscale uint8, shape (H, W).

    Raises ImageLoadError (never returns None, never silently substitutes
    a placeholder) if the file is missing, unreadable, or fails to decode.
    """
    path = Path(path)
    if not path.exists():
        raise ImageLoadError(f"Image file does not exist: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ImageLoadError(f"Failed to decode image (corrupt or unsupported content): {path}")

    if image.dtype != np.uint8:
        # cv2.IMREAD_GRAYSCALE normally guarantees uint8; this guards
        # against unusual source formats (e.g. 16-bit) rather than
        # silently reinterpreting the pixel range.
        raise ImageLoadError(
            f"Unexpected dtype {image.dtype} decoding {path}; expected uint8 grayscale."
        )

    return image
