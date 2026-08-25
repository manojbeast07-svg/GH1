"""Deterministic synthetic image fixtures for CPU filter/pipeline testing.

Small (64x64) and cheap to regenerate on every test run rather than
stored as files -- these are pure functions of a fixed seed, so "stored"
and "regenerated" are equivalent. Only computed *reference outputs*
(tests/reference/) are the things that must not silently change.
"""

from __future__ import annotations

import numpy as np

FIXTURE_SIZE = 64


def constant_image(size: int = FIXTURE_SIZE, value: int = 128) -> np.ndarray:
    return np.full((size, size), value, dtype=np.uint8)


def horizontal_gradient(size: int = FIXTURE_SIZE) -> np.ndarray:
    row = np.linspace(0, 255, size, dtype=np.float64)
    return np.tile(row, (size, 1)).astype(np.uint8)


def vertical_gradient(size: int = FIXTURE_SIZE) -> np.ndarray:
    return horizontal_gradient(size).T.copy()


def sharp_edge_square(size: int = FIXTURE_SIZE, border: int = None) -> np.ndarray:
    """Black background with a centered white square -> sharp edges in
    both x and y, useful for Sobel/Laplacian/border testing."""
    if border is None:
        border = size // 4
    image = np.zeros((size, size), dtype=np.uint8)
    image[border : size - border, border : size - border] = 255
    return image


def impulse_noise(size: int = FIXTURE_SIZE, seed: int = 0, density: float = 0.05) -> np.ndarray:
    """Mid-gray background with sparse salt-and-pepper impulses --
    exercises median filtering specifically."""
    rng = np.random.default_rng(seed)
    image = np.full((size, size), 128, dtype=np.uint8)
    mask = rng.random((size, size)) < density
    salt = rng.random((size, size)) < 0.5
    image[mask & salt] = 255
    image[mask & ~salt] = 0
    return image


def random_deterministic(size: int = FIXTURE_SIZE, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size), dtype=np.uint8)


# name -> generator, iterated by the reference-output generator script and
# by tests that want to exercise every fixture uniformly.
FIXTURES = {
    "constant": constant_image,
    "horizontal_gradient": horizontal_gradient,
    "vertical_gradient": vertical_gradient,
    "sharp_edge_square": sharp_edge_square,
    "impulse_noise": impulse_noise,
    "random_deterministic": random_deterministic,
}
