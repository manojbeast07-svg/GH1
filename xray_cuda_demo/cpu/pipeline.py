"""CPU reference pipeline: chains the five filters in canonical order.

--------------------------------------------------------------------------
Why this is not optimized
--------------------------------------------------------------------------
This is the *reference* implementation, not a performance target. No
multiprocessing, no OpenMP, no vectorized CPU libraries, no GPU-backed
OpenCV. The point of Section 3 is a simple, trustworthy baseline that the
Basic and Enhanced CUDA implementations (later sections) must reproduce
output-for-output; making the CPU path fast would only make it a moving
target.

--------------------------------------------------------------------------
Execution order
--------------------------------------------------------------------------
Original -> Gaussian -> Median -> Sobel -> Laplacian -> Threshold

This order is fixed. Disabling a stage removes it from the chain (the
next enabled stage receives the previous enabled stage's output, not the
disabled stage's would-have-been output); stages are never reordered.
If every stage is disabled, final_output is the original image,
unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np

from cpu.filters import FILTER_STAGES, FilterConfig, validate_input_contract
from pipeline.dataset import ImageSelection, SelectedItem
from pipeline.image_loader import load_image
from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """All intermediate outputs. A disabled stage's field stays None --
    it is never computed just to be discarded, and never silently
    recomputed from a different input."""

    original: np.ndarray
    gaussian_output: Optional[np.ndarray] = None
    median_output: Optional[np.ndarray] = None
    sobel_output: Optional[np.ndarray] = None
    laplacian_output: Optional[np.ndarray] = None
    threshold_output: Optional[np.ndarray] = None

    @property
    def final_output(self) -> np.ndarray:
        """The output of the last enabled stage, or the original image if
        every stage was disabled."""
        for stage in (
            self.threshold_output,
            self.laplacian_output,
            self.sobel_output,
            self.median_output,
            self.gaussian_output,
        ):
            if stage is not None:
                return stage
        return self.original

    def get(self, stage_name: str) -> Optional[np.ndarray]:
        return getattr(self, f"{stage_name}_output")


@dataclass
class TimingResult:
    """Per-filter wall-clock time in milliseconds. None for a disabled
    stage (never 0.0 -- 0.0 would falsely imply the stage ran instantly)."""

    gaussian_ms: Optional[float] = None
    median_ms: Optional[float] = None
    sobel_ms: Optional[float] = None
    laplacian_ms: Optional[float] = None
    threshold_ms: Optional[float] = None

    @property
    def total_ms(self) -> float:
        """Sum of enabled-stage times only -- this is the "processing
        benchmark" defined in the spec: image-processing time alone,
        excluding disk load."""
        return sum(v for v in self.as_dict().values() if v is not None)

    def as_dict(self) -> dict:
        return {
            "gaussian": self.gaussian_ms,
            "median": self.median_ms,
            "sobel": self.sobel_ms,
            "laplacian": self.laplacian_ms,
            "threshold": self.threshold_ms,
        }


def run_cpu_pipeline(image: np.ndarray, config: FilterConfig) -> Tuple[PipelineResult, TimingResult]:
    """Run the enabled stages of the CPU reference pipeline on one
    already-loaded image. Does not touch disk. Deterministic: the same
    (image, config) always produces bit-identical outputs and a new,
    independently-measured (but structurally identical) timing result.
    """
    validate_input_contract(image)

    result = PipelineResult(original=image)
    timing = TimingResult()
    current = image

    for name, apply_fn in FILTER_STAGES:
        if not getattr(config, f"{name}_enabled"):
            continue
        start = time.perf_counter()
        current = apply_fn(current, config)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        setattr(result, f"{name}_output", current)
        setattr(timing, f"{name}_ms", elapsed_ms)

    return result, timing


def run_cpu_image(path: "str | object", config: FilterConfig) -> Tuple[PipelineResult, TimingResult]:
    """Load one image from disk (grayscale uint8, per the Section 2
    contract) and run the pipeline on it."""
    image = load_image(path)
    return run_cpu_pipeline(image, config)


def run_cpu_selection(
    selection: ImageSelection, config: FilterConfig
) -> Iterator[Tuple[SelectedItem, PipelineResult, TimingResult]]:
    """Process every image in a Section 2 ImageSelection, one at a time.

    Deliberately a generator, not a list: results (including every
    intermediate uint8 array) are only held by the caller for as long as
    it keeps iterating, so a caller that only needs timings (e.g. a
    benchmark) is never forced to hold hundreds of decoded images in
    memory at once.

    The real dataset is mixed-resolution (mostly 224x224, but ~190
    images at 150+ other resolutions). This function makes no attempt to
    stack results into a single [N,H,W] array -- each image is loaded
    and processed independently at its native resolution, so mixed
    resolutions never crash this path. A fixed-shape GPU batching
    strategy is an explicit later-section decision, not made here.
    """
    for item in selection.items:
        image = load_image(item.absolute_path)
        result, timing = run_cpu_pipeline(image, config)
        yield item, result, timing
