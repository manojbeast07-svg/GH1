"""Production Basic CUDA pipeline (Section 5): Python orchestration
around the single native `xray_cuda.run_basic_cuda_pipeline_gpu()` call.

This module is the only place application code (benchmarks, the future
Streamlit UI, Enhanced-vs-Basic comparisons) should call to run the
five-filter GPU pipeline. The individual filter modules
(cuda/gaussian.py, cuda/median.py, ...) remain available for
per-filter testing and diagnostics, but production code should not call
them one at a time -- that reintroduces the five-Python-call overhead
Section 4F measured (~0.71ms of a ~1.035ms total for one 224x224 image).

Three responsibilities live here:
  1. run_basic_cuda_pipeline() -- run the pipeline on a list of images
     that already share one (height, width), via one native call.
  2. group_by_resolution() -- the real dataset is mixed-resolution
     (9,273 images at 224x224, ~190 spread across 150+ other sizes;
     Section 2 finding). Images are never resized/padded to force a
     common shape; instead they're grouped by their native shape so
     each native-pipeline call still gets a uniform [N,H,W] batch.
  3. compute_safe_gpu_batch_size() / run_basic_cuda_pipeline_selection()
     -- split a resolution group into GPU-sized chunks that fit safely
     within actual free VRAM (not relying on the WDDM over-commit
     behavior Section 4A observed), then reassemble results in the
     selection's original order regardless of chunking/grouping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import xray_cuda
from cpu.filters import FilterConfig
from cuda.gaussian import DEFAULT_ENHANCED_BLOCK, gaussian_kernel_1d, gaussian_kernel_2d
from cuda.gaussian import variant_to_int as gaussian_variant_to_int
from cuda.laplacian import DEFAULT_LAPLACIAN_ENHANCED_BLOCK
from cuda.laplacian import laplacian_kernel_2d
from cuda.laplacian import laplacian_variant_to_int
from cuda.median import DEFAULT_MEDIAN_ENHANCED_BLOCK
from cuda.median import median_variant_to_int
from cuda.sobel import DEFAULT_SOBEL_ENHANCED_BLOCK
from cuda.sobel import mode_to_int as sobel_mode_to_int
from cuda.sobel import sobel_variant_to_int
from cuda.threshold import DEFAULT_THRESHOLD_ENHANCED_BLOCK
from cuda.threshold import threshold_variant_to_int
from pipeline.dataset import ImageSelection, SelectedItem
from pipeline.image_loader import load_image


@dataclass
class PipelineTiming:
    """Mirrors xray_cuda.run_basic_cuda_pipeline_gpu()'s returned timing
    fields. A disabled stage's field is None (never 0.0 -- see
    cpu.pipeline.TimingResult for the same convention)."""

    h2d_ms: float
    gaussian_ms: Optional[float]
    median_ms: Optional[float]
    sobel_ms: Optional[float]
    laplacian_ms: Optional[float]
    threshold_ms: Optional[float]
    d2h_ms: float
    compute_ms: float
    total_ms: float

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineTiming":
        return cls(
            h2d_ms=d["h2d_ms"], gaussian_ms=d["gaussian_ms"], median_ms=d["median_ms"],
            sobel_ms=d["sobel_ms"], laplacian_ms=d["laplacian_ms"], threshold_ms=d["threshold_ms"],
            d2h_ms=d["d2h_ms"], compute_ms=d["compute_ms"], total_ms=d["total_ms"],
        )


def run_basic_cuda_pipeline(
    images: List[np.ndarray],
    config: FilterConfig,
    use_enhanced_gaussian: bool = False,
    gaussian_variant: str = "specialized",
    gaussian_block: tuple = DEFAULT_ENHANCED_BLOCK,
    use_enhanced_median: bool = False,
    median_variant: str = "network3x3",
    median_block: tuple = DEFAULT_MEDIAN_ENHANCED_BLOCK,
    use_enhanced_sobel: bool = False,
    sobel_variant: str = "specialized",
    sobel_block: tuple = DEFAULT_SOBEL_ENHANCED_BLOCK,
    use_enhanced_laplacian: bool = False,
    laplacian_variant: str = "specialized",
    laplacian_block: tuple = DEFAULT_LAPLACIAN_ENHANCED_BLOCK,
    use_enhanced_threshold: bool = False,
    threshold_variant: str = "vectorized",
    threshold_block: tuple = DEFAULT_THRESHOLD_ENHANCED_BLOCK,
) -> Tuple[np.ndarray, PipelineTiming]:
    """Run the full CUDA pipeline, in ONE native call, on a list of images
    that all share one (height, width). Returns a [N,H,W] uint8 array
    (order matches `images`) and the native timing breakdown.

    Coefficient generation (Gaussian/Laplacian) happens here in Python
    (same reasoning as cuda/gaussian.py and cuda/laplacian.py: bit-
    identical to what OpenCV would use) and is passed into the single
    native call -- it does not cost an extra Python/CUDA round trip.

    `use_enhanced_gaussian` (Section 6) / `use_enhanced_median` (Section
    7) / `use_enhanced_sobel` (Section 8) / `use_enhanced_laplacian`
    (Section 9) / `use_enhanced_threshold` (Section 10), all default
    False (preserves Section 5's all-Basic behavior): each independently
    swaps just that one stage for its Enhanced kernel, so any combination
    is directly comparable via one call path. With all five set True,
    this is the "Enhanced CUDA" pipeline -- see also
    run_enhanced_cuda_pipeline() below, a thin convenience wrapper around
    exactly that combination.
    """
    if not images:
        raise ValueError("run_basic_cuda_pipeline requires at least one image.")
    shape = images[0].shape
    for i, image in enumerate(images):
        if image.shape != shape:
            raise ValueError(
                f"run_basic_cuda_pipeline requires all images to share one resolution; "
                f"image 0 is {shape} but image {i} is {image.shape}. Use group_by_resolution() first."
            )

    batch = np.stack(images, axis=0)  # [N, H, W], uint8
    laplacian_coeffs = laplacian_kernel_2d(config.laplacian_kernel_size)

    if use_enhanced_gaussian:
        gaussian_coeffs_2d = gaussian_kernel_2d(3, 0.0)  # unused placeholder; C++ ignores it when gaussian_use_enhanced=True
        gaussian_coeffs_1d = gaussian_kernel_1d(config.gaussian_kernel_size, config.gaussian_sigma)
    else:
        gaussian_coeffs_2d = gaussian_kernel_2d(config.gaussian_kernel_size, config.gaussian_sigma)
        gaussian_coeffs_1d = None  # unused when gaussian_use_enhanced=False, but the binding requires *some* array

    result = xray_cuda.run_basic_cuda_pipeline_gpu(
        batch,
        config.gaussian_enabled, gaussian_coeffs_2d,
        config.median_enabled, config.median_kernel_size,
        config.sobel_enabled, sobel_mode_to_int(config.sobel_mode),
        config.laplacian_enabled, laplacian_coeffs, config.laplacian_scale, config.laplacian_delta,
        config.threshold_enabled, config.threshold_value, config.threshold_max_value,
        use_enhanced_gaussian, gaussian_coeffs_1d if use_enhanced_gaussian else gaussian_kernel_1d(3, 0.0),
        gaussian_variant_to_int(gaussian_variant), gaussian_block[0], gaussian_block[1],
        use_enhanced_median, median_variant_to_int(median_variant), median_block[0], median_block[1],
        use_enhanced_sobel, sobel_variant_to_int(sobel_variant), sobel_block[0], sobel_block[1],
        use_enhanced_laplacian, laplacian_variant_to_int(laplacian_variant), laplacian_block[0], laplacian_block[1],
        use_enhanced_threshold, threshold_variant_to_int(threshold_variant), threshold_block[0], threshold_block[1],
    )
    return result["output"], PipelineTiming.from_dict(result)


def run_enhanced_cuda_pipeline(
    images: List[np.ndarray],
    config: FilterConfig,
    gaussian_variant: str = "specialized",
    gaussian_block: tuple = DEFAULT_ENHANCED_BLOCK,
    median_variant: str = "network3x3",
    median_block: tuple = DEFAULT_MEDIAN_ENHANCED_BLOCK,
    sobel_variant: str = "specialized",
    sobel_block: tuple = DEFAULT_SOBEL_ENHANCED_BLOCK,
    laplacian_variant: str = "specialized",
    laplacian_block: tuple = DEFAULT_LAPLACIAN_ENHANCED_BLOCK,
    threshold_variant: str = "vectorized",
    threshold_block: tuple = DEFAULT_THRESHOLD_ENHANCED_BLOCK,
) -> Tuple[np.ndarray, PipelineTiming]:
    """The official Enhanced CUDA pipeline (Section 10 spec item 18):
    Enhanced Gaussian -> Enhanced Median -> Enhanced Sobel -> Enhanced
    Laplacian -> Enhanced Threshold, using each stage's Section 6-10
    measured-best variant by default. A thin wrapper around
    run_basic_cuda_pipeline() with all five `use_enhanced_*` flags set
    True -- kept as a separate, named function (not just a convention
    callers have to remember) so this exact configuration is never
    silently redefined elsewhere, mirroring run_basic_cuda_pipeline()'s
    role as the official all-Basic baseline (spec item 17).
    """
    return run_basic_cuda_pipeline(
        images, config,
        use_enhanced_gaussian=True, gaussian_variant=gaussian_variant, gaussian_block=gaussian_block,
        use_enhanced_median=True, median_variant=median_variant, median_block=median_block,
        use_enhanced_sobel=True, sobel_variant=sobel_variant, sobel_block=sobel_block,
        use_enhanced_laplacian=True, laplacian_variant=laplacian_variant, laplacian_block=laplacian_block,
        use_enhanced_threshold=True, threshold_variant=threshold_variant, threshold_block=threshold_block,
    )


@dataclass
class PipelineConfig:
    """CUDA *implementation* parameters -- which kernel family/variant/
    launch configuration runs each stage (Section 10 spec items 19-20).

    Deliberately separate from cpu.filters.FilterConfig, which holds
    *image-processing* parameters (Gaussian kernel/sigma, Median kernel,
    Sobel mode, Laplacian kernel/scale/delta, Threshold value/max_value)
    -- changing a PipelineConfig field can only affect performance and
    numerical rounding (within each stage's already-verified tolerance),
    never which processing the images conceptually receive; changing a
    FilterConfig field can only affect what processing happens, never
    which CUDA code path runs it. The future Streamlit UI exposes these
    as two separate control groups for exactly this reason -- an
    "Enhanced" toggle per filter plus its variant/block choice belongs
    here, not next to the image-processing sliders.

    Each `*_variant` is either "basic" (use the unchanged Basic kernel)
    or one of that filter's Enhanced variant names (GAUSSIAN_VARIANTS /
    MEDIAN_VARIANTS / SOBEL_VARIANTS / LAPLACIAN_VARIANTS /
    THRESHOLD_VARIANTS) -- never hardwired to one configuration, per
    spec item 19.
    """

    gaussian_variant: str = "basic"
    gaussian_block: tuple = DEFAULT_ENHANCED_BLOCK
    median_variant: str = "basic"
    median_block: tuple = DEFAULT_MEDIAN_ENHANCED_BLOCK
    sobel_variant: str = "basic"
    sobel_block: tuple = DEFAULT_SOBEL_ENHANCED_BLOCK
    laplacian_variant: str = "basic"
    laplacian_block: tuple = DEFAULT_LAPLACIAN_ENHANCED_BLOCK
    threshold_variant: str = "basic"
    threshold_block: tuple = DEFAULT_THRESHOLD_ENHANCED_BLOCK

    @classmethod
    def all_basic(cls) -> "PipelineConfig":
        """Equivalent to run_basic_cuda_pipeline()'s defaults."""
        return cls()

    @classmethod
    def all_enhanced(cls) -> "PipelineConfig":
        """Equivalent to run_enhanced_cuda_pipeline()'s defaults."""
        return cls(
            gaussian_variant="specialized", median_variant="network3x3", sobel_variant="specialized",
            laplacian_variant="specialized", threshold_variant="vectorized",
        )


def run_cuda_pipeline(
    images: List[np.ndarray], filter_config: FilterConfig, pipeline_config: PipelineConfig,
) -> Tuple[np.ndarray, PipelineTiming]:
    """Run the pipeline using `pipeline_config` to independently choose,
    per stage, Basic vs. a named Enhanced variant -- the general form
    run_basic_cuda_pipeline()/run_enhanced_cuda_pipeline() are named
    special cases of (all-Basic and all-Enhanced-with-defaults,
    respectively). Intended for the future Streamlit UI's per-filter
    "Enhanced" toggles (spec item 19), where any combination of
    Basic/Enhanced across the five stages must be selectable, not just
    the two canonical endpoints.
    """
    return run_basic_cuda_pipeline(
        images, filter_config,
        use_enhanced_gaussian=pipeline_config.gaussian_variant != "basic",
        gaussian_variant=pipeline_config.gaussian_variant if pipeline_config.gaussian_variant != "basic" else "specialized",
        gaussian_block=pipeline_config.gaussian_block,
        use_enhanced_median=pipeline_config.median_variant != "basic",
        median_variant=pipeline_config.median_variant if pipeline_config.median_variant != "basic" else "network3x3",
        median_block=pipeline_config.median_block,
        use_enhanced_sobel=pipeline_config.sobel_variant != "basic",
        sobel_variant=pipeline_config.sobel_variant if pipeline_config.sobel_variant != "basic" else "specialized",
        sobel_block=pipeline_config.sobel_block,
        use_enhanced_laplacian=pipeline_config.laplacian_variant != "basic",
        laplacian_variant=pipeline_config.laplacian_variant if pipeline_config.laplacian_variant != "basic" else "specialized",
        laplacian_block=pipeline_config.laplacian_block,
        use_enhanced_threshold=pipeline_config.threshold_variant != "basic",
        threshold_variant=pipeline_config.threshold_variant if pipeline_config.threshold_variant != "basic" else "vectorized",
        threshold_block=pipeline_config.threshold_block,
    )


def group_by_resolution(selection: ImageSelection) -> Dict[Tuple[int, int], List[SelectedItem]]:
    """Loads every image in `selection` (unavoidable -- shape isn't known
    without decoding) and groups items by (height, width), preserving
    each group's relative order from the original selection. Images are
    never resized or padded to force a common group.
    """
    groups: Dict[Tuple[int, int], List[SelectedItem]] = {}
    for item in selection.items:
        image = load_image(item.absolute_path)
        groups.setdefault(image.shape, []).append(item)
    return groups


def compute_safe_gpu_batch_size(
    height: int,
    width: int,
    free_bytes: Optional[int] = None,
    bytes_per_pixel: int = 1,
    num_buffers: int = 2,
    safety_factor: float = 0.7,
) -> int:
    """Maximum number of `height`x`width` images that can safely sit in
    `num_buffers` simultaneous device buffers (the pipeline's ping-pong
    pair) within `safety_factor` of currently free VRAM.

    Deliberately conservative and based on actual reported free VRAM,
    not on the WDDM over-commit behavior Section 4A observed (60x200MB
    allocations succeeding on an 8GB card by silently spilling into
    system RAM) -- that behavior is real but not something this project
    plans capacity around, since it isn't guaranteed and isn't available
    on non-Windows/non-WDDM systems.
    """
    if free_bytes is None:
        free_bytes = xray_cuda.device_memory_info()["free_bytes"]
    if not (0.0 < safety_factor <= 1.0):
        raise ValueError(f"safety_factor must be in (0, 1], got {safety_factor}")

    usable_bytes = free_bytes * safety_factor
    bytes_per_image_per_buffer = height * width * bytes_per_pixel
    bytes_per_image_total = bytes_per_image_per_buffer * num_buffers
    if bytes_per_image_total <= 0:
        raise ValueError("height/width/bytes_per_pixel/num_buffers must be positive.")

    max_batch = int(usable_bytes // bytes_per_image_total)
    return max(1, max_batch)


def run_basic_cuda_pipeline_selection(
    selection: ImageSelection,
    config: FilterConfig,
    max_batch_size: Optional[int] = None,
) -> List[Tuple[SelectedItem, np.ndarray]]:
    """Top-level orchestrator: groups `selection` by resolution, splits
    each group into GPU-safe chunks (`max_batch_size`, capped by
    compute_safe_gpu_batch_size() if not given), runs the native
    pipeline once per chunk, and returns (item, output_image) pairs in
    the SAME ORDER as `selection.items` -- regardless of how many
    resolution groups or GPU chunks that required internally.
    """
    groups = group_by_resolution(selection)

    # Keyed by SelectedItem.index (the item's unique position in the
    # scanned dataset -- guaranteed unique within one selection, per
    # Section 2's no-duplicates guarantee), not object identity, so
    # reassembly is explicit rather than relying on id().
    results_by_index: Dict[int, np.ndarray] = {}
    for (height, width), items in groups.items():
        # Effective GPU chunk size is always capped by what safely fits
        # in VRAM, regardless of what the caller requested (spec: a
        # requested batch larger than GPU capacity is chunked, not
        # rejected; a caller-requested size smaller than GPU capacity is
        # still respected).
        safe_capacity = compute_safe_gpu_batch_size(height, width)
        chunk_capacity = min(max_batch_size, safe_capacity) if max_batch_size is not None else safe_capacity
        chunk_capacity = max(1, chunk_capacity)

        for start in range(0, len(items), chunk_capacity):
            chunk_items = items[start : start + chunk_capacity]
            chunk_images = [load_image(item.absolute_path) for item in chunk_items]
            output_batch, _timing = run_basic_cuda_pipeline(chunk_images, config)
            for item, output_image in zip(chunk_items, output_batch):
                results_by_index[item.index] = output_image

    return [(item, results_by_index[item.index]) for item in selection.items]
