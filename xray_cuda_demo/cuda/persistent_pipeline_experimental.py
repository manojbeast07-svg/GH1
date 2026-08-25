"""Section 20C: EXPERIMENTAL Python-level API models around
xray_cuda.PersistentCudaPipeline (Section 20B's native persistent-buffer
class). Never imported by cuda/pipeline.py or ui/services.py's
production path -- the production run_basic_cuda_pipeline()/
run_enhanced_cuda_pipeline() functions are completely unaffected by
this module's existence.

Provides two API models for evaluation (spec items 3-4):

  Model B -- explicit caller-owned lifecycle:
      pipeline = PersistentCudaPipeline()
      try:
          result = pipeline.run(images, config)
      finally:
          pipeline.release()

  Model C -- context-managed:
      with PersistentCudaPipeline() as pipeline:
          result = pipeline.run(images, config)

Both models wrap the SAME underlying native object -- the only
difference is whether the caller manages the lifetime explicitly or via
`with`. `PersistentCudaPipelineService` (spec item 32) is a prototype of
session-scoped pipeline ownership, intended to be stored in
st.session_state exactly like every other piece of mutable state in
this project (never a module-level global -- see spec items 20-21).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import xray_cuda
from cpu.filters import FilterConfig
from cuda.gaussian import DEFAULT_ENHANCED_BLOCK, gaussian_kernel_1d, gaussian_kernel_2d
from cuda.gaussian import variant_to_int as gaussian_variant_to_int
from cuda.laplacian import DEFAULT_LAPLACIAN_ENHANCED_BLOCK, laplacian_kernel_2d, laplacian_variant_to_int
from cuda.median import DEFAULT_MEDIAN_ENHANCED_BLOCK, median_variant_to_int
from cuda.sobel import DEFAULT_SOBEL_ENHANCED_BLOCK, sobel_variant_to_int
from cuda.sobel import mode_to_int as sobel_mode_to_int
from cuda.threshold import DEFAULT_THRESHOLD_ENHANCED_BLOCK, threshold_variant_to_int


@dataclass
class PersistentPipelineTiming:
    """Mirrors cuda.pipeline.PipelineTiming's fields, plus the memory-
    strategy fields Section 20B's native RunTiming adds."""

    h2d_ms: float
    gaussian_ms: Optional[float]
    median_ms: Optional[float]
    sobel_ms: Optional[float]
    laplacian_ms: Optional[float]
    threshold_ms: Optional[float]
    d2h_ms: float
    compute_ms: float
    total_ms: float
    alloc_ms: float
    host_stage_ms: float
    used_persistent_buffers: bool
    used_pinned_memory: bool
    grew_gpu_buffers: bool

    @classmethod
    def from_dict(cls, d: dict) -> "PersistentPipelineTiming":
        return cls(
            h2d_ms=d["h2d_ms"], gaussian_ms=d["gaussian_ms"], median_ms=d["median_ms"],
            sobel_ms=d["sobel_ms"], laplacian_ms=d["laplacian_ms"], threshold_ms=d["threshold_ms"],
            d2h_ms=d["d2h_ms"], compute_ms=d["compute_ms"], total_ms=d["total_ms"],
            alloc_ms=d["alloc_ms"], host_stage_ms=d["host_stage_ms"],
            used_persistent_buffers=d["used_persistent_buffers"], used_pinned_memory=d["used_pinned_memory"],
            grew_gpu_buffers=d["grew_gpu_buffers"],
        )


class PersistentCudaPipeline:
    """EXPERIMENTAL Python-level wrapper around xray_cuda.PersistentCudaPipeline.

    Supports both Model B (explicit .release()) and Model C (`with`)
    lifecycles -- they are the same object, just two ways of ending its
    lifetime. GPU buffer persistence is controlled per-call
    (`use_persistent_buffers`, default True -- this class exists to
    demonstrate persistence, so it defaults on); pinned memory defaults
    OFF, matching Section 20B's REJECTED decision (spec item 39: do not
    re-adopt pinned memory here).
    """

    def __init__(self) -> None:
        self._native = xray_cuda.PersistentCudaPipeline()

    def __enter__(self) -> "PersistentCudaPipeline":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    @property
    def is_released(self) -> bool:
        return self._native.is_released

    def release(self) -> None:
        self._native.release()

    def run(
        self, images: List[np.ndarray], config: FilterConfig,
        use_enhanced_gaussian: bool = False, gaussian_variant: str = "specialized", gaussian_block: tuple = DEFAULT_ENHANCED_BLOCK,
        use_enhanced_median: bool = False, median_variant: str = "network3x3", median_block: tuple = DEFAULT_MEDIAN_ENHANCED_BLOCK,
        use_enhanced_sobel: bool = False, sobel_variant: str = "specialized", sobel_block: tuple = DEFAULT_SOBEL_ENHANCED_BLOCK,
        use_enhanced_laplacian: bool = False, laplacian_variant: str = "specialized", laplacian_block: tuple = DEFAULT_LAPLACIAN_ENHANCED_BLOCK,
        use_enhanced_threshold: bool = False, threshold_variant: str = "vectorized", threshold_block: tuple = DEFAULT_THRESHOLD_ENHANCED_BLOCK,
        use_persistent_buffers: bool = True, use_pinned_memory: bool = False,
    ) -> Tuple[np.ndarray, PersistentPipelineTiming]:
        """Same image-processing contract as cuda.pipeline.run_basic_cuda_pipeline()
        -- a list of same-shape images + a FilterConfig + per-stage
        enhanced flags/variants. Only the memory-strategy flags are new.
        Raises RuntimeError if called after release() (mirrors the
        native class's own explicit-lifecycle behavior)."""
        if not images:
            raise ValueError("PersistentCudaPipeline.run() requires at least one image.")
        shape = images[0].shape
        for i, image in enumerate(images):
            if image.shape != shape:
                raise ValueError(
                    f"PersistentCudaPipeline.run() requires all images to share one resolution; "
                    f"image 0 is {shape} but image {i} is {image.shape}.")

        batch = np.stack(images, axis=0)
        laplacian_coeffs = laplacian_kernel_2d(config.laplacian_kernel_size)
        if use_enhanced_gaussian:
            gaussian_coeffs_2d = gaussian_kernel_2d(3, 0.0)  # unused placeholder, matches production convention
            gaussian_coeffs_1d = gaussian_kernel_1d(config.gaussian_kernel_size, config.gaussian_sigma)
        else:
            gaussian_coeffs_2d = gaussian_kernel_2d(config.gaussian_kernel_size, config.gaussian_sigma)
            gaussian_coeffs_1d = None

        result = self._native.run(
            batch,
            config.gaussian_enabled, gaussian_coeffs_2d,
            config.median_enabled, config.median_kernel_size,
            config.sobel_enabled, sobel_mode_to_int(config.sobel_mode),
            config.laplacian_enabled, laplacian_coeffs, config.laplacian_scale, config.laplacian_delta,
            config.threshold_enabled, config.threshold_value, config.threshold_max_value,
            gaussian_use_enhanced=use_enhanced_gaussian,
            gaussian_coeffs_1d=gaussian_coeffs_1d if use_enhanced_gaussian else gaussian_kernel_1d(3, 0.0),
            gaussian_variant=gaussian_variant_to_int(gaussian_variant), gaussian_block_x=gaussian_block[0], gaussian_block_y=gaussian_block[1],
            median_use_enhanced=use_enhanced_median, median_variant=median_variant_to_int(median_variant),
            median_block_x=median_block[0], median_block_y=median_block[1],
            sobel_use_enhanced=use_enhanced_sobel, sobel_variant=sobel_variant_to_int(sobel_variant),
            sobel_block_x=sobel_block[0], sobel_block_y=sobel_block[1],
            laplacian_use_enhanced=use_enhanced_laplacian, laplacian_variant=laplacian_variant_to_int(laplacian_variant),
            laplacian_block_x=laplacian_block[0], laplacian_block_y=laplacian_block[1],
            threshold_use_enhanced=use_enhanced_threshold, threshold_variant=threshold_variant_to_int(threshold_variant),
            threshold_block_x=threshold_block[0], threshold_block_y=threshold_block[1],
            use_persistent_buffers=use_persistent_buffers, use_pinned_memory=use_pinned_memory,
        )
        output = result.pop("output")
        return output, PersistentPipelineTiming.from_dict(result)

    def run_enhanced(self, images: List[np.ndarray], config: FilterConfig) -> Tuple[np.ndarray, PersistentPipelineTiming]:
        """Convenience preset mirroring cuda.pipeline.run_enhanced_cuda_pipeline()'s
        all-Enhanced production defaults."""
        return self.run(
            images, config,
            use_enhanced_gaussian=True, gaussian_variant="specialized",
            use_enhanced_median=True, median_variant="network3x3",
            use_enhanced_sobel=True, sobel_variant="specialized",
            use_enhanced_laplacian=True, laplacian_variant="specialized",
            use_enhanced_threshold=True, threshold_variant="vectorized",
        )

    def run_basic(self, images: List[np.ndarray], config: FilterConfig) -> Tuple[np.ndarray, PersistentPipelineTiming]:
        """Convenience preset mirroring cuda.pipeline.run_basic_cuda_pipeline()'s
        all-Basic behavior."""
        return self.run(images, config)


class PersistentCudaPipelineService:
    """EXPERIMENTAL (spec item 32) prototype of session-scoped pipeline
    ownership: one Basic and one Enhanced PersistentCudaPipeline per
    service instance. Intended usage is ONE service instance stored per
    Streamlit session (e.g. in st.session_state), never a module-level
    global -- see spec items 20-21's session-isolation requirement,
    verified directly: two independent instances never share GPU
    buffers or state (each owns its own two native PersistentCudaPipeline
    objects).

    Deliberately NOT wired into ui/services.py -- this is a design
    prototype for evaluation, not a production integration.
    """

    def __init__(self) -> None:
        self._basic: Optional[PersistentCudaPipeline] = None
        self._enhanced: Optional[PersistentCudaPipeline] = None

    def get_basic_pipeline(self) -> PersistentCudaPipeline:
        if self._basic is None or self._basic.is_released:
            self._basic = PersistentCudaPipeline()
        return self._basic

    def get_enhanced_pipeline(self) -> PersistentCudaPipeline:
        if self._enhanced is None or self._enhanced.is_released:
            self._enhanced = PersistentCudaPipeline()
        return self._enhanced

    def release_all(self) -> None:
        if self._basic is not None:
            self._basic.release()
        if self._enhanced is not None:
            self._enhanced.release()
