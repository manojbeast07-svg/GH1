"""Section 17: single-variant kernel timing for one filter, isolated
from the other four pipeline stages -- shared by
scripts/run_optimization_lab_experiments.py (the historical backfill
sweep) and ui/services.py (the Streamlit Optimization Lab's live
variant comparison), so both measure exactly the same way and never
duplicate the CUDA-calling logic.

Deliberately calls each filter's standalone `xray_cuda.<filter>_
enhanced_batch_gpu` / `run_basic_cuda_pipeline_gpu` function directly,
the same direct-call methodology scripts/benchmark_<filter>_
optimization.py established in Sections 6-10 -- NOT
cuda.pipeline.run_cuda_pipeline()'s general 5-stage wrapper, whose
extra layer of overhead was measured to distort comparisons for these
sub-millisecond kernels (see run_optimization_lab_experiments.py's
module docstring for the reproduction).

No CUDA source is compiled or modified here -- every variant is an
already-built kernel the compiled `xray_cuda` extension already
exposes.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

import xray_cuda
from cpu.filters import FilterConfig
from cuda.gaussian import GAUSSIAN_VARIANTS, gaussian_kernel_1d, gaussian_kernel_2d, variant_to_int as gaussian_variant_to_int
from cuda.laplacian import LAPLACIAN_VARIANTS, laplacian_kernel_2d, laplacian_variant_to_int
from cuda.median import MEDIAN_VARIANTS, median_variant_to_int
from cuda.sobel import SOBEL_VARIANTS, mode_to_int, sobel_variant_to_int
from cuda.threshold import THRESHOLD_VARIANTS, threshold_variant_to_int

FILTER_VARIANTS = {
    "gaussian": GAUSSIAN_VARIANTS, "median": MEDIAN_VARIANTS, "sobel": SOBEL_VARIANTS,
    "laplacian": LAPLACIAN_VARIANTS, "threshold": THRESHOLD_VARIANTS,
}

PRODUCTION_DEFAULTS = {
    "gaussian": "specialized", "median": "network3x3", "sobel": "specialized",
    "laplacian": "specialized", "threshold": "vectorized",
}

DEFAULT_BLOCK = (16, 16)


def available_variants(filter_name: str) -> Tuple[str, ...]:
    if filter_name not in FILTER_VARIANTS:
        raise ValueError(f"Unknown filter: {filter_name!r}; expected one of {sorted(FILTER_VARIANTS)}")
    return ("basic",) + FILTER_VARIANTS[filter_name]


def measure_variant(
    filter_name: str, batch: np.ndarray, config: FilterConfig, variant: str,
    warmup_runs: int, measurement_runs: int, block: Tuple[int, int] = DEFAULT_BLOCK,
) -> Tuple[List[float], np.ndarray]:
    """Runs `variant` ("basic" or one of that filter's Enhanced variant
    names) `warmup_runs` times (discarded) then `measurement_runs` times
    (kept), on the SAME already-uploaded batch, and returns
    (per-run kernel_ms list, [N,H,W] uint8 output). `config` supplies
    that filter's own image-processing parameters (kernel_size/sigma/
    scale/delta/threshold values) -- every other stage is disabled so
    the measured kernel_ms is this filter's alone.
    """
    if filter_name not in FILTER_VARIANTS:
        raise ValueError(f"Unknown filter: {filter_name!r}; expected one of {sorted(FILTER_VARIANTS)}")
    valid = available_variants(filter_name)
    if variant not in valid:
        raise ValueError(f"variant={variant!r} is not valid for filter={filter_name!r}; expected one of {valid}")

    gaussian_placeholder = gaussian_kernel_2d(3, 0.0)
    laplacian_placeholder = laplacian_kernel_2d(3)

    if filter_name == "gaussian":
        coeffs2d = gaussian_kernel_2d(config.gaussian_kernel_size, config.gaussian_sigma)
        if variant == "basic":
            def run_once():
                return xray_cuda.run_basic_cuda_pipeline_gpu(
                    batch, True, coeffs2d, False, 3, False, 2, False, coeffs2d, 1.0, 0.0, False, 128, 255)
            key = "gaussian_ms"
        else:
            coeffs1d = gaussian_kernel_1d(config.gaussian_kernel_size, config.gaussian_sigma)
            vint = gaussian_variant_to_int(variant)

            def run_once():
                return xray_cuda.gaussian_enhanced_batch_gpu(batch, coeffs1d, vint, block[0], block[1])
            key = "kernel_ms"

    elif filter_name == "median":
        if variant == "basic":
            def run_once():
                return xray_cuda.run_basic_cuda_pipeline_gpu(
                    batch, False, gaussian_placeholder, True, config.median_kernel_size, False, 2,
                    False, laplacian_placeholder, 1.0, 0.0, False, 128, 255)
            key = "median_ms"
        else:
            vint = median_variant_to_int(variant)

            def run_once():
                return xray_cuda.median_enhanced_batch_gpu(batch, config.median_kernel_size, vint, block[0], block[1])
            key = "kernel_ms"

    elif filter_name == "sobel":
        mode_int = mode_to_int(config.sobel_mode)
        if variant == "basic":
            def run_once():
                return xray_cuda.run_basic_cuda_pipeline_gpu(
                    batch, False, gaussian_placeholder, False, 3, True, mode_int,
                    False, laplacian_placeholder, 1.0, 0.0, False, 128, 255)
            key = "sobel_ms"
        else:
            vint = sobel_variant_to_int(variant)

            def run_once():
                return xray_cuda.sobel_enhanced_batch_gpu(batch, mode_int, vint, block[0], block[1])
            key = "kernel_ms"

    elif filter_name == "laplacian":
        coeffs = laplacian_kernel_2d(config.laplacian_kernel_size)
        if variant == "basic":
            def run_once():
                return xray_cuda.run_basic_cuda_pipeline_gpu(
                    batch, False, gaussian_placeholder, False, 3, False, 2,
                    True, coeffs, config.laplacian_scale, config.laplacian_delta, False, 128, 255)
            key = "laplacian_ms"
        else:
            vint = laplacian_variant_to_int(variant)

            def run_once():
                return xray_cuda.laplacian_enhanced_batch_gpu(
                    batch, coeffs, config.laplacian_scale, config.laplacian_delta, vint, block[0], block[1])
            key = "kernel_ms"

    else:  # threshold
        if variant == "basic":
            def run_once():
                return xray_cuda.run_basic_cuda_pipeline_gpu(
                    batch, False, gaussian_placeholder, False, 3, False, 2,
                    False, laplacian_placeholder, 1.0, 0.0, True, config.threshold_value, config.threshold_max_value)
            key = "threshold_ms"
        else:
            vint = threshold_variant_to_int(variant)

            def run_once():
                return xray_cuda.threshold_enhanced_batch_gpu(
                    batch, config.threshold_value, config.threshold_max_value, vint, block[0], block[1])
            key = "kernel_ms"

    for _ in range(warmup_runs):
        run_once()
    values, out = [], None
    for _ in range(measurement_runs):
        r = run_once()
        values.append(r[key])
        out = r["output"]
    return values, out
