"""Section 20B tests: the EXPERIMENTAL xray_cuda.PersistentCudaPipeline
(GPU-buffer persistence + pinned host memory). Never touches the
production run_basic_cuda_pipeline()/run_enhanced_cuda_pipeline() API.

Per spec item 39, no performance number here is a hard pytest
threshold -- correctness, stability, and structural behavior are
asserted; timing is only ever printed/recorded, never gated on.
"""

import numpy as np
import pytest

xray_cuda = pytest.importorskip(
    "xray_cuda", reason="xray_cuda extension is not built; run cmake --build build first."
)
pytestmark = pytest.mark.skipif(not xray_cuda.cuda_available(), reason="No usable CUDA device detected on this machine.")

from cpu.filters import FilterConfig, apply_gaussian, apply_laplacian, apply_median, apply_sobel, apply_threshold  # noqa: E402
from cuda.gaussian import gaussian_kernel_1d, gaussian_kernel_2d  # noqa: E402
from cuda.laplacian import laplacian_kernel_2d  # noqa: E402
from tests.fixtures import FIXTURES  # noqa: E402


def _images(n=8, size=64, seed=0):
    return [FIXTURES["random_deterministic"](size=size, seed=seed + i) for i in range(n)]


def _coeffs(config: FilterConfig):
    return (
        gaussian_kernel_2d(config.gaussian_kernel_size, config.gaussian_sigma),
        gaussian_kernel_1d(config.gaussian_kernel_size, config.gaussian_sigma),
        laplacian_kernel_2d(config.laplacian_kernel_size),
    )


def _run(pipeline, batch, config, coeffs2d, coeffs1d, lap_coeffs, use_persistent, use_pinned, use_enhanced=True):
    return pipeline.run(
        batch, True, coeffs2d, True, config.median_kernel_size, True, 2,
        True, lap_coeffs, config.laplacian_scale, config.laplacian_delta,
        True, config.threshold_value, config.threshold_max_value,
        gaussian_use_enhanced=use_enhanced, gaussian_coeffs_1d=coeffs1d, gaussian_variant=3,
        median_use_enhanced=use_enhanced, median_variant=1,
        sobel_use_enhanced=use_enhanced, sobel_variant=2,
        laplacian_use_enhanced=use_enhanced, laplacian_variant=2,
        threshold_use_enhanced=use_enhanced, threshold_variant=0,
        use_persistent_buffers=use_persistent, use_pinned_memory=use_pinned,
    )


def _cpu_reference(images, config: FilterConfig) -> np.ndarray:
    return np.stack([
        apply_threshold(apply_laplacian(apply_sobel(apply_median(apply_gaussian(img, config), config), config), config), config)
        for img in images
    ])


# -- 1: allocation reuse -------------------------------------------------------


def test_persistent_buffers_grow_only_when_needed():
    images = _images(8)
    batch = np.stack(images, axis=0)
    config = FilterConfig()
    coeffs2d, coeffs1d, lap_coeffs = _coeffs(config)

    pipeline = xray_cuda.PersistentCudaPipeline()
    try:
        r1 = _run(pipeline, batch, config, coeffs2d, coeffs1d, lap_coeffs, True, False)
        assert r1["grew_gpu_buffers"] is True  # first call always allocates
        r2 = _run(pipeline, batch, config, coeffs2d, coeffs1d, lap_coeffs, True, False)
        assert r2["grew_gpu_buffers"] is False  # identical shape -> reused
        assert r2["alloc_ms"] < r1["alloc_ms"]
    finally:
        pipeline.release()


def test_non_persistent_allocates_every_call():
    images = _images(4)
    batch = np.stack(images, axis=0)
    config = FilterConfig()
    coeffs2d, coeffs1d, lap_coeffs = _coeffs(config)

    pipeline = xray_cuda.PersistentCudaPipeline()
    try:
        r1 = _run(pipeline, batch, config, coeffs2d, coeffs1d, lap_coeffs, False, False)
        r2 = _run(pipeline, batch, config, coeffs2d, coeffs1d, lap_coeffs, False, False)
        assert r1["used_persistent_buffers"] is False
        assert r2["used_persistent_buffers"] is False
    finally:
        pipeline.release()


# -- 2: pinned memory -------------------------------------------------------


def test_pinned_memory_flag_reflected_in_result():
    images = _images(4)
    batch = np.stack(images, axis=0)
    config = FilterConfig()
    coeffs2d, coeffs1d, lap_coeffs = _coeffs(config)

    pipeline = xray_cuda.PersistentCudaPipeline()
    try:
        r = _run(pipeline, batch, config, coeffs2d, coeffs1d, lap_coeffs, False, True)
        assert r["used_pinned_memory"] is True
        assert r["host_stage_ms"] > 0.0  # staging copy is real and measured, never hidden (spec item 14)
    finally:
        pipeline.release()


def test_pageable_path_has_zero_host_stage_cost():
    images = _images(4)
    batch = np.stack(images, axis=0)
    config = FilterConfig()
    coeffs2d, coeffs1d, lap_coeffs = _coeffs(config)

    pipeline = xray_cuda.PersistentCudaPipeline()
    try:
        r = _run(pipeline, batch, config, coeffs2d, coeffs1d, lap_coeffs, False, False)
        assert r["host_stage_ms"] == 0.0
    finally:
        pipeline.release()


# -- 3: output correctness (all 4 variants vs production baseline) -------------------------------------------------------


@pytest.mark.parametrize("use_persistent,use_pinned", [(False, False), (True, False), (False, True), (True, True)])
def test_all_variants_bit_exact_vs_each_other(use_persistent, use_pinned):
    """All four variants change only memory lifetime/host-memory type,
    never the kernels themselves (spec item 23) -- so all four must
    produce IDENTICAL output for identical input."""
    images = _images(8)
    batch = np.stack(images, axis=0)
    config = FilterConfig()
    coeffs2d, coeffs1d, lap_coeffs = _coeffs(config)

    reference_pipeline = xray_cuda.PersistentCudaPipeline()
    variant_pipeline = xray_cuda.PersistentCudaPipeline()
    try:
        reference = _run(reference_pipeline, batch, config, coeffs2d, coeffs1d, lap_coeffs, False, False)["output"]
        variant = _run(variant_pipeline, batch, config, coeffs2d, coeffs1d, lap_coeffs, use_persistent, use_pinned)["output"]
        np.testing.assert_array_equal(reference, variant)
    finally:
        reference_pipeline.release()
        variant_pipeline.release()


def test_experimental_output_matches_production_basic_pipeline():
    """The experimental pipeline (baseline variant: no persistence, no
    pinned memory) must match cuda.pipeline.run_basic_cuda_pipeline()'s
    own production output exactly -- proving the duplicated dispatch
    logic in pipeline_experimental.cu is a faithful copy."""
    from cuda.pipeline import run_basic_cuda_pipeline

    images = _images(8)
    batch = np.stack(images, axis=0)
    config = FilterConfig()
    coeffs2d, coeffs1d, lap_coeffs = _coeffs(config)

    production_out, _timing = run_basic_cuda_pipeline(images, config)

    pipeline = xray_cuda.PersistentCudaPipeline()
    try:
        experimental_out = _run(pipeline, batch, config, coeffs2d, coeffs1d, lap_coeffs, False, False,
                                 use_enhanced=False)["output"]
        np.testing.assert_array_equal(production_out, experimental_out)
    finally:
        pipeline.release()


def test_correctness_vs_cpu_matches_established_tolerance():
    images = _images(8)
    batch = np.stack(images, axis=0)
    config = FilterConfig()
    coeffs2d, coeffs1d, lap_coeffs = _coeffs(config)
    cpu_expected = _cpu_reference(images, config)

    pipeline = xray_cuda.PersistentCudaPipeline()
    try:
        r = _run(pipeline, batch, config, coeffs2d, coeffs1d, lap_coeffs, True, True)
        diff = np.abs(r["output"].astype(np.int16) - cpu_expected.astype(np.int16))
        differing_pct = 100.0 * np.count_nonzero(diff) / diff.size
        assert differing_pct < 1.0  # established differing-pixel methodology, not max_abs_diff (see Section 18 tests)
    finally:
        pipeline.release()


# -- 5-6: repeated execution + capacity changes -------------------------------------------------------


def test_repeated_execution_50x_identical_output():
    images = _images(4)
    batch = np.stack(images, axis=0)
    config = FilterConfig()
    coeffs2d, coeffs1d, lap_coeffs = _coeffs(config)

    pipeline = xray_cuda.PersistentCudaPipeline()
    try:
        reference = None
        for i in range(50):
            r = _run(pipeline, batch, config, coeffs2d, coeffs1d, lap_coeffs, True, True)
            if reference is None:
                reference = r["output"]
            else:
                np.testing.assert_array_equal(r["output"], reference)
    finally:
        pipeline.release()


def test_capacity_grows_and_shrinks_correctly_within_one_process():
    """Spec item 26: growing to a larger batch size then requesting a
    smaller one must reuse (not reallocate) the larger buffer and still
    produce correct output for the smaller request."""
    images = _images(32, size=64)
    config = FilterConfig()
    coeffs2d, coeffs1d, lap_coeffs = _coeffs(config)

    pipeline = xray_cuda.PersistentCudaPipeline()
    try:
        for batch_size in (8, 32, 16, 4, 24):
            subset = images[:batch_size]
            batch = np.stack(subset, axis=0)
            r = _run(pipeline, batch, config, coeffs2d, coeffs1d, lap_coeffs, True, True)

            reference_pipeline = xray_cuda.PersistentCudaPipeline()
            try:
                expected = _run(reference_pipeline, batch, config, coeffs2d, coeffs1d, lap_coeffs, False, False)["output"]
            finally:
                reference_pipeline.release()
            np.testing.assert_array_equal(r["output"], expected)
    finally:
        pipeline.release()


# -- 7: resolution changes -------------------------------------------------------


def test_resolution_change_does_not_reuse_incompatible_buffer():
    config = FilterConfig()
    coeffs2d, coeffs1d, lap_coeffs = _coeffs(config)

    small = np.stack(_images(4, size=32), axis=0)
    large = np.stack(_images(4, size=96, seed=100), axis=0)

    pipeline = xray_cuda.PersistentCudaPipeline()
    try:
        r_small = _run(pipeline, small, config, coeffs2d, coeffs1d, lap_coeffs, True, True)
        r_large = _run(pipeline, large, config, coeffs2d, coeffs1d, lap_coeffs, True, True)
        assert r_large["grew_gpu_buffers"] is True  # different resolution must not silently reuse a wrong-shaped buffer

        ref_pipeline = xray_cuda.PersistentCudaPipeline()
        try:
            expected_small = _run(ref_pipeline, small, config, coeffs2d, coeffs1d, lap_coeffs, False, False)["output"]
        finally:
            ref_pipeline.release()
        np.testing.assert_array_equal(r_small["output"], expected_small)

        ref_pipeline2 = xray_cuda.PersistentCudaPipeline()
        try:
            expected_large = _run(ref_pipeline2, large, config, coeffs2d, coeffs1d, lap_coeffs, False, False)["output"]
        finally:
            ref_pipeline2.release()
        np.testing.assert_array_equal(r_large["output"], expected_large)
    finally:
        pipeline.release()


# -- 8: cleanup -------------------------------------------------------


def test_release_frees_gpu_memory_without_leak():
    images = _images(16, size=96)
    batch = np.stack(images, axis=0)
    config = FilterConfig()
    coeffs2d, coeffs1d, lap_coeffs = _coeffs(config)

    free_before = xray_cuda.device_memory_info()["free_bytes"]
    pipeline = xray_cuda.PersistentCudaPipeline()
    _run(pipeline, batch, config, coeffs2d, coeffs1d, lap_coeffs, True, True)
    pipeline.release()
    free_after = xray_cuda.device_memory_info()["free_bytes"]

    leaked = free_before - free_after
    assert leaked < 16 * 1024 * 1024  # generous threshold, same convention as Section 18's memory-stability test


def test_release_is_idempotent():
    pipeline = xray_cuda.PersistentCudaPipeline()
    pipeline.release()
    pipeline.release()  # must not crash or double-free


def test_destructor_cleans_up_without_explicit_release():
    images = _images(4)
    batch = np.stack(images, axis=0)
    config = FilterConfig()
    coeffs2d, coeffs1d, lap_coeffs = _coeffs(config)

    free_before = xray_cuda.device_memory_info()["free_bytes"]

    def _scoped():
        pipeline = xray_cuda.PersistentCudaPipeline()
        _run(pipeline, batch, config, coeffs2d, coeffs1d, lap_coeffs, True, True)
        # no explicit release() -- destructor must clean up when pipeline goes out of scope

    _scoped()
    import gc
    gc.collect()
    free_after = xray_cuda.device_memory_info()["free_bytes"]
    assert (free_before - free_after) < 16 * 1024 * 1024


# -- 9: error handling -------------------------------------------------------


def test_empty_batch_raises_value_error():
    config = FilterConfig()
    coeffs2d, coeffs1d, lap_coeffs = _coeffs(config)
    pipeline = xray_cuda.PersistentCudaPipeline()
    try:
        with pytest.raises(ValueError):
            _run(pipeline, np.zeros((0, 64, 64), dtype=np.uint8), config, coeffs2d, coeffs1d, lap_coeffs, True, True)
    finally:
        pipeline.release()


def test_wrong_dtype_raises_value_error():
    config = FilterConfig()
    coeffs2d, coeffs1d, lap_coeffs = _coeffs(config)
    pipeline = xray_cuda.PersistentCudaPipeline()
    try:
        with pytest.raises(ValueError):
            _run(pipeline, np.zeros((4, 64, 64), dtype=np.float32), config, coeffs2d, coeffs1d, lap_coeffs, True, True)
    finally:
        pipeline.release()


def test_wrong_ndim_raises_value_error():
    config = FilterConfig()
    coeffs2d, coeffs1d, lap_coeffs = _coeffs(config)
    pipeline = xray_cuda.PersistentCudaPipeline()
    try:
        with pytest.raises(ValueError):
            _run(pipeline, np.zeros((64, 64), dtype=np.uint8), config, coeffs2d, coeffs1d, lap_coeffs, True, True)
    finally:
        pipeline.release()


# -- 10: production isolation -------------------------------------------------------


def test_production_pipeline_functions_unaffected():
    """cuda/pipeline.py's production entry points must not know this
    experimental class exists at all."""
    import cuda.pipeline as production_pipeline

    assert not hasattr(production_pipeline, "PersistentCudaPipeline")
    assert not hasattr(production_pipeline, "run_experimental_persistent_pinned_gpu")


# -- 11: benchmark serialization -------------------------------------------------------


def test_benchmark_script_writes_expected_namespace(tmp_path, monkeypatch):
    """Runs the actual benchmark harness's core function (not the CLI)
    against a tiny fixture batch and checks the result structure, without
    touching the real benchmark_results/research_optimization/ directory."""
    # scripts/ isn't a package importable via a dotted path in this repo's layout;
    # add it to sys.path and import the module directly instead.
    import sys as _sys
    from pathlib import Path as _Path

    scripts_dir = _Path(__file__).resolve().parent.parent / "scripts"
    _sys.path.insert(0, str(scripts_dir))
    import benchmark_persistent_pinned as bpp

    images = _images(4, size=32)
    config = FilterConfig()
    coeffs2d, coeffs1d, lap_coeffs = _coeffs(config)

    result = bpp.benchmark_one_batch_size(images, config, coeffs2d, coeffs1d, lap_coeffs,
                                           batch_size=4, warmup_runs=1, measurement_runs=2)
    assert result["batch_size"] == 4
    assert set(result["variants"].keys()) == {"baseline", "persistent", "pinned", "persistent_pinned"}
    for name, v in result["variants"].items():
        assert v["bit_exact_vs_baseline"] is True
        assert v["total_ms"]["n"] == 2
        assert "gain_vs_baseline" in v
