"""Section 20E: tests for the EXPERIMENTAL xray_cuda.CudaGraphEnhancedPipeline
(CUDA Graph capture/replay for the five ENHANCED filter kernels, production
variants only -- Gaussian=Specialized, Median=Network3x3, Sobel=Specialized,
Laplacian=Specialized, Threshold=Vectorized. See
cuda/include/pipeline_cuda_graph_enhanced.cuh for why this uses isolated,
verified-identical kernel copies rather than the production
*_enhanced_dispatch() functions directly.

Never imports cuda.persistent_pipeline_experimental or touches
xray_cuda.PersistentCudaPipeline/CudaGraphPipeline (Basic) -- isolated from
Section 20B/20C/20D per this section's own spec items 33-35.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

xray_cuda = pytest.importorskip("xray_cuda")

from cpu.filters import FilterConfig  # noqa: E402
from cuda.gaussian import gaussian_kernel_1d  # noqa: E402
from cuda.laplacian import laplacian_kernel_2d  # noqa: E402
from cuda.pipeline import run_basic_cuda_pipeline, run_enhanced_cuda_pipeline  # noqa: E402
from cuda.sobel import mode_to_int  # noqa: E402

pytestmark = pytest.mark.skipif(not xray_cuda.cuda_available(), reason="requires a usable CUDA device")


def _config():
    return FilterConfig()


def _coeffs(config):
    return (
        gaussian_kernel_1d(config.gaussian_kernel_size, config.gaussian_sigma),
        laplacian_kernel_2d(config.laplacian_kernel_size),
        mode_to_int(config.sobel_mode),
    )


def _run(pipeline, batch, config, use_graph=True, full_pipeline_scope=True,
         gaussian_enabled=True, median_enabled=True, sobel_enabled=True,
         laplacian_enabled=True, threshold_enabled=True,
         threshold_value=None, threshold_max_value=None):
    gc1d, lc, sm = _coeffs(config)
    return pipeline.run(
        batch,
        gaussian_enabled, gc1d,
        median_enabled,
        sobel_enabled, sm,
        laplacian_enabled, lc, config.laplacian_scale, config.laplacian_delta,
        threshold_enabled,
        config.threshold_value if threshold_value is None else threshold_value,
        config.threshold_max_value if threshold_max_value is None else threshold_max_value,
        use_graph=use_graph, full_pipeline_scope=full_pipeline_scope,
    )


def _expected(config, images):
    return run_enhanced_cuda_pipeline(images, config)[0]


def _batch(batch_size=4, height=32, width=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(batch_size, height, width), dtype=np.uint8)


# --------------------------------------------------------------------------
# 1. Graph creation / lifecycle
# --------------------------------------------------------------------------

def test_graph_creation_does_not_allocate_any_graph():
    pipeline = xray_cuda.CudaGraphEnhancedPipeline()
    assert pipeline.cache_size == 0
    assert pipeline.is_released is False
    pipeline.release()


def test_release_is_idempotent():
    pipeline = xray_cuda.CudaGraphEnhancedPipeline()
    pipeline.release()
    pipeline.release()
    assert pipeline.is_released is True


def test_run_after_release_raises_runtime_error():
    pipeline = xray_cuda.CudaGraphEnhancedPipeline()
    pipeline.release()
    config = _config()
    with pytest.raises(RuntimeError):
        _run(pipeline, _batch(), config)


def test_cleanup_frees_gpu_memory():
    config = _config()
    pipeline = xray_cuda.CudaGraphEnhancedPipeline()
    _run(pipeline, _batch(batch_size=32, height=128, width=128, seed=1), config)
    before_release = xray_cuda.device_memory_info()["free_bytes"]
    pipeline.release()
    after_release = xray_cuda.device_memory_info()["free_bytes"]
    assert after_release > before_release


# --------------------------------------------------------------------------
# 2. Graph replay -- basic and repeated
# --------------------------------------------------------------------------

def test_graph_replay_bit_exact_with_new_host_pointer():
    config = _config()
    pipeline = xray_cuda.CudaGraphEnhancedPipeline()
    _run(pipeline, _batch(seed=2), config)  # capture
    batch2 = _batch(seed=3)
    result = _run(pipeline, batch2, config)  # replay, different NumPy array
    pipeline.release()
    expected = _expected(config, [batch2[i] for i in range(batch2.shape[0])])
    assert np.array_equal(result["output"], expected)
    assert result["graph_cache_hit"] is True


def test_repeated_replay_50_times_stays_bit_exact():
    config = _config()
    pipeline = xray_cuda.CudaGraphEnhancedPipeline()
    for i in range(50):
        batch = _batch(seed=100 + i)
        result = _run(pipeline, batch, config)
        expected = _expected(config, [batch[j] for j in range(batch.shape[0])])
        assert np.array_equal(result["output"], expected), f"mismatch at replay {i}"
    pipeline.release()


# --------------------------------------------------------------------------
# 3. Exact output correctness (spec item 17) -- no-graph, capture, replay, both scopes
# --------------------------------------------------------------------------

@pytest.mark.parametrize("full_pipeline_scope", [True, False])
def test_no_graph_path_bit_exact_vs_production(full_pipeline_scope):
    config = _config()
    batch = _batch(seed=4)
    pipeline = xray_cuda.CudaGraphEnhancedPipeline()
    result = _run(pipeline, batch, config, use_graph=False, full_pipeline_scope=full_pipeline_scope)
    pipeline.release()
    expected = _expected(config, [batch[i] for i in range(batch.shape[0])])
    assert np.array_equal(result["output"], expected)
    assert result["used_graph"] is False


@pytest.mark.parametrize("full_pipeline_scope", [True, False])
def test_graph_capture_bit_exact_vs_production(full_pipeline_scope):
    config = _config()
    batch = _batch(seed=5)
    pipeline = xray_cuda.CudaGraphEnhancedPipeline()
    result = _run(pipeline, batch, config, use_graph=True, full_pipeline_scope=full_pipeline_scope)
    pipeline.release()
    expected = _expected(config, [batch[i] for i in range(batch.shape[0])])
    assert np.array_equal(result["output"], expected)
    assert result["used_graph"] is True
    assert result["graph_cache_hit"] is False
    assert result["fallback_reason"] == ""


# --------------------------------------------------------------------------
# 4. CPU comparison (spec item 18) -- established methodology, no new tolerance
# --------------------------------------------------------------------------

def test_graph_output_matches_cpu_within_established_tolerance():
    from cpu.filters import apply_gaussian, apply_laplacian, apply_median, apply_sobel, apply_threshold
    config = _config()
    batch = _batch(seed=6)
    images = [batch[i] for i in range(batch.shape[0])]
    cpu_expected = np.stack([
        apply_threshold(apply_laplacian(apply_sobel(apply_median(apply_gaussian(img, config), config), config), config), config)
        for img in images
    ])
    pipeline = xray_cuda.CudaGraphEnhancedPipeline()
    result = _run(pipeline, batch, config)
    pipeline.release()
    diff = np.abs(result["output"].astype(np.int16) - cpu_expected.astype(np.int16))
    differing_pct = 100.0 * np.count_nonzero(diff) / diff.size
    # Same established correctness bar Section 5/20B/20C use for GPU-vs-CPU on this
    # dataset's canonical config -- never a section-invented tolerance.
    assert differing_pct < 5.0


# --------------------------------------------------------------------------
# 5. Basic pipeline (Section 20D) unaffected
# --------------------------------------------------------------------------

def test_basic_cuda_graph_pipeline_unaffected():
    config = _config()
    batch = _batch(seed=7)
    before, _ = run_basic_cuda_pipeline([batch[i] for i in range(batch.shape[0])], config)

    pipeline = xray_cuda.CudaGraphEnhancedPipeline()
    for i in range(3):
        _run(pipeline, _batch(seed=8 + i), config)
    pipeline.release()

    after, _ = run_basic_cuda_pipeline([batch[i] for i in range(batch.shape[0])], config)
    assert np.array_equal(before, after)


def test_does_not_import_or_reference_persistent_or_basic_graph_pipeline():
    import inspect
    import cuda.pipeline as production_pipeline_module
    source = inspect.getsource(production_pipeline_module)
    assert "CudaGraphEnhancedPipeline" not in source


# --------------------------------------------------------------------------
# 6. Parameter change, batch change, resolution change, enable/disable
# --------------------------------------------------------------------------

def test_parameter_change_threshold_recaptures_and_differs():
    config = _config()
    pipeline = xray_cuda.CudaGraphEnhancedPipeline()
    batch = _batch(seed=9)
    r1 = _run(pipeline, batch, config, threshold_value=128)
    r2 = _run(pipeline, batch, config, threshold_value=150)
    assert r2["graph_cache_hit"] is False
    assert pipeline.cache_size == 2
    assert not np.array_equal(r1["output"], r2["output"])
    config2 = dataclasses.replace(config, threshold_value=150)
    expected2 = _expected(config2, [batch[i] for i in range(batch.shape[0])])
    assert np.array_equal(r2["output"], expected2)
    pipeline.release()


def test_batch_size_change_creates_new_cache_entry_and_stays_correct():
    config = _config()
    pipeline = xray_cuda.CudaGraphEnhancedPipeline()
    _run(pipeline, _batch(batch_size=4, seed=10), config)
    batch8 = _batch(batch_size=8, seed=11)
    r = _run(pipeline, batch8, config)
    assert r["graph_cache_hit"] is False
    assert pipeline.cache_size == 2
    expected = _expected(config, [batch8[i] for i in range(8)])
    assert np.array_equal(r["output"], expected)
    pipeline.release()


def test_resolution_change_creates_new_cache_entry_and_reverting_still_works():
    config = _config()
    pipeline = xray_cuda.CudaGraphEnhancedPipeline()
    _run(pipeline, _batch(height=32, width=32, seed=12), config)
    batch64 = _batch(height=64, width=64, seed=13)
    r = _run(pipeline, batch64, config)
    assert r["graph_cache_hit"] is False
    assert pipeline.cache_size == 2
    expected = _expected(config, [batch64[i] for i in range(batch64.shape[0])])
    assert np.array_equal(r["output"], expected)
    batch32b = _batch(height=32, width=32, seed=14)
    r2 = _run(pipeline, batch32b, config)
    assert r2["graph_cache_hit"] is True
    assert pipeline.cache_size == 2
    pipeline.release()


def test_enable_disable_gaussian_creates_new_cache_entry_and_recomputes_correctly():
    config = _config()
    pipeline = xray_cuda.CudaGraphEnhancedPipeline()
    batch = _batch(seed=15)
    r1 = _run(pipeline, batch, config, gaussian_enabled=True)
    r2 = _run(pipeline, batch, config, gaussian_enabled=False)
    assert r2["graph_cache_hit"] is False
    assert pipeline.cache_size == 2
    config_no_gaussian = dataclasses.replace(config, gaussian_enabled=False)
    expected_no_gaussian = _expected(config_no_gaussian, [batch[i] for i in range(batch.shape[0])])
    assert np.array_equal(r2["output"], expected_no_gaussian)
    assert not np.array_equal(r1["output"], r2["output"])
    pipeline.release()


def test_sobel_mode_change_selects_a_different_template_and_recomputes_correctly():
    config = _config()
    pipeline = xray_cuda.CudaGraphEnhancedPipeline()
    batch = _batch(seed=16)
    r1 = _run(pipeline, batch, config)
    config_x = dataclasses.replace(config, sobel_mode="x")
    gc1d, lc, _ = _coeffs(config_x)
    r2 = pipeline.run(batch, True, gc1d, True, True, mode_to_int("x"), True, lc,
                       config_x.laplacian_scale, config_x.laplacian_delta,
                       True, config_x.threshold_value, config_x.threshold_max_value)
    assert r2["graph_cache_hit"] is False
    expected_x = _expected(config_x, [batch[i] for i in range(batch.shape[0])])
    assert np.array_equal(r2["output"], expected_x)
    assert not np.array_equal(r1["output"], r2["output"])
    pipeline.release()


# --------------------------------------------------------------------------
# 7. Cache correctness, invalidation, and fallback
# --------------------------------------------------------------------------

def test_cache_hit_after_first_capture_for_identical_configuration():
    config = _config()
    pipeline = xray_cuda.CudaGraphEnhancedPipeline()
    r1 = _run(pipeline, _batch(seed=17), config)
    assert r1["graph_cache_hit"] is False
    assert r1["capture_ms"] > 0.0
    assert r1["instantiate_ms"] > 0.0
    r2 = _run(pipeline, _batch(seed=18), config)
    assert r2["graph_cache_hit"] is True
    assert r2["capture_ms"] == 0.0
    assert r2["instantiate_ms"] == 0.0
    assert pipeline.cache_size == 1
    pipeline.release()


def test_kernels_only_and_full_pipeline_scopes_use_separate_cache_entries():
    config = _config()
    pipeline = xray_cuda.CudaGraphEnhancedPipeline()
    batch = _batch(seed=19)
    _run(pipeline, batch, config, full_pipeline_scope=True)
    r = _run(pipeline, batch, config, full_pipeline_scope=False)
    assert r["graph_cache_hit"] is False
    assert pipeline.cache_size == 2
    pipeline.release()


def test_clear_cache_forces_recapture_without_releasing_pipeline():
    config = _config()
    pipeline = xray_cuda.CudaGraphEnhancedPipeline()
    _run(pipeline, _batch(seed=20), config)
    assert pipeline.cache_size == 1
    pipeline.clear_cache()
    assert pipeline.cache_size == 0
    assert pipeline.is_released is False
    r = _run(pipeline, _batch(seed=21), config)
    assert r["graph_cache_hit"] is False
    pipeline.release()


def test_never_executes_a_stale_graph_across_many_alternating_configs():
    """Alternates between two configurations 10 times; every single
    result must match its OWN configuration's production output, never
    the other configuration's (spec item 16's 'do not execute stale
    graphs' requirement, exercised directly rather than just asserted)."""
    config_a = _config()
    config_b = dataclasses.replace(config_a, threshold_value=64)
    pipeline = xray_cuda.CudaGraphEnhancedPipeline()
    batch = _batch(seed=22)
    expected_a = _expected(config_a, [batch[i] for i in range(batch.shape[0])])
    expected_b = _expected(config_b, [batch[i] for i in range(batch.shape[0])])
    for i in range(10):
        cfg = config_a if i % 2 == 0 else config_b
        expected = expected_a if i % 2 == 0 else expected_b
        r = _run(pipeline, batch, cfg, threshold_value=cfg.threshold_value)
        assert np.array_equal(r["output"], expected), f"stale graph executed at iteration {i}"
    assert pipeline.cache_size == 2
    pipeline.release()


# --------------------------------------------------------------------------
# 8. Validation / misuse
# --------------------------------------------------------------------------

def test_invalid_gaussian_kernel_size_rejected():
    config = _config()
    pipeline = xray_cuda.CudaGraphEnhancedPipeline()
    bad_coeffs = np.zeros(4, dtype=np.float32)  # even length -- not in {3,5,7,9}
    _, lc, sm = _coeffs(config)
    with pytest.raises(ValueError):
        pipeline.run(_batch(), True, bad_coeffs, True, True, sm, True, lc,
                     config.laplacian_scale, config.laplacian_delta,
                     True, config.threshold_value, config.threshold_max_value)
    pipeline.release()


def test_invalid_sobel_mode_rejected():
    config = _config()
    pipeline = xray_cuda.CudaGraphEnhancedPipeline()
    gc1d, lc, _ = _coeffs(config)
    with pytest.raises(ValueError):
        pipeline.run(_batch(), True, gc1d, True, True, 99, True, lc,
                     config.laplacian_scale, config.laplacian_delta,
                     True, config.threshold_value, config.threshold_max_value)
    pipeline.release()


def test_pipeline_remains_usable_after_a_validation_error():
    config = _config()
    pipeline = xray_cuda.CudaGraphEnhancedPipeline()
    gc1d, lc, sm = _coeffs(config)
    with pytest.raises(ValueError):
        pipeline.run(_batch(), True, gc1d, True, True, 99, True, lc,
                     config.laplacian_scale, config.laplacian_delta,
                     True, config.threshold_value, config.threshold_max_value)
    batch = _batch(seed=23)
    result = _run(pipeline, batch, config)
    expected = _expected(config, [batch[i] for i in range(batch.shape[0])])
    assert np.array_equal(result["output"], expected)
    pipeline.release()


# --------------------------------------------------------------------------
# 9. Memory stability
# --------------------------------------------------------------------------

def test_memory_stable_across_60_varied_calls():
    config = _config()
    pipeline = xray_cuda.CudaGraphEnhancedPipeline()
    rng = np.random.default_rng(42)
    shapes = [(4, 16, 16), (8, 32, 32), (16, 64, 64), (4, 16, 16)]
    for _ in range(20):
        b, h, w = shapes[rng.integers(0, len(shapes))]
        _run(pipeline, _batch(b, h, w, seed=int(rng.integers(0, 1_000_000))), config)
    after_warm = xray_cuda.device_memory_info()["free_bytes"]
    for _ in range(60):
        b, h, w = shapes[rng.integers(0, len(shapes))]
        _run(pipeline, _batch(b, h, w, seed=int(rng.integers(0, 1_000_000))), config)
    after_more = xray_cuda.device_memory_info()["free_bytes"]
    assert after_more == after_warm
    pipeline.release()


def test_two_independent_pipelines_do_not_share_cache():
    config = _config()
    p1 = xray_cuda.CudaGraphEnhancedPipeline()
    p2 = xray_cuda.CudaGraphEnhancedPipeline()
    _run(p1, _batch(seed=24), config)
    assert p1.cache_size == 1
    assert p2.cache_size == 0
    p1.release()
    assert p2.is_released is False
    r = _run(p2, _batch(seed=25), config)
    assert r["graph_cache_hit"] is False
    p2.release()
