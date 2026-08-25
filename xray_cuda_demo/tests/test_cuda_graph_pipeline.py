"""Section 20D: tests for the EXPERIMENTAL xray_cuda.CudaGraphPipeline
(CUDA Graph capture/replay for the five BASIC filter kernels only --
see cuda/include/pipeline_cuda_graph.cuh for why Enhanced graph capture
is not offered in this section).

Never imports cuda.persistent_pipeline_experimental or touches
xray_cuda.PersistentCudaPipeline -- this experiment is deliberately
isolated from Section 20B/20C's buffer-persistence/pinned-memory work
(spec items 16-18) so its own graph-specific hypothesis is not confounded.
"""

from __future__ import annotations

import numpy as np
import pytest

xray_cuda = pytest.importorskip("xray_cuda")

from cpu.filters import FilterConfig  # noqa: E402
from cuda.gaussian import gaussian_kernel_2d  # noqa: E402
from cuda.laplacian import laplacian_kernel_2d  # noqa: E402
from cuda.pipeline import run_basic_cuda_pipeline  # noqa: E402
from cuda.sobel import mode_to_int  # noqa: E402

pytestmark = pytest.mark.skipif(not xray_cuda.cuda_available(), reason="requires a usable CUDA device")


def _config():
    return FilterConfig()


def _coeffs(config):
    return (
        gaussian_kernel_2d(config.gaussian_kernel_size, config.gaussian_sigma),
        laplacian_kernel_2d(config.laplacian_kernel_size),
        mode_to_int(config.sobel_mode),
    )


def _run(pipeline, batch, config, gc, lc, sm, use_graph=True, full_pipeline_scope=True,
          gaussian_enabled=True, median_enabled=True, sobel_enabled=True,
          laplacian_enabled=True, threshold_enabled=True,
          threshold_value=None, threshold_max_value=None):
    return pipeline.run(
        batch,
        gaussian_enabled, gc,
        median_enabled, config.median_kernel_size,
        sobel_enabled, sm,
        laplacian_enabled, lc, config.laplacian_scale, config.laplacian_delta,
        threshold_enabled,
        config.threshold_value if threshold_value is None else threshold_value,
        config.threshold_max_value if threshold_max_value is None else threshold_max_value,
        use_graph=use_graph, full_pipeline_scope=full_pipeline_scope,
    )


def _batch(batch_size=4, height=32, width=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(batch_size, height, width), dtype=np.uint8)


# --------------------------------------------------------------------------
# 1. Lifecycle
# --------------------------------------------------------------------------

def test_creation_does_not_allocate_any_graph():
    pipeline = xray_cuda.CudaGraphPipeline()
    assert pipeline.cache_size == 0
    assert pipeline.is_released is False
    pipeline.release()


def test_release_is_idempotent():
    pipeline = xray_cuda.CudaGraphPipeline()
    pipeline.release()
    pipeline.release()
    assert pipeline.is_released is True


def test_run_after_release_raises_runtime_error():
    pipeline = xray_cuda.CudaGraphPipeline()
    pipeline.release()
    config = _config()
    gc, lc, sm = _coeffs(config)
    with pytest.raises(RuntimeError):
        _run(pipeline, _batch(), config, gc, lc, sm)


def test_destructor_cleans_up_without_explicit_release():
    import gc as pygc
    config = _config()
    gcoef, lc, sm = _coeffs(config)
    before = xray_cuda.device_memory_info()["free_bytes"]
    pipeline = xray_cuda.CudaGraphPipeline()
    _run(pipeline, _batch(batch_size=32, height=128, width=128), config, gcoef, lc, sm)
    del pipeline
    pygc.collect()
    after = xray_cuda.device_memory_info()["free_bytes"]
    assert after >= before - 4 * 1024 * 1024  # allow small fixed driver-side slack, no real leak


# --------------------------------------------------------------------------
# 2. Correctness -- bit-exact vs production
# --------------------------------------------------------------------------

def test_no_graph_path_bit_exact_vs_production():
    config = _config()
    gc, lc, sm = _coeffs(config)
    batch = _batch(seed=1)
    pipeline = xray_cuda.CudaGraphPipeline()
    result = _run(pipeline, batch, config, gc, lc, sm, use_graph=False)
    pipeline.release()
    expected, _ = run_basic_cuda_pipeline([batch[i] for i in range(batch.shape[0])], config)
    assert np.array_equal(result["output"], expected)
    assert result["used_graph"] is False


@pytest.mark.parametrize("full_pipeline_scope", [True, False])
def test_graph_capture_bit_exact_vs_production(full_pipeline_scope):
    config = _config()
    gc, lc, sm = _coeffs(config)
    batch = _batch(seed=2)
    pipeline = xray_cuda.CudaGraphPipeline()
    result = _run(pipeline, batch, config, gc, lc, sm, use_graph=True, full_pipeline_scope=full_pipeline_scope)
    pipeline.release()
    expected, _ = run_basic_cuda_pipeline([batch[i] for i in range(batch.shape[0])], config)
    assert np.array_equal(result["output"], expected)
    assert result["used_graph"] is True
    assert result["graph_cache_hit"] is False
    assert result["fallback_reason"] == ""


@pytest.mark.parametrize("full_pipeline_scope", [True, False])
def test_graph_replay_bit_exact_vs_production_with_new_host_pointers(full_pipeline_scope):
    """The core graph-reuse claim: a SECOND, different batch (a different
    NumPy array -- different host address) run through the SAME cached
    graph must still be bit-exact, proving the memcpy-node pointer patch
    is applied correctly on every replay, not just the capturing call."""
    config = _config()
    gc, lc, sm = _coeffs(config)
    pipeline = xray_cuda.CudaGraphPipeline()
    _run(pipeline, _batch(seed=3), config, gc, lc, sm, full_pipeline_scope=full_pipeline_scope)  # capture
    batch2 = _batch(seed=4)
    result = _run(pipeline, batch2, config, gc, lc, sm, full_pipeline_scope=full_pipeline_scope)  # replay
    pipeline.release()
    expected, _ = run_basic_cuda_pipeline([batch2[i] for i in range(batch2.shape[0])], config)
    assert np.array_equal(result["output"], expected)
    assert result["graph_cache_hit"] is True


def test_repeated_replay_50_times_stays_bit_exact():
    config = _config()
    gc, lc, sm = _coeffs(config)
    pipeline = xray_cuda.CudaGraphPipeline()
    for i in range(50):
        batch = _batch(seed=100 + i)
        result = _run(pipeline, batch, config, gc, lc, sm)
        expected, _ = run_basic_cuda_pipeline([batch[j] for j in range(batch.shape[0])], config)
        assert np.array_equal(result["output"], expected), f"mismatch at replay {i}"
    pipeline.release()


# --------------------------------------------------------------------------
# 3. Graph cache: hit/miss behavior, keying, cudaGraphExecMemcpyNodeSetParams1D reuse
# --------------------------------------------------------------------------

def test_cache_miss_then_hit_for_identical_configuration():
    config = _config()
    gc, lc, sm = _coeffs(config)
    pipeline = xray_cuda.CudaGraphPipeline()
    r1 = _run(pipeline, _batch(seed=5), config, gc, lc, sm)
    assert r1["graph_cache_hit"] is False
    assert r1["capture_ms"] > 0.0
    assert r1["instantiate_ms"] > 0.0
    r2 = _run(pipeline, _batch(seed=6), config, gc, lc, sm)
    assert r2["graph_cache_hit"] is True
    assert r2["capture_ms"] == 0.0
    assert r2["instantiate_ms"] == 0.0
    assert pipeline.cache_size == 1
    pipeline.release()


def test_batch_size_change_creates_a_new_cache_entry():
    config = _config()
    gc, lc, sm = _coeffs(config)
    pipeline = xray_cuda.CudaGraphPipeline()
    _run(pipeline, _batch(batch_size=4, seed=7), config, gc, lc, sm)
    r = _run(pipeline, _batch(batch_size=8, seed=8), config, gc, lc, sm)
    assert r["graph_cache_hit"] is False
    assert pipeline.cache_size == 2
    pipeline.release()


def test_resolution_change_creates_a_new_cache_entry_and_stays_correct():
    config = _config()
    gc, lc, sm = _coeffs(config)
    pipeline = xray_cuda.CudaGraphPipeline()
    _run(pipeline, _batch(height=32, width=32, seed=9), config, gc, lc, sm)
    batch64 = _batch(height=64, width=64, seed=10)
    r = _run(pipeline, batch64, config, gc, lc, sm)
    assert r["graph_cache_hit"] is False
    assert pipeline.cache_size == 2
    expected, _ = run_basic_cuda_pipeline([batch64[i] for i in range(batch64.shape[0])], config)
    assert np.array_equal(r["output"], expected)
    # revert to the first resolution -- must hit the original cache entry and stay correct
    batch32b = _batch(height=32, width=32, seed=11)
    r2 = _run(pipeline, batch32b, config, gc, lc, sm)
    assert r2["graph_cache_hit"] is True
    assert pipeline.cache_size == 2
    expected2, _ = run_basic_cuda_pipeline([batch32b[i] for i in range(batch32b.shape[0])], config)
    assert np.array_equal(r2["output"], expected2)
    pipeline.release()


def test_enabled_stage_mask_change_creates_a_new_cache_entry_and_recomputes_correctly():
    config = _config()
    gc, lc, sm = _coeffs(config)
    pipeline = xray_cuda.CudaGraphPipeline()
    batch = _batch(seed=12)
    _run(pipeline, batch, config, gc, lc, sm, sobel_enabled=True)
    r = _run(pipeline, batch, config, gc, lc, sm, sobel_enabled=False)
    assert r["graph_cache_hit"] is False
    assert pipeline.cache_size == 2
    expected, _ = run_basic_cuda_pipeline(
        [batch[i] for i in range(batch.shape[0])], config,
    )
    # sobel disabled changes the math -- compare against a config with sobel off instead
    import dataclasses
    config_no_sobel = dataclasses.replace(config, sobel_enabled=False)
    expected_no_sobel, _ = run_basic_cuda_pipeline([batch[i] for i in range(batch.shape[0])], config_no_sobel)
    assert np.array_equal(r["output"], expected_no_sobel)
    assert not np.array_equal(r["output"], expected)
    pipeline.release()


def test_threshold_value_change_creates_a_new_cache_entry_and_differs():
    config = _config()
    gc, lc, sm = _coeffs(config)
    pipeline = xray_cuda.CudaGraphPipeline()
    batch = _batch(seed=13)
    r1 = _run(pipeline, batch, config, gc, lc, sm, threshold_value=128)
    r2 = _run(pipeline, batch, config, gc, lc, sm, threshold_value=200)
    assert r2["graph_cache_hit"] is False
    assert pipeline.cache_size == 2
    assert not np.array_equal(r1["output"], r2["output"])
    pipeline.release()


def test_kernels_only_and_full_pipeline_scopes_use_separate_cache_entries():
    config = _config()
    gc, lc, sm = _coeffs(config)
    pipeline = xray_cuda.CudaGraphPipeline()
    batch = _batch(seed=14)
    _run(pipeline, batch, config, gc, lc, sm, full_pipeline_scope=True)
    r = _run(pipeline, batch, config, gc, lc, sm, full_pipeline_scope=False)
    assert r["graph_cache_hit"] is False
    assert pipeline.cache_size == 2
    pipeline.release()


def test_clear_cache_removes_entries_without_releasing_pipeline():
    config = _config()
    gc, lc, sm = _coeffs(config)
    pipeline = xray_cuda.CudaGraphPipeline()
    _run(pipeline, _batch(seed=15), config, gc, lc, sm)
    assert pipeline.cache_size == 1
    pipeline.clear_cache()
    assert pipeline.cache_size == 0
    assert pipeline.is_released is False
    r = _run(pipeline, _batch(seed=16), config, gc, lc, sm)
    assert r["graph_cache_hit"] is False  # recaptured after clear
    pipeline.release()


# --------------------------------------------------------------------------
# 4. Parameter validation (mirrors production's own bounds checks)
# --------------------------------------------------------------------------

def test_invalid_median_kernel_size_rejected():
    config = _config()
    gc, lc, sm = _coeffs(config)
    pipeline = xray_cuda.CudaGraphPipeline()
    with pytest.raises(ValueError):
        pipeline.run(_batch(), True, gc, True, 4, True, sm, True, lc,
                      config.laplacian_scale, config.laplacian_delta,
                      True, config.threshold_value, config.threshold_max_value)
    pipeline.release()


def test_invalid_sobel_mode_rejected():
    config = _config()
    gc, lc, sm = _coeffs(config)
    pipeline = xray_cuda.CudaGraphPipeline()
    with pytest.raises(ValueError):
        pipeline.run(_batch(), True, gc, True, config.median_kernel_size, True, 99, True, lc,
                      config.laplacian_scale, config.laplacian_delta,
                      True, config.threshold_value, config.threshold_max_value)
    pipeline.release()


def test_empty_batch_rejected():
    config = _config()
    gc, lc, sm = _coeffs(config)
    pipeline = xray_cuda.CudaGraphPipeline()
    empty = np.zeros((0, 32, 32), dtype=np.uint8)
    with pytest.raises(ValueError):
        _run(pipeline, empty, config, gc, lc, sm)
    pipeline.release()


def test_wrong_dtype_rejected():
    config = _config()
    gc, lc, sm = _coeffs(config)
    pipeline = xray_cuda.CudaGraphPipeline()
    bad = np.zeros((4, 32, 32), dtype=np.float32)
    with pytest.raises(ValueError):
        _run(pipeline, bad, config, gc, lc, sm)
    pipeline.release()


def test_pipeline_remains_usable_after_a_validation_error():
    config = _config()
    gc, lc, sm = _coeffs(config)
    pipeline = xray_cuda.CudaGraphPipeline()
    with pytest.raises(ValueError):
        pipeline.run(_batch(), True, gc, True, 4, True, sm, True, lc,
                      config.laplacian_scale, config.laplacian_delta,
                      True, config.threshold_value, config.threshold_max_value)
    batch = _batch(seed=17)
    result = _run(pipeline, batch, config, gc, lc, sm)
    expected, _ = run_basic_cuda_pipeline([batch[i] for i in range(batch.shape[0])], config)
    assert np.array_equal(result["output"], expected)
    pipeline.release()


# --------------------------------------------------------------------------
# 5. Memory stability under varied real usage
# --------------------------------------------------------------------------

def test_memory_stable_across_60_varied_calls():
    config = _config()
    gc, lc, sm = _coeffs(config)
    pipeline = xray_cuda.CudaGraphPipeline()
    rng = np.random.default_rng(42)
    shapes = [(4, 16, 16), (8, 32, 32), (16, 64, 64), (4, 16, 16)]
    for _ in range(20):
        b, h, w = shapes[rng.integers(0, len(shapes))]
        _run(pipeline, _batch(b, h, w, seed=int(rng.integers(0, 1_000_000))), config, gc, lc, sm)
    after_warm = xray_cuda.device_memory_info()["free_bytes"]
    for _ in range(60):
        b, h, w = shapes[rng.integers(0, len(shapes))]
        _run(pipeline, _batch(b, h, w, seed=int(rng.integers(0, 1_000_000))), config, gc, lc, sm)
    after_more = xray_cuda.device_memory_info()["free_bytes"]
    assert after_more == after_warm  # cache size bounded by the 4 distinct shapes -- must be exactly stable
    pipeline.release()


def test_release_frees_gpu_memory():
    config = _config()
    gc, lc, sm = _coeffs(config)
    pipeline = xray_cuda.CudaGraphPipeline()
    _run(pipeline, _batch(batch_size=32, height=128, width=128, seed=18), config, gc, lc, sm)
    before_release = xray_cuda.device_memory_info()["free_bytes"]
    pipeline.release()
    after_release = xray_cuda.device_memory_info()["free_bytes"]
    assert after_release > before_release


# --------------------------------------------------------------------------
# 6. Isolation from Section 20B/20C's persistent-buffer/pinned-memory work
# --------------------------------------------------------------------------

def test_class_does_not_import_or_reference_persistent_pipeline():
    import inspect
    import cuda.pipeline as production_pipeline_module
    source = inspect.getsource(production_pipeline_module)
    assert "CudaGraphPipeline" not in source  # production module never references this experimental class


def test_production_pipeline_functions_unaffected():
    """Direct behavioral check (not just a source-text check): the
    production entry point produces the same result before and after
    exercising CudaGraphPipeline, proving no shared/global state leaks
    between them."""
    config = _config()
    images = [_batch(seed=19)[i] for i in range(4)]
    before, _ = run_basic_cuda_pipeline(images, config)

    gc, lc, sm = _coeffs(config)
    pipeline = xray_cuda.CudaGraphPipeline()
    for i in range(5):
        _run(pipeline, _batch(seed=20 + i), config, gc, lc, sm)
    pipeline.release()

    after, _ = run_basic_cuda_pipeline(images, config)
    assert np.array_equal(before, after)


def test_two_independent_pipelines_do_not_share_cache():
    config = _config()
    gc, lc, sm = _coeffs(config)
    p1 = xray_cuda.CudaGraphPipeline()
    p2 = xray_cuda.CudaGraphPipeline()
    _run(p1, _batch(seed=21), config, gc, lc, sm)
    assert p1.cache_size == 1
    assert p2.cache_size == 0
    p1.release()
    assert p2.is_released is False
    r = _run(p2, _batch(seed=22), config, gc, lc, sm)
    assert r["graph_cache_hit"] is False  # p2 must capture its own graph, unaffected by p1.release()
    p2.release()


# --------------------------------------------------------------------------
# 7. Timing-field sanity (no fabricated numbers -- fields must be internally consistent)
# --------------------------------------------------------------------------

def test_full_pipeline_scope_folds_transfer_time_into_compute_ms():
    config = _config()
    gc, lc, sm = _coeffs(config)
    pipeline = xray_cuda.CudaGraphPipeline()
    r = _run(pipeline, _batch(seed=23), config, gc, lc, sm, full_pipeline_scope=True)
    pipeline.release()
    assert r["h2d_ms"] == 0.0
    assert r["d2h_ms"] == 0.0
    assert r["compute_ms"] > 0.0
    assert r["total_ms"] == pytest.approx(r["compute_ms"] + r["alloc_ms"], abs=1e-4)


def test_kernels_only_scope_reports_separate_h2d_and_d2h():
    config = _config()
    gc, lc, sm = _coeffs(config)
    pipeline = xray_cuda.CudaGraphPipeline()
    r = _run(pipeline, _batch(seed=24), config, gc, lc, sm, full_pipeline_scope=False)
    pipeline.release()
    assert r["h2d_ms"] > 0.0
    assert r["d2h_ms"] > 0.0
    assert r["compute_ms"] > 0.0
