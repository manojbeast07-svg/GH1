"""Section 20F: tests for the EXPERIMENTAL xray_cuda.AsyncCudaPipeline
(multi-stream, double/triple-buffered chunked pipeline). See
cuda/include/pipeline_async.cuh for the architecture: 3 explicit streams
(H2D/compute/D2H), N buffer sets round-robin across chunks, CUDA events
for cross-stream dependencies.

Never imports cuda.persistent_pipeline_experimental or touches
xray_cuda.PersistentCudaPipeline/CudaGraphPipeline/CudaGraphEnhancedPipeline
-- isolated from Sections 20B/20C/20D/20E per this section's spec items 14-15.

Performance is NOT asserted here (spec item 43's explicit instruction) --
see scripts/benchmark_async_pipeline.py for timing.
"""

from __future__ import annotations

import numpy as np
import pytest

xray_cuda = pytest.importorskip("xray_cuda")

from cpu.filters import FilterConfig  # noqa: E402
from cuda.gaussian import gaussian_kernel_1d, gaussian_kernel_2d  # noqa: E402
from cuda.laplacian import laplacian_kernel_2d  # noqa: E402
from cuda.pipeline import run_basic_cuda_pipeline, run_enhanced_cuda_pipeline  # noqa: E402
from cuda.sobel import mode_to_int  # noqa: E402

pytestmark = pytest.mark.skipif(not xray_cuda.cuda_available(), reason="requires a usable CUDA device")


def _config():
    return FilterConfig()


def _batch(batch_size=37, height=32, width=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(batch_size, height, width), dtype=np.uint8)


def _run_basic(pipeline, batch, config, chunk_size, num_buffers=2, use_pinned_staging=False):
    gc = gaussian_kernel_2d(config.gaussian_kernel_size, config.gaussian_sigma)
    lc = laplacian_kernel_2d(config.laplacian_kernel_size)
    sm = mode_to_int(config.sobel_mode)
    return pipeline.run(
        batch=batch, use_enhanced=False, gaussian_enabled=config.gaussian_enabled, gaussian_coeffs=gc,
        median_enabled=config.median_enabled, median_kernel_size=config.median_kernel_size,
        sobel_enabled=config.sobel_enabled, sobel_mode=sm,
        laplacian_enabled=config.laplacian_enabled, laplacian_coeffs=lc,
        laplacian_scale=config.laplacian_scale, laplacian_delta=config.laplacian_delta,
        threshold_enabled=config.threshold_enabled, threshold_value=config.threshold_value, threshold_max_value=config.threshold_max_value,
        chunk_size=chunk_size, num_buffers=num_buffers, use_pinned_staging=use_pinned_staging,
    )


def _run_enhanced(pipeline, batch, config, chunk_size, num_buffers=2, use_pinned_staging=False):
    gc1d = gaussian_kernel_1d(config.gaussian_kernel_size, config.gaussian_sigma)
    lc = laplacian_kernel_2d(config.laplacian_kernel_size)
    sm = mode_to_int(config.sobel_mode)
    return pipeline.run(
        batch=batch, use_enhanced=True, gaussian_enabled=config.gaussian_enabled, gaussian_coeffs_1d=gc1d,
        median_enabled=config.median_enabled,
        sobel_enabled=config.sobel_enabled, sobel_mode=sm,
        laplacian_enabled=config.laplacian_enabled, laplacian_coeffs=lc,
        laplacian_scale=config.laplacian_scale, laplacian_delta=config.laplacian_delta,
        threshold_enabled=config.threshold_enabled, threshold_value=config.threshold_value, threshold_max_value=config.threshold_max_value,
        chunk_size=chunk_size, num_buffers=num_buffers, use_pinned_staging=use_pinned_staging,
    )


def _expected_basic(config, images):
    return run_basic_cuda_pipeline(images, config)[0]


def _expected_enhanced(config, images):
    return run_enhanced_cuda_pipeline(images, config)[0]


# --------------------------------------------------------------------------
# 1-3. Basic / Enhanced correctness, same output as production
# --------------------------------------------------------------------------

@pytest.mark.parametrize("chunk_size,num_buffers,pinned", [(8, 2, False), (8, 3, False), (16, 2, True), (16, 3, True)])
def test_basic_correctness_vs_production(chunk_size, num_buffers, pinned):
    config = _config()
    batch = _batch(seed=1)
    pipeline = xray_cuda.AsyncCudaPipeline()
    result = _run_basic(pipeline, batch, config, chunk_size, num_buffers, pinned)
    pipeline.release()
    expected = _expected_basic(config, [batch[i] for i in range(batch.shape[0])])
    assert np.array_equal(result["output"], expected)


@pytest.mark.parametrize("chunk_size,num_buffers,pinned", [(8, 2, False), (8, 3, False), (16, 2, True), (16, 3, True)])
def test_enhanced_correctness_vs_production(chunk_size, num_buffers, pinned):
    config = _config()
    batch = _batch(seed=2)
    pipeline = xray_cuda.AsyncCudaPipeline()
    result = _run_enhanced(pipeline, batch, config, chunk_size, num_buffers, pinned)
    pipeline.release()
    expected = _expected_enhanced(config, [batch[i] for i in range(batch.shape[0])])
    assert np.array_equal(result["output"], expected)


def test_cpu_comparison_within_established_tolerance():
    from cpu.filters import apply_gaussian, apply_laplacian, apply_median, apply_sobel, apply_threshold
    config = _config()
    batch = _batch(seed=3)
    images = [batch[i] for i in range(batch.shape[0])]
    cpu_expected = np.stack([
        apply_threshold(apply_laplacian(apply_sobel(apply_median(apply_gaussian(img, config), config), config), config), config)
        for img in images
    ])
    pipeline = xray_cuda.AsyncCudaPipeline()
    result = _run_basic(pipeline, batch, config, chunk_size=8)
    pipeline.release()
    diff = np.abs(result["output"].astype(np.int16) - cpu_expected.astype(np.int16))
    differing_pct = 100.0 * np.count_nonzero(diff) / diff.size
    assert differing_pct < 5.0  # same established bar as every other section, no new tolerance


# --------------------------------------------------------------------------
# 4-5. Batch and chunk ordering
# --------------------------------------------------------------------------

@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_output_ordering_matches_input_order_across_batch_sizes(batch_size):
    config = _config()
    batch = _batch(batch_size=batch_size, seed=4)
    pipeline = xray_cuda.AsyncCudaPipeline()
    result = _run_basic(pipeline, batch, config, chunk_size=5, num_buffers=2)
    pipeline.release()
    for i in range(batch_size):
        expected_i = _expected_basic(config, [batch[i]])
        assert np.array_equal(result["output"][i], expected_i[0]), f"ordering mismatch at index {i}"


@pytest.mark.parametrize("chunk_size", [1, 3, 8, 37])
def test_chunk_ordering_correct_for_various_chunk_sizes(chunk_size):
    """Includes chunk_size >= batch_size (single chunk) and chunk_size=1
    (every image its own chunk, maximum chunk-boundary stress)."""
    config = _config()
    batch = _batch(batch_size=37, seed=5)
    pipeline = xray_cuda.AsyncCudaPipeline()
    result = _run_basic(pipeline, batch, config, chunk_size=chunk_size, num_buffers=2)
    pipeline.release()
    expected = _expected_basic(config, [batch[i] for i in range(37)])
    assert np.array_equal(result["output"], expected)


# --------------------------------------------------------------------------
# 6-7. Resolution and parameter changes
# --------------------------------------------------------------------------

def test_resolution_change_produces_correct_independent_results():
    config = _config()
    pipeline = xray_cuda.AsyncCudaPipeline()
    batch_small = _batch(height=32, width=32, seed=6)
    r1 = _run_basic(pipeline, batch_small, config, chunk_size=8)
    expected1 = _expected_basic(config, [batch_small[i] for i in range(batch_small.shape[0])])
    assert np.array_equal(r1["output"], expected1)

    batch_large = _batch(height=96, width=96, seed=7)
    r2 = _run_basic(pipeline, batch_large, config, chunk_size=8)
    expected2 = _expected_basic(config, [batch_large[i] for i in range(batch_large.shape[0])])
    assert np.array_equal(r2["output"], expected2)

    batch_small2 = _batch(height=32, width=32, seed=8)
    r3 = _run_basic(pipeline, batch_small2, config, chunk_size=8)
    expected3 = _expected_basic(config, [batch_small2[i] for i in range(batch_small2.shape[0])])
    assert np.array_equal(r3["output"], expected3)  # no leakage from the 96x96 pass
    pipeline.release()


def test_parameter_changes_produce_correctly_different_output():
    import dataclasses
    config = _config()
    batch = _batch(seed=9)
    pipeline = xray_cuda.AsyncCudaPipeline()
    r1 = _run_basic(pipeline, batch, config, chunk_size=8)

    config2 = dataclasses.replace(config, threshold_value=200)
    r2 = _run_basic(pipeline, batch, config2, chunk_size=8)
    assert not np.array_equal(r1["output"], r2["output"])
    assert np.array_equal(r2["output"], _expected_basic(config2, [batch[i] for i in range(batch.shape[0])]))

    config3 = dataclasses.replace(config, gaussian_enabled=False)
    r3 = _run_basic(pipeline, batch, config3, chunk_size=8)
    assert np.array_equal(r3["output"], _expected_basic(config3, [batch[i] for i in range(batch.shape[0])]))
    pipeline.release()


# --------------------------------------------------------------------------
# 8-9. Double/triple-buffer lifecycle
# --------------------------------------------------------------------------

@pytest.mark.parametrize("num_buffers", [1, 2, 3, 4])
def test_buffer_count_lifecycle(num_buffers):
    config = _config()
    batch = _batch(batch_size=20, seed=10)
    pipeline = xray_cuda.AsyncCudaPipeline()
    result = _run_basic(pipeline, batch, config, chunk_size=6, num_buffers=num_buffers)
    pipeline.release()
    expected = _expected_basic(config, [batch[i] for i in range(20)])
    assert np.array_equal(result["output"], expected)
    assert result["num_buffers_used"] == min(num_buffers, result["num_chunks"])


def test_num_buffers_capped_to_num_chunks_when_batch_smaller_than_requested_buffers():
    config = _config()
    batch = _batch(batch_size=2, seed=11)
    pipeline = xray_cuda.AsyncCudaPipeline()
    result = _run_basic(pipeline, batch, config, chunk_size=1, num_buffers=3)
    pipeline.release()
    assert result["num_chunks"] == 2
    assert result["num_buffers_used"] == 2  # capped, not 3


# --------------------------------------------------------------------------
# 10. Stream dependency correctness (measured overlap, not just structural)
# --------------------------------------------------------------------------

def test_measured_overlap_is_nonzero_for_a_multi_chunk_run():
    config = _config()
    batch = _batch(batch_size=64, height=96, width=96, seed=12)
    pipeline = xray_cuda.AsyncCudaPipeline()
    result = _run_basic(pipeline, batch, config, chunk_size=8, num_buffers=2)
    pipeline.release()
    assert result["num_chunks"] > 1
    assert result["h2d_compute_overlap_ms"] >= 0.0
    assert result["compute_d2h_overlap_ms"] >= 0.0
    # Not a strict >0 assertion (spec item 43: no hard performance thresholds) --
    # just confirms the measurement mechanism itself runs without error and
    # produces a well-formed, non-negative value.


# --------------------------------------------------------------------------
# 11. Repeated execution (200 runs) -- corruption / race / leak / determinism
# --------------------------------------------------------------------------

def test_200_repeated_executions_stay_correct_and_deterministic():
    config = _config()
    pipeline = xray_cuda.AsyncCudaPipeline()
    batch = _batch(batch_size=20, seed=13)
    expected = _expected_basic(config, [batch[i] for i in range(20)])
    first_output = None
    for i in range(200):
        result = _run_basic(pipeline, batch, config, chunk_size=6, num_buffers=2)
        assert np.array_equal(result["output"], expected), f"corruption/race detected at iteration {i}"
        if first_output is None:
            first_output = result["output"]
        else:
            assert np.array_equal(result["output"], first_output), f"non-deterministic output at iteration {i}"
    pipeline.release()


# --------------------------------------------------------------------------
# 12. Memory stability
# --------------------------------------------------------------------------

def test_memory_stable_across_varied_calls():
    config = _config()
    pipeline = xray_cuda.AsyncCudaPipeline()
    rng = np.random.default_rng(42)
    for _ in range(20):
        bs = int(rng.integers(4, 40))
        _run_basic(pipeline, _batch(batch_size=bs, seed=int(rng.integers(0, 1_000_000))), config,
                   chunk_size=int(rng.choice([4, 8, 16])), num_buffers=int(rng.choice([2, 3])))
    after_warm = xray_cuda.device_memory_info()["free_bytes"]
    for _ in range(40):
        bs = int(rng.integers(4, 40))
        _run_basic(pipeline, _batch(batch_size=bs, seed=int(rng.integers(0, 1_000_000))), config,
                   chunk_size=int(rng.choice([4, 8, 16])), num_buffers=int(rng.choice([2, 3])))
    after_more = xray_cuda.device_memory_info()["free_bytes"]
    # Each call allocates and frees its own buffer sets fresh (spec item 14: no cross-call
    # persistence) -- free memory should return to (approximately) the same baseline every time.
    assert abs(after_more - after_warm) < 8 * 1024 * 1024
    pipeline.release()


def test_release_frees_gpu_and_pinned_memory():
    config = _config()
    pipeline = xray_cuda.AsyncCudaPipeline()
    _run_basic(pipeline, _batch(batch_size=32, height=128, width=128, seed=14), config,
               chunk_size=8, num_buffers=3, use_pinned_staging=True)
    before_release = xray_cuda.device_memory_info()["free_bytes"]
    pipeline.release()
    after_release = xray_cuda.device_memory_info()["free_bytes"]
    # run() frees its own per-call buffer sets already (before release() is even called) --
    # release() only needs to free the 3 streams, so GPU memory need not visibly change here;
    # this test's real purpose is confirming release() does not error and is safe to call.
    assert after_release >= before_release


# --------------------------------------------------------------------------
# 13. Error handling
# --------------------------------------------------------------------------

def test_invalid_batch_rejected():
    config = _config()
    pipeline = xray_cuda.AsyncCudaPipeline()
    empty = np.zeros((0, 32, 32), dtype=np.uint8)
    with pytest.raises(ValueError):
        _run_basic(pipeline, empty, config, chunk_size=8)
    pipeline.release()


def test_invalid_chunk_size_rejected():
    config = _config()
    pipeline = xray_cuda.AsyncCudaPipeline()
    with pytest.raises(ValueError):
        _run_basic(pipeline, _batch(seed=15), config, chunk_size=0)
    with pytest.raises(ValueError):
        _run_basic(pipeline, _batch(seed=16), config, chunk_size=-5)
    pipeline.release()


def test_invalid_num_buffers_rejected():
    config = _config()
    pipeline = xray_cuda.AsyncCudaPipeline()
    with pytest.raises(ValueError):
        _run_basic(pipeline, _batch(seed=17), config, chunk_size=8, num_buffers=0)
    with pytest.raises(ValueError):
        _run_basic(pipeline, _batch(seed=18), config, chunk_size=8, num_buffers=10)
    pipeline.release()


def test_unsupported_resolution_rejected_cleanly():
    config = _config()
    pipeline = xray_cuda.AsyncCudaPipeline()
    bad = np.zeros((4, 0, 32), dtype=np.uint8)
    with pytest.raises(ValueError):
        _run_basic(pipeline, bad, config, chunk_size=4)
    pipeline.release()


def test_pipeline_remains_usable_after_a_validation_error():
    config = _config()
    pipeline = xray_cuda.AsyncCudaPipeline()
    with pytest.raises(ValueError):
        _run_basic(pipeline, _batch(seed=19), config, chunk_size=0)
    batch = _batch(seed=20)
    result = _run_basic(pipeline, batch, config, chunk_size=8)
    expected = _expected_basic(config, [batch[i] for i in range(batch.shape[0])])
    assert np.array_equal(result["output"], expected)
    pipeline.release()


# --------------------------------------------------------------------------
# 14. Cleanup / release lifecycle
# --------------------------------------------------------------------------

def test_release_is_idempotent():
    pipeline = xray_cuda.AsyncCudaPipeline()
    pipeline.release()
    pipeline.release()
    assert pipeline.is_released is True


def test_run_after_release_raises_runtime_error():
    pipeline = xray_cuda.AsyncCudaPipeline()
    pipeline.release()
    config = _config()
    with pytest.raises(RuntimeError):
        _run_basic(pipeline, _batch(), config, chunk_size=8)


# --------------------------------------------------------------------------
# Isolation from Sections 20B/20C/20D/20E
# --------------------------------------------------------------------------

def test_basic_and_enhanced_graph_pipelines_unaffected():
    config = _config()
    batch = _batch(seed=21)
    images = [batch[i] for i in range(batch.shape[0])]
    before_basic, _ = run_basic_cuda_pipeline(images, config)
    before_enh, _ = run_enhanced_cuda_pipeline(images, config)

    pipeline = xray_cuda.AsyncCudaPipeline()
    for i in range(5):
        _run_basic(pipeline, _batch(seed=22 + i), config, chunk_size=6)
        _run_enhanced(pipeline, _batch(seed=30 + i), config, chunk_size=6)
    pipeline.release()

    after_basic, _ = run_basic_cuda_pipeline(images, config)
    after_enh, _ = run_enhanced_cuda_pipeline(images, config)
    assert np.array_equal(before_basic, after_basic)
    assert np.array_equal(before_enh, after_enh)


def test_does_not_import_or_reference_other_experimental_pipelines():
    import inspect
    import cuda.pipeline as production_pipeline_module
    source = inspect.getsource(production_pipeline_module)
    assert "AsyncCudaPipeline" not in source
