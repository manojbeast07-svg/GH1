"""Section 20C tests: the experimental Python-level API models
(cuda.persistent_pipeline_experimental) built around Section 20B's
native xray_cuda.PersistentCudaPipeline. Never touches production
cuda/pipeline.py or ui/services.py.

Per spec item 42's 16 enumerated areas. No performance number here is a
hard pytest threshold (spec item 39's convention carried forward).
"""

import gc

import numpy as np
import pytest

xray_cuda = pytest.importorskip(
    "xray_cuda", reason="xray_cuda extension is not built; run cmake --build build first."
)
pytestmark = pytest.mark.skipif(not xray_cuda.cuda_available(), reason="No usable CUDA device detected on this machine.")

from cpu.filters import FilterConfig  # noqa: E402
from cuda.persistent_pipeline_experimental import (  # noqa: E402
    PersistentCudaPipeline,
    PersistentCudaPipelineService,
)
from cuda.pipeline import run_basic_cuda_pipeline, run_enhanced_cuda_pipeline  # noqa: E402
from tests.fixtures import FIXTURES  # noqa: E402


def _images(n=8, size=64, seed=0):
    return [FIXTURES["random_deterministic"](size=size, seed=seed + i) for i in range(n)]


# -- 1-2: creation + first run -------------------------------------------------------


def test_creation_does_not_allocate_gpu_memory_until_run():
    free_before = xray_cuda.device_memory_info()["free_bytes"]
    pipeline = PersistentCudaPipeline()
    free_after_create = xray_cuda.device_memory_info()["free_bytes"]
    assert abs(free_before - free_after_create) < 4 * 1024 * 1024  # construction alone allocates nothing GPU-side
    pipeline.release()


def test_first_run_grows_buffers():
    images = _images(4)
    config = FilterConfig()
    pipeline = PersistentCudaPipeline()
    try:
        out, timing = pipeline.run_basic(images, config)
        assert timing.grew_gpu_buffers is True
        assert out.shape == (4, 64, 64)
    finally:
        pipeline.release()


# -- 3: repeated run -------------------------------------------------------


def test_repeated_run_reuses_buffers_and_is_deterministic():
    images = _images(4)
    config = FilterConfig()
    pipeline = PersistentCudaPipeline()
    try:
        out1, t1 = pipeline.run_basic(images, config)
        out2, t2 = pipeline.run_basic(images, config)
        assert t1.grew_gpu_buffers is True
        assert t2.grew_gpu_buffers is False
        np.testing.assert_array_equal(out1, out2)
    finally:
        pipeline.release()


# -- 4-5: batch growth / reduction -------------------------------------------------------


def test_batch_growth_and_reduction_within_one_pipeline():
    images = _images(32, size=48)
    config = FilterConfig()
    pipeline = PersistentCudaPipeline()
    try:
        grew_flags = {}
        for batch_size in (4, 16, 32, 8, 24, 2):
            out, timing = pipeline.run_basic(images[:batch_size], config)
            grew_flags[batch_size] = timing.grew_gpu_buffers
            expected, _ = run_basic_cuda_pipeline(images[:batch_size], config)
            np.testing.assert_array_equal(out, expected)
        # first two requests grow (4 is first-ever, 16>4 grows); reductions (8, 24, 2 <= 32) reuse
        assert grew_flags[4] is True
        assert grew_flags[16] is True
        assert grew_flags[32] is True
        assert grew_flags[8] is False
        assert grew_flags[24] is False
        assert grew_flags[2] is False
    finally:
        pipeline.release()


# -- 6: resolution change -------------------------------------------------------


def test_resolution_change_reallocates_and_stays_correct():
    small = _images(4, size=32)
    large = _images(4, size=96, seed=100)
    config = FilterConfig()
    pipeline = PersistentCudaPipeline()
    try:
        out_small_1, _ = pipeline.run_basic(small, config)
        out_large, t_large = pipeline.run_basic(large, config)
        assert t_large.grew_gpu_buffers is True
        out_small_2, t_small_2 = pipeline.run_basic(small, config)
        assert t_small_2.grew_gpu_buffers is True  # switching back must not reuse the 96x96 buffer

        expected_small, _ = run_basic_cuda_pipeline(small, config)
        expected_large, _ = run_basic_cuda_pipeline(large, config)
        np.testing.assert_array_equal(out_small_1, expected_small)
        np.testing.assert_array_equal(out_large, expected_large)
        np.testing.assert_array_equal(out_small_2, expected_small)
    finally:
        pipeline.release()


# -- 7: configuration change -------------------------------------------------------


def test_configuration_change_does_not_reuse_stale_output():
    images = _images(4)
    pipeline = PersistentCudaPipeline()
    try:
        config_a = FilterConfig(threshold_value=100)
        config_b = FilterConfig(threshold_value=200)
        out_a, _ = pipeline.run_basic(images, config_a)
        out_b, _ = pipeline.run_basic(images, config_b)
        assert not np.array_equal(out_a, out_b), "different threshold_value must produce different output"

        expected_a, _ = run_basic_cuda_pipeline(images, config_a)
        expected_b, _ = run_basic_cuda_pipeline(images, config_b)
        np.testing.assert_array_equal(out_a, expected_a)
        np.testing.assert_array_equal(out_b, expected_b)
    finally:
        pipeline.release()


def test_gaussian_kernel_size_change_recomputes_coefficients_correctly():
    images = _images(4)
    pipeline = PersistentCudaPipeline()
    try:
        config_k3 = FilterConfig(gaussian_kernel_size=3)
        config_k5 = FilterConfig(gaussian_kernel_size=5)
        out_k3, _ = pipeline.run_basic(images, config_k3)
        out_k5, _ = pipeline.run_basic(images, config_k5)
        expected_k3, _ = run_basic_cuda_pipeline(images, config_k3)
        expected_k5, _ = run_basic_cuda_pipeline(images, config_k5)
        np.testing.assert_array_equal(out_k3, expected_k3)
        np.testing.assert_array_equal(out_k5, expected_k5)
    finally:
        pipeline.release()


# -- 8-9: release + repeated release -------------------------------------------------------


def test_release_frees_memory_and_repeated_release_is_idempotent():
    images = _images(8, size=96)
    config = FilterConfig()
    free_before = xray_cuda.device_memory_info()["free_bytes"]
    pipeline = PersistentCudaPipeline()
    pipeline.run_basic(images, config)
    pipeline.release()
    free_after = xray_cuda.device_memory_info()["free_bytes"]
    assert (free_before - free_after) < 16 * 1024 * 1024

    pipeline.release()  # must not raise
    pipeline.release()
    assert pipeline.is_released is True


def test_run_after_release_raises_runtime_error():
    images = _images(4)
    config = FilterConfig()
    pipeline = PersistentCudaPipeline()
    pipeline.release()
    with pytest.raises(RuntimeError):
        pipeline.run_basic(images, config)


# -- 10: destructor cleanup -------------------------------------------------------


def test_destructor_cleans_up_without_explicit_release():
    images = _images(4)
    config = FilterConfig()
    free_before = xray_cuda.device_memory_info()["free_bytes"]

    def _scoped():
        pipeline = PersistentCudaPipeline()
        pipeline.run_basic(images, config)

    _scoped()
    gc.collect()
    free_after = xray_cuda.device_memory_info()["free_bytes"]
    assert (free_before - free_after) < 16 * 1024 * 1024


def test_context_manager_releases_on_exit_including_on_exception():
    images = _images(4)
    config = FilterConfig()

    with PersistentCudaPipeline() as pipeline:
        pipeline.run_basic(images, config)
    assert pipeline.is_released is True

    with pytest.raises(ValueError):
        with PersistentCudaPipeline() as pipeline2:
            pipeline2.run_basic([], config)  # raises ValueError inside the with-block
    assert pipeline2.is_released is True  # __exit__ must still release even when the body raised


# -- 11: exception safety -------------------------------------------------------


def test_invalid_shape_raises_and_object_remains_usable():
    config = FilterConfig()
    pipeline = PersistentCudaPipeline()
    try:
        with pytest.raises(ValueError):
            pipeline.run_basic([], config)
        # object must remain valid and usable after a caught error, not silently corrupted
        images = _images(4)
        out, _ = pipeline.run_basic(images, config)
        expected, _ = run_basic_cuda_pipeline(images, config)
        np.testing.assert_array_equal(out, expected)
    finally:
        pipeline.release()


def test_mismatched_image_shapes_raises_value_error():
    config = FilterConfig()
    images = _images(3, size=32) + _images(1, size=48, seed=99)
    pipeline = PersistentCudaPipeline()
    try:
        with pytest.raises(ValueError):
            pipeline.run_basic(images, config)
    finally:
        pipeline.release()


# -- 12-14: stateless equivalence, Basic, Enhanced -------------------------------------------------------


def test_basic_pipeline_bit_exact_vs_production():
    images = _images(8)
    config = FilterConfig()
    pipeline = PersistentCudaPipeline()
    try:
        out, _ = pipeline.run_basic(images, config)
        expected, _ = run_basic_cuda_pipeline(images, config)
        np.testing.assert_array_equal(out, expected)
    finally:
        pipeline.release()


def test_enhanced_pipeline_bit_exact_vs_production():
    images = _images(8)
    config = FilterConfig()
    pipeline = PersistentCudaPipeline()
    try:
        out, _ = pipeline.run_enhanced(images, config)
        expected, _ = run_enhanced_cuda_pipeline(images, config)
        np.testing.assert_array_equal(out, expected)
    finally:
        pipeline.release()


def test_basic_and_enhanced_independent_on_same_service():
    """Spec item 12: Basic and Enhanced persistent pipelines must be
    independently usable -- via the service prototype's two separate
    native objects."""
    images = _images(8)
    config = FilterConfig()
    service = PersistentCudaPipelineService()
    try:
        basic_pipeline = service.get_basic_pipeline()
        enhanced_pipeline = service.get_enhanced_pipeline()
        assert basic_pipeline is not enhanced_pipeline

        basic_out, _ = basic_pipeline.run_basic(images, config)
        enhanced_out, _ = enhanced_pipeline.run_enhanced(images, config)
        expected_basic, _ = run_basic_cuda_pipeline(images, config)
        expected_enhanced, _ = run_enhanced_cuda_pipeline(images, config)
        np.testing.assert_array_equal(basic_out, expected_basic)
        np.testing.assert_array_equal(enhanced_out, expected_enhanced)

        # re-running basic after enhanced must not have drifted
        basic_out_2, _ = basic_pipeline.run_basic(images, config)
        np.testing.assert_array_equal(basic_out_2, expected_basic)
    finally:
        service.release_all()


# -- 15: session/service isolation -------------------------------------------------------


def test_two_independent_services_do_not_cross_contaminate():
    """Simulates two independent Streamlit sessions -- spec items 20-21:
    each service instance owns its own native objects; nothing is
    shared or global."""
    images_a = _images(4, seed=0)
    images_b = _images(4, seed=500)
    config = FilterConfig()

    service_a = PersistentCudaPipelineService()
    service_b = PersistentCudaPipelineService()
    try:
        out_a, _ = service_a.get_basic_pipeline().run_basic(images_a, config)
        out_b, _ = service_b.get_basic_pipeline().run_basic(images_b, config)

        expected_a, _ = run_basic_cuda_pipeline(images_a, config)
        expected_b, _ = run_basic_cuda_pipeline(images_b, config)
        np.testing.assert_array_equal(out_a, expected_a)
        np.testing.assert_array_equal(out_b, expected_b)

        service_a.release_all()
        # service_b must be completely unaffected by service_a's release
        out_b_2, _ = service_b.get_basic_pipeline().run_basic(images_b, config)
        np.testing.assert_array_equal(out_b_2, expected_b)
    finally:
        service_a.release_all()
        service_b.release_all()


def test_service_recreates_pipeline_after_release():
    """get_basic_pipeline()/get_enhanced_pipeline() must hand back a
    fresh, usable pipeline if the previous one was released -- a
    service instance itself should never become permanently unusable."""
    images = _images(4)
    config = FilterConfig()
    service = PersistentCudaPipelineService()
    try:
        p1 = service.get_basic_pipeline()
        p1.release()
        p2 = service.get_basic_pipeline()
        assert p2 is not p1
        assert p2.is_released is False
        out, _ = p2.run_basic(images, config)
        expected, _ = run_basic_cuda_pipeline(images, config)
        np.testing.assert_array_equal(out, expected)
    finally:
        service.release_all()


# -- 16: memory stability -------------------------------------------------------


def test_memory_stable_across_150_repeated_calls():
    images = _images(4, size=64)
    config = FilterConfig()
    pipeline = PersistentCudaPipeline()
    try:
        reference = None
        free_before = xray_cuda.device_memory_info()["free_bytes"]
        for i in range(150):
            out, _ = pipeline.run_basic(images, config)
            if reference is None:
                reference = out
            elif i % 30 == 0:
                np.testing.assert_array_equal(out, reference)
        free_after = xray_cuda.device_memory_info()["free_bytes"]
        # steady-state (buffers already grown to final capacity): should not keep growing
        assert (free_before - free_after) < 16 * 1024 * 1024
    finally:
        pipeline.release()
    free_after_release = xray_cuda.device_memory_info()["free_bytes"]
    assert free_after_release >= free_before - (16 * 1024 * 1024)
