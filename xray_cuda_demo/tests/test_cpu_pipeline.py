"""Section 3 tests: pipeline ordering, enable/disable, intermediate
outputs, mixed-resolution batch handling, and determinism."""

import cv2
import numpy as np
import pytest

from cpu.filters import FilterConfig, apply_gaussian, apply_sobel
from cpu.pipeline import run_cpu_image, run_cpu_pipeline, run_cpu_selection
from pipeline.dataset import DatasetManager
from tests.conftest import get_real_dataset_path
from tests.fixtures import FIXTURES


def _write_image(path, image):
    cv2.imwrite(str(path), image)


# -- ordering / enable-disable -------------------------------------------------------


def test_all_filters_enabled_runs_full_chain():
    image = FIXTURES["random_deterministic"]()
    config = FilterConfig()  # all enabled by default
    result, timing = run_cpu_pipeline(image, config)

    assert result.gaussian_output is not None
    assert result.median_output is not None
    assert result.sobel_output is not None
    assert result.laplacian_output is not None
    assert result.threshold_output is not None
    assert result.final_output is result.threshold_output

    for name in ("gaussian", "median", "sobel", "laplacian", "threshold"):
        assert getattr(timing, f"{name}_ms") is not None
        assert getattr(timing, f"{name}_ms") >= 0.0
    assert timing.total_ms >= 0.0


def test_all_filters_disabled_returns_original_unchanged():
    image = FIXTURES["random_deterministic"]()
    config = FilterConfig(
        gaussian_enabled=False,
        median_enabled=False,
        sobel_enabled=False,
        laplacian_enabled=False,
        threshold_enabled=False,
    )
    result, timing = run_cpu_pipeline(image, config)

    assert result.gaussian_output is None
    assert result.median_output is None
    assert result.sobel_output is None
    assert result.laplacian_output is None
    assert result.threshold_output is None
    np.testing.assert_array_equal(result.final_output, image)
    assert result.final_output is image  # not even copied

    assert timing.total_ms == 0.0
    for name in ("gaussian", "median", "sobel", "laplacian", "threshold"):
        assert getattr(timing, f"{name}_ms") is None


def test_only_gaussian_enabled():
    image = FIXTURES["sharp_edge_square"]()
    config = FilterConfig(
        gaussian_enabled=True, median_enabled=False, sobel_enabled=False,
        laplacian_enabled=False, threshold_enabled=False,
    )
    result, timing = run_cpu_pipeline(image, config)
    expected = apply_gaussian(image, config)
    np.testing.assert_array_equal(result.final_output, expected)
    assert result.final_output is result.gaussian_output
    assert timing.gaussian_ms is not None
    assert timing.median_ms is None


def test_only_sobel_enabled_receives_original_not_gaussian_output():
    """With Gaussian disabled, Sobel must run on the *original* image,
    not on a would-have-been Gaussian output -- stages are skipped, not
    reordered around a phantom previous stage."""
    image = FIXTURES["sharp_edge_square"]()
    config = FilterConfig(
        gaussian_enabled=False, median_enabled=False, sobel_enabled=True,
        laplacian_enabled=False, threshold_enabled=False,
    )
    result, _ = run_cpu_pipeline(image, config)
    expected = apply_sobel(image, config)
    np.testing.assert_array_equal(result.sobel_output, expected)


def test_gaussian_plus_sobel_chains_correctly():
    image = FIXTURES["random_deterministic"]()
    config = FilterConfig(
        gaussian_enabled=True, median_enabled=False, sobel_enabled=True,
        laplacian_enabled=False, threshold_enabled=False,
    )
    result, _ = run_cpu_pipeline(image, config)

    expected_gaussian = apply_gaussian(image, config)
    expected_sobel = apply_sobel(expected_gaussian, config)
    np.testing.assert_array_equal(result.gaussian_output, expected_gaussian)
    np.testing.assert_array_equal(result.sobel_output, expected_sobel)
    assert result.final_output is result.sobel_output


def test_threshold_only_operates_on_original():
    image = FIXTURES["random_deterministic"]()
    config = FilterConfig(
        gaussian_enabled=False, median_enabled=False, sobel_enabled=False,
        laplacian_enabled=False, threshold_enabled=True,
    )
    result, _ = run_cpu_pipeline(image, config)
    assert set(np.unique(result.threshold_output)).issubset({0, config.threshold_max_value})


# -- input contract -------------------------------------------------------


def test_run_cpu_pipeline_rejects_wrong_dtype():
    with pytest.raises(ValueError):
        run_cpu_pipeline(np.zeros((8, 8), dtype=np.float32), FilterConfig())


def test_run_cpu_pipeline_rejects_wrong_dtype_even_with_all_disabled():
    """Input contract must be enforced even when no filter would touch
    the pixels, so a malformed image never silently passes through."""
    config = FilterConfig(
        gaussian_enabled=False, median_enabled=False, sobel_enabled=False,
        laplacian_enabled=False, threshold_enabled=False,
    )
    with pytest.raises(ValueError):
        run_cpu_pipeline(np.zeros((8, 8), dtype=np.int32), config)


# -- determinism -------------------------------------------------------


def test_pipeline_is_deterministic():
    image = FIXTURES["random_deterministic"]()
    config = FilterConfig()
    result_a, timing_a = run_cpu_pipeline(image, config)
    result_b, timing_b = run_cpu_pipeline(image, config)

    np.testing.assert_array_equal(result_a.final_output, result_b.final_output)
    np.testing.assert_array_equal(result_a.gaussian_output, result_b.gaussian_output)
    np.testing.assert_array_equal(result_a.threshold_output, result_b.threshold_output)
    # Timings are independent measurements and may differ in value, but
    # the same set of stages must have run in both cases.
    assert timing_a.as_dict().keys() == timing_b.as_dict().keys()
    assert (timing_a.gaussian_ms is None) == (timing_b.gaussian_ms is None)


# -- single-image / disk loading -------------------------------------------------------


def test_run_cpu_image_loads_and_processes(tmp_path):
    image = FIXTURES["random_deterministic"]()
    path = tmp_path / "test.png"
    _write_image(path, image)

    result, timing = run_cpu_image(path, FilterConfig())
    np.testing.assert_array_equal(result.original, image)
    assert result.final_output is not None
    assert timing.total_ms >= 0.0


# -- mixed-resolution selection handling -------------------------------------------------------


def test_run_cpu_selection_handles_mixed_resolutions_without_crashing(tmp_path):
    _write_image(tmp_path / "small.png", FIXTURES["random_deterministic"](size=16, seed=1))
    _write_image(tmp_path / "medium.png", FIXTURES["random_deterministic"](size=32, seed=2))
    _write_image(tmp_path / "large.png", FIXTURES["random_deterministic"](size=48, seed=3))

    dm = DatasetManager(tmp_path)
    dm.scan()
    selection = dm.random_batch(batch_size=3, seed=42)

    seen_shapes = set()
    for item, result, timing in run_cpu_selection(selection, FilterConfig()):
        assert result.original.shape == result.final_output.shape  # native resolution preserved
        seen_shapes.add(result.original.shape)
        assert timing.total_ms >= 0.0

    assert seen_shapes == {(16, 16), (32, 32), (48, 48)}  # no resize happened


def test_run_cpu_selection_is_a_lazy_generator(tmp_path):
    for i in range(5):
        _write_image(tmp_path / f"img_{i}.png", FIXTURES["random_deterministic"](size=8, seed=i))
    dm = DatasetManager(tmp_path)
    dm.scan()
    selection = dm.random_batch(batch_size=5, seed=1)

    gen = run_cpu_selection(selection, FilterConfig())
    import inspect

    assert inspect.isgenerator(gen)
    first = next(gen)
    assert len(first) == 3  # (item, result, timing)


# -- real dataset compatibility (skipped if the dataset isn't present) -------------------------------------------------------


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_real_xray_images_process_without_crashing():
    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=5, seed=42)

    for item, result, timing in run_cpu_selection(selection, FilterConfig()):
        assert result.final_output.dtype == np.uint8
        assert result.final_output.shape == result.original.shape
        assert timing.total_ms >= 0.0
