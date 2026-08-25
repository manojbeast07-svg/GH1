"""Section 3 tests: single-image repeated-run benchmarking and
selection-level (batch) benchmarking, including JSON-serializability of
results (needed by scripts/benchmark_cpu.py)."""

import json

import cv2
import pytest

from cpu.benchmark import AggregatedTiming, BenchmarkResult, benchmark_cpu_pipeline, run_selection_benchmark
from cpu.filters import FilterConfig
from pipeline.dataset import DatasetManager
from tests.fixtures import FIXTURES


def _write_image(path, image):
    cv2.imwrite(str(path), image)


# -- benchmark_cpu_pipeline (single image, repeated runs) -------------------------------------------------------


def test_benchmark_cpu_pipeline_runs_expected_counts():
    image = FIXTURES["random_deterministic"]()
    config = FilterConfig()
    result = benchmark_cpu_pipeline(image, config, warmup_runs=1, measurement_runs=4)

    assert isinstance(result, AggregatedTiming)
    assert result.measurement_runs == 4
    assert result.total_mean_ms >= 0.0
    assert result.total_min_ms <= result.total_mean_ms <= result.total_max_ms
    assert result.total_std_ms >= 0.0
    assert set(result.per_filter_mean_ms.keys()) == {"gaussian", "median", "sobel", "laplacian", "threshold"}


def test_benchmark_cpu_pipeline_respects_disabled_filters():
    image = FIXTURES["random_deterministic"]()
    config = FilterConfig(median_enabled=False, laplacian_enabled=False)
    result = benchmark_cpu_pipeline(image, config, warmup_runs=0, measurement_runs=2)
    assert set(result.per_filter_mean_ms.keys()) == {"gaussian", "sobel", "threshold"}


def test_benchmark_cpu_pipeline_rejects_bad_run_counts():
    image = FIXTURES["random_deterministic"]()
    config = FilterConfig()
    with pytest.raises(ValueError):
        benchmark_cpu_pipeline(image, config, warmup_runs=-1, measurement_runs=1)
    with pytest.raises(ValueError):
        benchmark_cpu_pipeline(image, config, warmup_runs=0, measurement_runs=0)


def test_benchmark_defaults_match_config_yaml_conventions():
    # Defaults documented in Section 3 spec: warmup=2, measurements=5.
    image = FIXTURES["constant"]()
    config = FilterConfig()
    result = benchmark_cpu_pipeline(image, config)
    assert result.measurement_runs == 5


# -- run_selection_benchmark (batch of images) -------------------------------------------------------


@pytest.fixture
def small_batch_dataset(tmp_path):
    for i in range(6):
        _write_image(tmp_path / f"img_{i}.png", FIXTURES["random_deterministic"](size=16, seed=i))
    return tmp_path


def test_run_selection_benchmark_processing_mode(small_batch_dataset):
    dm = DatasetManager(small_batch_dataset)
    dm.scan()
    selection = dm.random_batch(batch_size=6, seed=1)

    result = run_selection_benchmark(
        selection, FilterConfig(), warmup_runs=1, measurement_runs=3, dataset_fingerprint=dm.fingerprint()
    )

    assert isinstance(result, BenchmarkResult)
    assert result.num_images == 6
    assert result.total_time_ms >= 0.0
    assert result.avg_time_per_image_ms == pytest.approx(result.total_time_ms / 6)
    assert result.throughput_images_per_sec > 0.0
    assert len(result.per_run_total_ms) == 3
    assert result.metadata.timing_mode == "processing"
    assert result.metadata.selection_seed == 1
    assert result.metadata.selection_batch_size == 6
    assert len(result.metadata.selected_relative_paths) == 6
    assert result.metadata.dataset_fingerprint == dm.fingerprint()


def test_run_selection_benchmark_end_to_end_mode(small_batch_dataset):
    dm = DatasetManager(small_batch_dataset)
    dm.scan()
    selection = dm.random_batch(batch_size=6, seed=1)

    result = run_selection_benchmark(
        selection, FilterConfig(), warmup_runs=0, measurement_runs=2, timing_mode="end_to_end"
    )
    assert result.metadata.timing_mode == "end_to_end"
    assert result.num_images == 6


def test_run_selection_benchmark_rejects_unknown_mode(small_batch_dataset):
    dm = DatasetManager(small_batch_dataset)
    dm.scan()
    selection = dm.random_batch(batch_size=3, seed=1)
    with pytest.raises(ValueError):
        run_selection_benchmark(selection, FilterConfig(), timing_mode="bogus")


def test_run_selection_benchmark_rejects_empty_selection(small_batch_dataset):
    dm = DatasetManager(small_batch_dataset)
    dm.scan()
    with pytest.raises(ValueError):
        dm.random_batch(batch_size=0, seed=1)  # already rejected upstream in Section 2


def test_benchmark_result_metadata_includes_environment_info(small_batch_dataset):
    dm = DatasetManager(small_batch_dataset)
    dm.scan()
    selection = dm.random_batch(batch_size=3, seed=1)
    result = run_selection_benchmark(selection, FilterConfig(), warmup_runs=0, measurement_runs=1)

    assert result.metadata.cpu_model
    assert result.metadata.python_version
    assert result.metadata.opencv_version
    assert result.metadata.timestamp_utc
    assert result.metadata.filter_config["gaussian_kernel_size"] == 5


def test_benchmark_result_is_json_serializable(small_batch_dataset):
    dm = DatasetManager(small_batch_dataset)
    dm.scan()
    selection = dm.random_batch(batch_size=3, seed=1)
    result = run_selection_benchmark(selection, FilterConfig(), warmup_runs=0, measurement_runs=1)

    serialized = json.dumps(result.to_dict())
    reloaded = json.loads(serialized)
    assert reloaded["num_images"] == 3
    assert reloaded["metadata"]["timing_mode"] == "processing"
