"""Section 11 tests: the final reproducible benchmark & experiment
framework (cuda/final_benchmark.py).

Per spec item 41, this suite does NOT make performance numbers hard
failures (a slow machine is not a test failure) -- it verifies
structure, serialization/deserialization round-trips, reproducibility
metadata, path/fingerprint validation, and that speedups/aggregations
are correctly COMPUTED from raw measurements (never hardcoded), using
small/fast synthetic or minimal real-dataset samples.
"""

from pathlib import Path

import numpy as np
import pytest

xray_cuda = pytest.importorskip(
    "xray_cuda", reason="xray_cuda extension is not built; run cmake --build build first."
)

import cuda.final_benchmark as fb  # noqa: E402
from cpu.filters import FilterConfig  # noqa: E402
from pipeline.dataset import DatasetManager  # noqa: E402
from pipeline.image_loader import load_image  # noqa: E402
from tests.conftest import get_real_dataset_path  # noqa: E402
from tests.fixtures import FIXTURES  # noqa: E402

cuda_present = xray_cuda.cuda_available()
pytestmark = pytest.mark.skipif(
    not cuda_present, reason="No usable CUDA device detected on this machine."
)


# -- AggregatedStat -------------------------------------------------------


def test_aggregated_stat_from_values_basic_statistics():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    stat = fb.AggregatedStat.from_values(values)
    assert stat.mean == 30.0
    assert stat.median == 30.0
    assert stat.min == 10.0
    assert stat.max == 50.0
    assert stat.n == 5
    assert stat.std > 0
    assert stat.coefficient_of_variation == pytest.approx(stat.std / stat.mean)


def test_aggregated_stat_does_not_drop_outliers():
    """An outlier must still be reflected in max/mean -- spec item 14:
    never silently delete slow runs."""
    values = [1.0, 1.0, 1.0, 1.0, 100.0]
    stat = fb.AggregatedStat.from_values(values)
    assert stat.max == 100.0
    assert stat.n == 5
    assert stat.mean == pytest.approx(20.8)


def test_aggregated_stat_single_value_has_zero_std():
    stat = fb.AggregatedStat.from_values([5.0])
    assert stat.std == 0.0
    assert stat.n == 1


def test_aggregated_stat_rejects_empty_values():
    with pytest.raises(ValueError):
        fb.AggregatedStat.from_values([])


def test_aggregated_stat_serializes_to_dict():
    stat = fb.AggregatedStat.from_values([1.0, 2.0, 3.0])
    d = stat.to_dict()
    assert set(d.keys()) == {"mean", "median", "min", "max", "std", "coefficient_of_variation", "n"}


# -- CudaImplementationConfig -------------------------------------------------------


def test_cuda_implementation_config_enhanced_defaults_are_not_basic():
    enhanced = fb.CudaImplementationConfig.enhanced_defaults()
    assert enhanced.gaussian_variant == "specialized"
    assert enhanced.median_variant == "network3x3"
    assert enhanced.sobel_variant == "specialized"
    assert enhanced.laplacian_variant == "specialized"
    assert enhanced.threshold_variant == "vectorized"


def test_basic_cuda_config_all_basic():
    basic = fb._basic_cuda_config()
    d = basic.to_dict()
    for key in ("gaussian_variant", "median_variant", "sobel_variant", "laplacian_variant", "threshold_variant"):
        assert d[key] == "basic"


# -- benchmark ID generation -------------------------------------------------------


def test_generate_benchmark_id_shape(tmp_path):
    bid = fb.generate_benchmark_id(seed=42, n_images=125, width=224, height=224, root=tmp_path)
    assert "seed42" in bid
    assert "batch125" in bid
    assert "224x224" in bid


def test_generate_benchmark_id_never_collides(tmp_path):
    """Two IDs requested for the identical config at the identical
    (frozen) timestamp must not collide -- spec item 30: never overwrite
    earlier data."""
    from datetime import datetime, timezone

    when = datetime(2026, 8, 24, 11, 15, 0, tzinfo=timezone.utc)
    (tmp_path / "manifests").mkdir(parents=True)
    first = fb.generate_benchmark_id(seed=42, n_images=125, width=224, height=224, when=when, root=tmp_path)
    (tmp_path / "manifests" / f"{first}.json").write_text("{}")
    second = fb.generate_benchmark_id(seed=42, n_images=125, width=224, height=224, when=when, root=tmp_path)
    assert first != second


# -- validation (spec items 33-34) -------------------------------------------------------


def test_validate_selection_paths_detects_missing_file():
    from pipeline.dataset import ImageSelection, SelectedItem

    selection = ImageSelection(
        mode="random_batch", seed=1, requested_batch_size=1, dataset_size=1,
        items=[SelectedItem(index=0, absolute_path=Path("/does/not/exist.jpg"),
                             relative_path="does/not/exist.jpg", filename="exist.jpg")],
    )
    problems = fb.validate_selection_paths(selection)
    assert len(problems) == 1
    assert "does not exist" in problems[0]


def test_sanity_check_timing_catches_negative_image_count():
    problems = fb.sanity_check_timing(0, 10.0, 5.0, 3.0)
    assert any("image count" in p for p in problems)


def test_sanity_check_timing_catches_negative_timings():
    problems = fb.sanity_check_timing(10, -1.0, 5.0, 3.0)
    assert any("cpu_total_ms" in p for p in problems)


def test_sanity_check_timing_passes_valid_input():
    problems = fb.sanity_check_timing(10, 10.0, 5.0, 3.0)
    assert problems == []


# -- canonical benchmark: serialization round-trip, raw data, correctness -------------------------------------------------------


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_canonical_benchmark_writes_raw_aggregated_manifest_and_round_trips(tmp_path):
    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    fingerprint = dm.fingerprint()
    selection = dm.random_batch(batch_size=8, seed=42)
    images = [load_image(p) for p in selection.selected_paths]
    shape = images[0].shape
    images_same = [img for img in images if img.shape == shape]
    if len(images_same) < 2:
        pytest.skip("Not enough same-resolution images in this sample.")

    result = fb.run_canonical_benchmark(
        images_same, selection, FilterConfig(),
        dataset_fingerprint=fingerprint, dataset_path=str(get_real_dataset_path()),
        valid_image_count=len(dm.paths), warmup_runs=1, measurement_runs=3, results_root=tmp_path,
    )
    bid = result.manifest.benchmark_id
    assert result.manifest.status == "VALID"

    # -- raw measurements: item 26, must not be only averages --
    import json
    raw_cpu = json.loads((tmp_path / "raw" / "cpu" / f"{bid}.json").read_text())
    raw_basic = json.loads((tmp_path / "raw" / "basic_cuda" / f"{bid}.json").read_text())
    raw_enhanced = json.loads((tmp_path / "raw" / "enhanced_cuda" / f"{bid}.json").read_text())
    assert len(raw_cpu["runs"]) == 3
    assert len(raw_basic["runs"]) == 3
    assert len(raw_enhanced["runs"]) == 3
    assert "compute_ms" in raw_basic["runs"][0]
    assert "gaussian_ms" in raw_basic["runs"][0]

    # -- round-trip: loader APIs must reconstruct exactly what was written --
    loaded_manifest = fb.load_manifest(bid, results_root=tmp_path)
    assert loaded_manifest["benchmark_id"] == bid
    assert loaded_manifest["dataset_fingerprint"] == fingerprint
    assert loaded_manifest["selected_relative_paths"] == [item.relative_path for item in selection.items[: len(images_same)]]

    loaded_aggregated = fb.load_aggregated(bid, results_root=tmp_path)
    assert loaded_aggregated["basic_speedup_vs_cpu"] == pytest.approx(result.basic_speedup_vs_cpu)

    # -- speedups computed from raw, not manually entered (spec item 34 check 4) --
    recomputed = result.cpu.mode4_end_to_end_ms.mean / result.basic.mode4_end_to_end_ms.mean
    assert result.basic_speedup_vs_cpu == pytest.approx(recomputed)


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_canonical_benchmark_enhanced_vs_basic_exact_correctness(tmp_path):
    """Enhanced vs Basic pipeline must be bit-exact -- the established
    Section 10 finding, re-verified here as a regression guard."""
    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=6, seed=7)
    images = [load_image(p) for p in selection.selected_paths]
    shape = images[0].shape
    images_same = [img for img in images if img.shape == shape]
    if len(images_same) < 2:
        pytest.skip("Not enough same-resolution images in this sample.")

    result = fb.run_canonical_benchmark(
        images_same, selection, FilterConfig(),
        dataset_fingerprint=dm.fingerprint(), dataset_path=str(get_real_dataset_path()),
        valid_image_count=len(dm.paths), warmup_runs=1, measurement_runs=2, results_root=tmp_path,
    )
    assert result.correctness_enhanced_vs_basic["max_abs_diff"] == 0
    assert result.correctness_enhanced_vs_basic["differing_pixel_percentage"] == 0.0


# -- batch sweep -------------------------------------------------------


def test_batch_sweep_produces_one_row_per_batch_size(tmp_path):
    images = [FIXTURES["random_deterministic"](size=32, seed=i) for i in range(20)]
    result = fb.run_batch_sweep(
        images, FilterConfig(), batch_sizes=[1, 4, 8, 16], warmup_runs=1, measurement_runs=2,
        seed=1, results_root=tmp_path,
    )
    assert len(result["rows"]) == 4
    effective_sizes = [row["effective_batch_size"] for row in result["rows"]]
    assert effective_sizes == [1, 4, 8, 16]


def test_batch_sweep_caps_requested_size_to_available_images(tmp_path):
    images = [FIXTURES["random_deterministic"](size=32, seed=i) for i in range(5)]
    result = fb.run_batch_sweep(
        images, FilterConfig(), batch_sizes=[1, 100], warmup_runs=1, measurement_runs=2,
        seed=1, results_root=tmp_path,
    )
    row = result["rows"][-1]
    assert row["requested_batch_size"] == 100
    assert row["effective_batch_size"] == 5  # capped, not padded or crashed


def test_batch_sweep_speedups_computed_from_aggregates(tmp_path):
    images = [FIXTURES["random_deterministic"](size=32, seed=i) for i in range(8)]
    result = fb.run_batch_sweep(
        images, FilterConfig(), batch_sizes=[8], warmup_runs=1, measurement_runs=2,
        seed=1, results_root=tmp_path,
    )
    row = result["rows"][0]
    recomputed = row["cpu_total_ms"]["mean"] / row["basic_total_ms"]["mean"]
    assert row["basic_speedup_vs_cpu"] == pytest.approx(recomputed)


# -- resolution sweep -------------------------------------------------------


def test_resolution_sweep_produces_one_row_per_group(tmp_path):
    groups = {
        (32, 32): [FIXTURES["random_deterministic"](size=32, seed=i) for i in range(4)],
        (48, 48): [FIXTURES["random_deterministic"](size=48, seed=i) for i in range(2)],
    }
    result = fb.run_resolution_sweep(groups, FilterConfig(), warmup_runs=1, measurement_runs=2,
                                      seed=1, results_root=tmp_path)
    assert len(result["rows"]) == 2
    for row in result["rows"]:
        assert row["pixel_count"] == row["width"] * row["height"]
        assert row["image_count"] > 0


def test_resolution_sweep_sorted_by_pixel_count_descending(tmp_path):
    groups = {
        (16, 16): [FIXTURES["random_deterministic"](size=16, seed=1)],
        (64, 64): [FIXTURES["random_deterministic"](size=64, seed=2)],
    }
    result = fb.run_resolution_sweep(groups, FilterConfig(), warmup_runs=1, measurement_runs=1,
                                      seed=1, results_root=tmp_path)
    pixel_counts = [row["pixel_count"] for row in result["rows"]]
    assert pixel_counts == sorted(pixel_counts, reverse=True)


# -- per-filter benchmark + Amdahl contribution -------------------------------------------------------


def test_per_filter_benchmark_covers_all_five_filters(tmp_path):
    images = [FIXTURES["random_deterministic"](size=32, seed=i) for i in range(4)]
    result = fb.run_per_filter_benchmark(
        images, FilterConfig(), warmup_runs=1, measurement_runs=2, seed=1, results_root=tmp_path,
    )
    filters = {row["filter"] for row in result["rows"]}
    assert filters == {"gaussian", "median", "sobel", "laplacian", "threshold"}


def test_per_filter_contribution_percentages_sum_to_100(tmp_path):
    """Amdahl-style contribution (spec item 21): each filter's share of
    the TOTAL compute-time reduction must sum to ~100% across all five
    filters -- a strong correctness check on the analysis itself."""
    images = [FIXTURES["random_deterministic"](size=48, seed=i) for i in range(6)]
    result = fb.run_per_filter_benchmark(
        images, FilterConfig(), warmup_runs=2, measurement_runs=5, seed=1, results_root=tmp_path,
    )
    total_pct = sum(row["pct_of_total_compute_reduction"] for row in result["rows"])
    assert total_pct == pytest.approx(100.0, abs=0.5)


def test_per_filter_kernel_speedup_matches_ratio(tmp_path):
    images = [FIXTURES["random_deterministic"](size=32, seed=i) for i in range(4)]
    result = fb.run_per_filter_benchmark(
        images, FilterConfig(), warmup_runs=1, measurement_runs=2, seed=1, results_root=tmp_path,
    )
    for row in result["rows"]:
        expected = row["basic_kernel_ms"]["mean"] / row["enhanced_kernel_ms"]["mean"]
        assert row["kernel_speedup"] == pytest.approx(expected)


# -- correctness benchmark -------------------------------------------------------


def test_correctness_benchmark_reports_all_five_filters_and_known_expected(tmp_path):
    images = [FIXTURES["random_deterministic"](size=32, seed=i) for i in range(3)]
    result = fb.run_correctness_benchmark(images, FilterConfig(), seed=1, results_root=tmp_path)
    assert set(result["filter_level"].keys()) == {"gaussian", "median", "sobel", "laplacian", "threshold"}
    assert set(result["known_expected_differences"].keys()) == {"gaussian", "median", "sobel", "laplacian", "threshold"}
    assert "pipeline_level" in result
    assert "overall_pass" in result


def test_correctness_benchmark_median_sobel_laplacian_threshold_exact(tmp_path):
    """These four filters have a documented exact (tolerance=0) standard
    -- a failure here indicates a real regression, not measurement noise."""
    images = [FIXTURES["random_deterministic"](size=32, seed=i) for i in range(3)]
    result = fb.run_correctness_benchmark(images, FilterConfig(), seed=1, results_root=tmp_path)
    for name in ("median", "sobel", "laplacian", "threshold"):
        assert result["filter_level"][name]["cpu_vs_basic_max_abs_diff"] == 0
        assert result["filter_level"][name]["cpu_vs_enhanced_max_abs_diff"] == 0
        assert result["filter_level"][name]["pass"] is True


def test_correctness_benchmark_pipeline_metrics_have_required_fields(tmp_path):
    images = [FIXTURES["random_deterministic"](size=32, seed=i) for i in range(3)]
    result = fb.run_correctness_benchmark(images, FilterConfig(), seed=1, results_root=tmp_path)
    for comparison in ("basic_vs_cpu", "enhanced_vs_cpu", "enhanced_vs_basic"):
        metrics = result["pipeline_level"][comparison]
        assert set(metrics.keys()) >= {
            "max_abs_diff", "mean_abs_diff", "rmse", "differing_pixel_count", "differing_pixel_percentage",
        }


# -- loader APIs -------------------------------------------------------


def test_load_benchmark_summary_returns_none_when_nothing_written(tmp_path):
    assert fb.load_benchmark_summary(results_root=tmp_path) is None


def test_load_batch_sweep_returns_none_when_nothing_written(tmp_path):
    assert fb.load_batch_sweep(results_root=tmp_path) is None


def test_load_by_explicit_id_returns_none_when_id_unknown(tmp_path):
    assert fb.load_batch_sweep(benchmark_id="does_not_exist", results_root=tmp_path) is None


def test_load_latest_picks_most_recently_written_file(tmp_path):
    images = [FIXTURES["random_deterministic"](size=32, seed=i) for i in range(4)]
    first = fb.run_batch_sweep(images, FilterConfig(), batch_sizes=[1], warmup_runs=1, measurement_runs=1,
                                seed=1, results_root=tmp_path)
    second = fb.run_batch_sweep(images, FilterConfig(), batch_sizes=[2], warmup_runs=1, measurement_runs=1,
                                 seed=2, results_root=tmp_path)
    loaded = fb.load_batch_sweep(results_root=tmp_path)
    assert loaded["benchmark_id"] == second["benchmark_id"]


# -- reproducibility -------------------------------------------------------


def test_check_reproducibility_missing_manifest(tmp_path):
    ok, notes = fb.check_reproducibility("nonexistent_benchmark_id", results_root=tmp_path)
    assert ok is False
    assert "No manifest found" in notes[0]


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_check_reproducibility_detects_fingerprint_mismatch(tmp_path):
    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=4, seed=42)
    images = [load_image(p) for p in selection.selected_paths]
    shape = images[0].shape
    images_same = [img for img in images if img.shape == shape]
    if len(images_same) < 2:
        pytest.skip("Not enough same-resolution images in this sample.")

    result = fb.run_canonical_benchmark(
        images_same, selection, FilterConfig(),
        dataset_fingerprint="deliberately_wrong_fingerprint_for_this_test",
        dataset_path=str(get_real_dataset_path()), valid_image_count=len(dm.paths),
        warmup_runs=1, measurement_runs=1, results_root=tmp_path,
    )
    ok, notes = fb.check_reproducibility(result.manifest.benchmark_id, results_root=tmp_path)
    assert ok is False
    assert any("fingerprint mismatch" in n for n in notes)


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_check_reproducibility_passes_when_dataset_unchanged(tmp_path):
    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    fingerprint = dm.fingerprint()
    selection = dm.random_batch(batch_size=4, seed=42)
    images = [load_image(p) for p in selection.selected_paths]
    shape = images[0].shape
    images_same = [img for img in images if img.shape == shape]
    if len(images_same) < 2:
        pytest.skip("Not enough same-resolution images in this sample.")

    result = fb.run_canonical_benchmark(
        images_same, selection, FilterConfig(),
        dataset_fingerprint=fingerprint, dataset_path=str(get_real_dataset_path()),
        valid_image_count=len(dm.paths), warmup_runs=1, measurement_runs=1, results_root=tmp_path,
    )
    ok, notes = fb.check_reproducibility(result.manifest.benchmark_id, results_root=tmp_path)
    assert ok is True


# -- experiment registry -------------------------------------------------------


def test_experiment_registry_seeds_known_experiments(tmp_path):
    registry = fb.update_experiment_registry(
        "Batch sweep", "some_id", ["batch_sweeps/some_id.json"], {"seed": 1}, results_root=tmp_path,
    )
    names = {e["experiment"] for e in registry["entries"]}
    assert "Gaussian optimization" in names
    assert "Batch sweep" in names


def test_experiment_registry_updates_existing_entry_not_duplicates(tmp_path):
    fb.update_experiment_registry("Batch sweep", "id1", ["a.json"], {"seed": 1}, results_root=tmp_path)
    registry = fb.update_experiment_registry("Batch sweep", "id2", ["b.json"], {"seed": 2}, results_root=tmp_path)
    matches = [e for e in registry["entries"] if e["experiment"] == "Batch sweep"]
    assert len(matches) == 1
    assert matches[0]["benchmark_id"] == "id2"


# -- final summary -------------------------------------------------------


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_build_final_summary_writes_latest_and_id_specific_files(tmp_path):
    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=4, seed=42)
    images = [load_image(p) for p in selection.selected_paths]
    shape = images[0].shape
    images_same = [img for img in images if img.shape == shape]
    if len(images_same) < 2:
        pytest.skip("Not enough same-resolution images in this sample.")

    result = fb.run_canonical_benchmark(
        images_same, selection, FilterConfig(),
        dataset_fingerprint=dm.fingerprint(), dataset_path=str(get_real_dataset_path()),
        valid_image_count=len(dm.paths), warmup_runs=1, measurement_runs=1, results_root=tmp_path,
    )
    summary = fb.build_final_summary(result, results_root=tmp_path)
    assert (tmp_path / "summary" / "latest.json").exists()
    assert (tmp_path / "summary" / f"{result.manifest.benchmark_id}.json").exists()

    loaded_latest = fb.load_benchmark_summary(results_root=tmp_path)
    assert loaded_latest["benchmark_id"] == summary["benchmark_id"]
    loaded_by_id = fb.load_benchmark_summary(benchmark_id=result.manifest.benchmark_id, results_root=tmp_path)
    assert loaded_by_id["benchmark_id"] == summary["benchmark_id"]
