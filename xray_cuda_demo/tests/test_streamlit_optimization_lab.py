"""Section 17 tests: the Interactive CUDA Optimization Lab
(ui.services' historical loaders + live variant-comparison functions,
backed by cuda.optimization_lab.measure_variant()).

Per spec item 43, unit tests use small deterministic fixtures where
possible; a handful require the real compiled xray_cuda extension and
a real CUDA device (skipped automatically when unavailable, matching
every other GPU-touching test file in this project), and one real-data
integration test exercises the full live comparison path end to end.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pytest

xray_cuda = pytest.importorskip(
    "xray_cuda", reason="xray_cuda extension is not built; run cmake --build build first."
)

from cpu.filters import FilterConfig  # noqa: E402
from cuda.gaussian import GAUSSIAN_VARIANTS  # noqa: E402
from tests.conftest import get_real_dataset_path  # noqa: E402
from ui import services  # noqa: E402

cuda_present = xray_cuda.cuda_available()


def _fake_correctness_dict(max_abs_diff=0):
    return {"max_abs_diff": max_abs_diff, "mean_abs_diff": 0.0, "rmse": 0.0,
            "differing_pixel_count": 0, "differing_pixel_percentage": 0.0}


def _make_fake_comparison(filter_name="median", variant_a="basic", variant_b="network3x3", n_images=4):
    out_a = [np.zeros((8, 8), dtype=np.uint8) for _ in range(n_images)]
    out_b = [np.zeros((8, 8), dtype=np.uint8) for _ in range(n_images)]
    result_a = services.VariantRunResult(variant_a, False, [1.0, 1.1, 0.9], 1.0, out_a)
    result_b = services.VariantRunResult(variant_b, True, [0.2, 0.25, 0.18], 0.21, out_b)
    return services.VariantComparisonResult(
        filter_name=filter_name, variant_a=variant_a, variant_b=variant_b, filter_config={"median_kernel_size": 3},
        n_images=n_images, warmup_runs=2, measurement_runs=3, result_a=result_a, result_b=result_b,
        speedup_a_over_b=1.0 / 0.21, correctness_a_vs_b=_fake_correctness_dict(),
        correctness_a_vs_cpu=_fake_correctness_dict(), correctness_b_vs_cpu=_fake_correctness_dict(),
        timestamp_utc="2026-01-01T00:00:00+00:00",
    )


# -- 1: filter selector -------------------------------------------------------


def test_optimization_lab_filters_lists_all_five():
    assert services.optimization_lab_filters() == ["gaussian", "median", "sobel", "laplacian", "threshold"]


# -- 2: variant selector -------------------------------------------------------


def test_optimization_lab_variants_gaussian_matches_repository_names():
    variants = services.optimization_lab_variants("gaussian")
    assert variants == ("basic",) + GAUSSIAN_VARIANTS


def test_optimization_lab_variants_unknown_filter_raises():
    with pytest.raises(services.ServiceError):
        services.optimization_lab_variants("not_a_real_filter")


# -- 3: production default detection -------------------------------------------------------


def test_production_defaults_match_enhanced_pipeline_defaults():
    from cuda.final_benchmark import CudaImplementationConfig

    enhanced = CudaImplementationConfig.enhanced_defaults()
    assert services.optimization_lab_production_default("gaussian") == enhanced.gaussian_variant
    assert services.optimization_lab_production_default("median") == enhanced.median_variant
    assert services.optimization_lab_production_default("sobel") == enhanced.sobel_variant
    assert services.optimization_lab_production_default("laplacian") == enhanced.laplacian_variant
    assert services.optimization_lab_production_default("threshold") == enhanced.threshold_variant


def test_production_default_unknown_filter_raises():
    with pytest.raises(services.ServiceError):
        services.optimization_lab_production_default("not_a_real_filter")


# -- 4: historical experiment loading -------------------------------------------------------


def test_load_variant_sweep_missing_returns_none(tmp_path):
    assert services.load_variant_sweep("gaussian", results_root=tmp_path) is None


def test_load_fusion_experiment_missing_returns_none(tmp_path):
    assert services.load_fusion_experiment(results_root=tmp_path) is None


def test_load_variant_sweep_real_backfilled_data():
    sweep = services.load_variant_sweep("gaussian")
    if sweep is None:
        pytest.skip("No historical variant sweep backfilled in this checkout "
                    "(run scripts/run_optimization_lab_experiments.py).")
    assert sweep["filter"] == "gaussian"
    variants = {r["variant"] for r in sweep["rows"]}
    assert variants == {"basic"} | set(GAUSSIAN_VARIANTS)
    basic_row = next(r for r in sweep["rows"] if r["variant"] == "basic")
    assert basic_row["speedup_vs_basic"] == 1.0


def test_load_variant_sweep_median_has_per_kernel_size_data():
    sweep = services.load_variant_sweep("median")
    if sweep is None:
        pytest.skip("No historical median sweep backfilled in this checkout.")
    assert set(sweep["by_kernel_size"].keys()) == {"3", "5", "7"}
    # network3x3 only applies at kernel_size=3 (spec item 13)
    k3_variants = {r["variant"] for r in sweep["by_kernel_size"]["3"]}
    k5_variants = {r["variant"] for r in sweep["by_kernel_size"]["5"]}
    assert "network3x3" in k3_variants
    assert "network3x3" not in k5_variants


def test_load_fusion_experiment_real_backfilled_data():
    fusion = services.load_fusion_experiment()
    if fusion is None:
        pytest.skip("No historical fusion experiment backfilled in this checkout.")
    assert fusion["wired_into_production"] is False
    assert fusion["fused_kernel_ms"]["mean"] > 0
    assert fusion["unfused_combined_kernel_ms"]["mean"] > 0


# -- 5-6: live experiment configuration + same-input enforcement -------------------------------------------------------


def _small_real_batch(n=8, seed=42, size_hint=224):
    dataset_path = get_real_dataset_path()
    if dataset_path is None:
        return None, None
    dm, _info = services.scan_dataset(str(dataset_path))
    selection = services.select_random_batch(dm, batch_size=n, seed=seed)
    images = services.load_selection_images(selection)
    groups = services.group_images_by_shape(images)
    dominant = max(groups, key=lambda k: len(groups[k]))
    return [images[i] for i in groups[dominant]], selection


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_compare_filter_variants_same_input_reaches_both_variants():
    images, _sel = _small_real_batch(n=8)
    config = FilterConfig()
    comparison = services.compare_filter_variants(images, config, "threshold", "basic", "vectorized",
                                                     warmup_runs=1, measurement_runs=2)
    assert comparison.n_images == len(images)
    assert len(comparison.result_a.final_outputs) == len(images)
    assert len(comparison.result_b.final_outputs) == len(images)
    # same shape confirms both variants processed the identical batch, not independently resampled
    assert comparison.result_a.final_outputs[0].shape == comparison.result_b.final_outputs[0].shape == images[0].shape


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_compare_filter_variants_rejects_mixed_resolution():
    images = [np.zeros((32, 32), dtype=np.uint8), np.zeros((48, 48), dtype=np.uint8)]
    with pytest.raises(services.ServiceError):
        services.compare_filter_variants(images, FilterConfig(), "sobel", "basic", "specialized")


# -- 7: speedup calculation -------------------------------------------------------


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_speedup_is_ratio_of_measured_means_not_estimated():
    images, _sel = _small_real_batch(n=8)
    config = FilterConfig()
    comparison = services.compare_filter_variants(images, config, "threshold", "basic", "vectorized",
                                                     warmup_runs=1, measurement_runs=3)
    expected = comparison.result_a.kernel_ms_mean / comparison.result_b.kernel_ms_mean
    assert comparison.speedup_a_over_b == pytest.approx(expected)


def test_speedup_uses_saved_fake_comparison_consistently():
    comparison = _make_fake_comparison()
    assert comparison.speedup_a_over_b == pytest.approx(comparison.result_a.kernel_ms_mean / comparison.result_b.kernel_ms_mean)


# -- 8: correctness calculation -------------------------------------------------------


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_correctness_exact_filter_is_bit_exact_vs_cpu():
    """Median is established exact (tolerance=0); network3x3 vs CPU must
    show max_abs_diff == 0, computed from THIS run, not hardcoded."""
    images, _sel = _small_real_batch(n=8)
    config = FilterConfig()
    comparison = services.compare_filter_variants(images, config, "median", "basic", "network3x3",
                                                     warmup_runs=1, measurement_runs=2)
    assert comparison.correctness_a_vs_cpu["max_abs_diff"] == 0
    assert comparison.correctness_b_vs_cpu["max_abs_diff"] == 0
    assert comparison.correctness_a_vs_b["max_abs_diff"] == 0


# -- 9: Save Experiment -------------------------------------------------------


def test_save_optimization_experiment_creates_expected_structure(tmp_path):
    comparison = _make_fake_comparison()
    run_dir = services.save_optimization_experiment(
        comparison, dataset_fingerprint="fake-fp", selected_relative_paths=["a.jpg", "b.jpg"], output_root=tmp_path)
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "timing.json").exists()
    assert (run_dir / "correctness.json").exists()

    import json
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["filter"] == "median"
    assert metadata["dataset_fingerprint"] == "fake-fp"
    timing = json.loads((run_dir / "timing.json").read_text(encoding="utf-8"))
    assert timing["speedup_a_over_b"] == pytest.approx(comparison.speedup_a_over_b)


def test_save_optimization_experiment_never_overwrites(tmp_path):
    comparison = _make_fake_comparison()
    run_dir_1 = services.save_optimization_experiment(comparison, output_root=tmp_path)
    run_dir_2 = services.save_optimization_experiment(comparison, output_root=tmp_path)
    assert run_dir_1 != run_dir_2


def test_save_optimization_experiment_never_saves_raw_image_outputs(tmp_path):
    """Spec item 36-37: this is a timing/correctness record, not an
    image export -- no PNG/array files, only the three JSON documents."""
    comparison = _make_fake_comparison()
    run_dir = services.save_optimization_experiment(comparison, output_root=tmp_path)
    files = sorted(p.name for p in run_dir.iterdir())
    assert files == ["correctness.json", "metadata.json", "timing.json"]


# -- 10: missing variant handling -------------------------------------------------------


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_compare_filter_variants_rejects_unknown_variant():
    images = [np.zeros((16, 16), dtype=np.uint8)]
    with pytest.raises(services.ServiceError):
        services.compare_filter_variants(images, FilterConfig(), "gaussian", "basic", "not_a_real_variant")


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_compare_filter_variants_rejects_network3x3_at_kernel_size_5():
    """Spec item 13: network3x3 requires kernel_size=3; must fail
    clearly (ServiceError) rather than crash or silently ignore."""
    images = [np.zeros((16, 16), dtype=np.uint8)]
    config = FilterConfig(median_kernel_size=5)
    with pytest.raises(services.ServiceError):
        services.compare_filter_variants(images, config, "median", "basic", "network3x3", warmup_runs=1, measurement_runs=1)


# -- 11: CUDA unavailable handling -------------------------------------------------------


def test_compare_filter_variants_raises_service_error_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(services, "cuda_available", lambda: False)
    images = [np.zeros((16, 16), dtype=np.uint8)]
    with pytest.raises(services.ServiceError):
        services.compare_filter_variants(images, FilterConfig(), "gaussian", "basic", "specialized")


def test_measure_pipeline_impact_raises_service_error_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(services, "cuda_available", lambda: False)
    images = [np.zeros((16, 16), dtype=np.uint8)]
    with pytest.raises(services.ServiceError):
        services.measure_pipeline_impact(images, FilterConfig(), "gaussian", "specialized")


def test_optimization_lab_variants_falls_back_when_cuda_import_fails(monkeypatch):
    monkeypatch.setattr(services, "_CUDA_IMPORT_ERROR", "extension not built (simulated)")
    variants = services.optimization_lab_variants("threshold")
    assert variants == ("basic", "vectorized", "multi_pixel")


# -- 12: historical/live separation -------------------------------------------------------


def test_live_output_root_never_equals_benchmark_results():
    from cuda.final_benchmark import DEFAULT_RESULTS_ROOT

    assert services.DEFAULT_OPTIMIZATION_LAB_OUTPUT_ROOT != DEFAULT_RESULTS_ROOT
    assert "benchmark_results" not in str(services.DEFAULT_OPTIMIZATION_LAB_OUTPUT_ROOT)


def test_variant_sweeps_root_lives_under_benchmark_results_but_separate_subdir():
    from cuda.final_benchmark import DEFAULT_RESULTS_ROOT

    root = services._variant_sweeps_root()
    assert root == DEFAULT_RESULTS_ROOT / "variant_sweeps"
    assert root != DEFAULT_RESULTS_ROOT / "summary"
    assert root != DEFAULT_RESULTS_ROOT / "manifests"


def test_save_optimization_experiment_never_writes_to_benchmark_results(tmp_path):
    from cuda.final_benchmark import DEFAULT_RESULTS_ROOT

    comparison = _make_fake_comparison()
    run_dir = services.save_optimization_experiment(comparison, output_root=tmp_path)
    assert DEFAULT_RESULTS_ROOT not in run_dir.parents
    assert run_dir != DEFAULT_RESULTS_ROOT


# -- real-data integration test (item 43's real-data requirement) -------------------------------------------------------


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_integration_real_live_variant_comparison_median_basic_vs_network3x3():
    """batch=8, seed=42, filter=median, basic vs network3x3 -- asserts
    structure/correctness only, never an exact performance threshold."""
    images, selection = _small_real_batch(n=8, seed=42)
    config = FilterConfig()

    comparison = services.compare_filter_variants(images, config, "median", "basic", "network3x3",
                                                     warmup_runs=2, measurement_runs=5)
    assert comparison.result_a.kernel_ms_mean > 0
    assert comparison.result_b.kernel_ms_mean > 0
    assert comparison.speedup_a_over_b > 0
    assert comparison.correctness_a_vs_b["max_abs_diff"] == 0

    impact = services.measure_pipeline_impact(images, config, "median", "network3x3", warmup_runs=1, measurement_runs=2)
    assert impact.basic_pipeline_ms > 0
    assert impact.enhanced_pipeline_ms > 0
