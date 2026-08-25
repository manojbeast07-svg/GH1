"""Section 16 tests: the Performance Analytics tab's historical
benchmark loading/analytics layer (ui.services + cuda.final_benchmark's
new Section 16 loaders). Per spec item 43, unit tests use small
deterministic fixture benchmarks built with the SAME dataclasses/
writer helpers production code uses (never a second, duplicated JSON
schema) -- the full Section 11 benchmark suite is never run to test
this section.
"""

import json
from pathlib import Path

import pytest

from cuda.final_benchmark import (
    AggregatedStat,
    BenchmarkManifest,
    CanonicalBenchmarkResult,
    FourModeMetrics,
    build_final_summary,
)
from ui import services


def _stat(values):
    return AggregatedStat.from_values(values).to_dict()


def _write_fixture_benchmark(
    root: Path, benchmark_id: str, *,
    cpu_ms=100.0, basic_ms=50.0, enhanced_ms=40.0, basic_kernel_ms=10.0, enhanced_kernel_ms=4.0,
    image_count=10, seed=7, resolution=(64, 64), gpu_name="Fake GPU", dataset_fingerprint="shared-fingerprint",
    include_per_filter=True, include_batch_sweep=True, include_resolution_sweep=True, include_correctness=True,
    include_raw=True, timestamp_utc="2026-01-01T00:00:00+00:00",
):
    """Builds one small, fully self-contained fixture benchmark under
    `root`, using the real BenchmarkManifest/FourModeMetrics/
    CanonicalBenchmarkResult/build_final_summary() production code path
    -- not a second, hand-rolled JSON writer -- so the fixture's schema
    can never silently drift from what run_final_benchmark.py actually
    produces."""
    manifest = BenchmarkManifest(
        benchmark_id=benchmark_id, timestamp_utc=timestamp_utc, status="VALID", validation_notes=[],
        dataset_fingerprint=dataset_fingerprint, dataset_path="fake/dataset", valid_image_count=image_count,
        environment={"gpu_name": gpu_name, "cuda_runtime_version": "12.0", "gpu_driver_version": "560.0",
                     "cpu_model": "Fake CPU", "cpu_logical_cores": 8},
        seed=seed, requested_batch_size=image_count, selected_image_count=image_count,
        selected_relative_paths=[f"img_{i}.jpg" for i in range(image_count)],
        resolution=resolution, filter_config={"gaussian_kernel_size": 5}, basic_cuda_config={}, enhanced_cuda_config={},
        warmup_runs=2, measurement_runs=5,
    )
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    with open(root / "manifests" / f"{benchmark_id}.json", "w", encoding="utf-8") as fh:
        json.dump(manifest.to_dict(), fh, default=str)

    cpu_modes = FourModeMetrics(None, None, AggregatedStat.from_values([cpu_ms * 0.9, cpu_ms * 0.95]), AggregatedStat.from_values([cpu_ms, cpu_ms * 1.02, cpu_ms * 0.98]))
    basic_modes = FourModeMetrics(
        AggregatedStat.from_values([basic_kernel_ms, basic_kernel_ms * 1.05]),
        AggregatedStat.from_values([basic_ms, basic_ms * 1.02]),
        AggregatedStat.from_values([basic_ms, basic_ms * 1.02]),
        AggregatedStat.from_values([basic_ms, basic_ms * 1.02, basic_ms * 0.98]),
    )
    enhanced_modes = FourModeMetrics(
        AggregatedStat.from_values([enhanced_kernel_ms, enhanced_kernel_ms * 1.05]),
        AggregatedStat.from_values([enhanced_ms, enhanced_ms * 1.02]),
        AggregatedStat.from_values([enhanced_ms, enhanced_ms * 1.02]),
        AggregatedStat.from_values([enhanced_ms, enhanced_ms * 1.02, enhanced_ms * 0.98]),
    )
    canonical = CanonicalBenchmarkResult(
        manifest=manifest, cpu=cpu_modes, basic=basic_modes, enhanced=enhanced_modes,
        per_stage_basic_ms={}, per_stage_enhanced_ms={},
        cpu_images_per_second=image_count / (cpu_ms / 1000.0),
        basic_images_per_second=image_count / (basic_ms / 1000.0),
        enhanced_images_per_second=image_count / (enhanced_ms / 1000.0),
        basic_speedup_vs_cpu=cpu_ms / basic_ms, enhanced_speedup_vs_cpu=cpu_ms / enhanced_ms,
        enhanced_speedup_vs_basic_compute_only=basic_kernel_ms / enhanced_kernel_ms,
        enhanced_speedup_vs_basic_end_to_end=basic_ms / enhanced_ms,
        correctness_basic_vs_cpu={"max_abs_diff": 255, "mean_abs_diff": 0.01, "rmse": 1.0,
                                   "differing_pixel_count": 5, "differing_pixel_percentage": 0.01},
        correctness_enhanced_vs_cpu={"max_abs_diff": 255, "mean_abs_diff": 0.01, "rmse": 1.0,
                                      "differing_pixel_count": 5, "differing_pixel_percentage": 0.01},
        correctness_enhanced_vs_basic={"max_abs_diff": 0, "mean_abs_diff": 0.0, "rmse": 0.0,
                                        "differing_pixel_count": 0, "differing_pixel_percentage": 0.0},
    )

    per_filter = None
    if include_per_filter:
        filters = ["gaussian", "median", "sobel", "laplacian", "threshold"]
        basic_vals = [4.0, 3.0, 1.0, 1.0, 1.0]
        enhanced_vals = [1.0, 0.5, 0.9, 0.6, 0.7]
        total_basic, total_enhanced = sum(basic_vals), sum(enhanced_vals)
        total_reduction = total_basic - total_enhanced
        rows = []
        for name, b, e in zip(filters, basic_vals, enhanced_vals):
            reduction = b - e
            rows.append({
                "filter": name, "basic_kernel_ms": _stat([b, b * 1.05]), "enhanced_kernel_ms": _stat([e, e * 1.05]),
                "kernel_speedup": b / e, "basic_pct_of_total_basic_compute": 100.0 * b / total_basic,
                "enhanced_pct_of_total_enhanced_compute": 100.0 * e / total_enhanced,
                "absolute_reduction_ms": reduction,
                "pct_of_total_compute_reduction": 100.0 * reduction / total_reduction if total_reduction > 0 else 0.0,
            })
        per_filter = {"benchmark_id": benchmark_id, "timestamp_utc": timestamp_utc, "resolution": list(resolution),
                      "image_count": image_count, "seed": seed, "total_basic_compute_ms": total_basic,
                      "total_enhanced_compute_ms": total_enhanced, "rows": rows}

    batch_sweep = None
    if include_batch_sweep:
        rows = []
        for bs in (1, 8, 32):
            scale = bs / image_count
            c, b, e = cpu_ms * scale, basic_ms * scale, enhanced_ms * scale
            rows.append({
                "requested_batch_size": bs, "effective_batch_size": bs,
                "cpu_total_ms": _stat([c, c * 1.02]), "basic_total_ms": _stat([b, b * 1.02]),
                "enhanced_total_ms": _stat([e, e * 1.02]),
                "cpu_images_per_second": bs / (c / 1000.0), "basic_images_per_second": bs / (b / 1000.0),
                "enhanced_images_per_second": bs / (e / 1000.0),
                "basic_speedup_vs_cpu": c / b, "enhanced_speedup_vs_cpu": c / e, "enhanced_vs_basic_gain": b / e,
            })
        batch_sweep = {"benchmark_id": benchmark_id, "timestamp_utc": timestamp_utc, "resolution": list(resolution),
                       "seed": seed, "warmup_runs": 1, "measurement_runs": 2, "rows": rows}

    resolution_sweep = None
    if include_resolution_sweep:
        rows = [{
            "width": resolution[1], "height": resolution[0], "pixel_count": resolution[0] * resolution[1],
            "image_count": image_count, "cpu_ms_per_image": cpu_ms / image_count, "basic_ms_per_image": basic_ms / image_count,
            "enhanced_ms_per_image": enhanced_ms / image_count,
            "cpu_images_per_second": image_count / (cpu_ms / 1000.0), "basic_images_per_second": image_count / (basic_ms / 1000.0),
            "enhanced_images_per_second": image_count / (enhanced_ms / 1000.0),
        }]
        resolution_sweep = {"benchmark_id": benchmark_id, "timestamp_utc": timestamp_utc, "seed": seed,
                             "warmup_runs": 1, "measurement_runs": 2, "rows": rows}

    correctness_detail = None
    if include_correctness:
        correctness_detail = {
            "benchmark_id": benchmark_id, "timestamp_utc": timestamp_utc, "resolution": list(resolution),
            "image_count": image_count,
            "filter_level": {"gaussian": {"pass": True, "tolerance": 1}, "median": {"pass": True, "tolerance": 0}},
            "known_expected_differences": {"gaussian": "documented ±1 tolerance"},
            "pipeline_level": {"enhanced_vs_cpu": {"max_abs_diff": 255, "mean_abs_diff": 0.01, "rmse": 1.0,
                                                     "differing_pixel_count": 5, "differing_pixel_percentage": 0.01}},
            "baseline_comparison_note": "fixture", "overall_pass": True,
        }

    summary = build_final_summary(canonical, batch_sweep=batch_sweep, resolution_sweep=resolution_sweep,
                                   per_filter=per_filter, correctness=correctness_detail, results_root=root)

    if include_raw:
        for impl, ms in (("basic_cuda", basic_ms), ("enhanced_cuda", enhanced_ms)):
            # segments sum to exactly `ms` (== total_ms) so the breakdown-averaging test can assert an exact total
            runs = [{"run": i, "h2d_ms": ms * 0.10, "gaussian_ms": ms * 0.25, "median_ms": ms * 0.25, "sobel_ms": ms * 0.10,
                     "laplacian_ms": ms * 0.10, "threshold_ms": ms * 0.10, "d2h_ms": ms * 0.10, "compute_ms": ms * 0.80,
                     "total_ms": ms} for i in range(3)]
            (root / "raw" / impl).mkdir(parents=True, exist_ok=True)
            with open(root / "raw" / impl / f"{benchmark_id}.json", "w", encoding="utf-8") as fh:
                json.dump({"benchmark_id": benchmark_id, "runs": runs}, fh)
        cpu_runs = [{"run": i, "processing_ms": cpu_ms * 0.9} for i in range(3)]
        (root / "raw" / "cpu").mkdir(parents=True, exist_ok=True)
        with open(root / "raw" / "cpu" / f"{benchmark_id}.json", "w", encoding="utf-8") as fh:
            json.dump({"benchmark_id": benchmark_id, "runs": cpu_runs}, fh)

    # Real Section 11 runs also write batch_sweep/resolution_sweep/per_filter/correctness as
    # STANDALONE files (each under its own sub-benchmark_id, produced by a separate sweep
    # invocation) in addition to being nested into the summary -- the standalone loaders
    # (load_batch_sweep() etc.) read from those directories directly, so the fixture writes
    # them too, keyed under this same benchmark_id for simplicity.
    for subdir, payload in (("batch_sweeps", batch_sweep), ("resolution_sweeps", resolution_sweep),
                             ("per_filter", per_filter), ("correctness", correctness_detail)):
        if payload is None:
            continue
        (root / subdir).mkdir(parents=True, exist_ok=True)
        with open(root / subdir / f"{benchmark_id}.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, default=str)

    return summary


# -- 1: benchmark loader -------------------------------------------------------


def test_load_benchmark_summary_by_id(tmp_path):
    _write_fixture_benchmark(tmp_path, "fixture_a")
    loaded = services.load_benchmark_summary("fixture_a", results_root=tmp_path)
    assert loaded is not None
    assert loaded["benchmark_id"] == "fixture_a"


def test_load_benchmark_summary_defaults_to_latest(tmp_path):
    _write_fixture_benchmark(tmp_path, "fixture_old", timestamp_utc="2026-01-01T00:00:00+00:00")
    loaded = services.load_benchmark_summary(None, results_root=tmp_path)
    assert loaded is not None
    assert loaded["benchmark_id"] == "fixture_old"


# -- 2: canonical benchmark loading -------------------------------------------------------


def test_get_canonical_benchmark_id_from_designation_file(tmp_path):
    _write_fixture_benchmark(tmp_path, "fixture_a")
    _write_fixture_benchmark(tmp_path, "fixture_b")
    with open(tmp_path / "canonical.json", "w", encoding="utf-8") as fh:
        json.dump({"benchmark_id": "fixture_a"}, fh)

    assert services.get_canonical_benchmark_id(results_root=tmp_path) == "fixture_a"
    assert services.is_canonical_benchmark("fixture_a", results_root=tmp_path) is True
    assert services.is_canonical_benchmark("fixture_b", results_root=tmp_path) is False


def test_get_canonical_benchmark_id_falls_back_to_registry(tmp_path):
    _write_fixture_benchmark(tmp_path, "fixture_a")
    registry = {"entries": [{"experiment": "Final pipeline", "benchmark_id": "fixture_a"}]}
    with open(tmp_path / "experiment_registry.json", "w", encoding="utf-8") as fh:
        json.dump(registry, fh)

    assert services.get_canonical_benchmark_id(results_root=tmp_path) == "fixture_a"


def test_get_canonical_benchmark_id_none_when_nothing_designated(tmp_path):
    assert services.get_canonical_benchmark_id(results_root=tmp_path) is None
    assert services.is_canonical_benchmark(None, results_root=tmp_path) is False


def test_benchmark_metadata_flattens_manifest_fields(tmp_path):
    summary = _write_fixture_benchmark(tmp_path, "fixture_a", image_count=15, seed=99, resolution=(32, 48))
    meta = services.benchmark_metadata(summary)
    assert meta["benchmark_id"] == "fixture_a"
    assert meta["image_count"] == 15
    assert meta["seed"] == 99
    assert meta["resolution"] == [32, 48] or tuple(meta["resolution"]) == (32, 48)
    assert meta["gpu_name"] == "Fake GPU"


# -- 3-6: batch-sweep / resolution-sweep / per-filter / correctness loading -------------------------------------------------------


def test_load_batch_sweep(tmp_path):
    _write_fixture_benchmark(tmp_path, "fixture_a")
    sweep = services.load_batch_sweep("fixture_a", results_root=tmp_path)
    assert sweep is not None
    assert {r["effective_batch_size"] for r in sweep["rows"]} == {1, 8, 32}


def test_load_resolution_sweep(tmp_path):
    _write_fixture_benchmark(tmp_path, "fixture_a", resolution=(64, 64))
    sweep = services.load_resolution_sweep("fixture_a", results_root=tmp_path)
    assert sweep is not None
    assert sweep["rows"][0]["pixel_count"] == 64 * 64


def test_load_per_filter_results(tmp_path):
    _write_fixture_benchmark(tmp_path, "fixture_a")
    per_filter = services.load_per_filter_results("fixture_a", results_root=tmp_path)
    assert per_filter is not None
    assert {r["filter"] for r in per_filter["rows"]} == {"gaussian", "median", "sobel", "laplacian", "threshold"}


def test_load_correctness_results(tmp_path):
    _write_fixture_benchmark(tmp_path, "fixture_a")
    corr = services.load_correctness_results("fixture_a", results_root=tmp_path)
    assert corr is not None
    assert corr["overall_pass"] is True


def test_gpu_compute_breakdown_averages_raw_runs(tmp_path):
    _write_fixture_benchmark(tmp_path, "fixture_a", basic_ms=50.0)
    breakdown = services.gpu_compute_breakdown("fixture_a", "basic_cuda", results_root=tmp_path)
    assert breakdown is not None
    assert breakdown["n_runs"] == 3
    assert breakdown["total_ms"] == pytest.approx(50.0)
    segment_sum = (breakdown["h2d_ms"] + breakdown["gaussian_ms"] + breakdown["median_ms"] + breakdown["sobel_ms"]
                   + breakdown["laplacian_ms"] + breakdown["threshold_ms"] + breakdown["d2h_ms"])
    assert segment_sum == pytest.approx(50.0)


# -- 7: speedup calculations -------------------------------------------------------


def test_speedups_computed_not_hardcoded(tmp_path):
    summary = _write_fixture_benchmark(tmp_path, "fixture_a", cpu_ms=100.0, basic_ms=50.0, enhanced_ms=25.0,
                                        basic_kernel_ms=10.0, enhanced_kernel_ms=2.0)
    assert summary["speedups"]["basic_vs_cpu"] == pytest.approx(2.0)
    assert summary["speedups"]["enhanced_vs_cpu"] == pytest.approx(4.0)
    assert summary["speedups"]["enhanced_vs_basic_compute_only"] == pytest.approx(5.0)
    assert summary["speedups"]["enhanced_vs_basic_end_to_end"] == pytest.approx(2.0)


# -- 8: missing-artifact handling -------------------------------------------------------


def test_missing_benchmark_returns_none_not_crash(tmp_path):
    assert services.load_benchmark_summary("does_not_exist", results_root=tmp_path) is None
    assert services.load_batch_sweep("does_not_exist", results_root=tmp_path) is None
    assert services.load_resolution_sweep("does_not_exist", results_root=tmp_path) is None
    assert services.load_per_filter_results("does_not_exist", results_root=tmp_path) is None
    assert services.load_correctness_results("does_not_exist", results_root=tmp_path) is None
    assert services.gpu_compute_breakdown("does_not_exist", "basic_cuda", results_root=tmp_path) is None


def test_summary_missing_optional_sub_artifacts_stays_none(tmp_path):
    summary = _write_fixture_benchmark(tmp_path, "fixture_sparse", include_per_filter=False,
                                        include_batch_sweep=False, include_resolution_sweep=False,
                                        include_correctness=False)
    assert summary["per_filter"] is None
    assert summary["batch_sweep"] is None
    assert summary["resolution_sweep"] is None
    assert summary["correctness"]["detailed"] is None


def test_legacy_summary_without_images_per_second_field_falls_back_cleanly(tmp_path):
    """Regression (found via manual AppTest verification against a real,
    older stored summary): a summary written before the images_per_second
    schema fix lacks the top-level cpu/basic/enhanced_images_per_second
    keys entirely. ui.analytics.resolve_images_per_second() must derive a
    fallback from the stored mean total time + image count instead of
    raising KeyError."""
    from ui import analytics

    summary = _write_fixture_benchmark(tmp_path, "fixture_legacy", cpu_ms=100.0, image_count=10)
    del summary["cpu_images_per_second"]
    del summary["basic_images_per_second"]
    del summary["enhanced_images_per_second"]

    mean_ms = summary["cpu"]["mode4_end_to_end_ms"]["mean"]
    fallback = analytics.resolve_images_per_second(summary, "cpu_images_per_second", mean_ms)
    assert fallback == pytest.approx(10 / (mean_ms / 1000.0))


def test_current_schema_images_per_second_uses_stored_field_not_fallback(tmp_path):
    """When the field IS present (the current schema), it must be used
    verbatim -- never silently replaced by the derived value, even if
    they'd be numerically close."""
    from ui import analytics

    summary = _write_fixture_benchmark(tmp_path, "fixture_current", cpu_ms=100.0, image_count=10)
    stored_value = summary["cpu_images_per_second"]
    result = analytics.resolve_images_per_second(summary, "cpu_images_per_second", 999999.0)
    assert result == stored_value


def test_resolve_images_per_second_returns_none_when_unresolvable():
    from ui import analytics

    assert analytics.resolve_images_per_second({"manifest": {}}, "cpu_images_per_second", None) is None
    assert analytics.resolve_images_per_second({"manifest": {}}, "cpu_images_per_second", 10.0) is None


# -- 9: benchmark comparison -------------------------------------------------------


def test_compare_benchmarks_same_config_no_differences(tmp_path):
    _write_fixture_benchmark(tmp_path, "fixture_a", cpu_ms=100.0)
    _write_fixture_benchmark(tmp_path, "fixture_b", cpu_ms=90.0)
    result = services.compare_benchmarks("fixture_a", "fixture_b", results_root=tmp_path)
    assert result["differences"] == []
    assert result["summary_a"]["benchmark_id"] == "fixture_a"
    assert result["summary_b"]["benchmark_id"] == "fixture_b"


def test_compare_benchmarks_flags_differing_configuration(tmp_path):
    _write_fixture_benchmark(tmp_path, "fixture_a", gpu_name="GPU One", image_count=10)
    _write_fixture_benchmark(tmp_path, "fixture_b", gpu_name="GPU Two", image_count=20)
    result = services.compare_benchmarks("fixture_a", "fixture_b", results_root=tmp_path)
    assert any("GPU" in d for d in result["differences"])
    assert any("Image count" in d for d in result["differences"])


def test_compare_benchmarks_missing_id_raises_service_error(tmp_path):
    _write_fixture_benchmark(tmp_path, "fixture_a")
    with pytest.raises(services.ServiceError):
        services.compare_benchmarks("fixture_a", "does_not_exist", results_root=tmp_path)


# -- 10-11: CSV / JSON export -------------------------------------------------------


def test_export_benchmark_json_round_trips(tmp_path):
    summary = _write_fixture_benchmark(tmp_path, "fixture_a")
    text = services.export_benchmark_json(summary)
    reloaded = json.loads(text)
    assert reloaded["benchmark_id"] == "fixture_a"
    # Normalize through JSON on both sides (e.g. manifest.resolution is an in-memory tuple
    # but becomes a JSON array) rather than a strict Python-type equality check.
    assert reloaded == json.loads(json.dumps(summary, default=str))


def test_export_benchmark_csv_contains_headline_rows(tmp_path):
    summary = _write_fixture_benchmark(tmp_path, "fixture_a")
    text = services.export_benchmark_csv(summary)
    assert "benchmark_id,implementation,metric" in text
    assert "fixture_a,CPU,mode4_end_to_end_ms" in text
    assert "fixture_a,Basic CUDA,mode1_kernel_only_ms" in text
    assert "fixture_a,per_filter,gaussian" in text


def test_export_never_modifies_source_file(tmp_path):
    summary = _write_fixture_benchmark(tmp_path, "fixture_a")
    path = tmp_path / "summary" / "fixture_a.json"
    before = path.read_text(encoding="utf-8")
    services.export_benchmark_csv(summary)
    services.export_benchmark_json(summary)
    after = path.read_text(encoding="utf-8")
    assert before == after


# -- 12: historical/live separation -------------------------------------------------------


def test_historical_loaders_never_touch_live_output_dirs(tmp_path):
    _write_fixture_benchmark(tmp_path, "fixture_a")
    live_root = tmp_path / "outputs"
    assert not live_root.exists(), "loading a historical benchmark must never create outputs/ (live) directories"


def test_historical_and_live_result_paths_are_disjoint():
    from cuda.final_benchmark import DEFAULT_RESULTS_ROOT

    live_single = Path(__file__).resolve().parent.parent / "outputs" / "live_processing"
    live_batch = Path(__file__).resolve().parent.parent / "outputs" / "live_batch_processing"
    assert DEFAULT_RESULTS_ROOT != live_single
    assert DEFAULT_RESULTS_ROOT != live_batch
    assert str(live_single) not in str(DEFAULT_RESULTS_ROOT)
    assert str(live_batch) not in str(DEFAULT_RESULTS_ROOT)


# -- real-data integration test (item 44's manual launch check is done separately;
#    this exercises the loaders against the actual benchmark_results/ on disk) -------------------------------------------------------


def test_integration_real_canonical_benchmark_loads_and_is_self_consistent():
    canonical_id = services.get_canonical_benchmark_id()
    if canonical_id is None:
        pytest.skip("No canonical benchmark designated in this checkout's benchmark_results/.")
    summary = services.load_benchmark_summary(canonical_id)
    if summary is None:
        pytest.skip("Designated canonical benchmark_id has no stored summary in this checkout.")

    assert services.is_canonical_benchmark(canonical_id) is True
    meta = services.benchmark_metadata(summary)
    assert meta["image_count"] > 0

    speedups = summary["speedups"]
    assert speedups["basic_vs_cpu"] > 0
    assert speedups["enhanced_vs_cpu"] > 0

    breakdown = services.gpu_compute_breakdown(canonical_id, "basic_cuda")
    if breakdown is not None:
        assert breakdown["compute_ms"] == pytest.approx(summary["basic_cuda"]["mode1_kernel_only_ms"]["mean"], rel=1e-6)
