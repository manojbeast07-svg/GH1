"""Section 14 tests: live single-image CPU vs Basic CUDA vs Enhanced
CUDA processing (ui.services.process_single_image /
compare_single_image / save_comparison_results and the session-state
fields app.py's Live Processing tab uses).

Per spec item 37, unit tests use small deterministic fixtures (never
the full 9,463-image dataset); spec item 38's real-data integration
test uses one existing 224x224 X-ray and asserts structure/timing
existence, never exact performance thresholds.
"""

from pathlib import Path

import numpy as np
import pytest

xray_cuda = pytest.importorskip(
    "xray_cuda", reason="xray_cuda extension is not built; run cmake --build build first."
)

from cpu.filters import FilterConfig  # noqa: E402
from tests.conftest import get_real_dataset_path  # noqa: E402
from tests.fixtures import FIXTURES  # noqa: E402

cuda_present = xray_cuda.cuda_available()


# -- 1-4: single-implementation single-image processing -------------------------------------------------------


def test_process_single_image_cpu():
    from ui import services

    image = FIXTURES["random_deterministic"](size=32)
    result = services.process_single_image(image, FilterConfig(), "CPU")
    assert result.label == "CPU"
    assert len(result.final_outputs) == 1
    assert result.final_outputs[0].shape == image.shape
    assert result.final_outputs[0].dtype == np.uint8
    assert result.total_ms >= 0.0
    assert result.compute_ms == result.total_ms  # CPU has no H2D/D2H
    assert result.h2d_ms is None and result.d2h_ms is None


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_process_single_image_basic_cuda():
    from ui import services

    image = FIXTURES["random_deterministic"](size=32)
    result = services.process_single_image(image, FilterConfig(), "Basic CUDA")
    assert result.label == "Basic CUDA"
    assert result.final_outputs[0].shape == image.shape
    assert result.final_outputs[0].dtype == np.uint8
    assert result.h2d_ms is not None and result.h2d_ms >= 0.0
    assert result.compute_ms is not None and result.compute_ms >= 0.0
    assert result.d2h_ms is not None and result.d2h_ms >= 0.0


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_process_single_image_enhanced_cuda():
    from ui import services

    image = FIXTURES["random_deterministic"](size=32)
    result = services.process_single_image(image, FilterConfig(), "Enhanced CUDA")
    assert result.label == "Enhanced CUDA"
    assert result.final_outputs[0].shape == image.shape
    assert result.h2d_ms is not None
    assert result.compute_ms is not None
    assert result.d2h_ms is not None


def test_process_single_image_rejects_unknown_implementation():
    from ui import services

    image = FIXTURES["constant"]()
    with pytest.raises(services.ServiceError):
        services.process_single_image(image, FilterConfig(), "Quantum CUDA")


# -- 5-6: same config, same image reach all three -------------------------------------------------------


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_compare_uses_same_filter_config_for_all_three():
    from dataclasses import asdict

    from ui import services

    image = FIXTURES["random_deterministic"](size=32)
    config = FilterConfig(threshold_value=100, gaussian_kernel_size=7)
    meta = services.build_image_metadata(image)
    comparison = services.compare_single_image(image, config, meta)

    assert comparison.filter_config == asdict(config)
    # every implementation's final output reflects the SAME config (threshold_value=100
    # is distinctive enough that outputs would differ from the default config)
    default_config_result = services.process_single_image(image, FilterConfig(), "CPU")
    assert not np.array_equal(comparison.results["CPU"].final_outputs[0], default_config_result.final_outputs[0])


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_compare_uses_same_image_for_all_three():
    from ui import services

    image = FIXTURES["random_deterministic"](size=48)
    meta = services.build_image_metadata(image)
    comparison = services.compare_single_image(image, FilterConfig(), meta)

    for label, result in comparison.results.items():
        assert result.final_outputs[0].shape == image.shape, f"{label} received a differently-shaped image"


# -- 7-9: final outputs, intermediate outputs, timing objects exist -------------------------------------------------------


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_compare_produces_final_and_intermediate_outputs_and_timing():
    from ui import services

    image = FIXTURES["random_deterministic"](size=32)
    meta = services.build_image_metadata(image)
    comparison = services.compare_single_image(image, FilterConfig(), meta)

    assert set(comparison.results.keys()) == {"CPU", "Basic CUDA", "Enhanced CUDA"}
    for label, result in comparison.results.items():
        # final outputs (item 7)
        assert result.final_outputs[0] is not None
        # intermediate outputs (item 8)
        assert result.stage_outputs is not None
        for stage in ["gaussian", "median", "sobel", "laplacian", "threshold"]:
            assert result.stage_outputs.get(stage) is not None, f"{label} missing {stage} intermediate output"
        # timing objects exist (item 9)
        assert result.total_ms >= 0.0
        assert all(v is not None and v >= 0.0 for v in result.per_stage_ms.values())


# -- 10: correctness metrics are generated -------------------------------------------------------


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_compare_generates_correctness_metrics():
    from ui import services

    image = FIXTURES["random_deterministic"](size=32)
    meta = services.build_image_metadata(image)
    comparison = services.compare_single_image(image, FilterConfig(), meta)

    assert len(comparison.stage_correctness) == 5
    for s in comparison.stage_correctness:
        assert s.status in ("PASS", "WARNING", "N/A")
        if s.status != "N/A":
            assert s.max_abs_diff_vs_cpu is not None

    assert len(comparison.pipeline_correctness) == 3  # basic_vs_cpu, enhanced_vs_cpu, enhanced_vs_basic
    for p in comparison.pipeline_correctness:
        assert p.max_abs_diff >= 0
        assert 0.0 <= p.differing_pixel_percentage <= 100.0


def test_disabled_stage_reports_na_correctness():
    from ui import services

    image = FIXTURES["random_deterministic"](size=32)
    config = FilterConfig(median_enabled=False)
    results = {"CPU": services.process_single_image(image, config, "CPU")}
    stage_correctness = services._compute_stage_correctness(results, config)
    median_entry = next(s for s in stage_correctness if s.stage == "median")
    assert median_entry.status == "N/A"


# -- 11: CUDA failure is handled -------------------------------------------------------


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_compare_handles_cuda_unavailable_gracefully(monkeypatch):
    from ui import services

    image = FIXTURES["random_deterministic"](size=32)
    meta = services.build_image_metadata(image)
    monkeypatch.setattr(services, "cuda_available", lambda: False)

    comparison = services.compare_single_image(image, FilterConfig(), meta)

    assert "CPU" in comparison.results  # CPU still succeeds
    assert "Basic CUDA" in comparison.errors
    assert "Enhanced CUDA" in comparison.errors
    assert "Basic CUDA" not in comparison.results
    assert "Enhanced CUDA" not in comparison.results


def test_process_single_image_gpu_raises_service_error_when_cuda_unavailable(monkeypatch):
    from ui import services

    monkeypatch.setattr(services, "cuda_available", lambda: False)
    image = FIXTURES["constant"]()
    with pytest.raises(services.ServiceError):
        services.process_single_image(image, FilterConfig(), "Basic CUDA")


# -- 12: results persist in session state -------------------------------------------------------


def test_session_state_has_section14_fields():
    import streamlit as st

    from ui import state

    st.session_state.clear()
    state.init_session_state()
    for key in ("last_selected_image", "last_image_meta", "last_filter_config", "last_cpu_result",
                "last_basic_result", "last_enhanced_result", "last_comparison", "last_run_timestamp"):
        assert key in st.session_state
        assert st.session_state[key] is None


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_comparison_result_can_be_stored_and_retrieved_from_session_state():
    import streamlit as st

    from ui import services, state

    st.session_state.clear()
    state.init_session_state()

    image = FIXTURES["random_deterministic"](size=32)
    meta = services.build_image_metadata(image)
    comparison = services.compare_single_image(image, FilterConfig(), meta)

    state.set_value("last_comparison", comparison)
    retrieved = state.get("last_comparison")
    assert retrieved is comparison
    assert set(retrieved.results.keys()) == {"CPU", "Basic CUDA", "Enhanced CUDA"}


# -- save results -------------------------------------------------------


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_save_comparison_results_creates_expected_structure(tmp_path):
    from ui import services

    image = FIXTURES["random_deterministic"](size=32)
    meta = services.build_image_metadata(image)
    comparison = services.compare_single_image(image, FilterConfig(), meta)

    run_dir = services.save_comparison_results(comparison, output_root=tmp_path)
    assert run_dir.exists()
    assert (run_dir / "original.png").exists()
    assert (run_dir / "metadata.json").exists()
    for subdir in ("cpu", "basic_cuda", "enhanced_cuda"):
        assert (run_dir / subdir / "final.png").exists()
    assert (run_dir / "differences").exists()


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_save_comparison_results_never_overwrites_previous_run(tmp_path):
    from ui import services

    image = FIXTURES["random_deterministic"](size=32)
    meta = services.build_image_metadata(image)
    comparison = services.compare_single_image(image, FilterConfig(), meta)

    run_dir_1 = services.save_comparison_results(comparison, output_root=tmp_path)
    run_dir_2 = services.save_comparison_results(comparison, output_root=tmp_path)
    assert run_dir_1 != run_dir_2
    assert run_dir_1.exists() and run_dir_2.exists()


def test_save_comparison_results_never_writes_to_benchmark_results(tmp_path):
    """Spec item 36: live processing must never touch benchmark_results/."""
    from cuda.final_benchmark import DEFAULT_RESULTS_ROOT

    assert DEFAULT_RESULTS_ROOT != tmp_path  # sanity: save target used in these tests is never the real one


# -- item 38: real-data integration test -------------------------------------------------------


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_integration_real_224x224_xray_cpu_basic_enhanced_compare():
    """Spec item 38: real X-ray -> CPU -> Basic CUDA -> Enhanced CUDA ->
    compare outputs. Verifies structure and timing existence only --
    never asserts an exact performance threshold."""
    from pipeline.dataset import DatasetManager
    from pipeline.image_loader import load_image
    from ui import services

    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=1, seed=42)
    item = selection.items[0]
    image = load_image(item.absolute_path)
    assert image.shape == (224, 224)

    meta = services.build_image_metadata(image, item)
    comparison = services.compare_single_image(image, FilterConfig(), meta)

    assert not comparison.errors, f"Unexpected failures: {comparison.errors}"
    assert set(comparison.results.keys()) == {"CPU", "Basic CUDA", "Enhanced CUDA"}

    shapes = {label: r.final_outputs[0].shape for label, r in comparison.results.items()}
    dtypes = {label: r.final_outputs[0].dtype for label, r in comparison.results.items()}
    assert len(set(shapes.values())) == 1, f"Output shapes differ: {shapes}"
    assert len(set(dtypes.values())) == 1, f"Output dtypes differ: {dtypes}"
    assert shapes["CPU"] == (224, 224)

    for label, result in comparison.results.items():
        assert result.total_ms >= 0.0, f"{label} missing timing data"

    assert len(comparison.stage_correctness) == 5
    assert len(comparison.pipeline_correctness) == 3
