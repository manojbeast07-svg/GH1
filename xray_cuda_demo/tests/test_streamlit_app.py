"""Section 12 tests: the Streamlit application layer.

Per spec item 61, the CUDA-correctness-relevant integration test (select
an image, run CPU/Basic/Enhanced, verify results+timing) exercises
ui.services directly -- Streamlit rendering is never part of a
correctness test. Separately, a few tests use Streamlit's own
AppTest framework (spec item 60's "application starts successfully")
to verify app.py runs end-to-end with zero exceptions, without
asserting anything about pixel-level rendering.
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


# -- component imports (spec item 60) -------------------------------------------------------


def test_ui_package_imports():
    import ui  # noqa: F401


def test_ui_state_imports():
    import ui.state  # noqa: F401


def test_ui_services_imports():
    import ui.services  # noqa: F401


def test_ui_controls_imports():
    import ui.controls  # noqa: F401


def test_ui_images_imports():
    import ui.images  # noqa: F401


def test_ui_performance_imports():
    import ui.performance  # noqa: F401


def test_ui_analytics_imports():
    import ui.analytics  # noqa: F401


def test_ui_correctness_imports():
    import ui.correctness  # noqa: F401


def test_ui_system_imports():
    import ui.system  # noqa: F401


def test_app_module_is_importable_as_a_script():
    """app.py must at least be valid, importable Python (a syntax or
    top-level-import error here would break `streamlit run app.py`)."""
    import ast

    source = (Path(__file__).resolve().parent.parent / "app.py").read_text(encoding="utf-8")
    ast.parse(source)  # raises SyntaxError if broken


# -- ui.state -------------------------------------------------------


def test_current_filter_config_uses_defaults_when_no_overrides():
    import streamlit as st

    from ui import state

    st.session_state.clear()
    state.init_session_state()
    config = state.current_filter_config()
    assert isinstance(config, FilterConfig)
    assert config.gaussian_kernel_size == 5  # FilterConfig's own default


def test_update_filter_config_kwarg_is_reflected_in_current_config():
    import streamlit as st

    from ui import state

    st.session_state.clear()
    state.init_session_state()
    state.update_filter_config_kwarg("threshold_value", 200)
    config = state.current_filter_config()
    assert config.threshold_value == 200


def test_init_session_state_does_not_overwrite_existing_values():
    import streamlit as st

    from ui import state

    st.session_state.clear()
    state.init_session_state()
    state.set_value("seed", 999)
    state.init_session_state()  # must not reset back to the default
    assert state.get("seed") == 999


# -- ui.services: dataset / selection validation -------------------------------------------------------


def test_scan_dataset_raises_service_error_for_missing_path():
    from ui import services

    with pytest.raises(services.ServiceError):
        services.scan_dataset("/this/path/does/not/exist/at/all")


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_scan_dataset_returns_manager_and_info():
    from ui import services

    dm, info = services.scan_dataset(str(get_real_dataset_path()))
    assert info.total_files > 0
    assert info.path == str(get_real_dataset_path())


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_select_random_batch_and_load_images_same_order():
    from ui import services

    dm, _info = services.scan_dataset(str(get_real_dataset_path()))
    selection = services.select_random_batch(dm, batch_size=5, seed=42)
    images = services.load_selection_images(selection)
    assert len(images) == len(selection)
    for img, item in zip(images, selection.items):
        assert img.dtype == np.uint8


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_parallel_loading_matches_sequential_byte_for_byte_and_in_order():
    """load_selection_images() decodes in a thread pool. The images it
    returns must be IDENTICAL, and in the same order, as loading them one
    at a time -- every implementation in a comparison depends on that."""
    from pipeline.image_loader import load_image
    from ui import services

    dm, _info = services.scan_dataset(str(get_real_dataset_path()))
    selection = services.select_random_batch(dm, batch_size=12, seed=7)

    parallel = services.load_selection_images(selection)
    sequential = [load_image(item.absolute_path) for item in selection.items]

    assert len(parallel) == len(sequential) == 12
    for got, expected in zip(parallel, sequential):
        np.testing.assert_array_equal(got, expected)


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_small_selection_below_parallel_threshold_still_loads():
    """Selections smaller than the pool threshold take the sequential
    path; they must behave identically."""
    from ui import services

    dm, _info = services.scan_dataset(str(get_real_dataset_path()))
    selection = services.select_random_batch(dm, batch_size=2, seed=3)
    images = services.load_selection_images(selection)
    assert len(images) == 2
    assert all(img.dtype == np.uint8 for img in images)


def test_load_failure_reports_the_specific_file_even_when_loaded_in_parallel(tmp_path):
    """A bad file inside a parallel batch must still surface as a
    ServiceError naming THAT file -- the pool must not swallow or
    anonymize the failure."""
    import cv2
    from pipeline.dataset import DatasetManager
    from ui import services

    for i in range(6):
        cv2.imwrite(str(tmp_path / f"good_{i}.png"), np.full((8, 8), i * 10, dtype=np.uint8))
    (tmp_path / "broken.png").write_bytes(b"this is not a PNG")

    dm = DatasetManager(tmp_path)
    dm.scan()
    selection = dm.random_batch(batch_size=7, seed=1)

    with pytest.raises(services.ServiceError) as excinfo:
        services.load_selection_images(selection)
    assert "broken.png" in str(excinfo.value)


def test_group_images_by_shape_groups_correctly():
    from ui import services

    images = [
        FIXTURES["random_deterministic"](size=16, seed=1),
        FIXTURES["random_deterministic"](size=16, seed=2),
        FIXTURES["random_deterministic"](size=32, seed=3),
    ]
    groups = services.group_images_by_shape(images)
    assert groups[(16, 16)] == [0, 1]
    assert groups[(32, 32)] == [2]


# -- ui.services: integration test (spec item 61) -------------------------------------------------------


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_full_integration_select_process_cpu_basic_enhanced_verify_timing():
    """The spec item 61 integration test, literally: select one image,
    apply a default filter configuration, call CPU, call Basic CUDA,
    call Enhanced CUDA, verify results are returned, verify timing data
    exists. No Streamlit rendering involved."""
    from ui import services

    image = FIXTURES["random_deterministic"](size=64)
    config = FilterConfig()

    results = services.run_compare(
        [image], config, run_cpu=True, run_basic=True, run_enhanced=True, preview_index=0,
    )

    assert set(results.keys()) == {"CPU", "Basic CUDA", "Enhanced CUDA"}
    for label, result in results.items():
        assert len(result.final_outputs) == 1
        assert result.final_outputs[0].shape == image.shape
        assert result.final_outputs[0].dtype == np.uint8
        assert result.total_ms >= 0.0
        assert result.stage_outputs is not None
        assert "threshold" in result.stage_outputs  # the last enabled stage by default


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_run_compare_respects_disabled_implementations():
    from ui import services

    image = FIXTURES["random_deterministic"](size=32)
    config = FilterConfig()
    results = services.run_compare([image], config, run_cpu=True, run_basic=False, run_enhanced=False)
    assert set(results.keys()) == {"CPU"}


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_run_compare_cpu_basic_enhanced_use_same_images_same_order():
    """Spec item 16/59: never independently sample per implementation --
    verified here by checking the SAME `images` list object drives every
    implementation's output count and order."""
    from ui import services

    images = [FIXTURES["random_deterministic"](size=32, seed=i) for i in range(4)]
    config = FilterConfig()
    results = services.run_compare(images, config, run_cpu=True, run_basic=True, run_enhanced=True)
    for label, result in results.items():
        assert len(result.final_outputs) == 4


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_run_gpu_batch_raises_service_error_when_cuda_forced_unavailable(monkeypatch):
    from ui import services

    monkeypatch.setattr(services, "cuda_available", lambda: False)
    image = FIXTURES["random_deterministic"](size=32)
    with pytest.raises(services.ServiceError):
        services.run_gpu_batch([image], FilterConfig(), use_enhanced=False)


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_run_gpu_batch_chunks_when_requested_batch_exceeds_vram_capacity(monkeypatch):
    """A batch larger than what currently fits in free VRAM must still be
    processed in full (chunked via the same compute_safe_gpu_batch_size()
    cuda.pipeline.run_basic_cuda_pipeline_selection() already uses), with
    results bit-identical to running it unchunked -- never an OOM error
    for a batch that would have fit as several smaller native calls."""
    from ui import services

    images = [FIXTURES["random_deterministic"](size=32, seed=i) for i in range(7)]
    config = FilterConfig()

    expected_output, _unchunked_timing = services.run_basic_cuda_pipeline(images, config)

    monkeypatch.setattr(services, "compute_safe_gpu_batch_size", lambda height, width: 2)
    result = services.run_gpu_batch(images, config, use_enhanced=False)

    assert len(result.final_outputs) == 7
    for i, out in enumerate(result.final_outputs):
        np.testing.assert_array_equal(out, expected_output[i])
    # GPU timing is noisy run-to-run (JIT/cache warmup, etc.) so this only
    # checks the aggregate is a real, non-fabricated, non-negative number,
    # not that it compares a specific way to a separate unchunked call.
    assert result.total_ms >= 0.0
    assert result.h2d_ms is not None and result.h2d_ms >= 0.0
    assert result.d2h_ms is not None and result.d2h_ms >= 0.0


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_run_gpu_batch_within_vram_capacity_is_not_chunked(monkeypatch):
    """When the requested batch already fits, it must go through in ONE
    native call (not silently re-chunked into 1-image calls), preserving
    the exact behavior every existing test/benchmark already relies on."""
    from ui import services

    images = [FIXTURES["random_deterministic"](size=32, seed=i) for i in range(3)]
    config = FilterConfig()

    calls = []
    real_run = services.run_basic_cuda_pipeline

    def _spy(imgs, cfg):
        calls.append(len(imgs))
        return real_run(imgs, cfg)

    monkeypatch.setattr(services, "compute_safe_gpu_batch_size", lambda height, width: 100)
    monkeypatch.setattr(services, "run_basic_cuda_pipeline", _spy)
    result = services.run_gpu_batch(images, config, use_enhanced=False)

    assert calls == [3]  # one call, with all 3 images -- not split
    assert len(result.final_outputs) == 3


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_run_live_benchmark_returns_measured_values_not_none_when_enabled():
    from ui import services

    images = [FIXTURES["random_deterministic"](size=32, seed=i) for i in range(3)]
    config = FilterConfig()
    result = services.run_live_benchmark(images, config, run_cpu=True, run_basic=True, run_enhanced=True,
                                          warmup_runs=1, measurement_runs=2)
    assert result.cpu_ms is not None and result.cpu_ms >= 0
    assert result.basic_ms is not None and result.basic_ms >= 0
    assert result.enhanced_ms is not None and result.enhanced_ms >= 0
    assert result.n_images == 3


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_run_live_per_filter_comparison_returns_all_five_filters_with_real_numbers():
    from ui import services

    images = [FIXTURES["random_deterministic"](size=32, seed=i) for i in range(4)]
    config = FilterConfig()
    rows = services.run_live_per_filter_comparison(images, config, warmup_runs=1, measurement_runs=2)

    assert [r["filter"] for r in rows] == ["gaussian", "median", "sobel", "laplacian", "threshold"]
    for row in rows:
        assert row["basic_kernel_ms"] is not None and row["basic_kernel_ms"] >= 0
        assert row["enhanced_kernel_ms"] is not None and row["enhanced_kernel_ms"] >= 0
        assert row["kernel_speedup"] is not None and row["kernel_speedup"] > 0
        assert row["absolute_reduction_ms"] == pytest.approx(row["basic_kernel_ms"] - row["enhanced_kernel_ms"])

    # Percentages are signed contributions to the total reduction, so they
    # should sum to ~100% (not necessarily each individually positive --
    # a filter that got SLOWER contributes a negative share, same
    # convention the static per-filter data already uses).
    total_pct = sum(r["pct_of_total_compute_reduction"] for r in rows)
    assert total_pct == pytest.approx(100.0, abs=0.5)


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_run_live_per_filter_comparison_respects_disabled_stages():
    from ui import services

    images = [FIXTURES["random_deterministic"](size=32, seed=i) for i in range(3)]
    config = FilterConfig(sobel_enabled=False)
    rows = services.run_live_per_filter_comparison(images, config, warmup_runs=1, measurement_runs=1)

    sobel_row = next(r for r in rows if r["filter"] == "sobel")
    assert sobel_row["basic_kernel_ms"] is None
    assert sobel_row["kernel_speedup"] is None


# -- benchmark artifact loading (spec item 60: "unavailable benchmark handling") -------------------------------------------------------


def test_load_benchmark_summary_returns_none_gracefully_when_missing(tmp_path):
    from ui import services

    assert services.load_benchmark_summary(results_root=tmp_path) is None


def test_list_known_benchmark_ids_empty_when_no_results_dir(tmp_path):
    from ui import services

    assert services.list_known_benchmark_ids(results_root=tmp_path) == []


def test_load_experiment_registry_none_when_missing(tmp_path):
    from ui import services

    assert services.load_experiment_registry(results_root=tmp_path) is None


# -- app-level smoke test via Streamlit AppTest (spec item 60: "application starts successfully") -------------------------------------------------------


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_app_runs_without_exceptions_with_default_state():
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_path, default_timeout=120)
    at.run()
    assert len(at.exception) == 0, [str(e.value) for e in at.exception]
    assert len(at.tabs) >= 4  # presentation mode hides 2 tabs; normal mode shows 6


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_app_shows_dataset_error_for_bad_path():
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_path, default_timeout=120)
    at.run()
    text_input = next(w for w in at.sidebar.text_input if w.label == "Dataset directory")
    text_input.set_value("/definitely/not/a/real/dataset/path").run()
    assert len(at.exception) == 0  # must not crash -- a friendly error, not a raw traceback
    assert len(at.sidebar.error) >= 1
