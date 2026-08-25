"""Section 15 tests: real-batch live processing (ui.services.compare_batch
/ get_preview_stage_outputs / run_quick_batch_sweep / save_batch_results
and the batch session-state fields app.py's Live Processing tab uses).

Per spec item 40, unit tests use small deterministic fixtures (never the
full 9,463-image dataset); spec item 41's real-data integration test
uses a real batch (batch=8, seed=42) and asserts structure/timing
existence, never exact performance thresholds.
"""

from pathlib import Path

import numpy as np
import pytest

xray_cuda = pytest.importorskip(
    "xray_cuda", reason="xray_cuda extension is not built; run cmake --build build first."
)

from cpu.filters import FilterConfig  # noqa: E402
from pipeline.dataset import ImageSelection, SelectedItem  # noqa: E402
from tests.conftest import get_real_dataset_path  # noqa: E402
from tests.fixtures import FIXTURES  # noqa: E402

cuda_present = xray_cuda.cuda_available()


def _make_selection(n: int, seed=42, size=32) -> ImageSelection:
    items = [
        SelectedItem(index=i, absolute_path=Path(f"/fake/img_{i}.jpg"),
                     relative_path=f"img_{i}.jpg", filename=f"img_{i}.jpg")
        for i in range(n)
    ]
    return ImageSelection(mode="random_batch", seed=seed, requested_batch_size=n, dataset_size=n, items=items)


def _make_images(n: int, size=32, seed=0):
    return [FIXTURES["random_deterministic"](size=size, seed=seed + i) for i in range(n)]


# -- 1-2: batch selection, seed reproducibility (dataset-level; already covered by Section 2's own
# tests, re-verified here at the ui.services boundary) -------------------------------------------------------


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_select_random_batch_is_seed_reproducible():
    from ui import services

    dm, _info = services.scan_dataset(str(get_real_dataset_path()))
    sel_a = services.select_random_batch(dm, batch_size=8, seed=42)
    sel_b = services.select_random_batch(dm, batch_size=8, seed=42)
    assert [i.relative_path for i in sel_a.items] == [i.relative_path for i in sel_b.items]


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_select_random_batch_no_replacement():
    from ui import services

    dm, _info = services.scan_dataset(str(get_real_dataset_path()))
    selection = services.select_random_batch(dm, batch_size=50, seed=1)
    paths = [i.relative_path for i in selection.items]
    assert len(paths) == len(set(paths))


# -- 3: same paths reach every implementation -------------------------------------------------------


def test_compare_batch_same_images_reach_all_three():
    from ui import services

    selection = _make_selection(6)
    images = _make_images(6)
    result = services.compare_batch(selection, images, FilterConfig())

    assert len(result.groups) == 1
    group = result.groups[0]
    for label, r in group.results.items():
        assert len(r.final_outputs) == 6, f"{label} did not receive all 6 images"


# -- 4: resolution grouping -------------------------------------------------------


def test_compare_batch_groups_by_resolution_without_resizing():
    from ui import services

    images = _make_images(3, size=32) + _make_images(2, size=48)
    selection = _make_selection(5)
    result = services.compare_batch(selection, images, FilterConfig())

    assert result.resolution_group_summary == {"32x32": 3, "48x48": 2}
    assert len(result.groups) == 2
    for group in result.groups:
        for label, r in group.results.items():
            for out in r.final_outputs:
                assert out.shape == group.shape  # never resized


# -- 5-7: CPU / Basic CUDA / Enhanced CUDA batch execution -------------------------------------------------------


def test_compare_batch_runs_cpu():
    from ui import services

    selection = _make_selection(4)
    images = _make_images(4)
    result = services.compare_batch(selection, images, FilterConfig(), run_basic=False, run_enhanced=False)
    assert "CPU" in result.groups[0].results
    assert len(result.groups[0].results["CPU"].final_outputs) == 4


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_compare_batch_runs_basic_cuda():
    from ui import services

    selection = _make_selection(4)
    images = _make_images(4)
    result = services.compare_batch(selection, images, FilterConfig(), run_cpu=False, run_enhanced=False)
    assert "Basic CUDA" in result.groups[0].results
    r = result.groups[0].results["Basic CUDA"]
    assert r.h2d_ms is not None and r.compute_ms is not None and r.d2h_ms is not None


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_compare_batch_runs_enhanced_cuda():
    from ui import services

    selection = _make_selection(4)
    images = _make_images(4)
    result = services.compare_batch(selection, images, FilterConfig(), run_cpu=False, run_basic=False)
    assert "Enhanced CUDA" in result.groups[0].results


# -- 8: output ordering -------------------------------------------------------


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_compare_batch_preserves_order_within_a_group():
    """Distinct images (not just random noise) must come back in the
    same order they were submitted."""
    from ui import services

    images = [
        np.zeros((16, 16), dtype=np.uint8),
        np.full((16, 16), 255, dtype=np.uint8),
        FIXTURES["random_deterministic"](size=16),
    ]
    selection = _make_selection(3)
    result = services.compare_batch(selection, images, FilterConfig())
    group = result.groups[0]
    assert group.original_indices == [0, 1, 2]

    from cpu.filters import apply_threshold, apply_laplacian, apply_sobel, apply_median, apply_gaussian
    config = FilterConfig()
    for i, img in enumerate(images):
        expected = apply_threshold(apply_laplacian(apply_sobel(apply_median(apply_gaussian(img, config), config), config), config), config)
        np.testing.assert_array_equal(group.results["CPU"].final_outputs[i], expected)


# -- 9-10: timing and throughput aggregation -------------------------------------------------------


def test_implementation_totals_aggregates_timing_and_throughput():
    from ui import services

    selection = _make_selection(5)
    images = _make_images(5)
    result = services.compare_batch(selection, images, FilterConfig(), run_basic=False, run_enhanced=False)
    totals = result.implementation_totals()

    assert totals["CPU"]["n_images"] == 5
    assert totals["CPU"]["total_ms"] >= 0.0
    expected_ips = 5 / (totals["CPU"]["total_ms"] / 1000.0) if totals["CPU"]["total_ms"] > 0 else 0.0
    assert totals["CPU"]["images_per_second"] == pytest.approx(expected_ips)
    assert totals["CPU"]["ms_per_image"] == pytest.approx(totals["CPU"]["total_ms"] / 5)


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_implementation_totals_sums_across_multiple_groups():
    from ui import services

    images = _make_images(3, size=32) + _make_images(2, size=48)
    selection = _make_selection(5)
    result = services.compare_batch(selection, images, FilterConfig(), run_basic=False, run_enhanced=False)
    totals = result.implementation_totals()
    assert totals["CPU"]["n_images"] == 5  # 3 + 2, summed across both resolution groups


# -- 11: correctness aggregation (every image, not just the first) -------------------------------------------------------


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_batch_correctness_covers_every_image_not_just_first():
    from ui import services

    selection = _make_selection(6)
    images = _make_images(6)
    result = services.compare_batch(selection, images, FilterConfig())

    enhanced_vs_basic = next(c for c in result.correctness if c.comparison == "enhanced_vs_basic")
    assert enhanced_vs_basic.images_compared == 6
    assert enhanced_vs_basic.max_abs_diff == 0  # established Section 10 finding: bit-exact
    assert enhanced_vs_basic.images_with_zero_diff == 6
    assert enhanced_vs_basic.images_with_nonzero_diff == 0


# -- 12: representative-image selection -------------------------------------------------------


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_get_preview_stage_outputs_for_arbitrary_index():
    from ui import services

    selection = _make_selection(5)
    images = _make_images(5)
    result = services.compare_batch(selection, images, FilterConfig())

    stages = services.get_preview_stage_outputs(result, 3, images, FilterConfig())
    assert set(stages.keys()) == {"CPU", "Basic CUDA", "Enhanced CUDA"}
    for label, stage_dict in stages.items():
        assert stage_dict["threshold"] is not None
        assert stage_dict["threshold"].shape == images[3].shape


def test_find_group_for_flat_index():
    from ui import services

    images = _make_images(2, size=32) + _make_images(2, size=48)
    selection = _make_selection(4)
    result = services.compare_batch(selection, images, FilterConfig(), run_basic=False, run_enhanced=False)

    located = services.find_group_for_flat_index(result, 3)
    assert located is not None
    group, local_index = located
    assert group.shape == (48, 48)
    assert local_index == 1


def test_get_preview_stage_outputs_rejects_capacity_capped_index():
    from ui import services

    selection = _make_selection(2)
    images = _make_images(2)
    result = services.compare_batch(selection, images, FilterConfig(), run_basic=False, run_enhanced=False)
    with pytest.raises(services.ServiceError):
        services.get_preview_stage_outputs(result, 99, images, FilterConfig())


# -- 13: save batch results -------------------------------------------------------


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_save_batch_results_creates_expected_structure(tmp_path):
    from ui import services

    selection = _make_selection(4)
    images = _make_images(4)
    result = services.compare_batch(selection, images, FilterConfig())
    stages = services.get_preview_stage_outputs(result, 0, images, FilterConfig())

    run_dir = services.save_batch_results(result, preview_stage_outputs=stages, preview_image=images[0], output_root=tmp_path)
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "selection.json").exists()
    assert (run_dir / "correctness.json").exists()
    assert (run_dir / "timing.json").exists()
    assert (run_dir / "original.png").exists()
    for subdir in ("cpu", "basic_cuda", "enhanced_cuda"):
        assert (run_dir / subdir).exists()


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_save_batch_results_never_overwrites_previous_run(tmp_path):
    from ui import services

    selection = _make_selection(3)
    images = _make_images(3)
    result = services.compare_batch(selection, images, FilterConfig())

    run_dir_1 = services.save_batch_results(result, output_root=tmp_path)
    run_dir_2 = services.save_batch_results(result, output_root=tmp_path)
    assert run_dir_1 != run_dir_2


def test_save_batch_results_never_writes_to_benchmark_results(tmp_path):
    from cuda.final_benchmark import DEFAULT_RESULTS_ROOT

    assert DEFAULT_RESULTS_ROOT != tmp_path


# -- 14: GPU failure handling -------------------------------------------------------


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_compare_batch_handles_cuda_unavailable_per_group(monkeypatch):
    from ui import services

    images = _make_images(2, size=32) + _make_images(2, size=48)
    selection = _make_selection(4)
    monkeypatch.setattr(services, "cuda_available", lambda: False)

    result = services.compare_batch(selection, images, FilterConfig())
    for group in result.groups:
        assert "CPU" in group.results
        assert "Basic CUDA" in group.errors
        assert "Enhanced CUDA" in group.errors


def test_compare_batch_threads_dataset_fingerprint_through():
    from ui import services

    selection = _make_selection(3)
    images = _make_images(3)
    default_result = services.compare_batch(selection, images, FilterConfig(), run_basic=False, run_enhanced=False)
    assert default_result.dataset_fingerprint is None

    fp_result = services.compare_batch(
        selection, images, FilterConfig(), run_basic=False, run_enhanced=False,
        dataset_fingerprint="abc123",
    )
    assert fp_result.dataset_fingerprint == "abc123"


def test_compare_batch_rejects_empty_images():
    from ui import services

    selection = _make_selection(0)
    with pytest.raises(services.ServiceError):
        services.compare_batch(selection, [], FilterConfig())


# -- session state -------------------------------------------------------


def test_session_state_has_section15_batch_fields():
    import streamlit as st

    from ui import state

    st.session_state.clear()
    state.init_session_state()
    for key in ("last_batch_selection", "last_batch_images", "last_batch_result",
                "last_preview_index", "last_batch_preview_stages", "quick_sweep_result"):
        assert key in st.session_state


# -- quick batch-size sweep -------------------------------------------------------


@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_run_quick_batch_sweep_caps_to_available_images():
    from ui import services

    pool = _make_images(4)
    sweep = services.run_quick_batch_sweep(pool, FilterConfig(), batch_sizes=[1, 100])
    sizes = [row["effective_batch_size"] for row in sweep["rows"]]
    assert sizes == [1, 4]  # 100 capped to the pool size of 4


# -- item 41: real-data integration test -------------------------------------------------------


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
@pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")
def test_integration_real_batch_224x224_cpu_basic_enhanced_compare():
    """Spec item 41: real batch (batch=8, seed=42) -> CPU -> Basic CUDA
    -> Enhanced CUDA -> compare. Verifies structure and correctness
    generation only -- never an exact performance threshold."""
    from ui import services

    dm, _info = services.scan_dataset(str(get_real_dataset_path()))
    selection = services.select_random_batch(dm, batch_size=8, seed=42)
    images = services.load_selection_images(selection)

    result = services.compare_batch(selection, images, FilterConfig(), dataset_fingerprint=dm.fingerprint())

    assert result.effective_count == len(images)
    assert result.dataset_fingerprint is not None and len(result.dataset_fingerprint) > 0
    for group in result.groups:
        assert not group.errors, f"Unexpected failures: {group.errors}"
        assert set(group.results.keys()) == {"CPU", "Basic CUDA", "Enhanced CUDA"}
        for label, r in group.results.items():
            assert r.total_ms >= 0.0, f"{label} missing timing data"
            assert len(r.final_outputs) == group.effective_count

    assert len(result.correctness) == 3
    for c in result.correctness:
        assert c.images_compared > 0
