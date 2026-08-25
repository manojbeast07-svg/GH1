"""Section 18 tests: verifies the GPU implementation is native C++/CUDA
end to end (the migration audit found nothing to migrate -- this file
proves that finding rather than just asserting it).

Per spec item 30's 9 enumerated areas, plus item 37's repeated-execution
memory-stability check. Uses the real compiled xray_cuda extension and
a real CUDA device throughout (skipped automatically when unavailable).
"""

import gc
from pathlib import Path

import numpy as np
import pytest

xray_cuda = pytest.importorskip(
    "xray_cuda", reason="xray_cuda extension is not built; run cmake --build build first."
)

from cpu.filters import FilterConfig, apply_gaussian, apply_laplacian, apply_median, apply_sobel, apply_threshold  # noqa: E402
from cuda.gaussian import gaussian_kernel_1d, gaussian_kernel_2d  # noqa: E402
from cuda.laplacian import laplacian_kernel_2d  # noqa: E402
from cuda.pipeline import run_basic_cuda_pipeline, run_enhanced_cuda_pipeline, run_cuda_pipeline, PipelineConfig  # noqa: E402
from tests.conftest import get_real_dataset_path  # noqa: E402
from tests.fixtures import FIXTURES  # noqa: E402

cuda_present = xray_cuda.cuda_available()
pytestmark = pytest.mark.skipif(not cuda_present, reason="No usable CUDA device detected on this machine.")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _images(n=4, size=32, seed=0):
    return [FIXTURES["random_deterministic"](size=size, seed=seed + i) for i in range(n)]


# -- 1-2: Basic / Enhanced backend is native -------------------------------------------------------


def test_extension_is_a_compiled_binary_not_python():
    ext_path = Path(xray_cuda.__file__).resolve()
    assert ext_path.suffix in (".pyd", ".so")
    assert ext_path.exists()


def test_basic_pipeline_dispatches_to_the_compiled_extension_module():
    fn = xray_cuda.run_basic_cuda_pipeline_gpu
    assert getattr(fn, "__module__", None) == "xray_cuda"


def test_all_cu_source_files_exist_and_are_referenced_by_cmake():
    cu_files = sorted((PROJECT_ROOT / "cuda" / "src").glob("*.cu"))
    assert len(cu_files) > 0
    cmakelists = (PROJECT_ROOT / "cuda" / "CMakeLists.txt").read_text(encoding="utf-8")
    for cu_file in cu_files:
        assert cu_file.name in cmakelists, f"{cu_file.name} is not referenced in cuda/CMakeLists.txt"


def test_no_python_gpu_computation_libraries_in_cuda_package():
    import ast

    forbidden = {"cupy", "numba", "torch", "pycuda"}
    for py_file in (PROJECT_ROOT / "cuda").glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in forbidden, f"{py_file.name} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden, f"{py_file.name} imports from {node.module}"


# -- 3-4: Basic / Enhanced output correctness -------------------------------------------------------


def _differing_pct(out: np.ndarray, expected: np.ndarray) -> float:
    diff = np.abs(out.astype(np.int16) - expected.astype(np.int16))
    return 100.0 * np.count_nonzero(diff) / diff.size


def test_basic_pipeline_output_matches_cpu_reference():
    """Full 5-stage pipeline output vs. CPU: max_abs_diff can
    legitimately hit 255 (Threshold can amplify a tiny upstream
    Gaussian +/-1 difference into a 0<->255 flip on a handful of
    pixels -- the established project finding), so the differing-pixel
    PERCENTAGE is the meaningful measure, not max_abs_diff."""
    images = _images(4)
    config = FilterConfig()
    out, _timing = run_basic_cuda_pipeline(images, config)
    expected = np.stack([
        apply_threshold(apply_laplacian(apply_sobel(apply_median(apply_gaussian(img, config), config), config), config), config)
        for img in images
    ])
    assert _differing_pct(out, expected) < 1.0


def test_enhanced_pipeline_output_matches_cpu_reference():
    images = _images(4)
    config = FilterConfig()
    out, _timing = run_enhanced_cuda_pipeline(images, config)
    expected = np.stack([
        apply_threshold(apply_laplacian(apply_sobel(apply_median(apply_gaussian(img, config), config), config), config), config)
        for img in images
    ])
    assert _differing_pct(out, expected) < 1.0


def test_enhanced_bit_exact_vs_basic():
    """Established project finding (Section 10): Enhanced vs Basic is
    bit-exact for the production configuration."""
    images = _images(4)
    config = FilterConfig()
    basic_out, _ = run_basic_cuda_pipeline(images, config)
    enhanced_out, _ = run_enhanced_cuda_pipeline(images, config)
    assert np.array_equal(basic_out, enhanced_out)


# -- 5: GPU timing exists -------------------------------------------------------


def test_pipeline_timing_is_populated_from_native_cuda_events():
    images = _images(4)
    config = FilterConfig()
    _out, timing = run_basic_cuda_pipeline(images, config)
    assert timing.h2d_ms >= 0.0
    assert timing.d2h_ms >= 0.0
    assert timing.gaussian_ms is not None and timing.gaussian_ms >= 0.0
    assert timing.median_ms is not None and timing.median_ms >= 0.0
    assert timing.sobel_ms is not None and timing.sobel_ms >= 0.0
    assert timing.laplacian_ms is not None and timing.laplacian_ms >= 0.0
    assert timing.threshold_ms is not None and timing.threshold_ms >= 0.0
    assert timing.total_ms == pytest.approx(timing.h2d_ms + timing.compute_ms + timing.d2h_ms, rel=1e-4)


def test_disabled_stage_timing_is_none_not_zero():
    images = _images(2)
    config = FilterConfig(median_enabled=False)
    _out, timing = run_basic_cuda_pipeline(images, config)
    assert timing.median_ms is None
    assert timing.gaussian_ms is not None


# -- 6: GPU-resident pipeline (no GPU->CPU->GPU between stages) -------------------------------------------------------


def test_pipeline_is_gpu_resident_single_h2d_single_d2h():
    """Indirect proof: run the full 5-stage pipeline and a single-stage
    pipeline on the same batch; if stages round-tripped through the
    host between filters, H2D+D2H time would scale with stage count.
    Instead it should stay roughly constant (one upload + one download
    regardless of how many stages ran).

    Warms up both configurations first and takes the median of several
    runs -- first-call CUDA context/driver overhead otherwise makes this
    heuristic noticeably flaky (observed: passes reliably alone, can
    occasionally fail inside the full suite where it's not the first
    CUDA call in the process). The deterministic proof is
    test_source_confirms_ping_pong_buffers_no_intermediate_host_copy()
    below; this test is a supporting empirical sanity check, not the
    primary proof, so a generous threshold + warmup is appropriate."""
    images = _images(8, size=64)
    full_config = FilterConfig()
    single_config = FilterConfig(median_enabled=False, sobel_enabled=False, laplacian_enabled=False, threshold_enabled=False)

    for _ in range(3):  # warmup: absorb first-call CUDA context/driver jitter
        run_basic_cuda_pipeline(images, full_config)
        run_basic_cuda_pipeline(images, single_config)

    full_h2d = sorted(run_basic_cuda_pipeline(images, full_config)[1].h2d_ms for _ in range(5))
    single_h2d = sorted(run_basic_cuda_pipeline(images, single_config)[1].h2d_ms for _ in range(5))
    median_full, median_single = full_h2d[len(full_h2d) // 2], single_h2d[len(single_h2d) // 2]

    # H2D transfers ~batch_size*H*W bytes regardless of stage count -- allow generous
    # tolerance for measurement jitter, but reject an obviously-scaling (5x) relationship.
    h2d_ratio = median_full / median_single if median_single > 0 else 1.0
    assert h2d_ratio < 5.0, f"H2D time scaled with stage count ({h2d_ratio:.2f}x) -- suggests per-stage host round trips"


def test_source_confirms_ping_pong_buffers_no_intermediate_host_copy():
    """Direct proof by inspecting the native pipeline source: exactly
    one upload_from_host / one download_to_host call, everything else
    operates on device pointers (current->data()/other->data())."""
    source = (PROJECT_ROOT / "cuda" / "src" / "pipeline_basic.cu").read_text(encoding="utf-8")
    assert source.count("upload_from_host") == 1
    assert source.count("download_to_host") == 1


# -- 7: existing batch behavior works -------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_batch_output_ordering_preserved(batch_size):
    images = [np.full((32, 32), fill_value=i * 10, dtype=np.uint8) for i in range(batch_size)]
    config = FilterConfig(gaussian_enabled=False, median_enabled=False, sobel_enabled=False, laplacian_enabled=False)
    out, _timing = run_basic_cuda_pipeline(images, config)
    assert out.shape[0] == batch_size
    for i, img in enumerate(images):
        expected = apply_threshold(img, config)
        np.testing.assert_array_equal(out[i], expected)


# -- 8-9: memory cleanup + repeated execution -------------------------------------------------------


def test_repeated_execution_10x_stable_and_correct():
    images = _images(4)
    config = FilterConfig()
    outputs = []
    for _ in range(10):
        out, timing = run_basic_cuda_pipeline(images, config)
        outputs.append(out)
        assert timing.total_ms > 0
    for out in outputs[1:]:
        np.testing.assert_array_equal(out, outputs[0])  # deterministic: identical input -> identical output every time


def test_memory_stable_across_200_iterations():
    """Spec item 37: 200 iterations of upload -> Basic -> Enhanced ->
    download -> release. No crash, no growing device allocation, no
    correctness degradation."""
    images = _images(4, size=64)
    config = FilterConfig()

    free_before, total = xray_cuda.device_memory_info()["free_bytes"], xray_cuda.device_memory_info()["total_bytes"]
    reference_basic, _ = run_basic_cuda_pipeline(images, config)
    reference_enhanced, _ = run_enhanced_cuda_pipeline(images, config)

    for i in range(200):
        basic_out, _ = run_basic_cuda_pipeline(images, config)
        enhanced_out, _ = run_enhanced_cuda_pipeline(images, config)
        if i % 50 == 0:  # spot-check correctness periodically, not every iteration (keep the test fast)
            np.testing.assert_array_equal(basic_out, reference_basic)
            np.testing.assert_array_equal(enhanced_out, reference_enhanced)

    gc.collect()
    free_after = xray_cuda.device_memory_info()["free_bytes"]
    leaked_bytes = free_before - free_after
    # Generous threshold (32 MB) -- catches a real per-call leak (which would be gigabytes
    # after 400 pipeline calls) without failing on ordinary allocator fragmentation/caching.
    assert leaked_bytes < 32 * 1024 * 1024, (
        f"Possible GPU memory leak: {leaked_bytes / (1024*1024):.1f} MB less free after 200 iterations "
        f"(free_before={free_before}, free_after={free_after}, total={total})"
    )


def test_pipeline_config_variant_selection_repeated_execution():
    """Repeated execution through the general PipelineConfig/
    run_cuda_pipeline() path (used by the Optimization Lab and mixed
    Basic/Enhanced configurations), not just the two named presets."""
    images = _images(4)
    config = FilterConfig()
    pipeline_config = PipelineConfig(gaussian_variant="specialized", median_variant="basic",
                                      sobel_variant="specialized", laplacian_variant="basic",
                                      threshold_variant="vectorized")
    outputs = [run_cuda_pipeline(images, config, pipeline_config)[0] for _ in range(5)]
    for out in outputs[1:]:
        np.testing.assert_array_equal(out, outputs[0])


# -- real X-ray verification (spec item 31) -------------------------------------------------------


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_integration_real_224x224_and_large_image_cpu_basic_enhanced_agree():
    from pipeline.dataset import DatasetManager
    from pipeline.image_loader import load_image

    dm = DatasetManager(get_real_dataset_path())
    dm.scan()

    # 224x224 (the dominant resolution) and at least one larger image.
    selection = dm.random_batch(batch_size=200, seed=42)
    images_by_shape = {}
    for p in selection.selected_paths:
        img = load_image(p)
        images_by_shape.setdefault(img.shape, []).append(img)

    shape_224 = next((s for s in images_by_shape if s == (224, 224)), None)
    assert shape_224 is not None, "expected at least one 224x224 image in this dataset sample"
    large_shape = max((s for s in images_by_shape if s != (224, 224)), key=lambda s: s[0] * s[1], default=None)

    config = FilterConfig()
    for shape in [shape_224] + ([large_shape] if large_shape else []):
        imgs = images_by_shape[shape][:4]
        cpu_expected = np.stack([
            apply_threshold(apply_laplacian(apply_sobel(apply_median(apply_gaussian(img, config), config), config), config), config)
            for img in imgs
        ])
        basic_out, basic_timing = run_basic_cuda_pipeline(imgs, config)
        enhanced_out, enhanced_timing = run_enhanced_cuda_pipeline(imgs, config)

        # Full-pipeline diff vs CPU: max_abs_diff can legitimately hit 255 (Threshold can
        # amplify a tiny upstream Gaussian +/-1 difference into a 0<->255 flip on a handful
        # of pixels -- the established, documented project finding). The differing-PIXEL
        # PERCENTAGE is the meaningful measure, matching every other full-pipeline
        # correctness check in this project.
        diff = np.abs(basic_out.astype(np.int16) - cpu_expected.astype(np.int16))
        differing_pct = 100.0 * np.count_nonzero(diff) / diff.size
        assert differing_pct < 1.0, f"{differing_pct:.4f}% of pixels differ from CPU for shape {shape} -- too high"
        assert np.array_equal(basic_out, enhanced_out)
        assert basic_timing.total_ms > 0
        assert enhanced_timing.total_ms > 0
