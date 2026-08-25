"""Section 10 tests: Enhanced CUDA binary threshold -- two variants
(uchar4 vectorized, multi-pixel-per-thread scalar), each required to
match CPU and Basic CUDA EXACTLY (max_abs_diff == 0). Also covers the
experimental fused Laplacian+Threshold kernel.

--------------------------------------------------------------------------
Correctness standard: exact match, no tolerance
--------------------------------------------------------------------------
Threshold is a single uint8 comparison, no floating-point arithmetic
anywhere -- max_abs_diff == 0 is the only acceptable outcome, exactly
the same standard test_cuda_threshold.py already established for Basic.

--------------------------------------------------------------------------
Why non-4-aligned widths get dedicated coverage
--------------------------------------------------------------------------
The Vectorized variant only uses uchar4 loads when width % 4 == 0;
otherwise threshold_enhanced_dispatch() automatically falls back to an
equivalent scalar kernel (Section 10 spec item 7's explicit safety
requirement -- the real dataset has widths like 1733 that are NOT
4-aligned). This suite exercises both paths directly, including the
real 1733 width, rather than assuming the fallback is exercised
incidentally.
"""

from pathlib import Path

import numpy as np
import pytest

xray_cuda = pytest.importorskip(
    "xray_cuda", reason="xray_cuda extension is not built; run cmake --build build first."
)

from cpu.filters import FilterConfig, apply_laplacian, apply_threshold  # noqa: E402
from cuda.laplacian import laplacian_kernel_2d  # noqa: E402
from cuda.threshold import (  # noqa: E402
    THRESHOLD_VARIANTS,
    threshold_cuda,
    threshold_enhanced_cuda,
    threshold_enhanced_cuda_gpu,
    threshold_variant_to_int,
)
from pipeline.dataset import DatasetManager  # noqa: E402
from pipeline.image_loader import load_image  # noqa: E402
from tests.conftest import get_real_dataset_path, stable_seed  # noqa: E402
from tests.fixtures import FIXTURES  # noqa: E402

cuda_present = xray_cuda.cuda_available()
pytestmark = pytest.mark.skipif(
    not cuda_present, reason="No usable CUDA device detected on this machine."
)

REFERENCE_DIR = Path(__file__).resolve().parent / "reference"
THRESHOLDS = [0, 1, 127, 128, 254, 255]

# Widths deliberately spanning both the fast (4-aligned) and safe
# (fallback) paths -- 1733 is a real dataset width (Section 2 finding).
WIDTHS = [224, 1733, 15, 33, 4, 5]


def _assert_exact_match(image, threshold_value=128, max_value=255, variant="vectorized"):
    config = FilterConfig(threshold_value=threshold_value, threshold_max_value=max_value)
    cpu_out = apply_threshold(image, config)
    basic_out = threshold_cuda(image, threshold_value=threshold_value, max_value=max_value)
    enhanced_out = threshold_enhanced_cuda(image, threshold_value=threshold_value, max_value=max_value, variant=variant)

    assert enhanced_out.dtype == np.uint8
    assert enhanced_out.shape == image.shape

    diff_cpu = np.abs(cpu_out.astype(np.int16) - enhanced_out.astype(np.int16))
    assert diff_cpu.max() == 0, (
        f"variant={variant} threshold={threshold_value} max_value={max_value}: "
        f"vs CPU max_abs_diff={diff_cpu.max()} (expected exact match)"
    )
    diff_basic = np.abs(basic_out.astype(np.int16) - enhanced_out.astype(np.int16))
    assert diff_basic.max() == 0, (
        f"variant={variant} threshold={threshold_value} max_value={max_value}: "
        f"vs Basic CUDA max_abs_diff={diff_basic.max()} (expected exact match)"
    )


# -- correctness: all variants x fixtures x thresholds -------------------------------------------------------


@pytest.mark.parametrize("fixture_name", list(FIXTURES.keys()))
@pytest.mark.parametrize("threshold_value", THRESHOLDS)
@pytest.mark.parametrize("variant", THRESHOLD_VARIANTS)
def test_exact_match_on_synthetic_fixtures(fixture_name, threshold_value, variant):
    image = FIXTURES[fixture_name]()
    _assert_exact_match(image, threshold_value=threshold_value, variant=variant)


@pytest.mark.parametrize("max_value", [1, 100, 200, 255])
@pytest.mark.parametrize("variant", THRESHOLD_VARIANTS)
def test_exact_match_various_max_values(max_value, variant):
    image = FIXTURES["random_deterministic"]()
    _assert_exact_match(image, threshold_value=128, max_value=max_value, variant=variant)


# -- equality-boundary tests -------------------------------------------------------


@pytest.mark.parametrize("threshold_value", [0, 1, 50, 127, 128, 200, 254, 255])
@pytest.mark.parametrize("variant", THRESHOLD_VARIANTS)
def test_equality_boundary_exact(threshold_value, variant):
    """threshold-1, threshold, threshold+1 in one tiny image -- the
    classic off-by-one trap for a threshold implementation."""
    lo = max(0, threshold_value - 1)
    hi = min(255, threshold_value + 1)
    image = np.array([[lo, threshold_value, hi]], dtype=np.uint8)
    _assert_exact_match(image, threshold_value=threshold_value, max_value=255, variant=variant)


@pytest.mark.parametrize("variant", THRESHOLD_VARIANTS)
def test_threshold_value_128_boundary_explicit(variant):
    image = np.array([[126, 127, 128, 129, 130]], dtype=np.uint8)
    out = threshold_enhanced_cuda(image, threshold_value=128, max_value=255, variant=variant)
    np.testing.assert_array_equal(out, np.array([[0, 0, 0, 255, 255]], dtype=np.uint8))


@pytest.mark.parametrize("variant", THRESHOLD_VARIANTS)
def test_threshold_value_zero_edge_case(variant):
    image = np.array([[0, 1, 2, 254, 255]], dtype=np.uint8)
    out = threshold_enhanced_cuda(image, threshold_value=0, max_value=255, variant=variant)
    np.testing.assert_array_equal(out, np.array([[0, 255, 255, 255, 255]], dtype=np.uint8))


@pytest.mark.parametrize("variant", THRESHOLD_VARIANTS)
def test_threshold_value_255_maps_everything_to_zero(variant):
    image = np.array([[0, 1, 254, 255]], dtype=np.uint8)
    out = threshold_enhanced_cuda(image, threshold_value=255, max_value=255, variant=variant)
    np.testing.assert_array_equal(out, np.array([[0, 0, 0, 0]], dtype=np.uint8))


# -- width alignment: Vectorized's fast path vs. its automatic scalar fallback -------------------------------------------------------


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("variant", THRESHOLD_VARIANTS)
def test_exact_match_various_widths(width, variant):
    rng = np.random.default_rng(stable_seed(width, variant))
    image = rng.integers(0, 256, size=(9, width), dtype=np.uint8)
    _assert_exact_match(image, variant=variant)


def test_vectorized_fast_path_and_fallback_agree():
    """Same logical threshold applied to a 4-aligned and a non-aligned
    image must both be exact vs CPU -- verifies the fallback isn't
    silently wrong, not just that SOME output is produced."""
    rng = np.random.default_rng(99)
    image_aligned = rng.integers(0, 256, size=(9, 224), dtype=np.uint8)   # 224 % 4 == 0
    image_unaligned = rng.integers(0, 256, size=(9, 1733), dtype=np.uint8)  # real dataset width, 1733 % 4 == 1
    assert 224 % 4 == 0
    assert 1733 % 4 != 0
    _assert_exact_match(image_aligned, variant="vectorized")
    _assert_exact_match(image_unaligned, variant="vectorized")


# -- real X-rays -------------------------------------------------------


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
@pytest.mark.parametrize("threshold_value", THRESHOLDS)
@pytest.mark.parametrize("variant", THRESHOLD_VARIANTS)
def test_exact_match_on_real_224x224_xray(threshold_value, variant):
    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=1, seed=42)
    image = load_image(selection.selected_paths[0])
    assert image.shape == (224, 224)
    _assert_exact_match(image, threshold_value=threshold_value, variant=variant)


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
@pytest.mark.parametrize("variant", THRESHOLD_VARIANTS)
def test_exact_match_on_large_real_xray(variant):
    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    large_path = None
    for path in dm.paths[:2000]:
        image = load_image(path)
        if image.shape != (224, 224) and image.shape[0] * image.shape[1] > 500_000:
            large_path = path
            break
    if large_path is None:
        pytest.skip("No sufficiently large non-224x224 image found in the first 2000 scanned files.")
    image = load_image(large_path)
    _assert_exact_match(image, variant=variant)


# -- frozen reference regression -------------------------------------------------------


@pytest.mark.parametrize("fixture_name", list(FIXTURES.keys()))
@pytest.mark.parametrize("variant", THRESHOLD_VARIANTS)
def test_matches_frozen_cpu_reference_output(fixture_name, variant):
    ref_path = REFERENCE_DIR / fixture_name / "threshold.npy"
    if not ref_path.exists():
        pytest.skip(f"No frozen reference at {ref_path}")
    expected = np.load(ref_path)
    image = FIXTURES[fixture_name]()
    gpu_out = threshold_enhanced_cuda(image, threshold_value=128, max_value=255, variant=variant)
    np.testing.assert_array_equal(gpu_out, expected)


# -- determinism -------------------------------------------------------


@pytest.mark.parametrize("variant", THRESHOLD_VARIANTS)
def test_repeated_execution_is_deterministic(variant):
    image = FIXTURES["random_deterministic"]()
    first = threshold_enhanced_cuda(image, variant=variant)
    for _ in range(5):
        again = threshold_enhanced_cuda(image, variant=variant)
        np.testing.assert_array_equal(first, again)


# -- Basic-unchanged regression -------------------------------------------------------


def test_threshold_basic_output_unchanged_by_enhanced_addition():
    """threshold_basic must remain byte-for-byte identical to its
    pre-Section-10 behavior -- Enhanced Threshold is a separate
    implementation family."""
    image = FIXTURES["random_deterministic"]()
    config = FilterConfig()
    cpu_out = apply_threshold(image, config)
    basic_out = threshold_cuda(image)
    diff = np.abs(cpu_out.astype(np.int16) - basic_out.astype(np.int16))
    assert diff.max() == 0


# -- batched correctness -------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 8, 32])
@pytest.mark.parametrize("variant", THRESHOLD_VARIANTS)
def test_batched_exact_match(batch_size, variant):
    config = FilterConfig()
    images = [FIXTURES["random_deterministic"](size=32, seed=i) for i in range(batch_size)]
    batch = np.stack(images, axis=0)

    result = xray_cuda.threshold_enhanced_batch_gpu(batch, 128, 255, threshold_variant_to_int(variant), 16, 16)
    output = result["output"]
    assert output.shape == (batch_size, 32, 32)

    for i, image in enumerate(images):
        cpu_out = apply_threshold(image, config)
        diff = np.abs(cpu_out.astype(np.int16) - output[i].astype(np.int16))
        assert diff.max() == 0, f"batch index {i}: max_abs_diff={diff.max()}"


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
@pytest.mark.parametrize("variant", THRESHOLD_VARIANTS)
def test_batched_exact_match_real_125_images(variant):
    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=125, seed=42)
    images = [load_image(p) for p in selection.selected_paths if load_image(p).shape == (224, 224)]
    if len(images) < 2:
        pytest.skip("Not enough same-resolution images in this sample.")
    batch = np.stack(images, axis=0)

    config = FilterConfig()
    result = xray_cuda.threshold_enhanced_batch_gpu(batch, 128, 255, threshold_variant_to_int(variant), 16, 16)
    output = result["output"]

    cpu_outputs = np.stack([apply_threshold(img, config) for img in images])
    diff = np.abs(cpu_outputs.astype(np.int16) - output.astype(np.int16))
    assert diff.max() == 0, f"variant={variant}: max_abs_diff={diff.max()} across {len(images)} real images"


def test_batch_ordering_no_cross_contamination():
    images = [
        np.zeros((16, 16), dtype=np.uint8),
        np.full((16, 16), 255, dtype=np.uint8),
        FIXTURES["random_deterministic"](size=16),
    ]
    batch = np.stack(images, axis=0)
    result = xray_cuda.threshold_enhanced_batch_gpu(batch, 128, 255, threshold_variant_to_int("vectorized"), 16, 16)
    output = result["output"]

    config = FilterConfig()
    for i, image in enumerate(images):
        expected = apply_threshold(image, config)
        np.testing.assert_array_equal(output[i], expected)


# -- GPU-native API -------------------------------------------------------


@pytest.mark.parametrize("variant", THRESHOLD_VARIANTS)
def test_threshold_enhanced_cuda_gpu_chains_without_host_round_trip(variant):
    image = FIXTURES["random_deterministic"]()
    gpu_image = xray_cuda.upload_image(image)
    result = threshold_enhanced_cuda_gpu(gpu_image, variant=variant)
    assert "output" in result and "kernel_ms" in result
    assert result["kernel_ms"] >= 0.0

    downloaded = xray_cuda.download_image(result["output"])
    expected = threshold_enhanced_cuda(image, variant=variant)
    np.testing.assert_array_equal(downloaded, expected)


def test_threshold_enhanced_gpu_input_not_modified():
    image = FIXTURES["random_deterministic"]()
    gpu_image = xray_cuda.upload_image(image)
    before = xray_cuda.download_image(gpu_image)
    xray_cuda.threshold_enhanced_gpu(gpu_image, 128, 255, threshold_variant_to_int("vectorized"), 16, 16)
    after = xray_cuda.download_image(gpu_image)
    np.testing.assert_array_equal(before, after)


# -- block configuration -------------------------------------------------------


@pytest.mark.parametrize("block", [(8, 8), (16, 16), (32, 8), (32, 16)])
@pytest.mark.parametrize("variant", THRESHOLD_VARIANTS)
def test_block_configuration_exact_match(block, variant):
    image = FIXTURES["random_deterministic"](size=64)
    config = FilterConfig()
    cpu_out = apply_threshold(image, config)
    out = threshold_enhanced_cuda(image, variant=variant, block=block)
    diff = np.abs(cpu_out.astype(np.int16) - out.astype(np.int16))
    assert diff.max() == 0, f"variant={variant} block={block}: max_abs_diff={diff.max()}"


# -- input validation -------------------------------------------------------


def test_threshold_enhanced_cuda_rejects_invalid_variant():
    image = FIXTURES["constant"]()
    with pytest.raises(ValueError):
        threshold_enhanced_cuda(image, variant="not_a_variant")


def test_threshold_enhanced_cuda_rejects_out_of_range_threshold_value():
    image = FIXTURES["constant"]()
    with pytest.raises(ValueError):
        threshold_enhanced_cuda(image, threshold_value=256, variant="vectorized")


def test_threshold_enhanced_gpu_rejects_null_image():
    with pytest.raises(ValueError):
        xray_cuda.threshold_enhanced_gpu(None, 128, 255, 0, 16, 16)


def test_threshold_enhanced_gpu_rejects_out_of_range_variant():
    image = FIXTURES["constant"]()
    gpu_image = xray_cuda.upload_image(image)
    with pytest.raises(ValueError):
        xray_cuda.threshold_enhanced_gpu(gpu_image, 128, 255, 99, 16, 16)


def test_threshold_enhanced_batch_gpu_rejects_wrong_dtype():
    batch = np.zeros((4, 16, 16), dtype=np.float32)
    with pytest.raises(ValueError):
        xray_cuda.threshold_enhanced_batch_gpu(batch, 128, 255, 0, 16, 16)


def test_threshold_enhanced_batch_gpu_rejects_wrong_ndim():
    batch = np.zeros((16, 16), dtype=np.uint8)  # missing batch dim
    with pytest.raises(ValueError):
        xray_cuda.threshold_enhanced_batch_gpu(batch, 128, 255, 0, 16, 16)


# -- memory: repeated allocation/release -------------------------------------------------------


@pytest.mark.parametrize("variant", THRESHOLD_VARIANTS)
def test_repeated_enhanced_threshold_no_crash_and_no_leak(variant):
    image = FIXTURES["random_deterministic"]()
    expected = threshold_enhanced_cuda(image, variant=variant)

    free_before = xray_cuda.device_memory_info()["free_bytes"]

    for _ in range(100):
        out = threshold_enhanced_cuda(image, variant=variant)
        np.testing.assert_array_equal(out, expected)

    free_after = xray_cuda.device_memory_info()["free_bytes"]
    leaked_bytes = free_before - free_after
    assert leaked_bytes < 50 * 1024 * 1024, (
        f"variant={variant}: possible GPU memory leak: {leaked_bytes} bytes not reclaimed after 100 calls"
    )


# -- pipeline integration: Enhanced Threshold + Enhanced Gaussian/Median/Sobel/Laplacian -------------------------------------------------------


@pytest.mark.skipif(get_real_dataset_path() is None, reason="Real X-ray dataset not present on this machine.")
def test_enhanced_threshold_pipeline_matches_cpu_and_basic_pipeline():
    """cuda.pipeline.run_basic_cuda_pipeline(use_enhanced_threshold=True):
    Gaussian/Median/Sobel/Laplacian stay Basic; only Threshold changes.
    Threshold's comparison is exact-integer, so the default (vectorized)
    variant must match the Basic pipeline EXACTLY, not within a
    tolerance."""
    from cpu.filters import apply_gaussian, apply_laplacian, apply_median, apply_sobel
    from cuda.pipeline import run_basic_cuda_pipeline

    dm = DatasetManager(get_real_dataset_path())
    dm.scan()
    selection = dm.random_batch(batch_size=10, seed=42)
    images = [load_image(p) for p in selection.selected_paths if load_image(p).shape == (224, 224)]
    if len(images) < 2:
        pytest.skip("Not enough same-resolution images in this sample.")

    config = FilterConfig()
    basic_output, _t1 = run_basic_cuda_pipeline(images, config, use_enhanced_threshold=False)
    enhanced_output, _t2 = run_basic_cuda_pipeline(images, config, use_enhanced_threshold=True)

    diff_vs_basic = np.abs(basic_output.astype(np.int16) - enhanced_output.astype(np.int16))
    assert diff_vs_basic.max() == 0, (
        f"Enhanced-Threshold pipeline differs from Basic pipeline "
        f"(max_abs_diff={diff_vs_basic.max()}); expected bit-exact"
    )

    def cpu_pipeline(image):
        return apply_threshold(
            apply_laplacian(apply_sobel(apply_median(apply_gaussian(image, config), config), config), config), config
        )

    cpu_outputs = np.stack([cpu_pipeline(img) for img in images])
    diff_vs_cpu = np.abs(cpu_outputs.astype(np.int16) - enhanced_output.astype(np.int16))
    pct_differing = 100.0 * np.count_nonzero(diff_vs_cpu) / diff_vs_cpu.size
    assert pct_differing <= 1.0, (
        f"Enhanced-Threshold pipeline vs CPU: {pct_differing:.4f}% pixels differ, exceeds the "
        f"Section 4F/5-established 1% chain tolerance"
    )


@pytest.mark.parametrize("variant", THRESHOLD_VARIANTS)
def test_enhanced_threshold_pipeline_all_variants_run_without_crash(variant):
    from cuda.pipeline import run_basic_cuda_pipeline

    images = [FIXTURES["random_deterministic"](size=32, seed=i) for i in range(4)]
    config = FilterConfig()
    output, timing = run_basic_cuda_pipeline(images, config, use_enhanced_threshold=True, threshold_variant=variant)
    assert output.shape == (4, 32, 32)
    assert timing.threshold_ms is not None and timing.threshold_ms >= 0.0


def test_enhanced_threshold_pipeline_disabled_threshold_ignores_enhanced_flag():
    from cpu.filters import apply_gaussian, apply_laplacian, apply_median, apply_sobel
    from cuda.pipeline import run_basic_cuda_pipeline

    images = [FIXTURES["constant"](size=16)]
    config = FilterConfig(threshold_enabled=False)
    output, timing = run_basic_cuda_pipeline(images, config, use_enhanced_threshold=True)
    assert timing.threshold_ms is None

    cpu_expected = apply_laplacian(apply_sobel(apply_median(apply_gaussian(images[0], config), config), config), config)
    np.testing.assert_array_equal(output[0], cpu_expected)


def test_all_five_enhanced_stages_pipeline_combination():
    """All five Enhanced stages together must still match the all-Basic
    pipeline exactly at default settings (sigma=0.0) -- this is the
    official 'Enhanced CUDA' pipeline (spec item 18)."""
    from cuda.pipeline import run_basic_cuda_pipeline, run_enhanced_cuda_pipeline

    images = [FIXTURES["random_deterministic"](size=64, seed=i) for i in range(3)]
    config = FilterConfig()
    basic_output, _t1 = run_basic_cuda_pipeline(images, config)
    combined_output, _t2 = run_basic_cuda_pipeline(
        images, config,
        use_enhanced_gaussian=True, use_enhanced_median=True,
        use_enhanced_sobel=True, use_enhanced_laplacian=True, use_enhanced_threshold=True,
    )
    enhanced_output, _t3 = run_enhanced_cuda_pipeline(images, config)

    diff = np.abs(basic_output.astype(np.int16) - combined_output.astype(np.int16))
    assert diff.max() == 0, (
        f"All-five-Enhanced pipeline differs from all-Basic pipeline "
        f"(max_abs_diff={diff.max()}); expected bit-exact at sigma=0.0"
    )
    np.testing.assert_array_equal(combined_output, enhanced_output)


# -- experimental fused Laplacian+Threshold kernel -------------------------------------------------------


def test_fused_laplacian_threshold_matches_unfused_and_cpu():
    """The experimental fused kernel must produce EXACTLY the same
    output as running Enhanced Laplacian (Specialized) then Enhanced
    Threshold (Vectorized) as two separate launches, and match CPU."""
    images = [FIXTURES["random_deterministic"](size=48, seed=i) for i in range(4)]
    batch = np.stack(images, axis=0)
    coeffs = laplacian_kernel_2d(3)
    scale, delta, tv, mv = 1.0, 0.0, 128, 255

    lap_result = xray_cuda.laplacian_enhanced_batch_gpu(batch, coeffs, scale, delta, 2, 16, 16)  # 2 = Specialized
    unfused_result = xray_cuda.threshold_enhanced_batch_gpu(lap_result["output"], tv, mv, 0, 16, 16)  # 0 = Vectorized
    fused_result = xray_cuda.laplacian_threshold_fused_batch_gpu(batch, coeffs, scale, delta, tv, mv, 16, 16)

    diff_unfused = np.abs(unfused_result["output"].astype(np.int16) - fused_result["output"].astype(np.int16))
    assert diff_unfused.max() == 0, f"fused vs unfused: max_abs_diff={diff_unfused.max()}"

    config = FilterConfig(laplacian_kernel_size=3, laplacian_scale=scale, laplacian_delta=delta,
                           threshold_value=tv, threshold_max_value=mv)
    cpu_out = np.stack([apply_threshold(apply_laplacian(img, config), config) for img in images])
    diff_cpu = np.abs(cpu_out.astype(np.int16) - fused_result["output"].astype(np.int16))
    assert diff_cpu.max() == 0, f"fused vs CPU: max_abs_diff={diff_cpu.max()}"


@pytest.mark.parametrize("kernel_size", [3, 5])
def test_fused_laplacian_threshold_kernel_size_3_and_5(kernel_size):
    images = [FIXTURES["sharp_edge_square"]()]
    batch = np.stack(images, axis=0)
    coeffs = laplacian_kernel_2d(kernel_size)

    fused_result = xray_cuda.laplacian_threshold_fused_batch_gpu(batch, coeffs, 1.0, 0.0, 128, 255, 16, 16)

    config = FilterConfig(laplacian_kernel_size=kernel_size, threshold_value=128, threshold_max_value=255)
    cpu_out = np.stack([apply_threshold(apply_laplacian(img, config), config) for img in images])
    diff = np.abs(cpu_out.astype(np.int16) - fused_result["output"].astype(np.int16))
    assert diff.max() == 0


def test_fused_laplacian_threshold_rejects_unsupported_kernel_size():
    image = FIXTURES["constant"]()
    batch = image[None, :, :]
    bad_coeffs = np.zeros((7, 7), dtype=np.float32)
    with pytest.raises(ValueError):
        xray_cuda.laplacian_threshold_fused_batch_gpu(batch, bad_coeffs, 1.0, 0.0, 128, 255, 16, 16)
