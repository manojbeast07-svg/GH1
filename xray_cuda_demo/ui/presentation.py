"""Section 20: the dedicated Presentation Mode tab -- a self-contained,
~30-second "story" screen for a technical demo/presentation. Renders
only; every number comes from an already-loaded historical benchmark
summary, already-computed live session state, or a freshly-verified
GPU-backend fact -- nothing here reprocesses images or reruns a
benchmark.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from ui import performance
from ui import theme

ARCHITECTURE_DIAGRAM = """\
                    X-RAY DATASET
                         │
                         ▼
                  SAME INPUT BATCH
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    Python/OpenCV   C++ Basic CUDA   C++ Enhanced CUDA
        CPU              .cu              .cu
          │              │                │
          └──────────────┼────────────────┘
                         ▼
                 5 FILTER PIPELINE
                         │
                         ▼
                 PERFORMANCE RESULTS"""

PIPELINE_DIAGRAM = """\
Original X-ray
      ↓
Gaussian
      ↓
Median
      ↓
Sobel
      ↓
Laplacian
      ↓
Threshold
      ↓
Final Output"""

OPTIMIZATION_METHODS = {
    "Gaussian": "Separable convolution · Shared memory · Specialization",
    "Median": "3×3 sorting network",
    "Sobel": "Compile-time specialization",
    "Laplacian": "Shared memory · Constant coefficients · Specialization",
    "Threshold": "uchar4 vectorization",
}


def render_header() -> None:
    st.markdown("## 🩻 CUDA X-RAY PROCESSING LAB")
    st.markdown("#### CPU vs Basic CUDA vs Enhanced CUDA")
    st.caption("Real X-ray dataset  ·  5-filter image-processing pipeline  ·  Custom CUDA C++")
    st.warning("Technical demonstration only — not a medical diagnostic system.")


def render_architecture_diagram() -> None:
    st.markdown("#### Architecture")
    st.code(ARCHITECTURE_DIAGRAM, language="text")
    cols = st.columns(3)
    cols[0].caption("**CPU** / Python + OpenCV")
    cols[1].caption("**Basic** / C++ CUDA")
    cols[2].caption("**Enhanced** / C++ CUDA")


def render_pipeline_diagram() -> None:
    st.markdown("#### Pipeline")
    st.code(PIPELINE_DIAGRAM, language="text")


def render_live_image(comparison) -> None:
    """`comparison` is a ui.services.ComparisonResult from the Live
    Processing tab's session state, or None if nothing has been run
    yet -- never auto-triggers a new run."""
    st.markdown("#### Live Result")
    if comparison is None:
        st.info("Run \"⚖ Compare CPU vs Basic CUDA vs Enhanced CUDA\" in Live Processing to show a "
                 "live image here.")
        return

    st.caption(f"{comparison.image_meta.filename}  ·  {comparison.image_meta.width}×{comparison.image_meta.height}")
    cols = st.columns(4)
    original = comparison.results.get("CPU")
    with cols[0]:
        st.markdown("**Original**")
        if original and original.stage_outputs:
            st.image(original.stage_outputs.get("original"), clamp=True, channels="GRAY", width="stretch")
    for col, label in zip(cols[1:], ("CPU", "Basic CUDA", "Enhanced CUDA")):
        with col:
            st.markdown(f"**{label}**")
            r = comparison.results.get(label)
            if r and r.final_outputs:
                st.image(r.final_outputs[0], clamp=True, channels="GRAY", width="stretch")
            else:
                st.caption(comparison.errors.get(label, "n/a"))


def render_performance_cards(canonical: dict) -> None:
    st.markdown("#### Performance")
    performance.render_metric_cards({
        "CPU / OpenCV": canonical["cpu"]["mode4_end_to_end_ms"]["mean"],
        "Basic CUDA": canonical["basic_cuda"]["mode4_end_to_end_ms"]["mean"],
        "Enhanced CUDA": canonical["enhanced_cuda"]["mode4_end_to_end_ms"]["mean"],
    })
    performance.render_speedup_row({
        "CPU → Basic": canonical["speedups"]["basic_vs_cpu"],
        "CPU → Enhanced": canonical["speedups"]["enhanced_vs_cpu"],
        "Basic → Enhanced": canonical["speedups"]["enhanced_vs_basic_end_to_end"],
    })


def render_compute_vs_end_to_end(canonical: dict) -> None:
    performance.render_compute_vs_end_to_end(
        canonical["basic_cuda"]["mode1_kernel_only_ms"]["mean"],
        canonical["enhanced_cuda"]["mode1_kernel_only_ms"]["mean"],
        canonical["basic_cuda"]["mode4_end_to_end_ms"]["mean"],
        canonical["enhanced_cuda"]["mode4_end_to_end_ms"]["mean"],
    )


def render_per_filter_cards(per_filter_rows: Optional[list]) -> None:
    st.markdown("#### Per-Filter Optimization")
    if not per_filter_rows:
        st.caption("No per-filter benchmark stored with this result.")
        return
    cols = st.columns(len(per_filter_rows))
    for col, row in zip(cols, per_filter_rows):
        name = row["filter"].capitalize()
        with col:
            st.markdown(f"**{name}**")
            st.metric("Basic → Enhanced", f"{row['kernel_speedup']:.2f}×")
            st.caption(OPTIMIZATION_METHODS.get(name, ""))


def render_threading_summary(width: int = 224, height: int = 224, batch_size: int = 1) -> None:
    """Section 23: compact Threading & Parallelism summary for the
    presentation story (spec item 40). Queries the same
    pipeline.threading_metrics service the full Threading tab uses --
    every value here is real and freshly measured, never hardcoded."""
    from pipeline.threading_metrics import get_threading_metrics

    st.markdown("#### Threading & Parallelism")
    m = get_threading_metrics(width=width, height=height, batch_size=batch_size)
    if not m.gpu_available:
        st.caption("No CUDA device detected.")
        return
    cols = st.columns(5)
    cols[0].metric("CPU Cores", m.cpu_physical_cores if m.cpu_physical_cores is not None else "N/A")
    cols[1].metric("GPU SMs", m.gpu_sm_count if m.gpu_sm_count is not None else "N/A")
    cols[2].metric("CUDA Threads", f"{m.total_threads_launched:,}" if m.total_threads_launched else "N/A")
    cols[3].metric("Threads/Block", m.threads_per_block if m.threads_per_block is not None else "N/A")
    cols[4].metric("Warps/Block", m.warps_per_block if m.warps_per_block is not None else "N/A")
    st.caption(f"Launch config for {width}×{height}, batch={batch_size} — see the Threading & Parallelism tab for full detail.")


def render_key_insight(per_filter_rows: Optional[list]) -> None:
    st.markdown("#### Key Insight — Measured Contribution Analysis")
    if not per_filter_rows:
        st.caption("No per-filter benchmark stored with this result.")
        return
    reduction_by_filter = {r["filter"]: r["absolute_reduction_ms"] for r in per_filter_rows}
    total = sum(reduction_by_filter.values())
    if total <= 0:
        st.caption("No measured compute-time reduction to summarize.")
        return
    gaussian_median_pct = 100.0 * (reduction_by_filter.get("gaussian", 0.0) + reduction_by_filter.get("median", 0.0)) / total
    st.success(f"**Gaussian + Median ≈ {gaussian_median_pct:.0f}%** of the total measured compute-time reduction "
               f"(measured contribution analysis, this benchmark).")


def render_gpu_implementation(facts: dict) -> None:
    st.markdown("#### GPU Implementation")
    lines = [
        ("Native C++ / CUDA", facts["native_extension"]),
        (f".cu kernels ({facts['cu_file_count']} files, compiled with NVCC)", facts["cu_file_count"] > 0),
        ("GPU-resident pipeline (single native call)", facts["single_pipeline_call"]),
        ("RAII GPU memory", facts["raii_memory"]),
        ("CUDA-event timing", facts["cuda_event_timing"]),
        ("No Python GPU compute libraries", facts["no_python_gpu_libs"]),
        ("Thin pybind11 bridge", facts["native_extension"]),
    ]
    for label, ok in lines:
        st.markdown(f"{'✅' if ok else '⚠️'} {label}")


def render_correctness(canonical: dict) -> None:
    st.markdown("#### Correctness")
    corr = canonical.get("correctness") or {}
    detail = corr.get("detailed")
    cols = st.columns(5)
    if detail and detail.get("filter_level"):
        for col, (name, result) in zip(cols, detail["filter_level"].items()):
            with col:
                st.metric(name.capitalize(), "✅ PASS" if result["pass"] else "❌ FAIL")
    else:
        st.caption("No stored per-filter correctness for this benchmark.")

    canonical_pipeline = corr.get("canonical_pipeline") or {}
    enhanced_vs_basic = canonical_pipeline.get("enhanced_vs_basic")
    enhanced_vs_cpu = canonical_pipeline.get("enhanced_vs_cpu")
    col1, col2 = st.columns(2)
    if enhanced_vs_basic:
        col1.metric("Enhanced vs Basic", "✅ PASS" if enhanced_vs_basic["max_abs_diff"] == 0 else "⚠️ NOT bit-exact")
    if enhanced_vs_cpu:
        col2.metric("Final pipeline differing pixels", f"{enhanced_vs_cpu['differing_pixel_percentage']:.4f}%")


def render_batch_insight(batch_sweep: Optional[dict]) -> None:
    st.markdown("#### Batch-Size Insight")
    st.markdown("**GPU advantage changes with workload size.**")
    if not batch_sweep or not batch_sweep.get("rows"):
        st.caption("No batch-size sweep stored with this benchmark.")
        return
    rows = batch_sweep["rows"]
    smallest = min(rows, key=lambda r: r["effective_batch_size"])
    largest = max(rows, key=lambda r: r["effective_batch_size"])
    st.caption(f"At batch={smallest['effective_batch_size']}: Enhanced vs Basic = {smallest['enhanced_vs_basic_gain']:.2f}×  ·  "
               f"at batch={largest['effective_batch_size']}: Enhanced vs Basic = {largest['enhanced_vs_basic_gain']:.2f}×  "
               f"(measured, this benchmark's stored sweep — see Performance Analytics for the full chart).")
