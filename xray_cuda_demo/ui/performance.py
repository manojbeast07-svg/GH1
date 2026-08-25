"""Performance dashboard components (Section 12 spec items 19-22, 35,
55). Every function ACCEPTS already-measured results (from
ui.services or a loaded benchmark_results/ JSON) and only renders them
-- no timing logic, no recalculation of speedups, lives here (spec item
55: render_performance_summary must not recompute timing logic).
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

FILTER_OPTIMIZATION_NOTES = {
    "gaussian": ("Separable convolution + shared memory + constant memory + compile-time specialization",
                 "Basic's O(k²) 2-D convolution was genuinely unexploited -- algorithmic + memory optimization."),
    "median": ("Branchless 3×3 sorting network (Devillard opt_med9)",
               "Removing insertion-sort's data-dependent branching was the real lever, not memory reuse alone."),
    "sobel": ("Compile-time mode specialization",
              "Basic Sobel was already hand-optimized (1 launch, minimal arithmetic) -- limited headroom remained."),
    "laplacian": ("Shared memory + constant memory + compile-time specialization",
                  "Basic's generic runtime-sized loop had real, unexploited headroom -- specialization helped a lot."),
    "threshold": ("uchar4 vectorized memory access",
                  "Pure pointwise op, no neighborhood lever at all -- the only gain available is wider memory transactions."),
}


def render_metric_cards(label_ms: Dict[str, Optional[float]], title: str = "") -> None:
    if title:
        st.subheader(title)
    cols = st.columns(len(label_ms))
    for col, (label, ms) in zip(cols, label_ms.items()):
        with col:
            st.metric(label, f"{ms:.3f} ms" if ms is not None else "n/a")


def render_speedup_row(speedups: Dict[str, float]) -> None:
    cols = st.columns(len(speedups))
    for col, (label, value) in zip(cols, speedups.items()):
        with col:
            st.metric(label, f"{value:.3f}×")


def render_compute_vs_end_to_end(
    basic_compute_ms: float, enhanced_compute_ms: float,
    basic_e2e_ms: float, enhanced_e2e_ms: float,
) -> None:
    """Spec item 20 -- the essential distinction. Two clearly separate
    sections, never blended into one misleading number."""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Compute-only")
        st.metric("Basic", f"{basic_compute_ms:.3f} ms")
        st.metric("Enhanced", f"{enhanced_compute_ms:.3f} ms")
        gain = (basic_compute_ms / enhanced_compute_ms) if enhanced_compute_ms > 0 else 0.0
        st.metric("Improvement", f"{gain:.3f}×")
    with col2:
        st.markdown("#### End-to-end")
        st.metric("Basic", f"{basic_e2e_ms:.3f} ms")
        st.metric("Enhanced", f"{enhanced_e2e_ms:.3f} ms")
        gain_e2e = (basic_e2e_ms / enhanced_e2e_ms) if enhanced_e2e_ms > 0 else 0.0
        st.metric("Improvement", f"{gain_e2e:.3f}×")

    with st.expander("Why is the compute-only improvement so much larger than the end-to-end improvement?"):
        st.markdown(
            "GPU kernel optimization can produce a much larger compute-only improvement than end-to-end "
            "improvement because disk I/O, host-side work, memory transfers, and other fixed overheads remain "
            "in the full application path. Both numbers are shown above -- neither is hidden."
        )


def render_per_filter_chart(rows: list) -> None:
    """`rows` is per_filter benchmark's `rows` list (Section 11 shape:
    filter/basic_kernel_ms/enhanced_kernel_ms/kernel_speedup/...)."""
    df = pd.DataFrame([{
        "Filter": r["filter"].capitalize(),
        "Basic (ms)": r["basic_kernel_ms"]["mean"] if isinstance(r["basic_kernel_ms"], dict) else r["basic_kernel_ms"],
        "Enhanced (ms)": r["enhanced_kernel_ms"]["mean"] if isinstance(r["enhanced_kernel_ms"], dict) else r["enhanced_kernel_ms"],
        "Speedup": r["kernel_speedup"],
    } for r in rows])

    col1, col2 = st.columns(2)
    with col1:
        melted = df.melt(id_vars="Filter", value_vars=["Basic (ms)", "Enhanced (ms)"], var_name="Implementation", value_name="ms")
        fig = px.bar(melted, x="Filter", y="ms", color="Implementation", barmode="group", title="Per-filter kernel time: Basic vs Enhanced")
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig2 = px.bar(df, x="Filter", y="Speedup", title="Per-filter speedup (Basic → Enhanced)", text="Speedup")
        fig2.update_traces(texttemplate="%{text:.2f}×", textposition="outside")
        st.plotly_chart(fig2, width="stretch")

    st.dataframe(df.style.format({"Basic (ms)": "{:.4f}", "Enhanced (ms)": "{:.4f}", "Speedup": "{:.3f}×"}),
                 width="stretch", hide_index=True)


def render_filter_optimization_cards(rows: list) -> None:
    """Spec items 22, 45-46 -- the five filter cards with measured
    speedup + a one-line explanation of WHY the optimization differs
    per filter."""
    cols = st.columns(5)
    for col, r in zip(cols, rows):
        name = r["filter"]
        technique, note = FILTER_OPTIMIZATION_NOTES.get(name, ("", ""))
        with col:
            st.markdown(f"**{name.capitalize()}**")
            st.metric("Basic → Enhanced", f"{r['kernel_speedup']:.2f}×")
            st.caption(technique)
    st.markdown("---")
    for r in rows:
        name = r["filter"]
        technique, note = FILTER_OPTIMIZATION_NOTES.get(name, ("", ""))
        st.markdown(f"**{name.capitalize()}:** {note}")


def render_amdahl_contribution(rows: list) -> None:
    """Spec item 21/23 -- measured (not theoretical) contribution to
    total compute-time reduction."""
    df = pd.DataFrame([{
        "Filter": r["filter"].capitalize(),
        "Absolute reduction (ms)": r["absolute_reduction_ms"],
        "% of total compute reduction": r["pct_of_total_compute_reduction"],
    } for r in rows])
    fig = px.pie(df, names="Filter", values="% of total compute reduction",
                 title="Measured contribution to total compute-time reduction (not a theoretical Amdahl bound)")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(df.style.format({"Absolute reduction (ms)": "{:.4f}", "% of total compute reduction": "{:.1f}%"}),
                 width="stretch", hide_index=True)

    top_two = df.nlargest(2, "% of total compute reduction")
    total_top_two = top_two["% of total compute reduction"].sum()
    st.info(f"**{' and '.join(top_two['Filter'])}** together account for approximately "
            f"**{total_top_two:.0f}%** of the measured compute-time reduction.")


IMPLEMENTATION_EXPLANATIONS = {
    "CPU": "Standard CPU image-processing implementation (Python + OpenCV).",
    "Basic CUDA": "Custom CUDA kernels + GPU batch processing + GPU-resident pipeline (C++ + CUDA).",
    "Enhanced CUDA": "Basic CUDA + filter-specific optimization + memory optimization + specialized kernels "
                     "(C++ + optimized CUDA).",
}


def render_live_performance_cards(comparison) -> None:
    """Spec item 23: three cards (CPU/Basic/Enhanced total_ms) + the
    three measured speedups, computed from THIS run's ComparisonResult
    -- never recalculated elsewhere, never hardcoded."""
    results = comparison.results
    cols = st.columns(3)
    for col, label in zip(cols, ("CPU", "Basic CUDA", "Enhanced CUDA")):
        with col:
            if label in results:
                st.metric(label, f"{results[label].total_ms:.3f} ms")
            elif label in comparison.errors:
                st.metric(label, "FAILED")
                st.caption(comparison.errors[label])
            else:
                st.metric(label, "n/a")

    cpu, basic, enhanced = results.get("CPU"), results.get("Basic CUDA"), results.get("Enhanced CUDA")
    speedups = {}
    if cpu and basic:
        speedups["Basic speedup (CPU / Basic)"] = cpu.total_ms / basic.total_ms if basic.total_ms > 0 else 0.0
    if cpu and enhanced:
        speedups["Enhanced speedup (CPU / Enhanced)"] = cpu.total_ms / enhanced.total_ms if enhanced.total_ms > 0 else 0.0
    if basic and enhanced:
        speedups["Basic → Enhanced (Basic / Enhanced)"] = basic.total_ms / enhanced.total_ms if enhanced.total_ms > 0 else 0.0
    if speedups:
        render_speedup_row(speedups)


def render_h2d_kernel_d2h_breakdown(comparison) -> None:
    """Spec item 24: H2D / Kernel / D2H / Total for Basic and Enhanced
    CUDA, directly from the backend's CUDA-event timing -- demonstrates
    why GPU kernel optimization and end-to-end application speedup
    differ."""
    cols = st.columns(2)
    for col, label in zip(cols, ("Basic CUDA", "Enhanced CUDA")):
        with col:
            st.markdown(f"**{label}**")
            r = comparison.results.get(label)
            if r is None:
                st.caption(comparison.errors.get(label, "n/a"))
                continue
            st.write(f"H2D: {r.h2d_ms:.4f} ms" if r.h2d_ms is not None else "H2D: n/a")
            st.write(f"Kernel (compute): {r.compute_ms:.4f} ms" if r.compute_ms is not None else "Kernel: n/a")
            st.write(f"D2H: {r.d2h_ms:.4f} ms" if r.d2h_ms is not None else "D2H: n/a")
            st.write(f"**Total: {r.total_ms:.4f} ms**")


def render_per_filter_live_table(comparison) -> None:
    """Spec item 25: per-filter timing across CPU/Basic/Enhanced, from
    the backend's own per-stage timing fields -- never inferred by
    dividing a total."""
    rows = []
    for stage in ["gaussian", "median", "sobel", "laplacian", "threshold"]:
        row = {"Filter": stage.capitalize()}
        for label in ("CPU", "Basic CUDA", "Enhanced CUDA"):
            r = comparison.results.get(label)
            ms = r.per_stage_ms.get(stage) if r else None
            row[label] = f"{ms:.4f}" if ms is not None else "—"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_gpu_optimization_summary(production_defaults: Dict[str, str]) -> None:
    """Spec item 26: current Enhanced configuration, explicitly labeled
    as a CUDA implementation detail, not an image-processing parameter."""
    st.markdown("**CUDA implementation** (not an image-processing parameter)")
    cols = st.columns(len(production_defaults) + 1)
    for col, (filt, variant) in zip(cols, production_defaults.items()):
        with col:
            st.caption(filt.capitalize())
            st.write(variant)
    with cols[-1]:
        st.caption("Block")
        st.write("16×16")


def render_implementation_explanation() -> None:
    """Spec item 27: a short, factual explanation of what each
    implementation actually is."""
    cols = st.columns(3)
    for col, (label, text) in zip(cols, IMPLEMENTATION_EXPLANATIONS.items()):
        with col:
            st.markdown(f"**{label}**")
            st.caption(text)


# -- Section 15: batch performance rendering -----------------------------------------------------


def render_batch_performance_cards(batch_result) -> None:
    """Spec item 19: N images + total ms + img/s per implementation,
    plus the three measured speedups -- all from
    BatchComparisonResult.implementation_totals(), never recalculated
    here from throughput (item 18: speedups come from the underlying
    measured timings, not a second derivation)."""
    totals = batch_result.implementation_totals()
    cols = st.columns(3)
    for col, label in zip(cols, ("CPU", "Basic CUDA", "Enhanced CUDA")):
        with col:
            t = totals.get(label)
            if t is None:
                st.metric(label, "n/a")
                continue
            st.markdown(f"**{label}**")
            st.metric("Images", t["n_images"])
            st.metric("Total time", f"{t['total_ms']:.3f} ms")
            st.metric("Throughput", f"{t['images_per_second']:.1f} img/s")

    cpu, basic, enhanced = totals.get("CPU"), totals.get("Basic CUDA"), totals.get("Enhanced CUDA")
    speedups = {}
    if cpu and basic:
        speedups["CPU → Basic"] = cpu["total_ms"] / basic["total_ms"] if basic["total_ms"] > 0 else 0.0
    if cpu and enhanced:
        speedups["CPU → Enhanced"] = cpu["total_ms"] / enhanced["total_ms"] if enhanced["total_ms"] > 0 else 0.0
    if basic and enhanced:
        speedups["Basic → Enhanced"] = basic["total_ms"] / enhanced["total_ms"] if enhanced["total_ms"] > 0 else 0.0
    if speedups:
        render_speedup_row(speedups)


def render_batch_h2d_kernel_d2h_breakdown(batch_result) -> None:
    """Spec item 20: aggregated H2D / kernels (per-filter) / D2H / Total
    for Basic and Enhanced CUDA, summed across every resolution group
    this batch spanned -- live batch execution data, not historical."""
    totals = batch_result.implementation_totals()
    cols = st.columns(2)
    for col, label in zip(cols, ("Basic CUDA", "Enhanced CUDA")):
        with col:
            st.markdown(f"**{label}**")
            t = totals.get(label)
            if t is None:
                st.caption("n/a")
                continue
            st.write(f"H2D: {t['h2d_ms']:.4f} ms" if t["h2d_ms"] is not None else "H2D: n/a")
            for stage in ["gaussian", "median", "sobel", "laplacian", "threshold"]:
                ms = t["per_stage_ms"].get(stage)
                if ms is not None:
                    st.write(f"{stage.capitalize()}: {ms:.4f} ms")
            st.write(f"D2H: {t['d2h_ms']:.4f} ms" if t["d2h_ms"] is not None else "D2H: n/a")
            st.write(f"**Total: {t['total_ms']:.4f} ms**")


def render_batch_per_filter_table(batch_result) -> None:
    """Spec item 21: per-filter timing across CPU/Basic/Enhanced,
    summed across all resolution groups. 'N/A' (never estimated) when a
    filter's CPU timing isn't available for this batch result."""
    totals = batch_result.implementation_totals()
    rows = []
    for stage in ["gaussian", "median", "sobel", "laplacian", "threshold"]:
        row = {"Filter": stage.capitalize()}
        for label in ("CPU", "Basic CUDA", "Enhanced CUDA"):
            t = totals.get(label)
            ms = t["per_stage_ms"].get(stage) if t else None
            row[label] = f"{ms:.4f}" if ms is not None else "N/A"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_batch_message(batch_result) -> None:
    """Spec item 33: a concise, measurement-backed interpretation --
    only claims what the live totals actually show."""
    totals = batch_result.implementation_totals()
    cpu, basic, enhanced = totals.get("CPU"), totals.get("Basic CUDA"), totals.get("Enhanced CUDA")
    lines = []
    if cpu and basic:
        lines.append(f"Basic CUDA = {cpu['total_ms']/basic['total_ms']:.2f}× CPU" if basic["total_ms"] > 0 else "")
    if cpu and enhanced:
        lines.append(f"Enhanced CUDA = {cpu['total_ms']/enhanced['total_ms']:.2f}× CPU" if enhanced["total_ms"] > 0 else "")
    if basic and enhanced:
        lines.append(f"Enhanced vs Basic = {basic['total_ms']/enhanced['total_ms']:.2f}×" if enhanced["total_ms"] > 0 else "")
    if lines:
        st.markdown("**GPU acceleration (this batch):** " + "  ·  ".join(l for l in lines if l))
    st.caption("GPU performance depends on workload size. Small batches may be dominated by transfer and launch "
               "overhead, while larger batches provide more parallel work.")


def render_quick_batch_sweep_table(sweep: dict) -> None:
    """Spec item 35."""
    rows = []
    for row in sweep["rows"]:
        rows.append({
            "Batch": row["effective_batch_size"],
            "CPU img/s": f"{row['cpu_images_per_second']:.1f}" if "cpu_images_per_second" in row else "N/A",
            "Basic img/s": f"{row['basic_images_per_second']:.1f}" if "basic_images_per_second" in row else "N/A",
            "Enhanced img/s": f"{row['enhanced_images_per_second']:.1f}" if "enhanced_images_per_second" in row else "N/A",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# -- Section 16: historical benchmark analytics -----------------------------------------------------


def render_measurement_summary(canonical: dict) -> None:
    """Spec item 42: three dynamic, factual observations computed from
    the SELECTED benchmark -- never a generic fixed claim that might not
    match it."""
    st.markdown("### What the measurements show")
    cpu_vs_basic = canonical["speedups"]["basic_vs_cpu"]
    cpu_vs_enhanced = canonical["speedups"]["enhanced_vs_cpu"]
    basic_vs_enhanced_compute = canonical["speedups"]["enhanced_vs_basic_compute_only"]
    st.markdown(
        f"1. **CPU vs Basic CUDA:** Basic CUDA end-to-end is **{cpu_vs_basic:.2f}×** faster than the CPU "
        f"reference for benchmark `{canonical['benchmark_id']}`.\n"
        f"2. **CPU vs Enhanced CUDA:** Enhanced CUDA end-to-end is **{cpu_vs_enhanced:.2f}×** faster than the "
        f"CPU reference.\n"
        f"3. **Basic vs Enhanced (compute-only):** Enhanced CUDA's GPU compute time is **{basic_vs_enhanced_compute:.2f}×** "
        f"faster than Basic CUDA's, before accounting for fixed end-to-end overhead."
    )


def render_live_vs_canonical_label(is_live: bool) -> None:
    if is_live:
        st.warning("🔴 LIVE RESULT -- measured just now for the current configuration, not a stored canonical benchmark.")
    else:
        st.success("📊 CANONICAL BENCHMARK -- loaded from `benchmark_results/` (Section 11).")
