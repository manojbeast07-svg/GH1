"""Section 17: the Interactive CUDA Optimization Lab tab's rendering
layer. All rendering only -- every timing/correctness number comes from
ui.services (historical JSON loaders or the live measurement functions
that call cuda.optimization_lab.measure_variant()); nothing here
computes a measurement, compiles anything, or mutates production
defaults.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

FILTER_LABELS = {
    "gaussian": "Gaussian", "median": "Median", "sobel": "Sobel",
    "laplacian": "Laplacian", "threshold": "Threshold",
}

# Curated from this project's own measured findings (Sections 6-10, README) --
# narrative explanation text, not a numeric measurement, so (unlike timings/
# speedups) it is legitimately static rather than loaded from a JSON artifact.
WHY_IT_WORKS = {
    "gaussian": "Gaussian is linear and separable: a 2-D k*k convolution can be done as two 1-D passes "
                "(O(2k) instead of O(k^2) work per pixel). Basic's direct 2-D convolution left real headroom, "
                "so separable convolution + shared-memory tiling + constant-memory coefficients + compile-time "
                "specialization each added a further, progressively smaller gain.",
    "median": "Median filtering is nonlinear and non-separable, so Gaussian-style separable convolution does "
              "not apply. The major improvement came from a fixed-size branchless sorting network for the 3x3 "
              "neighborhood -- shared memory alone was not the main lever.",
    "sobel": "Basic Sobel was already relatively efficient (one launch, minimal arithmetic, fixed 3x3 "
             "coefficients), so the optimization headroom was small. Compile-time mode specialization gave a "
             "modest, real gain; several other approaches were measured and rejected.",
    "laplacian": "The Basic Laplacian implementation contained more generic loop work than the already "
                 "hand-optimized Basic Sobel implementation, so specialization and memory optimizations had "
                 "more headroom -- and that headroom grows with kernel size (k=5 shows a larger gain than k=3).",
    "threshold": "Threshold is a simple pointwise operation with no neighborhood to reuse. The optimization "
                 "opportunity is in memory access width and processing multiple pixels per thread, not "
                 "algorithmic restructuring -- register usage was already minimal before optimization.",
}

TECHNIQUE_CATEGORIES = {
    "gaussian": ["Algorithmic optimization (separable convolution)", "Memory optimization (shared + constant memory)",
                 "Specialization (compile-time kernel size)"],
    "median": ["Algorithmic optimization (branchless sorting network)", "Specialization (compile-time kernel size)"],
    "sobel": ["Specialization (compile-time mode)"],
    "laplacian": ["Memory optimization (shared + constant memory)", "Specialization (compile-time kernel size)"],
    "threshold": ["Instruction optimization (uchar4 vectorized memory access)"],
}


def render_filter_selector(filters: list, current: Optional[str]) -> str:
    labels = [FILTER_LABELS.get(f, f.capitalize()) for f in filters]
    index = filters.index(current) if current in filters else 0
    chosen_label = st.radio("Filter", labels, index=index, horizontal=True, key="opt_lab_filter_radio")
    return filters[labels.index(chosen_label)]


def render_production_configuration_panel(production_defaults: dict) -> None:
    st.markdown("**CURRENT PRODUCTION ENHANCED CONFIGURATION**")
    cols = st.columns(len(production_defaults) + 1)
    for col, (filt, variant) in zip(cols, production_defaults.items()):
        with col:
            st.caption(FILTER_LABELS.get(filt, filt.capitalize()))
            st.write(variant)
    with cols[-1]:
        st.caption("Block")
        st.write("16×16")
    st.caption("Informational only -- selections made below never alter this.")


def render_historical_variant_table_and_chart(filter_name: str, sweep: dict, kernel_size: Optional[int] = None) -> None:
    """Spec items 7-10: historical per-variant kernel time + speedup,
    loaded verbatim from the stored sweep -- never hardcoded."""
    if sweep is None:
        st.warning("⚠ No historical variant sweep found for this filter. Run "
                    "`python scripts/run_optimization_lab_experiments.py`.")
        return

    if "by_kernel_size" in sweep:
        available = sorted(sweep["by_kernel_size"].keys(), key=int)
        if kernel_size is not None and str(kernel_size) in sweep["by_kernel_size"]:
            ksize_key = str(kernel_size)
        else:
            ksize_key = available[0]
        rows = sweep["by_kernel_size"][ksize_key]
        st.caption(f"Historical kernel_size={ksize_key} (available: {', '.join(available)})")
    else:
        rows = sweep["rows"]

    st.caption(f"📊 HISTORICAL — {sweep['image_count']} images, {sweep['resolution'][1]}×{sweep['resolution'][0]}, "
               f"seed={sweep['seed']}, {sweep['measurement_runs']} measurement runs, GPU: {sweep.get('gpu_name') or 'n/a'}")

    df_rows = []
    for r in rows:
        label = r["variant"] + (" (production default)" if r["is_production_default"] else "")
        df_rows.append({
            "Variant": label, "Kernel time (ms)": r["kernel_ms"]["mean"] if r["kernel_ms"] else None,
            "Speedup vs Basic": r["speedup_vs_basic"], "Correctness": r["correctness"] or (r["error"] or "n/a"),
        })
    df = pd.DataFrame(df_rows)

    col1, col2 = st.columns(2)
    with col1:
        plot_df = df.dropna(subset=["Kernel time (ms)"])
        fig = px.bar(plot_df, x="Variant", y="Kernel time (ms)", title=f"{FILTER_LABELS[filter_name]}: variant vs. kernel time")
        st.plotly_chart(fig, width="stretch")
    with col2:
        plot_df2 = df.dropna(subset=["Speedup vs Basic"])
        fig2 = px.bar(plot_df2, x="Variant", y="Speedup vs Basic", title=f"{FILTER_LABELS[filter_name]}: variant vs. speedup over Basic",
                      text="Speedup vs Basic")
        fig2.update_traces(texttemplate="%{text:.2f}×", textposition="outside")
        fig2.add_hline(y=1.0, line_dash="dot")
        st.plotly_chart(fig2, width="stretch")

    st.dataframe(df.style.format({"Kernel time (ms)": "{:.4f}", "Speedup vs Basic": "{:.3f}×"}, na_rep="n/a"),
                 width="stretch", hide_index=True)

    basic = next((r for r in rows if r["variant"] == "basic"), None)
    best = max((r for r in rows if r.get("speedup_vs_basic")), key=lambda r: r["speedup_vs_basic"], default=None)
    if basic and best:
        st.markdown(f"**Basic:** {basic['kernel_ms']['mean']:.4f} ms &nbsp;&nbsp; "
                    f"**Best measured ({best['variant']}):** {best['kernel_ms']['mean']:.4f} ms &nbsp;&nbsp; "
                    f"**Speedup:** {best['speedup_vs_basic']:.2f}×")


def render_optimization_progression(sweep: dict, kernel_size: Optional[int] = None) -> None:
    """Spec item 29: Basic -> ... -> Enhanced as a step-down visual,
    using the ACTUAL measured ordering (not assumed monotonic)."""
    if sweep is None:
        return
    rows = sweep["by_kernel_size"][str(kernel_size)] if "by_kernel_size" in sweep and kernel_size is not None and str(kernel_size) in sweep.get("by_kernel_size", {}) else sweep.get("rows")
    if rows is None:
        return
    valid_rows = [r for r in rows if r["kernel_ms"] is not None]
    ordered = sorted(valid_rows, key=lambda r: -r["kernel_ms"]["mean"])  # slowest first, matching "Basic -> ... -> Enhanced"

    st.markdown("**Optimization progression (measured, slowest → fastest)**")
    steps = " → ".join(f"{r['variant']} ({r['kernel_ms']['mean']:.3f} ms)" for r in ordered)
    st.markdown(steps)


def render_rejected_variants(sweep: dict) -> None:
    """Spec items 15, 31: rejected variants (measured but not chosen as
    production) with a real reason, never implying CUDA itself failed."""
    if sweep is None or not sweep.get("rejected_notes"):
        return
    rows_by_variant = {r["variant"]: r for r in sweep.get("rows", [])} if "rows" in sweep else {}
    st.markdown("**Rejected / non-default variants**")
    for variant, note in sweep["rejected_notes"].items():
        measured = rows_by_variant.get(variant)
        speedup_str = f" (measured {measured['speedup_vs_basic']:.2f}× vs Basic)" if measured and measured.get("speedup_vs_basic") else ""
        st.markdown(f"- **{variant}**{speedup_str}: {note}")


def render_why_it_works(filter_name: str) -> None:
    st.markdown("**Why it works**")
    st.markdown(WHY_IT_WORKS.get(filter_name, ""))


def render_technique_cards(filter_name: str) -> None:
    techniques = TECHNIQUE_CATEGORIES.get(filter_name, [])
    if not techniques:
        return
    cols = st.columns(len(techniques))
    for col, technique in zip(cols, techniques):
        with col:
            st.info(technique)


def render_fusion_experiment(fusion: Optional[dict]) -> None:
    """Spec item 19: only rendered if the fusion experiment was actually
    measured/backfilled -- never invented."""
    if fusion is None:
        st.caption("No Laplacian+Threshold fusion experiment data available.")
        return
    st.markdown("**EXPERIMENTAL — Laplacian+Threshold fusion**")
    st.caption("Not wired into the production pipeline -- kept as a tested, benchmarked capability that "
               "sacrifices per-stage introspection (no standalone Laplacian output) for a modest additional gain.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Unfused (2 launches)", f"{fusion['unfused_combined_kernel_ms']['mean']:.4f} ms")
    col2.metric("Fused (1 launch)", f"{fusion['fused_kernel_ms']['mean']:.4f} ms")
    col3.metric("Combined-kernel speedup", f"{fusion['combined_kernel_speedup']:.2f}×" if fusion['combined_kernel_speedup'] else "n/a")
    st.caption(f"Correctness vs. unfused: {'✅ bit-exact' if fusion['bit_exact_vs_unfused'] else '⚠️ NOT bit-exact'} "
               f"(max_abs_diff={fusion['max_abs_diff_fused_vs_unfused']})")


def render_live_variant_controls(filter_name: str, variants: tuple, production_default: str) -> tuple:
    st.markdown("**Live Experiment**")
    labels = [f"{v} (production default)" if v == production_default else
              ("basic (no optimization)" if v == "basic" else f"{v} (experimental)") for v in variants]
    col1, col2 = st.columns(2)
    with col1:
        a_label = st.selectbox("Variant A", labels, index=0, key=f"opt_lab_variant_a_{filter_name}")
    with col2:
        default_b_idx = variants.index(production_default) if production_default in variants else min(1, len(variants) - 1)
        b_label = st.selectbox("Variant B", labels, index=default_b_idx, key=f"opt_lab_variant_b_{filter_name}")
    variant_a = variants[labels.index(a_label)]
    variant_b = variants[labels.index(b_label)]
    return variant_a, variant_b


def render_experimental_configuration_panel(filter_name: str, variant_a: str, variant_b: str) -> None:
    st.markdown("**EXPERIMENTAL CONFIGURATION** (this live run only -- production defaults are unaffected)")
    col1, col2 = st.columns(2)
    col1.write(f"Variant A: `{variant_a}`")
    col2.write(f"Variant B: `{variant_b}`")


def render_live_result_card(comparison) -> None:
    """Spec item 25: Variant A/B kernel times + measured speedup +
    correctness, from THIS run's ComparisonResult only."""
    performance_speedup = comparison.speedup_a_over_b
    st.markdown("### 🔴 LIVE EXPERIMENT — Variant Comparison")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"Variant A: {comparison.variant_a}", f"{comparison.result_a.kernel_ms_mean:.4f} ms")
    with col2:
        st.metric(f"Variant B: {comparison.variant_b}", f"{comparison.result_b.kernel_ms_mean:.4f} ms")
    with col3:
        faster = comparison.variant_a if performance_speedup >= 1 else comparison.variant_b
        shown = performance_speedup if performance_speedup >= 1 else (1 / performance_speedup if performance_speedup > 0 else 0.0)
        st.metric("Measured speedup", f"{shown:.2f}×", help=f"{faster} is faster, measured directly from these {comparison.measurement_runs} runs.")

    st.caption(f"{comparison.n_images} image(s)  ·  {comparison.warmup_runs} warmup + {comparison.measurement_runs} measured runs "
               f"(same input for both variants)")


def render_live_correctness(comparison) -> None:
    """Spec item 26: max_abs_diff/mean_abs_diff/RMSE/differing pixels
    for A vs B and each vs CPU, computed from THIS run."""
    st.markdown("**Correctness**")
    a_vs_b = comparison.correctness_a_vs_b
    exact = a_vs_b["max_abs_diff"] == 0
    status_text = "✅ PASS (bit-exact)" if exact else f"⚠️ max_abs_diff={a_vs_b['max_abs_diff']}"
    st.markdown(f"**{comparison.variant_a} vs {comparison.variant_b}:** {status_text}")

    cols = st.columns(2)
    for col, (label, metrics) in zip(cols, [
        (f"{comparison.variant_a} vs CPU", comparison.correctness_a_vs_cpu),
        (f"{comparison.variant_b} vs CPU", comparison.correctness_b_vs_cpu),
    ]):
        with col:
            st.markdown(f"_{label}_")
            st.write(f"max_abs_diff = {metrics['max_abs_diff']}")
            st.write(f"mean_abs_diff = {metrics['mean_abs_diff']:.4f}")
            st.write(f"RMSE = {metrics['rmse']:.4f}")
            st.write(f"differing pixels = {metrics['differing_pixel_count']:,} ({metrics['differing_pixel_percentage']:.4f}%)")


def render_pipeline_impact_panel(impact) -> None:
    """Spec item 33: measured (never inferred) whole-pipeline effect of
    this one filter's enhancement."""
    st.markdown("**Pipeline impact**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Basic 5-filter pipeline", f"{impact.basic_pipeline_ms:.3f} ms")
    col2.metric(f"Pipeline with {impact.variant} {impact.filter_name}", f"{impact.enhanced_pipeline_ms:.3f} ms")
    col3.metric("Pipeline improvement", f"{impact.improvement:.3f}×")
    st.caption("Measured directly (both pipelines actually run) -- not inferred by multiplying the isolated "
               "per-filter speedup.")


def render_amdahl_connection(filter_name: str, per_filter_rows: Optional[list]) -> None:
    """Spec item 32: links this filter's optimization to its share of
    the measured whole-pipeline compute-time reduction."""
    if not per_filter_rows:
        st.caption("No historical per-filter contribution data available (Performance Analytics tab).")
        return
    row = next((r for r in per_filter_rows if r["filter"] == filter_name), None)
    if row is None:
        return
    st.markdown("**Connection to the full pipeline (Performance Analytics)**")
    st.metric(f"{FILTER_LABELS[filter_name]}'s share of total measured compute-time reduction",
              f"{row['pct_of_total_compute_reduction']:.1f}%")
    gaussian_median_pct = sum(r["pct_of_total_compute_reduction"] for r in per_filter_rows if r["filter"] in ("gaussian", "median"))
    st.caption(f"For context: Gaussian + Median together account for ≈{gaussian_median_pct:.0f}% of the measured "
               f"compute-time reduction (canonical benchmark). Optimizing a filter matters in proportion to how "
               f"much time the application actually spends there.")


def render_technical_detail_panel(filter_name: str, variant: str) -> None:
    """Spec item 40: only already-known facts -- N/A for anything not
    actually measured/documented, never a fabricated profiler value."""
    with st.expander("Show Technical Details"):
        col1, col2, col3 = st.columns(3)
        col1.write(f"**CUDA variant:** {variant}")
        col2.write("**Block dimensions:** 16×16" if variant != "basic" else "**Block dimensions:** N/A (Basic uses a fixed internal launch configuration)")
        launches = "2 (separable two-pass)" if filter_name == "gaussian" and variant in ("naive", "shared", "shared_const", "specialized") else "1"
        col3.write(f"**Kernel launches:** {launches}")
        st.caption("Registers/thread and shared-memory-per-block are only available from this project's "
                   "`nvcc -Xptxas -v` profiling notes (see README) rather than at Streamlit runtime, since this "
                   "lab never compiles or profiles from the UI (spec item 41).")
