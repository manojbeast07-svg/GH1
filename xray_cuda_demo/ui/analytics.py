"""Batch-size / resolution analytics + benchmark history (Section 12
spec items 24-26, 40, 42, 57). All charts are built directly from
stored Section 11 JSON rows -- nothing is smoothed, interpolated, or
recomputed (spec item 25's explicit "do not smooth or interpolate").
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_batch_sweep(batch_sweep: dict) -> None:
    rows = batch_sweep["rows"]
    df = pd.DataFrame([{
        "Batch size": r["effective_batch_size"],
        "CPU (ms)": r["cpu_total_ms"]["mean"],
        "Basic CUDA (ms)": r["basic_total_ms"]["mean"],
        "Enhanced CUDA (ms)": r["enhanced_total_ms"]["mean"],
        "CPU img/s": r["cpu_images_per_second"],
        "Basic img/s": r["basic_images_per_second"],
        "Enhanced img/s": r["enhanced_images_per_second"],
        "Enhanced vs Basic gain": r["enhanced_vs_basic_gain"],
    } for r in rows])

    st.caption(f"benchmark_id: `{batch_sweep['benchmark_id']}`  ·  resolution: "
               f"{batch_sweep['resolution'][1]}×{batch_sweep['resolution'][0]}  ·  seed={batch_sweep['seed']}")

    col1, col2 = st.columns(2)
    with col1:
        melted = df.melt(id_vars="Batch size", value_vars=["CPU img/s", "Basic img/s", "Enhanced img/s"],
                          var_name="Implementation", value_name="Images/sec")
        fig = px.line(melted, x="Batch size", y="Images/sec", color="Implementation", markers=True,
                       log_x=True, title="Batch size vs. throughput")
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig2 = px.line(df, x="Batch size", y="Enhanced vs Basic gain", markers=True, log_x=True,
                        title="Batch size vs. Enhanced-vs-Basic speedup")
        fig2.add_hline(y=1.0, line_dash="dot", annotation_text="1.0× (no gain)")
        st.plotly_chart(fig2, width="stretch")

    st.markdown("**GPU advantage depends on workload size** -- tiny batches can be dominated by fixed launch "
                "overhead (see the batch=1 row below, often close to or below 1.0×), while the advantage grows "
                "and stabilizes as batch size increases.")
    st.dataframe(df.style.format({
        "CPU (ms)": "{:.3f}", "Basic CUDA (ms)": "{:.3f}", "Enhanced CUDA (ms)": "{:.3f}",
        "CPU img/s": "{:.1f}", "Basic img/s": "{:.1f}", "Enhanced img/s": "{:.1f}", "Enhanced vs Basic gain": "{:.3f}×",
    }), width="stretch", hide_index=True)


def render_batch_sweep_speedup_chart(batch_sweep: dict) -> None:
    """Spec item 21: CPU/Basic and CPU/Enhanced speedup vs. batch size,
    directly from each row's already-computed `basic_speedup_vs_cpu` /
    `enhanced_speedup_vs_cpu` fields -- no re-derivation, raw (not
    smoothed) data, so overhead effects at small batch sizes stay
    visible."""
    rows = batch_sweep["rows"]
    df = pd.DataFrame([{
        "Batch size": r["effective_batch_size"],
        "CPU / Basic CUDA": r["basic_speedup_vs_cpu"],
        "CPU / Enhanced CUDA": r["enhanced_speedup_vs_cpu"],
    } for r in rows])
    melted = df.melt(id_vars="Batch size", value_vars=["CPU / Basic CUDA", "CPU / Enhanced CUDA"],
                      var_name="Comparison", value_name="Speedup (×)")
    fig = px.line(melted, x="Batch size", y="Speedup (×)", color="Comparison", markers=True, log_x=True,
                  title="Batch size vs. speedup (CPU / Basic CUDA, CPU / Enhanced CUDA)")
    fig.add_hline(y=1.0, line_dash="dot", annotation_text="1.0× (no speedup)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Raw measured points, not smoothed or interpolated -- overhead-dominated batch sizes (often small "
               "batches close to or below 1.0×) are visible exactly as measured.")


def render_resolution_sweep(resolution_sweep: dict) -> None:
    rows = resolution_sweep["rows"]
    df = pd.DataFrame([{
        "Resolution": f"{r['width']}×{r['height']}",
        "Pixels/image": r["pixel_count"],
        "Images": r["image_count"],
        "CPU ms/image": r["cpu_ms_per_image"],
        "Basic ms/image": r["basic_ms_per_image"],
        "Enhanced ms/image": r["enhanced_ms_per_image"],
        "CPU img/s": r["cpu_images_per_second"],
        "Basic img/s": r["basic_images_per_second"],
        "Enhanced img/s": r["enhanced_images_per_second"],
    } for r in rows])

    st.caption(f"benchmark_id: `{resolution_sweep['benchmark_id']}`  ·  seed={resolution_sweep['seed']}  ·  "
               f"{len(rows)} resolution group(s), never averaged together")

    melted = df.melt(id_vars=["Resolution", "Pixels/image"],
                      value_vars=["CPU ms/image", "Basic ms/image", "Enhanced ms/image"],
                      var_name="Implementation", value_name="ms/image")
    fig = px.bar(melted.sort_values("Pixels/image"), x="Resolution", y="ms/image", color="Implementation",
                 barmode="group", title="Resolution vs. ms/image")
    st.plotly_chart(fig, width="stretch")

    st.dataframe(df.style.format({
        "Pixels/image": "{:,}", "CPU ms/image": "{:.4f}", "Basic ms/image": "{:.4f}", "Enhanced ms/image": "{:.4f}",
        "CPU img/s": "{:.1f}", "Basic img/s": "{:.1f}", "Enhanced img/s": "{:.1f}",
    }), width="stretch", hide_index=True)


def render_resolution_throughput_chart(resolution_sweep: dict) -> None:
    """Spec item 24: resolution vs. images/sec for all three
    implementations, one bar group per resolution -- never averaged
    across resolution groups."""
    rows = resolution_sweep["rows"]
    df = pd.DataFrame([{
        "Resolution": f"{r['width']}×{r['height']}",
        "Pixels/image": r["pixel_count"],
        "CPU img/s": r["cpu_images_per_second"],
        "Basic img/s": r["basic_images_per_second"],
        "Enhanced img/s": r["enhanced_images_per_second"],
    } for r in rows])
    melted = df.melt(id_vars=["Resolution", "Pixels/image"], value_vars=["CPU img/s", "Basic img/s", "Enhanced img/s"],
                      var_name="Implementation", value_name="Images/sec")
    fig = px.bar(melted.sort_values("Pixels/image"), x="Resolution", y="Images/sec", color="Implementation",
                 barmode="group", title="Resolution vs. throughput")
    st.plotly_chart(fig, width="stretch")


def render_resolution_scaling_note() -> None:
    """Spec item 25: an explicit, hedged note -- not a guaranteed rule
    for every configuration, only what the measured data supports."""
    with st.expander("How does resolution affect GPU acceleration?"):
        st.markdown(
            "GPU parallelism generally becomes more useful as the number of pixels processed per image "
            "increases, but transfer and launch overhead can dominate smaller workloads. This is not a "
            "guaranteed rule for every hardware configuration -- the chart above shows what was actually "
            "measured on this system for the resolution groups present in the dataset."
        )


IMPLEMENTATION_ORDER = ["CPU", "Basic CUDA", "Enhanced CUDA"]


def resolve_images_per_second(canonical: dict, explicit_key: str, mean_ms: Optional[float]) -> Optional[float]:
    """Looks up an already-stored `*_images_per_second` field, falling
    back to deriving it from the stored mean total time + image count
    when an older summary (written before that field existed) doesn't
    have it -- never a fabricated value, both inputs are themselves
    stored measurements. Returns None only when neither is available,
    so callers can show a missing-data notice instead of crashing."""
    if explicit_key in canonical:
        return canonical[explicit_key]
    image_count = canonical.get("manifest", {}).get("selected_image_count")
    if image_count and mean_ms and mean_ms > 0:
        return image_count / (mean_ms / 1000.0)
    return None


def render_total_and_throughput_charts(canonical: dict) -> None:
    """Chart 1 (total time) + Chart 2 (throughput) for the selected
    historical benchmark -- Section 16 spec items 11-12. Legend/x-axis
    order is always CPU / Basic CUDA / Enhanced CUDA (spec item 40:
    never randomly reordered)."""
    cpu_ms = canonical["cpu"]["mode4_end_to_end_ms"]["mean"]
    basic_ms = canonical["basic_cuda"]["mode4_end_to_end_ms"]["mean"]
    enhanced_ms = canonical["enhanced_cuda"]["mode4_end_to_end_ms"]["mean"]

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Bar(x=IMPLEMENTATION_ORDER, y=[cpu_ms, basic_ms, enhanced_ms],
                                text=[f"{v:.3f} ms" for v in (cpu_ms, basic_ms, enhanced_ms)]))
        fig.update_traces(textposition="outside")
        fig.update_layout(title="CPU vs Basic CUDA vs Enhanced CUDA — Total Execution Time", yaxis_title="Total time (ms)")
        st.plotly_chart(fig, width="stretch")
    with col2:
        ips = [resolve_images_per_second(canonical, "cpu_images_per_second", cpu_ms),
               resolve_images_per_second(canonical, "basic_images_per_second", basic_ms),
               resolve_images_per_second(canonical, "enhanced_images_per_second", enhanced_ms)]
        if any(v is None for v in ips):
            render_missing_artifact("throughput could not be determined for one or more implementations.")
        else:
            fig2 = go.Figure(go.Bar(x=IMPLEMENTATION_ORDER, y=ips, text=[f"{v:.1f}" for v in ips]))
            fig2.update_traces(textposition="outside")
            fig2.update_layout(title="CPU vs GPU Throughput", yaxis_title="Images / second")
            st.plotly_chart(fig2, width="stretch")


def render_gpu_compute_breakdown_chart(basic_breakdown: Optional[dict], enhanced_breakdown: Optional[dict]) -> None:
    """Chart 3 (spec item 13): stacked H2D / per-filter / D2H timing for
    Basic and Enhanced CUDA, from averaged (never re-measured) per-run
    raw data -- makes visible where GPU time actually goes."""
    if not basic_breakdown and not enhanced_breakdown:
        st.warning("⚠ Historical result unavailable — no raw per-run GPU timing stored for this benchmark.")
        return

    segments = [("H2D", "h2d_ms"), ("Gaussian", "gaussian_ms"), ("Median", "median_ms"), ("Sobel", "sobel_ms"),
                ("Laplacian", "laplacian_ms"), ("Threshold", "threshold_ms"), ("D2H", "d2h_ms")]
    fig = go.Figure()
    for label, key in segments:
        fig.add_trace(go.Bar(
            name=label,
            x=["Basic CUDA", "Enhanced CUDA"],
            y=[
                (basic_breakdown or {}).get(key) or 0.0,
                (enhanced_breakdown or {}).get(key) or 0.0,
            ],
        ))
    fig.update_layout(barmode="stack", title="Where does GPU time go? (H2D / per-filter kernels / D2H)",
                       yaxis_title="ms (mean per batch, averaged across measurement runs)")
    st.plotly_chart(fig, width="stretch")

    n_runs = (basic_breakdown or enhanced_breakdown or {}).get("n_runs")
    if n_runs:
        st.caption(f"Averaged across {n_runs} measurement run(s), from the stored raw per-run timing.")


def render_benchmark_history(benchmark_ids: list, load_manifest_fn) -> Optional[str]:
    """Spec item 57: a selectable table of historical canonical
    benchmarks; returns the benchmark_id the user picked, if any."""
    if not benchmark_ids:
        st.info("No canonical benchmarks found yet. Run `python scripts/run_final_benchmark.py` to create one.")
        return None

    rows = []
    for bid in benchmark_ids:
        manifest = load_manifest_fn(bid)
        if manifest is None:
            continue
        rows.append({
            "Benchmark ID": bid, "Date (UTC)": manifest["timestamp_utc"][:19],
            "Images": manifest["selected_image_count"], "Seed": manifest["seed"],
            "Resolution": f"{manifest['resolution'][1]}×{manifest['resolution'][0]}",
            "GPU": manifest["environment"].get("gpu_name") or "n/a",
            "Status": manifest["status"],
        })
    if not rows:
        st.info("No readable benchmark manifests found.")
        return None

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)
    return st.selectbox("Select a benchmark to inspect", options=[r["Benchmark ID"] for r in rows])


def render_missing_artifact(message: str) -> None:
    """Spec item 39: a consistent 'unavailable' notice -- never a crash,
    never a fabricated substitute value."""
    st.warning(f"⚠ Historical result unavailable — {message}")


def render_canonical_badge(is_canonical: bool) -> None:
    """Spec item 32."""
    if is_canonical:
        st.success("★ CANONICAL BENCHMARK")
    else:
        st.info("HISTORICAL EXPERIMENT")


def render_benchmark_metadata_panel(meta: dict, is_canonical: bool) -> None:
    """Spec items 6, 29: the full metadata panel for the selected
    historical benchmark."""
    render_canonical_badge(is_canonical)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Benchmark ID", meta["benchmark_id"])
    col1.caption(f"Date: {meta['timestamp_utc'][:19]} UTC")
    col2.metric("Images", meta["image_count"])
    col2.caption(f"Seed: {meta['seed']}")
    res = meta.get("resolution")
    col3.metric("Resolution", f"{res[1]}×{res[0]}" if res else "n/a")
    col3.caption(f"Runs: {meta['runs']}  (+{meta['warmup_runs']} warmup)")
    col4.metric("GPU", meta.get("gpu_name") or "n/a")
    col4.caption(f"CUDA {meta.get('cuda_runtime_version') or 'n/a'}  ·  Driver {meta.get('gpu_driver_version') or 'n/a'}")
    with st.expander("Full metadata"):
        st.write(f"**Dataset fingerprint:** `{meta.get('dataset_fingerprint') or 'n/a'}`")
        st.write(f"**CPU:** {meta.get('cpu_model') or 'n/a'}")


def render_run_statistics(canonical: dict) -> None:
    """Spec items 30-31: mean/median/min/max/std for CPU/Basic/Enhanced
    total time -- already-stored AggregatedStat fields, not
    recomputed, so variability is never hidden."""
    rows = []
    for label, key in zip(IMPLEMENTATION_ORDER, ("cpu", "basic_cuda", "enhanced_cuda")):
        stat = canonical[key]["mode4_end_to_end_ms"]
        rows.append({
            "Implementation": label, "Mean (ms)": stat["mean"], "Median (ms)": stat["median"],
            "Min (ms)": stat["min"], "Max (ms)": stat["max"], "Std Dev (ms)": stat["std"], "n": stat["n"],
        })
    st.dataframe(pd.DataFrame(rows).style.format({
        "Mean (ms)": "{:.4f}", "Median (ms)": "{:.4f}", "Min (ms)": "{:.4f}", "Max (ms)": "{:.4f}", "Std Dev (ms)": "{:.4f}",
    }), width="stretch", hide_index=True)


def render_raw_measurements_expander(
    cpu_raw: Optional[dict], basic_raw: Optional[dict], enhanced_raw: Optional[dict],
) -> None:
    """Spec item 30: individual per-run measurements -- variability is
    shown, not hidden."""
    with st.expander("Raw Measurements (every individual run)"):
        if cpu_raw:
            st.markdown("**CPU** (`processing_ms` per run)")
            st.dataframe(pd.DataFrame(cpu_raw["runs"]), width="stretch", hide_index=True)
        if basic_raw:
            st.markdown("**Basic CUDA** (per run)")
            st.dataframe(pd.DataFrame(basic_raw["runs"]), width="stretch", hide_index=True)
        if enhanced_raw:
            st.markdown("**Enhanced CUDA** (per run)")
            st.dataframe(pd.DataFrame(enhanced_raw["runs"]), width="stretch", hide_index=True)
        if not (cpu_raw or basic_raw or enhanced_raw):
            render_missing_artifact("no raw per-run measurements stored for this benchmark.")


def render_contribution_summary(rows: list) -> None:
    """Spec items 17-18: the higher-level Gaussian / Median / other-
    filters grouping of measured compute-time reduction, explicitly
    labeled as a measured analysis, not a theoretical prediction."""
    reduction_by_filter = {r["filter"]: r["absolute_reduction_ms"] for r in rows}
    gaussian = reduction_by_filter.get("gaussian", 0.0)
    median = reduction_by_filter.get("median", 0.0)
    other = sum(v for k, v in reduction_by_filter.items() if k not in ("gaussian", "median"))
    total = gaussian + median + other

    st.markdown("**Measured contribution analysis**")
    df = pd.DataFrame([
        {"Group": "Gaussian", "% of total compute reduction": 100.0 * gaussian / total if total > 0 else 0.0},
        {"Group": "Median", "% of total compute reduction": 100.0 * median / total if total > 0 else 0.0},
        {"Group": "Sobel + Laplacian + Threshold", "% of total compute reduction": 100.0 * other / total if total > 0 else 0.0},
    ])
    fig = px.pie(df, names="Group", values="% of total compute reduction", hole=0.45,
                 title="Measured contribution to total compute-time reduction")
    st.plotly_chart(fig, width="stretch")

    gaussian_pct = 100.0 * gaussian / total if total > 0 else 0.0
    median_pct = 100.0 * median / total if total > 0 else 0.0
    st.info(f"**Gaussian and Median account for approximately {gaussian_pct + median_pct:.0f}%** of the measured "
            f"compute-time reduction (Gaussian ≈{gaussian_pct:.0f}%, Median ≈{median_pct:.0f}%).")


def render_benchmark_comparison(comparison: dict) -> None:
    """Spec items 33-34: Benchmark A vs Benchmark B, with an explicit
    warning when the underlying configurations differ (never implying
    two different datasets/GPUs/resolutions are directly comparable
    without saying so)."""
    summary_a, summary_b = comparison["summary_a"], comparison["summary_b"]
    if comparison["differences"]:
        st.warning("⚠ These benchmarks were run under different configurations -- the comparison below may not "
                    "be apples-to-apples:\n\n" + "\n".join(f"- {d}" for d in comparison["differences"]))
    else:
        st.success("Configurations match (same GPU, dataset, image count, resolution, filter/CUDA config).")

    rows = []
    for label, key in zip(IMPLEMENTATION_ORDER, ("cpu", "basic_cuda", "enhanced_cuda")):
        rows.append({
            "Implementation": label,
            f"A: {summary_a['benchmark_id']} (ms)": summary_a[key]["mode4_end_to_end_ms"]["mean"],
            f"B: {summary_b['benchmark_id']} (ms)": summary_b[key]["mode4_end_to_end_ms"]["mean"],
        })
    df = pd.DataFrame(rows)
    numeric_cols = [c for c in df.columns if c != "Implementation"]
    st.dataframe(df.style.format({c: "{:.3f}" for c in numeric_cols}), width="stretch", hide_index=True)

    melted = df.melt(id_vars="Implementation", var_name="Benchmark", value_name="ms")
    fig = px.bar(melted, x="Implementation", y="ms", color="Benchmark", barmode="group",
                 title="Benchmark A vs Benchmark B — total execution time")
    st.plotly_chart(fig, width="stretch")


def render_export_buttons(summary: dict, csv_text: str, json_text: str) -> None:
    """Spec item 36: export the SELECTED benchmark's stored data --
    never modifies the source file."""
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("⬇ Export Selected Benchmark (CSV)", data=csv_text,
                            file_name=f"{summary['benchmark_id']}.csv", mime="text/csv")
    with col2:
        st.download_button("⬇ Export Selected Benchmark (JSON)", data=json_text,
                            file_name=f"{summary['benchmark_id']}.json", mime="application/json")
