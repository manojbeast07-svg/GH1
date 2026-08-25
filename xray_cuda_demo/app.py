"""Streamlit entry point: the final CUDA X-Ray Processing Lab (Section 12).

Launch:
    streamlit run app.py

Thin orchestrator only -- every processing/benchmark call goes through
ui/services.py, which is the sole caller into the existing, already-
tested backend (DatasetManager, FilterConfig, cpu.pipeline,
cuda.pipeline, cuda.final_benchmark). No filter/kernel/benchmark logic
lives in this file or anywhere under ui/.
"""

from __future__ import annotations

import time
import traceback
from datetime import datetime, timezone

import streamlit as st

from ui import analytics, controls, correctness, images, optimization_lab, performance, services, state, theme
from ui import presentation as presentation_ui
from ui import threading_view

st.set_page_config(page_title="CUDA X-Ray Processing Lab", layout="wide")
state.init_session_state()
theme.inject_css()


def _show_error(exc: Exception) -> None:
    if state.get("debug_mode"):
        st.error(str(exc))
        st.code(traceback.format_exc())
    else:
        st.error(str(exc))


# -- Section 16: cached historical benchmark loaders -----------------------------------------------------
# Historical benchmark_results/ artifacts are static once written, so they're safe to cache across
# reruns (spec item 38). Live/GPU state (compare_batch, process_single_image, etc.) is never cached.


@st.cache_data(show_spinner=False)
def _load_benchmark_summary_cached(benchmark_id):
    return services.load_benchmark_summary(benchmark_id)


@st.cache_data(show_spinner=False)
def _load_batch_sweep_cached(benchmark_id):
    return services.load_batch_sweep(benchmark_id)


@st.cache_data(show_spinner=False)
def _load_resolution_sweep_cached(benchmark_id):
    return services.load_resolution_sweep(benchmark_id)


@st.cache_data(show_spinner=False)
def _load_per_filter_cached(benchmark_id):
    return services.load_per_filter_results(benchmark_id)


@st.cache_data(show_spinner=False)
def _load_correctness_cached(benchmark_id):
    return services.load_correctness_results(benchmark_id)


@st.cache_data(show_spinner=False)
def _gpu_compute_breakdown_cached(benchmark_id, implementation):
    return services.gpu_compute_breakdown(benchmark_id, implementation)


@st.cache_data(show_spinner=False)
def _load_raw_runs_cached(benchmark_id, implementation):
    return services.load_raw_runs(benchmark_id, implementation)


# -- header: the story in under 30 seconds (spec item 43) -----------------------------------------------------

st.title("🩻 CUDA X-Ray Processing Lab")
st.caption("Python/OpenCV CPU → Basic CUDA C++ → Enhanced CUDA C++")

# -- live status badges (spec item 5): queried fresh on this render, reflecting this session's
# actual current state -- the "Dataset" badge is rendered after the scan below (line ~130), once
# a real DatasetManager exists, rather than guessed here before the sidebar has even run.
_gpu_ok = services.cuda_available()
_status_badges_placeholder = st.empty()
_status_badges_placeholder.markdown(" ".join([
    theme.render_status_badge("GPU", "Available" if _gpu_ok else "Unavailable", "pass" if _gpu_ok else "error"),
    theme.render_status_badge("CUDA", "Active" if _gpu_ok else "Inactive", "pass" if _gpu_ok else "error"),
    theme.render_status_badge("Backend", "Native C++/CUDA", "pass"),
    theme.render_status_badge("Dataset", "Pending selection", "warning"),
]), unsafe_allow_html=True)

_summary = services.load_benchmark_summary()
if _summary:
    canonical_desc = (
        f"{_summary['manifest']['selected_image_count']} real X-rays · "
        f"{_summary['manifest']['resolution'][1]}×{_summary['manifest']['resolution'][0]} · "
        f"seed {_summary['manifest']['seed']}"
    )
    st.caption(f"Canonical benchmark: {canonical_desc}  (`{_summary['benchmark_id']}`)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CPU", f"{_summary['cpu']['mode4_end_to_end_ms']['mean']:.1f} ms")
    col2.metric("Basic CUDA", f"{_summary['basic_cuda']['mode4_end_to_end_ms']['mean']:.1f} ms",
                delta=f"{_summary['speedups']['basic_vs_cpu']:.2f}× vs CPU")
    col3.metric("Enhanced CUDA", f"{_summary['enhanced_cuda']['mode4_end_to_end_ms']['mean']:.1f} ms",
                delta=f"{_summary['speedups']['enhanced_vs_cpu']:.2f}× vs CPU")
    col4.metric("GPU compute optimization", f"{_summary['speedups']['enhanced_vs_basic_compute_only']:.2f}×",
                help="Basic→Enhanced, GPU compute time only (H2D+kernels+D2H), excluding disk I/O")
else:
    st.info("No canonical benchmark found yet. Run `python scripts/run_final_benchmark.py` to populate the "
            "Performance Analytics tab, or use Live Processing below with your own dataset now.")

st.markdown("---")

# -- sidebar -----------------------------------------------------

controls.render_presentation_and_debug_toggles()
dataset_path = controls.render_dataset_controls()

if dataset_path is None:
    st.stop()

try:
    dm, dataset_info = services.scan_dataset(dataset_path)
except services.ServiceError as exc:
    _show_error(exc)
    st.stop()

_status_badges_placeholder.markdown(" ".join([
    theme.render_status_badge("GPU", "Available" if _gpu_ok else "Unavailable", "pass" if _gpu_ok else "error"),
    theme.render_status_badge("CUDA", "Active" if _gpu_ok else "Inactive", "pass" if _gpu_ok else "error"),
    theme.render_status_badge("Backend", "Native C++/CUDA", "pass"),
    theme.render_status_badge("Dataset", f"Ready ({dataset_info.total_files:,} images)", "pass"),
]), unsafe_allow_html=True)

selection = controls.render_selection_controls(dm)
if selection is None:
    st.stop()

controls.render_filter_controls()
controls.render_implementation_controls()
controls.render_compare_controls()

filter_config = state.current_filter_config()

try:
    _load_start = time.perf_counter()
    images_all = services.load_selection_images(selection)
    selection_load_ms = (time.perf_counter() - _load_start) * 1000.0
except services.ServiceError as exc:
    _show_error(exc)
    st.stop()

groups = services.group_images_by_shape(images_all)
dominant_shape = max(groups, key=lambda k: len(groups[k]))
images_main = [images_all[i] for i in groups[dominant_shape]]

st.sidebar.markdown("---")
st.sidebar.caption(f"Selected: {len(selection)} images  ·  Dataset: {dataset_info.total_files:,} images")
st.sidebar.caption(f"Seed: {selection.seed}" if selection.seed is not None else "Single image")
st.sidebar.caption(f"Dominant resolution: {dominant_shape[1]}×{dominant_shape[0]} ({len(images_main)} images)")
if len(groups) > 1:
    with st.sidebar.expander("Mixed resolutions (never resized)"):
        for shape, idxs in groups.items():
            st.caption(f"{shape[1]}×{shape[0]}: {len(idxs)} image(s)")

presentation = state.get("presentation_mode", False)

# -- tabs -----------------------------------------------------

if presentation:
    tab_names = ["Live Processing", "CPU vs GPU", "Performance Analytics", "Threading & Parallelism",
                 "Correctness", "Presentation Mode"]
else:
    tab_names = ["Live Processing", "CPU vs GPU", "Performance Analytics", "Optimization Lab",
                 "Threading & Parallelism", "Correctness", "System", "Presentation Mode"]

tabs = st.tabs(tab_names)
tab_map = dict(zip(tab_names, tabs))


# ================================================================
# TAB: Live Processing
# ================================================================
with tab_map["Live Processing"]:
    st.markdown("### Pipeline")
    stage_flags = {
        "gaussian": filter_config.gaussian_enabled, "median": filter_config.median_enabled,
        "sobel": filter_config.sobel_enabled, "laplacian": filter_config.laplacian_enabled,
        "threshold": filter_config.threshold_enabled,
    }
    diagram = " → ".join(
        (name.capitalize() if enabled else f"~~{name.capitalize()}~~")
        for name, enabled in stage_flags.items()
    )
    st.markdown(f"X-ray → {diagram} → Output")

    is_single_image = state.get("selection_mode") == "Single image"

    if is_single_image:
        # ============================================================
        # Section 14: single-image live processing / comparison
        # ============================================================
        image = images_main[0]
        item = selection.items[0] if selection.items else None
        image_meta = services.build_image_metadata(image, item)

        st.markdown("### Selected X-ray")
        meta_cols = st.columns(4)
        meta_cols[0].metric("Filename", image_meta.filename)
        meta_cols[1].metric("Width", image_meta.width)
        meta_cols[2].metric("Height", image_meta.height)
        meta_cols[3].metric("dtype", image_meta.dtype)
        if image_meta.relative_path:
            st.caption(f"Path: {image_meta.relative_path}")
        st.caption(f"Image loaded in {selection_load_ms:.3f} ms (measured separately from any processing time).")

        display_mode = st.radio("Original X-ray display", ["Fit to container", "Actual size"], horizontal=True)
        st.markdown("**Original X-ray**")
        if display_mode == "Fit to container":
            st.image(image, clamp=True, channels="GRAY", width="stretch")
        else:
            st.image(image, clamp=True, channels="GRAY")

        col1, col2 = st.columns(2)
        with col1:
            process_clicked = st.button(f"▶ Process Selected Image ({state.get('implementation')})", type="primary")
        with col2:
            compare_clicked = st.button("⚖ Compare CPU vs Basic CUDA vs Enhanced CUDA", type="primary")

        if process_clicked:
            impl_label = state.get("implementation")
            try:
                with st.spinner(f"Processing with {impl_label}..."):
                    result = services.process_single_image(image, filter_config, impl_label)
                state.set_value("last_selected_image", image)
                state.set_value("last_image_meta", image_meta)
                slot = {"CPU": "last_cpu_result", "Basic CUDA": "last_basic_result", "Enhanced CUDA": "last_enhanced_result"}[impl_label]
                state.set_value(slot, result)
                state.set_value("last_run_timestamp", datetime.now(timezone.utc).isoformat())
            except services.ServiceError as exc:
                _show_error(exc)

        if compare_clicked:
            try:
                with st.spinner("Running CPU, Basic CUDA, and Enhanced CUDA on the same image..."):
                    comparison = services.compare_single_image(image, filter_config, image_meta, load_ms=selection_load_ms)
                state.set_value("last_comparison", comparison)
                state.set_value("last_selected_image", image)
                state.set_value("last_image_meta", image_meta)
                state.set_value("last_run_timestamp", comparison.timestamp_utc)
            except services.ServiceError as exc:
                _show_error(exc)

        single_label = state.get("implementation")
        single_slot = {"CPU": "last_cpu_result", "Basic CUDA": "last_basic_result", "Enhanced CUDA": "last_enhanced_result"}[single_label]
        single_result = state.get(single_slot)
        if single_result is not None and state.get("last_comparison") is None:
            performance.render_live_vs_canonical_label(is_live=True)
            st.markdown(f"### {single_label} result")
            performance.render_metric_cards({single_label: single_result.total_ms})
            if single_result.h2d_ms is not None:
                st.caption(f"H2D: {single_result.h2d_ms:.4f} ms · Kernel: {single_result.compute_ms:.4f} ms · "
                           f"D2H: {single_result.d2h_ms:.4f} ms")
            if single_result.stage_outputs:
                st.markdown("#### Pipeline stages")
                images.render_pipeline_stage_grid(single_result.stage_outputs, stage_flags)

        comparison = state.get("last_comparison")
        if comparison is not None:
            performance.render_live_vs_canonical_label(is_live=True)
            st.markdown("### Comparison result")
            for label, err in comparison.errors.items():
                st.warning(f"{label} failed: {err}")

            performance.render_live_performance_cards(comparison)

            st.markdown("#### Compute vs. total (H2D / Kernel / D2H)")
            performance.render_h2d_kernel_d2h_breakdown(comparison)

            st.markdown("#### Per-filter timing")
            performance.render_per_filter_live_table(comparison)

            final_outputs = {label: r.final_outputs[0] for label, r in comparison.results.items()}
            if final_outputs:
                st.markdown("#### Final output, three-way")
                images.render_side_by_side_final_outputs(final_outputs)

            show_intermediate = st.checkbox("Compare intermediate stages", value=state.get("show_intermediate_comparison", False))
            state.set_value("show_intermediate_comparison", show_intermediate)
            if show_intermediate:
                stage_outputs_by_impl = {label: r.stage_outputs for label, r in comparison.results.items() if r.stage_outputs}
                images.render_three_way_stage_comparison(stage_outputs_by_impl, stage_flags)

                show_diffs = st.checkbox("Show per-stage difference images", value=state.get("show_stage_differences", False))
                state.set_value("show_stage_differences", show_diffs)
                if show_diffs:
                    cpu_r = comparison.results.get("CPU")
                    for stage in ["gaussian", "median", "sobel", "laplacian", "threshold"]:
                        if not stage_flags[stage] or cpu_r is None or not cpu_r.stage_outputs:
                            continue
                        cpu_stage_out = cpu_r.stage_outputs.get(stage)
                        if cpu_stage_out is None:
                            continue
                        for label in ("Basic CUDA", "Enhanced CUDA"):
                            r = comparison.results.get(label)
                            if r and r.stage_outputs and r.stage_outputs.get(stage) is not None:
                                st.markdown(f"**{stage.capitalize()}: CPU vs {label}**")
                                images.render_absolute_difference("CPU", cpu_stage_out, label, r.stage_outputs[stage])

            st.markdown("#### Correctness")
            correctness.render_live_stage_correctness(comparison.stage_correctness)
            correctness.render_live_pipeline_correctness(comparison.pipeline_correctness)
            correctness.render_threshold_amplification_note()

            st.markdown("#### CUDA implementation")
            performance.render_gpu_optimization_summary(controls.PRODUCTION_ENHANCED_DEFAULTS)
            performance.render_implementation_explanation()

            st.markdown("---")
            if st.button("💾 Save Results"):
                try:
                    run_dir = services.save_comparison_results(comparison)
                    st.success(f"Saved to `{run_dir}` (a new, timestamped directory -- previous runs are never overwritten).")
                except Exception as exc:
                    _show_error(exc)

    else:
        # ============================================================
        # Section 15: real-batch live processing
        # ============================================================
        st.markdown("### Selection summary")
        sum_cols = st.columns(3)
        sum_cols[0].metric("Selected", f"{len(selection)} images")
        sum_cols[1].metric("Seed", selection.seed if selection.seed is not None else "n/a")
        sum_cols[2].metric("Dataset", f"{dataset_info.total_files:,} images")
        st.markdown("**Resolution groups**")
        st.write("  ·  ".join(f"{shape[1]}×{shape[0]} → {len(idxs)}" for shape, idxs in groups.items()))
        st.caption(f"Images loaded in {selection_load_ms:.3f} ms (measured separately from any processing time).")

        st.markdown("### Batch preview")
        images.render_batch_thumbnail_grid(images_all)

        col1, col2 = st.columns(2)
        with col1:
            process_batch_clicked = st.button(f"▶ Process Batch ({state.get('implementation')})", type="primary")
        with col2:
            compare_batch_clicked = st.button("⚖ Compare CPU vs Basic CUDA vs Enhanced CUDA", type="primary")

        if process_batch_clicked:
            impl_label = state.get("implementation")
            try:
                with st.spinner(f"Processing {len(images_main)} image(s) ({dominant_shape[1]}×{dominant_shape[0]}) with {impl_label}..."):
                    if impl_label == "CPU":
                        result = services.run_cpu_batch(images_main, filter_config)
                    else:
                        result = services.run_gpu_batch(images_main, filter_config, use_enhanced=(impl_label == "Enhanced CUDA"))
                state.set_value("last_result", {impl_label: result})
                state.set_value("last_images", images_main)
                state.set_value("last_batch_result", None)  # a single-implementation run supersedes any stale comparison view
            except services.ServiceError as exc:
                _show_error(exc)

        if compare_batch_clicked:
            try:
                with st.spinner(f"Processing {len(images_all)} image(s) across {len(groups)} resolution group(s) "
                                 f"for CPU, Basic CUDA, and Enhanced CUDA..."):
                    batch_result = services.compare_batch(
                        selection, images_all, filter_config, load_ms=selection_load_ms,
                        dataset_fingerprint=dm.fingerprint(),
                    )
                state.set_value("last_batch_result", batch_result)
                state.set_value("last_batch_selection", selection)
                state.set_value("last_batch_images", images_all)
                state.set_value("last_preview_index", 0)
                state.set_value("last_batch_preview_stages", None)
                state.set_value("last_result", None)
            except services.ServiceError as exc:
                _show_error(exc)

        last_result = state.get("last_result")
        if last_result:
            performance.render_live_vs_canonical_label(is_live=True)
            st.markdown("### Timing")
            performance.render_metric_cards({label: r.total_ms for label, r in last_result.items()})
            st.markdown("### Pipeline stages (dominant-group preview image)")
            for label, r in last_result.items():
                if r.stage_outputs:
                    images.render_pipeline_stage_grid(r.stage_outputs, stage_flags)
                else:
                    st.caption("No per-stage output available for this run.")

        batch_result = state.get("last_batch_result")
        if batch_result is not None:
            performance.render_live_vs_canonical_label(is_live=True)
            st.markdown("### Batch comparison result")
            st.caption(f"{batch_result.effective_count} of {batch_result.requested_batch_size} requested images "
                       f"processed (capped by safe GPU capacity where applicable)  ·  "
                       f"resolution groups: {', '.join(f'{k} → {v}' for k, v in batch_result.resolution_group_summary.items())}")
            for group in batch_result.groups:
                for label, err in group.errors.items():
                    st.warning(f"{label} failed for the {group.shape[1]}×{group.shape[0]} group: {err}")

            performance.render_batch_performance_cards(batch_result)
            performance.render_batch_message(batch_result)

            st.markdown("#### Compute vs. total (H2D / per-filter kernels / D2H)")
            performance.render_batch_h2d_kernel_d2h_breakdown(batch_result)

            st.markdown("#### Per-filter timing")
            performance.render_batch_per_filter_table(batch_result)

            st.markdown("#### Representative image")
            batch_images = state.get("last_batch_images") or images_all
            preview_index = images.render_preview_navigation(len(batch_images), state.get("last_preview_index", 0))
            if preview_index != state.get("last_preview_index", 0):
                state.set_value("last_preview_index", preview_index)
                state.set_value("last_batch_preview_stages", None)  # invalidate the cache -- recompute lazily below

            preview_stages = state.get("last_batch_preview_stages")
            if preview_stages is None:
                try:
                    preview_stages = services.get_preview_stage_outputs(batch_result, preview_index, batch_images, filter_config)
                    state.set_value("last_batch_preview_stages", preview_stages)
                except services.ServiceError as exc:
                    _show_error(exc)
                    preview_stages = {}

            preview_final_outputs = {label: stages["threshold"] if "threshold" in stages and stages["threshold"] is not None
                                      else next((v for v in reversed(list(stages.values())) if v is not None), None)
                                      for label, stages in preview_stages.items()} if preview_stages else {}
            if preview_final_outputs:
                st.markdown("##### Final output, three-way")
                images.render_side_by_side_final_outputs(preview_final_outputs)

            show_batch_intermediate = st.checkbox("Compare intermediate stages", value=state.get("show_batch_intermediate", False),
                                                   key="batch_intermediate_cb")
            state.set_value("show_batch_intermediate", show_batch_intermediate)
            if show_batch_intermediate and preview_stages:
                images.render_three_way_stage_comparison(preview_stages, stage_flags)

                show_batch_diffs = st.checkbox("Show per-stage difference images", value=state.get("show_batch_differences", False),
                                                key="batch_diffs_cb")
                state.set_value("show_batch_differences", show_batch_diffs)
                if show_batch_diffs:
                    cpu_stages = preview_stages.get("CPU")
                    for stage in ["gaussian", "median", "sobel", "laplacian", "threshold"]:
                        if not stage_flags[stage] or not cpu_stages or cpu_stages.get(stage) is None:
                            continue
                        for label in ("Basic CUDA", "Enhanced CUDA"):
                            gpu_stages = preview_stages.get(label)
                            if gpu_stages and gpu_stages.get(stage) is not None:
                                st.markdown(f"**{stage.capitalize()}: CPU vs {label}**")
                                images.render_absolute_difference("CPU", cpu_stages[stage], label, gpu_stages[stage])

            st.markdown("#### Correctness")
            correctness.render_batch_correctness(batch_result)
            correctness.render_threshold_amplification_note()

            st.markdown("---")
            if st.button("💾 Save Batch Results"):
                try:
                    preview_image = batch_images[preview_index] if preview_index < len(batch_images) else None
                    run_dir = services.save_batch_results(
                        batch_result, preview_stage_outputs=preview_stages, preview_image=preview_image,
                    )
                    st.success(f"Saved to `{run_dir}` (a new, timestamped directory -- previous runs are never overwritten).")
                except Exception as exc:
                    _show_error(exc)

            st.markdown("---")
            st.markdown("### 🔬 Quick batch-size sweep")
            st.caption("A small, live, session-local comparison across a few batch sizes on the dominant resolution "
                       "group -- NOT the full Section 11 benchmark matrix, only runs when clicked.")
            sweep_sizes = st.multiselect("Batch sizes", [1, 8, 16, 32, 64, 128, 256, 512],
                                          default=[1, 8, 32, 128])
            if st.button("Run Quick Batch-Size Sweep") and sweep_sizes:
                try:
                    with st.spinner("Running quick sweep..."):
                        sweep = services.run_quick_batch_sweep(images_main, filter_config, sorted(sweep_sizes))
                    state.set_value("quick_sweep_result", sweep)
                except services.ServiceError as exc:
                    _show_error(exc)

            sweep = state.get("quick_sweep_result")
            if sweep:
                performance.render_live_vs_canonical_label(is_live=True)
                st.caption(f"Resolution: {sweep['resolution'][1]}×{sweep['resolution'][0]}")
                performance.render_quick_batch_sweep_table(sweep)


# ================================================================
# TAB: CPU vs GPU
# ================================================================
with tab_map["CPU vs GPU"]:
    comparison = state.get("last_comparison")
    batch_result_tab = state.get("last_batch_result")
    last_result = state.get("last_result")

    if batch_result_tab is not None:
        # Batch comparison (Section 15) -- full detail (per-stage grid,
        # correctness, quick sweep, Save Batch Results) lives in Live
        # Processing; this tab gives the same three-way summary.
        performance.render_live_vs_canonical_label(is_live=True)
        st.caption(f"Batch: {batch_result_tab.effective_count} images  ·  "
                   f"resolution groups: {', '.join(f'{k} → {v}' for k, v in batch_result_tab.resolution_group_summary.items())}")
        performance.render_batch_performance_cards(batch_result_tab)
        st.caption("Full per-stage detail, per-image correctness distribution, and Save Batch Results are in the "
                   "Live Processing tab.")

    elif comparison is not None:
        # Single-image comparison (Section 14) -- full detail already lives in
        # Live Processing; this tab gives the same side-by-side + diff view
        # without needing to scroll back up.
        performance.render_live_vs_canonical_label(is_live=True)
        st.caption(f"Image: {comparison.image_meta.filename}  ·  {comparison.image_meta.width}×{comparison.image_meta.height}")
        final_outputs = {label: r.final_outputs[0] for label, r in comparison.results.items()}
        if final_outputs:
            st.markdown("### Final output, side-by-side")
            images.render_side_by_side_final_outputs(final_outputs)

        if "CPU" in final_outputs and "Enhanced CUDA" in final_outputs:
            st.markdown("### Absolute difference: CPU vs Enhanced CUDA")
            images.render_absolute_difference("CPU", final_outputs["CPU"], "Enhanced CUDA", final_outputs["Enhanced CUDA"])
        elif len(final_outputs) >= 2:
            labels = list(final_outputs.keys())
            st.markdown(f"### Absolute difference: {labels[0]} vs {labels[1]}")
            images.render_absolute_difference(labels[0], final_outputs[labels[0]], labels[1], final_outputs[labels[1]])

        st.markdown("### Timing (this run)")
        performance.render_metric_cards({label: r.total_ms for label, r in comparison.results.items()})
        st.caption("Full per-stage detail, correctness, and Save Results are in the Live Processing tab.")

    elif last_result and len(last_result) >= 2:
        preview_index = state.get("preview_index", 0)
        st.markdown("### Final output, side-by-side")
        final_outputs = {label: r.final_outputs[preview_index] for label, r in last_result.items()}
        images.render_side_by_side_final_outputs(final_outputs)

        if "CPU" in last_result and "Enhanced CUDA" in last_result:
            st.markdown("### Absolute difference: CPU vs Enhanced CUDA")
            images.render_absolute_difference("CPU", final_outputs["CPU"], "Enhanced CUDA", final_outputs["Enhanced CUDA"])
        elif len(final_outputs) >= 2:
            labels = list(final_outputs.keys())
            st.markdown(f"### Absolute difference: {labels[0]} vs {labels[1]}")
            images.render_absolute_difference(labels[0], final_outputs[labels[0]], labels[1], final_outputs[labels[1]])

        st.markdown("### Timing (this run)")
        performance.render_live_vs_canonical_label(is_live=True)
        performance.render_metric_cards({label: r.total_ms for label, r in last_result.items()})
    else:
        st.info("Run a comparison (Live Processing tab: 'Compare CPU vs Basic CUDA vs Enhanced CUDA', in either "
                "single-image or batch mode) to see side-by-side outputs and differences here.")

    st.markdown("---")
    st.markdown("### 🔬 Benchmark current configuration")
    st.caption("A small, session-local benchmark for the CURRENT filter parameters -- separate from the "
               "canonical Section 11 benchmark, and never written to benchmark_results/.")
    col1, col2 = st.columns(2)
    warmup = col1.number_input("Warmup runs", min_value=0, max_value=10, value=1)
    runs = col2.number_input("Measurement runs", min_value=1, max_value=50, value=3)
    if st.button("⏱ Benchmark Current Configuration"):
        try:
            with st.spinner("Benchmarking..."):
                live = services.run_live_benchmark(
                    images_main, filter_config,
                    run_cpu=state.get("compare_cpu", True), run_basic=state.get("compare_basic", False),
                    run_enhanced=state.get("compare_enhanced", False), warmup_runs=warmup, measurement_runs=runs,
                )
            state.set_value("live_benchmark_result", live)
        except services.ServiceError as exc:
            _show_error(exc)

    live = state.get("live_benchmark_result")
    if live:
        performance.render_live_vs_canonical_label(is_live=True)
        performance.render_metric_cards({
            "CPU": live.cpu_ms, "Basic CUDA": live.basic_ms, "Enhanced CUDA": live.enhanced_ms,
        }, title=f"{live.n_images} images, {live.warmup_runs} warmup + {live.measurement_runs} measurement runs (median)")


# ================================================================
# TAB: Performance Analytics
# ================================================================
with tab_map["Performance Analytics"]:
    benchmark_ids = services.list_known_benchmark_ids()
    from cuda.final_benchmark import load_manifest as _load_manifest

    if not presentation:
        selected_id = analytics.render_benchmark_history(benchmark_ids, _load_manifest)
    else:
        selected_id = None

    canonical = _load_benchmark_summary_cached(selected_id)
    if canonical is None:
        analytics.render_missing_artifact("no historical benchmark found. Run `python scripts/run_final_benchmark.py`.")
    else:
        bench_id = canonical["benchmark_id"]
        performance.render_live_vs_canonical_label(is_live=False)

        meta = services.benchmark_metadata(canonical)
        is_canonical = services.is_canonical_benchmark(bench_id)
        analytics.render_benchmark_metadata_panel(meta, is_canonical)

        st.markdown("### Executive summary")
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

        st.markdown("### Compute-only vs. end-to-end")
        performance.render_compute_vs_end_to_end(
            canonical["basic_cuda"]["mode1_kernel_only_ms"]["mean"],
            canonical["enhanced_cuda"]["mode1_kernel_only_ms"]["mean"],
            canonical["basic_cuda"]["mode4_end_to_end_ms"]["mean"],
            canonical["enhanced_cuda"]["mode4_end_to_end_ms"]["mean"],
        )

        st.markdown("### CPU vs Basic CUDA vs Enhanced CUDA")
        analytics.render_total_and_throughput_charts(canonical)

        st.markdown("### GPU compute breakdown")
        basic_breakdown = _gpu_compute_breakdown_cached(bench_id, "basic_cuda")
        enhanced_breakdown = _gpu_compute_breakdown_cached(bench_id, "enhanced_cuda")
        analytics.render_gpu_compute_breakdown_chart(basic_breakdown, enhanced_breakdown)

        per_filter = canonical.get("per_filter") or _load_per_filter_cached(selected_id)
        if per_filter:
            st.markdown("### Per-filter optimization")
            performance.render_filter_optimization_cards(per_filter["rows"])
            performance.render_per_filter_chart(per_filter["rows"])
            st.markdown("### Measured contribution to total compute reduction")
            performance.render_amdahl_contribution(per_filter["rows"])
            analytics.render_contribution_summary(per_filter["rows"])
        else:
            analytics.render_missing_artifact(f"no per-filter benchmark stored with benchmark `{bench_id}`.")

        batch_sweep = canonical.get("batch_sweep") or _load_batch_sweep_cached(selected_id)
        if batch_sweep:
            st.markdown("### Batch-size scaling")
            analytics.render_batch_sweep(batch_sweep)
            analytics.render_batch_sweep_speedup_chart(batch_sweep)
        else:
            analytics.render_missing_artifact(f"no batch-size sweep found for benchmark `{bench_id}`.")

        resolution_sweep = canonical.get("resolution_sweep") or _load_resolution_sweep_cached(selected_id)
        if resolution_sweep and not presentation:
            st.markdown("### Resolution scaling")
            analytics.render_resolution_sweep(resolution_sweep)
            analytics.render_resolution_throughput_chart(resolution_sweep)
            analytics.render_resolution_scaling_note()
        elif not presentation:
            analytics.render_missing_artifact(f"no resolution sweep found for benchmark `{bench_id}`.")

        st.markdown("### Correctness (this benchmark)")
        corr_bundle = canonical.get("correctness") or {}
        canonical_pipeline = corr_bundle.get("canonical_pipeline")
        corr_detail = corr_bundle.get("detailed") or _load_correctness_cached(selected_id)
        if canonical_pipeline:
            correctness.render_canonical_pipeline_correctness(canonical_pipeline)
        if corr_detail:
            correctness.render_correctness_summary(corr_detail)
            correctness.render_threshold_amplification_note()
        if not canonical_pipeline and not corr_detail:
            analytics.render_missing_artifact(f"no correctness benchmark stored with benchmark `{bench_id}`.")

        if not presentation:
            st.markdown("### Raw measurements & run statistics")
            analytics.render_run_statistics(canonical)
            cpu_raw = _load_raw_runs_cached(bench_id, "cpu")
            basic_raw = _load_raw_runs_cached(bench_id, "basic_cuda")
            enhanced_raw = _load_raw_runs_cached(bench_id, "enhanced_cuda")
            analytics.render_raw_measurements_expander(cpu_raw, basic_raw, enhanced_raw)

        performance.render_measurement_summary(canonical)

        st.markdown("---")
        st.markdown("### Export")
        analytics.render_export_buttons(canonical, services.export_benchmark_csv(canonical), services.export_benchmark_json(canonical))

        if not presentation and len(benchmark_ids) >= 2:
            st.markdown("---")
            st.markdown("### Compare Benchmarks")
            col_a, col_b = st.columns(2)
            default_a = benchmark_ids.index(bench_id) if bench_id in benchmark_ids else 0
            default_b = 1 if len(benchmark_ids) > 1 and default_a != 1 else 0
            with col_a:
                cmp_id_a = st.selectbox("Benchmark A", options=benchmark_ids, index=default_a, key="cmp_bench_a")
            with col_b:
                cmp_id_b = st.selectbox("Benchmark B", options=benchmark_ids, index=default_b, key="cmp_bench_b")
            if st.button("⚖ Compare Benchmarks"):
                try:
                    st.session_state["_benchmark_comparison"] = services.compare_benchmarks(cmp_id_a, cmp_id_b)
                except services.ServiceError as exc:
                    _show_error(exc)
            benchmark_comparison = st.session_state.get("_benchmark_comparison")
            if benchmark_comparison:
                analytics.render_benchmark_comparison(benchmark_comparison)


# ================================================================
# TAB: Optimization Lab (hidden in Presentation mode)
# ================================================================
if "Optimization Lab" in tab_map:
    with tab_map["Optimization Lab"]:
        st.markdown("### 🧪 CUDA Optimization Lab")
        st.caption("Different image-processing operations benefit from different CUDA optimization techniques. "
                   "This lab explains and demonstrates each filter's Basic → Enhanced progression using "
                   "already-compiled, already-tested kernels -- nothing here compiles CUDA code or changes the "
                   "production pipeline.")

        selected_filter = optimization_lab.render_filter_selector(
            services.optimization_lab_filters(), state.get("opt_lab_filter"))
        if selected_filter != state.get("opt_lab_filter"):
            state.set_value("opt_lab_filter", selected_filter)
            state.set_value("opt_lab_comparison", None)
            state.set_value("opt_lab_pipeline_impact", None)

        optimization_lab.render_production_configuration_panel(services.OPTIMIZATION_LAB_PRODUCTION_DEFAULTS)
        production_default = services.optimization_lab_production_default(selected_filter)

        # -- Historical Performance --
        st.markdown("---")
        st.markdown("#### Historical Performance")
        ksize_for_history = {
            "gaussian": filter_config.gaussian_kernel_size, "median": filter_config.median_kernel_size,
            "sobel": None, "laplacian": filter_config.laplacian_kernel_size, "threshold": None,
        }[selected_filter]
        sweep = services.load_variant_sweep(selected_filter)
        optimization_lab.render_historical_variant_table_and_chart(selected_filter, sweep, kernel_size=ksize_for_history)
        optimization_lab.render_optimization_progression(sweep, kernel_size=ksize_for_history)
        optimization_lab.render_rejected_variants(sweep)
        if selected_filter == "laplacian":
            fusion = services.load_fusion_experiment()
            optimization_lab.render_fusion_experiment(fusion)

        canonical_for_amdahl = services.load_benchmark_summary()
        per_filter_rows = (canonical_for_amdahl or {}).get("per_filter", {}).get("rows") if canonical_for_amdahl else None
        optimization_lab.render_amdahl_connection(selected_filter, per_filter_rows)

        # -- Why it works --
        st.markdown("---")
        st.markdown("#### Why it works")
        optimization_lab.render_why_it_works(selected_filter)
        optimization_lab.render_technique_cards(selected_filter)

        # -- Live Experiment --
        st.markdown("---")
        variants = services.optimization_lab_variants(selected_filter)
        variant_a, variant_b = optimization_lab.render_live_variant_controls(selected_filter, variants, production_default)
        optimization_lab.render_experimental_configuration_panel(selected_filter, variant_a, variant_b)

        batch_options = sorted({n for n in (1, 8, 32, 64, 128) if n <= len(images_main)} | {len(images_main)})
        live_batch_size = st.selectbox("Live test batch size", batch_options,
                                        index=len(batch_options) - 1 if len(batch_options) < 3 else min(2, len(batch_options) - 1),
                                        key=f"opt_lab_batch_size_{selected_filter}")
        live_images = images_main[:live_batch_size]

        if not services.cuda_available():
            st.warning(f"⚠ {services.cuda_unavailable_reason()} Live comparison is unavailable; historical "
                       f"results above are still shown.")

        run_col, impact_col = st.columns(2)
        with run_col:
            run_clicked = st.button("▶ Run Live Variant Comparison", type="primary",
                                     disabled=not services.cuda_available(), key=f"opt_lab_run_{selected_filter}")
        with impact_col:
            impact_clicked = st.button("📈 Measure Pipeline Impact (this filter's production variant)",
                                        disabled=not services.cuda_available(), key=f"opt_lab_impact_{selected_filter}")

        if run_clicked:
            try:
                with st.spinner(f"Running {variant_a} vs {variant_b} on {len(live_images)} image(s) "
                                 f"(2 warmup + 5 measured runs)..."):
                    comparison = services.compare_filter_variants(
                        live_images, filter_config, selected_filter, variant_a, variant_b,
                        warmup_runs=2, measurement_runs=5,
                    )
                state.set_value("opt_lab_comparison", comparison)
            except services.ServiceError as exc:
                _show_error(exc)

        if impact_clicked:
            try:
                with st.spinner("Measuring whole-pipeline impact (Basic vs. this filter enhanced)..."):
                    impact = services.measure_pipeline_impact(
                        live_images, filter_config, selected_filter, production_default,
                        warmup_runs=2, measurement_runs=5,
                    )
                state.set_value("opt_lab_pipeline_impact", impact)
            except services.ServiceError as exc:
                _show_error(exc)

        comparison = state.get("opt_lab_comparison")
        if comparison is not None and comparison.filter_name == selected_filter:
            optimization_lab.render_live_result_card(comparison)
            optimization_lab.render_live_correctness(comparison)
            optimization_lab.render_technical_detail_panel(selected_filter, comparison.variant_b)

            if st.button("💾 Save Experiment", key=f"opt_lab_save_{selected_filter}"):
                try:
                    run_dir = services.save_optimization_experiment(
                        comparison, dataset_fingerprint=dm.fingerprint(),
                        selected_relative_paths=[item.relative_path for item in selection.items[:live_batch_size]],
                    )
                    st.success(f"Saved to `{run_dir}`")
                except Exception as exc:  # noqa: BLE001
                    _show_error(exc)

        impact = state.get("opt_lab_pipeline_impact")
        if impact is not None and impact.filter_name == selected_filter:
            optimization_lab.render_pipeline_impact_panel(impact)

        registry = services.load_experiment_registry()
        if registry and not presentation:
            st.markdown("---")
            st.markdown("### Known optimization experiments")
            import pandas as pd

            reg_rows = [{"Experiment": e["experiment"], "Script": e["script"],
                         "Last benchmark_id": e.get("benchmark_id", "-")} for e in registry["entries"]]
            st.dataframe(pd.DataFrame(reg_rows), width="stretch", hide_index=True)


# ================================================================
# TAB: Threading & Parallelism (Section 23) -- pure observability, reads real
# CPU/GPU/launch-configuration data via pipeline.threading_metrics; never
# invokes cpu.pipeline/cuda.pipeline just to render this tab.
# ================================================================
with tab_map["Threading & Parallelism"]:
    threading_view.render_threading_tab(
        width=dominant_shape[1], height=dominant_shape[0], batch_size=len(images_main))


# ================================================================
# TAB: Correctness
# ================================================================
with tab_map["Correctness"]:
    corr = services.load_correctness_results()
    if corr is None:
        st.info("No stored correctness benchmark found. Run `python scripts/run_final_benchmark.py`.")
    else:
        correctness.render_correctness_summary(corr)
        if not presentation:
            manifest = services.load_benchmark_summary()
            correctness.render_correctness_details(
                corr, manifest["manifest"]["filter_config"] if manifest else {},
                corr["resolution"], corr["image_count"],
            )


# ================================================================
# TAB: System (hidden in Presentation mode)
# ================================================================
if "System" in tab_map:
    with tab_map["System"]:
        from ui import system

        system.render_system_info()
        st.markdown("---")
        estimated_bytes = len(images_main) * dominant_shape[0] * dominant_shape[1] * 2  # ping-pong buffers, uint8
        system.render_gpu_memory_info(estimated_batch_bytes=estimated_bytes)

        st.markdown("---")
        with st.expander("How CUDA improves the pipeline"):
            st.markdown("""
- **CPU**: one processing engine, sequential per pixel/row.
- **Basic CUDA**: many GPU threads working in parallel, one thread per output pixel.
- **Enhanced CUDA**: parallel execution **+** optimized memory access (shared/constant memory) **+**
  algorithm-specific optimization (e.g. a sorting network for Median) **+** compile-time specialized kernels **+**
  batch processing (one native call for many images, amortizing fixed overhead).

Measured speedups vary by filter -- see Performance Analytics for exactly why (per-filter optimization notes).
""")


# ================================================================
# TAB: Presentation Mode (Section 20) -- always available, a self-contained ~30s story
# ================================================================
with tab_map["Presentation Mode"]:
    presentation_ui.render_header()
    st.markdown("---")

    presentation_ui.render_architecture_diagram()
    st.markdown("---")
    presentation_ui.render_pipeline_diagram()
    st.markdown("---")

    presentation_ui.render_live_image(state.get("last_comparison"))
    st.markdown("---")

    presentation_canonical = _load_benchmark_summary_cached(None)
    if presentation_canonical is None:
        st.info("No canonical benchmark found. Run `python scripts/run_final_benchmark.py` to populate "
                 "Presentation Mode's performance sections.")
    else:
        presentation_ui.render_performance_cards(presentation_canonical)
        st.markdown("---")
        presentation_ui.render_compute_vs_end_to_end(presentation_canonical)
        st.markdown("---")

        presentation_per_filter = presentation_canonical.get("per_filter", {}).get("rows") if presentation_canonical.get("per_filter") else None
        presentation_ui.render_per_filter_cards(presentation_per_filter)
        st.markdown("---")
        presentation_ui.render_threading_summary(
            width=dominant_shape[1], height=dominant_shape[0], batch_size=len(images_main))
        st.markdown("---")
        presentation_ui.render_key_insight(presentation_per_filter)
        st.markdown("---")

        presentation_ui.render_correctness(presentation_canonical)
        st.markdown("---")
        presentation_ui.render_batch_insight(presentation_canonical.get("batch_sweep"))
        st.markdown("---")

    presentation_ui.render_gpu_implementation(services.gpu_implementation_facts())
