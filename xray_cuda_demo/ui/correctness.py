"""Correctness dashboard components (Section 12 spec items 27-29, 56).
Renders already-computed correctness results (from a loaded
correctness/{id}.json or from ui.services) -- never recomputes a
diff itself.
"""

from __future__ import annotations

import streamlit as st

ESTABLISHED_BASELINE_PCT = 0.0178


def render_correctness_summary(correctness: dict) -> None:
    """`correctness` is a Section 11 correctness-benchmark payload
    (filter_level / pipeline_level / overall_pass / baseline_comparison_note)."""
    st.subheader("Correctness")

    st.markdown("**Filter-level**")
    cols = st.columns(5)
    for col, (name, result) in zip(cols, correctness["filter_level"].items()):
        with col:
            status = "✅ PASS" if result["pass"] else "❌ FAIL"
            st.metric(name.capitalize(), status)
            st.caption(f"tolerance: ±{result['tolerance']}")

    st.markdown("**Final pipeline**")
    p = correctness["pipeline_level"]["enhanced_vs_cpu"]
    pct = p["differing_pixel_percentage"]
    within_baseline = abs(pct - ESTABLISHED_BASELINE_PCT) < 0.01
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Differing pixels", f"{pct:.4f}%", delta=f"baseline {ESTABLISHED_BASELINE_PCT}%" if within_baseline else "⚠ differs from baseline")
    col2.metric("max_abs_diff", p["max_abs_diff"])
    col3.metric("mean_abs_diff", f"{p['mean_abs_diff']:.4f}")
    col4.metric("RMSE", f"{p['rmse']:.4f}")

    if p["max_abs_diff"] == 255:
        st.caption("Threshold can amplify a tiny upstream floating-point difference (e.g. Gaussian's documented "
                   "±1 tolerance) into a full 0↔255 pixel flip. `max_abs_diff = 255` does not by itself imply a "
                   "bad pipeline -- the differing-pixel percentage above is the meaningful measure.")

    st.markdown(f"**Overall:** {'✅ PASS' if correctness['overall_pass'] else '❌ FAIL'} — {correctness['baseline_comparison_note']}")


def render_live_stage_correctness(stage_correctness: list) -> None:
    """Spec item 20: per-stage PASS/WARNING indicators for a live
    ComparisonResult, computed from THIS run's actual outputs (never
    hardcoded)."""
    present = [s for s in stage_correctness if s.status != "N/A"]
    if not present:
        st.caption("No stage-level correctness available (need CPU plus at least one GPU implementation).")
        return
    cols = st.columns(len(present))
    for col, s in zip(cols, present):
        with col:
            icon = "✅" if s.status == "PASS" else "⚠️"
            st.metric(s.stage.capitalize(), f"{icon} {s.status}")
            st.caption(f"max_abs_diff={s.max_abs_diff_vs_cpu}  (tolerance ±{s.tolerance})")


def render_live_pipeline_correctness(pipeline_correctness: list) -> None:
    """Spec item 21: full-pipeline correctness computed from the CURRENT
    run, never a hardcoded established value."""
    if not pipeline_correctness:
        st.caption("No pipeline-level correctness available.")
        return
    for p in pipeline_correctness:
        st.markdown(f"**{p.comparison.replace('_', ' ')}**")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("max_abs_diff", p.max_abs_diff)
        col2.metric("mean_abs_diff", f"{p.mean_abs_diff:.4f}")
        col3.metric("RMSE", f"{p.rmse:.4f}")
        col4.metric("Differing pixels", f"{p.differing_pixel_count:,}")
        col5.metric("Differing %", f"{p.differing_pixel_percentage:.4f}%")


def render_threshold_amplification_note() -> None:
    """Spec item 22."""
    with st.expander("Why can max_abs_diff be 255 even when almost nothing differs?"):
        st.markdown(
            "Small upstream floating-point differences, particularly from Gaussian filtering, can cause a pixel "
            "near the threshold boundary to switch from 0 to 255. Therefore the final thresholded output can have "
            "a large `max_abs_diff` even when the number of affected pixels is very small. The differing-pixel "
            "percentage is the more useful measure of overall agreement."
        )


def render_batch_correctness(batch_result) -> None:
    """Spec items 22-24, 26: aggregate correctness across EVERY
    processed image (never just the first), plus the per-image
    distribution (images with zero vs. nonzero differences, max/mean
    per-image differing %). Basic-vs-Enhanced is expected bit-exact
    (Section 10's established finding) -- flagged as a WARNING, not
    silently tolerated, if it ever isn't."""
    by_name = {c.comparison: c for c in batch_result.correctness}

    enhanced_vs_basic = by_name.get("enhanced_vs_basic")
    if enhanced_vs_basic is not None:
        exact = enhanced_vs_basic.max_abs_diff == 0
        st.markdown(f"**Basic vs Enhanced exactness:** {'✅ bit-exact' if exact else '⚠️ WARNING -- NOT bit-exact'}")
        if not exact:
            st.warning(f"max_abs_diff={enhanced_vs_basic.max_abs_diff}, "
                       f"{enhanced_vs_basic.differing_pixel_count:,} differing pixels "
                       f"({enhanced_vs_basic.differing_pixel_percentage:.4f}%) across "
                       f"{enhanced_vs_basic.images_compared} images -- unexpected, not silently tolerated.")

    enhanced_vs_cpu = by_name.get("enhanced_vs_cpu")
    if enhanced_vs_cpu is not None:
        st.markdown("**CPU vs Enhanced**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Differing pixels", f"{enhanced_vs_cpu.differing_pixel_count:,}")
            st.metric("Differing %", f"{enhanced_vs_cpu.differing_pixel_percentage:.4f}%")
        with col2:
            st.metric("max_abs_diff", enhanced_vs_cpu.max_abs_diff)
            st.metric("RMSE", f"{enhanced_vs_cpu.rmse:.4f}")

    st.markdown("**Per-image distribution**")
    for c in batch_result.correctness:
        st.markdown(f"_{c.comparison.replace('_', ' ')}_")
        cols = st.columns(4)
        cols[0].metric("Images w/ 0 diff", c.images_with_zero_diff)
        cols[1].metric("Images w/ diff", c.images_with_nonzero_diff)
        cols[2].metric("Max image diff %", f"{c.max_image_diff_percentage:.4f}%")
        cols[3].metric("Mean image diff %", f"{c.mean_image_diff_percentage:.4f}%")


def render_canonical_pipeline_correctness(canonical_pipeline: dict) -> None:
    """Spec items 26-27: the selected historical benchmark's
    basic_vs_cpu / enhanced_vs_cpu / enhanced_vs_basic pipeline-level
    correctness, loaded from that benchmark's own stored data (never a
    hardcoded baseline)."""
    st.markdown("**Final pipeline correctness (this benchmark)**")
    labels = {"basic_vs_cpu": "Basic vs CPU", "enhanced_vs_cpu": "Enhanced vs CPU", "enhanced_vs_basic": "Enhanced vs Basic"}
    for key, label in labels.items():
        metrics = canonical_pipeline.get(key)
        if metrics is None:
            continue
        st.markdown(f"_{label}_")
        cols = st.columns(4)
        cols[0].metric("Differing pixels", f"{metrics['differing_pixel_percentage']:.4f}%")
        cols[1].metric("max_abs_diff", metrics["max_abs_diff"])
        cols[2].metric("mean_abs_diff", f"{metrics['mean_abs_diff']:.4f}")
        cols[3].metric("RMSE", f"{metrics['rmse']:.4f}")


def render_correctness_details(correctness: dict, filter_config: dict, resolution, image_count: int) -> None:
    with st.expander("Correctness Details"):
        st.markdown("**Known expected differences**")
        for name, note in correctness["known_expected_differences"].items():
            st.markdown(f"- **{name.capitalize()}**: {note}")

        st.markdown("**Pipeline-level comparisons**")
        for comparison, metrics in correctness["pipeline_level"].items():
            st.markdown(f"- `{comparison}`: max_abs_diff={metrics['max_abs_diff']}, "
                        f"mean_abs_diff={metrics['mean_abs_diff']:.4f}, rmse={metrics['rmse']:.4f}, "
                        f"differing={metrics['differing_pixel_percentage']:.4f}%")

        st.markdown("**Selected images / configuration**")
        st.write(f"Resolution: {resolution[1]}×{resolution[0]}  ·  Images: {image_count}")
        st.json(filter_config)
