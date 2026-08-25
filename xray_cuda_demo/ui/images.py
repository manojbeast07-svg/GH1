"""Image display components (Section 12 spec items 17-18, 47-49).

All rendering only -- no processing happens here (that's ui/services.py,
called from app.py). Batch mode never renders more than one
representative image's full pipeline (spec item 18: no attempt to
display hundreds of full-resolution images).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import streamlit as st

STAGE_ORDER = ["original", "gaussian", "median", "sobel", "laplacian", "threshold"]
STAGE_LABELS = {
    "original": "Original", "gaussian": "Gaussian", "median": "Median",
    "sobel": "Sobel", "laplacian": "Laplacian", "threshold": "Threshold",
}


def render_pipeline_stage_grid(stage_outputs: Dict[str, Optional[np.ndarray]], enabled_stages: Dict[str, bool]) -> None:
    """2x3 grid: Original/Gaussian/Median top row, Sobel/Laplacian/
    Threshold bottom row (spec item 17). A disabled stage is greyed out
    (spec item 47) rather than hidden, so the fixed filter order stays
    visible."""
    present = [s for s in STAGE_ORDER if s in stage_outputs]
    rows = [present[:3], present[3:]]
    for row in rows:
        if not row:
            continue
        cols = st.columns(len(row))
        for col, stage in zip(cols, row):
            with col:
                image = stage_outputs.get(stage)
                is_enabled = stage == "original" or enabled_stages.get(stage, True)
                label = STAGE_LABELS[stage]
                if image is None:
                    st.markdown(f"**{label}** _(disabled)_")
                    st.empty()
                elif not is_enabled:
                    st.markdown(f"~~{label}~~ _(disabled)_")
                    st.image(image, clamp=True, channels="GRAY", width="stretch")
                else:
                    st.markdown(f"**{label}**")
                    st.image(image, clamp=True, channels="GRAY", width="stretch")


def render_batch_thumbnail_grid(images: List[np.ndarray], max_thumbnails: int = 24) -> None:
    """Small thumbnail grid, never a full-resolution render of every
    image in a large batch (spec item 18)."""
    shown = images[:max_thumbnails]
    cols_per_row = 6
    for start in range(0, len(shown), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, img in zip(cols, shown[start:start + cols_per_row]):
            with col:
                st.image(img, clamp=True, channels="GRAY", width="stretch")
    if len(images) > max_thumbnails:
        st.caption(f"Showing {max_thumbnails} of {len(images)} images. Use the preview selector below for full detail on one image.")


def render_preview_selector(n_images: int, current: int) -> int:
    return st.selectbox(f"Preview image (1 of {n_images})", options=list(range(n_images)),
                         index=min(current, n_images - 1), format_func=lambda i: f"Image #{i + 1}")


def render_preview_navigation(n_images: int, current: int, key_prefix: str = "preview_nav") -> int:
    """Spec item 28: Previous/Next buttons for stepping through a
    batch's representative image without recomputing anything --
    callers cache per-image results keyed by the returned index and
    only recompute when it actually changes."""
    current = min(max(current, 0), max(n_images - 1, 0))
    col_prev, col_label, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("◀ Previous", key=f"{key_prefix}_prev", disabled=current <= 0):
            current -= 1
    with col_next:
        if st.button("Next ▶", key=f"{key_prefix}_next", disabled=current >= n_images - 1):
            current += 1
    with col_label:
        st.caption(f"Image {current + 1} of {n_images}")
    return current


def render_side_by_side_final_outputs(labeled_outputs: Dict[str, np.ndarray]) -> None:
    """CPU / Basic CUDA / Enhanced CUDA final output side-by-side (spec
    items 48-49)."""
    cols = st.columns(len(labeled_outputs))
    for col, (label, img) in zip(cols, labeled_outputs.items()):
        with col:
            st.markdown(f"**{label}**")
            st.image(img, clamp=True, channels="GRAY", width="stretch")


def render_three_way_stage_comparison(
    stage_outputs_by_impl: Dict[str, Dict[str, Optional[np.ndarray]]], enabled_stages: Dict[str, bool],
) -> None:
    """Spec item 18: for each enabled stage, one row of CPU / Basic CUDA
    / Enhanced CUDA columns -- `stage_outputs_by_impl` is
    {implementation_label: {stage_name: image_or_None}}. Skips a stage
    entirely when its filter is disabled (never shows a stale/greyed
    row for it here -- greying is Live Processing's per-implementation
    grid, this is the dedicated compare view)."""
    implementations = list(stage_outputs_by_impl.keys())
    for stage in STAGE_ORDER:
        if stage == "original" or not enabled_stages.get(stage, False):
            continue
        present = [impl for impl in implementations if stage_outputs_by_impl[impl].get(stage) is not None]
        if not present:
            continue
        st.markdown(f"**{STAGE_LABELS[stage]}**")
        cols = st.columns(len(present))
        for col, impl in zip(cols, present):
            with col:
                st.caption(impl)
                st.image(stage_outputs_by_impl[impl][stage], clamp=True, channels="GRAY", width="stretch")


def render_absolute_difference(label_a: str, img_a: np.ndarray, label_b: str, img_b: np.ndarray) -> None:
    """Difference visualization -- amplified for visibility, with the
    real (not exaggerated) statistics printed alongside (spec items 28,
    48-49: never overstate a handful of thresholded pixels)."""
    if img_a.shape != img_b.shape:
        st.warning(f"Cannot diff {label_a} ({img_a.shape}) against {label_b} ({img_b.shape}) -- different shapes.")
        return
    diff = np.abs(img_a.astype(np.int16) - img_b.astype(np.int16)).astype(np.uint8)
    differing = int(np.count_nonzero(diff))
    pct = 100.0 * differing / diff.size

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**|{label_a} − {label_b}|** (amplified ×4 for visibility)")
        amplified = np.clip(diff.astype(np.int32) * 4, 0, 255).astype(np.uint8)
        st.image(amplified, clamp=True, channels="GRAY", width="stretch")
    with col2:
        st.metric("Differing pixels", f"{pct:.4f}%")
        st.caption(f"{differing:,} / {diff.size:,} pixels")
        st.caption(f"max_abs_diff = {int(diff.max())}")
        if int(diff.max()) == 255:
            st.caption("A max_abs_diff of 255 alone does not imply failure -- Threshold can amplify a "
                       "tiny upstream difference into a full 0↔255 flip on a handful of pixels. The "
                       "differing-pixel percentage above is the meaningful measure.")
