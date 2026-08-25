"""Sidebar controls (Section 12 spec items 6-16, 30-32): dataset,
selection, filter tuning, implementation choice, compare mode. Every
widget here reads/writes ui.state's session_state -- no local module
state.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from cpu.filters import (
    ALLOWED_GAUSSIAN_KERNELS,
    ALLOWED_LAPLACIAN_KERNELS,
    ALLOWED_MEDIAN_KERNELS,
    SOBEL_MODES,
)
from ui import services, state

PRODUCTION_ENHANCED_DEFAULTS = {
    "gaussian": "specialized", "median": "network3x3", "sobel": "specialized",
    "laplacian": "specialized", "threshold": "vectorized",
}


def render_dataset_controls() -> Optional[str]:
    st.sidebar.header("Dataset")
    default_path = state.get("dataset_path") or "../data"
    dataset_path = st.sidebar.text_input("Dataset directory", value=default_path)
    state.set_value("dataset_path", dataset_path)

    if not dataset_path:
        st.sidebar.info("Enter a dataset directory to begin.")
        return None

    try:
        dm, info = services.scan_dataset(dataset_path)
    except services.ServiceError as exc:
        st.sidebar.error(str(exc))
        return None

    st.sidebar.caption(f"{info.total_files:,} images found ({info.resolution_note})")
    return dataset_path


def render_selection_controls(dm) -> object:
    st.sidebar.header("Image Selection")
    mode = st.sidebar.radio("Selection mode", ["Random batch", "Single image"],
                             index=0 if state.get("selection_mode") == "Random batch" else 1)
    state.set_value("selection_mode", mode)

    if mode == "Single image":
        max_index = max(0, len(dm.paths) - 1)
        index = st.sidebar.number_input("Image index", min_value=0, max_value=max_index,
                                         value=min(state.get("single_image_index", 0), max_index))
        state.set_value("single_image_index", index)
        try:
            return services.select_single(dm, index)
        except services.ServiceError as exc:
            st.sidebar.error(str(exc))
            return None

    st.sidebar.caption("Batch size presets")
    preset_cols = st.sidebar.columns(4)
    presets = [1, 8, 16, 32, 64, 128, 256, 512]
    for col, preset in zip(preset_cols * 2, presets):
        if col.button(str(preset), key=f"batch_preset_{preset}"):
            state.set_value("batch_size", preset)

    col1, col2 = st.sidebar.columns(2)
    batch_size = col1.number_input("Batch size", min_value=1, max_value=2048, value=state.get("batch_size", 125), step=1)
    seed = col2.number_input("Seed", min_value=0, max_value=2**31 - 1, value=state.get("seed", 42), step=1)
    if st.sidebar.button("🎲 Randomize Batch (new seed)"):
        import random

        seed = random.randint(0, 2**31 - 1)
    state.set_value("batch_size", batch_size)
    state.set_value("seed", seed)

    try:
        return services.select_random_batch(dm, batch_size=batch_size, seed=seed)
    except services.ServiceError as exc:
        st.sidebar.error(str(exc))
        return None


def render_filter_controls() -> None:
    st.sidebar.header("Filter Parameters")
    kwargs = dict(state.get("filter_config_kwargs", {}))

    with st.sidebar.expander("Gaussian", expanded=False):
        kwargs["gaussian_enabled"] = st.checkbox("Enable Gaussian", value=kwargs.get("gaussian_enabled", True))
        kwargs["gaussian_kernel_size"] = st.select_slider(
            "Kernel size", options=sorted(ALLOWED_GAUSSIAN_KERNELS), value=kwargs.get("gaussian_kernel_size", 5))
        kwargs["gaussian_sigma"] = st.slider("Sigma (0 = auto from kernel size)", 0.0, 5.0,
                                              value=float(kwargs.get("gaussian_sigma", 0.0)), step=0.1)

    with st.sidebar.expander("Median", expanded=False):
        kwargs["median_enabled"] = st.checkbox("Enable Median", value=kwargs.get("median_enabled", True))
        kwargs["median_kernel_size"] = st.select_slider(
            "Kernel size", options=sorted(ALLOWED_MEDIAN_KERNELS), value=kwargs.get("median_kernel_size", 3))

    with st.sidebar.expander("Sobel", expanded=False):
        kwargs["sobel_enabled"] = st.checkbox("Enable Sobel", value=kwargs.get("sobel_enabled", True))
        kwargs["sobel_mode"] = st.selectbox("Mode", sorted(SOBEL_MODES),
                                             index=sorted(SOBEL_MODES).index(kwargs.get("sobel_mode", "magnitude")))

    with st.sidebar.expander("Laplacian", expanded=False):
        kwargs["laplacian_enabled"] = st.checkbox("Enable Laplacian", value=kwargs.get("laplacian_enabled", True))
        kwargs["laplacian_kernel_size"] = st.select_slider(
            "Kernel size", options=sorted(ALLOWED_LAPLACIAN_KERNELS), value=kwargs.get("laplacian_kernel_size", 3))
        kwargs["laplacian_scale"] = st.slider("Scale", 0.1, 3.0, value=float(kwargs.get("laplacian_scale", 1.0)), step=0.1)
        kwargs["laplacian_delta"] = st.slider("Delta", 0.0, 50.0, value=float(kwargs.get("laplacian_delta", 0.0)), step=1.0)

    with st.sidebar.expander("Threshold", expanded=False):
        kwargs["threshold_enabled"] = st.checkbox("Enable Threshold", value=kwargs.get("threshold_enabled", True))
        kwargs["threshold_value"] = st.slider("Threshold value", 0, 255, value=int(kwargs.get("threshold_value", 128)))
        kwargs["threshold_max_value"] = st.slider("Max output value", 0, 255, value=int(kwargs.get("threshold_max_value", 255)))

    state.set_value("filter_config_kwargs", kwargs)


def render_implementation_controls() -> None:
    st.sidebar.header("Implementation")
    implementation = st.sidebar.radio(
        "Processing implementation", ["CPU", "Basic CUDA", "Enhanced CUDA"],
        index=["CPU", "Basic CUDA", "Enhanced CUDA"].index(state.get("implementation", "Enhanced CUDA")),
    )
    state.set_value("implementation", implementation)
    if implementation != "CPU" and not services.cuda_available():
        st.sidebar.warning(f"{implementation} selected but unavailable: {services.cuda_unavailable_reason()}")


def render_compare_controls() -> None:
    st.sidebar.header("Compare Mode")
    cuda_ok = services.cuda_available()
    compare_cpu = st.sidebar.checkbox("Run CPU", value=state.get("compare_cpu", True))
    compare_basic = st.sidebar.checkbox("Run Basic CUDA", value=state.get("compare_basic", True) and cuda_ok, disabled=not cuda_ok)
    compare_enhanced = st.sidebar.checkbox("Run Enhanced CUDA", value=state.get("compare_enhanced", True) and cuda_ok, disabled=not cuda_ok)
    state.set_value("compare_cpu", compare_cpu)
    state.set_value("compare_basic", compare_basic if cuda_ok else False)
    state.set_value("compare_enhanced", compare_enhanced if cuda_ok else False)
    if not cuda_ok:
        st.sidebar.caption("GPU comparisons unavailable -- CPU-only mode.")


def render_presentation_and_debug_toggles() -> None:
    st.sidebar.header("Display")
    presentation = st.sidebar.checkbox("🎤 Presentation mode", value=state.get("presentation_mode", False))
    debug = st.sidebar.checkbox("🐛 Debug mode (show raw errors)", value=state.get("debug_mode", False))
    state.set_value("presentation_mode", presentation)
    state.set_value("debug_mode", debug)
