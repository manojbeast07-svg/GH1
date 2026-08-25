"""System / GPU information panel (Section 12 spec items 36-37)."""

from __future__ import annotations

from typing import Optional

import streamlit as st

from ui import services, theme


def render_system_info() -> None:
    theme.inject_css()
    st.subheader("System Information")
    label = services.environment_label()
    st.markdown(f"**{'🖥️' if label == 'LOCAL ENVIRONMENT' else '☁️'} {label}**")

    env = services.environment_fingerprint()
    tools = services.build_tool_versions()
    vram_gb = (env["gpu_vram_bytes"] / (1024 ** 3)) if env.get("gpu_vram_bytes") else None

    theme.render_section_label("CPU")
    c1, c2, c3 = st.columns(3)
    with c1:
        theme.render_metric_card("Model", (env["cpu_model"] or "N/A")[:30], status="cpu")
    with c2:
        theme.render_metric_card("Logical Cores", str(env["cpu_logical_cores"] or "N/A"), status="cpu")
    with c3:
        theme.render_metric_card("OS", env["os_platform"].split("-")[0])

    theme.render_section_label("GPU")
    if env["gpu_available"]:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            theme.render_metric_card("Model", (env["gpu_name"] or "N/A")[:26], status="enhanced")
        with c2:
            theme.render_metric_card("VRAM", f"{vram_gb:.2f} GB" if vram_gb is not None else "N/A", status="enhanced")
        with c3:
            theme.render_metric_card("Compute Capability", env["gpu_compute_capability"] or "N/A", status="enhanced")
        with c4:
            theme.render_metric_card("Driver", env["gpu_driver_version"] or "N/A", status="enhanced")
    else:
        st.warning("No usable CUDA device detected.")

    theme.render_section_label("CUDA")
    c1, c2 = st.columns(2)
    with c1:
        theme.render_metric_card("CUDA Runtime/Toolkit", env["cuda_runtime_version"] or "N/A", status="enhanced")
    with c2:
        theme.render_metric_card("NVCC", tools["nvcc"], status="enhanced")

    theme.render_section_label("Software / Build")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        theme.render_metric_card("Python", env["python_version"])
    with c2:
        theme.render_metric_card("OpenCV", env["opencv_version"] or "N/A")
    with c3:
        theme.render_metric_card("NumPy", env["numpy_version"] or "N/A")
    with c4:
        theme.render_metric_card("CMake", tools["cmake"])
    if env.get("git_commit"):
        st.caption(f"git commit: {env['git_commit'][:12]}")


def render_gpu_memory_info(estimated_batch_bytes: Optional[int] = None) -> None:
    st.subheader("GPU Memory")
    mem = services.gpu_memory_info()
    if mem is None:
        st.warning("GPU memory info unavailable (no usable CUDA device).")
        return

    free_gb = mem["free_bytes"] / (1024 ** 3)
    total_gb = mem["total_bytes"] / (1024 ** 3)
    used_gb = total_gb - free_gb

    col1, col2, col3 = st.columns(3)
    col1.metric("Total VRAM", f"{total_gb:.2f} GB")
    col2.metric("Free VRAM", f"{free_gb:.2f} GB")
    col3.metric("Used (other processes)", f"{used_gb:.2f} GB")
    st.progress(min(1.0, used_gb / total_gb) if total_gb > 0 else 0.0)

    if estimated_batch_bytes is not None:
        est_gb = estimated_batch_bytes / (1024 ** 3)
        st.caption(f"Estimated memory for current batch: {est_gb:.3f} GB "
                   f"({'within' if estimated_batch_bytes < mem['free_bytes'] else '⚠ EXCEEDS'} free VRAM)")

    st.caption("This project deliberately caps GPU batch sizes conservatively (Section 5's "
               "`compute_safe_gpu_batch_size()`, 70% of reported free VRAM) rather than encouraging "
               "allocation of all available memory.")
