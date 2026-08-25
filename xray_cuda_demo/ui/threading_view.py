"""Section 23: "Threading & Parallelism" tab -- renders
pipeline.threading_metrics.ThreadingMetrics. Pure rendering, like every
other ui/ module; never calls into cpu/cuda pipeline code itself, never
runs a filter or launches a kernel just to render this tab (spec item 53
test #12). All data comes from the centralized
pipeline.threading_metrics.get_threading_metrics() service (spec item 47).
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from pipeline.threading_metrics import ThreadingMetrics, get_threading_metrics
from ui import theme


def _fmt(value, suffix: str = "", none_text: str = "N/A") -> str:
    if value is None:
        return none_text
    return f"{value}{suffix}"


def _fmt_bytes(n: Optional[int]) -> str:
    if n is None:
        return "N/A"
    gb = n / (1024 ** 3)
    return f"{gb:.2f} GB"


def _fmt_dims(dims) -> str:
    if dims is None:
        return "N/A"
    x, y, z = dims
    return f"{x} × {y} × {z}" if z != 1 else f"{x} × {y}"


def render_cpu_section(m: ThreadingMetrics) -> None:
    theme.render_section_label("CPU")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        theme.render_metric_card("CPU Model", (m.cpu_model or "N/A")[:28], status="cpu")
    with c2:
        theme.render_metric_card("Physical Cores", _fmt(m.cpu_physical_cores), status="cpu")
    with c3:
        theme.render_metric_card("Logical Processors", _fmt(m.cpu_logical_processors), status="cpu")
    with c4:
        theme.render_metric_card("OpenCV Threads", _fmt(m.opencv_threads), subtitle="configured", status="cpu")
    st.caption(m.opencv_threads_source)

    with st.expander("CPU execution model"):
        st.markdown(
            "```text\n"
            "Image\n"
            "  ↓\n"
            "OpenCV (cv2.*)\n"
            "  ↓\n"
            "CPU worker threads (OpenCV's own internal thread pool)\n"
            "  ↓\n"
            "CPU cores\n"
            "```"
        )
        st.caption(
            "OpenCV may internally parallelize individual operations (e.g. across rows) using its own "
            "threading backend, depending on the build and the specific operation -- this project does not "
            "assume a fixed per-operation thread count beyond what `cv2.getNumThreads()` reports as the "
            "configured maximum above. Whether every operation actually uses all configured threads is "
            "operation-dependent and was not independently measured here."
        )


def render_gpu_hardware_section(m: ThreadingMetrics) -> None:
    theme.render_section_label("GPU Hardware")
    if not m.gpu_available:
        st.warning("No CUDA device detected.")
        return
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        theme.render_metric_card("GPU", (m.gpu_model or "N/A")[:24], status="enhanced")
    with c2:
        theme.render_metric_card("Compute Capability", _fmt(m.gpu_compute_capability), status="enhanced")
    with c3:
        theme.render_metric_card("SM Count", _fmt(m.gpu_sm_count), subtitle="streaming multiprocessors", status="enhanced")
    with c4:
        theme.render_metric_card("Warp Size", _fmt(m.gpu_warp_size), subtitle="threads/warp", status="enhanced")
    c5, c6, c7 = st.columns(3)
    with c5:
        theme.render_metric_card("VRAM", _fmt_bytes(m.gpu_vram_bytes), status="enhanced")
    with c6:
        theme.render_metric_card("Free VRAM", _fmt_bytes(m.gpu_free_vram_bytes), status="enhanced")
    with c7:
        theme.render_metric_card("Max Threads/Block", _fmt(m.gpu_max_threads_per_block), status="enhanced")


def render_cuda_hierarchy_explanation() -> None:
    theme.render_section_label("CUDA Execution Hierarchy")
    st.markdown(
        "```text\n"
        "Thread\n"
        "  ↓  (grouped in units of warp_size, scheduled together)\n"
        "Warp\n"
        "  ↓  (many warps grouped, share a block's shared memory)\n"
        "Thread Block\n"
        "  ↓  (blocks are scheduled onto SMs, many blocks per SM)\n"
        "Streaming Multiprocessor (SM)\n"
        "  ↓  (many SMs on one device)\n"
        "GPU\n"
        "```"
    )
    st.caption(
        "A CUDA thread is not the same execution model as a CPU thread -- see the comparison table below. "
        "Threads within a warp execute the same instruction in lockstep on the SM's hardware; a block's "
        "warps are scheduled independently, and many blocks can be resident on one SM simultaneously, "
        "which is how a GPU keeps thousands of threads in flight."
    )


def render_launch_metrics_section(m: ThreadingMetrics) -> None:
    theme.render_section_label(
        f"Kernel Launch Configuration — {m.representative_width}×{m.representative_height}, "
        f"batch={m.representative_batch_size}")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        theme.render_metric_card("Block", _fmt_dims(m.block_dimensions))
    with c2:
        theme.render_metric_card("Grid", _fmt_dims(m.grid_dimensions))
    with c3:
        theme.render_metric_card("Threads/Block", _fmt(m.threads_per_block))
    with c4:
        theme.render_metric_card("Warps/Block", _fmt(m.warps_per_block))
    with c5:
        theme.render_metric_card("Total Threads Launched", _fmt(f"{m.total_threads_launched:,}" if m.total_threads_launched else None))

    if m.grid_dimensions and m.block_dimensions:
        gx, gy, gz = m.grid_dimensions
        bx, by, bz = m.block_dimensions
        st.caption(
            f"grid.z = {gz} maps directly to the batch dimension (`blockIdx.z`) every production kernel "
            f"uses for batched execution -- one 3-D grid launch processes the whole batch, not one launch "
            f"per image."
        )


def render_parallelism_metric(m: ThreadingMetrics) -> None:
    theme.render_section_label("Parallelism")
    w, h, n = m.representative_width, m.representative_height, m.representative_batch_size
    pixels_per_image = w * h
    total_pixels = pixels_per_image * n
    threads = m.total_threads_launched
    threads_per_pixel = (threads / total_pixels) if (threads and total_pixels) else None
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        theme.render_metric_card("Images", f"{n:,}")
    with c2:
        theme.render_metric_card("Pixels/Image", f"{pixels_per_image:,}", subtitle=f"{w}×{h}")
    with c3:
        theme.render_metric_card("Total Pixels", f"{total_pixels:,}")
    with c4:
        theme.render_metric_card("CUDA Threads Launched", f"{threads:,}" if threads else "N/A")
    with c5:
        theme.render_metric_card("Threads/Pixel", f"{threads_per_pixel:.2f}" if threads_per_pixel else "N/A",
                                  subtitle="per single-pass filter")
    st.caption(
        "One CUDA thread computes one output pixel for a single-pass filter (Median/Sobel/Laplacian), one "
        "element of an intermediate/output plane per pass for separable Gaussian (2 launches), or up to 4 "
        "pixels for Threshold's vectorized uchar4 path -- see the per-filter breakdown below. A single "
        "thread never computes an entire filter's output for a whole image."
    )


def render_per_filter_breakdown(m: ThreadingMetrics) -> None:
    theme.render_section_label("Per-Filter Threading (Production Enhanced Variants)")
    cols = st.columns(len(m.per_filter))
    for col, f in zip(cols, m.per_filter):
        with col:
            st.markdown(f"**{f.filter_name}**")
            st.caption(f"{f.launches_per_call} launch(es)/call")
            st.caption(f"{f.pixels_per_thread} px/thread")
            st.caption(f"Block: {_fmt_dims(m.block_dimensions)}")
            st.caption(f.note)


def render_cpu_vs_cuda_comparison() -> None:
    theme.render_section_label("CPU Thread vs CUDA Thread")
    st.table({
        "CPU Thread": [
            "Software execution context",
            "Scheduled by CPU OS/runtime",
            "Runs on a CPU core",
            "Fewer concurrent threads (tens)",
            "General purpose",
        ],
        "CUDA Thread": [
            "GPU execution thread",
            "Scheduled by GPU hardware",
            "Executes on a GPU SM",
            "Thousands of threads resident at once",
            "Designed for massively parallel, data-independent workloads",
        ],
    })
    st.caption(
        "These are different execution models, not the same concept at different scale -- a CUDA thread is "
        "far lighter-weight than a CPU thread, and GPU hardware schedules warps of 32 directly, which has "
        "no CPU equivalent."
    )


def render_why_gpu_accelerates() -> None:
    theme.render_section_label("Why GPUs Can Accelerate This Workload")
    st.markdown(
        "The five-filter image pipeline contains large numbers of independent pixel operations -- each "
        "output pixel (for most filters) depends only on a small, fixed neighborhood of input pixels, not "
        "on any other output pixel. CUDA maps these independent operations across many GPU threads, "
        "allowing thousands of threads to be resident while the hardware schedules them across streaming "
        "multiprocessors. Not every operation in the pipeline is perfectly parallel end-to-end -- the five "
        "filters still run one after another (each depends on the previous stage's full output), and "
        "host↔device transfers are a separate, partly serial cost (see Performance Analytics for the "
        "measured H2D/D2H breakdown)."
    )


def render_pipeline_parallelism_timeline() -> None:
    theme.render_section_label("Pipeline Execution Timeline")
    st.markdown("**Production (sequential):**")
    st.markdown(
        "```text\n"
        "H2D ──┐\n"
        "      ├─ Gaussian ─ Median ─ Sobel ─ Laplacian ─ Threshold ─┐\n"
        "      │                                                     ├─ D2H\n"
        "```"
    )
    st.caption("This is what `run_basic_cuda_pipeline()` / `run_enhanced_cuda_pipeline()` actually execute today.")

    with st.expander("EXPERIMENTAL — Multi-stream overlap (Section 20F, not production)", expanded=False):
        st.markdown(
            "```text\n"
            "H2D(chunk N)      ──┐\n"
            "Compute(chunk N-1)  ├─ overlap possible on separate CUDA streams\n"
            "D2H(chunk N-2)    ──┘\n"
            "```"
        )
        st.markdown(theme.render_pill("EXPERIMENTAL — NOT ADOPTED", "experimental"), unsafe_allow_html=True)
        st.caption(
            "Section 20F measured this directly: genuine H2D/compute and compute/D2H overlap were confirmed "
            "via CUDA events, but isolated microbenchmarks regressed 1.4-3.3x (chunking overhead exceeded "
            "the achievable overlap on this GPU's single async copy engine) while the realistic workload "
            "showed a modest ~1.07-1.12x gain -- an EXPERIMENTAL result, not adopted into production. See "
            "`research/async_pipeline_decision.md`."
        )


def render_cuda_graph_status() -> None:
    theme.render_section_label("CUDA Graphs")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Basic pipeline** {theme.render_pill('EXPERIMENTAL', 'experimental')}", unsafe_allow_html=True)
        st.caption("Section 20D — realistic-workload gain ~1.44-1.56x; a wash at the largest batch size tested. Not wired into production.")
    with c2:
        st.markdown(f"**Enhanced pipeline** {theme.render_pill('EXPERIMENTAL', 'experimental')}", unsafe_allow_html=True)
        st.caption("Section 20E — realistic-workload gain ~1.89x, the strongest result of the optimization arc. Not wired into production.")


def render_basic_vs_enhanced_launch_comparison(m: ThreadingMetrics) -> None:
    theme.render_section_label("Basic vs Enhanced — Launch Configuration")
    st.markdown(
        f"Both implementations use the **same block** ({_fmt_dims(m.block_dimensions)}) and the **same grid** "
        f"({_fmt_dims(m.grid_dimensions)}) for this image size and batch -- verified directly across all five "
        f"filters (Section 22 audit: every `DEFAULT_*_ENHANCED_BLOCK` constant is `(16, 16)`, identical to "
        f"Basic's `default_block_dim()`). The two implementations differ in **kernel algorithm** (shared "
        f"memory tiling, constant memory, compile-time specialization for Enhanced) — never in the "
        f"launch geometry itself for this project's production kernels."
    )


def render_navigation_links() -> None:
    st.caption("→ Per-filter variant history and live comparisons: see the **Optimization Lab** tab.")
    st.caption("→ Historical benchmark charts and canonical run data: see the **Performance Analytics** tab.")


def render_threading_tab(width: int = 224, height: int = 224, batch_size: int = 1) -> ThreadingMetrics:
    """Renders the full Threading & Parallelism tab and returns the
    ThreadingMetrics used, so callers (e.g. Presentation Mode) can reuse
    the same measured values without re-querying."""
    theme.inject_css()
    m = get_threading_metrics(width=width, height=height, batch_size=batch_size)

    st.markdown(
        "CPU threads and CUDA threads are different execution models. This tab shows the real, measured "
        "configuration this session's hardware and pipeline actually use — nothing here is estimated."
    )

    render_cpu_section(m)
    render_gpu_hardware_section(m)
    render_cuda_hierarchy_explanation()
    render_launch_metrics_section(m)
    render_parallelism_metric(m)
    render_per_filter_breakdown(m)
    render_cpu_vs_cuda_comparison()
    render_why_gpu_accelerates()
    render_basic_vs_enhanced_launch_comparison(m)
    render_pipeline_parallelism_timeline()
    render_cuda_graph_status()

    with st.expander("Profiler-only metrics (registers/thread, occupancy, GPU utilization)"):
        c1, c2, c3 = st.columns(3)
        with c1:
            theme.render_metric_card("Registers/Thread", "Not profiled")
        with c2:
            theme.render_metric_card("Occupancy", "Not profiled")
        with c3:
            theme.render_metric_card("GPU Utilization", "Not profiled")
        st.caption(
            "Nsight Compute (registers/occupancy) and Nsight Systems (GPU utilization) both require "
            "Administrator privileges that were not available in this environment (confirmed independently "
            "in Sections 20D, 20E, and 20F) -- these values are never estimated from threads/block or any "
            "other proxy. Section 20A's static, historical register-count data (from a fresh `nvcc "
            "--ptxas-options=-v` compile, not a live profile) is available in "
            "`research/filter_optimization_matrix.md` and informed which optimizations were kept."
        )

    render_navigation_links()
    return m
