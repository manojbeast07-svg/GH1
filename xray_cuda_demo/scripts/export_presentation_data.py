"""Section 24: exports real, already-measured project data into
presentation/public/data/*.json for the standalone React presentation
app to consume.

This script NEVER computes a new number -- every value here is read
directly from an existing stored artifact (benchmark_results/, the
canonical designation, pipeline.threading_metrics, pipeline.environment,
xray_cuda.device_info()) or hand-transcribed from this project's own
research/*_decision.md documents (Sections 20B-20F), which themselves
record real measured results. If a source is unavailable, the exported
field is `null`, never a fabricated placeholder -- the React app renders
`null` as "Not available" (spec item 6).

Usage:
    python scripts/export_presentation_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "presentation" / "public" / "data"


def _write(name: str, data) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    print(f"wrote {path} ({path.stat().st_size:,} bytes)")


def export_benchmark_summary(canonical_id):
    from ui import services

    summary = services.load_benchmark_summary(canonical_id)
    if summary is None:
        _write("benchmark_summary.json", None)
        return None
    out = {
        "benchmark_id": summary["benchmark_id"],
        "timestamp_utc": summary["timestamp_utc"],
        "dataset_fingerprint": summary["dataset_fingerprint"],
        "manifest": {
            "seed": summary["manifest"]["seed"],
            "selected_image_count": summary["manifest"]["selected_image_count"],
            "resolution": summary["manifest"]["resolution"],
        },
        "cpu": summary["cpu"],
        "basic_cuda": summary["basic_cuda"],
        "enhanced_cuda": summary["enhanced_cuda"],
        "cpu_images_per_second": summary.get("cpu_images_per_second"),
        "basic_images_per_second": summary.get("basic_images_per_second"),
        "enhanced_images_per_second": summary.get("enhanced_images_per_second"),
        "speedups": summary["speedups"],
    }
    _write("benchmark_summary.json", out)
    return summary


def export_per_filter(summary):
    per_filter = summary.get("per_filter") if summary else None
    if not per_filter:
        _write("per_filter_results.json", None)
        return
    rows = per_filter.get("rows", [])
    out = [{
        "filter": r["filter"],
        "basic_kernel_ms": r["basic_kernel_ms"]["median"],
        "enhanced_kernel_ms": r["enhanced_kernel_ms"]["median"],
        "kernel_speedup": r["kernel_speedup"],
        "absolute_reduction_ms": r["absolute_reduction_ms"],
        "pct_of_total_compute_reduction": r["pct_of_total_compute_reduction"],
    } for r in rows]
    _write("per_filter_results.json", out)


def export_batch_sweep(summary):
    bs = summary.get("batch_sweep") if summary else None
    if not bs:
        _write("batch_sweep.json", None)
        return
    rows = bs.get("rows", [])
    out = [{
        "batch_size": r["effective_batch_size"],
        "cpu_total_ms": r["cpu_total_ms"]["median"],
        "basic_total_ms": r["basic_total_ms"]["median"],
        "enhanced_total_ms": r["enhanced_total_ms"]["median"],
        "cpu_images_per_second": r["cpu_images_per_second"],
        "basic_images_per_second": r["basic_images_per_second"],
        "enhanced_images_per_second": r["enhanced_images_per_second"],
        "basic_speedup_vs_cpu": r["basic_speedup_vs_cpu"],
        "enhanced_speedup_vs_cpu": r["enhanced_speedup_vs_cpu"],
    } for r in rows]
    _write("batch_sweep.json", out)


def export_resolution_sweep(summary):
    rs = summary.get("resolution_sweep") if summary else None
    if not rs:
        _write("resolution_sweep.json", None)
        return
    rows = rs.get("rows", [])
    out = [{
        "width": r["width"],
        "height": r["height"],
        "pixel_count": r["pixel_count"],
        "cpu_ms_per_image": r["cpu_ms_per_image"],
        "basic_ms_per_image": r["basic_ms_per_image"],
        "enhanced_ms_per_image": r["enhanced_ms_per_image"],
        "cpu_images_per_second": r["cpu_images_per_second"],
        "basic_images_per_second": r["basic_images_per_second"],
        "enhanced_images_per_second": r["enhanced_images_per_second"],
    } for r in rows]
    _write("resolution_sweep.json", out)


def export_correctness(summary):
    corr = summary.get("correctness") if summary else None
    if not corr:
        _write("correctness.json", None)
        return
    filter_level = corr.get("detailed", {}).get("filter_level", {})
    out = {
        "canonical_pipeline": corr["canonical_pipeline"],
        "filter_level": {
            name: {"tolerance": v["tolerance"], "pass": v["pass"],
                   "cpu_vs_basic_max_abs_diff": v["cpu_vs_basic_max_abs_diff"],
                   "basic_vs_enhanced_max_abs_diff": v["basic_vs_enhanced_max_abs_diff"]}
            for name, v in filter_level.items()
        },
    }
    _write("correctness.json", out)


def export_system():
    from ui import services

    env = services.environment_fingerprint()
    tools = services.build_tool_versions()
    mem = services.gpu_memory_info()
    out = {
        "environment_label": services.environment_label(),
        "os_platform": env["os_platform"],
        "cpu_model": env["cpu_model"],
        "cpu_logical_cores": env["cpu_logical_cores"],
        "gpu_available": env["gpu_available"],
        "gpu_name": env["gpu_name"],
        "gpu_vram_bytes": env["gpu_vram_bytes"],
        "gpu_free_vram_bytes": mem["free_bytes"] if mem else None,
        "gpu_compute_capability": env["gpu_compute_capability"],
        "gpu_driver_version": env["gpu_driver_version"],
        "cuda_runtime_version": env["cuda_runtime_version"],
        "nvcc_version": tools["nvcc"],
        "cmake_version": tools["cmake"],
        "python_version": env["python_version"],
        "opencv_version": env["opencv_version"],
        "numpy_version": env["numpy_version"],
        "pybind11_version": env["pybind11_version"],
    }
    _write("system.json", out)


def export_threading():
    from pipeline.threading_metrics import get_threading_metrics

    m = get_threading_metrics(width=224, height=224, batch_size=122)
    _write("threading.json", m.to_dict())


OPTIMIZATION_RESULTS = [
    {
        "id": "persistent_buffers",
        "name": "Persistent GPU Buffers",
        "status": "REJECTED",
        "section": "20B / 20C",
        "what_we_tried": "Keep GPU input/output buffers allocated across calls instead of allocating fresh "
                          "buffers on every request, to avoid repeated cudaMalloc/cudaFree overhead.",
        "what_happened": "1.15x-1.62x faster in a microbenchmark that repeated the exact same image shape and "
                          "configuration on every call -- but ~28% SLOWER in a realistic workload where batch "
                          "size, filter parameters, and the selected image change between calls, which is how "
                          "the real Streamlit application is actually used.",
        "why_not_production": "The realistic-workload measurement is the deciding one, and it was a clear "
                               "regression. The microbenchmark's own repeated-identical-call pattern doesn't "
                               "match real usage.",
    },
    {
        "id": "pinned_memory",
        "name": "Pinned Host Memory",
        "status": "REJECTED",
        "section": "20B",
        "what_we_tried": "Stage image data through page-locked ('pinned') host memory before transferring it "
                          "to the GPU, since pinned memory transfers are faster in isolation.",
        "what_happened": "An isolated transfer-only benchmark showed ~1.8-1.9x faster transfers. But the real "
                          "data path is always a plain NumPy array (pageable memory), so a copy into a pinned "
                          "buffer is required first -- and that staging copy's cost was larger than the "
                          "transfer speedup it unlocked.",
        "why_not_production": "The isolated benefit did not survive contact with how the application actually "
                               "receives its data.",
    },
    {
        "id": "cuda_graph_basic",
        "name": "CUDA Graphs — Basic Pipeline",
        "status": "EXPERIMENTAL",
        "section": "20D",
        "what_we_tried": "Record the sequence of GPU operations for the Basic pipeline once ('capture'), then "
                          "replay that recording on future calls instead of re-issuing each step individually, "
                          "reducing host-side launch overhead.",
        "what_happened": "~1.44x-1.56x faster in the realistic workload test -- a genuine, reproducible win -- "
                          "but roughly a wash at the largest batch size this project tests, where per-call "
                          "overhead is already a small fraction of total time.",
        "why_not_production": "Kept as a tested, working, documented capability -- not yet wired into the "
                               "default application, since its benefit is batch-size-dependent.",
    },
    {
        "id": "cuda_graph_enhanced",
        "name": "CUDA Graphs — Enhanced Pipeline",
        "status": "EXPERIMENTAL",
        "section": "20E",
        "what_we_tried": "The same graph-capture idea as above, applied to the Enhanced (optimized) kernels.",
        "what_happened": "~1.89x faster in the realistic workload test -- the single strongest result of the "
                          "entire optimization research effort.",
        "why_not_production": "No production-ready API surface has been built for it yet, and the Enhanced "
                               "kernels required copying (not modifying) code into an isolated file to make "
                               "capture possible, which is a real maintenance cost to weigh before adoption.",
    },
    {
        "id": "async_multistream",
        "name": "Async Multi-Stream Pipeline",
        "status": "EXPERIMENTAL",
        "section": "20F",
        "what_we_tried": "Split a batch into smaller chunks and overlap host-to-device transfer, GPU compute, "
                          "and device-to-host transfer across separate CUDA streams, so the GPU is never idle "
                          "waiting on a transfer.",
        "what_happened": "Isolated, same-shape microbenchmarks were 1.4x-3.3x SLOWER (the chunking overhead "
                          "outweighed the overlap gained on this GPU's single data-transfer engine) -- but the "
                          "realistic workload showed a modest, reproducible ~1.07x-1.12x improvement.",
        "why_not_production": "A genuinely mixed result -- positive on the metric that matters most, but "
                               "negative everywhere else, and the extra complexity (three CUDA streams, "
                               "careful chunk-size tuning) is hard to justify for a gain this small.",
    },
    {
        "id": "fusion",
        "name": "Laplacian + Threshold Fusion",
        "status": "EXPERIMENTAL",
        "section": "10",
        "what_we_tried": "Combine the Laplacian and Threshold stages into a single GPU kernel, avoiding one "
                          "round-trip through GPU memory between the two stages.",
        "what_happened": "A working, correctness-verified fused kernel exists, evaluated alongside the "
                          "per-filter optimization work.",
        "why_not_production": "Kept as a documented, tested alternative implementation rather than the "
                               "default production configuration.",
    },
]


def export_optimization_results():
    _write("optimization_results.json", OPTIMIZATION_RESULTS)


def main() -> int:
    from ui import services

    canonical_id = services.get_canonical_benchmark_id()
    print(f"Using canonical benchmark_id: {canonical_id}")

    summary = export_benchmark_summary(canonical_id)
    export_per_filter(summary)
    export_batch_sweep(summary)
    export_resolution_sweep(summary)
    export_correctness(summary)
    export_system()
    export_threading()
    export_optimization_results()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
