"""Section 24 (follow-up): a small local HTTP API so the React
presentation app can trigger a REAL, live CPU/Basic/Enhanced run
(select a batch size or a single image, run it, see live metrics) --
the same interactive capability the Streamlit app's Live Processing tab
already has, exposed over HTTP instead of Streamlit's in-process model.

This file is a thin integration layer, exactly parallel to app.py: it
calls into ui/services.py's EXISTING functions (run_compare(),
_compute_stage_correctness(), _compute_pipeline_correctness(), dataset
selection/loading) -- it does not reimplement any pipeline, kernel, or
correctness logic, and it never modifies cpu/, cuda/, or ui/services.py
itself. No CUDA kernel, CPU filter, or benchmark methodology is touched.

Usage:
    python scripts/presentation_api_server.py [--port 5001] [--dataset PATH]

CORS is enabled manually (no new dependency) so the Vite dev server
(a different origin/port) can call this API directly.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml
from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cpu.filters import FilterConfig  # noqa: E402
from pipeline.image_loader import load_image  # noqa: E402
from pipeline.threading_metrics import get_threading_metrics  # noqa: E402
from ui import services  # noqa: E402
from ui.services import ServiceError, _compute_pipeline_correctness, _compute_stage_correctness  # noqa: E402

FILTER_CONFIG_FIELDS = [
    "gaussian_enabled", "gaussian_kernel_size", "gaussian_sigma",
    "median_enabled", "median_kernel_size",
    "sobel_enabled", "sobel_kernel_size", "sobel_mode",
    "laplacian_enabled", "laplacian_kernel_size", "laplacian_scale", "laplacian_delta",
    "threshold_enabled", "threshold_value", "threshold_max_value",
]


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")


def _filter_config_from_body(body: dict) -> FilterConfig:
    """Builds a REAL FilterConfig from whatever fields the request actually
    sent (nested under "filter_config", or flat in the body) -- unset
    fields keep FilterConfig's own defaults, never a value invented here.
    Section 25 spec item 9: full filter configuration control."""
    source = body.get("filter_config") if isinstance(body.get("filter_config"), dict) else body
    kwargs = {k: source[k] for k in FILTER_CONFIG_FIELDS if k in source}
    try:
        return FilterConfig(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ServiceError(f"Invalid filter configuration: {exc}") from exc


def _difference_png(result_a, result_b):
    """Real pixel difference (spec item 22) between two implementations'
    final output for the preview image -- same computation
    ui.services.save_comparison_results() already performs inline, reused
    here rather than duplicated as a new formula."""
    if not result_a or not result_b or not result_a.final_outputs or not result_b.final_outputs:
        return None
    diff = np.abs(result_a.final_outputs[0].astype(np.int16) - result_b.final_outputs[0].astype(np.int16)).astype(np.uint8)
    return _encode_png(diff)

app = Flask(__name__)


def _load_dataset_path() -> str:
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    raw = (cfg.get("dataset") or {}).get("path", "../data")
    p = Path(raw)
    return str(p if p.is_absolute() else (PROJECT_ROOT / p).resolve())


DATASET_PATH = _load_dataset_path()


def _encode_png(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


# Scanning the dataset walks every file under DATASET_PATH (9,463 files,
# ~470 ms measured) and produced identical results on every request, since
# the dataset does not change while the server is running. Section 25 item
# 65 lists dataset metadata as safe to cache; mutable CUDA execution state
# is deliberately NOT cached anywhere. Restart the server to pick up files
# added to the dataset.
_dataset_cache = {"dm": None, "info": None}
_dataset_lock = threading.Lock()


def _get_dataset():
    """Returns the (DatasetManager, DatasetInfo) pair, scanning once per
    process. Both are only read after the initial scan, so sharing them
    across Flask's request threads is safe."""
    with _dataset_lock:
        if _dataset_cache["dm"] is None:
            _dataset_cache["dm"], _dataset_cache["info"] = services.scan_dataset(DATASET_PATH)
        return _dataset_cache["dm"], _dataset_cache["info"]


@app.after_request
def _add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "available": True,
        "cuda_available": services.cuda_available(),
        "dataset_path": DATASET_PATH,
    })


def _threading_payload(width: int, height: int, batch_size: int, block_x: int, block_y: int) -> dict:
    """Reuses pipeline.threading_metrics.get_threading_metrics() (Section 23) --
    the exact same real, live-queried CPU/GPU/launch-configuration data the
    static Threading & Parallelism slide shows, just parameterized by the
    caller's chosen width/height/batch/block instead of the fixed default.
    Never a duplicated formula -- this is the identical function, called with
    different arguments."""
    m = get_threading_metrics(width=width, height=height, batch_size=batch_size, block=(block_x, block_y))
    return {
        "cpu_logical_processors": m.cpu_logical_processors,
        "gpu_sm_count": m.gpu_sm_count,
        "gpu_warp_size": m.gpu_warp_size,
        "gpu_max_threads_per_block": m.gpu_max_threads_per_block,
        "block_dimensions": m.block_dimensions,
        "grid_dimensions": m.grid_dimensions,
        "threads_per_block": m.threads_per_block,
        "warps_per_block": m.warps_per_block,
        "total_threads_launched": m.total_threads_launched,
        "representative_width": m.representative_width,
        "representative_height": m.representative_height,
        "representative_batch_size": m.representative_batch_size,
    }


@app.route("/api/launch_config", methods=["GET"])
def launch_config():
    """Live, interactive launch-configuration explorer backend (spec
    follow-up: 'more interactive metrics based on threads'). Query params:
    width, height, batch_size, block_x, block_y -- all optional, defaulting
    to this project's dominant resolution / a single image / the
    production 16x16 block."""
    try:
        width = int(request.args.get("width", 224))
        height = int(request.args.get("height", 224))
        batch_size = int(request.args.get("batch_size", 1))
        block_x = int(request.args.get("block_x", 16))
        block_y = int(request.args.get("block_y", 16))
    except (TypeError, ValueError):
        return jsonify({"error": "width/height/batch_size/block_x/block_y must be integers."}), 400

    if width <= 0 or height <= 0 or batch_size <= 0 or block_x <= 0 or block_y <= 0:
        return jsonify({"error": "width/height/batch_size/block_x/block_y must all be positive."}), 400
    if not services.cuda_available():
        return jsonify({"error": services.cuda_unavailable_reason()}), 503

    try:
        return jsonify(_threading_payload(width, height, batch_size, block_x, block_y))
    except Exception as exc:  # pragma: no cover -- surfaced to the UI, never silently swallowed
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


STAGE_ORDER = ["original", "gaussian", "median", "sobel", "laplacian", "threshold"]


@app.route("/api/preview_stages", methods=["GET"])
def preview_stages():
    """Real, per-stage filtered images for ONE image (spec follow-up:
    'try displaying images with filter') -- runs the actual production
    Enhanced CUDA pipeline (via ui.services.run_gpu_batch's existing
    preview_index mechanism, the same one Streamlit's Live Processing tab
    uses) on a single selected image and returns the original plus each
    of the five real filtered stage outputs as PNGs. No illustration, no
    synthetic gradient -- every image returned is an actual measured
    pipeline output."""
    try:
        image_index = int(request.args.get("image_index", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "image_index must be an integer."}), 400

    if not services.cuda_available():
        return jsonify({"error": services.cuda_unavailable_reason()}), 503

    try:
        dm, _info = _get_dataset()
        selection = services.select_single(dm, image_index)
        images = services.load_selection_images(selection)
        config = FilterConfig()
        result = services.run_gpu_batch(images, config, use_enhanced=True, preview_index=0)
    except ServiceError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover -- surfaced to the UI, never silently swallowed
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    if not result.stage_outputs:
        return jsonify({"error": "Stage outputs were not produced for this image."}), 500

    stages = {}
    for stage in STAGE_ORDER:
        arr = result.stage_outputs.get(stage)
        stages[stage] = _encode_png(arr) if arr is not None else None

    return jsonify({"image_index": image_index, "implementation": "Enhanced CUDA", "stages": stages})


def _sobel_coefficients_from_source() -> dict:
    """Reads the Sobel Gx/Gy operators out of the ACTUAL kernel source
    (cuda/src/sobel_enhanced.cu's kHostSobel*Coeffs) rather than restating
    them here -- same approach ui.services.gpu_implementation_facts()
    already uses to verify claims against source instead of asserting
    them. Returns None if the constants can't be located, so the UI shows
    "not available" instead of a plausible-looking guess."""
    import re

    source_path = PROJECT_ROOT / "cuda" / "src" / "sobel_enhanced.cu"
    if not source_path.exists():
        return None
    text = source_path.read_text(encoding="utf-8")

    def _parse(name):
        match = re.search(rf"{name}\[9\]\s*=\s*\{{([^}}]*)\}}", text)
        if not match:
            return None
        values = [float(v.strip().rstrip("f")) for v in match.group(1).split(",")]
        return [values[0:3], values[3:6], values[6:9]] if len(values) == 9 else None

    gx = _parse("kHostSobelGxCoeffs")
    gy = _parse("kHostSobelGyCoeffs")
    if gx is None or gy is None:
        return None
    return {"gx": gx, "gy": gy, "source": "cuda/src/sobel_enhanced.cu"}


@app.route("/api/filter_math", methods=["GET"])
def filter_math():
    """The REAL numbers behind one filter at one pixel: the actual
    neighborhood values that filter's input stage contained, the actual
    coefficients the production kernel uses (generated by the same
    cuda.gaussian / cuda.laplacian functions the kernels are fed with, or
    read straight out of the .cu source for Sobel), and the actual output
    value the pipeline produced at that pixel.

    Nothing here is illustrative: the neighborhood comes from a real
    dataset X-ray run through the real Enhanced CUDA pipeline, so the
    input values, the coefficients and the output value are all things
    that genuinely happened on the GPU."""
    try:
        image_index = int(request.args.get("image_index", 0))
        req_x = request.args.get("x")
        req_y = request.args.get("y")
    except (TypeError, ValueError):
        return jsonify({"error": "image_index must be an integer."}), 400

    stage = request.args.get("stage", "gaussian")
    if stage not in STAGE_ORDER or stage == "original":
        return jsonify({"error": f"stage must be one of {STAGE_ORDER[1:]}."}), 400

    if not services.cuda_available():
        return jsonify({"error": services.cuda_unavailable_reason()}), 503

    try:
        dm, _info = _get_dataset()
        selection = services.select_single(dm, image_index)
        images = services.load_selection_images(selection)
        config = FilterConfig()
        result = services.run_gpu_batch(images, config, use_enhanced=True, preview_index=0)
    except ServiceError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover -- surfaced to the UI, never silently swallowed
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    if not result.stage_outputs:
        return jsonify({"error": "Stage outputs were not produced for this image."}), 500

    input_stage = STAGE_ORDER[STAGE_ORDER.index(stage) - 1]
    input_arr = result.stage_outputs.get(input_stage)
    output_arr = result.stage_outputs.get(stage)
    if input_arr is None or output_arr is None:
        return jsonify({"error": f"Stage {stage!r} is disabled in this configuration, so it has no output."}), 400

    kernel_size = {
        "gaussian": config.gaussian_kernel_size,
        "median": config.median_kernel_size,
        "sobel": 3,
        "laplacian": max(3, config.laplacian_kernel_size),
        "threshold": 1,
    }[stage]

    height, width = input_arr.shape[:2]
    half = kernel_size // 2
    # Clamp to the interior so the full neighborhood genuinely exists --
    # no border rule is claimed or applied, we simply show a pixel whose
    # neighbours are all real pixels of this image.
    cx = int(req_x) if req_x is not None else width // 2
    cy = int(req_y) if req_y is not None else height // 2
    cx = max(half, min(width - 1 - half, cx))
    cy = max(half, min(height - 1 - half, cy))

    window = input_arr[cy - half: cy + half + 1, cx - half: cx + half + 1]
    neighborhood = [[int(v) for v in row] for row in window]

    coefficients = None
    if stage == "gaussian":
        from cuda.gaussian import gaussian_kernel_2d

        coeffs = gaussian_kernel_2d(config.gaussian_kernel_size, config.gaussian_sigma)
        coefficients = {
            "kind": "weights",
            "values": [[float(v) for v in row] for row in coeffs],
            "source": f"cuda.gaussian.gaussian_kernel_2d({config.gaussian_kernel_size}, {config.gaussian_sigma})",
        }
    elif stage == "laplacian":
        from cuda.laplacian import laplacian_kernel_2d

        coeffs = laplacian_kernel_2d(config.laplacian_kernel_size)
        coefficients = {
            "kind": "weights",
            "values": [[float(v) for v in row] for row in coeffs],
            "source": f"cuda.laplacian.laplacian_kernel_2d({config.laplacian_kernel_size})",
        }
    elif stage == "sobel":
        sobel = _sobel_coefficients_from_source()
        if sobel is not None:
            coefficients = {"kind": "sobel", "gx": sobel["gx"], "gy": sobel["gy"], "source": sobel["source"], "mode": config.sobel_mode}
    elif stage == "median":
        coefficients = {"kind": "rank", "source": "no coefficients -- median is a rank filter, it sorts the neighborhood"}
    elif stage == "threshold":
        coefficients = {
            "kind": "threshold",
            "threshold_value": config.threshold_value,
            "max_value": config.threshold_max_value,
            "source": "cpu.filters.FilterConfig (the same config passed to the kernel)",
        }

    return jsonify({
        "image_index": image_index,
        "stage": stage,
        "input_stage": input_stage,
        "x": cx, "y": cy, "width": int(width), "height": int(height),
        "kernel_size": kernel_size,
        "neighborhood": neighborhood,
        "input_center": int(input_arr[cy, cx]),
        "output_value": int(output_arr[cy, cx]),
        "coefficients": coefficients,
        "implementation": "Enhanced CUDA",
    })


@app.route("/api/image_thumbnails", methods=["GET"])
def image_thumbnails():
    """A browsable page of the real dataset, so a user can pick the image
    they actually want instead of guessing an index. Returns small
    thumbnails plus each file's real dataset index and relative path --
    the index is exactly what /api/run and /api/preview_stages expect, so
    what you pick is what gets processed.

    Optional `q` filters on the relative path (e.g. "fractured"), which is
    how the folder structure of this dataset encodes its classes."""
    try:
        start = max(0, int(request.args.get("start", 0)))
        count = max(1, min(60, int(request.args.get("count", 24))))
        size = max(32, min(160, int(request.args.get("size", 96))))
    except (TypeError, ValueError):
        return jsonify({"error": "start/count/size must be integers."}), 400

    query = (request.args.get("q") or "").strip().lower()

    try:
        dm, _info = _get_dataset()
        paths = dm.paths
        # Keep the ORIGINAL dataset index with each path: filtering must not
        # renumber images, or a picked thumbnail would run a different file.
        indexed = list(enumerate(paths))
        if query:
            indexed = [(i, p) for i, p in indexed if query in dm.relative_path(p).lower().replace("\\", "/")]

        total = len(indexed)
        page = indexed[start: start + count]

        items = []
        for index, path in page:
            image = load_image(path)
            h, w = image.shape[:2]
            scale = size / max(h, w)
            thumb = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
            items.append({
                "index": index,
                "relative_path": dm.relative_path(path).replace("\\", "/"),
                "filename": Path(path).name,
                "width": int(w), "height": int(h),
                "thumbnail": _encode_png(thumb),
            })
    except ServiceError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover -- surfaced to the UI, never silently swallowed
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    return jsonify({"total": total, "start": start, "count": len(items), "query": query, "items": items})


@app.route("/api/dataset_info", methods=["GET"])
def dataset_info():
    try:
        _dm, info = _get_dataset()
    except ServiceError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({
        "path": info.path,
        "total_files": info.total_files,
        "resolution_note": info.resolution_note,
    })


@app.route("/api/system_info", methods=["GET"])
def system_info():
    """Live environment/hardware/toolchain facts (Section 25 System page)
    -- cheap, read-only introspection (no pipeline execution), so this is
    safe to auto-load, unlike anything that runs the CPU/CUDA pipeline."""
    fingerprint = services.environment_fingerprint()
    return jsonify({
        "cuda_available": services.cuda_available(),
        "environment_label": services.environment_label(),
        "fingerprint": fingerprint,
        "tool_versions": services.build_tool_versions(),
        "gpu_implementation_facts": services.gpu_implementation_facts(),
        "gpu_memory": services.gpu_memory_info(),
    })


@app.route("/api/run", methods=["POST", "OPTIONS"])
def run():
    """The core "explicit action" endpoint (Section 25 spec items 10-22):
    runs exactly the implementations the caller asked for (never all
    three unless requested -- [Run CPU]/[Run Basic CUDA]/[Run Enhanced
    CUDA]/[Compare All] are all this one endpoint with different
    run_cpu/run_basic/run_enhanced flags), on the SAME loaded images/
    config for every implementation (the "same input" guarantee is
    enforced by construction: one `images` list and one `config` object
    are passed to every implementation below, never resampled). Every
    field in the response is measured on THIS call -- LIVE, never a
    historical/pre-recorded value."""
    if request.method == "OPTIONS":
        return "", 204

    body = request.get_json(force=True, silent=True) or {}
    mode = body.get("mode", "single")
    batch_size = int(body.get("batch_size", 8))
    seed = int(body.get("seed", 42))
    image_index = int(body.get("image_index", 0))
    run_cpu = bool(body.get("run_cpu", True))
    run_basic = bool(body.get("run_basic", True))
    run_enhanced = bool(body.get("run_enhanced", True))

    if not (run_cpu or run_basic or run_enhanced):
        return jsonify({"error": "At least one of run_cpu/run_basic/run_enhanced must be true."}), 400
    if (run_basic or run_enhanced) and not services.cuda_available():
        return jsonify({"error": services.cuda_unavailable_reason()}), 503

    try:
        config = _filter_config_from_body(body)
        dm, dataset_info_obj = _get_dataset()
        selection = (
            services.select_single(dm, image_index)
            if mode == "single"
            else services.select_random_batch(dm, batch_size, seed)
        )
        t0 = time.perf_counter()
        images_all = services.load_selection_images(selection)
        load_ms = (time.perf_counter() - t0) * 1000.0
        groups = services.group_images_by_shape(images_all)
        dominant_shape = max(groups, key=lambda k: len(groups[k]))
        images = [images_all[i] for i in groups[dominant_shape]]  # the ONE image list every implementation below receives

        results = services.run_compare(images, config, run_cpu, run_basic, run_enhanced, preview_index=0)
        stage_correctness = _compute_stage_correctness(results, config)
        pipeline_correctness = _compute_pipeline_correctness(results)
    except ServiceError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover -- surfaced to the UI, never silently swallowed
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    implementations = {}
    for label, r in results.items():
        preview_png = _encode_png(r.final_outputs[0]) if r.final_outputs else None
        implementations[label] = {
            "total_ms": r.total_ms,
            "h2d_ms": r.h2d_ms,
            "compute_ms": r.compute_ms,
            "d2h_ms": r.d2h_ms,
            "per_stage_ms": r.per_stage_ms,  # actual per-stage CUDA-event / CPU-timer values, never total_ms / stage_count
            "images_per_second": (len(r.final_outputs) / (r.total_ms / 1000.0)) if r.total_ms else None,
            "preview_png": preview_png,
        }

    cpu_total = implementations.get("CPU", {}).get("total_ms")
    speedups = {}
    for label in ("Basic CUDA", "Enhanced CUDA"):
        t = implementations.get(label, {}).get("total_ms")
        speedups[label] = (cpu_total / t) if (cpu_total and t) else None
    if implementations.get("Basic CUDA", {}).get("total_ms") and implementations.get("Enhanced CUDA", {}).get("total_ms"):
        speedups["Enhanced vs Basic"] = implementations["Basic CUDA"]["total_ms"] / implementations["Enhanced CUDA"]["total_ms"]

    differences = {
        "basic_vs_cpu": _difference_png(results.get("CPU"), results.get("Basic CUDA")),
        "enhanced_vs_cpu": _difference_png(results.get("CPU"), results.get("Enhanced CUDA")),
        "enhanced_vs_basic": _difference_png(results.get("Basic CUDA"), results.get("Enhanced CUDA")),
    }

    threading_config = _threading_payload(
        width=dominant_shape[1], height=dominant_shape[0], batch_size=len(images), block_x=16, block_y=16)

    return jsonify({
        "run_id": _new_run_id(),
        "provenance": "LIVE",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "batch_size": len(images),
        "resolution": [dominant_shape[0], dominant_shape[1]],
        "seed": seed if mode == "batch" else None,
        "load_ms": load_ms,
        "dataset_total_files": dataset_info_obj.total_files,
        "filter_config": asdict(config),
        "implementations_requested": {"CPU": run_cpu, "Basic CUDA": run_basic, "Enhanced CUDA": run_enhanced},
        "implementations": implementations,
        "speedups_vs_cpu": speedups,
        "same_input_verification": {
            # True by construction: every implementation above received this
            # exact same `images` list and `config` object, never resampled
            # or reconfigured between calls -- not an assumption, the fact
            # that they share one Python object IS the guarantee.
            "same_input": True,
            "same_configuration": True,
        },
        "threading": threading_config,
        "differences": differences,
        "stage_correctness": [
            {"stage": s.stage, "tolerance": s.tolerance, "max_abs_diff_vs_cpu": s.max_abs_diff_vs_cpu, "status": s.status}
            for s in stage_correctness
        ],
        "pipeline_correctness": [
            {
                "comparison": p.comparison, "max_abs_diff": p.max_abs_diff, "mean_abs_diff": p.mean_abs_diff,
                "rmse": p.rmse, "differing_pixel_count": p.differing_pixel_count,
                "differing_pixel_percentage": p.differing_pixel_percentage,
            }
            for p in pipeline_correctness
        ],
    })


@app.route("/api/live/per_filter", methods=["POST", "OPTIONS"])
def live_per_filter():
    """Explicit, user-triggered per-filter Basic-vs-Enhanced live timing
    (Section 25 Filters page) -- never runs automatically on page load
    (spec item 56)."""
    if request.method == "OPTIONS":
        return "", 204
    if not services.cuda_available():
        return jsonify({"error": services.cuda_unavailable_reason()}), 503

    body = request.get_json(force=True, silent=True) or {}
    batch_size = int(body.get("batch_size", 16))
    seed = int(body.get("seed", 42))

    try:
        config = _filter_config_from_body(body)
        dm, _info = _get_dataset()
        selection = services.select_random_batch(dm, batch_size, seed)
        images_all = services.load_selection_images(selection)
        groups = services.group_images_by_shape(images_all)
        dominant_shape = max(groups, key=lambda k: len(groups[k]))
        images = [images_all[i] for i in groups[dominant_shape]]
        rows = services.run_live_per_filter_comparison(images, config, warmup_runs=1, measurement_runs=3)
    except ServiceError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover -- surfaced to the UI, never silently swallowed
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    return jsonify({
        "run_id": _new_run_id(), "provenance": "LIVE",
        "resolution": [dominant_shape[0], dominant_shape[1]], "n_images": len(images),
        "per_filter_results": rows,
    })


@app.route("/api/live/batch_sweep", methods=["POST", "OPTIONS"])
def live_batch_sweep():
    """Explicit, user-triggered batch-size sweep (Section 25 spec items
    24-25: "[Run Batch Size Test]" -- only executes when this endpoint is
    called, plots only completed measurements, never a fabricated curve)."""
    if request.method == "OPTIONS":
        return "", 204
    if not services.cuda_available():
        return jsonify({"error": services.cuda_unavailable_reason()}), 503

    body = request.get_json(force=True, silent=True) or {}
    batch_sizes = sorted({int(b) for b in body.get("batch_sizes", [1, 8, 32, 64]) if int(b) > 0})
    seed = int(body.get("seed", 42))
    if not batch_sizes:
        return jsonify({"error": "batch_sizes must contain at least one positive integer."}), 400

    try:
        config = _filter_config_from_body(body)
        dm, _info = _get_dataset()
        pool_size = max(batch_sizes) * 2  # headroom so the dominant-shape group still covers the largest requested size
        selection = services.select_random_batch(dm, min(pool_size, len(dm.paths)), seed)
        images_all = services.load_selection_images(selection)
        groups = services.group_images_by_shape(images_all)
        dominant_shape = max(groups, key=lambda k: len(groups[k]))
        images_pool = [images_all[i] for i in groups[dominant_shape]]
        sweep = services.run_quick_batch_sweep(images_pool, config, batch_sizes, True, True, True)
    except ServiceError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover -- surfaced to the UI, never silently swallowed
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    return jsonify({
        "run_id": _new_run_id(), "provenance": "LIVE",
        "resolution": [dominant_shape[0], dominant_shape[1]], "pool_size": len(images_pool),
        "rows": sweep["rows"],
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Bind address. Default 0.0.0.0 so this is reachable from another machine "
             "(e.g. viewing the presentation over a remote desktop session by IP). "
             "Pass --host 127.0.0.1 to restrict it to this machine only.",
    )
    args = parser.parse_args()
    print(f"Dataset: {DATASET_PATH}")
    print(f"Presentation API server listening on http://{args.host}:{args.port}")
    if args.host == "0.0.0.0":
        print("Reachable from other machines on this network -- there is no authentication on this API,")
        print("so only run it on a trusted network. Use --host 127.0.0.1 to restrict it to this machine.")
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
