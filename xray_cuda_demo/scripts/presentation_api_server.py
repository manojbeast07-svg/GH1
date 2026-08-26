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
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cpu.filters import FilterConfig  # noqa: E402
from pipeline.threading_metrics import get_threading_metrics  # noqa: E402
from ui import services  # noqa: E402
from ui.services import ServiceError, _compute_pipeline_correctness, _compute_stage_correctness  # noqa: E402

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
        dm, _info = services.scan_dataset(DATASET_PATH)
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


@app.route("/api/dataset_info", methods=["GET"])
def dataset_info():
    try:
        _dm, info = services.scan_dataset(DATASET_PATH)
    except ServiceError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({
        "path": info.path,
        "total_files": info.total_files,
        "resolution_note": info.resolution_note,
    })


@app.route("/api/run", methods=["POST", "OPTIONS"])
def run():
    if request.method == "OPTIONS":
        return "", 204

    body = request.get_json(force=True, silent=True) or {}
    mode = body.get("mode", "single")
    batch_size = int(body.get("batch_size", 8))
    seed = int(body.get("seed", 42))
    image_index = int(body.get("image_index", 0))

    if not services.cuda_available():
        return jsonify({"error": services.cuda_unavailable_reason()}), 503

    try:
        dm, dataset_info_obj = services.scan_dataset(DATASET_PATH)
        selection = (
            services.select_single(dm, image_index)
            if mode == "single"
            else services.select_random_batch(dm, batch_size, seed)
        )
        images_all = services.load_selection_images(selection)
        groups = services.group_images_by_shape(images_all)
        dominant_shape = max(groups, key=lambda k: len(groups[k]))
        images = [images_all[i] for i in groups[dominant_shape]]

        config = FilterConfig()
        results = services.run_compare(images, config, True, True, True, preview_index=0)
        stage_correctness = _compute_stage_correctness(results, config)
        pipeline_correctness = _compute_pipeline_correctness(results)
    except ServiceError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover -- surfaced to the UI, never silently swallowed
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    implementations = {}
    for label, r in results.items():
        preview_png = None
        if r.final_outputs:
            preview_png = _encode_png(r.final_outputs[0])
        implementations[label] = {
            "total_ms": r.total_ms,
            "h2d_ms": r.h2d_ms,
            "compute_ms": r.compute_ms,
            "d2h_ms": r.d2h_ms,
            "per_stage_ms": r.per_stage_ms,
            "images_per_second": (len(r.final_outputs) / (r.total_ms / 1000.0)) if r.total_ms else None,
            "preview_png": preview_png,
        }

    cpu_total = implementations.get("CPU", {}).get("total_ms")
    speedups = {}
    for label in ("Basic CUDA", "Enhanced CUDA"):
        t = implementations.get(label, {}).get("total_ms")
        speedups[label] = (cpu_total / t) if (cpu_total and t) else None

    threading_config = _threading_payload(
        width=dominant_shape[1], height=dominant_shape[0], batch_size=len(images), block_x=16, block_y=16)

    return jsonify({
        "mode": mode,
        "batch_size": len(images),
        "resolution": [dominant_shape[0], dominant_shape[1]],
        "seed": seed if mode == "batch" else None,
        "dataset_total_files": dataset_info_obj.total_files,
        "implementations": implementations,
        "speedups_vs_cpu": speedups,
        "threading": threading_config,
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
