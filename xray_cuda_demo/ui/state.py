"""Session state (Section 12 spec item 53): st.session_state is the
single source of truth for current selection/seed/filter configuration/
last processing result/benchmark selection -- never a module-level
global variable (which would leak across Streamlit sessions/reruns).
"""

from __future__ import annotations

from typing import Any, Optional

import streamlit as st

from cpu.filters import FilterConfig

DEFAULTS = {
    "dataset_path": None,
    "selection_mode": "Random batch",
    "batch_size": 125,
    "seed": 42,
    "single_image_index": 0,
    "filter_config_kwargs": {},  # overrides on top of FilterConfig()'s defaults
    "implementation": "Enhanced CUDA",
    "compare_cpu": True,
    "compare_basic": True,
    "compare_enhanced": True,
    "presentation_mode": False,
    "debug_mode": False,
    "last_selection": None,           # ImageSelection
    "last_images": None,              # List[np.ndarray], same order as last_selection
    "last_result": None,              # dict from ui.services.run_compare(...) (batch mode)
    "preview_index": 0,               # which image in a batch to show detailed stages for
    "live_benchmark_result": None,
    "selected_benchmark_id": None,
    # -- Section 14: single-image live processing / comparison (spec item 30) --
    "last_selected_image": None,          # np.ndarray, the currently-loaded single image
    "last_image_meta": None,              # ui.services.ImageMetadata
    "last_filter_config": None,           # dict snapshot of the FilterConfig used for the last run
    "last_cpu_result": None,              # ui.services.ImplementationResult
    "last_basic_result": None,            # ui.services.ImplementationResult
    "last_enhanced_result": None,         # ui.services.ImplementationResult
    "last_comparison": None,              # ui.services.ComparisonResult (Process or Compare action)
    "last_run_timestamp": None,           # ISO-8601 UTC string
    "show_intermediate_comparison": False,
    "show_stage_differences": False,
    # -- Section 15: real-batch live processing (spec item 37) --
    "last_batch_selection": None,       # ImageSelection used for the last batch run
    "last_batch_images": None,          # List[np.ndarray], same order as last_batch_selection
    "last_batch_result": None,          # ui.services.BatchComparisonResult
    "last_preview_index": 0,            # flat (original-order) index of the batch preview image
    "last_batch_preview_stages": None,  # cached ui.services.get_preview_stage_outputs(...) for last_preview_index
    "quick_sweep_result": None,         # ui.services.run_quick_batch_sweep(...) result
    "show_batch_intermediate": False,
    "show_batch_differences": False,
    # -- Section 17: Interactive CUDA Optimization Lab (spec item 38-40) --
    "opt_lab_filter": "gaussian",       # currently selected filter in the lab
    "opt_lab_comparison": None,         # ui.services.VariantComparisonResult from the last live run
    "opt_lab_pipeline_impact": None,    # ui.services.PipelineImpactResult from the last pipeline-impact run
    "opt_lab_show_technical": False,
}


def init_session_state() -> None:
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)


def set_value(key: str, value: Any) -> None:
    st.session_state[key] = value


def current_filter_config() -> FilterConfig:
    """Builds a FilterConfig from session_state's stored overrides --
    the ONLY place app code constructs FilterConfig, so every tab reads
    the identical, currently-tuned configuration (spec item 33: slider
    changes must never mutate anything except this live in-memory
    config -- benchmark_results/ artifacts are never touched)."""
    return FilterConfig(**st.session_state.get("filter_config_kwargs", {}))


def update_filter_config_kwarg(key: str, value: Any) -> None:
    kwargs = dict(st.session_state.get("filter_config_kwargs", {}))
    kwargs[key] = value
    st.session_state["filter_config_kwargs"] = kwargs
