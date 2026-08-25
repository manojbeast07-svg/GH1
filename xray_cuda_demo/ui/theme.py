"""Section 23: shared visual theme (CSS injection) and reusable UI
components for the CUDA X-Ray Processing Lab. Presentation only -- this
module never calls into the backend, never touches a compute path, and
has zero effect on any measured value. Base color theme lives in
`.streamlit/config.toml`; this module adds card/badge styling on top and
the `render_metric_card()` / `render_status_badge()` components every
tab uses instead of duplicating HTML/CSS per call site (spec item 42).
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

# Semantic colors (spec item 43: consistent meaning, always paired with a text label,
# never color-only).
COLOR_CPU = "#4C8DF6"
COLOR_BASIC = "#F5A623"
COLOR_ENHANCED = "#39D98A"
COLOR_EXPERIMENTAL = "#B892FF"
COLOR_PASS = "#39D98A"
COLOR_WARNING = "#F5A623"
COLOR_ERROR = "#F0555A"
COLOR_MUTED = "#8B949E"

_CSS = """
<style>
.xrl-card {
    background: #161B22;
    border: 1px solid #2A313C;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
}
.xrl-card-title {
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #8B949E;
    margin-bottom: 0.15rem;
}
.xrl-card-value {
    font-size: 1.55rem;
    font-weight: 650;
    line-height: 1.2;
    color: #E6EDF3;
}
.xrl-card-subtitle {
    font-size: 0.80rem;
    color: #8B949E;
    margin-top: 0.15rem;
}
.xrl-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: #161B22;
    border: 1px solid #2A313C;
    border-radius: 999px;
    padding: 0.18rem 0.7rem;
    font-size: 0.78rem;
    color: #E6EDF3;
    margin-right: 0.4rem;
}
.xrl-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}
.xrl-section-label {
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8B949E;
    border-bottom: 1px solid #2A313C;
    padding-bottom: 0.25rem;
    margin: 0.9rem 0 0.5rem 0;
}
.xrl-pill {
    display: inline-block;
    border-radius: 4px;
    padding: 0.05rem 0.45rem;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
</style>
"""


def inject_css() -> None:
    """Injects the shared CSS once per page render. Idempotent -- safe to
    call from every tab/module that needs it; Streamlit re-renders the
    whole script per interaction anyway, so no session-state guard is
    needed."""
    st.markdown(_CSS, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, subtitle: Optional[str] = None,
                        status: Optional[str] = None) -> None:
    """Reusable compact metric card (spec item 42) -- the ONE place this
    HTML is written, reused across the header, Threading tab, System tab,
    and Presentation Mode rather than duplicated per call site.

    `status` is optional and purely decorative (a small colored dot) --
    the text `title`/`value`/`subtitle` always carry the actual meaning,
    per spec item 43's "never rely only on color" requirement.
    """
    dot = ""
    if status:
        color = {"pass": COLOR_PASS, "warning": COLOR_WARNING, "error": COLOR_ERROR,
                 "cpu": COLOR_CPU, "basic": COLOR_BASIC, "enhanced": COLOR_ENHANCED,
                 "experimental": COLOR_EXPERIMENTAL}.get(status, COLOR_MUTED)
        dot = f'<span class="xrl-dot" style="background:{color};margin-right:0.3rem;"></span>'
    subtitle_html = f'<div class="xrl-card-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="xrl-card">'
        f'<div class="xrl-card-title">{dot}{title}</div>'
        f'<div class="xrl-card-value">{value}</div>'
        f'{subtitle_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_status_badge(label: str, value: str, status: str = "pass") -> str:
    """Returns (does not render) one status-badge HTML fragment; callers
    join several into one st.markdown call so badges lay out inline on
    one row (spec item 5's header status badges)."""
    color = {"pass": COLOR_PASS, "warning": COLOR_WARNING, "error": COLOR_ERROR}.get(status, COLOR_MUTED)
    return (
        f'<span class="xrl-badge"><span class="xrl-dot" style="background:{color};"></span>'
        f'{label}: <b>{value}</b></span>'
    )


def render_section_label(text: str) -> None:
    st.markdown(f'<div class="xrl-section-label">{text}</div>', unsafe_allow_html=True)


def render_pill(text: str, status: str = "experimental") -> str:
    """Returns an inline HTML pill (e.g. "EXPERIMENTAL", "PASS") for
    embedding inside other markdown strings."""
    color = {"pass": COLOR_PASS, "warning": COLOR_WARNING, "error": COLOR_ERROR,
              "experimental": COLOR_EXPERIMENTAL, "cpu": COLOR_CPU, "basic": COLOR_BASIC,
              "enhanced": COLOR_ENHANCED}.get(status, COLOR_MUTED)
    return f'<span class="xrl-pill" style="background:{color}22;color:{color};border:1px solid {color}55;">{text}</span>'
