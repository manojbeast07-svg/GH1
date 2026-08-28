# CUDA X-Ray Processing Lab — Presentation App (Section 24)

A standalone, interactive React presentation for a **non-CUDA audience**. It explains the project visually:

```text
X-ray → CPU/OpenCV → Basic CUDA → Enhanced CUDA → "What changed?" → "Why is GPU faster?"
```

**This is a presentation/education layer, not the production backend, and it does not replace the
existing Streamlit application** (`../app.py`).

**No presentation mode: nothing here is pre-recorded.** Every benchmark, per-filter speedup, batch/resolution
sweep, correctness check, and system/threading number is computed **fresh, on every load**, by calling a small
local API server (`scripts/presentation_api_server.py`) that runs the real CPU/Basic/Enhanced pipeline through
`ui/services.py` — the same, unmodified backend the Streamlit app uses. This app REQUIRES that server running;
without it, every metric renders as "Not available" rather than falling back to a cached/historical number. A
"🔄 Recalculate Metrics" button in the sidebar re-runs everything on demand, so you can watch the numbers
change between runs (small run-to-run GPU timing variance is real, not a bug).

The one exception is the "What Didn't Work?" slide's five optimization-experiment summaries — those document
this project's own past research (Sections 20B–20F, `research/*_decision.md`), a historical record of what
was tried and why, not a metric that can be "recalculated" by re-running the pipeline. That one section still
loads from `public/data/optimization_results.json`.

## Data source — no fabricated numbers

Everything except the historical "What Didn't Work?" data comes from `GET /api/live_presentation_data`
(`scripts/presentation_api_server.py`), which calls `ui/services.py`'s existing `run_live_benchmark()`,
`run_live_per_filter_comparison()`, `run_quick_batch_sweep()`, `run_compare()`, the correctness helpers, and
the environment/threading introspection functions directly — no pipeline, kernel, or benchmark methodology is
reimplemented anywhere in this app or its API server. If a value isn't computable, the field is `null`; the
app renders that as "Not available" rather than guessing.

To (re)generate the static export used only for the historical optimization-experiment data:

```bash
# from the xray_cuda_demo/ project root, with the project's Python environment active
python scripts/export_presentation_data.py
```

## Live server (required)

The whole app needs this running alongside the React dev server — there is no static/pre-recorded fallback:

```bash
# from the xray_cuda_demo/ project root, with the project's Python environment active
python scripts/presentation_api_server.py            # listens on http://0.0.0.0:5001 by default
```

It is a thin integration layer, exactly parallel to `app.py` (Streamlit): it only calls `ui/services.py`'s
existing functions — no pipeline, kernel, or correctness logic is reimplemented, and no production file is
modified. If this server (or CUDA) isn't reachable, the app still loads and renders every slide, but every
live metric shows "Not available" instead of a number; the "Try It Yourself" section additionally shows a
message with the start command instead of a form.

## Requirements

- Node.js 18+ (developed and verified against Node v24.14.1 / npm 11.11.0)
- Python environment with `flask` (already in the project's `requirements.txt`) for the optional live-run
  server above

## Install and run

```bash
npm install
npm run dev
```

Opens at `http://localhost:5174` by default. Start `scripts/presentation_api_server.py` (see above) too if
you want the "Try It Yourself" and real-image pipeline sections to work.

## Running on a remote system / remote desktop

Both the Vite dev server and the Flask API server bind to `0.0.0.0` by default, so they're reachable by the
remote machine's IP address, not just `localhost` on that box:

```bash
python scripts/presentation_api_server.py     # http://0.0.0.0:5001
npm run dev                                   # http://0.0.0.0:5174
```

- **Connecting to the remote machine's desktop directly (RDP/VNC) and opening a browser there**: just use
  `http://localhost:5174` as usual — nothing else to configure.
- **Browsing to it from a different machine on the same network**: use the remote machine's IP or hostname
  instead, e.g. `http://<remote-ip>:5174`. `src/data/apiClient.js` calls the API on whatever hostname the
  page itself was loaded from (not a hardcoded `127.0.0.1`), so the "Try It Yourself" and pipeline-image
  features work the same way from a remote browser as they do locally.
- Make sure the remote machine's firewall allows inbound TCP on **5174** (Vite) and **5001** (Flask API) if
  you're accessing it remotely.
- The Flask API has no authentication — only run it with the default `0.0.0.0` bind on a trusted network.
  Pass `--host 127.0.0.1` to `presentation_api_server.py` to restrict it to the local machine only.
- For a production-style deployment instead of the dev server, use `npm run build && npm run preview`
  (also bound to `0.0.0.0:5174` — see below).

## Production build

```bash
npm run build
npm run preview
```

`vite.config.js` uses a relative `base: "./"`, so the built `dist/` folder can be opened from any path or
served as static files (a plain `npm run preview`, or any static file host) without assuming it's deployed
at a domain root.

## Tests

```bash
npm run test
```

Runs the Vitest + React Testing Library suite (`src/test/*.test.jsx`) covering data loading, benchmark/
threading rendering with real and missing data, filter/optimization interactivity, Simple/Technical mode,
the quiz, performance bars, navigation (keyboard and buttons), and presenter/fullscreen controls.

## Navigation

- `→` / `Space` — next slide
- `←` — previous slide
- `Home` / `End` — first / last slide
- `O` — toggle slide overview
- `P` — toggle presenter mode
- `F` — toggle fullscreen

## Structure

```text
presentation/
├── public/data/          generated JSON (scripts/export_presentation_data.py output)
├── src/
│   ├── components/       reusable UI primitives (MetricCard, BarChart, NavBar, ...)
│   ├── data/              loadPresentationData.js — the single data loader
│   ├── hooks/             usePresentationData, useModeContext, useKeyboardNav
│   ├── slides/            the 18 presentation screens
│   ├── styles/            theme.css
│   └── test/              Vitest + React Testing Library suite
├── index.html
├── package.json
└── vite.config.js
```

## What this app does NOT do

- Does not modify, call, or duplicate any CUDA kernel, CPU filter, or the production benchmark
  methodology.
- Does not replace or reimplement the production Streamlit application.
- Does not compute or estimate a value that wasn't actually measured — a missing metric always renders as
  "Not available" / "Not measured", never an invented number.
