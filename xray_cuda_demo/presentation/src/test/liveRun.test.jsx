import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { LiveRun } from "../slides/19_LiveRun.jsx";

const FAKE_RUN_RESULT = {
  mode: "batch",
  batch_size: 4,
  resolution: [224, 224],
  implementations: {
    CPU: { total_ms: 9, h2d_ms: null, compute_ms: 9, d2h_ms: null, preview_png: null },
    "Basic CUDA": { total_ms: 5, h2d_ms: 0.1, compute_ms: 4.5, d2h_ms: 0.4, preview_png: "data:image/png;base64,AAA" },
    "Enhanced CUDA": { total_ms: 1, h2d_ms: 0.06, compute_ms: 0.8, d2h_ms: 0.14, preview_png: "data:image/png;base64,BBB" },
  },
  speedups_vs_cpu: { "Basic CUDA": 1.8, "Enhanced CUDA": 9.0 },
  threading: {
    grid_dimensions: [14, 14, 4],
    threads_per_block: 256,
    warps_per_block: 8,
    total_threads_launched: 200704,
  },
  stage_correctness: [
    { stage: "gaussian", tolerance: 1, max_abs_diff_vs_cpu: 1, status: "PASS" },
    { stage: "median", tolerance: 0, max_abs_diff_vs_cpu: 0, status: "PASS" },
  ],
  pipeline_correctness: [
    { comparison: "basic_vs_cpu", max_abs_diff: 255, mean_abs_diff: 0.03, rmse: 2.9, differing_pixel_count: 27, differing_pixel_percentage: 0.013 },
    { comparison: "enhanced_vs_basic", max_abs_diff: 0, mean_abs_diff: 0, rmse: 0, differing_pixel_count: 0, differing_pixel_percentage: 0 },
  ],
};

describe("Live Run (interactive real pipeline execution)", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("shows an unavailable message and start instructions when the server can't be reached", async () => {
    global.fetch = vi.fn(() => Promise.reject(new Error("connection refused")));
    render(<LiveRun />);
    await waitFor(() => expect(screen.getByText(/not reachable/)).toBeInTheDocument());
    expect(screen.getByText(/presentation_api_server\.py/)).toBeInTheDocument();
  });

  it("shows the Run form once the server is available", async () => {
    global.fetch = vi.fn((url) => {
      if (url.includes("/api/status")) return Promise.resolve({ ok: true, json: () => Promise.resolve({ available: true }) });
      if (url.includes("/api/dataset_info")) return Promise.resolve({ ok: true, json: () => Promise.resolve({ total_files: 9463 }) });
      return Promise.resolve({ ok: false, json: () => Promise.resolve({}) });
    });
    render(<LiveRun />);
    await waitFor(() => expect(screen.getByText("Run")).toBeInTheDocument());
    expect(screen.getByText(/9,463 images available/)).toBeInTheDocument();
  });

  it("running a live batch shows real returned metrics, correctness, and previews", async () => {
    global.fetch = vi.fn((url, opts) => {
      if (url.includes("/api/status")) return Promise.resolve({ ok: true, json: () => Promise.resolve({ available: true }) });
      if (url.includes("/api/dataset_info")) return Promise.resolve({ ok: true, json: () => Promise.resolve({ total_files: 100 }) });
      if (url.includes("/api/run")) return Promise.resolve({ ok: true, json: () => Promise.resolve(FAKE_RUN_RESULT) });
      return Promise.resolve({ ok: false, json: () => Promise.resolve({}) });
    });
    render(<LiveRun />);
    await waitFor(() => expect(screen.getByText("Run")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Run"));
    await waitFor(() => expect(screen.getByText(/Results — 4 images/)).toBeInTheDocument());

    expect(screen.getByText("1.80×")).toBeInTheDocument(); // Basic vs CPU speedup
    expect(screen.getByText("9.00×")).toBeInTheDocument(); // Enhanced vs CPU speedup
    expect(screen.getAllByAltText(/output/).length).toBe(2); // Basic + Enhanced previews (CPU has none)
    expect(screen.getByText(/0.0130% differ/)).toBeInTheDocument();

    // Real thread/launch-configuration metrics for THIS run, not the static defaults.
    expect(screen.getByText("14 × 14 × 4")).toBeInTheDocument();
    expect(screen.getByText("200,704")).toBeInTheDocument();
  });

  it("shows a clear error message when a run fails, never fabricates results", async () => {
    global.fetch = vi.fn((url) => {
      if (url.includes("/api/status")) return Promise.resolve({ ok: true, json: () => Promise.resolve({ available: true }) });
      if (url.includes("/api/dataset_info")) return Promise.resolve({ ok: true, json: () => Promise.resolve({ total_files: 100 }) });
      if (url.includes("/api/run")) return Promise.resolve({ ok: false, status: 400, json: () => Promise.resolve({ error: "No usable CUDA device found." }) });
      return Promise.resolve({ ok: false, json: () => Promise.resolve({}) });
    });
    render(<LiveRun />);
    await waitFor(() => expect(screen.getByText("Run")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Run"));
    await waitFor(() => expect(screen.getByText(/No usable CUDA device found/)).toBeInTheDocument());
  });
});
