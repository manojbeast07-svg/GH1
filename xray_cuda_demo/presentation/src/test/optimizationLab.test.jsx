import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { OptimizationLab } from "../pages/OptimizationLab.jsx";

function mockServer() {
  global.fetch = vi.fn((url) => {
    if (url.includes("/api/optimization_lab/filters")) return Promise.resolve({ ok: true, json: () => Promise.resolve({ filters: ["gaussian", "median"] }) });
    if (url.includes("/api/optimization_lab/variants")) return Promise.resolve({ ok: true, json: () => Promise.resolve({ filter_name: "gaussian", variants: ["basic", "specialized"], production_default: "specialized" }) });
    if (url.includes("/api/optimization_lab/historical_sweep")) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          filter: "gaussian", image_count: 125, resolution: [224, 224], measurement_runs: 20, warmup_runs: 5,
          gpu_name: "Test GPU", timestamp_utc: "2026-01-01T00:00:00Z",
          rows: [
            { variant: "basic", kernel_ms: { mean: 1.259 }, speedup_vs_basic: 1.0, is_production_default: false, correctness: null },
            { variant: "specialized", kernel_ms: { mean: 0.371 }, speedup_vs_basic: 3.39, is_production_default: true, correctness: "PASS" },
          ],
        }),
      });
    }
    if (url.includes("/api/live/variant_comparison")) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          run_id: "R1", variant_a: "basic", variant_b: "specialized", mean_a_ms: 1.25, mean_b_ms: 0.37,
          is_production_default_a: false, is_production_default_b: true, speedup_a_over_b: 3.38,
          correctness_a_vs_b: { differing_pixel_percentage: 0, max_abs_diff: 0 },
        }),
      });
    }
    return Promise.resolve({ ok: false, json: () => Promise.resolve({}) });
  });
}

describe("Optimization Lab page", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("shows the historical sweep as a formatted table, not raw JSON", async () => {
    mockServer();
    render(<OptimizationLab />);
    await waitFor(() => expect(screen.getByText("3.39×")).toBeInTheDocument());
    expect(screen.getAllByText(/\(production\)/).length).toBeGreaterThan(0);
    expect(screen.getByText("PASS")).toBeInTheDocument();
    expect(screen.queryByText(/"rows":/)).not.toBeInTheDocument(); // no raw JSON leaking through
  });

  it("running a live experiment shows fresh LIVE results", async () => {
    mockServer();
    render(<OptimizationLab />);
    await waitFor(() => expect(screen.getByText("Run Experiment")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Run Experiment"));
    await waitFor(() => expect(screen.getByText("3.38×")).toBeInTheDocument());
  });
});
