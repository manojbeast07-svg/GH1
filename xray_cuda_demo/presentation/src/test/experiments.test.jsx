import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { Experiments } from "../pages/Experiments.jsx";
import { fakeLiveRunResult } from "./testData.js";

const EXPERIMENTS = [
  { run_id: "R1", label: "Run 1", mode: "batch", batch_size: 4, resolution: [224, 224], saved_at_utc: "2026-01-01T00:00:00Z" },
  { run_id: "R2", label: "Run 2", mode: "batch", batch_size: 8, resolution: [224, 224], saved_at_utc: "2026-01-02T00:00:00Z" },
];

function mockServer() {
  global.fetch = vi.fn((url) => {
    if (url.includes("/api/experiments/R1")) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ metadata: { run_id: "R1", filter_config: { a: 1 } }, timing: { CPU: { total_ms: 10 }, "Enhanced CUDA": { total_ms: 2 } } }) });
    }
    if (url.includes("/api/experiments/R2")) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ metadata: { run_id: "R2", filter_config: { a: 2 } }, timing: { CPU: { total_ms: 12 }, "Enhanced CUDA": { total_ms: 3 } } }) });
    }
    if (url.includes("/api/experiments")) return Promise.resolve({ ok: true, json: () => Promise.resolve({ experiments: EXPERIMENTS }) });
    if (url.includes("/api/save_experiment")) return Promise.resolve({ ok: true, json: () => Promise.resolve({ run_id: "NEW1", saved_to: "x" }) });
    return Promise.resolve({ ok: false, json: () => Promise.resolve({}) });
  });
}

describe("Experiments page", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("lists saved experiments and saves the current live run", async () => {
    mockServer();
    const live = { result: fakeLiveRunResult() };
    render(<Experiments live={live} />);
    await waitFor(() => expect(screen.getByText(/Run 1/)).toBeInTheDocument());
    fireEvent.click(screen.getByText("💾 Save Current Run"));
    await waitFor(() => expect(screen.getByText(/Saved as NEW1/)).toBeInTheDocument());
  });

  it("comparing two experiments shows a formatted delta table, not raw JSON, and warns on config mismatch", async () => {
    mockServer();
    render(<Experiments live={{ result: null }} />);
    await waitFor(() => expect(screen.getByText(/Run 1/)).toBeInTheDocument());

    const selects = screen.getAllByRole("combobox");
    fireEvent.change(selects[0], { target: { value: "R1" } });
    fireEvent.change(selects[1], { target: { value: "R2" } });

    await waitFor(() => expect(screen.getByText("+1.000 ms")).toBeInTheDocument()); // Enhanced CUDA: 3 - 2
    expect(screen.getByText(/WARNING: configurations differ/)).toBeInTheDocument();
    expect(screen.queryByText(/"total_ms":/)).not.toBeInTheDocument();
  });
});
