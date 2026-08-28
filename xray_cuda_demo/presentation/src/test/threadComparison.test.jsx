import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { ThreadComparison } from "../components/ThreadComparison.jsx";

function mockLaunchConfig(byBatchSize) {
  global.fetch = vi.fn((url) => {
    const m = url.match(/batch_size=(\d+)/);
    const batchSize = m ? Number(m[1]) : null;
    const data = byBatchSize[batchSize] ?? { grid_dimensions: [1, 1, 1], threads_per_block: 256, warps_per_block: 8, total_threads_launched: 256 };
    return Promise.resolve({ ok: true, json: () => Promise.resolve(data) });
  });
}

describe("ThreadComparison", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("fetches and displays real launch configs for both A and B independently", async () => {
    mockLaunchConfig({
      32: { grid_dimensions: [14, 14, 32], threads_per_block: 256, warps_per_block: 8, total_threads_launched: 1605632 },
      256: { grid_dimensions: [14, 14, 256], threads_per_block: 256, warps_per_block: 8, total_threads_launched: 12845056 },
    });
    render(<ThreadComparison />);
    await waitFor(() => expect(screen.getByText("14 × 14 × 32")).toBeInTheDocument(), { timeout: 2000 });
    await waitFor(() => expect(screen.getByText("14 × 14 × 256")).toBeInTheDocument(), { timeout: 2000 });
    expect(screen.getByText("1,605,632")).toBeInTheDocument();
    expect(screen.getByText("12,845,056")).toBeInTheDocument();
  });

  it("computes a real relative-thread-count summary once both sides resolve", async () => {
    mockLaunchConfig({
      32: { grid_dimensions: [14, 14, 32], threads_per_block: 256, warps_per_block: 8, total_threads_launched: 1000000 },
      256: { grid_dimensions: [14, 14, 256], threads_per_block: 256, warps_per_block: 8, total_threads_launched: 8000000 },
    });
    render(<ThreadComparison />);
    await waitFor(() => expect(screen.getByText("8.00× more threads")).toBeInTheDocument(), { timeout: 2000 });
  });

  it("re-fetches configuration A when its batch size slider changes", async () => {
    const calls = [];
    global.fetch = vi.fn((url) => {
      calls.push(url);
      const m = url.match(/batch_size=(\d+)/);
      const total = m ? Number(m[1]) * 1000 : 1000;
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ grid_dimensions: [1, 1, 1], threads_per_block: 256, warps_per_block: 8, total_threads_launched: total }) });
    });
    render(<ThreadComparison />);
    await waitFor(() => expect(calls.some((u) => u.includes("batch_size=32"))).toBe(true), { timeout: 2000 });

    const sliders = screen.getAllByRole("slider");
    fireEvent.change(sliders[0], { target: { value: "64" } });
    await waitFor(() => expect(calls.some((u) => u.includes("batch_size=64"))).toBe(true), { timeout: 2000 });
  });

  it("shows a per-slot error message rather than a fabricated number when a request fails", async () => {
    global.fetch = vi.fn(() => Promise.resolve({ ok: false, status: 503, json: () => Promise.resolve({ error: "No usable CUDA device found." }) }));
    render(<ThreadComparison />);
    await waitFor(() => expect(screen.getAllByText("No usable CUDA device found.").length).toBeGreaterThan(0), { timeout: 2000 });
  });
});
