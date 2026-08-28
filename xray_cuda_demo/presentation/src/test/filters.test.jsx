import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { Filters } from "../pages/Filters.jsx";

const HISTORICAL = [
  { filter: "gaussian", basic_kernel_ms: 1.9, enhanced_kernel_ms: 0.55, kernel_speedup: 3.45 },
  { filter: "median", basic_kernel_ms: 2.5, enhanced_kernel_ms: 0.45, kernel_speedup: 5.6 },
];

function mockServer() {
  global.fetch = vi.fn((url) => {
    if (url.includes("/api/status")) return Promise.resolve({ ok: true, json: () => Promise.resolve({ available: true }) });
    if (url.includes("/api/dataset_info")) return Promise.resolve({ ok: true, json: () => Promise.resolve({ total_files: 100 }) });
    if (url.includes("/api/preview_stages")) {
      const m = url.match(/image_index=(\d+)/);
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          image_index: m ? Number(m[1]) : 0,
          stages: { original: "data:image/png;base64,ORIG", gaussian: "data:image/png;base64,GAUSS", median: "data:image/png;base64,MED", sobel: "data:image/png;base64,SOBEL", laplacian: "data:image/png;base64,LAPL", threshold: "data:image/png;base64,THRESH" },
        }),
      });
    }
    if (url.includes("/api/live/per_filter")) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          run_id: "R1",
          per_filter_results: [
            { filter: "gaussian", basic_kernel_ms: 1.2, enhanced_kernel_ms: 0.3, kernel_speedup: 4.0 },
            { filter: "median", basic_kernel_ms: 1.0, enhanced_kernel_ms: 0.2, kernel_speedup: 5.0 },
          ],
        }),
      });
    }
    return Promise.resolve({ ok: false, json: () => Promise.resolve({}) });
  });
}

describe("Filters page", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("shows historical per-filter data and switches filter selection", async () => {
    mockServer();
    render(<Filters historical={HISTORICAL} />);
    await waitFor(() => expect(screen.getByText("3.45×")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Median" }));
    await waitFor(() => expect(screen.getByText("5.60×")).toBeInTheDocument());
  });

  it("running live timing shows fresh LIVE per-filter results without touching the historical panel", async () => {
    mockServer();
    render(<Filters historical={HISTORICAL} />);
    await waitFor(() => expect(screen.getByText("Run Live Timing (all 5 filters)")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Run Live Timing (all 5 filters)"));
    await waitFor(() => expect(screen.getByText("4.00×")).toBeInTheDocument());
    expect(screen.getByText("3.45×")).toBeInTheDocument(); // historical panel unchanged
  });

  it("clicking a real preview image opens a lightbox with the full-size image", async () => {
    mockServer();
    render(<Filters historical={HISTORICAL} />);
    await waitFor(() => expect(screen.getByAltText("After")).toBeInTheDocument());
    fireEvent.click(screen.getByAltText("After"));
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("dialog"));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("Prev/Next step through real dataset images, re-fetching stages each time", async () => {
    mockServer();
    render(<Filters historical={HISTORICAL} />);
    await waitFor(() => expect(screen.getByText(/Image 0/)).toBeInTheDocument());
    fireEvent.click(screen.getByText("Next ▶"));
    await waitFor(() => expect(screen.getByText(/Image 1/)).toBeInTheDocument());
    expect(global.fetch.mock.calls.some(([u]) => u.includes("image_index=1"))).toBe(true);
  });
});
