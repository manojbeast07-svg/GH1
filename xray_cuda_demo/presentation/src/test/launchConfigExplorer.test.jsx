import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { LaunchConfigExplorer } from "../components/LaunchConfigExplorer.jsx";

const FAKE_CONFIG = {
  grid_dimensions: [14, 14, 122],
  threads_per_block: 256,
  warps_per_block: 8,
  total_threads_launched: 6121472,
};

describe("Launch Configuration Explorer (interactive thread metrics)", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("shows a fallback message when the live server is unavailable", async () => {
    global.fetch = vi.fn(() => Promise.reject(new Error("down")));
    render(<LaunchConfigExplorer width={224} height={224} />);
    await waitFor(() => expect(screen.getByText(/needs the live server/)).toBeInTheDocument());
  });

  it("fetches and displays real launch-configuration numbers once the server is available", async () => {
    global.fetch = vi.fn((url) => {
      if (url.includes("/api/status")) return Promise.resolve({ ok: true, json: () => Promise.resolve({ available: true }) });
      if (url.includes("/api/launch_config")) return Promise.resolve({ ok: true, json: () => Promise.resolve(FAKE_CONFIG) });
      return Promise.resolve({ ok: false, json: () => Promise.resolve({}) });
    });
    render(<LaunchConfigExplorer width={224} height={224} />);
    await waitFor(() => expect(screen.getByText("14 × 14 × 122")).toBeInTheDocument(), { timeout: 2000 });
    expect(screen.getByText("256")).toBeInTheDocument();
    expect(screen.getByText("8")).toBeInTheDocument();
    expect(screen.getByText("6,121,472")).toBeInTheDocument();
  });

  it("re-fetches with the new batch size after the slider changes", async () => {
    const calls = [];
    global.fetch = vi.fn((url) => {
      if (url.includes("/api/status")) return Promise.resolve({ ok: true, json: () => Promise.resolve({ available: true }) });
      if (url.includes("/api/launch_config")) {
        calls.push(url);
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ ...FAKE_CONFIG, grid_dimensions: [14, 14, url.includes("batch_size=64") ? 64 : 122] }),
        });
      }
      return Promise.resolve({ ok: false, json: () => Promise.resolve({}) });
    });
    render(<LaunchConfigExplorer width={224} height={224} />);
    await waitFor(() => expect(screen.getByText("14 × 14 × 122")).toBeInTheDocument(), { timeout: 2000 });

    fireEvent.change(screen.getByLabelText("Batch size"), { target: { value: "64" } });
    await waitFor(() => expect(screen.getByText("14 × 14 × 64")).toBeInTheDocument(), { timeout: 2000 });
    expect(calls.some((u) => u.includes("batch_size=64"))).toBe(true);
  });

  it("shows a clear error message rather than fabricated numbers when the request fails", async () => {
    global.fetch = vi.fn((url) => {
      if (url.includes("/api/status")) return Promise.resolve({ ok: true, json: () => Promise.resolve({ available: true }) });
      if (url.includes("/api/launch_config")) return Promise.resolve({ ok: false, status: 503, json: () => Promise.resolve({ error: "No usable CUDA device found." }) });
      return Promise.resolve({ ok: false, json: () => Promise.resolve({}) });
    });
    render(<LaunchConfigExplorer width={224} height={224} />);
    await waitFor(() => expect(screen.getByText(/No usable CUDA device found/)).toBeInTheDocument(), { timeout: 2000 });
  });
});
