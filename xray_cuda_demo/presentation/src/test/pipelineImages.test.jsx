import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { ModeProvider } from "../hooks/useModeContext.jsx";
import { Pipeline } from "../slides/03_Pipeline.jsx";

function withMode(ui) {
  return <ModeProvider>{ui}</ModeProvider>;
}

const FAKE_STAGES = {
  image_index: 0,
  implementation: "Enhanced CUDA",
  stages: {
    original: "data:image/png;base64,ORIG",
    gaussian: "data:image/png;base64,GAUSS",
    median: "data:image/png;base64,MED",
    sobel: "data:image/png;base64,SOBEL",
    laplacian: "data:image/png;base64,LAPL",
    threshold: "data:image/png;base64,THRESH",
  },
};

function mockLiveServer({ totalFiles = 100 } = {}) {
  global.fetch = vi.fn((url) => {
    if (url.includes("/api/status")) return Promise.resolve({ ok: true, json: () => Promise.resolve({ available: true }) });
    if (url.includes("/api/dataset_info")) return Promise.resolve({ ok: true, json: () => Promise.resolve({ total_files: totalFiles }) });
    if (url.includes("/api/preview_stages")) {
      const m = url.match(/image_index=(\d+)/);
      const idx = m ? Number(m[1]) : 0;
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ ...FAKE_STAGES, image_index: idx }) });
    }
    return Promise.resolve({ ok: false, json: () => Promise.resolve({}) });
  });
}

describe("Pipeline slide — real filtered image display", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("shows the illustration fallback when the live server is unavailable", async () => {
    global.fetch = vi.fn(() => Promise.reject(new Error("down")));
    render(withMode(<Pipeline />));
    await waitFor(() => expect(screen.getByText(/live server/)).toBeInTheDocument());
    expect(screen.queryByAltText("Before")).not.toBeInTheDocument();
  });

  it("fetches and displays real before/after images once the server is available", async () => {
    global.fetch = vi.fn((url) => {
      if (url.includes("/api/status")) return Promise.resolve({ ok: true, json: () => Promise.resolve({ available: true }) });
      if (url.includes("/api/preview_stages")) return Promise.resolve({ ok: true, json: () => Promise.resolve(FAKE_STAGES) });
      return Promise.resolve({ ok: false, json: () => Promise.resolve({}) });
    });
    render(withMode(<Pipeline />));
    await waitFor(() => expect(screen.getByAltText("Before")).toBeInTheDocument());
    expect(screen.getByAltText("Before").src).toContain("ORIG"); // Gaussian's "before" is the original
    expect(screen.getByAltText("After").src).toContain("GAUSS");

    fireEvent.click(screen.getByRole("button", { name: "Median" }));
    expect(screen.getByAltText("Before").src).toContain("GAUSS"); // Median's "before" is Gaussian's output
    expect(screen.getByAltText("After").src).toContain("MED");
  });

  it("steps to the next/previous real image using Prev/Next, refetching real stage data", async () => {
    mockLiveServer({ totalFiles: 5 });
    render(withMode(<Pipeline />));
    await waitFor(() => expect(screen.getByText("Image 0 of 4")).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: "Next image" }));
    await waitFor(() => expect(screen.getByText("Image 1 of 4")).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: "Previous image" }));
    await waitFor(() => expect(screen.getByText("Image 0 of 4")).toBeInTheDocument());
    expect(screen.getByRole("button", { name: "Previous image" })).toBeDisabled();
  });

  it("opens a lightbox with the full-size image when the comparison image is clicked", async () => {
    mockLiveServer();
    render(withMode(<Pipeline />));
    await waitFor(() => expect(screen.getByAltText("After")).toBeInTheDocument());

    fireEvent.click(screen.getByAltText("After"));
    expect(screen.getByRole("dialog")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("dialog"));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("falls back to the illustration and shows the error when the fetch fails", async () => {
    global.fetch = vi.fn((url) => {
      if (url.includes("/api/status")) return Promise.resolve({ ok: true, json: () => Promise.resolve({ available: true }) });
      if (url.includes("/api/preview_stages")) return Promise.resolve({ ok: false, status: 503, json: () => Promise.resolve({ error: "No usable CUDA device found." }) });
      return Promise.resolve({ ok: false, json: () => Promise.resolve({}) });
    });
    render(withMode(<Pipeline />));
    await waitFor(() => expect(screen.getByText(/No usable CUDA device found/)).toBeInTheDocument());
    expect(screen.queryByAltText("Before")).not.toBeInTheDocument();
  });
});
