import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { ModeProvider } from "../hooks/useModeContext.jsx";
import { CudaArchitecture } from "../pages/CudaArchitecture.jsx";

describe("CUDA Architecture page", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("shows the live CPU logical core count in the Why GPU visual", async () => {
    global.fetch = vi.fn((url) => {
      if (url.includes("/api/system_info")) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ fingerprint: { cpu_logical_cores: 16 } }),
        });
      }
      return Promise.resolve({ ok: false, json: () => Promise.resolve({}) });
    });
    render(<ModeProvider><CudaArchitecture /></ModeProvider>);
    await waitFor(() => expect(screen.getByText(/16 logical processors on this machine/)).toBeInTheDocument());
  });

  it("switching to Beginner/Technical mode toggles the deeper explanation", async () => {
    global.fetch = vi.fn(() => Promise.resolve({ ok: false, json: () => Promise.resolve({}) }));
    render(<ModeProvider><CudaArchitecture /></ModeProvider>);
    expect(screen.getByText(/An image contains many pixels\./)).toBeInTheDocument();
    fireEvent.click(screen.getByText("Technical"));
    expect(screen.getByText(/Technical mode also covers:/)).toBeInTheDocument();
  });
});
