import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { Gpu3D } from "../components/Gpu3D.jsx";
import { ThreadHierarchy3D } from "../components/ThreadHierarchy3D.jsx";

describe("Gpu3D (interactive 3D GPU chip)", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("renders one tile per real SM, read live from the device, never a fixed illustrative count", async () => {
    global.fetch = vi.fn(() => Promise.resolve({ ok: true, json: () => Promise.resolve({ gpu_sm_count: 24 }) }));
    const { container } = render(<Gpu3D />);
    await waitFor(() => expect(screen.getByText(/this GPU has 24 Streaming Multiprocessors/)).toBeInTheDocument());
    expect(container.querySelectorAll(".gpu3d-sm")).toHaveLength(24);
  });

  it("shows an honest unavailable message instead of a fabricated SM count when the query fails", async () => {
    global.fetch = vi.fn(() => Promise.reject(new Error("down")));
    render(<Gpu3D />);
    await waitFor(() => expect(screen.getByText(/Live SM count unavailable/)).toBeInTheDocument());
  });

  it("falls back to the 2D grid (no WebGL in this test environment) without camera controls that would do nothing", async () => {
    global.fetch = vi.fn(() => Promise.resolve({ ok: true, json: () => Promise.resolve({ gpu_sm_count: 8 }) }));
    render(<Gpu3D />);
    await waitFor(() => expect(screen.getByText(/8 Streaming Multiprocessors/)).toBeInTheDocument());
    // No WebGL camera controls should appear when there is no 3D scene to control.
    expect(screen.queryByText("Perspective")).not.toBeInTheDocument();
    expect(screen.queryByText("Auto Rotate")).not.toBeInTheDocument();
  });

  it("clicking an SM tile opens its details, honestly stating per-SM utilization is not measured", async () => {
    global.fetch = vi.fn(() => Promise.resolve({ ok: true, json: () => Promise.resolve({ gpu_sm_count: 4 }) }));
    const { container } = render(<Gpu3D />);
    await waitFor(() => expect(container.querySelectorAll(".gpu3d-sm")).toHaveLength(4));
    fireEvent.click(container.querySelectorAll(".gpu3d-sm")[2]);
    expect(screen.getByText("SM #2")).toBeInTheDocument();
    expect(screen.getByText(/Per-SM utilization: Not measured/)).toBeInTheDocument();
  });
});

describe("ThreadHierarchy3D (interactive containment visualization)", () => {
  it("shows the concept-only explanation with no fabricated number when no threading data is supplied", () => {
    render(<ThreadHierarchy3D threading={null} />);
    expect(screen.getByText(/The complete processor executing CUDA workloads\./)).toBeInTheDocument();
    expect(screen.getByText(/No live value yet/)).toBeInTheDocument();
  });

  it("shows the real per-run number for each level once threading data is supplied, clicking switches levels", () => {
    const threading = { gpu_sm_count: 24, gpu_warp_size: 32, threads_per_block: 256, warps_per_block: 8, total_threads_launched: 1605632 };
    render(<ThreadHierarchy3D threading={threading} />);
    expect(screen.getByText(/24 SMs on this GPU/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Warp" }));
    expect(screen.getByText(/8 warps\/block for this run/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Thread" }));
    expect(screen.getByText(/1,605,632 total logical threads launched this run/)).toBeInTheDocument();
  });
});
