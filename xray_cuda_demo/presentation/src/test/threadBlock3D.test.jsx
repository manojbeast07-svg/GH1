import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ThreadBlock3D } from "../components/ThreadBlock3D.jsx";

describe("ThreadBlock3D (real CUDA indexing, 2D fallback in this environment)", () => {
  it("renders exactly threadsPerBlock cells for the given block dimensions, never a fixed illustrative count", () => {
    const { container } = render(<ThreadBlock3D blockX={16} blockY={16} warpSize={32} />);
    expect(container.querySelectorAll(".threadblock2d-cell")).toHaveLength(256);
    expect(screen.getByText("256")).toBeInTheDocument(); // Threads/block
    expect(screen.getByText("8")).toBeInTheDocument(); // Warps/block = 256/32
  });

  it("computes the real (x, y) thread coordinate and warp index for a clicked cell, using the same linearization CUDA/production kernels use", () => {
    const { container } = render(<ThreadBlock3D blockX={16} blockY={16} warpSize={32} />);
    const cells = container.querySelectorAll(".threadblock2d-cell");
    // Flat index 37 in a 16-wide block -> (x=5, y=2); warp = floor(37/32) = 1.
    fireEvent.click(cells[37]);
    expect(screen.getByText("(5, 2)")).toBeInTheDocument();
    expect(screen.getByText("Warp 1")).toBeInTheDocument();
  });

  it("recomputes threads/warps per block for a different, real block size (e.g. 8x8)", () => {
    render(<ThreadBlock3D blockX={8} blockY={8} warpSize={32} />);
    expect(screen.getByText("Interactive Thread Block — 8×8")).toBeInTheDocument();
    expect(screen.getByText("64")).toBeInTheDocument(); // 8*8 threads
    expect(screen.getByText("2")).toBeInTheDocument(); // ceil(64/32) warps
  });

  it("Show Warps toggles the warp-coloring mode without changing the real thread/warp counts", () => {
    render(<ThreadBlock3D blockX={16} blockY={16} warpSize={32} />);
    const toggle = screen.getByText("Show Warps");
    fireEvent.click(toggle);
    expect(screen.getByText("Hide Warps")).toBeInTheDocument();
    expect(screen.getByText("256")).toBeInTheDocument();
    expect(screen.getByText("8")).toBeInTheDocument();
  });
});
