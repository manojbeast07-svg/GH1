import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { ModeProvider } from "../hooks/useModeContext.jsx";

import { Performance } from "../slides/10_Performance.jsx";
import { Threading } from "../slides/13_Threading.jsx";
import { Pipeline } from "../slides/03_Pipeline.jsx";
import { EnhancedCuda } from "../slides/09_EnhancedCuda.jsx";
import { Correctness } from "../slides/14_Correctness.jsx";
import { WhatFailed } from "../slides/15_WhatFailed.jsx";
import { Quiz } from "../slides/17_Quiz.jsx";
import { FAKE_DATA, EMPTY_DATA } from "./testData.js";

function withMode(ui) {
  return <ModeProvider>{ui}</ModeProvider>;
}

// Expanding Threading's technical detail also mounts LaunchConfigExplorer,
// which checks the live API server (see launchConfigExplorer.test.jsx for
// that component's own dedicated tests) -- mock it unavailable here so
// these rendering-only tests stay deterministic and don't depend on a
// real server.
beforeEach(() => {
  vi.restoreAllMocks();
  global.fetch = vi.fn(() => Promise.resolve({ ok: true, json: () => Promise.resolve({ available: false }) }));
});

describe("2. benchmark rendering", () => {
  it("renders real benchmark numbers on the Performance slide", () => {
    render(withMode(<Performance benchmark={FAKE_DATA.benchmark_summary} />));
    expect(screen.getByText(/90.00 ms/)).toBeInTheDocument(); // CPU end-to-end
    expect(screen.getByText(/1.50×/)).toBeInTheDocument(); // basic_vs_cpu
  });

  it("shows 'Not available' when benchmark data is entirely missing", () => {
    render(withMode(<Performance benchmark={null} />));
    expect(screen.getAllByText("Not available").length).toBeGreaterThan(0);
  });
});

describe("3. threading metrics rendering", () => {
  it("shows CPU/GPU threading numbers after revealing technical detail", async () => {
    render(withMode(<Threading threading={FAKE_DATA.threading} />));
    fireEvent.click(screen.getByText(/Show technical detail/));
    expect(screen.getByText("24")).toBeInTheDocument(); // SM count
    expect(screen.getByText("256")).toBeInTheDocument(); // threads/block
    // LaunchConfigExplorer's own async server check resolves after this render;
    // wait for it so no update happens outside act() after the test body ends.
    await waitFor(() => expect(screen.getByText(/needs the live server/)).toBeInTheDocument());
  });

  it("renders 'Not available' fields as N/A, never a fabricated number", async () => {
    render(withMode(<Threading threading={null} />));
    fireEvent.click(screen.getByText(/Show technical detail/));
    expect(screen.getAllByText("Not available").length).toBeGreaterThan(0);
    await waitFor(() => expect(screen.getByText(/needs the live server/)).toBeInTheDocument());
  });
});

describe("4. filter visualization", () => {
  it("switches the displayed filter explanation when a filter card is clicked", async () => {
    render(withMode(<Pipeline />));
    expect(screen.getByText(/Smooths the image/)).toBeInTheDocument(); // Gaussian, default selection
    fireEvent.click(screen.getByRole("button", { name: "Median" }));
    expect(screen.getByText(/median of nearby pixels/)).toBeInTheDocument();
    // Pipeline's own async server check resolves after this render; wait for it
    // so no update happens outside act() after the test body ends.
    await waitFor(() => expect(screen.getByText(/live server/)).toBeInTheDocument());
  });

  it("Enhanced CUDA slide shows real per-filter speedup for the selected filter", () => {
    render(withMode(<EnhancedCuda perFilter={FAKE_DATA.per_filter_results} />));
    expect(screen.getByText("3.45×")).toBeInTheDocument(); // Gaussian, default selection
    fireEvent.click(screen.getByRole("button", { name: "median" }));
    expect(screen.getByText("5.60×")).toBeInTheDocument();
  });
});

describe("correctness slide", () => {
  it("renders PASS/FAIL pills from real correctness data", () => {
    render(withMode(<Correctness correctness={FAKE_DATA.correctness} />));
    expect(screen.getAllByText("PASS").length).toBe(2);
  });
});

describe("what-failed slide (optimization experiments)", () => {
  it("shows status pills and switches detail on click, using real research data", () => {
    render(withMode(<WhatFailed optimizationResults={FAKE_DATA.optimization_results} />));
    expect(screen.getByText("REJECTED")).toBeInTheDocument();
    expect(screen.getByText("EXPERIMENTAL")).toBeInTheDocument();
    fireEvent.click(screen.getByText("CUDA Graphs — Basic"));
    expect(screen.getAllByText(/CUDA Graphs — Basic/).length).toBeGreaterThan(0);
  });
});

describe("7. quiz", () => {
  it("marks the chosen answer correct/incorrect and reveals an explanation", () => {
    render(withMode(<Quiz />));
    const correctOption = screen.getByText("It can execute many compatible operations in parallel");
    fireEvent.click(correctOption);
    expect(correctOption.className).toContain("correct");
  });

  it("marks a wrong answer incorrect and still shows which one was correct", () => {
    render(withMode(<Quiz />));
    const wrongOption = screen.getByText("It always has a faster CPU");
    fireEvent.click(wrongOption);
    expect(wrongOption.className).toContain("incorrect");
    expect(screen.getByText("It can execute many compatible operations in parallel").className).toContain("correct");
  });
});
