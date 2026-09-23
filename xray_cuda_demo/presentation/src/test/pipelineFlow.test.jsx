import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { PipelineFlow, buildPipelineStages } from "../components/PipelineFlow.jsx";
import { fakeLiveRunResult } from "./testData.js";

describe("buildPipelineStages", () => {
  it("orders the stages as the data actually flows: H2D, the five filters, D2H", () => {
    const stages = buildPipelineStages({ h2d_ms: 0.1, d2h_ms: 0.2, per_stage_ms: {} });
    expect(stages.map((s) => s.id)).toEqual(["h2d", "gaussian", "median", "sobel", "laplacian", "threshold", "d2h"]);
  });

  it("carries an unmeasured stage through as missing rather than as zero", () => {
    const stages = buildPipelineStages({ h2d_ms: null, d2h_ms: null, per_stage_ms: { gaussian: 2 } });
    expect(stages.find((s) => s.id === "h2d").ms).toBeNull();
    expect(stages.find((s) => s.id === "median").ms).toBeUndefined();
    expect(stages.find((s) => s.id === "gaussian").ms).toBe(2);
  });
});

describe("PipelineFlow", () => {
  it("renders every stage with its real measured time from the current run", () => {
    render(<PipelineFlow result={fakeLiveRunResult()} />);
    // Defaults to the last implementation in the result (Enhanced CUDA).
    expect(screen.getByText("0.300 ms")).toBeInTheDocument(); // gaussian
    expect(screen.getByText("0.060 ms")).toBeInTheDocument(); // h2d
    expect(screen.getByText("0.140 ms")).toBeInTheDocument(); // d2h
  });

  it("switching implementation shows that implementation's own measured stage times", () => {
    render(<PipelineFlow result={fakeLiveRunResult()} />);
    fireEvent.click(screen.getByRole("button", { name: "Basic CUDA" }));
    expect(screen.getByText("2.000 ms")).toBeInTheDocument(); // Basic's gaussian
    expect(screen.getByText("0.100 ms")).toBeInTheDocument(); // Basic's h2d
  });

  it("shows CPU's absent transfers as n/a, never as a zero-length bar", () => {
    const { container } = render(<PipelineFlow result={fakeLiveRunResult()} />);
    fireEvent.click(screen.getByRole("button", { name: "CPU" }));
    expect(screen.getAllByText("n/a")).toHaveLength(2); // H2D and D2H
    expect(container.querySelectorAll(".pipeline-stage.unmeasured")).toHaveLength(2);
  });

  it("clicking a stage reveals its real share of the run and the per-implementation comparison", () => {
    render(<PipelineFlow result={fakeLiveRunResult()} />);
    fireEvent.click(screen.getByRole("button", { name: /Gaussian/ }));
    // Enhanced: gaussian 0.3 of (0.06+0.3+0.2+0.1+0.15+0.05+0.14) = 1.0 measured ms -> 30.0%
    expect(screen.getByText(/30\.0% of this run's/)).toBeInTheDocument();
    expect(screen.getByText(/CPU: 3\.000 ms/)).toBeInTheDocument();
  });

  it("clicking an unmeasured stage explains why it has no number instead of inventing one", () => {
    render(<PipelineFlow result={fakeLiveRunResult()} />);
    fireEvent.click(screen.getByRole("button", { name: "CPU" }));
    fireEvent.click(screen.getByRole("button", { name: /H2D/ }));
    expect(screen.getByText(/Not measured for/)).toHaveTextContent("this implementation does no host↔device transfer");
  });

  it("describes itself as a post-run breakdown, not a live kernel trace", () => {
    render(<PipelineFlow result={fakeLiveRunResult()} />);
    expect(screen.getByText(/not a live trace/)).toBeInTheDocument();
  });
});
