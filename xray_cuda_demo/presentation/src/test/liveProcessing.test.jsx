import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { renderHook, act } from "@testing-library/react";
import { LiveProcessing } from "../pages/LiveProcessing.jsx";
import { useLiveRun } from "../hooks/useLiveRun.js";
import { fakeLiveRunResult } from "./testData.js";

function mockServer({ statusAvailable = true, cudaAvailable = true, runResponses = [], runHandler } = {}) {
  let call = 0;
  const requestBodies = [];
  global.fetch = vi.fn((url, opts) => {
    if (url.includes("/api/status")) return Promise.resolve({ ok: true, json: () => Promise.resolve({ available: statusAvailable, cuda_available: cudaAvailable }) });
    if (url.includes("/api/dataset_info")) return Promise.resolve({ ok: true, json: () => Promise.resolve({ total_files: 9463 }) });
    if (url.includes("/api/system_info")) return Promise.resolve({ ok: true, json: () => Promise.resolve({ fingerprint: { gpu_name: "Test GPU" } }) });
    if (url.includes("/api/run")) {
      const body = JSON.parse(opts.body);
      requestBodies.push(body);
      if (runHandler) return runHandler(body, call++);
      const response = runResponses[call] ?? runResponses[runResponses.length - 1];
      call += 1;
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ ...response, _requestBody: body }) });
    }
    return Promise.resolve({ ok: false, json: () => Promise.resolve({}) });
  });
  return { requestBodies };
}

// Renders LiveProcessing wired to a real useLiveRun() hook instance, the
// same way App.jsx does, instead of a disconnected mock -- so this
// exercises the actual state flow a user experiences.
function Harness() {
  const live = useLiveRun();
  return <LiveProcessing live={live} />;
}

describe("Live Processing page", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("shows a clear message and no controls when the live server is unreachable", async () => {
    mockServer({ statusAvailable: false });
    render(<Harness />);
    await waitFor(() => expect(screen.getByText(/not reachable/)).toBeInTheDocument());
    expect(screen.queryByText("Compare All")).not.toBeInTheDocument();
  });

  it("does not run anything until an explicit action button is clicked", async () => {
    mockServer({ runResponses: [fakeLiveRunResult()] });
    render(<Harness />);
    await waitFor(() => expect(screen.getByText("Compare All")).toBeInTheDocument());
    expect(screen.getByText("No run yet — pick an action above.")).toBeInTheDocument();
    expect(global.fetch.mock.calls.some(([u]) => u.includes("/api/run"))).toBe(false);
  });

  it("Compare All sends run_cpu/run_basic/run_enhanced all true; selecting CPU and running sends only run_cpu true", async () => {
    mockServer({ runResponses: [fakeLiveRunResult()] });
    render(<Harness />);
    await waitFor(() => expect(screen.getByText("Compare All")).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: /^CPU \/ OpenCV/ }));
    fireEvent.click(screen.getByText("Run CPU / OpenCV"));
    await waitFor(() => expect(screen.getByText("LIVE")).toBeInTheDocument());
    const runCall = global.fetch.mock.calls.find(([u]) => u.includes("/api/run"));
    const body = JSON.parse(runCall[1].body);
    expect(body).toMatchObject({ run_cpu: true, run_basic: false, run_enhanced: false });
  });

  it("GPU-FIRST: defaults to Enhanced CUDA the moment CUDA is confirmed available", async () => {
    mockServer({ cudaAvailable: true, runResponses: [fakeLiveRunResult()] });
    render(<Harness />);
    await waitFor(() => expect(screen.getByText("Run Enhanced CUDA / C++")).toBeInTheDocument());
    expect(screen.getByRole("button", { name: /^Enhanced CUDA \/ C\+\+/ }).className).toContain("active");

    fireEvent.click(screen.getByText("Run Enhanced CUDA / C++"));
    await waitFor(() => expect(screen.getByText("LIVE")).toBeInTheDocument());
    const body = JSON.parse(global.fetch.mock.calls.find(([u]) => u.includes("/api/run"))[1].body);
    expect(body).toMatchObject({ run_cpu: false, run_basic: false, run_enhanced: true });
  });

  it("GPU UNAVAILABLE: defaults to CPU and shows the GPU-unavailable status indicator", async () => {
    mockServer({ cudaAvailable: false });
    render(<Harness />);
    await waitFor(() => expect(screen.getByText("GPU UNAVAILABLE")).toBeInTheDocument());
    expect(screen.getByRole("button", { name: /^CPU \/ OpenCV/ }).className).toContain("active");
  });

  it("GPU UNAVAILABLE + user still selects Enhanced: falls back to CPU automatically with a clear reason, never silently", async () => {
    mockServer({ cudaAvailable: false, runResponses: [fakeLiveRunResult()] });
    render(<Harness />);
    await waitFor(() => expect(screen.getByText("GPU UNAVAILABLE")).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: /^Enhanced CUDA \/ C\+\+/ }));
    fireEvent.click(screen.getByText(/Run Enhanced CUDA/));
    await waitFor(() => expect(screen.getByText(/GPU unavailable — CPU\/OpenCV fallback active/)).toBeInTheDocument());
    const body = JSON.parse(global.fetch.mock.calls.find(([u]) => u.includes("/api/run"))[1].body);
    expect(body).toMatchObject({ run_cpu: true, run_basic: false, run_enhanced: false });
  });

  it("GPU FAILURE with fallback OFF (default): shows an explicit error and never silently runs CPU", async () => {
    mockServer({
      cudaAvailable: true,
      runHandler: () => Promise.resolve({ ok: false, status: 500, json: () => Promise.resolve({ error: "CUDA out of memory" }) }),
    });
    render(<Harness />);
    await waitFor(() => expect(screen.getByText("Run Enhanced CUDA / C++")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Run Enhanced CUDA / C++"));

    await waitFor(() => expect(screen.getByText(/GPU execution failed: CUDA out of memory/)).toBeInTheDocument());
    expect(global.fetch.mock.calls.filter(([u]) => u.includes("/api/run"))).toHaveLength(1); // no automatic retry
  });

  it("GPU FAILURE with fallback explicitly enabled: automatically retries with CPU, clearly labeled FALLBACK", async () => {
    const { requestBodies } = mockServer({
      cudaAvailable: true,
      runHandler: (body, call) => {
        if (call === 0) return Promise.resolve({ ok: false, status: 500, json: () => Promise.resolve({ error: "CUDA out of memory" }) });
        return Promise.resolve({ ok: true, json: () => Promise.resolve(fakeLiveRunResult()) });
      },
    });
    render(<Harness />);
    await waitFor(() => expect(screen.getByText("Run Enhanced CUDA / C++")).toBeInTheDocument());
    fireEvent.click(screen.getByLabelText(/Allow CPU fallback on GPU failure/));
    fireEvent.click(screen.getByText("Run Enhanced CUDA / C++"));

    await waitFor(() => expect(screen.getByText("FALLBACK")).toBeInTheDocument());
    expect(screen.getByText(/falling back to CPU\/OpenCV because fallback is enabled/)).toBeInTheDocument();
    expect(requestBodies).toHaveLength(2);
    expect(requestBodies[0]).toMatchObject({ run_enhanced: true, run_cpu: false });
    expect(requestBodies[1]).toMatchObject({ run_cpu: true, run_basic: false, run_enhanced: false });
  });

  it("never mislabels an implementation's backend: CPU rows say Python + OpenCV, GPU rows say Native C++/CUDA", async () => {
    mockServer({ runResponses: [fakeLiveRunResult()] });
    render(<Harness />);
    await waitFor(() => expect(screen.getByText("Compare All")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Compare All"));
    await waitFor(() => expect(screen.getByText("LIVE")).toBeInTheDocument());

    const rows = screen.getAllByRole("row");
    const cpuRow = rows.find((r) => r.textContent.includes("CPU") && !r.textContent.includes("Basic") && !r.textContent.includes("Enhanced"));
    const enhancedRow = rows.find((r) => r.textContent.includes("Enhanced CUDA"));
    expect(cpuRow.textContent).toContain("Python + OpenCV");
    expect(enhancedRow.textContent).toContain("Native C++/CUDA");
  });

  it("displays LIVE provenance, same-input verification, and the narrative generated from the result", async () => {
    mockServer({ runResponses: [fakeLiveRunResult()] });
    render(<Harness />);
    await waitFor(() => expect(screen.getByText("Compare All")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Compare All"));

    await waitFor(() => expect(screen.getByText("LIVE")).toBeInTheDocument());
    expect(screen.getAllByText("PASS").length).toBeGreaterThanOrEqual(2); // Same Input + Same Configuration
    expect(screen.getByText(/Enhanced CUDA took 1.000 ms/)).toBeInTheDocument();
  });

  it("FRESH-METRIC REGRESSION: a second run with different results fully replaces the first — no stale numbers, run_id changes", async () => {
    const runA = fakeLiveRunResult({ run_id: "RUN_A" });
    const runB = fakeLiveRunResult({
      run_id: "RUN_B",
      implementations: {
        ...runA.implementations,
        "Enhanced CUDA": { ...runA.implementations["Enhanced CUDA"], total_ms: 42.5 },
      },
      speedups_vs_cpu: { ...runA.speedups_vs_cpu, "Enhanced CUDA": 3.33 },
      pipeline_correctness: [
        { comparison: "basic_vs_cpu", max_abs_diff: 9, mean_abs_diff: 0.9, rmse: 9.9, differing_pixel_count: 999, differing_pixel_percentage: 9.9999 },
      ],
    });
    mockServer({ runResponses: [runA, runB] });
    render(<Harness />);
    await waitFor(() => expect(screen.getByText("Compare All")).toBeInTheDocument());

    fireEvent.click(screen.getByText("Compare All"));
    await waitFor(() => expect(screen.getByText(/run RUN_A/)).toBeInTheDocument());
    expect(screen.getByText("9.00×")).toBeInTheDocument(); // Enhanced vs CPU speedup from run A
    expect(screen.getByText("0.0130% differ")).toBeInTheDocument(); // basic_vs_cpu pipeline correctness from run A

    fireEvent.click(screen.getByText("Compare All"));
    await waitFor(() => expect(screen.getByText(/run RUN_B/)).toBeInTheDocument());

    // Run A's specific numbers must be GONE, not just supplemented -- a
    // fresh run replaces the whole LiveRunResult, it never merges.
    expect(screen.queryByText(/run RUN_A/)).not.toBeInTheDocument();
    expect(screen.queryByText("9.00×")).not.toBeInTheDocument();
    expect(screen.queryByText("0.0130% differ")).not.toBeInTheDocument();
    expect(screen.getAllByText(/42\.500 ms/).length).toBeGreaterThan(0); // run B's fresh Enhanced CUDA timing
    expect(screen.getByText("3.33×")).toBeInTheDocument(); // run B's fresh speedup
    expect(screen.getByText("9.9999% differ")).toBeInTheDocument(); // run B's fresh correctness
  });

  it("marks the result stale when the selection changes, without discarding it", async () => {
    mockServer({ runResponses: [fakeLiveRunResult()] });
    render(<Harness />);
    await waitFor(() => expect(screen.getByText("Compare All")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Compare All"));
    await waitFor(() => expect(screen.getByText("LIVE")).toBeInTheDocument());

    fireEvent.change(screen.getByLabelText("Image index"), { target: { value: "5" } });
    expect(screen.getByText(/Configuration changed/)).toBeInTheDocument();
    expect(screen.getByText("LIVE")).toBeInTheDocument(); // old result still shown, just flagged
  });

  it("shows a clear error message rather than a fabricated result when a run fails", async () => {
    global.fetch = vi.fn((url) => {
      if (url.includes("/api/status")) return Promise.resolve({ ok: true, json: () => Promise.resolve({ available: true }) });
      if (url.includes("/api/dataset_info")) return Promise.resolve({ ok: true, json: () => Promise.resolve({ total_files: 100 }) });
      if (url.includes("/api/run")) return Promise.resolve({ ok: false, status: 503, json: () => Promise.resolve({ error: "No usable CUDA device found." }) });
      return Promise.resolve({ ok: false, json: () => Promise.resolve({}) });
    });
    render(<Harness />);
    await waitFor(() => expect(screen.getByText("Compare All")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Compare All"));
    await waitFor(() => expect(screen.getByText(/No usable CUDA device found/)).toBeInTheDocument());
  });
});

describe("useLiveRun hook", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("marks the previous result stale (not cleared) when the filter config changes", async () => {
    mockServer({ runResponses: [fakeLiveRunResult()] });
    const { result } = renderHook(() => useLiveRun());
    await act(async () => { await result.current.run(); });
    expect(result.current.result.stale).toBe(false);

    act(() => result.current.setFilterConfig({ median_enabled: false }));
    expect(result.current.result.stale).toBe(true);
    expect(result.current.result.run_id).toBeTruthy(); // still holds the old result, just flagged
  });

  it("a fresh run always overwrites the previous result entirely, never merges", async () => {
    const runA = fakeLiveRunResult({ run_id: "A" });
    const runB = fakeLiveRunResult({ run_id: "B", batch_size: 999 });
    mockServer({ runResponses: [runA, runB] });
    const { result } = renderHook(() => useLiveRun());

    await act(async () => { await result.current.run(); });
    expect(result.current.result.run_id).toBe("A");
    await act(async () => { await result.current.run(); });
    expect(result.current.result.run_id).toBe("B");
    expect(result.current.result.batch_size).toBe(999);
  });
});
