import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import App from "../App.jsx";
import { FAKE_DATA, mockFetchWith } from "./testData.js";

describe("App integration (Section 25 paged interactive application)", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    mockFetchWith(FAKE_DATA);
  });

  it("renders the Dashboard by default, with every page reachable from the sidebar", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByRole("heading", { name: "Dashboard" })).toBeInTheDocument());
    const nav = screen.getByRole("navigation", { name: "Section navigation" });
    for (const title of ["Dashboard", "Live Processing", "Performance", "Filters", "CUDA Architecture", "Threading & Parallelism", "Optimization Lab", "Correctness", "System", "Experiments", "Research / Decisions"]) {
      expect(nav.textContent).toContain(title);
    }
  });

  it("does NOT call the live /api/run endpoint just from loading the app", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByRole("heading", { name: "Dashboard" })).toBeInTheDocument());
    const runCalls = global.fetch.mock.calls.filter(([url]) => url.includes("/api/run"));
    expect(runCalls).toHaveLength(0);
  });

  it("clicking a sidebar link switches the active page", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByRole("heading", { name: "Dashboard" })).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "System" }));
    await waitFor(() => expect(screen.getByRole("heading", { name: "System" })).toBeInTheDocument());
  });

  it("keyboard ArrowRight moves to the next page", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByRole("heading", { name: "Dashboard" })).toBeInTheDocument());
    fireEvent.keyDown(window, { key: "ArrowRight" });
    await waitFor(() => expect(screen.getByRole("heading", { name: "Live Processing" })).toBeInTheDocument());
  });

  it("Home/End jump to the first/last page", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByRole("heading", { name: "Dashboard" })).toBeInTheDocument());
    fireEvent.keyDown(window, { key: "End" });
    await waitFor(() => expect(screen.getByRole("heading", { name: "What Didn't Work?" })).toBeInTheDocument()); // Research page's content heading
    fireEvent.keyDown(window, { key: "Home" });
    await waitFor(() => expect(screen.getByRole("heading", { name: "Dashboard" })).toBeInTheDocument());
  });

  it("Focus Mode hides the sidebar and shows a 'Show Sidebar' control", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByRole("heading", { name: "Dashboard" })).toBeInTheDocument());
    expect(screen.getByRole("navigation", { name: "Section navigation" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Toggle focus mode" }));
    await waitFor(() => expect(screen.queryByRole("navigation", { name: "Section navigation" })).not.toBeInTheDocument());
    expect(screen.getByRole("button", { name: "Toggle focus mode" }).textContent).toBe("Show Sidebar");
  });

  it("does not crash when historical data is entirely unavailable", async () => {
    global.fetch = vi.fn(() => Promise.resolve({ ok: false }));
    render(<App />);
    await waitFor(() => expect(screen.getByRole("heading", { name: "Dashboard" })).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Performance" }));
    await waitFor(() => expect(screen.getAllByText("Historical benchmark unavailable.").length).toBeGreaterThan(0));
  });
});
