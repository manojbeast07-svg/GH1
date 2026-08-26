import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import App from "../App.jsx";
import { FAKE_DATA, mockFetchWith } from "./testData.js";

describe("App integration (single-scroll layout)", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    mockFetchWith(FAKE_DATA);
  });

  it("loads data, then renders every section on one scrollable page", async () => {
    render(<App />);
    expect(screen.getByText(/Loading presentation data/)).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("CUDA X-RAY PROCESSING LAB")).toBeInTheDocument());

    // Single-scroll: every section is present in the DOM at once, not swapped
    // in/out like the old paginated model.
    expect(document.getElementById("opening")).toBeInTheDocument();
    expect(document.getElementById("quiz")).toBeInTheDocument();
    expect(document.getElementById("summary")).toBeInTheDocument();
    expect(screen.getByText("Quick Quiz")).toBeInTheDocument();
    expect(screen.getAllByText("Summary").length).toBeGreaterThan(0); // sidebar link + slide heading
  });

  it("sidebar lists a jump-link for every section", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText("CUDA X-RAY PROCESSING LAB")).toBeInTheDocument());
    const nav = screen.getByRole("navigation", { name: "Section navigation" });
    expect(nav.textContent).toContain("Opening");
    expect(nav.textContent).toContain("Threading & Parallelism");
    expect(nav.textContent).toContain("Quiz");
  });

  it("clicking a sidebar link scrolls that section into view", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText("CUDA X-RAY PROCESSING LAB")).toBeInTheDocument());
    const target = document.getElementById("quiz");
    const spy = vi.spyOn(target, "scrollIntoView");
    fireEvent.click(screen.getByRole("navigation", { name: "Section navigation" }).querySelector('button[aria-current="false"]'));
    // (clicks whichever non-active link is first; assert the mechanism works generically instead)
    fireEvent.click(screen.getAllByText("Quiz")[0]);
    expect(spy).toHaveBeenCalled();
  });

  it("5. keyboard navigation: ArrowRight/ArrowDown/Space call scrollIntoView on the next section", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText("CUDA X-RAY PROCESSING LAB")).toBeInTheDocument());
    // IntersectionObserver never fires in jsdom (see setupTests.js), so activeId is only ever set by
    // useScrollSpy's corrective effect defaulting it to the first section -- wait for that to land
    // (visible as the "Opening" sidebar link becoming active) before exercising keyboard nav.
    await waitFor(() => expect(screen.getByText("Opening").closest("button").className).toContain("active"));

    const next = document.getElementById("problem"); // activeId is "opening" (index 0) -> goNext targets index 1.
    const spy = vi.spyOn(next, "scrollIntoView");
    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(spy).toHaveBeenCalled();
  });

  it("Home/End jump to the first/last section", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText("CUDA X-RAY PROCESSING LAB")).toBeInTheDocument());
    const first = document.getElementById("opening");
    const last = document.getElementById("summary");
    const spyFirst = vi.spyOn(first, "scrollIntoView");
    const spyLast = vi.spyOn(last, "scrollIntoView");
    fireEvent.keyDown(window, { key: "End" });
    expect(spyLast).toHaveBeenCalled();
    fireEvent.keyDown(window, { key: "Home" });
    expect(spyFirst).toHaveBeenCalled();
  });

  it("Focus Mode hides the sidebar and shows a 'Show Sidebar' control", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText("CUDA X-RAY PROCESSING LAB")).toBeInTheDocument());
    expect(screen.getByRole("navigation", { name: "Section navigation" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Toggle focus mode" }));
    await waitFor(() => expect(screen.queryByRole("navigation", { name: "Section navigation" })).not.toBeInTheDocument());
    expect(screen.getByRole("button", { name: "Toggle focus mode" }).textContent).toBe("Show Sidebar");
  });

  it("does not crash when every data file is unavailable, and still renders all sections", async () => {
    global.fetch = vi.fn(() => Promise.resolve({ ok: false }));
    render(<App />);
    await waitFor(() => expect(screen.getByText("CUDA X-RAY PROCESSING LAB")).toBeInTheDocument());
    expect(document.getElementById("summary")).toBeInTheDocument();
    expect(screen.getAllByText("Not available").length).toBeGreaterThan(0);
  });
});
