import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { ImagePicker } from "../components/ImagePicker.jsx";

function item(index, path) {
  return {
    index, relative_path: path, filename: path.split("/").pop(),
    width: 224, height: 224, thumbnail: `data:image/png;base64,T${index}`,
  };
}

function mockThumbnails(pages) {
  const calls = [];
  global.fetch = vi.fn((url) => {
    calls.push(url);
    if (!url.includes("/api/image_thumbnails")) {
      return Promise.resolve({ ok: false, json: () => Promise.resolve({}) });
    }
    const params = new URL(url, "http://x").searchParams;
    const key = `${params.get("start") || 0}|${params.get("q") || ""}`;
    const body = pages[key] ?? { total: 0, start: 0, count: 0, query: "", items: [] };
    return Promise.resolve({ ok: true, json: () => Promise.resolve(body) });
  });
  return calls;
}

const FIRST_PAGE = {
  "0|": { total: 9463, start: 0, count: 2, query: "", items: [item(0, "train/fractured/a.jpg"), item(1, "train/fractured/b.jpg")] },
};

describe("ImagePicker", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("shows real dataset thumbnails with their real dataset indices", async () => {
    mockThumbnails(FIRST_PAGE);
    render(<ImagePicker selectedIndex={0} onSelect={() => {}} />);

    await waitFor(() => expect(screen.getByAltText("train/fractured/a.jpg")).toBeInTheDocument());
    expect(screen.getByAltText("train/fractured/b.jpg")).toBeInTheDocument();
    expect(screen.getByText("#1")).toBeInTheDocument();
    expect(screen.getByText(/1–2 of 9,463/)).toBeInTheDocument();
  });

  it("reports the picked image's real index to the caller, not its position on the page", async () => {
    mockThumbnails({
      "0|": { total: 9463, start: 0, count: 2, query: "", items: [item(4480, "train/not fractured/x.jpg"), item(4481, "train/not fractured/y.jpg")] },
    });
    const onSelect = vi.fn();
    render(<ImagePicker selectedIndex={0} onSelect={onSelect} />);

    await waitFor(() => expect(screen.getByAltText("train/not fractured/y.jpg")).toBeInTheDocument());
    fireEvent.click(screen.getByAltText("train/not fractured/y.jpg"));
    expect(onSelect).toHaveBeenCalledWith(4481);
  });

  it("marks the currently selected tile so the choice is visible", async () => {
    mockThumbnails(FIRST_PAGE);
    const { container } = render(<ImagePicker selectedIndex={1} onSelect={() => {}} />);

    await waitFor(() => expect(container.querySelectorAll(".image-picker-tile")).toHaveLength(2));
    const selected = container.querySelectorAll(".image-picker-tile.selected");
    expect(selected).toHaveLength(1);
    expect(selected[0].querySelector("img").alt).toBe("train/fractured/b.jpg");
  });

  it("filtering requests the filter and restarts at its first page", async () => {
    const calls = mockThumbnails({
      ...FIRST_PAGE,
      "24|": { total: 9463, start: 24, count: 1, query: "", items: [item(24, "train/fractured/c.jpg")] },
      "0|not fractured": { total: 4623, start: 0, count: 1, query: "not fractured", items: [item(4480, "train/not fractured/x.jpg")] },
    });
    render(<ImagePicker selectedIndex={0} onSelect={() => {}} />);
    await waitFor(() => expect(screen.getByAltText("train/fractured/a.jpg")).toBeInTheDocument());

    fireEvent.click(screen.getByText("Next ▶"));
    await waitFor(() => expect(calls.some((u) => u.includes("start=24"))).toBe(true));

    fireEvent.change(screen.getByLabelText("Filter images"), { target: { value: "not fractured" } });
    await waitFor(() => expect(screen.getByAltText("train/not fractured/x.jpg")).toBeInTheDocument());
    // The filtered request must start from 0 again, not from the page we had paged to.
    expect(calls.some((u) => u.includes("start=0") && u.includes("not+fractured"))).toBe(true);
    expect(screen.getByText(/1–1 of 4,623/)).toBeInTheDocument();
  });

  it("surfaces a backend error rather than an empty grid with no explanation", async () => {
    global.fetch = vi.fn(() => Promise.resolve({
      ok: false, status: 400, json: () => Promise.resolve({ error: "No supported image files found." }),
    }));
    render(<ImagePicker selectedIndex={0} onSelect={() => {}} />);
    await waitFor(() => expect(screen.getByText(/No supported image files found/)).toBeInTheDocument());
  });
});
