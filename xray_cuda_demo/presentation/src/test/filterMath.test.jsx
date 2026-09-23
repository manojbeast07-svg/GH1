import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { FilterMath, explainFilterMath } from "../components/FilterMath.jsx";

const GAUSSIAN_RESPONSE = {
  image_index: 0, stage: "gaussian", input_stage: "original",
  x: 112, y: 112, width: 224, height: 224, kernel_size: 3,
  neighborhood: [[10, 10, 10], [10, 100, 10], [10, 10, 10]],
  input_center: 100, output_value: 21,
  coefficients: {
    kind: "weights",
    values: [[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]],
    source: "cuda.gaussian.gaussian_kernel_2d(3, 0.0)",
  },
  implementation: "Enhanced CUDA",
};

function mockFilterMath(response = GAUSSIAN_RESPONSE) {
  const calls = [];
  global.fetch = vi.fn((url) => {
    calls.push(url);
    if (url.includes("/api/filter_math")) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(response) });
    }
    return Promise.resolve({ ok: false, json: () => Promise.resolve({}) });
  });
  return calls;
}

describe("explainFilterMath — the arithmetic shown to the user", () => {
  it("weighted kernels: multiplies each real value by its real coefficient and sums", () => {
    const result = explainFilterMath({
      neighborhood: [[1, 2], [3, 4]],
      coefficients: { kind: "weights", values: [[0.5, 0.5], [0.5, 0.5]] },
    });
    expect(result.kind).toBe("weights");
    expect(result.sum).toBeCloseTo(5, 10); // (1+2+3+4) * 0.5
  });

  it("median: sorts the real neighbourhood and takes the middle value", () => {
    const result = explainFilterMath({
      neighborhood: [[72, 72, 72], [72, 71, 71], [72, 72, 72]],
      coefficients: { kind: "rank" },
    });
    expect(result.sorted).toEqual([71, 71, 72, 72, 72, 72, 72, 72, 72]);
    expect(result.median).toBe(72);
  });

  it("sobel: a vertical edge produces a large Gx and a zero Gy", () => {
    const result = explainFilterMath({
      neighborhood: [[0, 0, 255], [0, 0, 255], [0, 0, 255]],
      coefficients: {
        kind: "sobel",
        gx: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        gy: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        mode: "magnitude",
      },
    });
    expect(result.gx).toBe(1020);
    expect(result.gy).toBe(0);
    expect(result.magnitude).toBeCloseTo(1020, 6);
  });

  it("sobel: a flat neighbourhood produces no gradient at all", () => {
    const result = explainFilterMath({
      neighborhood: [[72, 72, 72], [72, 72, 72], [72, 72, 72]],
      coefficients: {
        kind: "sobel",
        gx: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        gy: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        mode: "magnitude",
      },
    });
    expect(result.magnitude).toBe(0);
  });

  it("threshold: compares the real pixel against the real configured threshold", () => {
    const above = explainFilterMath({
      input_center: 200, neighborhood: [[200]],
      coefficients: { kind: "threshold", threshold_value: 128, max_value: 255 },
    });
    expect(above).toMatchObject({ passes: true, result: 255 });

    const below = explainFilterMath({
      input_center: 40, neighborhood: [[40]],
      coefficients: { kind: "threshold", threshold_value: 128, max_value: 255 },
    });
    expect(below).toMatchObject({ passes: false, result: 0 });
  });

  it("returns null rather than inventing an explanation when there are no coefficients", () => {
    expect(explainFilterMath(null)).toBeNull();
    expect(explainFilterMath({ neighborhood: [[1]] })).toBeNull();
  });
});

describe("FilterMath component", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("shows the real neighbourhood, the real coefficient source, and the GPU's actual output", async () => {
    mockFilterMath();
    render(<FilterMath stage="gaussian" imageIndex={0} imageSrc="data:image/png;base64,AAA" />);

    await waitFor(() => expect(screen.getByText("100")).toBeInTheDocument()); // real centre pixel
    expect(screen.getByText("cuda.gaussian.gaussian_kernel_2d(3, 0.0)")).toBeInTheDocument();
    expect(screen.getByText("What the GPU actually wrote")).toBeInTheDocument();
    expect(screen.getByText("21")).toBeInTheDocument(); // real pipeline output
    expect(screen.getByText(/input stage:/)).toHaveTextContent("original");
  });

  it("clicking the image refetches the math for that exact pixel", async () => {
    const calls = mockFilterMath();
    render(<FilterMath stage="gaussian" imageIndex={0} imageSrc="data:image/png;base64,AAA" />);
    await waitFor(() => expect(screen.getByText("100")).toBeInTheDocument());

    const img = screen.getByAltText(/click to choose a pixel/);
    Object.defineProperty(img, "naturalWidth", { value: 224 });
    Object.defineProperty(img, "naturalHeight", { value: 224 });
    vi.spyOn(img, "getBoundingClientRect").mockReturnValue({ left: 0, top: 0, width: 224, height: 224 });

    fireEvent.click(img, { clientX: 50, clientY: 30 });
    await waitFor(() => expect(calls.some((u) => u.includes("x=50") && u.includes("y=30"))).toBe(true));
  });

  it("median shows the sorted real values with the middle one marked, and no coefficient grid", async () => {
    mockFilterMath({
      ...GAUSSIAN_RESPONSE,
      stage: "median", input_stage: "gaussian", output_value: 72,
      neighborhood: [[72, 72, 72], [72, 71, 71], [72, 72, 72]],
      coefficients: { kind: "rank", source: "no coefficients -- median is a rank filter, it sorts the neighborhood" },
    });
    const { container } = render(<FilterMath stage="median" imageIndex={0} imageSrc="data:image/png;base64,AAA" />);

    await waitFor(() => expect(container.querySelectorAll(".sorted-value")).toHaveLength(9));
    expect(container.querySelector(".sorted-value.median").textContent).toBe("72");
    expect(screen.getByText(/no multiplication at all/)).toBeInTheDocument();
  });

  it("surfaces a backend error instead of rendering fabricated numbers", async () => {
    global.fetch = vi.fn(() => Promise.resolve({
      ok: false, status: 503, json: () => Promise.resolve({ error: "No usable CUDA device found." }),
    }));
    render(<FilterMath stage="gaussian" imageIndex={0} imageSrc="data:image/png;base64,AAA" />);
    await waitFor(() => expect(screen.getByText(/No usable CUDA device found/)).toBeInTheDocument());
    expect(screen.queryByText("What the GPU actually wrote")).not.toBeInTheDocument();
  });
});
