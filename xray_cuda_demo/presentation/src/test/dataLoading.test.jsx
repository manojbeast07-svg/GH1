import { describe, it, expect, vi, beforeEach } from "vitest";
import { loadHistoricalData } from "../data/loadHistoricalData.js";
import { isMissing } from "../utils/isMissing.js";
import { FAKE_DATA, mockFetchWith } from "./testData.js";

describe("historical data loading (static export, never LIVE)", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("loads every historical file from public/data/*.json", async () => {
    mockFetchWith(FAKE_DATA);
    const data = await loadHistoricalData();
    expect(data.benchmark_summary.manifest.selected_image_count).toBe(122);
    expect(data.threading.gpu_sm_count).toBe(24);
    expect(data.optimization_results).toHaveLength(2);
  });

  it("resolves missing files to null, never throws", async () => {
    global.fetch = vi.fn(() => Promise.resolve({ ok: false }));
    const data = await loadHistoricalData();
    expect(data.benchmark_summary).toBeNull();
    expect(data.system).toBeNull();
  });

  it("resolves a network failure to null, never throws", async () => {
    global.fetch = vi.fn(() => Promise.reject(new Error("network down")));
    const data = await loadHistoricalData();
    expect(data.threading).toBeNull();
  });
});

describe("missing data handling", () => {
  it("isMissing() treats null and undefined as missing, real values (including 0) as present", () => {
    expect(isMissing(null)).toBe(true);
    expect(isMissing(undefined)).toBe(true);
    expect(isMissing(0)).toBe(false);
    expect(isMissing(0.0)).toBe(false);
    expect(isMissing("")).toBe(false);
    expect(isMissing(42)).toBe(false);
  });
});
