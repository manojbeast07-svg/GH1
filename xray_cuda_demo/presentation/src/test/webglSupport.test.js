import { describe, it, expect } from "vitest";
import { isWebglAvailable } from "../three/webglSupport.js";

describe("isWebglAvailable", () => {
  it("returns false in this test environment (jsdom has no real WebGL context) -- documents why 3D tests exercise the 2D fallback path", () => {
    expect(isWebglAvailable()).toBe(false);
  });

  it("never throws even if canvas/WebGL APIs are missing", () => {
    expect(() => isWebglAvailable()).not.toThrow();
  });
});
