// Section 25A spec item 1/54: every 3D visualization needs a real 2D/text
// fallback when WebGL is unavailable -- this is the single check that
// decides which path renders, checked once (WebGL support doesn't change
// during a session).
export function isWebglAvailable() {
  try {
    const canvas = document.createElement("canvas");
    return !!(window.WebGLRenderingContext && (canvas.getContext("webgl") || canvas.getContext("experimental-webgl")));
  } catch {
    return false;
  }
}
