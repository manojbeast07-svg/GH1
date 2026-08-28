import { useCallback, useState } from "react";
import { runLive } from "../data/apiClient.js";

export const DEFAULT_FILTER_CONFIG = {
  gaussian_enabled: true, gaussian_kernel_size: 5, gaussian_sigma: 0.0,
  median_enabled: true, median_kernel_size: 3,
  sobel_enabled: true, sobel_kernel_size: 3, sobel_mode: "magnitude",
  laplacian_enabled: true, laplacian_kernel_size: 3, laplacian_scale: 1.0, laplacian_delta: 0.0,
  threshold_enabled: true, threshold_value: 128, threshold_max_value: 255,
};

export const DEFAULT_SELECTION = {
  mode: "single", batchSize: 8, seed: 42, imageIndex: 0,
};

// Section 25 spec item 58: ONE centralized LiveRunResult object. Every
// "current run" card across every page reads from this same state -- no
// component holds its own copy of a timing/speedup/correctness number.
//
// Spec item 57 (stale-result invalidation): changing the selection or
// filter config does NOT clear the previous result -- it flags it `stale`
// so the UI can keep showing it (grayed, with a "configuration changed"
// banner) instead of abruptly losing context, while making it impossible
// to mistake the old numbers for the current configuration's numbers.
export function useLiveRun() {
  const [selection, setSelectionState] = useState(DEFAULT_SELECTION);
  const [filterConfig, setFilterConfigState] = useState(DEFAULT_FILTER_CONFIG);
  const [result, setResult] = useState(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);

  const markStale = useCallback(() => {
    setResult((r) => (r && !r.stale ? { ...r, stale: true } : r));
  }, []);

  const setSelection = useCallback((patch) => {
    setSelectionState((s) => ({ ...s, ...patch }));
    markStale();
  }, [markStale]);

  const setFilterConfig = useCallback((patch) => {
    setFilterConfigState((c) => ({ ...c, ...patch }));
    markStale();
  }, [markStale]);

  // Returns { success, data } or { success: false, error } -- callers that
  // need to make a control-flow decision based on the outcome (e.g. "GPU
  // failed, should I now explicitly retry with CPU?") read THIS return
  // value, never the `error`/`result` state, since state updates from
  // setError/setResult above are not yet visible in this closure by the
  // time the caller's `await run(...)` resolves.
  const run = useCallback(async ({ runCpu = true, runBasic = true, runEnhanced = true, fallbackReason } = {}) => {
    setRunning(true);
    setError(null);
    try {
      const data = await runLive({ ...selection, runCpu, runBasic, runEnhanced, filterConfig });
      const finalData = fallbackReason ? { ...data, fallback_reason: fallbackReason } : data;
      setResult({ ...finalData, stale: false });
      return { success: true, data: finalData };
    } catch (e) {
      setError(e.message);
      return { success: false, error: e.message };
    } finally {
      setRunning(false);
    }
  }, [selection, filterConfig]);

  return { selection, setSelection, filterConfig, setFilterConfig, result, running, error, run };
}
