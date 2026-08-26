import { useEffect, useState } from "react";
import { loadPresentationData } from "../data/loadPresentationData.js";

// Section 24 spec item 45-46: centralized data hook, so slide components
// never fetch or parse raw JSON themselves.
export function usePresentationData() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    loadPresentationData().then((result) => {
      if (!cancelled) {
        setData(result);
        setLoading(false);
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return { data, loading };
}

// Convenience derived hooks (spec item 45: useBenchmarkData(), useSystemData(), etc.)
// -- thin accessors over the one loaded data object, not independent fetches.
export function useBenchmarkData(data) {
  return data?.benchmark_summary ?? null;
}
export function useSystemData(data) {
  return data?.system ?? null;
}
export function useThreadingData(data) {
  return data?.threading ?? null;
}
export function useOptimizationData(data) {
  return data?.optimization_results ?? null;
}
export function usePerFilterData(data) {
  return data?.per_filter_results ?? null;
}
export function useBatchSweepData(data) {
  return data?.batch_sweep ?? null;
}
export function useResolutionSweepData(data) {
  return data?.resolution_sweep ?? null;
}
export function useCorrectnessData(data) {
  return data?.correctness ?? null;
}
