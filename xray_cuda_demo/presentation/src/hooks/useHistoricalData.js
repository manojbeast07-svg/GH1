import { useEffect, useState } from "react";
import { loadHistoricalData } from "../data/loadHistoricalData.js";

// Loads the static HISTORICAL export ONCE on mount -- this is pre-recorded
// data (see loadHistoricalData.js) and is never re-fetched by user actions,
// unlike useLiveRun's state which only ever changes via an explicit run.
export function useHistoricalData() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    loadHistoricalData().then((result) => {
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
