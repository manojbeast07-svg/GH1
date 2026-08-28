// Section 25: HISTORICAL data only -- pre-recorded benchmark artifacts,
// exported once by scripts/export_presentation_data.py from real project
// results (benchmark_results/, research/*_decision.md). This is explicitly
// never mixed with LIVE data (src/hooks/useLiveRun.js): every value loaded
// here must be labeled "HISTORICAL" wherever it's rendered. A file that is
// missing, unreadable, or `null` resolves to `null`, never a fabricated
// fallback -- callers render "Not available" for a `null` field.

const DATA_FILES = [
  "benchmark_summary",
  "per_filter_results",
  "batch_sweep",
  "resolution_sweep",
  "correctness",
  "system",
  "threading",
  "optimization_results",
];

async function loadOne(name) {
  try {
    const res = await fetch(`${import.meta.env.BASE_URL}data/${name}.json`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function loadHistoricalData() {
  const entries = await Promise.all(DATA_FILES.map(async (name) => [name, await loadOne(name)]));
  return Object.fromEntries(entries);
}

export { isMissing } from "../utils/isMissing.js";
