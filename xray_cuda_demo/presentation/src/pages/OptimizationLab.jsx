import { useEffect, useState } from "react";
import { ProvenanceBadge } from "../components/ProvenanceBadge.jsx";
import { Pill } from "../components/Pill.jsx";
import {
  fetchOptimizationLabFilters, fetchOptimizationLabVariants, fetchOptimizationLabHistoricalSweep,
  runLiveVariantComparison,
} from "../data/apiClient.js";

function HistoricalSweepTable({ historical }) {
  const rows = historical.rows ?? [];
  return (
    <div>
      <p style={{ color: "var(--muted)", fontSize: "0.82rem" }}>
        {historical.image_count} images, {historical.resolution?.join("×")}, {historical.measurement_runs} measurement runs
        (after {historical.warmup_runs} warmup) — recorded {historical.timestamp_utc?.slice(0, 10)} on {historical.gpu_name}.
      </p>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ color: "var(--muted)", fontSize: "0.78rem", textAlign: "right" }}>
            <th style={{ textAlign: "left" }}>Variant</th><th>Mean kernel</th><th>Speedup vs basic</th><th>Correctness</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.variant} style={{ borderTop: "1px solid var(--border)" }}>
              <td style={{ padding: "0.3rem 0" }}>
                {r.variant}{r.is_production_default ? <span style={{ color: "var(--enhanced)", fontSize: "0.75rem" }}> (production)</span> : ""}
              </td>
              <td style={{ textAlign: "right" }}>{r.kernel_ms?.mean != null ? `${r.kernel_ms.mean.toFixed(3)} ms` : "N/A"}</td>
              <td style={{ textAlign: "right" }}>{r.speedup_vs_basic != null ? `${r.speedup_vs_basic.toFixed(2)}×` : "N/A"}</td>
              <td style={{ textAlign: "right" }}>
                {r.correctness ? <Pill status={r.correctness === "PASS" ? "pass" : "warning"}>{r.correctness}</Pill> : <span className="data-unavailable">N/A</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function OptimizationLab() {
  const [filters, setFilters] = useState([]);
  const [filterName, setFilterName] = useState("");
  const [variants, setVariants] = useState([]);
  const [productionDefault, setProductionDefault] = useState(null);
  const [variantA, setVariantA] = useState("basic");
  const [variantB, setVariantB] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [historical, setHistorical] = useState(null);

  useEffect(() => {
    fetchOptimizationLabFilters().then((d) => {
      setFilters(d.filters);
      if (d.filters.length) setFilterName(d.filters[0]);
    });
  }, []);

  useEffect(() => {
    if (!filterName) return;
    setResult(null);
    fetchOptimizationLabVariants(filterName).then((d) => {
      setVariants(d.variants);
      setProductionDefault(d.production_default);
      setVariantB(d.production_default);
    });
    fetchOptimizationLabHistoricalSweep(filterName).then(setHistorical);
  }, [filterName]);

  async function runExperiment() {
    setRunning(true);
    setError(null);
    try {
      const data = await runLiveVariantComparison({ filterName, variantA, variantB, batchSize: 16 });
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="page">
      <h1>Optimization Lab</h1>
      <p className="subtitle">Select a filter and two kernel variants, then run a fresh, live A/B comparison.</p>

      <div className="card-row cols-3">
        <div>
          <label style={{ display: "block", fontSize: "0.8rem", color: "var(--muted)" }}>Filter</label>
          <select value={filterName} onChange={(e) => setFilterName(e.target.value)} style={{ width: "100%", padding: "0.4rem" }}>
            {filters.map((f) => <option key={f} value={f}>{f}</option>)}
          </select>
        </div>
        <div>
          <label style={{ display: "block", fontSize: "0.8rem", color: "var(--muted)" }}>Variant A</label>
          <select value={variantA} onChange={(e) => setVariantA(e.target.value)} style={{ width: "100%", padding: "0.4rem" }}>
            {variants.map((v) => <option key={v} value={v}>{v}{v === productionDefault ? " (production)" : ""}</option>)}
          </select>
        </div>
        <div>
          <label style={{ display: "block", fontSize: "0.8rem", color: "var(--muted)" }}>Variant B</label>
          <select value={variantB} onChange={(e) => setVariantB(e.target.value)} style={{ width: "100%", padding: "0.4rem" }}>
            {variants.map((v) => <option key={v} value={v}>{v}{v === productionDefault ? " (production)" : ""}</option>)}
          </select>
        </div>
      </div>

      <button className="btn-primary" onClick={runExperiment} disabled={running} style={{ marginTop: "0.8rem" }}>
        {running ? "Running…" : "Run Experiment"}
      </button>
      {error && <p style={{ color: "var(--error)", fontSize: "0.82rem" }}>{error}</p>}

      <h2><ProvenanceBadge kind="LIVE" /> Experiment</h2>
      <div className="section-live">
        {result ? (
          <div className="card-row cols-2">
            <div className="card">
              <div style={{ fontWeight: 600 }}>{result.variant_a}{result.is_production_default_a ? " (production)" : ""}</div>
              <div className="metric-value">{result.mean_a_ms.toFixed(3)} ms</div>
            </div>
            <div className="card">
              <div style={{ fontWeight: 600 }}>{result.variant_b}{result.is_production_default_b ? " (production)" : ""}</div>
              <div className="metric-value">{result.mean_b_ms.toFixed(3)} ms</div>
            </div>
            <div className="card" style={{ gridColumn: "1 / -1" }}>
              <div className="metric-title">Speedup (A over B)</div>
              <div className="metric-value" style={{ color: "var(--enhanced)" }}>{result.speedup_a_over_b.toFixed(2)}×</div>
              <p style={{ fontSize: "0.8rem", color: "var(--muted)" }}>
                A vs B correctness: {result.correctness_a_vs_b.differing_pixel_percentage.toFixed(4)}% differ, max abs diff {result.correctness_a_vs_b.max_abs_diff}.
              </p>
            </div>
          </div>
        ) : (
          <p className="data-unavailable">Run an experiment to see fresh results here.</p>
        )}
      </div>

      <h2><ProvenanceBadge kind="HISTORICAL" /> Historical Experiments</h2>
      <div className="section-historical">
        {historical ? (
          <HistoricalSweepTable historical={historical} />
        ) : (
          <p className="data-unavailable">No historical variant sweep recorded for {filterName || "this filter"}.</p>
        )}
      </div>
    </div>
  );
}
