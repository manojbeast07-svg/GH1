import { useEffect, useRef, useState } from "react";
import { fetchLaunchConfig } from "../data/apiClient.js";
import { MetricCard } from "./MetricCard.jsx";
import { isMissing } from "../utils/isMissing.js";

const BLOCK_PRESETS = [
  { label: "8 × 8", x: 8, y: 8 },
  { label: "16 × 16 (production default)", x: 16, y: 16 },
  { label: "32 × 8", x: 32, y: 8 },
  { label: "32 × 16", x: 32, y: 16 },
  { label: "32 × 32", x: 32, y: 32 },
];

function ConfigSlot({ label, config, setConfig, result, error }) {
  return (
    <div className="card">
      <h4 style={{ marginTop: 0 }}>{label}</h4>
      <div className="card-row cols-2">
        <label style={{ display: "block", fontSize: "0.78rem", color: "var(--muted)" }}>
          Width
          <input type="number" min={1} value={config.width} onChange={(e) => setConfig({ ...config, width: Number(e.target.value) })} style={{ width: "100%", padding: "0.4rem", marginTop: "0.1rem" }} />
        </label>
        <label style={{ display: "block", fontSize: "0.78rem", color: "var(--muted)" }}>
          Height
          <input type="number" min={1} value={config.height} onChange={(e) => setConfig({ ...config, height: Number(e.target.value) })} style={{ width: "100%", padding: "0.4rem", marginTop: "0.1rem" }} />
        </label>
      </div>
      <label style={{ display: "block", fontSize: "0.78rem", color: "var(--muted)", marginTop: "var(--space-2)" }}>
        Batch size: <strong style={{ color: "var(--text)" }}>{config.batchSize}</strong>
        <input type="range" min={1} max={512} value={config.batchSize} onChange={(e) => setConfig({ ...config, batchSize: Number(e.target.value) })} style={{ width: "100%", marginTop: "0.2rem" }} />
      </label>
      <label style={{ display: "block", fontSize: "0.78rem", color: "var(--muted)", marginTop: "var(--space-2)" }}>
        Block size
        <select value={config.blockIdx} onChange={(e) => setConfig({ ...config, blockIdx: Number(e.target.value) })} style={{ width: "100%", padding: "0.4rem", marginTop: "0.1rem" }}>
          {BLOCK_PRESETS.map((b, i) => <option key={b.label} value={i}>{b.label}</option>)}
        </select>
      </label>

      {error && <p style={{ color: "var(--error)", fontSize: "0.8rem" }}>{error}</p>}
      {result && (
        <div className="card-row cols-2" style={{ marginTop: "0.6rem" }}>
          <MetricCard title="Grid" value={result.grid_dimensions ? result.grid_dimensions.join(" × ") : null} />
          <MetricCard title="Threads/Block" value={result.threads_per_block ?? null} />
          <MetricCard title="Warps/Block" value={result.warps_per_block ?? null} />
          <MetricCard title="Total Threads" value={isMissing(result.total_threads_launched) ? null : result.total_threads_launched.toLocaleString("en-US")} />
        </div>
      )}
    </div>
  );
}

// Section 25 follow-up: compare two independently-configurable launch
// geometries side by side, live -- every number below comes from a real
// call to /api/launch_config (the exact function every production kernel
// launch already uses), never a duplicated grid/warp formula in JS.
export function ThreadComparison() {
  const [configA, setConfigA] = useState({ width: 224, height: 224, batchSize: 32, blockIdx: 1 });
  const [configB, setConfigB] = useState({ width: 224, height: 224, batchSize: 256, blockIdx: 1 });
  const [resultA, setResultA] = useState(null);
  const [resultB, setResultB] = useState(null);
  const [errorA, setErrorA] = useState(null);
  const [errorB, setErrorB] = useState(null);
  const debounceRef = useRef(null);

  useEffect(() => {
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      const blockA = BLOCK_PRESETS[configA.blockIdx];
      fetchLaunchConfig({ width: configA.width, height: configA.height, batchSize: configA.batchSize, blockX: blockA.x, blockY: blockA.y })
        .then((d) => { setResultA(d); setErrorA(null); })
        .catch((e) => setErrorA(e.message));

      const blockB = BLOCK_PRESETS[configB.blockIdx];
      fetchLaunchConfig({ width: configB.width, height: configB.height, batchSize: configB.batchSize, blockX: blockB.x, blockY: blockB.y })
        .then((d) => { setResultB(d); setErrorB(null); })
        .catch((e) => setErrorB(e.message));
    }, 150);
    return () => clearTimeout(debounceRef.current);
  }, [configA, configB]);

  const bothReady = resultA && resultB && !isMissing(resultA.total_threads_launched) && !isMissing(resultB.total_threads_launched);
  const ratio = bothReady ? resultB.total_threads_launched / resultA.total_threads_launched : null;

  return (
    <div>
      <p style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
        Configure A and B independently — every field recalculates live from this machine's real GPU.
      </p>
      <div className="card-row cols-2">
        <ConfigSlot label="Configuration A" config={configA} setConfig={setConfigA} result={resultA} error={errorA} />
        <ConfigSlot label="Configuration B" config={configB} setConfig={setConfigB} result={resultB} error={errorB} />
      </div>
      {bothReady && (
        <div className="card" style={{ marginTop: "0.6rem", textAlign: "center" }}>
          <div className="metric-title">B launches, relative to A</div>
          <div className="metric-value" style={{ color: "var(--enhanced)" }}>
            {ratio >= 1 ? `${ratio.toFixed(2)}× more threads` : `${(1 / ratio).toFixed(2)}× fewer threads`}
          </div>
          <p style={{ color: "var(--muted)", fontSize: "0.78rem", margin: "0.3rem 0 0" }}>
            {resultA.total_threads_launched.toLocaleString("en-US")} → {resultB.total_threads_launched.toLocaleString("en-US")} logical threads (configured, not measured occupancy)
          </p>
        </div>
      )}
    </div>
  );
}
