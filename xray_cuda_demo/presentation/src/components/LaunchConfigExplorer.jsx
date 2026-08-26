import { useEffect, useRef, useState } from "react";
import { checkApiStatus, fetchLaunchConfig } from "../data/apiClient.js";
import { MetricCard } from "./MetricCard.jsx";
import { isMissing } from "../data/loadPresentationData.js";

const BLOCK_PRESETS = [
  { label: "16 × 16 (production default)", x: 16, y: 16 },
  { label: "8 × 8", x: 8, y: 8 },
  { label: "32 × 8", x: 32, y: 8 },
  { label: "32 × 16", x: 32, y: 16 },
  { label: "32 × 32", x: 32, y: 32 },
];

// Interactive "what launch configuration would THIS batch size / block size
// actually produce?" widget. Every number comes from a live call to
// /api/launch_config, which itself calls the exact same
// pipeline.threading_metrics.get_threading_metrics() function the static
// Threading slide uses -- never a duplicated grid/warp formula in JS.
export function LaunchConfigExplorer({ width = 224, height = 224 }) {
  const [serverAvailable, setServerAvailable] = useState(null);
  const [batchSize, setBatchSize] = useState(122);
  const [blockIdx, setBlockIdx] = useState(0);
  const [config, setConfig] = useState(null);
  const [error, setError] = useState(null);
  const debounceRef = useRef(null);

  useEffect(() => {
    checkApiStatus().then((s) => setServerAvailable(!!s.available));
  }, []);

  useEffect(() => {
    if (!serverAvailable) return;
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      const block = BLOCK_PRESETS[blockIdx];
      fetchLaunchConfig({ width, height, batchSize, blockX: block.x, blockY: block.y })
        .then((data) => {
          setConfig(data);
          setError(null);
        })
        .catch((e) => setError(e.message));
    }, 150);
    return () => clearTimeout(debounceRef.current);
  }, [serverAvailable, batchSize, blockIdx, width, height]);

  if (serverAvailable === null) {
    return <p className="data-unavailable">Checking for the live server…</p>;
  }
  if (!serverAvailable) {
    return (
      <p className="data-unavailable">
        Interactive explorer needs the live server (<code>python scripts/presentation_api_server.py</code>) —
        showing the fixed, pre-recorded numbers above instead.
      </p>
    );
  }

  return (
    <div className="card" style={{ marginTop: "0.8rem" }}>
      <h4 style={{ marginTop: 0 }}>Launch Configuration Explorer — {width}×{height}</h4>
      <p style={{ color: "var(--muted)", fontSize: "0.82rem" }}>
        Change the batch size or block size below — every number recalculates live, from this machine's real
        GPU, via the same function the production kernels' launch geometry is built from.
      </p>

      <label style={{ display: "block", fontSize: "0.8rem", color: "var(--muted)", marginTop: "0.6rem" }}>
        Batch size: <strong style={{ color: "var(--text)" }}>{batchSize}</strong>
      </label>
      <input
        type="range" min={1} max={512} value={batchSize}
        onChange={(e) => setBatchSize(Number(e.target.value))}
        style={{ width: "100%" }}
        aria-label="Batch size"
      />

      <label style={{ display: "block", fontSize: "0.8rem", color: "var(--muted)", marginTop: "0.6rem" }}>
        Block size
      </label>
      <select value={blockIdx} onChange={(e) => setBlockIdx(Number(e.target.value))} style={{ width: "100%", padding: "0.4rem" }}>
        {BLOCK_PRESETS.map((b, i) => (
          <option key={b.label} value={i}>{b.label}</option>
        ))}
      </select>

      {error && <p style={{ color: "var(--error)", fontSize: "0.82rem" }}>{error}</p>}

      {config && (
        <div className="card-row cols-4" style={{ marginTop: "0.8rem" }}>
          <MetricCard title="Grid" value={config.grid_dimensions ? config.grid_dimensions.join(" × ") : null} />
          <MetricCard title="Threads/Block" value={config.threads_per_block ?? null} />
          <MetricCard title="Warps/Block" value={config.warps_per_block ?? null} />
          <MetricCard
            title="Total Threads Launched"
            value={isMissing(config.total_threads_launched) ? null : config.total_threads_launched.toLocaleString("en-US")}
          />
        </div>
      )}
    </div>
  );
}
