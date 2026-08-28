import { useEffect, useState } from "react";
import { checkApiStatus, fetchDatasetInfo, fetchSystemInfo } from "../data/apiClient.js";
import { MetricCard } from "../components/MetricCard.jsx";
import { ProvenanceBadge } from "../components/ProvenanceBadge.jsx";
import { isMissing } from "../utils/isMissing.js";
import { Why } from "../components/Why.jsx";

function StatusBanner({ status, running, error, hasResult }) {
  let text = "System Ready";
  let color = "var(--pass)";
  if (!status.checked) { text = "Checking…"; color = "var(--muted)"; }
  else if (!status.available) { text = "Backend Unavailable"; color = "var(--error)"; }
  else if (running) { text = "Processing…"; color = "var(--basic)"; }
  else if (error) { text = "Error"; color = "var(--error)"; }
  else if (hasResult) { text = "Completed"; color = "var(--pass)"; }

  return (
    <div className="card" style={{ display: "inline-flex", alignItems: "center", gap: "0.5rem", borderColor: color }}>
      <span style={{ width: 10, height: 10, borderRadius: "50%", background: color, display: "inline-block" }} />
      <strong>{text}</strong>
    </div>
  );
}

export function Dashboard({ live, onNavigate }) {
  const [status, setStatus] = useState({ checked: false, available: false });
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [systemInfo, setSystemInfo] = useState(null);

  useEffect(() => {
    checkApiStatus().then((s) => setStatus({ checked: true, ...s }));
    fetchDatasetInfo().then(setDatasetInfo);
    fetchSystemInfo().then(setSystemInfo).catch(() => setSystemInfo(null));
  }, []);

  const { result, running, error, selection, filterConfig } = live;
  const latestLabel = result ? (result.implementations["Enhanced CUDA"] ? "Enhanced CUDA" : result.implementations["Basic CUDA"] ? "Basic CUDA" : "CPU") : null;
  const latestMs = latestLabel ? result.implementations[latestLabel].total_ms : null;

  return (
    <div className="page">
      <h1>Dashboard</h1>
      <p className="subtitle">A real-time summary of this application's current state — this is a complete interactive lab, not a slideshow.</p>

      <StatusBanner status={status} running={running} error={error} hasResult={!!result} />
      <Why question="Why is GPU faster?">
        A GPU has hundreds of lightweight cores that run the same operation on many pixels at once, while a CPU
        has a handful of powerful cores that mostly work through pixels one at a time. For an independent,
        per-pixel operation like these five filters, that parallelism wins — but see the Live Processing page's
        own measured H2D/compute/D2H breakdown for exactly where the time actually goes on a given run.
      </Why>

      <div className="card-row cols-3" style={{ marginTop: "1.2rem" }}>
        <MetricCard title="Dataset" value={datasetInfo ? datasetInfo.total_files.toLocaleString("en-US") : null} subtitle="images available" />
        <MetricCard title="GPU" value={systemInfo?.fingerprint?.gpu_name ?? (status.checked && !status.available ? null : "…")} />
        <MetricCard title="Backend" value={systemInfo?.gpu_implementation_facts?.native_extension ? "Native C++/CUDA" : "CPU only"} />
      </div>

      <div className="card-row cols-3">
        <MetricCard
          title="Current Selection"
          value={selection.mode === "single" ? `Image #${selection.imageIndex}` : `${selection.batchSize} images (batch)`}
        />
        <MetricCard
          title="Current Configuration"
          value={Object.entries(filterConfig).filter(([k, v]) => k.endsWith("_enabled") && v).length}
          subtitle="filters enabled"
        />
        <div className="card">
          <div className="metric-title">Latest Run</div>
          {result ? (
            <>
              <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                <ProvenanceBadge kind="LIVE" />
                {result.stale && <span style={{ fontSize: "0.7rem", color: "var(--warning)" }}>stale</span>}
              </div>
              <div className="metric-value" style={{ marginTop: "0.3rem" }}>{isMissing(latestMs) ? "N/A" : `${latestMs.toFixed(3)} ms`}</div>
              <div className="metric-subtitle">{latestLabel}</div>
            </>
          ) : (
            <span className="data-unavailable">No run yet</span>
          )}
        </div>
      </div>

      <h2>Where to go next</h2>
      <div className="card-row cols-4">
        {[
          ["live-processing", "Live Processing", "Run CPU/Basic/Enhanced and see fresh metrics"],
          ["filters", "Filters", "Explore each of the five filters"],
          ["threading", "Threading & Parallelism", "GPU launch configuration, live"],
          ["experiments", "Experiments", "Save and compare runs"],
        ].map(([id, title, desc]) => (
          <button key={id} className="card filter-card" style={{ textAlign: "left", cursor: "pointer" }} onClick={() => onNavigate(id)}>
            <div style={{ fontWeight: 700 }}>{title}</div>
            <div style={{ fontSize: "0.8rem", color: "var(--muted)", marginTop: "0.3rem" }}>{desc}</div>
          </button>
        ))}
      </div>
    </div>
  );
}
