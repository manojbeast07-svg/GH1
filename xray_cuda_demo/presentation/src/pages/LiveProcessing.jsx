import { useEffect, useState } from "react";
import { checkApiStatus, fetchDatasetInfo, fetchSystemInfo, saveExperiment } from "../data/apiClient.js";
import { MetricCard } from "../components/MetricCard.jsx";
import { BarChart } from "../components/BarChart.jsx";
import { Pill } from "../components/Pill.jsx";
import { ProvenanceBadge, SourceTag } from "../components/ProvenanceBadge.jsx";
import { FilterConfigPanel } from "../components/FilterConfigPanel.jsx";
import { Lightbox, Zoomable } from "../components/Lightbox.jsx";
import { isMissing } from "../utils/isMissing.js";
import { buildRunNarrative } from "../utils/narrative.js";

const IMPL_KIND = { CPU: "cpu", "Basic CUDA": "basic", "Enhanced CUDA": "enhanced" };
const BACKEND_LABEL = { CPU: "Python + OpenCV", "Basic CUDA": "Native C++/CUDA", "Enhanced CUDA": "Native C++/CUDA" };
const IMPL_META = {
  cpu: { label: "CPU / OpenCV", flags: { runCpu: true, runBasic: false, runEnhanced: false } },
  basic: { label: "Basic CUDA / C++", flags: { runCpu: false, runBasic: true, runEnhanced: false } },
  enhanced: { label: "Enhanced CUDA / C++", flags: { runCpu: false, runBasic: false, runEnhanced: true } },
};

function fmtMs(v) {
  return isMissing(v) ? "N/A" : `${v.toFixed(3)} ms`;
}

export function LiveProcessing({ live }) {
  const { selection, setSelection, filterConfig, setFilterConfig, result, running, error, run } = live;
  const [serverStatus, setServerStatus] = useState({ checked: false, available: false });
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [systemInfo, setSystemInfo] = useState(null);
  const [diffView, setDiffView] = useState("basic_vs_cpu");
  const [saveMessage, setSaveMessage] = useState(null);
  const [zoomSrc, setZoomSrc] = useState(null);
  const [implementation, setImplementation] = useState(null);
  const [allowCpuFallback, setAllowCpuFallback] = useState(false);
  const [fallbackNotice, setFallbackNotice] = useState(null);

  useEffect(() => {
    checkApiStatus().then((s) => setServerStatus({ checked: true, ...s }));
    fetchDatasetInfo().then(setDatasetInfo);
    fetchSystemInfo().then(setSystemInfo).catch(() => setSystemInfo(null));
  }, []);

  // GPU-first: default to Enhanced CUDA the moment we know CUDA is
  // available; only fall back to CPU as the default selection when CUDA
  // genuinely isn't available on this machine.
  useEffect(() => {
    if (serverStatus.checked && implementation === null) {
      setImplementation(serverStatus.cuda_available ? "enhanced" : "cpu");
    }
  }, [serverStatus, implementation]);

  async function runSelected() {
    setFallbackNotice(null);
    let effectiveImpl = implementation;
    if (effectiveImpl !== "cpu" && !serverStatus.cuda_available) {
      // CUDA is unavailable outright -- this is a known, permanent state,
      // not a failure -- so CPU is used automatically, clearly labeled.
      effectiveImpl = "cpu";
      setFallbackNotice({ kind: "unavailable", message: "GPU unavailable — CPU/OpenCV fallback active." });
    }
    const outcome = await run(IMPL_META[effectiveImpl].flags);
    if (!outcome.success && effectiveImpl !== "cpu") {
      if (allowCpuFallback) {
        setFallbackNotice({ kind: "failure", message: `GPU execution failed (${outcome.error}) — falling back to CPU/OpenCV because fallback is enabled.` });
        await run({ runCpu: true, runBasic: false, runEnhanced: false, fallbackReason: outcome.error });
      } else {
        setFallbackNotice({ kind: "error", message: `GPU execution failed: ${outcome.error}` });
      }
    }
  }

  async function handleSave() {
    if (!result) return;
    setSaveMessage(null);
    try {
      const saved = await saveExperiment(result, `Live Processing — ${result.mode}`);
      setSaveMessage(`Saved as experiment ${saved.run_id}.`);
    } catch (e) {
      setSaveMessage(`Save failed: ${e.message}`);
    }
  }

  if (!serverStatus.checked) {
    return <p className="data-unavailable">Checking for the live processing server…</p>;
  }
  if (!serverStatus.available) {
    return (
      <div className="card" style={{ borderColor: "var(--error)" }}>
        <p style={{ margin: 0 }}>
          <strong>Live processing server not reachable.</strong> Start it from the project root with:
        </p>
        <pre className="diagram" style={{ marginTop: "0.5rem" }}>{"python scripts/presentation_api_server.py"}</pre>
        <p style={{ color: "var(--muted)", fontSize: "0.85rem", margin: 0 }}>
          Every metric on this page requires an actual execution against the live backend — there is no
          pre-recorded fallback here.
        </p>
      </div>
    );
  }

  const narrative = buildRunNarrative(result);

  return (
    <div className="page">
      <h1>Live Processing</h1>
      <p className="subtitle">
        Select images, configure the filters, and run an implementation. Every metric below is calculated from
        that exact execution — nothing here is pre-recorded.
      </p>

      <div className="gpu-status-indicator" data-status={serverStatus.cuda_available ? "available" : "unavailable"}>
        <span className="gpu-status-dot" />
        {serverStatus.cuda_available ? "CUDA AVAILABLE" : "GPU UNAVAILABLE"}
        <span style={{ color: "var(--muted)", fontWeight: 400, marginLeft: "0.6rem" }}>
          Backend: {serverStatus.cuda_available ? "Native C++/CUDA" : "Python + OpenCV only"}
          {systemInfo?.fingerprint?.gpu_name ? ` · GPU: ${systemInfo.fingerprint.gpu_name}` : ""}
        </span>
      </div>

      <h2>1. Dataset &amp; Selection</h2>
      <div className="card">
        <div className="card-row cols-2">
          <div>
            <label style={{ display: "block", fontSize: "0.8rem", color: "var(--muted)" }}>
              Mode
              <select value={selection.mode} onChange={(e) => setSelection({ mode: e.target.value })} style={{ width: "100%", padding: "0.4rem" }}>
                <option value="single">Single image</option>
                <option value="batch">Batch</option>
              </select>
            </label>
          </div>
          {selection.mode === "batch" ? (
            <div>
              <label style={{ display: "block", fontSize: "0.8rem", color: "var(--muted)" }}>
                Batch size
                <input type="number" min={1} max={512} value={selection.batchSize}
                  onChange={(e) => setSelection({ batchSize: Number(e.target.value) })} style={{ width: "100%", padding: "0.4rem" }} />
              </label>
            </div>
          ) : (
            <div>
              <label style={{ display: "block", fontSize: "0.8rem", color: "var(--muted)" }}>
                Image index
                <input type="number" min={0} value={selection.imageIndex}
                  onChange={(e) => setSelection({ imageIndex: Number(e.target.value) })} style={{ width: "100%", padding: "0.4rem" }} />
              </label>
            </div>
          )}
        </div>
        {selection.mode === "batch" && (
          <div style={{ marginTop: "0.6rem" }}>
            <label style={{ display: "block", fontSize: "0.8rem", color: "var(--muted)" }}>
              Random seed
              <input type="number" value={selection.seed} onChange={(e) => setSelection({ seed: Number(e.target.value) })}
                style={{ width: "100%", padding: "0.4rem" }} />
            </label>
          </div>
        )}
        {datasetInfo && (
          <p style={{ color: "var(--muted)", fontSize: "0.8rem", marginTop: "0.6rem" }}>
            Dataset: {datasetInfo.total_files?.toLocaleString("en-US")} images available.
          </p>
        )}
      </div>

      <h2>2. Filter Configuration</h2>
      <FilterConfigPanel config={filterConfig} onChange={setFilterConfig} />

      <h2>3. Implementation</h2>
      <div className="card-row cols-3">
        {["cpu", "basic", "enhanced"].map((impl) => (
          <button
            key={impl}
            className={`card filter-card impl-card ${impl}${implementation === impl ? " active" : ""}`}
            onClick={() => setImplementation(impl)}
            disabled={running}
          >
            <div style={{ fontWeight: 700 }}>{IMPL_META[impl].label}</div>
            {impl === "enhanced" && <div style={{ marginTop: "0.3rem" }}><ProvenanceBadge kind="PRODUCTION" /></div>}
            {impl === "enhanced" && serverStatus.cuda_available && implementation === "enhanced" && (
              <div style={{ fontSize: "0.72rem", color: "var(--muted)", marginTop: "0.3rem" }}>GPU-first default</div>
            )}
          </button>
        ))}
      </div>

      <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.82rem", color: "var(--muted)", marginTop: "0.8rem" }}>
        <input type="checkbox" checked={allowCpuFallback} onChange={(e) => setAllowCpuFallback(e.target.checked)} />
        Allow CPU fallback on GPU failure (off by default — a GPU failure shows an error, it never silently runs CPU)
      </label>

      <div className="card-row" style={{ gridTemplateColumns: "repeat(2, 1fr)", marginTop: "0.6rem" }}>
        <button className="btn-primary" disabled={running || !implementation} onClick={runSelected}>
          {running ? "Running…" : `Run ${implementation ? IMPL_META[implementation].label : "…"}`}
        </button>
        <button className="btn-action enhanced" disabled={running} onClick={() => run({ runCpu: true, runBasic: true, runEnhanced: true })}>
          Compare All
        </button>
      </div>

      {fallbackNotice && (
        <div className="card" style={{ borderColor: fallbackNotice.kind === "error" ? "var(--error)" : "var(--warning)", marginTop: "1rem" }}>
          <ProvenanceBadge kind={fallbackNotice.kind === "error" ? "ERROR" : "FALLBACK"} />
          <p style={{ margin: "0.4rem 0 0" }}>{fallbackNotice.message}</p>
        </div>
      )}

      {error && !fallbackNotice && (
        <div className="card" style={{ borderColor: "var(--error)", marginTop: "1rem" }}>
          <strong>Run failed:</strong> {error}
        </div>
      )}

      {!result && !error && (
        <p className="data-unavailable" style={{ marginTop: "1rem" }}>No run yet — pick an action above.</p>
      )}

      {result && (
        <div style={{ marginTop: "1.5rem" }}>
          {result.stale && (
            <div className="card" style={{ borderColor: "var(--warning)", marginBottom: "1rem" }}>
              <strong>Configuration changed.</strong> Run again to update results — the numbers below are from
              the previous configuration.
            </div>
          )}

          <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", flexWrap: "wrap" }}>
            <ProvenanceBadge kind="LIVE" />
            <span style={{ color: "var(--muted)", fontSize: "0.8rem" }}>
              run {result.run_id} — {result.batch_size} image{result.batch_size === 1 ? "" : "s"}, {result.resolution[0]}×{result.resolution[1]}
            </span>
            <button className="tech-toggle" onClick={handleSave}>💾 Save Experiment</button>
            {saveMessage && <span style={{ fontSize: "0.8rem", color: "var(--muted)" }}>{saveMessage}</span>}
          </div>

          <div className="card-row cols-2" style={{ marginTop: "0.8rem" }}>
            <div className="card" style={{ display: "flex", justifyContent: "space-between" }}>
              <span>Same Input</span>
              <Pill status={result.same_input_verification?.same_input ? "pass" : "error"}>
                {result.same_input_verification?.same_input ? "PASS" : "FAIL"}
              </Pill>
            </div>
            <div className="card" style={{ display: "flex", justifyContent: "space-between" }}>
              <span>Same Configuration</span>
              <Pill status={result.same_input_verification?.same_configuration ? "pass" : "error"}>
                {result.same_input_verification?.same_configuration ? "PASS" : "FAIL"}
              </Pill>
            </div>
          </div>

          <h3 style={{ marginTop: "1.2rem" }}>Total time <SourceTag source="current live run" /></h3>
          <BarChart
            bars={Object.entries(result.implementations).map(([label, impl]) => ({ label, value: impl.total_ms, kind: IMPL_KIND[label] }))}
            formatValue={fmtMs}
          />

          <div className="card-row cols-3">
            {Object.entries(result.speedups_vs_cpu).map(([label, v]) => (
              <MetricCard
                key={label} title={`${label} vs CPU`} value={isMissing(v) ? null : `${v.toFixed(2)}×`}
                kind={label.includes("Enhanced") ? "enhanced" : label.includes("Basic") ? "basic" : undefined}
              />
            ))}
          </div>

          <h3>H2D / Compute / D2H breakdown <SourceTag source="CUDA event / CPU timer" /></h3>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ color: "var(--muted)", fontSize: "0.78rem", textAlign: "right" }}>
                <th style={{ textAlign: "left" }}>Implementation</th><th style={{ textAlign: "left" }}>Backend</th><th>H2D</th><th>Compute</th><th>D2H</th><th>Total</th><th>Images/s</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(result.implementations).map(([label, impl]) => (
                <tr key={label} style={{ borderTop: "1px solid var(--border)" }}>
                  <td style={{ padding: "0.3rem 0" }}>{label}</td>
                  <td style={{ color: "var(--muted)", fontSize: "0.8rem" }}>{BACKEND_LABEL[label] ?? "Unknown"}</td>
                  <td style={{ textAlign: "right" }}>{fmtMs(impl.h2d_ms)}</td>
                  <td style={{ textAlign: "right" }}>{fmtMs(impl.compute_ms)}</td>
                  <td style={{ textAlign: "right" }}>{fmtMs(impl.d2h_ms)}</td>
                  <td style={{ textAlign: "right", fontWeight: 600 }}>{fmtMs(impl.total_ms)}</td>
                  <td style={{ textAlign: "right" }}>{isMissing(impl.images_per_second) ? "N/A" : impl.images_per_second.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p style={{ color: "var(--muted)", fontSize: "0.82rem" }}>CPU has no H2D/D2H (no GPU transfer involved), so those cells show N/A rather than zero.</p>

          <h3>Per-filter GPU timing <SourceTag source="CUDA event timing, per stage" /></h3>
          {Object.entries(result.implementations).filter(([l]) => l !== "CPU").map(([label, impl]) => (
            <div key={label} style={{ marginBottom: "0.8rem" }}>
              <p style={{ fontWeight: 600, marginBottom: "0.3rem" }}>{label}</p>
              <BarChart
                bars={["gaussian", "median", "sobel", "laplacian", "threshold"].map((stage) => ({
                  label: stage, value: impl.per_stage_ms?.[stage], kind: IMPL_KIND[label],
                }))}
                formatValue={fmtMs}
              />
            </div>
          ))}

          <h3>Correctness — this run <SourceTag source="current live run output comparison" /></h3>
          <div className="card-row cols-1" style={{ gridTemplateColumns: "1fr" }}>
            {result.stage_correctness.map((s) => (
              <div key={s.stage} className="card" style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.4rem" }}>
                <span style={{ textTransform: "capitalize" }}>{s.stage}</span>
                <span>
                  {s.status === "N/A" ? "Disabled" : `max diff ${s.max_abs_diff_vs_cpu} (tolerance ${s.tolerance})`}{" "}
                  <Pill status={s.status === "PASS" ? "pass" : s.status === "WARNING" ? "warning" : "experimental"}>{s.status}</Pill>
                </span>
              </div>
            ))}
          </div>
          <div className="card-row cols-3" style={{ marginTop: "0.6rem" }}>
            {result.pipeline_correctness.map((p) => (
              <MetricCard key={p.comparison} title={p.comparison.replace(/_/g, " ")}
                value={`${p.differing_pixel_percentage.toFixed(4)}% differ`}
                subtitle={`${p.differing_pixel_count} pixel(s), RMSE ${p.rmse.toFixed(2)}`} />
            ))}
          </div>

          <h3>Output — final result</h3>
          <div className="card-row cols-3">
            {Object.entries(result.implementations).map(([label, impl]) => (
              <div key={label} className="card" style={{ textAlign: "center" }}>
                <div style={{ marginBottom: "0.4rem" }}>{label}</div>
                {impl.preview_png ? <Zoomable src={impl.preview_png} alt={`${label} output`} onZoom={setZoomSrc} /> : <span className="data-unavailable">No preview</span>}
              </div>
            ))}
          </div>

          {Object.values(result.differences || {}).some(Boolean) && (
            <>
              <h3>Difference image</h3>
              <div className="mode-toggle" role="group" aria-label="Difference view">
                {Object.entries(result.differences).filter(([, v]) => v).map(([key]) => (
                  <button key={key} className={diffView === key ? "active" : ""} onClick={() => setDiffView(key)}>
                    {key.replace(/_/g, " ")}
                  </button>
                ))}
              </div>
              {result.differences[diffView] && (
                <div style={{ marginTop: "0.6rem" }}>
                  <Zoomable src={result.differences[diffView]} alt={`Difference: ${diffView}`} onZoom={setZoomSrc} />
                </div>
              )}
            </>
          )}

          {narrative.length > 0 && (
            <>
              <h3>What just happened?</h3>
              <div className="card">
                {narrative.map((line, i) => <p key={i} style={{ margin: i === 0 ? 0 : "0.4rem 0 0" }}>{line}</p>)}
              </div>
            </>
          )}
        </div>
      )}
      {zoomSrc && <Lightbox src={zoomSrc} label="Output" onClose={() => setZoomSrc(null)} />}
    </div>
  );
}
