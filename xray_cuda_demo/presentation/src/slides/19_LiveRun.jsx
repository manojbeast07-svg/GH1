import { useEffect, useState } from "react";
import { checkApiStatus, fetchDatasetInfo, runLive } from "../data/apiClient.js";
import { MetricCard } from "../components/MetricCard.jsx";
import { BarChart } from "../components/BarChart.jsx";
import { Pill } from "../components/Pill.jsx";
import { isMissing } from "../data/loadPresentationData.js";

const IMPL_KIND = { CPU: "cpu", "Basic CUDA": "basic", "Enhanced CUDA": "enhanced" };

function fmtMs(v) {
  return isMissing(v) ? "N/A" : `${v.toFixed(3)} ms`;
}

export function LiveRun() {
  const [serverStatus, setServerStatus] = useState({ checked: false, available: false });
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [mode, setMode] = useState("batch");
  const [batchSize, setBatchSize] = useState(8);
  const [seed, setSeed] = useState(42);
  const [imageIndex, setImageIndex] = useState(0);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    checkApiStatus().then((s) => setServerStatus({ checked: true, ...s }));
    fetchDatasetInfo().then(setDatasetInfo);
  }, []);

  async function handleRun() {
    setRunning(true);
    setError(null);
    try {
      const data = await runLive({ mode, batchSize, seed, imageIndex });
      setResult(data);
    } catch (e) {
      setError(e.message);
      setResult(null);
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="slide">
      <h1>Try It Yourself</h1>
      <p className="subtitle">
        Pick a batch size (or a single image), run it, and see real, freshly-measured metrics — not the
        pre-recorded benchmark shown elsewhere in this presentation.
      </p>

      {!serverStatus.checked && <p className="data-unavailable">Checking for the live processing server…</p>}
      {serverStatus.checked && !serverStatus.available && (
        <div className="card" style={{ borderColor: "var(--error)" }}>
          <p style={{ margin: 0 }}>
            <strong>Live processing server not reachable.</strong> Start it from the project root with:
          </p>
          <pre className="diagram" style={{ marginTop: "0.5rem" }}>
            {"python scripts/presentation_api_server.py"}
          </pre>
          <p style={{ color: "var(--muted)", fontSize: "0.85rem", margin: 0 }}>
            The rest of this presentation still works from pre-recorded data — this section specifically needs
            the live server, since it runs the real pipeline on demand.
          </p>
        </div>
      )}

      {serverStatus.available && (
        <>
          <div className="card">
            <div className="card-row cols-2">
              <div>
                <label style={{ display: "block", fontSize: "0.8rem", color: "var(--muted)", marginBottom: "0.3rem" }}>
                  Mode
                </label>
                <select value={mode} onChange={(e) => setMode(e.target.value)} style={{ width: "100%", padding: "0.4rem" }}>
                  <option value="batch">Batch</option>
                  <option value="single">Single image</option>
                </select>
              </div>
              {mode === "batch" ? (
                <div>
                  <label style={{ display: "block", fontSize: "0.8rem", color: "var(--muted)", marginBottom: "0.3rem" }}>
                    Batch size
                  </label>
                  <input
                    type="number" min={1} max={512} value={batchSize}
                    onChange={(e) => setBatchSize(Number(e.target.value))}
                    style={{ width: "100%", padding: "0.4rem" }}
                  />
                </div>
              ) : (
                <div>
                  <label style={{ display: "block", fontSize: "0.8rem", color: "var(--muted)", marginBottom: "0.3rem" }}>
                    Image index
                  </label>
                  <input
                    type="number" min={0} value={imageIndex}
                    onChange={(e) => setImageIndex(Number(e.target.value))}
                    style={{ width: "100%", padding: "0.4rem" }}
                  />
                </div>
              )}
            </div>
            {mode === "batch" && (
              <div style={{ marginTop: "0.8rem" }}>
                <label style={{ display: "block", fontSize: "0.8rem", color: "var(--muted)", marginBottom: "0.3rem" }}>
                  Random seed
                </label>
                <input
                  type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))}
                  style={{ width: "100%", padding: "0.4rem" }}
                />
              </div>
            )}
            {datasetInfo && (
              <p style={{ color: "var(--muted)", fontSize: "0.8rem", marginTop: "0.6rem" }}>
                Dataset: {datasetInfo.total_files?.toLocaleString("en-US")} images available.
              </p>
            )}
            <button
              onClick={handleRun}
              disabled={running}
              className="tech-toggle"
              style={{ marginTop: "0.8rem", padding: "0.5rem 1.2rem", fontSize: "0.95rem" }}
            >
              {running ? "Running…" : "Run"}
            </button>
          </div>

          {error && (
            <div className="card" style={{ borderColor: "var(--error)", marginTop: "1rem" }}>
              <strong>Run failed:</strong> {error}
            </div>
          )}

          {result && (
            <div style={{ marginTop: "1.5rem" }}>
              <h2>Results — {result.batch_size} image{result.batch_size === 1 ? "" : "s"}, {result.resolution[0]}×{result.resolution[1]}</h2>

              <h3>Total time</h3>
              <BarChart
                bars={Object.entries(result.implementations).map(([label, impl]) => ({
                  label, value: impl.total_ms, kind: IMPL_KIND[label],
                }))}
                formatValue={fmtMs}
              />
              <p style={{ color: "var(--muted)", fontSize: "0.82rem" }}>
                Total time for this exact run, this exact batch — freshly measured just now, not a historical average.
              </p>

              <div className="card-row cols-3">
                {Object.entries(result.speedups_vs_cpu).map(([label, v]) => (
                  <MetricCard key={label} title={`${label} vs CPU`} value={isMissing(v) ? null : `${v.toFixed(2)}×`} />
                ))}
              </div>

              <h3>H2D / Compute / D2H breakdown</h3>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr style={{ color: "var(--muted)", fontSize: "0.78rem", textAlign: "right" }}>
                    <th style={{ textAlign: "left" }}>Implementation</th>
                    <th>H2D</th><th>Compute</th><th>D2H</th><th>Total</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.implementations).map(([label, impl]) => (
                    <tr key={label} style={{ borderTop: "1px solid var(--border)" }}>
                      <td style={{ padding: "0.3rem 0" }}>{label}</td>
                      <td style={{ textAlign: "right" }}>{fmtMs(impl.h2d_ms)}</td>
                      <td style={{ textAlign: "right" }}>{fmtMs(impl.compute_ms)}</td>
                      <td style={{ textAlign: "right" }}>{fmtMs(impl.d2h_ms)}</td>
                      <td style={{ textAlign: "right", fontWeight: 600 }}>{fmtMs(impl.total_ms)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p style={{ color: "var(--muted)", fontSize: "0.82rem" }}>
                CPU has no H2D/D2H (no GPU transfer involved), so those cells show N/A rather than zero.
              </p>

              <h3>Correctness — this run</h3>
              <div className="card-row cols-1" style={{ gridTemplateColumns: "1fr" }}>
                {result.stage_correctness.map((s) => (
                  <div key={s.stage} className="card" style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.4rem" }}>
                    <span style={{ textTransform: "capitalize" }}>{s.stage}</span>
                    <span>
                      max diff {isMissing(s.max_abs_diff_vs_cpu) ? "N/A" : s.max_abs_diff_vs_cpu} (tolerance {s.tolerance}) {" "}
                      <Pill status={s.status === "PASS" ? "pass" : s.status === "WARNING" ? "warning" : "experimental"}>{s.status}</Pill>
                    </span>
                  </div>
                ))}
              </div>
              <p style={{ color: "var(--muted)", fontSize: "0.82rem" }}>
                A WARNING here doesn't necessarily mean a bug — small floating-point differences are expected
                and can occasionally exceed a filter's tolerance on any given random batch; see the full-pipeline
                differing-pixel percentage below for the metric that actually matters.
              </p>
              <div className="card-row cols-3" style={{ marginTop: "0.6rem" }}>
                {result.pipeline_correctness.map((p) => (
                  <MetricCard
                    key={p.comparison}
                    title={p.comparison.replace(/_/g, " ")}
                    value={`${p.differing_pixel_percentage.toFixed(4)}% differ`}
                    subtitle={`${p.differing_pixel_count} pixel(s), RMSE ${p.rmse.toFixed(2)}`}
                  />
                ))}
              </div>

              {result.threading && (
                <>
                  <h3>Thread launch configuration — this run</h3>
                  <p style={{ color: "var(--muted)", fontSize: "0.82rem" }}>
                    The actual GPU launch geometry for the {result.batch_size} image{result.batch_size === 1 ? "" : "s"} you
                    just ran, at {result.resolution[1]}×{result.resolution[0]}, on the production 16×16 block.
                  </p>
                  <div className="card-row cols-4">
                    <MetricCard
                      title="Grid"
                      value={result.threading.grid_dimensions ? result.threading.grid_dimensions.join(" × ") : null}
                    />
                    <MetricCard title="Threads/Block" value={result.threading.threads_per_block ?? null} />
                    <MetricCard title="Warps/Block" value={result.threading.warps_per_block ?? null} />
                    <MetricCard
                      title="Total Threads Launched"
                      value={isMissing(result.threading.total_threads_launched) ? null : result.threading.total_threads_launched.toLocaleString("en-US")}
                    />
                  </div>
                </>
              )}

              <h3>Difference — visual preview</h3>
              <p style={{ color: "var(--muted)", fontSize: "0.82rem" }}>
                Final output of the first image in this run, from each implementation. Basic and Enhanced should
                look pixel-identical to each other (verified above).
              </p>
              <div className="card-row cols-3">
                {Object.entries(result.implementations).map(([label, impl]) => (
                  <div key={label} className="card" style={{ textAlign: "center" }}>
                    <div style={{ marginBottom: "0.4rem" }}>{label}</div>
                    {impl.preview_png ? (
                      <img src={impl.preview_png} alt={`${label} output`} style={{ width: "100%", borderRadius: 6 }} />
                    ) : (
                      <span className="data-unavailable">No preview</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
