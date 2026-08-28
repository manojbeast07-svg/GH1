import { useState } from "react";
import { BarChart } from "../components/BarChart.jsx";
import { MetricCard } from "../components/MetricCard.jsx";
import { ProvenanceBadge } from "../components/ProvenanceBadge.jsx";
import { runLiveBatchSweep, runLiveResolutionBenchmark } from "../data/apiClient.js";
import { isMissing } from "../utils/isMissing.js";
import { Why } from "../components/Why.jsx";

const BATCH_SIZE_OPTIONS = [1, 8, 16, 32, 64, 128, 256, 512];

function HistoricalBenchmark({ benchmark }) {
  const cpu = benchmark?.cpu?.mode4_end_to_end_ms?.median;
  const basic = benchmark?.basic_cuda?.mode4_end_to_end_ms?.median;
  const enhanced = benchmark?.enhanced_cuda?.mode4_end_to_end_ms?.median;

  return (
    <>
      <BarChart
        bars={[
          { label: "CPU / OpenCV", value: cpu, kind: "cpu" },
          { label: "Basic CUDA", value: basic, kind: "basic" },
          { label: "Enhanced CUDA", value: enhanced, kind: "enhanced" },
        ]}
        formatValue={(v) => `${v.toFixed(2)} ms`}
      />
      <div className="card-row cols-3">
        <MetricCard title="Basic vs CPU" kind="basic" value={isMissing(benchmark?.speedups?.basic_vs_cpu) ? null : `${benchmark.speedups.basic_vs_cpu.toFixed(2)}×`} />
        <MetricCard title="Enhanced vs CPU" kind="enhanced" value={isMissing(benchmark?.speedups?.enhanced_vs_cpu) ? null : `${benchmark.speedups.enhanced_vs_cpu.toFixed(2)}×`} />
        <MetricCard title="Enhanced vs Basic" kind="enhanced" value={isMissing(benchmark?.speedups?.enhanced_vs_basic_compute_only) ? null : `${benchmark.speedups.enhanced_vs_basic_compute_only.toFixed(2)}×`} />
      </div>
    </>
  );
}

function BatchSizeExplorer() {
  const [selected, setSelected] = useState(new Set([1, 8, 32, 64]));
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [rows, setRows] = useState(null);

  function toggle(size) {
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(size) ? next.delete(size) : next.add(size);
      return next;
    });
  }

  async function runTest() {
    setRunning(true);
    setError(null);
    try {
      const data = await runLiveBatchSweep({ batchSizes: Array.from(selected) });
      setRows(data.rows);
    } catch (e) {
      setError(e.message);
    } finally {
      setRunning(false);
    }
  }

  return (
    <div>
      <p style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
        Select batch sizes, then run the test — nothing is plotted until the measurement actually completes.
      </p>
      <div style={{ display: "flex", gap: "0.4rem", flexWrap: "wrap", marginBottom: "0.6rem" }}>
        {BATCH_SIZE_OPTIONS.map((size) => (
          <button key={size} className={`filter-card${selected.has(size) ? " active" : ""}`} style={{ padding: "0.4rem 0.8rem" }} onClick={() => toggle(size)}>
            {size}
          </button>
        ))}
      </div>
      <button className="btn-primary" onClick={runTest} disabled={running || selected.size === 0}>
        {running ? "Running…" : "Run Batch Size Test"}
      </button>
      {error && <p style={{ color: "var(--error)", fontSize: "0.82rem" }}>{error}</p>}
      {rows && (
        <>
          <ProvenanceBadge kind="LIVE" />
          <h4>Throughput (images/second)</h4>
          {rows.map((r) => (
            <div key={r.requested_batch_size} style={{ marginBottom: "0.6rem" }}>
              <p style={{ margin: "0.2rem 0", fontSize: "0.8rem", color: "var(--muted)" }}>Batch size {r.effective_batch_size}</p>
              <BarChart
                bars={[
                  { label: "CPU", value: r.cpu_images_per_second, kind: "cpu" },
                  { label: "Basic CUDA", value: r.basic_images_per_second, kind: "basic" },
                  { label: "Enhanced CUDA", value: r.enhanced_images_per_second, kind: "enhanced" },
                ]}
                formatValue={(v) => `${v.toFixed(0)} img/s`}
              />
            </div>
          ))}
        </>
      )}
    </div>
  );
}

function ResolutionExplorer() {
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [rows, setRows] = useState(null);

  async function runTest() {
    setRunning(true);
    setError(null);
    try {
      const data = await runLiveResolutionBenchmark({});
      setRows(data.rows);
    } catch (e) {
      setError(e.message);
    } finally {
      setRunning(false);
    }
  }

  return (
    <div>
      <p style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
        Samples the real dataset for whatever distinct resolutions are actually present, then benchmarks each —
        sparse results (even one image for a rare shape) are expected, not a bug.
      </p>
      <button className="btn-primary" onClick={runTest} disabled={running}>{running ? "Running…" : "Benchmark Resolutions"}</button>
      {error && <p style={{ color: "var(--error)", fontSize: "0.82rem" }}>{error}</p>}
      {rows && (
        <>
          <ProvenanceBadge kind="LIVE" />
          {rows.map((r) => (
            <div key={`${r.width}x${r.height}`} style={{ marginBottom: "0.6rem" }}>
              <p style={{ margin: "0.2rem 0", fontSize: "0.8rem", color: "var(--muted)" }}>{r.width}×{r.height} ({r.n_images} image{r.n_images === 1 ? "" : "s"} sampled)</p>
              <BarChart
                bars={[
                  { label: "CPU", value: r.cpu_ms_per_image, kind: "cpu" },
                  { label: "Basic CUDA", value: r.basic_ms_per_image, kind: "basic" },
                  { label: "Enhanced CUDA", value: r.enhanced_ms_per_image, kind: "enhanced" },
                ]}
                formatValue={(v) => `${v.toFixed(3)} ms/img`}
              />
            </div>
          ))}
        </>
      )}
    </div>
  );
}

export function Performance({ historical }) {
  return (
    <div className="page">
      <h1>Performance</h1>

      <h2><ProvenanceBadge kind="HISTORICAL" /> Canonical Benchmark</h2>
      <div className="section-historical">
        {historical?.benchmark_summary ? <HistoricalBenchmark benchmark={historical.benchmark_summary} /> : <p className="data-unavailable">Historical benchmark unavailable.</p>}
      </div>

      <h2><ProvenanceBadge kind="LIVE" /> Batch-Size Explorer</h2>
      <div className="section-live">
        <Why question="Why does batch size matter?">
          Every GPU call has fixed overhead — a kernel launch, a host-to-device transfer setup — that doesn't grow
          with batch size. Processing more images in one call amortizes that fixed cost across more work, so
          throughput (images/second) typically rises with batch size, up to whatever this GPU's actual free VRAM
          can safely hold at once (see the "adapt to VRAM" behavior on the Live Processing page for what happens
          past that point).
        </Why>
        <BatchSizeExplorer />
      </div>

      <h2><ProvenanceBadge kind="LIVE" /> Resolution Explorer</h2>
      <div className="section-live">
        <Why question="Why can CPU win for small images?">
          A GPU run always pays a small fixed cost — uploading the image, launching kernels, downloading the
          result — before any parallelism advantage kicks in. For a very small or very simple workload, that fixed
          cost can outweigh the benefit of running in parallel, so CPU/OpenCV (with no transfer overhead at all)
          can come out ahead. Run the Resolution Explorer below on this machine's actual smallest images to see
          whether that happens here.
        </Why>
        <ResolutionExplorer />
      </div>
    </div>
  );
}
