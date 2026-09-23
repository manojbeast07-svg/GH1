import { useEffect, useState } from "react";
import { TechnicalDetail } from "../components/TechnicalDetail.jsx";
import { ProvenanceBadge } from "../components/ProvenanceBadge.jsx";
import { MetricCard } from "../components/MetricCard.jsx";
import { checkApiStatus, fetchDatasetInfo, fetchPreviewStages, runLivePerFilter } from "../data/apiClient.js";
import { Lightbox, Zoomable } from "../components/Lightbox.jsx";
import { FilterMath } from "../components/FilterMath.jsx";
import { isMissing } from "../utils/isMissing.js";

const FILTERS = [
  {
    id: "gaussian", name: "Gaussian", prevStage: "original", productionVariant: "Specialized",
    simple: "Smooths the image and reduces small variations/noise.",
    technical: "Separable convolution, shared-memory tiling, and compile-time coefficient specialization.",
  },
  {
    id: "median", name: "Median", prevStage: "gaussian", productionVariant: "Network3x3",
    simple: "Replaces a pixel using the median of nearby pixels, which can reduce isolated noise.",
    technical: "3×3 branchless sorting network instead of a general sort.",
  },
  {
    id: "sobel", name: "Sobel", prevStage: "median", productionVariant: "Specialized",
    simple: "Highlights intensity changes such as edges. The optimization headroom here is relatively small — Sobel was already efficient.",
    technical: "Compile-time mode specialization over the gradient magnitude computation.",
  },
  {
    id: "laplacian", name: "Laplacian", prevStage: "sobel", productionVariant: "Specialized",
    simple: "Detects rapid changes in intensity.",
    technical: "Shared-memory tiling, constant-memory coefficients, compile-time specialization. Kernel size (1/3/5) affects the neighborhood examined.",
  },
  {
    id: "threshold", name: "Threshold", prevStage: "laplacian", productionVariant: "Vectorized",
    simple: "A pointwise operation performed independently for each pixel.",
    technical: "uchar4 vectorized processing — each thread handles 4 pixels at once.",
  },
];

export function Filters({ historical }) {
  const [selected, setSelected] = useState("gaussian");
  const [serverAvailable, setServerAvailable] = useState(null);
  const [datasetTotal, setDatasetTotal] = useState(null);
  const [imageIndex, setImageIndex] = useState(0);
  const [stages, setStages] = useState(null);
  const [stagesLoading, setStagesLoading] = useState(false);
  const [liveResult, setLiveResult] = useState(null);
  const [liveRunning, setLiveRunning] = useState(false);
  const [liveError, setLiveError] = useState(null);
  const [zoomSrc, setZoomSrc] = useState(null);

  const filter = FILTERS.find((f) => f.id === selected);
  const historicalRow = (historical ?? []).find((r) => r.filter === selected);

  useEffect(() => {
    checkApiStatus().then((s) => setServerAvailable(!!s?.available));
    fetchDatasetInfo().then((d) => { if (d?.total_files) setDatasetTotal(d.total_files); });
  }, []);

  useEffect(() => {
    if (!serverAvailable) return;
    setStagesLoading(true);
    fetchPreviewStages(imageIndex).then(setStages).catch(() => setStages(null)).finally(() => setStagesLoading(false));
  }, [serverAvailable, imageIndex]);

  async function runLiveTiming() {
    setLiveRunning(true);
    setLiveError(null);
    try {
      const data = await runLivePerFilter({ batchSize: 16, seed: 42 });
      setLiveResult(data);
    } catch (e) {
      setLiveError(e.message);
    } finally {
      setLiveRunning(false);
    }
  }

  const liveRow = liveResult?.per_filter_results?.find((r) => r.filter === selected);

  return (
    <div className="page">
      <h1>Filters</h1>
      <p className="subtitle">Five image-processing filters, each with a Basic and an Enhanced CUDA implementation.</p>

      <div className="card-row cols-5">
        {FILTERS.map((f) => (
          <button key={f.id} className={`filter-card${selected === f.id ? " active" : ""}`} onClick={() => setSelected(f.id)}>{f.name}</button>
        ))}
      </div>

      <div className="card" style={{ marginTop: "1rem" }}>
        <h3 style={{ marginTop: 0 }}>{filter.name} — production variant: {filter.productionVariant}</h3>
        <p>{filter.simple}</p>
        <TechnicalDetail label="what CUDA does with it"><p>{filter.technical}</p></TechnicalDetail>

        {serverAvailable && stages?.stages && (
          <div style={{ marginTop: "1rem" }}>
            <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginBottom: "0.5rem" }}>
              <button className="tech-toggle" disabled={imageIndex === 0} onClick={() => setImageIndex((i) => Math.max(0, i - 1))}>◀ Prev</button>
              <span style={{ fontSize: "0.82rem", color: "var(--muted)" }}>Image {imageIndex}{datasetTotal ? ` of ${datasetTotal - 1}` : ""}</span>
              <button className="tech-toggle" onClick={() => setImageIndex((i) => i + 1)}>Next ▶</button>
              {stagesLoading && <span className="data-unavailable">loading…</span>}
            </div>
            <div className="card-row cols-2">
              <div className="card" style={{ textAlign: "center" }}>
                <div style={{ marginBottom: "0.3rem", fontSize: "0.8rem", color: "var(--muted)" }}>Before ({filter.prevStage})</div>
                <Zoomable src={stages.stages[filter.prevStage]} alt="Before" onZoom={setZoomSrc} />
              </div>
              <div className="card" style={{ textAlign: "center" }}>
                <div style={{ marginBottom: "0.3rem", fontSize: "0.8rem", color: "var(--muted)" }}>After ({filter.name})</div>
                <Zoomable src={stages.stages[filter.id]} alt="After" onZoom={setZoomSrc} />
              </div>
            </div>
          </div>
        )}
        {zoomSrc && <Lightbox src={zoomSrc} label={filter.name} onClose={() => setZoomSrc(null)} />}

        {serverAvailable && stages?.stages && (
          <FilterMath stage={selected} imageIndex={imageIndex} imageSrc={stages.stages[filter.prevStage]} />
        )}

        <h4 style={{ marginTop: "1.2rem" }}><ProvenanceBadge kind="LIVE" /> Current live kernel time</h4>
        <button className="btn-primary" onClick={runLiveTiming} disabled={liveRunning || !serverAvailable}>
          {liveRunning ? "Running…" : "Run Live Timing (all 5 filters)"}
        </button>
        {liveError && <p style={{ color: "var(--error)", fontSize: "0.82rem" }}>{liveError}</p>}
        {liveRow && (
          <div className="card-row cols-3" style={{ marginTop: "0.6rem" }}>
            <MetricCard title="Basic kernel" kind="basic" value={isMissing(liveRow.basic_kernel_ms) ? "Disabled" : `${liveRow.basic_kernel_ms.toFixed(3)} ms`} />
            <MetricCard title="Enhanced kernel" kind="enhanced" value={isMissing(liveRow.enhanced_kernel_ms) ? "Disabled" : `${liveRow.enhanced_kernel_ms.toFixed(3)} ms`} />
            <MetricCard title="Speedup" kind="enhanced" value={isMissing(liveRow.kernel_speedup) ? "N/A" : `${liveRow.kernel_speedup.toFixed(2)}×`} />
          </div>
        )}

        <h4 style={{ marginTop: "1.2rem" }}><ProvenanceBadge kind="HISTORICAL" /> Canonical optimization result</h4>
        {historicalRow ? (
          <div className="card-row cols-3">
            <MetricCard title="Basic kernel (median)" kind="basic" value={isMissing(historicalRow.basic_kernel_ms) ? "N/A" : `${historicalRow.basic_kernel_ms.toFixed(3)} ms`} />
            <MetricCard title="Enhanced kernel (median)" kind="enhanced" value={isMissing(historicalRow.enhanced_kernel_ms) ? "N/A" : `${historicalRow.enhanced_kernel_ms.toFixed(3)} ms`} />
            <MetricCard title="Measured speedup" kind="enhanced" value={isMissing(historicalRow.kernel_speedup) ? "N/A" : `${historicalRow.kernel_speedup.toFixed(2)}×`} />
          </div>
        ) : <p className="data-unavailable">Historical data unavailable.</p>}
      </div>
    </div>
  );
}
