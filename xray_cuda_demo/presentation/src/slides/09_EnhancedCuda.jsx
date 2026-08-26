import { useState } from "react";
import { TechnicalDetail } from "../components/TechnicalDetail.jsx";
import { isMissing } from "../data/loadPresentationData.js";

const PRODUCTION_VARIANTS = {
  gaussian: "Specialized",
  median: "Network3x3",
  sobel: "Specialized",
  laplacian: "Specialized",
  threshold: "Vectorized",
};

const EXPLANATIONS = {
  gaussian: {
    simple: "Gaussian filtering repeatedly uses nearby pixels. The optimized implementation reduces "
      + "unnecessary memory movement and specializes the computation.",
    technical: "Separable convolution, shared memory, coefficient specialization, compile-time specialization.",
  },
  median: {
    simple: "Median filtering requires comparing neighboring values. For the common 3×3 case, a specialized "
      + "comparison network reduces the amount of unnecessary sorting work.",
    technical: "Branchless sorting network.",
  },
  sobel: {
    simple: "Sobel was already relatively efficient, so optimization produced only a small additional "
      + "improvement.",
    technical: "Compile-time mode specialization over the gradient computation. An honest result: not every "
      + "optimization is dramatic.",
  },
  laplacian: {
    simple: "The original implementation performed more generic work, leaving more room for specialization.",
    technical: "Shared memory, constant coefficients, compile-time specialization.",
  },
  threshold: {
    simple: "Thresholding is a simple operation performed independently for each pixel. The optimization "
      + "therefore focuses mainly on moving data efficiently.",
    technical: "Vectorized uchar4 processing — 4 pixels per thread.",
  },
};

export function EnhancedCuda({ perFilter }) {
  const [selected, setSelected] = useState("gaussian");
  const rows = perFilter ?? [];
  const row = rows.find((r) => r.filter === selected);
  const info = EXPLANATIONS[selected];

  return (
    <div className="slide">
      <h1>Enhanced CUDA — The Optimization Story</h1>
      <pre className="diagram">{"BASIC CUDA\n     ↓\nProfile\n     ↓\nFind bottlenecks\n     ↓\nOptimize each filter\n     ↓\nENHANCED CUDA"}</pre>

      <h2>Production configuration (from the actual repository)</h2>
      <div className="card-row cols-5">
        {Object.entries(PRODUCTION_VARIANTS).map(([f, v]) => (
          <div className="card" key={f} style={{ textAlign: "center" }}>
            <div style={{ textTransform: "capitalize", fontWeight: 600 }}>{f}</div>
            <div style={{ color: "var(--enhanced)", fontSize: "0.85rem" }}>{v}</div>
          </div>
        ))}
      </div>

      <h2>Click a filter to see its optimization story</h2>
      <div className="card-row cols-5">
        {Object.keys(PRODUCTION_VARIANTS).map((f) => (
          <button
            key={f}
            className={`filter-card${selected === f ? " active" : ""}`}
            onClick={() => setSelected(f)}
            style={{ textTransform: "capitalize" }}
          >
            {f}
          </button>
        ))}
      </div>

      <div className="card" style={{ marginTop: "1rem" }}>
        <pre className="diagram" style={{ textAlign: "center" }}>{"Basic\n ↓\nOptimization technique\n ↓\nEnhanced"}</pre>
        <p>{info.simple}</p>
        <TechnicalDetail label="technical detail">
          <p>{info.technical}</p>
        </TechnicalDetail>
        <div className="card-row cols-3" style={{ marginTop: "0.8rem" }}>
          <div className="metric-card card">
            <div className="metric-title">Basic kernel (median)</div>
            <div className="metric-value">{isMissing(row?.basic_kernel_ms) ? <span className="data-unavailable">Not available</span> : `${row.basic_kernel_ms.toFixed(3)} ms`}</div>
          </div>
          <div className="metric-card card">
            <div className="metric-title">Enhanced kernel (median)</div>
            <div className="metric-value">{isMissing(row?.enhanced_kernel_ms) ? <span className="data-unavailable">Not available</span> : `${row.enhanced_kernel_ms.toFixed(3)} ms`}</div>
          </div>
          <div className="metric-card card">
            <div className="metric-title">Measured speedup</div>
            <div className="metric-value" style={{ color: "var(--enhanced)" }}>
              {isMissing(row?.kernel_speedup) ? <span className="data-unavailable">Not available</span> : `${row.kernel_speedup.toFixed(2)}×`}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
