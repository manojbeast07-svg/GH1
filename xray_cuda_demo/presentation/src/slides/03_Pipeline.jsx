import { useEffect, useState } from "react";
import { TechnicalDetail } from "../components/TechnicalDetail.jsx";
import { checkApiStatus, fetchDatasetInfo, fetchPreviewStages } from "../data/apiClient.js";

const FILTERS = [
  {
    id: "gaussian",
    name: "Gaussian",
    prevStage: "original",
    simple: "Smooths the image and reduces small variations/noise.",
    technical: "Separable convolution, shared-memory tiling, and compile-time coefficient specialization " +
      "(production variant: Specialized).",
  },
  {
    id: "median",
    name: "Median",
    prevStage: "gaussian",
    simple: "Replaces a pixel using the median of nearby pixels, which can reduce isolated noise.",
    technical: "3×3 branchless sorting network (production variant: Network3x3) instead of a general sort.",
  },
  {
    id: "sobel",
    name: "Sobel",
    prevStage: "median",
    simple: "Highlights intensity changes such as edges.",
    technical: "Compile-time mode specialization (production variant: Specialized) over the gradient magnitude " +
      "computation.",
  },
  {
    id: "laplacian",
    name: "Laplacian",
    prevStage: "sobel",
    simple: "Detects rapid changes in intensity.",
    technical: "Shared-memory tiling, constant-memory coefficients, compile-time specialization (production " +
      "variant: Specialized).",
  },
  {
    id: "threshold",
    name: "Threshold",
    prevStage: "laplacian",
    simple: "Converts the image into a simpler binary-style representation using a threshold value.",
    technical: "uchar4 vectorized processing — each thread handles 4 pixels at once (production variant: " +
      "Vectorized).",
  },
];

function Lightbox({ src, label, onClose }) {
  useEffect(() => {
    function onKey(e) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      onClick={onClose}
      role="dialog"
      aria-label={`${label} — full size`}
      style={{
        position: "fixed", inset: 0, background: "rgba(4,6,12,0.86)", zIndex: 50,
        display: "flex", alignItems: "center", justifyContent: "center", cursor: "zoom-out",
        animation: "lightboxFadeIn 0.15s ease",
      }}
    >
      <img
        src={src}
        alt={`${label} — full size`}
        style={{ maxWidth: "90vw", maxHeight: "88vh", borderRadius: 8, boxShadow: "0 20px 60px -20px rgba(0,0,0,0.7)" }}
      />
      <div style={{ position: "absolute", top: 16, right: 20, color: "#fff", fontSize: "0.85rem" }}>Click anywhere or press Esc to close</div>
    </div>
  );
}

function ImageCompareSlider({ beforeSrc, afterSrc, slider, onChange, loading, onZoom }) {
  return (
    <div>
      <div
        style={{
          position: "relative", borderRadius: 8, overflow: "hidden", border: "1px solid var(--border)",
          lineHeight: 0, cursor: "zoom-in", opacity: loading ? 0.55 : 1, transition: "opacity 0.2s ease",
        }}
        onClick={() => onZoom(afterSrc)}
        title="Click to zoom in"
      >
        <img src={afterSrc} alt="After" style={{ width: "100%", display: "block" }} />
        <div style={{ position: "absolute", inset: 0, clipPath: `inset(0 ${100 - slider}% 0 0)`, lineHeight: 0 }}>
          <img src={beforeSrc} alt="Before" style={{ width: "100%", display: "block" }} />
        </div>
        <div
          style={{
            position: "absolute", top: 0, bottom: 0, left: `${slider}%`,
            width: 2, background: "var(--enhanced)", boxShadow: "0 0 8px var(--enhanced)",
          }}
        />
        <div style={{ position: "absolute", left: 8, top: 8, fontSize: "0.7rem", color: "#fff", textShadow: "0 1px 2px #000" }}>Before</div>
        <div style={{ position: "absolute", right: 8, top: 8, fontSize: "0.7rem", color: "#fff", textShadow: "0 1px 2px #000" }}>After</div>
      </div>
      <input
        type="range" min={0} max={100} value={slider}
        onClick={(e) => e.stopPropagation()}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ width: "100%", marginTop: "0.5rem" }}
        aria-label="Before/after slider"
      />
    </div>
  );
}

export function Pipeline() {
  const [selected, setSelected] = useState("gaussian");
  const [slider, setSlider] = useState(50);
  const [serverAvailable, setServerAvailable] = useState(null);
  const [datasetTotal, setDatasetTotal] = useState(null);
  const [imageIndex, setImageIndex] = useState(0);
  const [stages, setStages] = useState(null);
  const [stagesLoading, setStagesLoading] = useState(false);
  const [error, setError] = useState(null);
  const [zoomSrc, setZoomSrc] = useState(null);
  const filter = FILTERS.find((f) => f.id === selected);

  useEffect(() => {
    checkApiStatus().then((s) => setServerAvailable(!!s?.available));
    fetchDatasetInfo().then((d) => {
      if (d?.total_files) setDatasetTotal(d.total_files);
    });
  }, []);

  useEffect(() => {
    if (!serverAvailable) return;
    let cancelled = false;
    setStagesLoading(true);
    setError(null);
    fetchPreviewStages(imageIndex)
      .then((data) => {
        if (!cancelled) setStages(data);
      })
      .catch((e) => {
        if (!cancelled) setError(e.message);
      })
      .finally(() => {
        if (!cancelled) setStagesLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [serverAvailable, imageIndex]);

  const hasRealImages = serverAvailable && stages && !error;
  const maxIndex = datasetTotal ? datasetTotal - 1 : null;

  function stepImage(delta) {
    setImageIndex((i) => {
      const next = i + delta;
      if (next < 0) return 0;
      if (maxIndex !== null && next > maxIndex) return maxIndex;
      return next;
    });
  }

  function randomImage() {
    if (!maxIndex) return;
    setImageIndex(Math.floor(Math.random() * (maxIndex + 1)));
  }

  return (
    <div className="slide">
      <h1>The X-ray Pipeline</h1>
      <p className="subtitle">Every image passes through the same five filters, in this order.</p>

      <pre className="diagram">{"Original\n   ↓\nGaussian\n   ↓\nMedian\n   ↓\nSobel\n   ↓\nLaplacian\n   ↓\nThreshold"}</pre>

      <h2>Click a filter to explore it</h2>
      <div className="card-row cols-5">
        {FILTERS.map((f) => (
          <button
            key={f.id}
            className={`filter-card${selected === f.id ? " active" : ""}`}
            onClick={() => setSelected(f.id)}
          >
            {f.name}
          </button>
        ))}
      </div>

      <div className="card" style={{ marginTop: "1rem" }}>
        <h3 style={{ marginTop: 0 }}>{filter.name}</h3>
        <p>{filter.simple}</p>
        <TechnicalDetail label="What CUDA does with it">
          <p>{filter.technical}</p>
        </TechnicalDetail>

        <div style={{ marginTop: "1.2rem" }}>
          {serverAvailable && (
            <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", marginBottom: "0.7rem" }}>
              <button className="tech-toggle" onClick={() => stepImage(-1)} disabled={imageIndex === 0} aria-label="Previous image">
                ◀ Prev
              </button>
              <span style={{ fontSize: "0.82rem", color: "var(--muted)" }}>
                Image {imageIndex}{maxIndex !== null ? ` of ${maxIndex}` : ""}
              </span>
              <button className="tech-toggle" onClick={() => stepImage(1)} disabled={maxIndex !== null && imageIndex >= maxIndex} aria-label="Next image">
                Next ▶
              </button>
              <button className="tech-toggle" onClick={randomImage} disabled={!maxIndex} aria-label="Random image">
                🎲 Random
              </button>
              {stagesLoading && <span className="data-unavailable">loading…</span>}
            </div>
          )}

          {hasRealImages ? (
            <>
              <p style={{ color: "var(--muted)", fontSize: "0.82rem" }}>
                Real output from the production Enhanced CUDA pipeline, on a real dataset X-ray (index {imageIndex}) —
                "Before" is the pipeline's state entering this stage ({filter.prevStage === "original" ? "the original image" : `after ${filter.prevStage}`}), "After" is this stage's real output. Click the image to zoom in.
              </p>
              <ImageCompareSlider
                beforeSrc={stages.stages[filter.prevStage]}
                afterSrc={stages.stages[filter.id]}
                slider={slider}
                onChange={setSlider}
                loading={stagesLoading}
                onZoom={setZoomSrc}
              />
            </>
          ) : (
            <>
              <p style={{ color: "var(--muted)", fontSize: "0.82rem" }}>
                {serverAvailable === null
                  ? "Loading a real example…"
                  : error
                    ? `Live server error: ${error} — showing an illustration instead.`
                    : "Illustration — start the live server (python scripts/presentation_api_server.py) to see a real filtered X-ray here instead."}
              </p>
              <div style={{ position: "relative", height: 120, borderRadius: 8, overflow: "hidden", border: "1px solid var(--border)" }}>
                <div style={{ position: "absolute", inset: 0, background: "linear-gradient(90deg, #333 0%, #999 50%, #333 100%)" }} />
                <div
                  style={{
                    position: "absolute", inset: 0,
                    clipPath: `inset(0 ${100 - slider}% 0 0)`,
                    background: "linear-gradient(90deg, #1a1a1a 0%, #eee 50%, #1a1a1a 100%)",
                    filter: "contrast(1.6)",
                  }}
                />
                <div style={{ position: "absolute", left: 8, top: 8, fontSize: "0.7rem", color: "#fff" }}>Before</div>
                <div style={{ position: "absolute", right: 8, top: 8, fontSize: "0.7rem", color: "#fff" }}>After</div>
              </div>
              <input
                type="range" min={0} max={100} value={slider}
                onChange={(e) => setSlider(Number(e.target.value))}
                style={{ width: "100%", marginTop: "0.5rem" }}
                aria-label="Before/after slider"
              />
            </>
          )}
        </div>
      </div>

      {zoomSrc && <Lightbox src={zoomSrc} label={filter.name} onClose={() => setZoomSrc(null)} />}
    </div>
  );
}
