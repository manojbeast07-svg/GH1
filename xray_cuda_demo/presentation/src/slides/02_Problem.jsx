import { useState } from "react";
import { MetricCard } from "../components/MetricCard.jsx";
import { isMissing } from "../data/loadPresentationData.js";

const GRID_SIZE = 16; // 16x16 illustrative grid -- a real 224x224 image has far more pixels than we can render

export function Problem({ benchmark }) {
  const [hoverCount, setHoverCount] = useState(0);
  const imageCount = benchmark?.manifest?.selected_image_count;
  const resolution = benchmark?.manifest?.resolution;

  return (
    <div className="slide">
      <h1>What Are We Trying to Solve?</h1>
      <p className="subtitle">
        Thousands of X-ray images, each passing through five image-processing filters.
      </p>

      <div className="card-row cols-3">
        <MetricCard title="X-ray images (this dataset)" value={isMissing(imageCount) ? null : imageCount.toLocaleString("en-US")} />
        <MetricCard title="Resolution (canonical benchmark)" value={resolution ? `${resolution[1]} × ${resolution[0]}` : null} subtitle="width × height" />
        <MetricCard title="Filters per image" value="5" subtitle="Gaussian, Median, Sobel, Laplacian, Threshold" />
      </div>

      <h2>An image is a grid of independent pixel operations</h2>
      <p className="subtitle">
        Hover the grid below — each cell is one illustrative pixel. Many filter operations can be computed
        independently, one output pixel at a time, without waiting on any other output pixel.
      </p>
      <div
        className="pixel-grid card"
        style={{ gridTemplateColumns: `repeat(${GRID_SIZE}, 1fr)`, maxWidth: 420, margin: "0 auto" }}
        onMouseLeave={() => setHoverCount(0)}
      >
        {Array.from({ length: GRID_SIZE * GRID_SIZE }).map((_, i) => (
          <div
            key={i}
            className="pixel"
            onMouseEnter={() => setHoverCount(i + 1)}
            style={{
              background: i < hoverCount ? "var(--enhanced)" : "#0b0e13",
              border: "1px solid var(--border)",
              transition: "background 0.1s",
            }}
          />
        ))}
      </div>
      <p className="subtitle" style={{ textAlign: "center" }}>
        {hoverCount > 0
          ? `${hoverCount} of ${GRID_SIZE * GRID_SIZE} illustrative pixels — each can be processed independently.`
          : "Hover to explore."}
      </p>
      <p style={{ color: "var(--muted)", fontSize: "0.85rem", textAlign: "center" }}>
        Illustration — a real {resolution ? `${resolution[1]}×${resolution[0]}` : "224×224"} image has far more
        pixels than shown here; this grid exists to build intuition, not to represent an exact measurement.
      </p>
    </div>
  );
}
