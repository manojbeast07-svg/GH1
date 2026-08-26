import { useState } from "react";
import { BarChart } from "../components/BarChart.jsx";

export function ResolutionExplorer({ resolutionSweep }) {
  const rows = resolutionSweep ?? [];
  const [idx, setIdx] = useState(0);
  const row = rows[idx];

  if (!rows.length) {
    return (
      <div className="slide">
        <h1>Resolution Explorer</h1>
        <p className="data-unavailable">Resolution-sweep data unavailable.</p>
      </div>
    );
  }

  return (
    <div className="slide">
      <h1>Resolution Explorer</h1>
      <p className="subtitle">Larger images provide more work per GPU launch, so fixed overhead matters relatively less.</p>

      <div className="card-row" style={{ gridTemplateColumns: `repeat(${rows.length}, 1fr)` }}>
        {rows.map((r, i) => (
          <button
            key={`${r.width}x${r.height}`}
            className={`filter-card${i === idx ? " active" : ""}`}
            onClick={() => setIdx(i)}
          >
            {r.width}×{r.height}
          </button>
        ))}
      </div>

      <h2>Milliseconds per image</h2>
      <BarChart
        bars={[
          { label: "CPU", value: row?.cpu_ms_per_image, kind: "cpu" },
          { label: "Basic CUDA", value: row?.basic_ms_per_image, kind: "basic" },
          { label: "Enhanced CUDA", value: row?.enhanced_ms_per_image, kind: "enhanced" },
        ]}
        formatValue={(v) => `${v.toFixed(3)} ms`}
      />
      <p style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
        This is a measured trend from this project's own resolution sweep, not an absolute rule for every
        workload.
      </p>
    </div>
  );
}
