import { useMemo, useState } from "react";
import { BarChart } from "../components/BarChart.jsx";

export function BatchExplorer({ batchSweep }) {
  const rows = batchSweep ?? [];
  const sizes = useMemo(() => rows.map((r) => r.batch_size).sort((a, b) => a - b), [rows]);
  const [idx, setIdx] = useState(0);
  const row = rows.find((r) => r.batch_size === sizes[idx]);

  if (!rows.length) {
    return (
      <div className="slide">
        <h1>Batch Size Explorer</h1>
        <p className="data-unavailable">Batch-sweep data unavailable.</p>
      </div>
    );
  }

  return (
    <div className="slide">
      <h1>Batch Size Explorer</h1>
      <p className="subtitle">Drag the slider to see how throughput changes with batch size (measured data).</p>

      <input
        type="range" min={0} max={sizes.length - 1} value={idx}
        onChange={(e) => setIdx(Number(e.target.value))}
        style={{ width: "100%" }}
        aria-label="Batch size"
      />
      <p style={{ textAlign: "center", fontSize: "1.3rem", fontWeight: 700 }}>Batch size: {row?.batch_size ?? "—"}</p>

      <h2>Throughput (images/second)</h2>
      <BarChart
        bars={[
          { label: "CPU", value: row?.cpu_images_per_second, kind: "cpu" },
          { label: "Basic CUDA", value: row?.basic_images_per_second, kind: "basic" },
          { label: "Enhanced CUDA", value: row?.enhanced_images_per_second, kind: "enhanced" },
        ]}
        formatValue={(v) => `${v.toFixed(0)} img/s`}
      />
    </div>
  );
}
