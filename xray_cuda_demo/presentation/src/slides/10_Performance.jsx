import { useState } from "react";
import { BarChart } from "../components/BarChart.jsx";
import { MetricCard } from "../components/MetricCard.jsx";
import { isMissing } from "../data/loadPresentationData.js";

export function Performance({ benchmark }) {
  const [showWhy, setShowWhy] = useState(false);
  const cpu = benchmark?.cpu?.mode4_end_to_end_ms?.median;
  const basic = benchmark?.basic_cuda?.mode4_end_to_end_ms?.median;
  const enhanced = benchmark?.enhanced_cuda?.mode4_end_to_end_ms?.median;
  const cpuCompute = benchmark?.cpu?.mode3_pipeline_ms?.median;
  const basicCompute = benchmark?.basic_cuda?.mode2_gpu_processing_ms?.median;
  const enhancedCompute = benchmark?.enhanced_cuda?.mode2_gpu_processing_ms?.median;

  return (
    <div className="slide">
      <h1>Performance</h1>
      <p className="subtitle">Same X-ray images, three implementations, measured end-to-end.</p>

      <h2>Full application time (disk load + processing)</h2>
      <BarChart
        bars={[
          { label: "CPU / OpenCV", value: cpu, kind: "cpu" },
          { label: "Basic CUDA", value: basic, kind: "basic" },
          { label: "Enhanced CUDA", value: enhanced, kind: "enhanced" },
        ]}
        formatValue={(v) => `${v.toFixed(2)} ms`}
      />

      <button className="tech-toggle" onClick={() => setShowWhy((s) => !s)} style={{ marginTop: "0.5rem" }}>
        Why is this faster?
      </button>
      {showWhy && (
        <div className="card" style={{ marginTop: "0.6rem" }}>
          <pre className="diagram">{"More parallel work\n+\nGPU-resident processing\n+\nKernel optimization\n=\nLower measured processing time"}</pre>
          <p style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
            GPU processing does not always win — see "Compute vs. End-to-End" below, and the "What Didn't
            Work" slide for cases where an optimization made things worse.
          </p>
        </div>
      )}

      <h2>GPU compute only (excludes disk I/O)</h2>
      <BarChart
        bars={[
          { label: "CPU pipeline", value: cpuCompute, kind: "cpu" },
          { label: "Basic GPU compute", value: basicCompute, kind: "basic" },
          { label: "Enhanced GPU compute", value: enhancedCompute, kind: "enhanced" },
        ]}
        formatValue={(v) => `${v.toFixed(2)} ms`}
      />

      <div className="card-row cols-3">
        <MetricCard title="Basic vs CPU" value={isMissing(benchmark?.speedups?.basic_vs_cpu) ? null : `${benchmark.speedups.basic_vs_cpu.toFixed(2)}×`} />
        <MetricCard title="Enhanced vs CPU" value={isMissing(benchmark?.speedups?.enhanced_vs_cpu) ? null : `${benchmark.speedups.enhanced_vs_cpu.toFixed(2)}×`} />
        <MetricCard title="Enhanced vs Basic (compute only)" value={isMissing(benchmark?.speedups?.enhanced_vs_basic_compute_only) ? null : `${benchmark.speedups.enhanced_vs_basic_compute_only.toFixed(2)}×`} />
      </div>

      <h2>Compute vs. End-to-End</h2>
      <p>
        Making the kernels faster does not remove disk I/O, data transfer, or host-side overhead. The
        "GPU compute" bars above measure only the processing itself; the "full application" bars above include
        everything the real application actually pays for.
      </p>
      <div className="card-row cols-2">
        <MetricCard title="Enhanced vs Basic — compute only" value={isMissing(benchmark?.speedups?.enhanced_vs_basic_compute_only) ? null : `${benchmark.speedups.enhanced_vs_basic_compute_only.toFixed(2)}×`} />
        <MetricCard title="Enhanced vs Basic — full application" value={isMissing(benchmark?.speedups?.enhanced_vs_basic_end_to_end) ? null : `${benchmark.speedups.enhanced_vs_basic_end_to_end.toFixed(2)}×`} />
      </div>
    </div>
  );
}
