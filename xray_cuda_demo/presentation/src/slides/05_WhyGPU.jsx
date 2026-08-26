import { MetricCard } from "../components/MetricCard.jsx";

export function WhyGPU({ system, threading }) {
  return (
    <div className="slide">
      <h1>Why GPU?</h1>
      <p className="subtitle">
        The CPU has a smaller number of powerful, general-purpose cores. The GPU has a very large number of
        lightweight, parallel threads.
      </p>

      <div className="card-row cols-2">
        <div className="card">
          <div className="pill cpu">CPU</div>
          <p style={{ marginTop: "0.6rem" }}>A smaller number of powerful general-purpose cores.</p>
          <div style={{ fontFamily: "monospace", color: "var(--cpu)", fontSize: "1.4rem", letterSpacing: "2px" }}>
            {"█".repeat(Math.max(1, Math.min(16, system?.cpu_logical_cores ?? 4)))}
          </div>
        </div>
        <div className="card">
          <div className="pill enhanced">GPU</div>
          <p style={{ marginTop: "0.6rem" }}>A very large number of lightweight parallel threads.</p>
          <div style={{ fontFamily: "monospace", color: "var(--enhanced)", fontSize: "1.4rem", letterSpacing: "1px", lineHeight: 1.3, wordBreak: "break-all" }}>
            {"█".repeat(64)}
          </div>
        </div>
      </div>
      <p style={{ color: "var(--muted)", fontSize: "0.82rem" }}>
        Conceptual animation — bar lengths are illustrative proportions, not exact thread counts. See the
        Threading &amp; Parallelism slide for real, measured launch-configuration numbers.
      </p>

      <div className="card-row cols-2">
        <MetricCard title="CPU logical cores (this machine)" value={system?.cpu_logical_cores ?? null} />
        <MetricCard title="GPU streaming multiprocessors (SMs)" value={threading?.gpu_sm_count ?? null} />
      </div>
    </div>
  );
}
