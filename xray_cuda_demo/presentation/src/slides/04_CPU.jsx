import { MetricCard } from "../components/MetricCard.jsx";

export function CPU({ system }) {
  return (
    <div className="slide">
      <h1>CPU / OpenCV</h1>
      <p className="subtitle">
        The CPU executes the image-processing operations using general-purpose processing cores and optimized
        OpenCV routines.
      </p>

      <pre className="diagram">{"Image\n ↓\nOpenCV\n ↓\nCPU cores"}</pre>

      <div className="card-row cols-2">
        <MetricCard title="CPU" value={system?.cpu_model?.slice(0, 30) ?? null} />
        <MetricCard title="Logical Cores (this machine)" value={system?.cpu_logical_cores ?? null} />
      </div>
    </div>
  );
}
