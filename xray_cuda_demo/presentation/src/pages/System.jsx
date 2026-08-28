import { useEffect, useState } from "react";
import { fetchSystemInfo } from "../data/apiClient.js";
import { MetricCard } from "../components/MetricCard.jsx";
import { ProvenanceBadge } from "../components/ProvenanceBadge.jsx";

export function System() {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchSystemInfo().then(setInfo).catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="page"><h1>System</h1><p className="data-unavailable">{error}</p></div>;
  if (!info) return <div className="page"><h1>System</h1><p className="data-unavailable">Loading…</p></div>;

  const fp = info.fingerprint;

  return (
    <div className="page">
      <h1>System</h1>
      <ProvenanceBadge kind="LIVE" />
      <p className="subtitle" style={{ marginTop: "0.6rem" }}>Live environment introspection from this machine — queried on load, never hardcoded.</p>

      <h2>Hardware</h2>
      <div className="card-row cols-3">
        <MetricCard title="CPU" value={fp.cpu_model} />
        <MetricCard title="Logical Cores" value={fp.cpu_logical_cores} />
        <MetricCard title="GPU" value={fp.gpu_name ?? null} />
      </div>
      <div className="card-row cols-3">
        <MetricCard title="VRAM" value={fp.gpu_vram_bytes ? `${(fp.gpu_vram_bytes / 1e9).toFixed(1)} GB` : null} />
        <MetricCard title="Compute Capability" value={fp.gpu_compute_capability} />
        <MetricCard title="Driver Version" value={fp.gpu_driver_version} />
      </div>

      <h2>Toolchain</h2>
      <div className="card-row cols-3">
        <MetricCard title="CUDA Runtime" value={fp.cuda_runtime_version} />
        <MetricCard title="NVCC" value={info.tool_versions?.nvcc} />
        <MetricCard title="CMake" value={info.tool_versions?.cmake} />
      </div>
      <div className="card-row cols-3">
        <MetricCard title="Python" value={fp.python_version} />
        <MetricCard title="NumPy" value={fp.numpy_version} />
        <MetricCard title="OpenCV" value={fp.opencv_version} />
      </div>
      <div className="card-row cols-2">
        <MetricCard title="pybind11" value={fp.pybind11_version} />
        <MetricCard title="Git commit" value={fp.git_commit?.slice(0, 12)} />
      </div>

      <h2>Live GPU memory</h2>
      <div className="card-row cols-2">
        <MetricCard title="VRAM total" value={info.gpu_memory ? `${(info.gpu_memory.total_bytes / 1e9).toFixed(2)} GB` : null} />
        <MetricCard title="VRAM free (right now)" value={info.gpu_memory ? `${(info.gpu_memory.free_bytes / 1e9).toFixed(2)} GB` : null} />
      </div>
      <p style={{ color: "var(--muted)", fontSize: "0.82rem" }}>
        VRAM allocated specifically by the current pipeline run is not separately tracked by the native
        extension — only the device-wide free/total figures above are queried directly (cudaMemGetInfo).
      </p>

      <h2>Implementation facts</h2>
      <div className="card-row cols-3">
        <MetricCard title="Native extension" value={info.gpu_implementation_facts?.native_extension ? "Yes" : "No"} />
        <MetricCard title="Single fused pipeline call" value={info.gpu_implementation_facts?.single_pipeline_call ? "Yes" : "No"} />
        <MetricCard title=".cu source files" value={info.gpu_implementation_facts?.cu_file_count} />
      </div>
      <div className="card-row cols-2">
        <MetricCard title="No Python GPU libraries" value={info.gpu_implementation_facts?.no_python_gpu_libs ? "Confirmed" : "Not confirmed"} />
        <MetricCard title="CUDA event timing" value={info.gpu_implementation_facts?.cuda_event_timing ? "Confirmed" : "Not confirmed"} />
      </div>
    </div>
  );
}
