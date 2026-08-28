import { useEffect, useState } from "react";
import { useMode } from "../hooks/useModeContext.jsx";
import { fetchSystemInfo } from "../data/apiClient.js";
import { CudaExplained } from "../slides/06_CudaExplained.jsx";
import { FinalArchitecture } from "../slides/16_FinalArchitecture.jsx";
import { ProvenanceBadge } from "../components/ProvenanceBadge.jsx";
import { Why } from "../components/Why.jsx";
import { Gpu3D } from "../components/Gpu3D.jsx";
import { ThreadHierarchy3D } from "../components/ThreadHierarchy3D.jsx";

function WhyGpuVisual() {
  const [info, setInfo] = useState(null);

  useEffect(() => {
    fetchSystemInfo().then(setInfo).catch(() => setInfo(null));
  }, []);

  const cpuCores = info?.fingerprint?.cpu_logical_cores ?? 4;

  return (
    <>
      <h2>Why GPU?</h2>
      <div className="card-row cols-2">
        <div className="card">
          <div className="pill cpu">CPU</div>
          <p style={{ marginTop: "0.6rem" }}>A smaller number of powerful general-purpose cores.</p>
          <div style={{ fontFamily: "monospace", color: "var(--cpu)", fontSize: "1.4rem", letterSpacing: "2px" }}>
            {"█".repeat(Math.max(1, Math.min(16, cpuCores)))}
          </div>
          {info && <p style={{ color: "var(--muted)", fontSize: "0.78rem" }}>{cpuCores} logical processors on this machine <ProvenanceBadge kind="LIVE" /></p>}
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
        Conceptual illustration — bar lengths are illustrative proportions, not exact thread counts. See the
        Threading &amp; Parallelism page for real, measured launch-configuration numbers and a direct CPU-vs-GPU
        thread-count comparison for your last run.
      </p>
    </>
  );
}

export function CudaArchitecture({ live }) {
  const { mode, setMode } = useMode();

  return (
    <div className="page">
      <h1>CUDA Architecture</h1>
      <div className="mode-toggle" role="group" aria-label="Explanation mode" style={{ margin: "0.5rem 0 1rem" }}>
        <button className={mode === "simple" ? "active" : ""} onClick={() => setMode("simple")}>Beginner</button>
        <button className={mode === "technical" ? "active" : ""} onClick={() => setMode("technical")}>Technical</button>
      </div>

      {mode === "simple" ? (
        <div className="card" style={{ fontSize: "1.05rem" }}>
          <p>An image contains many pixels.</p>
          <p>Many pixel operations can be performed independently.</p>
          <p>CUDA lets us distribute that work across many GPU threads.</p>
        </div>
      ) : (
        <CudaExplained />
      )}

      <WhyGpuVisual />
      <Why question="Why is GPU faster for this specific workload?">
        Each of the five filters computes one output pixel from a small, fixed neighborhood of input pixels,
        completely independently of every other output pixel. That's exactly the pattern GPUs are built for —
        thousands of threads each doing the same small amount of work on a different piece of data.
      </Why>

      <h2>Interactive 3D GPU</h2>
      <Gpu3D />

      <h2>CUDA Threading Hierarchy</h2>
      <ThreadHierarchy3D threading={live?.result?.threading} />

      {mode === "technical" && (
        <div className="card">
          <p style={{ marginTop: 0, fontWeight: 600 }}>Technical mode also covers:</p>
          <p style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
            registers, shared memory, global memory, H2D/D2H transfers, and kernel launch overhead — see the
            Threading &amp; Parallelism page for this machine's real, measured numbers for each of these.
          </p>
        </div>
      )}

      <FinalArchitecture />
    </div>
  );
}
