import { useState } from "react";
import { ProvenanceBadge } from "./ProvenanceBadge.jsx";

const LEVELS = [
  { id: "gpu", name: "GPU", desc: "The complete processor executing CUDA workloads." },
  { id: "sm", name: "SM", desc: "A hardware execution unit that schedules and executes thread blocks." },
  { id: "block", name: "Thread Block", desc: "A group of CUDA threads scheduled together onto one SM." },
  { id: "warp", name: "Warp", desc: "A group of CUDA threads executed together by the GPU hardware." },
  { id: "thread", name: "Thread", desc: "A single CUDA execution context used to process part of the workload." },
];

// Section 25 spec items 29/33/34: depth is used LITERALLY to communicate
// containment (GPU contains SMs, a block runs on an SM, a block contains
// warps, a warp contains threads) -- each level renders visually behind
// the previous one, not as decoration. Real numbers only appear when a
// `threading` payload from an actual run/query is supplied; otherwise the
// level shows its concept-only explanation, never an invented number.
export function ThreadHierarchy3D({ threading }) {
  const [active, setActive] = useState("gpu");
  const activeIdx = LEVELS.findIndex((l) => l.id === active);
  const activeLevel = LEVELS[activeIdx];

  const REAL_VALUE = {
    gpu: threading?.gpu_sm_count != null ? `${threading.gpu_sm_count} SMs on this GPU` : null,
    sm: threading?.gpu_warp_size != null ? `warp size ${threading.gpu_warp_size} threads` : null,
    block: threading?.threads_per_block != null ? `${threading.threads_per_block} threads/block for this run's launch geometry` : null,
    warp: threading?.warps_per_block != null ? `${threading.warps_per_block} warps/block for this run` : null,
    thread: threading?.total_threads_launched != null
      ? `${threading.total_threads_launched.toLocaleString("en-US")} total logical threads launched this run`
      : null,
  };

  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>CUDA Threading Hierarchy</h3>
      <p style={{ color: "var(--muted)", fontSize: "0.82rem" }}>Click a level — each one is literally rendered behind the last, since it lives inside it.</p>

      <div className="hierarchy3d-stage">
        {LEVELS.map((l, i) => (
          <button
            key={l.id}
            className={`hierarchy3d-layer${active === l.id ? " active" : ""}`}
            style={{ transform: `translateZ(${-i * 45}px) translateY(${i * 14}px)`, zIndex: LEVELS.length - i }}
            onClick={() => setActive(l.id)}
          >
            {l.name}
          </button>
        ))}
      </div>

      <div className="card" style={{ marginTop: "0.8rem" }}>
        <strong>{activeLevel.name}</strong>
        <p style={{ margin: "0.4rem 0" }}>{activeLevel.desc}</p>
        {REAL_VALUE[active] ? (
          <p style={{ margin: 0 }}><ProvenanceBadge kind="LIVE" /> {REAL_VALUE[active]}</p>
        ) : (
          <p className="data-unavailable" style={{ margin: 0 }}>No live value yet — run something on Live Processing or the Launch Configuration Explorer.</p>
        )}
      </div>
    </div>
  );
}
