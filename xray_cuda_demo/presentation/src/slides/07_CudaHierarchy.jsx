import { useState } from "react";

const LEVELS = [
  { id: "gpu", name: "GPU", desc: "The whole graphics processing unit — contains everything below." },
  { id: "sm", name: "SM (Streaming Multiprocessor)", desc: "A hardware unit on the GPU that executes groups of threads." },
  { id: "block", name: "Thread Block", desc: "A group of CUDA threads scheduled together onto one SM." },
  { id: "warp", name: "Warp", desc: "A group of CUDA threads executed together by the GPU hardware." },
  { id: "thread", name: "Thread", desc: "The smallest unit of work — one lightweight execution context." },
];

export function CudaHierarchy() {
  const [open, setOpen] = useState(null);

  return (
    <div className="slide">
      <h1>How a GPU Is Organized</h1>
      <p className="subtitle">Click a level below to learn what it means. Start from the top.</p>

      <pre className="diagram" style={{ textAlign: "center" }}>{"GPU\n ↓\nSMs\n ↓\nThread Blocks\n ↓\nWarps\n ↓\nThreads"}</pre>

      <div className="card-row cols-1" style={{ gridTemplateColumns: "1fr" }}>
        {LEVELS.map((l) => (
          <div key={l.id} className="card" style={{ marginBottom: "0.6rem", cursor: "pointer" }} onClick={() => setOpen(open === l.id ? null : l.id)}>
            <strong>{l.name}</strong>
            {open === l.id && <p style={{ margin: "0.5rem 0 0 0", color: "var(--muted)" }}>{l.desc}</p>}
          </div>
        ))}
      </div>
      <p style={{ color: "var(--muted)", fontSize: "0.82rem" }}>
        We won't dwell on the hardware details here — the Threading &amp; Parallelism slide shows the real
        numbers for this exact GPU when you're ready for them.
      </p>
    </div>
  );
}
