import { useState } from "react";

const GRID = 10;

export function CudaExplained() {
  const [active, setActive] = useState(new Set());

  function toggle(i) {
    setActive((prev) => {
      const next = new Set(prev);
      next.has(i) ? next.delete(i) : next.add(i);
      return next;
    });
  }

  return (
    <div className="slide">
      <h1>CUDA in One Sentence</h1>
      <div className="card" style={{ fontSize: "1.2rem", textAlign: "center" }}>
        CUDA lets us divide a large amount of compatible work into many GPU threads that execute across the GPU.
      </div>

      <h2>Pixels map to threads</h2>
      <p className="subtitle">Click a few cells — each one conceptually becomes its own CUDA thread.</p>
      <div
        className="pixel-grid card"
        style={{ gridTemplateColumns: `repeat(${GRID}, 1fr)`, maxWidth: 360, margin: "0 auto" }}
      >
        {Array.from({ length: GRID * GRID }).map((_, i) => (
          <button
            key={i}
            className="pixel"
            onClick={() => toggle(i)}
            style={{
              background: active.has(i) ? "var(--enhanced)" : "#0b0e13",
              border: "1px solid var(--border)",
              cursor: "pointer",
              padding: 0,
            }}
            aria-label={`pixel ${i}`}
          />
        ))}
      </div>
      <p style={{ textAlign: "center", color: "var(--muted)" }}>
        {active.size > 0
          ? `${active.size} pixel${active.size === 1 ? "" : "s"} selected → ${active.size} conceptual CUDA thread${active.size === 1 ? "" : "s"}`
          : "Click cells to select pixels."}
      </p>
      <pre className="diagram">{"Pixel 1 → Thread 1\nPixel 2 → Thread 2\nPixel 3 → Thread 3\n..."}</pre>
      <p style={{ color: "var(--muted)", fontSize: "0.82rem" }}>
        This is a simplified visualization. Actual kernels may process data in different patterns — some
        filters (like Threshold's vectorized variant) have one thread handle several pixels at once. See the
        Threading &amp; Parallelism slide for the real, filter-by-filter breakdown.
      </p>
    </div>
  );
}
