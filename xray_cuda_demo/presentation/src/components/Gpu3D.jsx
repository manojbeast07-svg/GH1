import { useEffect, useRef, useState, lazy, Suspense } from "react";
import { fetchLaunchConfig } from "../data/apiClient.js";
import { ProvenanceBadge } from "./ProvenanceBadge.jsx";
import { isWebglAvailable } from "../three/webglSupport.js";

const GpuChipScene = lazy(() => import("../three/GpuChipScene.jsx").then((m) => ({ default: m.GpuChipScene })));

const PRESETS = { perspective: [4, 3, 6], top: [0, 8, 0.01], front: [0, 0, 8] };

// Section 25A: a real WebGL scene (React Three Fiber, lazy-loaded so its
// cost is only paid when this page is actually visited) with a full CSS
// 2D fallback -- spec item 54: the textual/2D equivalent must be visible
// even when 3D is disabled, not just as an error message. The SM count
// itself is read live from this machine's actual device query in both
// branches; only the rendering technique differs.
function FallbackGrid({ count, hoveredSm, setHoveredSm, setSelectedSm }) {
  const cols = count ? Math.ceil(Math.sqrt(count * 1.6)) : 6;
  return (
    <div className="gpu3d-die-2d" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
      {Array.from({ length: count }).map((_, i) => (
        <div
          key={i}
          className={`gpu3d-sm${hoveredSm === i ? " hovered" : ""}`}
          onPointerEnter={() => setHoveredSm(i)}
          onPointerLeave={() => setHoveredSm((h) => (h === i ? null : h))}
          onClick={() => setSelectedSm(i)}
          title={`SM ${i}`}
        >
          SM
        </div>
      ))}
    </div>
  );
}

export function Gpu3D() {
  const [smCount, setSmCount] = useState(null);
  const [webgl] = useState(isWebglAvailable);
  const [hoveredSm, setHoveredSm] = useState(null);
  const [selectedSm, setSelectedSm] = useState(null);
  const [autoRotate, setAutoRotate] = useState(false);
  const presetRef = useRef(null);

  useEffect(() => {
    fetchLaunchConfig({ width: 224, height: 224, batchSize: 1, blockX: 16, blockY: 16 })
      .then((config) => setSmCount(config?.gpu_sm_count ?? null))
      .catch(() => setSmCount(null));
  }, []);

  const count = smCount ?? 0;

  return (
    <div className="card gpu3d-card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.4rem" }}>
        <h3 style={{ margin: 0 }}>Interactive GPU{webgl && smCount ? " (drag to rotate, scroll to zoom)" : ""}</h3>
        {smCount != null && <ProvenanceBadge kind="LIVE" />}
      </div>
      <p style={{ color: "var(--muted)", fontSize: "0.82rem" }}>
        Each tile is one Streaming Multiprocessor (SM) — the hardware unit that schedules and executes CUDA
        thread blocks. Hover a tile for what it is; click one for details.
      </p>

      {webgl && smCount ? (
        <>
          <div style={{ display: "flex", gap: "0.4rem", marginBottom: "0.6rem", flexWrap: "wrap" }}>
            <button className="tech-toggle" onClick={() => { presetRef.current = PRESETS.perspective; }}>Perspective</button>
            <button className="tech-toggle" onClick={() => { presetRef.current = PRESETS.top; }}>Top</button>
            <button className="tech-toggle" onClick={() => { presetRef.current = PRESETS.front; }}>Front</button>
            <button className={`tech-toggle${autoRotate ? " active" : ""}`} onClick={() => setAutoRotate((a) => !a)}>
              {autoRotate ? "Stop Auto-Rotate" : "Auto Rotate"}
            </button>
          </div>
          <div className="gpu3d-canvas-wrap">
            <Suspense fallback={<p className="data-unavailable">Loading 3D scene…</p>}>
              <GpuChipScene
                smCount={count} hoveredIndex={hoveredSm} onHover={setHoveredSm}
                onLeave={() => setHoveredSm(null)} onSelect={setSelectedSm}
                autoRotate={autoRotate} presetRef={presetRef}
              />
            </Suspense>
          </div>
        </>
      ) : (
        <FallbackGrid count={count} hoveredSm={hoveredSm} setHoveredSm={setHoveredSm} setSelectedSm={setSelectedSm} />
      )}

      <p style={{ textAlign: "center", color: "var(--muted)", fontSize: "0.85rem", marginTop: "0.6rem" }}>
        {smCount != null
          ? `ACTUAL: this GPU has ${smCount} Streaming Multiprocessors.${hoveredSm != null ? ` Hovering SM ${hoveredSm}.` : ""}`
          : "Live SM count unavailable — start scripts/presentation_api_server.py to see this machine's real value."}
      </p>

      {selectedSm != null && (
        <div className="card" style={{ marginTop: "0.6rem" }}>
          <strong>SM #{selectedSm}</strong>
          <p style={{ margin: "0.4rem 0" }}>
            Streaming Multiprocessor — a hardware execution unit that schedules and executes CUDA thread blocks.
          </p>
          <p className="data-unavailable" style={{ margin: 0 }}>
            Per-SM utilization: Not measured (would require profiler integration this app does not have).
          </p>
        </div>
      )}
    </div>
  );
}
