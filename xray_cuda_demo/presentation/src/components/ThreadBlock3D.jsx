import { useMemo, useRef, useState, lazy, Suspense } from "react";
import { ProvenanceBadge } from "./ProvenanceBadge.jsx";
import { isWebglAvailable } from "../three/webglSupport.js";

const ThreadBlockScene = lazy(() => import("../three/ThreadBlockScene.jsx").then((m) => ({ default: m.ThreadBlockScene })));

const PRESETS = { perspective: [3, 2, 5], top: [0, 5, 0.01], front: [0, 0, 5] };

// Real CUDA indexing semantics (Section 25A item 6): threadIdx is the 2D
// coordinate within the block; the linear index used for warp assignment
// is threadIdx.y * blockDim.x + threadIdx.x -- exactly how this project's
// production kernels linearize thread index, never an invented mapping.
function describeThread(index, blockX, warpSize) {
  const tx = index % blockX;
  const ty = Math.floor(index / blockX);
  const warp = Math.floor(index / warpSize);
  return { tx, ty, warp };
}

function FallbackGrid({ blockX, blockY, warpSize, showWarps, hoveredId, setHoveredId, setSelectedId }) {
  const count = blockX * blockY;
  return (
    <div className="threadblock2d" style={{ gridTemplateColumns: `repeat(${blockX}, 1fr)` }}>
      {Array.from({ length: count }).map((_, i) => {
        const warpIdx = Math.floor(i / warpSize);
        const hue = showWarps ? (warpIdx * 61) % 360 : 158;
        return (
          <div
            key={i}
            className={`threadblock2d-cell${hoveredId === i ? " hovered" : ""}`}
            style={{ background: hoveredId === i ? "#fff" : `hsl(${hue}, 55%, ${showWarps ? 50 : 45}%)` }}
            onPointerEnter={() => setHoveredId(i)}
            onPointerLeave={() => setHoveredId((h) => (h === i ? null : h))}
            onClick={() => setSelectedId(i)}
          />
        );
      })}
    </div>
  );
}

export function ThreadBlock3D({ blockX = 16, blockY = 16, warpSize = 32 }) {
  const [webgl] = useState(isWebglAvailable);
  const [showWarps, setShowWarps] = useState(false);
  const [autoRotate, setAutoRotate] = useState(false);
  const [hoveredId, setHoveredId] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const presetRef = useRef(null);

  const count = blockX * blockY;
  const threadsPerBlock = count;
  const warpsPerBlock = Math.ceil(threadsPerBlock / warpSize);
  const focused = hoveredId ?? selectedId;
  const focusedInfo = focused != null ? describeThread(focused, blockX, warpSize) : null;

  return (
    <div className="card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.4rem" }}>
        <h3 style={{ margin: 0 }}>Interactive Thread Block — {blockX}×{blockY}</h3>
        <ProvenanceBadge kind="LIVE" />
      </div>
      <p style={{ color: "var(--muted)", fontSize: "0.82rem" }}>
        Each cube conceptually represents one CUDA thread in a {blockX}×{blockY} block. Hover a cube for its
        exact thread/warp indices, computed the same way the production kernels linearize thread index.
      </p>

      <div style={{ display: "flex", gap: "0.4rem", marginBottom: "0.6rem", flexWrap: "wrap" }}>
        <button className={`tech-toggle${showWarps ? " active" : ""}`} onClick={() => setShowWarps((s) => !s)}>
          {showWarps ? "Hide Warps" : "Show Warps"}
        </button>
        {webgl && (
          <>
            <button className="tech-toggle" onClick={() => { presetRef.current = PRESETS.perspective; }}>Perspective</button>
            <button className="tech-toggle" onClick={() => { presetRef.current = PRESETS.top; }}>Top</button>
            <button className="tech-toggle" onClick={() => { presetRef.current = PRESETS.front; }}>Front</button>
            <button className={`tech-toggle${autoRotate ? " active" : ""}`} onClick={() => setAutoRotate((a) => !a)}>
              {autoRotate ? "Stop Auto-Rotate" : "Auto Rotate"}
            </button>
          </>
        )}
      </div>

      {webgl ? (
        <div className="gpu3d-canvas-wrap">
          <Suspense fallback={<p className="data-unavailable">Loading 3D scene…</p>}>
            <ThreadBlockScene
              blockX={blockX} blockY={blockY} warpSize={warpSize} showWarps={showWarps}
              hoveredId={hoveredId} onHover={setHoveredId} onLeave={() => setHoveredId(null)} onSelect={setSelectedId}
              autoRotate={autoRotate} presetRef={presetRef}
            />
          </Suspense>
        </div>
      ) : (
        <FallbackGrid blockX={blockX} blockY={blockY} warpSize={warpSize} showWarps={showWarps} hoveredId={hoveredId} setHoveredId={setHoveredId} setSelectedId={setSelectedId} />
      )}

      <div className="card-row cols-3" style={{ marginTop: "0.6rem" }}>
        <div className="metric-card card"><div className="metric-title">Threads/block (ACTUAL)</div><div className="metric-value">{threadsPerBlock}</div></div>
        <div className="metric-card card"><div className="metric-title">Warps/block (ACTUAL)</div><div className="metric-value">{warpsPerBlock}</div></div>
        <div className="metric-card card">
          <div className="metric-title">Focused thread</div>
          <div className="metric-value" style={{ fontSize: "1.1rem" }}>
            {focusedInfo ? `(${focusedInfo.tx}, ${focusedInfo.ty})` : "—"}
          </div>
          {focusedInfo && <div className="metric-subtitle">Warp {focusedInfo.warp}</div>}
        </div>
      </div>
    </div>
  );
}
