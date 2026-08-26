import { useMode } from "../hooks/useModeContext.jsx";

export function Sidebar({ slides, activeId, onJump, onFullscreen, focusMode, onToggleFocus, progress = 0 }) {
  const { mode, setMode } = useMode();

  return (
    <nav className="sidebar" aria-label="Section navigation">
      <div className="sidebar-title">CUDA X-Ray Lab</div>

      <div
        className="sidebar-progress"
        role="progressbar"
        aria-label="Presentation progress"
        aria-valuenow={Math.round(progress)}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div className="sidebar-progress-fill" style={{ width: `${progress}%` }} />
      </div>

      <div className="mode-toggle" role="group" aria-label="Explanation mode" style={{ margin: "0.8rem 0" }}>
        <button className={mode === "simple" ? "active" : ""} onClick={() => setMode("simple")}>
          Simple Mode
        </button>
        <button className={mode === "technical" ? "active" : ""} onClick={() => setMode("technical")}>
          Technical Mode
        </button>
      </div>

      <ol className="sidebar-links">
        {slides.map((s, i) => (
          <li key={s.id}>
            <button
              className={activeId === s.id ? "active" : ""}
              onClick={() => onJump(s.id)}
              aria-current={activeId === s.id ? "true" : "false"}
            >
              <span className="sidebar-num">{String(i + 1).padStart(2, "0")}</span> {s.title}
            </button>
          </li>
        ))}
      </ol>

      <div className="sidebar-actions">
        <button onClick={onToggleFocus} aria-label="Toggle focus mode">
          {focusMode ? "Show Sidebar" : "Focus Mode"}
        </button>
        <button onClick={onFullscreen} aria-label="Toggle fullscreen">Fullscreen</button>
      </div>
    </nav>
  );
}
