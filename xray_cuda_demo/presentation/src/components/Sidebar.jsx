import { useMode } from "../hooks/useModeContext.jsx";

export function Sidebar({ pages, activeId, onSelect, statusLabel }) {
  const { mode, setMode } = useMode();

  return (
    <nav className="sidebar" aria-label="Section navigation">
      <div className="sidebar-title">CUDA X-Ray Lab</div>
      {statusLabel && <div className="sidebar-status">{statusLabel}</div>}

      <div className="mode-toggle" role="group" aria-label="Explanation mode" style={{ margin: "0.8rem 0" }}>
        <button className={mode === "simple" ? "active" : ""} onClick={() => setMode("simple")}>Simple Mode</button>
        <button className={mode === "technical" ? "active" : ""} onClick={() => setMode("technical")}>Technical Mode</button>
      </div>

      <ol className="sidebar-links">
        {pages.map((p) => (
          <li key={p.id}>
            <button
              className={activeId === p.id ? "active" : ""}
              onClick={() => onSelect(p.id)}
              aria-current={activeId === p.id ? "true" : "false"}
            >
              {p.title}
            </button>
          </li>
        ))}
      </ol>
    </nav>
  );
}
