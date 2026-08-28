import { useCallback, useState } from "react";
import { useLiveRun } from "./hooks/useLiveRun.js";
import { useHistoricalData } from "./hooks/useHistoricalData.js";
import { ModeProvider } from "./hooks/useModeContext.jsx";
import { useKeyboardNav } from "./hooks/useKeyboardNav.js";
import { Sidebar } from "./components/Sidebar.jsx";

import { Dashboard } from "./pages/Dashboard.jsx";
import { LiveProcessing } from "./pages/LiveProcessing.jsx";
import { Performance } from "./pages/Performance.jsx";
import { Filters } from "./pages/Filters.jsx";
import { CudaArchitecture } from "./pages/CudaArchitecture.jsx";
import { Threading } from "./pages/Threading.jsx";
import { OptimizationLab } from "./pages/OptimizationLab.jsx";
import { Correctness } from "./pages/Correctness.jsx";
import { System } from "./pages/System.jsx";
import { Experiments } from "./pages/Experiments.jsx";
import { Research } from "./pages/Research.jsx";

const PAGES = [
  { id: "dashboard", title: "Dashboard" },
  { id: "live-processing", title: "Live Processing" },
  { id: "performance", title: "Performance" },
  { id: "filters", title: "Filters" },
  { id: "cuda-architecture", title: "CUDA Architecture" },
  { id: "threading", title: "Threading & Parallelism" },
  { id: "optimization-lab", title: "Optimization Lab" },
  { id: "correctness", title: "Correctness" },
  { id: "system", title: "System" },
  { id: "experiments", title: "Experiments" },
  { id: "research", title: "Research / Decisions" },
];

export default function App() {
  const live = useLiveRun();
  const { data: historical, loading: historicalLoading } = useHistoricalData();
  const [activeId, setActiveId] = useState(PAGES[0].id);
  const [focusMode, setFocusMode] = useState(false);

  const goTo = useCallback((id) => setActiveId(id), []);
  const goNext = useCallback(() => {
    const i = PAGES.findIndex((p) => p.id === activeId);
    if (i < PAGES.length - 1) goTo(PAGES[i + 1].id);
  }, [activeId, goTo]);
  const goPrev = useCallback(() => {
    const i = PAGES.findIndex((p) => p.id === activeId);
    if (i > 0) goTo(PAGES[i - 1].id);
  }, [activeId, goTo]);
  const goHome = useCallback(() => goTo(PAGES[0].id), [goTo]);
  const goEnd = useCallback(() => goTo(PAGES[PAGES.length - 1].id), [goTo]);
  const toggleFocus = useCallback(() => setFocusMode((f) => !f), []);
  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen?.();
    } else {
      document.exitFullscreen?.();
    }
  }, []);

  useKeyboardNav({ onNext: goNext, onPrev: goPrev, onHome: goHome, onEnd: goEnd, onToggleFullscreen: toggleFullscreen, onToggleFocus: toggleFocus });

  function renderPage() {
    switch (activeId) {
      case "dashboard": return <Dashboard live={live} onNavigate={goTo} />;
      case "live-processing": return <LiveProcessing live={live} />;
      case "performance": return <Performance historical={historical} />;
      case "filters": return <Filters historical={historical?.per_filter_results} />;
      case "cuda-architecture": return <CudaArchitecture live={live} />;
      case "threading": return <Threading live={live} />;
      case "optimization-lab": return <OptimizationLab />;
      case "correctness": return <Correctness live={live} historical={historical?.correctness} />;
      case "system": return <System />;
      case "experiments": return <Experiments live={live} />;
      case "research": return <Research historical={historical} />;
      default: return null;
    }
  }

  return (
    <ModeProvider>
      <div className="app-shell app-shell-scroll" data-focus-mode={focusMode ? "true" : "false"}>
        {!focusMode && (
          <Sidebar
            pages={PAGES}
            activeId={activeId}
            onSelect={goTo}
            statusLabel={historicalLoading ? "Loading historical data…" : null}
          />
        )}
        <div className="scroll-content">
          <div className="page-container" key={activeId}>
            {renderPage()}
          </div>
          <div style={{ padding: "0 4rem 3rem", display: "flex", gap: "0.5rem" }}>
            <button className="tech-toggle" onClick={toggleFullscreen} aria-label="Toggle fullscreen">Fullscreen</button>
            <button className="tech-toggle" onClick={toggleFocus} aria-label="Toggle focus mode">{focusMode ? "Show Sidebar" : "Focus Mode"}</button>
          </div>
        </div>
      </div>
    </ModeProvider>
  );
}
