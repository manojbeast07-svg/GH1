import { useCallback, useMemo, useRef, useState } from "react";
import { usePresentationData } from "./hooks/usePresentationData.js";
import { ModeProvider } from "./hooks/useModeContext.jsx";
import { useKeyboardNav } from "./hooks/useKeyboardNav.js";
import { useScrollSpy } from "./hooks/useScrollSpy.js";
import { useRevealOnScroll } from "./hooks/useRevealOnScroll.js";
import { useScrollProgress } from "./hooks/useScrollProgress.js";
import { Sidebar } from "./components/Sidebar.jsx";

import { Opening } from "./slides/01_Opening.jsx";
import { Problem } from "./slides/02_Problem.jsx";
import { Pipeline } from "./slides/03_Pipeline.jsx";
import { CPU } from "./slides/04_CPU.jsx";
import { WhyGPU } from "./slides/05_WhyGPU.jsx";
import { CudaExplained } from "./slides/06_CudaExplained.jsx";
import { CudaHierarchy } from "./slides/07_CudaHierarchy.jsx";
import { BasicCuda } from "./slides/08_BasicCuda.jsx";
import { EnhancedCuda } from "./slides/09_EnhancedCuda.jsx";
import { LiveRun } from "./slides/19_LiveRun.jsx";
import { Performance } from "./slides/10_Performance.jsx";
import { BatchExplorer } from "./slides/11_BatchExplorer.jsx";
import { ResolutionExplorer } from "./slides/12_ResolutionExplorer.jsx";
import { Threading } from "./slides/13_Threading.jsx";
import { Correctness } from "./slides/14_Correctness.jsx";
import { WhatFailed } from "./slides/15_WhatFailed.jsx";
import { FinalArchitecture } from "./slides/16_FinalArchitecture.jsx";
import { Quiz } from "./slides/17_Quiz.jsx";
import { Summary } from "./slides/18_Summary.jsx";

function buildSlides(data) {
  return [
    { id: "opening", title: "Opening", render: () => <Opening /> },
    { id: "problem", title: "Problem", render: () => <Problem benchmark={data.benchmark_summary} /> },
    { id: "pipeline", title: "X-ray Pipeline", render: () => <Pipeline /> },
    { id: "cpu", title: "CPU Approach", render: () => <CPU system={data.system} /> },
    { id: "why-gpu", title: "Why GPU?", render: () => <WhyGPU system={data.system} threading={data.threading} /> },
    { id: "cuda-explained", title: "CUDA Explained", render: () => <CudaExplained /> },
    { id: "cuda-hierarchy", title: "GPU Hierarchy", render: () => <CudaHierarchy /> },
    { id: "basic-cuda", title: "Basic CUDA", render: () => <BasicCuda /> },
    { id: "enhanced-cuda", title: "Enhanced CUDA", render: () => <EnhancedCuda perFilter={data.per_filter_results} /> },
    { id: "live-run", title: "Try It Yourself", render: () => <LiveRun /> },
    { id: "performance", title: "Performance", render: () => <Performance benchmark={data.benchmark_summary} /> },
    { id: "batch", title: "Batch Explorer", render: () => <BatchExplorer batchSweep={data.batch_sweep} /> },
    { id: "resolution", title: "Resolution Explorer", render: () => <ResolutionExplorer resolutionSweep={data.resolution_sweep} /> },
    { id: "threading", title: "Threading & Parallelism", render: () => <Threading threading={data.threading} /> },
    { id: "correctness", title: "Correctness", render: () => <Correctness correctness={data.correctness} /> },
    { id: "what-failed", title: "What Didn't Work", render: () => <WhatFailed optimizationResults={data.optimization_results} /> },
    { id: "final-architecture", title: "Final Architecture", render: () => <FinalArchitecture /> },
    { id: "quiz", title: "Quiz", render: () => <Quiz /> },
    { id: "summary", title: "Summary", render: () => <Summary /> },
  ];
}

export default function App() {
  const { data, loading } = usePresentationData();
  const [focusMode, setFocusMode] = useState(false);
  const scrollRef = useRef(null);

  const slides = useMemo(() => (data ? buildSlides(data) : []), [data]);
  const sectionIds = useMemo(() => slides.map((s) => s.id), [slides]);
  const activeId = useScrollSpy(sectionIds, scrollRef);
  useRevealOnScroll(sectionIds, scrollRef);
  const progress = useScrollProgress(scrollRef);

  const jumpTo = useCallback((id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, []);

  const goNext = useCallback(() => {
    const i = sectionIds.indexOf(activeId);
    if (i >= 0 && i < sectionIds.length - 1) jumpTo(sectionIds[i + 1]);
  }, [activeId, sectionIds, jumpTo]);

  const goPrev = useCallback(() => {
    const i = sectionIds.indexOf(activeId);
    if (i > 0) jumpTo(sectionIds[i - 1]);
  }, [activeId, sectionIds, jumpTo]);

  const goHome = useCallback(() => jumpTo(sectionIds[0]), [sectionIds, jumpTo]);
  const goEnd = useCallback(() => jumpTo(sectionIds[sectionIds.length - 1]), [sectionIds, jumpTo]);
  const toggleFocus = useCallback(() => setFocusMode((f) => !f), []);
  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen?.();
    } else {
      document.exitFullscreen?.();
    }
  }, []);

  useKeyboardNav({
    onNext: goNext, onPrev: goPrev, onHome: goHome, onEnd: goEnd,
    onToggleFullscreen: toggleFullscreen, onToggleFocus: toggleFocus,
  });

  if (loading) {
    return (
      <div className="app-shell" style={{ alignItems: "center", justifyContent: "center" }}>
        <p>Loading presentation data…</p>
      </div>
    );
  }

  return (
    <ModeProvider>
      <div className="app-shell app-shell-scroll" data-focus-mode={focusMode ? "true" : "false"}>
        {!focusMode && (
          <Sidebar
            slides={slides}
            activeId={activeId}
            onJump={jumpTo}
            onFullscreen={toggleFullscreen}
            focusMode={focusMode}
            onToggleFocus={toggleFocus}
            progress={progress}
          />
        )}
        <div className="scroll-content" ref={scrollRef}>
          {focusMode && (
            <button className="focus-exit" onClick={toggleFocus} aria-label="Toggle focus mode">
              Show Sidebar
            </button>
          )}
          {slides.map((s) => (
            <section key={s.id} id={s.id} className="scroll-section">
              {s.render()}
            </section>
          ))}
        </div>
      </div>
    </ModeProvider>
  );
}
