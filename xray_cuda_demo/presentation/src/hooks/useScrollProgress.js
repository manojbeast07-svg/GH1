import { useEffect, useState } from "react";

// Tracks how far the visitor has scrolled through the single-scroll page,
// as a 0-100 percentage, for the sidebar's progress bar.
export function useScrollProgress(scrollRootRef) {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const el = scrollRootRef.current;
    if (!el) return undefined;

    function update() {
      const max = el.scrollHeight - el.clientHeight;
      setProgress(max > 0 ? Math.min(100, Math.max(0, (el.scrollTop / max) * 100)) : 0);
    }

    update();
    el.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    return () => {
      el.removeEventListener("scroll", update);
      window.removeEventListener("resize", update);
    };
  }, [scrollRootRef]);

  return progress;
}
