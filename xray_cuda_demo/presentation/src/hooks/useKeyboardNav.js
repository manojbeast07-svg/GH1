import { useEffect } from "react";

// Single-scroll layout: ArrowDown/ArrowRight/Space scroll to the next
// section, ArrowUp/ArrowLeft scroll to the previous one, Home/End jump to
// the first/last section. 'f' toggles fullscreen. Ignores keystrokes
// while the user is typing in an input/textarea.
export function useKeyboardNav({ onNext, onPrev, onHome, onEnd, onToggleFullscreen, onToggleFocus }) {
  useEffect(() => {
    function handler(e) {
      const tag = document.activeElement?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;

      switch (e.key) {
        case "ArrowRight":
        case "ArrowDown":
        case " ":
          e.preventDefault();
          onNext?.();
          break;
        case "ArrowLeft":
        case "ArrowUp":
          e.preventDefault();
          onPrev?.();
          break;
        case "Home":
          e.preventDefault();
          onHome?.();
          break;
        case "End":
          e.preventDefault();
          onEnd?.();
          break;
        case "f":
        case "F":
          onToggleFullscreen?.();
          break;
        case "p":
        case "P":
          onToggleFocus?.();
          break;
        default:
          break;
      }
    }
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onNext, onPrev, onHome, onEnd, onToggleFullscreen, onToggleFocus]);
}
