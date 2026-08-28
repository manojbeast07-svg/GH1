import { useEffect } from "react";

// Shared full-size image viewer -- click any preview thumbnail to inspect
// it, Esc or click-anywhere to close. Used by Filters and Live Processing
// wherever a real (not illustrative) output image is shown.
export function Lightbox({ src, label, onClose }) {
  useEffect(() => {
    function onKey(e) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      onClick={onClose}
      role="dialog"
      aria-label={`${label} — full size`}
      style={{
        position: "fixed", inset: 0, background: "rgba(4,6,12,0.86)", zIndex: 50,
        display: "flex", alignItems: "center", justifyContent: "center", cursor: "zoom-out",
        animation: "lightboxFadeIn 0.15s ease",
      }}
    >
      <img
        src={src}
        alt={`${label} — full size`}
        style={{ maxWidth: "90vw", maxHeight: "88vh", borderRadius: 8, boxShadow: "0 20px 60px -20px rgba(0,0,0,0.7)" }}
      />
      <div style={{ position: "absolute", top: 16, right: 20, color: "#fff", fontSize: "0.85rem" }}>
        Click anywhere or press Esc to close
      </div>
    </div>
  );
}

export function Zoomable({ src, alt, onZoom, children }) {
  return (
    <div onClick={() => onZoom(src)} style={{ cursor: "zoom-in" }} title="Click to zoom in">
      {children ?? <img src={src} alt={alt} style={{ width: "100%", borderRadius: 6 }} />}
    </div>
  );
}
