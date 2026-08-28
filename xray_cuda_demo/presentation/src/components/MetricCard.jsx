import { useEffect, useRef, useState } from "react";
import { isMissing } from "../utils/isMissing.js";

const KIND_CLASS = { cpu: "cpu", basic: "basic", enhanced: "enhanced", live: "live", historical: "historical" };

// The ONE reusable metric-card component, used everywhere a single
// measured value is shown -- enhancing it here (3D tilt, colorful accent,
// pulse-on-change motion) applies the same treatment to every metric in
// the app without touching each call site's own markup. Thread/launch
// configuration detail stays on the dedicated Threading page -- this
// component never renders thread counts itself.
export function MetricCard({ title, value, subtitle, kind }) {
  const display = isMissing(value) ? <span className="data-unavailable">Not available</span> : value;
  const cardRef = useRef(null);
  const prevValue = useRef(value);
  const [pulsing, setPulsing] = useState(false);
  const [tilt, setTilt] = useState({ x: 0, y: 0 });

  useEffect(() => {
    if (prevValue.current !== value && !isMissing(value)) {
      setPulsing(true);
      const t = setTimeout(() => setPulsing(false), 500);
      prevValue.current = value;
      return () => clearTimeout(t);
    }
    prevValue.current = value;
    return undefined;
  }, [value]);

  function handleMouseMove(e) {
    const el = cardRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width - 0.5;
    const py = (e.clientY - rect.top) / rect.height - 0.5;
    setTilt({ x: py * -10, y: px * 10 });
  }

  function handleMouseLeave() {
    setTilt({ x: 0, y: 0 });
  }

  return (
    <div
      ref={cardRef}
      className={`card metric-card metric-card-3d${kind ? ` ${KIND_CLASS[kind] || ""}` : ""}${pulsing ? " metric-pulse" : ""}`}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{ transform: `perspective(700px) rotateX(${tilt.x}deg) rotateY(${tilt.y}deg)` }}
    >
      <div className="metric-title">{title}</div>
      <div className="metric-value">{display}</div>
      {subtitle && <div className="metric-subtitle">{subtitle}</div>}
    </div>
  );
}
