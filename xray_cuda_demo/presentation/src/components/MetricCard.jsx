import { useEffect, useRef, useState } from "react";
import { isMissing } from "../utils/isMissing.js";

const KIND_CLASS = { cpu: "cpu", basic: "basic", enhanced: "enhanced", live: "live", historical: "historical" };

// The ONE reusable metric-card component, used everywhere a single
// measured value is shown, so its treatment applies to every metric in the
// app without touching each call site's own markup.
//
// It used to tilt in 3D under the pointer. That is gone: a tile whose job
// is to state a measured number should not move while someone is reading
// it, and on a projector the rotation just blurred the digits. `kind` now
// shows up as a solid coloured rule down the leading edge (see
// .metric-card-3d in theme.css), which survives being seen from across a
// room. The one motion left is a brief tint when the value itself
// changed, so a re-measurement is noticed rather than silently swapped.
export function MetricCard({ title, value, subtitle, kind }) {
  const display = isMissing(value) ? <span className="data-unavailable">Not available</span> : value;
  const prevValue = useRef(value);
  const [pulsing, setPulsing] = useState(false);

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

  return (
    <div className={`card metric-card metric-card-3d${kind ? ` ${KIND_CLASS[kind] || ""}` : ""}${pulsing ? " metric-pulse" : ""}`}>
      <div className="metric-title">{title}</div>
      <div className="metric-value">{display}</div>
      {subtitle && <div className="metric-subtitle">{subtitle}</div>}
    </div>
  );
}
