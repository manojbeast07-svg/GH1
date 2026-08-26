import { isMissing } from "../data/loadPresentationData.js";

// Section 24 spec item 45: the ONE reusable metric-card component, used
// by every slide instead of duplicating markup.
export function MetricCard({ title, value, subtitle }) {
  const display = isMissing(value) ? <span className="data-unavailable">Not available</span> : value;
  return (
    <div className="card metric-card">
      <div className="metric-title">{title}</div>
      <div className="metric-value">{display}</div>
      {subtitle && <div className="metric-subtitle">{subtitle}</div>}
    </div>
  );
}
