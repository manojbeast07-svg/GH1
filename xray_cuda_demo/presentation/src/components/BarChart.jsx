import { isMissing } from "../data/loadPresentationData.js";

const COLORS = { cpu: "var(--cpu)", basic: "var(--basic)", enhanced: "var(--enhanced)" };

// Animated horizontal bars, driven entirely by real numbers passed in
// (spec item 27: "bar lengths must be generated from actual values").
// `bars` = [{ label, value, kind, unit }]. Bar widths are relative to
// the largest value in the set (the slowest bar = 100% width), so a
// smaller bar visually reads as "faster."
export function BarChart({ bars, formatValue }) {
  const known = bars.filter((b) => !isMissing(b.value));
  const max = known.length ? Math.max(...known.map((b) => b.value)) : 1;

  return (
    <div>
      {bars.map((b) => {
        const missing = isMissing(b.value);
        const widthPct = missing ? 0 : Math.max(4, (b.value / max) * 100);
        return (
          <div className="bar-row" key={b.label}>
            <div className="bar-label">{b.label}</div>
            <div className="bar-track">
              {!missing && (
                <div
                  className="bar-fill"
                  style={{ width: `${widthPct}%`, background: COLORS[b.kind] || "var(--muted)" }}
                >
                  {formatValue ? formatValue(b.value) : b.value}
                </div>
              )}
            </div>
            {missing && <span className="data-unavailable" style={{ marginLeft: "0.5rem" }}>Not available</span>}
          </div>
        );
      })}
    </div>
  );
}
