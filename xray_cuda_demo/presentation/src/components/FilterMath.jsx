import { useEffect, useState } from "react";
import { fetchFilterMath } from "../data/apiClient.js";
import { ProvenanceBadge } from "./ProvenanceBadge.jsx";

// Computes the filter's arithmetic here in the browser FROM THE REAL
// numbers the backend returned, purely so the steps can be shown. The
// authoritative answer is always `data.output_value` -- what the GPU
// actually produced at this pixel. Where the two differ slightly, that
// difference is shown honestly rather than hidden (see the note in the
// component): the production kernels use a separable two-pass Gaussian,
// float intermediates and their own rounding, so an exact match of the
// recomputation is expected but not guaranteed.
export function explainFilterMath(data) {
  if (!data?.coefficients) return null;
  const { neighborhood, coefficients } = data;

  if (coefficients.kind === "weights") {
    let sum = 0;
    neighborhood.forEach((row, r) => row.forEach((v, c) => { sum += v * coefficients.values[r][c]; }));
    return { kind: "weights", sum };
  }
  if (coefficients.kind === "rank") {
    const sorted = neighborhood.flat().slice().sort((a, b) => a - b);
    return { kind: "rank", sorted, median: sorted[Math.floor(sorted.length / 2)] };
  }
  if (coefficients.kind === "sobel") {
    let gx = 0;
    let gy = 0;
    neighborhood.forEach((row, r) => row.forEach((v, c) => {
      gx += v * coefficients.gx[r][c];
      gy += v * coefficients.gy[r][c];
    }));
    return { kind: "sobel", gx, gy, magnitude: Math.sqrt(gx * gx + gy * gy), absSum: Math.abs(gx) + Math.abs(gy) };
  }
  if (coefficients.kind === "threshold") {
    const passes = data.input_center > coefficients.threshold_value;
    return { kind: "threshold", passes, result: passes ? coefficients.max_value : 0 };
  }
  return null;
}

function intensityStyle(v) {
  const shade = Math.max(0, Math.min(255, v));
  return {
    background: `rgb(${shade * 0.35 + 10}, ${shade * 0.45 + 14}, ${shade * 0.6 + 24})`,
    color: shade > 140 ? "#04140f" : "var(--text)",
  };
}

function Grid({ values, format, cellStyle, highlight }) {
  return (
    <div className="pixel-matrix" style={{ gridTemplateColumns: `repeat(${values[0].length}, 1fr)` }}>
      {values.map((row, r) => row.map((v, c) => {
        const isCenter = r === Math.floor(values.length / 2) && c === Math.floor(row.length / 2);
        return (
          <div
            key={`${r}-${c}`}
            className={`pixel-matrix-cell${isCenter && highlight ? " center" : ""}`}
            style={cellStyle ? cellStyle(v) : undefined}
          >
            {format ? format(v) : v}
          </div>
        );
      }))}
    </div>
  );
}

export function FilterMath({ stage, imageIndex, imageSrc }) {
  const [data, setData] = useState(null);
  const [coord, setCoord] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchFilterMath({ imageIndex, stage, x: coord?.x, y: coord?.y })
      .then((d) => { if (!cancelled) setData(d); })
      .catch((e) => { if (!cancelled) { setError(e.message); setData(null); } })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [stage, imageIndex, coord]);

  function handleImageClick(e) {
    const img = e.currentTarget;
    const rect = img.getBoundingClientRect();
    const x = Math.round(((e.clientX - rect.left) / rect.width) * img.naturalWidth);
    const y = Math.round(((e.clientY - rect.top) / rect.height) * img.naturalHeight);
    setCoord({ x, y });
  }

  const math = explainFilterMath(data);
  const coeffs = data?.coefficients;

  return (
    <div className="card" style={{ marginTop: "1rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.4rem" }}>
        <h4 style={{ margin: 0 }}>The actual arithmetic, at one real pixel</h4>
        <ProvenanceBadge kind="LIVE" />
      </div>
      <p style={{ color: "var(--muted)", fontSize: "0.82rem" }}>
        Click anywhere on the image to pick a pixel. Every number below is real: the neighbourhood values are
        what this stage actually received, the coefficients are the ones the production kernel is actually fed,
        and the result is what the GPU actually wrote at that pixel.
      </p>

      {error && <p style={{ color: "var(--error)", fontSize: "0.82rem" }}>{error}</p>}

      <div className="card-row cols-2">
        <div>
          {imageSrc ? (
            <div style={{ position: "relative", display: "inline-block", width: "100%" }}>
              <img
                src={imageSrc}
                alt={`Input to ${stage} — click to choose a pixel`}
                onClick={handleImageClick}
                style={{ width: "100%", borderRadius: 6, cursor: "crosshair", display: "block", imageRendering: "pixelated" }}
              />
              {data && (
                <span
                  className="pixel-crosshair"
                  style={{ left: `${(data.x / data.width) * 100}%`, top: `${(data.y / data.height) * 100}%` }}
                />
              )}
            </div>
          ) : (
            <p className="data-unavailable">Input image unavailable.</p>
          )}
          {data && (
            <p style={{ color: "var(--muted)", fontSize: "0.78rem", marginTop: "0.4rem" }}>
              Pixel ({data.x}, {data.y}) of {data.width}×{data.height} — input stage: <strong>{data.input_stage}</strong>
              {loading ? " · updating…" : ""}
            </p>
          )}
        </div>

        <div>
          {data ? (
            <>
              <div style={{ fontSize: "0.78rem", color: "var(--muted)", marginBottom: "0.3rem" }}>
                Neighbourhood it read ({data.kernel_size}×{data.kernel_size}), real pixel values
              </div>
              <Grid values={data.neighborhood} cellStyle={intensityStyle} highlight />

              {coeffs?.kind === "weights" && (
                <>
                  <div style={{ fontSize: "0.78rem", color: "var(--muted)", margin: "0.6rem 0 0.3rem" }}>
                    Coefficients it multiplied by
                  </div>
                  <Grid values={coeffs.values} format={(v) => (Math.abs(v) < 0.01 ? v.toFixed(0) : v.toFixed(3))} highlight />
                </>
              )}

              {coeffs?.kind === "sobel" && (
                <div className="card-row cols-2" style={{ marginTop: "0.6rem" }}>
                  <div>
                    <div style={{ fontSize: "0.78rem", color: "var(--muted)", marginBottom: "0.3rem" }}>Gx</div>
                    <Grid values={coeffs.gx} format={(v) => v.toFixed(0)} />
                  </div>
                  <div>
                    <div style={{ fontSize: "0.78rem", color: "var(--muted)", marginBottom: "0.3rem" }}>Gy</div>
                    <Grid values={coeffs.gy} format={(v) => v.toFixed(0)} />
                  </div>
                </div>
              )}

              {coeffs?.source && (
                <p style={{ color: "var(--muted)", fontSize: "0.72rem", marginTop: "0.4rem" }}>
                  Source: <code>{coeffs.source}</code>
                </p>
              )}
            </>
          ) : (
            !error && <p className="data-unavailable">Loading real pixel data…</p>
          )}
        </div>
      </div>

      {data && math && (
        <div className="card" style={{ marginTop: "0.8rem" }}>
          <div style={{ fontWeight: 600, marginBottom: "0.4rem" }}>What that works out to</div>

          {math.kind === "weights" && (
            <p style={{ margin: 0 }}>
              Multiply each of the {data.kernel_size * data.kernel_size} values by its coefficient and add them
              up: <strong>{math.sum.toFixed(2)}</strong>.
            </p>
          )}

          {math.kind === "rank" && (
            <>
              <p style={{ margin: "0 0 0.4rem" }}>
                Median sorts the {math.sorted.length} values and takes the middle one — no multiplication at all:
              </p>
              <div className="sorted-values">
                {math.sorted.map((v, i) => (
                  <span key={i} className={`sorted-value${i === Math.floor(math.sorted.length / 2) ? " median" : ""}`}>{v}</span>
                ))}
              </div>
              <p style={{ margin: "0.4rem 0 0" }}>Middle value: <strong>{math.median}</strong></p>
            </>
          )}

          {math.kind === "sobel" && (
            <p style={{ margin: 0 }}>
              Gx = <strong>{math.gx.toFixed(1)}</strong>, Gy = <strong>{math.gy.toFixed(1)}</strong> →{" "}
              {coeffs.mode === "magnitude"
                ? <>magnitude √(Gx²+Gy²) = <strong>{math.magnitude.toFixed(2)}</strong></>
                : coeffs.mode === "absolute_sum"
                  ? <>|Gx|+|Gy| = <strong>{math.absSum.toFixed(2)}</strong></>
                  : <>mode <strong>{coeffs.mode}</strong></>}
              . A flat area gives ~0 (no edge); a strong edge gives a large value.
            </p>
          )}

          {math.kind === "threshold" && (
            <p style={{ margin: 0 }}>
              This pixel is <strong>{data.input_center}</strong>, the threshold is{" "}
              <strong>{coeffs.threshold_value}</strong> → {math.passes ? "above" : "not above"} it, so the output
              is <strong>{math.result}</strong>.
            </p>
          )}

          <div className="card-row cols-2" style={{ marginTop: "0.6rem" }}>
            <div className="metric-card card">
              <div className="metric-title">Recomputed here</div>
              <div className="metric-value" style={{ fontSize: "1.3rem" }}>
                {math.kind === "weights" ? math.sum.toFixed(2)
                  : math.kind === "rank" ? math.median
                    : math.kind === "sobel" ? (coeffs.mode === "absolute_sum" ? math.absSum : math.magnitude).toFixed(2)
                      : math.result}
              </div>
            </div>
            <div className="metric-card card">
              <div className="metric-title">What the GPU actually wrote</div>
              <div className="metric-value" style={{ fontSize: "1.3rem", color: "var(--enhanced)" }}>{data.output_value}</div>
            </div>
          </div>

          <p style={{ color: "var(--muted)", fontSize: "0.78rem", marginTop: "0.5rem", marginBottom: 0 }}>
            The GPU value on the right is the ground truth — it is read back from the real pipeline output. The
            left-hand figure is this same arithmetic redone in your browser from the values above, so you can
            follow it; small differences are normal, since the production kernels clamp and round to 8-bit and
            Gaussian runs as two separable passes rather than one 2-D sum.
          </p>
        </div>
      )}
    </div>
  );
}
