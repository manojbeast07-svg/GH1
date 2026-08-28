import { Pill } from "../components/Pill.jsx";
import { ProvenanceBadge } from "../components/ProvenanceBadge.jsx";
import { Why } from "../components/Why.jsx";
import { MetricCard } from "../components/MetricCard.jsx";
import { isMissing } from "../utils/isMissing.js";

function HistoricalCorrectness({ correctness }) {
  const filterLevel = correctness?.filter_level ?? {};
  const canonical = correctness?.canonical_pipeline ?? {};

  return (
    <div className="card-row cols-1" style={{ gridTemplateColumns: "1fr" }}>
      {Object.entries(filterLevel).map(([name, v]) => (
        <div key={name} className="card" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
          <span style={{ textTransform: "capitalize" }}>{name}</span>
          <Pill status={v.pass ? "pass" : "error"}>{v.pass ? "PASS" : "FAIL"}</Pill>
        </div>
      ))}
      <div className="card" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span>Basic vs Enhanced (GPU-to-GPU)</span>
        <Pill status={isMissing(canonical?.enhanced_vs_basic?.differing_pixel_percentage) ? "warning" : canonical.enhanced_vs_basic.differing_pixel_percentage === 0 ? "pass" : "warning"}>
          {isMissing(canonical?.enhanced_vs_basic?.differing_pixel_percentage) ? "Not available" : `${canonical.enhanced_vs_basic.differing_pixel_percentage.toFixed(4)}% differ`}
        </Pill>
      </div>
      <div className="card" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span>CPU vs Basic — final pipeline</span>
        <span>{isMissing(canonical?.basic_vs_cpu?.differing_pixel_percentage) ? <span className="data-unavailable">Not available</span> : `${canonical.basic_vs_cpu.differing_pixel_percentage.toFixed(4)}% of pixels differ`}</span>
      </div>
    </div>
  );
}

function LiveCorrectness({ result }) {
  return (
    <div>
      <div className="card-row cols-1" style={{ gridTemplateColumns: "1fr" }}>
        {result.stage_correctness.map((s) => (
          <div key={s.stage} className="card" style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.4rem" }}>
            <span style={{ textTransform: "capitalize" }}>{s.stage}</span>
            <span>
              {s.status === "N/A" ? "Disabled" : `max diff ${s.max_abs_diff_vs_cpu} (tolerance ${s.tolerance})`}{" "}
              <Pill status={s.status === "PASS" ? "pass" : s.status === "WARNING" ? "warning" : "experimental"}>{s.status}</Pill>
            </span>
          </div>
        ))}
      </div>
      <div className="card-row cols-3">
        {result.pipeline_correctness.map((p) => (
          <MetricCard
            key={p.comparison}
            title={p.comparison.replace(/_/g, " ")}
            kind="live"
            value={`${p.differing_pixel_percentage.toFixed(4)}% differ`}
            subtitle={`${p.differing_pixel_count} pixel(s), RMSE ${p.rmse.toFixed(2)}`}
          />
        ))}
      </div>
    </div>
  );
}

export function Correctness({ live, historical }) {
  const { result } = live;

  return (
    <div className="page">
      <h1>Correctness</h1>
      <p className="subtitle">
        A GPU implementation is only useful if it produces the same result. Every filter is checked against
        both the CPU reference and the other GPU implementation.
      </p>

      <Why question="Why are CPU/GPU outputs slightly different?">
        CPU (OpenCV) and CUDA kernels don't always evaluate floating-point arithmetic in exactly the same order,
        so rounding can differ in the last bit or two of a pixel value. That's normally invisible — but a
        threshold operation converts a continuous value into one of two extremes, so a difference of 1 right at
        the threshold boundary can occasionally flip a whole pixel from 0 to 255. That's expected numerical
        behavior, not a bug — see the aggregate percentages below for how rare it actually is.
      </Why>

      <h2><ProvenanceBadge kind="LIVE" /> This run</h2>
      <div className="section-live">
        {result ? (
          <>
            {result.stale && <p style={{ color: "var(--warning)", fontSize: "0.85rem" }}>Configuration changed — run again to update this section.</p>}
            <LiveCorrectness result={result} />
          </>
        ) : (
          <p className="data-unavailable">Run something on the Live Processing page to see this run's correctness here.</p>
        )}
      </div>

      <h2><ProvenanceBadge kind="HISTORICAL" /> Canonical benchmark</h2>
      <div className="section-historical">
        {historical ? (
          <HistoricalCorrectness correctness={historical} />
        ) : (
          <p className="data-unavailable">Historical correctness data unavailable.</p>
        )}
      </div>

      <p style={{ marginTop: "1rem", color: "var(--muted)", fontSize: "0.85rem" }}>
        The per-filter checks above use a strict, zero-tolerance comparison, so an occasional single-run FAIL
        from normal floating-point/threshold-boundary variance is expected and does not by itself indicate a
        broken pipeline — the "final pipeline" percentages are the more representative signal, since they
        aggregate over every image in the run.
      </p>
    </div>
  );
}
