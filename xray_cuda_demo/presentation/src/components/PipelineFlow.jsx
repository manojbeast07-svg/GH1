import { useState } from "react";
import { ProvenanceBadge, SourceTag } from "./ProvenanceBadge.jsx";
import { isMissing } from "../utils/isMissing.js";

const FILTER_STAGES = ["gaussian", "median", "sobel", "laplacian", "threshold"];

// Builds the stage list for one implementation straight from its measured
// timings. A stage that wasn't measured (CPU has no H2D/D2H; a disabled
// filter has no kernel time) is carried through as null and rendered as
// "not applicable" rather than as a zero-width bar pretending it ran.
export function buildPipelineStages(impl) {
  if (!impl) return [];
  const stages = [{ id: "h2d", label: "H2D", ms: impl.h2d_ms, kind: "transfer" }];
  FILTER_STAGES.forEach((id) => {
    stages.push({ id, label: id[0].toUpperCase() + id.slice(1), ms: impl.per_stage_ms?.[id], kind: "kernel" });
  });
  stages.push({ id: "d2h", label: "D2H", ms: impl.d2h_ms, kind: "transfer" });
  return stages;
}

export function PipelineFlow({ result }) {
  const labels = Object.keys(result?.implementations ?? {});
  const [implLabel, setImplLabel] = useState(null);
  const active = implLabel && labels.includes(implLabel) ? implLabel : labels[labels.length - 1];
  const impl = result?.implementations?.[active];
  const [selectedStage, setSelectedStage] = useState(null);

  if (!impl) return null;

  const stages = buildPipelineStages(impl);
  const measured = stages.filter((s) => !isMissing(s.ms));
  const measuredTotal = measured.reduce((acc, s) => acc + s.ms, 0);
  const widest = measured.length ? Math.max(...measured.map((s) => s.ms)) : 0;
  const selected = stages.find((s) => s.id === selectedStage);

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", flexWrap: "wrap", marginBottom: "0.5rem" }}>
        <ProvenanceBadge kind="LIVE" />
        <span style={{ color: "var(--muted)", fontSize: "0.82rem" }}>Execution breakdown for this run</span>
        <div className="mode-toggle" role="group" aria-label="Implementation to break down">
          {labels.map((label) => (
            <button key={label} className={active === label ? "active" : ""} onClick={() => setImplLabel(label)}>{label}</button>
          ))}
        </div>
      </div>

      <div className="pipeline-flow">
        {stages.map((stage) => {
          const unmeasured = isMissing(stage.ms);
          const pct = !unmeasured && widest > 0 ? Math.max(6, (stage.ms / widest) * 100) : 0;
          return (
            <div key={stage.id} className="pipeline-stage-row">
              <button
                className={`pipeline-stage${selectedStage === stage.id ? " active" : ""}${unmeasured ? " unmeasured" : ""}`}
                onClick={() => setSelectedStage(selectedStage === stage.id ? null : stage.id)}
              >
                <span className="pipeline-stage-label">{stage.label}</span>
                <span className="pipeline-stage-track">
                  {!unmeasured && (
                    <span className={`pipeline-stage-fill ${stage.kind}`} style={{ width: `${pct}%` }} />
                  )}
                </span>
                <span className="pipeline-stage-value">
                  {unmeasured ? "n/a" : `${stage.ms.toFixed(3)} ms`}
                </span>
              </button>
            </div>
          );
        })}
      </div>

      <p style={{ color: "var(--muted)", fontSize: "0.78rem", marginTop: "0.5rem" }}>
        Bar lengths are the measured times for this run, relative to the longest stage{" "}
        <SourceTag source="CUDA event timing per stage / CPU timer" />. This is a breakdown of a completed run,
        not a live trace — nothing here was monitored while the kernels were executing. Stages showing
        &ldquo;n/a&rdquo; genuinely had no measurement (CPU performs no host↔device transfers).
      </p>

      {selected && (
        <div className="card" style={{ marginTop: "0.4rem" }}>
          <strong>{selected.label}</strong>
          {isMissing(selected.ms) ? (
            <p style={{ margin: "0.4rem 0 0" }} className="data-unavailable">
              Not measured for {active} — {selected.kind === "transfer"
                ? "this implementation does no host↔device transfer."
                : "this stage is disabled in the current filter configuration."}
            </p>
          ) : (
            <>
              <p style={{ margin: "0.4rem 0 0" }}>
                {selected.ms.toFixed(3)} ms — {((selected.ms / measuredTotal) * 100).toFixed(1)}% of this run's{" "}
                {measuredTotal.toFixed(3)} ms of measured stage time.
              </p>
              {selected.kind === "kernel" && labels.length > 1 && (
                <p style={{ margin: "0.4rem 0 0", color: "var(--muted)", fontSize: "0.82rem" }}>
                  {labels.map((label) => {
                    const v = result.implementations[label]?.per_stage_ms?.[selected.id];
                    return `${label}: ${isMissing(v) ? "n/a" : `${v.toFixed(3)} ms`}`;
                  }).join("  ·  ")}
                </p>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
