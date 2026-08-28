// Section 25 spec items 51-52: "What just happened?" -- generated ENTIRELY
// from the current LiveRunResult's own measured fields. No hardcoded
// sentence templates that assume an outcome; every claim (faster/slower,
// which stage was largest) is computed from `result` right here.
export function buildRunNarrative(result) {
  if (!result) return [];
  const impls = result.implementations || {};
  const cpu = impls["CPU"];
  const basic = impls["Basic CUDA"];
  const enhanced = impls["Enhanced CUDA"];
  const lines = [];

  const primaryLabel = enhanced ? "Enhanced CUDA" : basic ? "Basic CUDA" : cpu ? "CPU" : null;
  const primary = enhanced || basic || cpu;
  if (!primary) return lines;

  lines.push(`${primaryLabel} took ${primary.total_ms.toFixed(3)} ms for this run.`);

  function compareLine(other, otherLabel) {
    if (!other || other === primary) return;
    const ratio = other.total_ms / primary.total_ms;
    if (ratio >= 1) {
      lines.push(`${primaryLabel} was ${ratio.toFixed(2)}× faster than ${otherLabel} for this run.`);
    } else {
      lines.push(`${primaryLabel} was ${(1 / ratio).toFixed(2)}× slower than ${otherLabel} for this run.`);
    }
  }
  compareLine(cpu, "CPU");
  if (primary !== basic) compareLine(basic, "Basic CUDA");

  const stageSource = enhanced || basic;
  if (stageSource?.per_stage_ms) {
    const entries = Object.entries(stageSource.per_stage_ms).filter(([, v]) => v !== null && v !== undefined);
    if (entries.length) {
      const [stage, ms] = entries.reduce((a, b) => (b[1] > a[1] ? b : a));
      lines.push(`The largest GPU stage this run was ${stage} at ${ms.toFixed(3)} ms.`);
    }
  }

  return lines;
}
