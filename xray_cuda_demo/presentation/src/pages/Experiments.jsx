import { useEffect, useState } from "react";
import { fetchExperiments, fetchExperiment, saveExperiment } from "../data/apiClient.js";
import { ProvenanceBadge } from "../components/ProvenanceBadge.jsx";

function configsDiffer(a, b) {
  if (!a || !b) return true;
  return JSON.stringify(a) !== JSON.stringify(b);
}

function TimingCompareTable({ timingA, timingB }) {
  const labels = Array.from(new Set([...Object.keys(timingA ?? {}), ...Object.keys(timingB ?? {})]));
  return (
    <table style={{ width: "100%", borderCollapse: "collapse" }}>
      <thead>
        <tr style={{ color: "var(--muted)", fontSize: "0.78rem", textAlign: "right" }}>
          <th style={{ textAlign: "left" }}>Implementation</th><th>A total</th><th>B total</th><th>Δ (B − A)</th>
        </tr>
      </thead>
      <tbody>
        {labels.map((label) => {
          const ta = timingA?.[label]?.total_ms;
          const tb = timingB?.[label]?.total_ms;
          const delta = ta != null && tb != null ? tb - ta : null;
          return (
            <tr key={label} style={{ borderTop: "1px solid var(--border)" }}>
              <td style={{ padding: "0.3rem 0" }}>{label}</td>
              <td style={{ textAlign: "right" }}>{ta != null ? `${ta.toFixed(3)} ms` : "N/A"}</td>
              <td style={{ textAlign: "right" }}>{tb != null ? `${tb.toFixed(3)} ms` : "N/A"}</td>
              <td style={{ textAlign: "right", color: delta == null ? "var(--muted)" : delta > 0 ? "var(--error)" : "var(--pass)" }}>
                {delta == null ? "N/A" : `${delta > 0 ? "+" : ""}${delta.toFixed(3)} ms`}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export function Experiments({ live }) {
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [saveMessage, setSaveMessage] = useState(null);
  const [compareIds, setCompareIds] = useState([null, null]);
  const [compareData, setCompareData] = useState([null, null]);

  function refresh() {
    setLoading(true);
    fetchExperiments().then((d) => setExperiments(d.experiments)).finally(() => setLoading(false));
  }

  useEffect(() => { refresh(); }, []);

  async function handleSaveCurrent() {
    if (!live.result) return;
    try {
      const saved = await saveExperiment(live.result, `Experiments page — ${live.result.mode}`);
      setSaveMessage(`Saved as ${saved.run_id}.`);
      refresh();
    } catch (e) {
      setSaveMessage(`Save failed: ${e.message}`);
    }
  }

  async function loadCompareSlot(slot, runId) {
    // Functional updates -- selecting A then quickly selecting B must
    // never let B's resolved fetch clobber A's slot with a stale
    // snapshot of compareData captured before A's own update landed.
    setCompareIds((prev) => {
      const next = [...prev];
      next[slot] = runId;
      return next;
    });
    if (!runId) {
      setCompareData((prev) => {
        const next = [...prev];
        next[slot] = null;
        return next;
      });
      return;
    }
    const data = await fetchExperiment(runId);
    setCompareData((prev) => {
      const next = [...prev];
      next[slot] = data;
      return next;
    });
  }

  const [a, b] = compareData;
  const differ = a && b && configsDiffer(a.metadata?.filter_config, b.metadata?.filter_config);

  return (
    <div className="page">
      <h1>Experiments</h1>
      <p className="subtitle">Save the current live run, and compare any two saved experiments.</p>

      <div className="card">
        <p style={{ margin: 0 }}>
          {live.result ? `Latest live run: ${live.result.run_id} (${live.result.mode}, ${live.result.resolution[0]}×${live.result.resolution[1]})` : "No live run yet — go to Live Processing first."}
        </p>
        <button className="tech-toggle" onClick={handleSaveCurrent} disabled={!live.result} style={{ marginTop: "0.5rem" }}>💾 Save Current Run</button>
        {saveMessage && <span style={{ marginLeft: "0.6rem", fontSize: "0.82rem", color: "var(--muted)" }}>{saveMessage}</span>}
      </div>

      <h2>Saved Experiments</h2>
      {loading ? (
        <p className="data-unavailable">Loading…</p>
      ) : experiments.length === 0 ? (
        <p className="data-unavailable">No experiments saved yet.</p>
      ) : (
        <div className="card-row cols-1" style={{ gridTemplateColumns: "1fr" }}>
          {experiments.map((e) => (
            <div key={e.run_id} className="card" style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.4rem" }}>
              <span>{e.run_id} — {e.label || e.mode} — {e.batch_size} image(s), {e.resolution ? `${e.resolution[0]}×${e.resolution[1]}` : "?"}</span>
              <span style={{ fontSize: "0.78rem", color: "var(--muted)" }}>{e.saved_at_utc}</span>
            </div>
          ))}
        </div>
      )}

      <h2>Compare Two Experiments</h2>
      <div className="card-row cols-2">
        {[0, 1].map((slot) => (
          <select key={slot} value={compareIds[slot] ?? ""} onChange={(e) => loadCompareSlot(slot, e.target.value || null)} style={{ padding: "0.4rem" }}>
            <option value="">Select experiment {slot === 0 ? "A" : "B"}</option>
            {experiments.map((e) => <option key={e.run_id} value={e.run_id}>{e.run_id}</option>)}
          </select>
        ))}
      </div>

      {a && b && (
        <div className="card" style={{ marginTop: "0.8rem" }}>
          <ProvenanceBadge kind="HISTORICAL" />
          {differ && (
            <p style={{ color: "var(--warning)", fontWeight: 600 }}>
              WARNING: configurations differ — this is not an apples-to-apples comparison.
            </p>
          )}
          <p style={{ color: "var(--muted)", fontSize: "0.78rem" }}>A — {a.metadata?.run_id} · B — {b.metadata?.run_id}</p>
          <TimingCompareTable timingA={a.timing} timingB={b.timing} />
        </div>
      )}
    </div>
  );
}
