import { useState } from "react";
import { Pill } from "../components/Pill.jsx";

const STATUS_MAP = { REJECTED: "error", EXPERIMENTAL: "experimental" };

export function WhatFailed({ optimizationResults }) {
  const [selected, setSelected] = useState(optimizationResults?.[0]?.id ?? null);
  const rows = optimizationResults ?? [];
  const item = rows.find((r) => r.id === selected);

  return (
    <div className="slide">
      <h1>What Didn't Work?</h1>
      <p className="subtitle">
        Not every optimization that looks promising in isolation survives contact with the real application.
        This is honest engineering.
      </p>

      <div className="card-row cols-1" style={{ gridTemplateColumns: "1fr" }}>
        {rows.map((r) => (
          <div
            key={r.id}
            className="card"
            style={{ display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer", marginBottom: "0.5rem", borderColor: selected === r.id ? "var(--enhanced)" : undefined }}
            onClick={() => setSelected(r.id)}
          >
            <span>{r.name}</span>
            <Pill status={STATUS_MAP[r.status] || "warning"}>{r.status}</Pill>
          </div>
        ))}
      </div>

      {item && (
        <div className="card" style={{ marginTop: "1rem" }}>
          <h3 style={{ marginTop: 0 }}>{item.name} <span style={{ color: "var(--muted)", fontSize: "0.8rem" }}>(Section {item.section})</span></h3>
          <p><strong>What we tried:</strong> {item.what_we_tried}</p>
          <p><strong>What happened:</strong> {item.what_happened}</p>
          <p><strong>Why not production:</strong> {item.why_not_production}</p>
        </div>
      )}

      <h2>The engineering lesson</h2>
      <div className="card">
        <p style={{ fontSize: "1.1rem" }}>
          The fastest-looking optimization in a microbenchmark is not necessarily the best optimization for the
          real application.
        </p>
        <p style={{ color: "var(--muted)" }}>
          Persistent GPU buffers looked like a clear 1.15×-1.62× win in a repeated-identical-call microbenchmark
          — but measured ~28% <em>slower</em> in a realistic workload where batch size, parameters, and the
          selected image change between calls, which is how the real Streamlit application is actually used.
          Because we measured the realistic case too, that regression was caught before it ever reached
          production.
        </p>
      </div>
    </div>
  );
}
