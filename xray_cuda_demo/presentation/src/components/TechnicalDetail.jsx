import { useState } from "react";
import { useMode } from "../hooks/useModeContext.jsx";

// Section 24 spec items 22-26, 38: technical detail is hidden by default
// in Simple Mode, and always click-to-reveal even in Technical Mode
// (progressive disclosure -- spec item 1's core instruction: "Technical
// details can appear progressively"). In Technical Mode the section
// starts expanded so a technical audience doesn't need extra clicks.
export function TechnicalDetail({ label = "Technical detail", children }) {
  const { mode } = useMode();
  const [open, setOpen] = useState(mode === "technical");

  return (
    <div className="tech-detail">
      <button className="tech-toggle" onClick={() => setOpen((o) => !o)}>
        {open ? "Hide" : "Show"} {label}
      </button>
      {open && <div style={{ marginTop: "0.6rem" }}>{children}</div>}
    </div>
  );
}
