import { useState } from "react";

// Section 25 spec item 50: small "Why?" quick-explainer controls, scattered
// near the metric they explain rather than bundled into one FAQ page --
// each answer is short, concrete, and grounded in this project's own
// measured behavior (not a generic platitude).
export function Why({ question, children }) {
  const [open, setOpen] = useState(false);
  return (
    <span className="why-inline">
      <button className="why-toggle" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        ❓ {question}
      </button>
      {open && <div className="why-answer">{children}</div>}
    </span>
  );
}
