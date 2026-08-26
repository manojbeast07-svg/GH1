import { createContext, useContext, useState } from "react";

// Section 24 spec item 38: global Simple/Technical mode toggle. Context
// so any slide/component can read it without prop-drilling through the
// whole slide tree.
const ModeContext = createContext({ mode: "simple", setMode: () => {} });

export function ModeProvider({ children }) {
  const [mode, setMode] = useState("simple"); // "simple" | "technical"
  return <ModeContext.Provider value={{ mode, setMode }}>{children}</ModeContext.Provider>;
}

export function useMode() {
  return useContext(ModeContext);
}
