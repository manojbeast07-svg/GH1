import { ProvenanceBadge } from "../components/ProvenanceBadge.jsx";
import { WhatFailed } from "../slides/15_WhatFailed.jsx";

export function Research({ historical }) {
  return (
    <div className="page">
      <ProvenanceBadge kind="HISTORICAL" />
      <WhatFailed optimizationResults={historical?.optimization_results} />
    </div>
  );
}
