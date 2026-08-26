import { Pill } from "../components/Pill.jsx";
import { isMissing } from "../data/loadPresentationData.js";

export function Correctness({ correctness }) {
  const filterLevel = correctness?.filter_level ?? {};
  const canonical = correctness?.canonical_pipeline ?? {};

  return (
    <div className="slide">
      <h1>Correctness</h1>
      <p className="subtitle">
        A GPU implementation is only useful if it produces the same result. Every filter is checked against
        both the CPU reference and the other GPU implementation.
      </p>

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

      <p style={{ marginTop: "1rem" }}>
        Tiny floating-point differences can occasionally propagate through thresholding, so correctness is
        evaluated using both numerical difference and differing-pixel percentage, not just a single number.
      </p>
    </div>
  );
}
