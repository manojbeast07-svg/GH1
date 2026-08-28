const FIELD_LABEL_STYLE = { display: "block", fontSize: "0.78rem", color: "var(--muted)", marginBottom: "var(--space-1)" };
const DETAIL_STYLE = { marginTop: "var(--space-3)", paddingTop: "var(--space-2)", borderTop: "1px dashed var(--border)", display: "flex", flexDirection: "column", gap: "var(--space-2)" };
const INPUT_STYLE = { width: "100%", padding: "0.4rem", marginTop: "0.1rem" };

function Field({ label, children }) {
  return (
    <label style={FIELD_LABEL_STYLE}>
      {label}
      {children}
    </label>
  );
}

// Section 25 spec item 9: full filter configuration control, reused by
// Live Processing and anywhere else a run needs a configurable pipeline.
export function FilterConfigPanel({ config, onChange }) {
  function set(patch) {
    onChange(patch);
  }

  return (
    <div className="card-row cols-2">
      <div className="card">
        <label style={{ display: "flex", alignItems: "center", gap: "var(--space-2)", fontWeight: 600, fontSize: "1.02rem" }}>
          <input type="checkbox" checked={config.gaussian_enabled} onChange={(e) => set({ gaussian_enabled: e.target.checked })} style={{ width: 16, height: 16 }} />
          Gaussian
        </label>
        {config.gaussian_enabled && (
          <div style={DETAIL_STYLE}>
            <Field label="Kernel size">
              <input type="number" min={1} step={2} value={config.gaussian_kernel_size}
                onChange={(e) => set({ gaussian_kernel_size: Number(e.target.value) })} style={INPUT_STYLE} />
            </Field>
            <Field label="Sigma (0 = auto)">
              <input type="number" min={0} step={0.1} value={config.gaussian_sigma}
                onChange={(e) => set({ gaussian_sigma: Number(e.target.value) })} style={INPUT_STYLE} />
            </Field>
          </div>
        )}
      </div>

      <div className="card">
        <label style={{ display: "flex", alignItems: "center", gap: "var(--space-2)", fontWeight: 600, fontSize: "1.02rem" }}>
          <input type="checkbox" checked={config.median_enabled} onChange={(e) => set({ median_enabled: e.target.checked })} style={{ width: 16, height: 16 }} />
          Median
        </label>
        {config.median_enabled && (
          <div style={DETAIL_STYLE}>
            <Field label="Kernel size">
              <input type="number" min={1} step={2} value={config.median_kernel_size}
                onChange={(e) => set({ median_kernel_size: Number(e.target.value) })} style={INPUT_STYLE} />
            </Field>
          </div>
        )}
      </div>

      <div className="card">
        <label style={{ display: "flex", alignItems: "center", gap: "var(--space-2)", fontWeight: 600, fontSize: "1.02rem" }}>
          <input type="checkbox" checked={config.sobel_enabled} onChange={(e) => set({ sobel_enabled: e.target.checked })} style={{ width: 16, height: 16 }} />
          Sobel
        </label>
        {config.sobel_enabled && (
          <div style={DETAIL_STYLE}>
            <Field label="Mode">
              <select value={config.sobel_mode} onChange={(e) => set({ sobel_mode: e.target.value })} style={INPUT_STYLE}>
                <option value="magnitude">magnitude</option>
                <option value="x">x</option>
                <option value="y">y</option>
              </select>
            </Field>
          </div>
        )}
      </div>

      <div className="card">
        <label style={{ display: "flex", alignItems: "center", gap: "var(--space-2)", fontWeight: 600, fontSize: "1.02rem" }}>
          <input type="checkbox" checked={config.laplacian_enabled} onChange={(e) => set({ laplacian_enabled: e.target.checked })} style={{ width: 16, height: 16 }} />
          Laplacian
        </label>
        {config.laplacian_enabled && (
          <div style={DETAIL_STYLE}>
            <Field label="Kernel size">
              <select value={config.laplacian_kernel_size} onChange={(e) => set({ laplacian_kernel_size: Number(e.target.value) })} style={INPUT_STYLE}>
                <option value={1}>1</option>
                <option value={3}>3</option>
                <option value={5}>5</option>
              </select>
            </Field>
            <Field label="Scale">
              <input type="number" step={0.1} value={config.laplacian_scale}
                onChange={(e) => set({ laplacian_scale: Number(e.target.value) })} style={INPUT_STYLE} />
            </Field>
            <Field label="Delta">
              <input type="number" step={0.1} value={config.laplacian_delta}
                onChange={(e) => set({ laplacian_delta: Number(e.target.value) })} style={INPUT_STYLE} />
            </Field>
          </div>
        )}
      </div>

      <div className="card">
        <label style={{ display: "flex", alignItems: "center", gap: "var(--space-2)", fontWeight: 600, fontSize: "1.02rem" }}>
          <input type="checkbox" checked={config.threshold_enabled} onChange={(e) => set({ threshold_enabled: e.target.checked })} style={{ width: 16, height: 16 }} />
          Threshold
        </label>
        {config.threshold_enabled && (
          <div style={DETAIL_STYLE}>
            <Field label="Value">
              <input type="number" min={0} max={255} value={config.threshold_value}
                onChange={(e) => set({ threshold_value: Number(e.target.value) })} style={INPUT_STYLE} />
            </Field>
            <Field label="Max value">
              <input type="number" min={0} max={255} value={config.threshold_max_value}
                onChange={(e) => set({ threshold_max_value: Number(e.target.value) })} style={INPUT_STYLE} />
            </Field>
          </div>
        )}
      </div>
    </div>
  );
}
