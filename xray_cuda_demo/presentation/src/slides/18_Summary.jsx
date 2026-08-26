const STEPS = [
  "We have thousands of X-rays.",
  "Each image passes through five processing filters.",
  "CPU/OpenCV works.",
  "GPU can process many pixel operations in parallel.",
  "Basic CUDA moves the pipeline to custom GPU kernels.",
  "We profile each filter.",
  "We optimize the filters differently.",
  "Enhanced CUDA improves the important bottlenecks.",
  "Some optimizations fail and are rejected.",
  "Therefore the final system is based on measurement, not assumptions.",
];

export function Summary() {
  return (
    <div className="slide">
      <h1>Summary</h1>
      <ol style={{ fontSize: "1.1rem", lineHeight: 2 }}>
        {STEPS.map((s, i) => (
          <li key={i}>{s}</li>
        ))}
      </ol>
      <p className="subtitle" style={{ textAlign: "center", marginTop: "2rem" }}>Thank you.</p>
    </div>
  );
}
