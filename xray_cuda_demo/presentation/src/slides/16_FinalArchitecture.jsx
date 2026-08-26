const DIAGRAM = `                X-RAY DATASET
                       │
                       ▼
                SAME WORKLOAD
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
      CPU/OpenCV   Basic CUDA   Enhanced CUDA
       Python       C++/.cu       C++/.cu
          │            │            │
          └────────────┼────────────┘
                       ▼
                SAME 5 FILTERS
                       │
                       ▼
                 MEASURE RESULTS`;

export function FinalArchitecture() {
  return (
    <div className="slide">
      <h1>Final Architecture</h1>
      <pre className="diagram" style={{ fontSize: "0.95rem" }}>{DIAGRAM}</pre>

      <div className="card-row cols-2">
        <div className="card">
          <div className="pill pass">Production</div>
          <ul>
            <li>CPU / OpenCV</li>
            <li>Basic CUDA</li>
            <li>Enhanced CUDA</li>
          </ul>
        </div>
        <div className="card">
          <div className="pill experimental">Experimental</div>
          <ul>
            <li>CUDA Graphs</li>
            <li>Async multi-stream pipeline</li>
            <li>Laplacian + Threshold fusion</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
