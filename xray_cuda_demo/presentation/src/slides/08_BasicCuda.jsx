export function BasicCuda() {
  return (
    <div className="slide">
      <h1>Custom C++ CUDA</h1>
      <p className="subtitle">
        Instead of relying on a generic GPU image library, this project implements the image-processing
        kernels directly in CUDA C++.
      </p>

      <pre className="diagram">{"Python\n  ↓\npybind11\n  ↓\nC++ CUDA\n  ↓\nGPU"}</pre>

      <h2>Five hand-written kernels</h2>
      <div className="card-row cols-5">
        {["Gaussian kernel", "Median kernel", "Sobel kernel", "Laplacian kernel", "Threshold kernel"].map((k) => (
          <div className="card" key={k} style={{ textAlign: "center", fontSize: "0.9rem" }}>{k}</div>
        ))}
      </div>
    </div>
  );
}
