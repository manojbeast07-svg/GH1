export function Opening() {
  return (
    <div className="slide" style={{ textAlign: "center", paddingTop: "3rem" }}>
      <h1 style={{ fontSize: "3rem" }}>CUDA X-RAY PROCESSING LAB</h1>
      <p className="subtitle" style={{ fontSize: "1.3rem" }}>
        From Python/OpenCV<br />to Custom C++ CUDA<br />to Optimized CUDA
      </p>
      <div className="card-row cols-3" style={{ marginTop: "2.5rem" }}>
        <div className="card">
          <div className="pill cpu">CPU</div>
          <h3 style={{ marginTop: "0.6rem" }}>OpenCV</h3>
        </div>
        <div className="card">
          <div className="pill basic">Basic GPU</div>
          <h3 style={{ marginTop: "0.6rem" }}>Custom C++ CUDA</h3>
        </div>
        <div className="card">
          <div className="pill enhanced">Enhanced GPU</div>
          <h3 style={{ marginTop: "0.6rem" }}>Optimized C++ CUDA</h3>
        </div>
      </div>
      <p className="subtitle" style={{ marginTop: "2.5rem" }}>
        We process the same X-ray images three different ways and measure what changes.
      </p>
    </div>
  );
}
