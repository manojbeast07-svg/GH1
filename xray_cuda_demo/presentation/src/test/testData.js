// Shared fake HISTORICAL dataset for tests -- shaped like the real
// exported JSON (public/data/*.json) but with small, deterministic
// numbers so assertions are exact.
export const FAKE_DATA = {
  benchmark_summary: {
    manifest: { seed: 42, selected_image_count: 122, resolution: [224, 224] },
    cpu: { mode3_pipeline_ms: { median: 30 }, mode4_end_to_end_ms: { median: 90 } },
    basic_cuda: { mode2_gpu_processing_ms: { median: 5 }, mode4_end_to_end_ms: { median: 60 } },
    enhanced_cuda: { mode2_gpu_processing_ms: { median: 3 }, mode4_end_to_end_ms: { median: 58 } },
    speedups: {
      basic_vs_cpu: 1.5,
      enhanced_vs_cpu: 1.55,
      enhanced_vs_basic_compute_only: 2.7,
      enhanced_vs_basic_end_to_end: 1.03,
    },
  },
  per_filter_results: [
    { filter: "gaussian", basic_kernel_ms: 1.9, enhanced_kernel_ms: 0.55, kernel_speedup: 3.45, absolute_reduction_ms: 1.3, pct_of_total_compute_reduction: 37 },
    { filter: "median", basic_kernel_ms: 2.5, enhanced_kernel_ms: 0.45, kernel_speedup: 5.6, absolute_reduction_ms: 2.1, pct_of_total_compute_reduction: 57 },
    { filter: "sobel", basic_kernel_ms: 0.64, enhanced_kernel_ms: 0.7, kernel_speedup: 0.9, absolute_reduction_ms: -0.06, pct_of_total_compute_reduction: -2 },
    { filter: "laplacian", basic_kernel_ms: 0.86, enhanced_kernel_ms: 0.66, kernel_speedup: 1.35, absolute_reduction_ms: 0.2, pct_of_total_compute_reduction: 5 },
    { filter: "threshold", basic_kernel_ms: 0.14, enhanced_kernel_ms: 0.08, kernel_speedup: 1.64, absolute_reduction_ms: 0.06, pct_of_total_compute_reduction: 2 },
  ],
  batch_sweep: [
    { batch_size: 1, cpu_images_per_second: 4000, basic_images_per_second: 5000, enhanced_images_per_second: 4500 },
    { batch_size: 32, cpu_images_per_second: 4200, basic_images_per_second: 8000, enhanced_images_per_second: 9500 },
  ],
  resolution_sweep: [
    { width: 224, height: 224, pixel_count: 50176, cpu_ms_per_image: 2, basic_ms_per_image: 0.5, enhanced_ms_per_image: 0.3 },
  ],
  correctness: {
    canonical_pipeline: {
      basic_vs_cpu: { differing_pixel_percentage: 0.018 },
      enhanced_vs_cpu: { differing_pixel_percentage: 0.018 },
      enhanced_vs_basic: { differing_pixel_percentage: 0 },
    },
    filter_level: {
      gaussian: { tolerance: 1, pass: true },
      median: { tolerance: 0, pass: true },
    },
  },
  system: {
    environment_label: "LOCAL ENVIRONMENT",
    cpu_model: "Test CPU",
    cpu_logical_cores: 16,
    gpu_available: true,
    gpu_name: "Test GPU",
  },
  threading: {
    cpu_logical_processors: 16,
    gpu_sm_count: 24,
    gpu_warp_size: 32,
    gpu_max_threads_per_block: 1024,
    block_dimensions: [16, 16, 1],
    grid_dimensions: [14, 14, 122],
    threads_per_block: 256,
    warps_per_block: 8,
    total_threads_launched: 6121472,
    representative_width: 224,
    representative_height: 224,
    representative_batch_size: 122,
  },
  optimization_results: [
    { id: "persistent_buffers", name: "Persistent GPU Buffers", status: "REJECTED", section: "20B / 20C", what_we_tried: "x", what_happened: "y", why_not_production: "z" },
    { id: "cuda_graph_basic", name: "CUDA Graphs — Basic", status: "EXPERIMENTAL", section: "20D", what_we_tried: "x", what_happened: "y", why_not_production: "z" },
  ],
};

export const EMPTY_DATA = {
  benchmark_summary: null,
  per_filter_results: null,
  batch_sweep: null,
  resolution_sweep: null,
  correctness: null,
  system: null,
  threading: null,
  optimization_results: null,
};

// Mocks the HISTORICAL static-file loader (loadHistoricalData.js) AND a
// generically "available" live server, so page-navigation tests can
// render every page's full content without each needing its own status
// mock. Tests that care about a specific live scenario (unavailable, a
// particular /api/run response, etc.) should set global.fetch themselves.
export function mockFetchWith(dataByFile) {
  global.fetch = vi.fn((url) => {
    const name = Object.keys(dataByFile).find((n) => url.includes(`${n}.json`));
    if (name) {
      const payload = dataByFile[name];
      return Promise.resolve({ ok: payload !== undefined, json: () => Promise.resolve(payload ?? null) });
    }
    if (url.includes("/api/status")) return Promise.resolve({ ok: true, json: () => Promise.resolve({ available: true }) });
    if (url.includes("/api/dataset_info")) return Promise.resolve({ ok: true, json: () => Promise.resolve({ total_files: 9463 }) });
    if (url.includes("/api/system_info")) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          cuda_available: true, environment_label: "LOCAL ENVIRONMENT",
          fingerprint: { cpu_model: "Test CPU", cpu_logical_cores: 16, gpu_name: "Test GPU", gpu_available: true },
          tool_versions: {}, gpu_implementation_facts: { native_extension: true }, gpu_memory: null,
        }),
      });
    }
    return Promise.resolve({ ok: false, json: () => Promise.resolve(null) });
  });
}

// A real LiveRunResult shape (see scripts/presentation_api_server.py's
// /api/run), for tests that exercise LIVE pages (Live Processing,
// Dashboard, Threading, Correctness) without hitting a real server.
export function fakeLiveRunResult(overrides = {}) {
  return {
    run_id: "20260101T000000_000000",
    provenance: "LIVE",
    timestamp_utc: "2026-01-01T00:00:00Z",
    mode: "batch",
    batch_size: 4,
    resolution: [224, 224],
    seed: 42,
    load_ms: 2,
    dataset_total_files: 9463,
    filter_config: { gaussian_enabled: true },
    implementations_requested: { CPU: true, "Basic CUDA": true, "Enhanced CUDA": true },
    implementations: {
      CPU: { total_ms: 9, h2d_ms: null, compute_ms: 9, d2h_ms: null, per_stage_ms: { gaussian: 3, median: 2, sobel: 1, laplacian: 2, threshold: 1 }, images_per_second: 444, preview_png: null },
      "Basic CUDA": { total_ms: 5, h2d_ms: 0.1, compute_ms: 4.5, d2h_ms: 0.4, per_stage_ms: { gaussian: 2, median: 1, sobel: 0.5, laplacian: 0.7, threshold: 0.3 }, images_per_second: 800, preview_png: "data:image/png;base64,AAA" },
      "Enhanced CUDA": { total_ms: 1, h2d_ms: 0.06, compute_ms: 0.8, d2h_ms: 0.14, per_stage_ms: { gaussian: 0.3, median: 0.2, sobel: 0.1, laplacian: 0.15, threshold: 0.05 }, images_per_second: 4000, preview_png: "data:image/png;base64,BBB" },
    },
    speedups_vs_cpu: { "Basic CUDA": 1.8, "Enhanced CUDA": 9.0, "Enhanced vs Basic": 5.0 },
    same_input_verification: { same_input: true, same_configuration: true },
    threading: {
      grid_dimensions: [14, 14, 4], threads_per_block: 256, warps_per_block: 8, total_threads_launched: 200704,
      cpu_logical_processors: 16, gpu_sm_count: 24, gpu_warp_size: 32, gpu_max_threads_per_block: 1024,
      block_dimensions: [16, 16, 1], representative_width: 224, representative_height: 224, representative_batch_size: 4,
    },
    differences: { basic_vs_cpu: "data:image/png;base64,DIFF1", enhanced_vs_cpu: "data:image/png;base64,DIFF2", enhanced_vs_basic: null },
    stage_correctness: [
      { stage: "gaussian", tolerance: 1, max_abs_diff_vs_cpu: 1, status: "PASS" },
      { stage: "median", tolerance: 0, max_abs_diff_vs_cpu: 0, status: "PASS" },
    ],
    pipeline_correctness: [
      { comparison: "basic_vs_cpu", max_abs_diff: 255, mean_abs_diff: 0.03, rmse: 2.9, differing_pixel_count: 27, differing_pixel_percentage: 0.013 },
      { comparison: "enhanced_vs_basic", max_abs_diff: 0, mean_abs_diff: 0, rmse: 0, differing_pixel_count: 0, differing_pixel_percentage: 0 },
    ],
    stale: false,
    ...overrides,
  };
}
