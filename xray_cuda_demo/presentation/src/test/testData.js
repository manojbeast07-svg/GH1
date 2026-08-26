// Shared fake dataset for tests -- shaped like the real exported JSON
// but with small, deterministic numbers so assertions are exact.
export const FAKE_DATA = {
  benchmark_summary: {
    benchmark_id: "test_bench",
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
    { batch_size: 1, cpu_total_ms: 0.2, basic_total_ms: 0.2, enhanced_total_ms: 0.2, cpu_images_per_second: 4000, basic_images_per_second: 5000, enhanced_images_per_second: 4500 },
    { batch_size: 32, cpu_total_ms: 1, basic_total_ms: 0.5, enhanced_total_ms: 0.3, cpu_images_per_second: 4200, basic_images_per_second: 8000, enhanced_images_per_second: 9500 },
  ],
  resolution_sweep: [
    { width: 224, height: 224, pixel_count: 50176, cpu_ms_per_image: 2, basic_ms_per_image: 0.5, enhanced_ms_per_image: 0.3, cpu_images_per_second: 500, basic_images_per_second: 2000, enhanced_images_per_second: 3300 },
  ],
  correctness: {
    canonical_pipeline: {
      basic_vs_cpu: { differing_pixel_percentage: 0.018 },
      enhanced_vs_cpu: { differing_pixel_percentage: 0.018 },
      enhanced_vs_basic: { differing_pixel_percentage: 0 },
    },
    filter_level: {
      gaussian: { tolerance: 1, pass: true, cpu_vs_basic_max_abs_diff: 1, basic_vs_enhanced_max_abs_diff: 0 },
      median: { tolerance: 0, pass: true, cpu_vs_basic_max_abs_diff: 0, basic_vs_enhanced_max_abs_diff: 0 },
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

export function mockFetchWith(dataByFile) {
  global.fetch = vi.fn((url) => {
    const name = Object.keys(dataByFile).find((n) => url.includes(n));
    const payload = name ? dataByFile[name] : null;
    return Promise.resolve({
      ok: payload !== undefined,
      json: () => Promise.resolve(payload ?? null),
    });
  });
}
