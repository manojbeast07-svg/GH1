// Client for the local presentation API server (scripts/presentation_api_server.py).
// Section 25: this is a complete interactive application, not a slideshow --
// nothing here auto-executes the pipeline. Every function that runs CPU/CUDA
// work is only ever called in response to an explicit user action (a button
// click). checkApiStatus()/fetchDatasetInfo() are cheap, read-only queries and
// are the only ones safe to call automatically.

// Uses the page's own hostname rather than a hardcoded 127.0.0.1, so this
// works both when opened locally and when the presentation is served to a
// remote/RDP session and accessed by IP or hostname -- the API call follows
// wherever the page itself was loaded from.
const API_BASE = `http://${window.location.hostname}:5001`;

async function getJson(path, { signal } = {}) {
  const res = await fetch(`${API_BASE}${path}`, signal ? { signal } : undefined);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

async function postJson(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

export async function checkApiStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/status`, { signal: AbortSignal.timeout(2000) });
    if (!res.ok) return { available: false };
    return await res.json();
  } catch {
    return { available: false };
  }
}

export async function fetchDatasetInfo() {
  try {
    return await getJson("/api/dataset_info", { signal: AbortSignal.timeout(4000) });
  } catch {
    return null;
  }
}

export async function fetchSystemInfo() {
  return getJson("/api/system_info");
}

export async function fetchLaunchConfig({ width, height, batchSize, blockX, blockY }) {
  const params = new URLSearchParams({
    width: String(width), height: String(height), batch_size: String(batchSize),
    block_x: String(blockX), block_y: String(blockY),
  });
  return getJson(`/api/launch_config?${params}`);
}

export async function fetchPreviewStages(imageIndex = 0) {
  return getJson(`/api/preview_stages?image_index=${imageIndex}`);
}

// A browsable page of real dataset thumbnails, so an image can be picked
// visually. `index` on each item is the real dataset index /api/run takes.
export async function fetchImageThumbnails({ start = 0, count = 24, q = "" } = {}) {
  const params = new URLSearchParams({ start: String(start), count: String(count) });
  if (q) params.set("q", q);
  return getJson(`/api/image_thumbnails?${params}`);
}

// The real neighbourhood values, real production coefficients and real
// output value for one filter at one pixel (see /api/filter_math).
export async function fetchFilterMath({ imageIndex = 0, stage, x, y }) {
  const params = new URLSearchParams({ image_index: String(imageIndex), stage });
  if (x != null) params.set("x", String(x));
  if (y != null) params.set("y", String(y));
  return getJson(`/api/filter_math?${params}`);
}

// The core explicit-action endpoint: [Run CPU] / [Run Basic CUDA] /
// [Run Enhanced CUDA] / [Compare All] are all this one call with different
// runCpu/runBasic/runEnhanced flags -- never all three unless requested.
export async function runLive({
  mode, batchSize, seed, imageIndex, runCpu = true, runBasic = true, runEnhanced = true, filterConfig,
}) {
  return postJson("/api/run", {
    mode, batch_size: batchSize, seed, image_index: imageIndex,
    run_cpu: runCpu, run_basic: runBasic, run_enhanced: runEnhanced,
    filter_config: filterConfig,
  });
}

export async function runLivePerFilter({ batchSize, seed, filterConfig }) {
  return postJson("/api/live/per_filter", { batch_size: batchSize, seed, filter_config: filterConfig });
}

export async function runLiveBatchSweep({ batchSizes, seed, filterConfig }) {
  return postJson("/api/live/batch_sweep", { batch_sizes: batchSizes, seed, filter_config: filterConfig });
}

