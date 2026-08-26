// Client for the local presentation API server (scripts/presentation_api_server.py).
// Unlike loadPresentationData.js (which reads static, pre-exported JSON), this
// triggers a REAL, live CPU/Basic/Enhanced run through the actual production
// pipeline -- every number returned is freshly measured, never cached alongside
// the static benchmark data, and never fabricated if the server is unreachable
// (checkApiStatus()/runLive() surface a clear error instead).

// Uses the page's own hostname rather than a hardcoded 127.0.0.1, so this
// works both when opened locally and when the presentation is served to a
// remote/RDP session and accessed by IP or hostname -- the API call follows
// wherever the page itself was loaded from.
const API_BASE = `http://${window.location.hostname}:5001`;

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
    const res = await fetch(`${API_BASE}/api/dataset_info`, { signal: AbortSignal.timeout(4000) });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function fetchLaunchConfig({ width, height, batchSize, blockX, blockY }) {
  const params = new URLSearchParams({
    width: String(width), height: String(height), batch_size: String(batchSize),
    block_x: String(blockX), block_y: String(blockY),
  });
  const res = await fetch(`${API_BASE}/api/launch_config?${params}`);
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || `Request failed (${res.status})`);
  }
  return data;
}

export async function fetchPreviewStages(imageIndex = 0) {
  const res = await fetch(`${API_BASE}/api/preview_stages?image_index=${imageIndex}`);
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || `Request failed (${res.status})`);
  }
  return data;
}

export async function runLive({ mode, batchSize, seed, imageIndex }) {
  const res = await fetch(`${API_BASE}/api/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      mode,
      batch_size: batchSize,
      seed,
      image_index: imageIndex,
    }),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || `Request failed (${res.status})`);
  }
  return data;
}
