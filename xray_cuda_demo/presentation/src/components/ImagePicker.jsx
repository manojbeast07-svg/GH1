import { useEffect, useState } from "react";
import { fetchImageThumbnails } from "../data/apiClient.js";

const PAGE_SIZE = 24;

// Pick the X-ray to process by looking at it, rather than by typing a
// dataset index. Each tile carries its real dataset index, so filtering
// or paging never changes which file a selection actually runs.
export function ImagePicker({ selectedIndex, onSelect }) {
  const [query, setQuery] = useState("");
  const [start, setStart] = useState(0);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchImageThumbnails({ start, count: PAGE_SIZE, q: query })
      .then((d) => { if (!cancelled) setData(d); })
      .catch((e) => { if (!cancelled) { setError(e.message); setData(null); } })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [start, query]);

  function applyQuery(value) {
    setQuery(value);
    setStart(0); // a new filter starts from its own first page
  }

  const total = data?.total ?? 0;
  // Derived from what the server actually returned, not from PAGE_SIZE --
  // a filtered or final page can legitimately be shorter than a full page.
  const pageEnd = start + (data?.items?.length ?? 0);
  const selectedItem = data?.items?.find((it) => it.index === selectedIndex);

  return (
    <div className="image-picker">
      <div className="image-picker-toolbar">
        <input
          id="image-picker-search"
          type="search"
          value={query}
          onChange={(e) => applyQuery(e.target.value)}
          placeholder="Filter by folder or filename — e.g. fractured"
          aria-label="Filter images"
        />
        <div className="image-picker-paging">
          <button className="tech-toggle" disabled={start === 0 || loading} onClick={() => setStart((s) => Math.max(0, s - PAGE_SIZE))}>
            ◀ Prev
          </button>
          <span className="image-picker-count">
            {total > 0 ? `${start + 1}–${pageEnd} of ${total.toLocaleString("en-US")}` : loading ? "Loading…" : "No matches"}
          </span>
          <button className="tech-toggle" disabled={pageEnd >= total || loading} onClick={() => setStart((s) => s + PAGE_SIZE)}>
            Next ▶
          </button>
        </div>
      </div>

      {error && <p style={{ color: "var(--error)", fontSize: "0.82rem" }}>{error}</p>}

      {data?.items?.length ? (
        <div className="image-picker-grid">
          {data.items.map((item) => (
            <button
              key={item.index}
              className={`image-picker-tile${item.index === selectedIndex ? " selected" : ""}`}
              onClick={() => onSelect(item.index)}
              title={`${item.relative_path} — ${item.width}×${item.height}`}
              aria-pressed={item.index === selectedIndex}
            >
              <img src={item.thumbnail} alt={item.relative_path} loading="lazy" />
              <span className="image-picker-tile-index">#{item.index}</span>
            </button>
          ))}
        </div>
      ) : (
        !error && !loading && <p className="data-unavailable">No images match that filter.</p>
      )}

      <p className="image-picker-selection">
        {selectedItem
          ? <>Selected: <code>{selectedItem.relative_path}</code> <span className="muted-inline">(index {selectedItem.index}, {selectedItem.width}×{selectedItem.height})</span></>
          : <>Selected image index <code>{selectedIndex}</code> <span className="muted-inline">— not on this page; browse or filter to see it.</span></>}
      </p>
    </div>
  );
}
