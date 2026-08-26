import { TechnicalDetail } from "../components/TechnicalDetail.jsx";
import { MetricCard } from "../components/MetricCard.jsx";
import { LaunchConfigExplorer } from "../components/LaunchConfigExplorer.jsx";
import { isMissing } from "../data/loadPresentationData.js";

export function Threading({ threading }) {
  return (
    <div className="slide">
      <h1>Threading &amp; Parallelism</h1>
      <p className="subtitle">
        CPU processing uses a relatively small number of general-purpose execution units. GPU processing exposes
        much more fine-grained parallelism, allowing a large number of pixel-related operations to be in flight.
      </p>

      <div className="card-row cols-2">
        <div className="card">
          <div className="pill cpu">CPU</div>
          <div style={{ marginTop: "0.6rem", fontFamily: "monospace" }}>
            {"● Core\n● Core\n● Core\n● Core".split("\n").map((l, i) => <div key={i}>{l}</div>)}
          </div>
        </div>
        <div className="card">
          <div className="pill enhanced">GPU</div>
          <pre style={{ marginTop: "0.6rem", color: "var(--enhanced)" }}>{"SM   SM   SM   SM\n│    │    │    │\n↓↓↓  ↓↓↓  ↓↓↓  ↓↓↓\nMany CUDA threads"}</pre>
        </div>
      </div>

      <TechnicalDetail label="technical detail">
        <div className="card-row cols-4">
          <MetricCard title="CPU Logical Processors" value={threading?.cpu_logical_processors ?? null} />
          <MetricCard title="GPU SMs" value={threading?.gpu_sm_count ?? null} />
          <MetricCard title="Warp Size" value={threading?.gpu_warp_size ?? null} />
          <MetricCard title="Max Threads/Block" value={threading?.gpu_max_threads_per_block ?? null} />
        </div>
        <div className="card-row cols-4">
          <MetricCard title="Block" value={threading?.block_dimensions ? threading.block_dimensions.slice(0, 2).join(" × ") : null} />
          <MetricCard title="Grid" value={threading?.grid_dimensions ? threading.grid_dimensions.join(" × ") : null} />
          <MetricCard title="Threads/Block" value={threading?.threads_per_block ?? null} />
          <MetricCard
            title="Total Threads Launched"
            value={isMissing(threading?.total_threads_launched) ? null : threading.total_threads_launched.toLocaleString("en-US")}
            subtitle={threading ? `for ${threading.representative_width}×${threading.representative_height}, batch=${threading.representative_batch_size}` : null}
          />
        </div>
        <p style={{ color: "var(--muted)", fontSize: "0.82rem" }}>
          All values above are read live from this machine's actual GPU and CPU — never estimated.
        </p>

        <LaunchConfigExplorer
          width={threading?.representative_width ?? 224}
          height={threading?.representative_height ?? 224}
        />
      </TechnicalDetail>
    </div>
  );
}
