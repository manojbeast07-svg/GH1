import { TechnicalDetail } from "../components/TechnicalDetail.jsx";
import { MetricCard } from "../components/MetricCard.jsx";
import { LaunchConfigExplorer } from "../components/LaunchConfigExplorer.jsx";
import { ThreadComparison } from "../components/ThreadComparison.jsx";
import { ThreadBlock3D } from "../components/ThreadBlock3D.jsx";
import { ProvenanceBadge, SourceTag } from "../components/ProvenanceBadge.jsx";
import { isMissing } from "../utils/isMissing.js";

export function Threading({ live }) {
  const { result } = live;
  const threading = result?.threading;

  return (
    <div className="page">
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

      {threading ? (
        <>
          <h2>Live Threading Summary — this run <SourceTag source="current live run" /></h2>
          <ProvenanceBadge kind="LIVE" />
          <div className="card-row cols-4" style={{ marginTop: "0.6rem" }}>
            <MetricCard title="CPU Logical Processors" value={threading.cpu_logical_processors} />
            <MetricCard title="GPU SMs" value={threading.gpu_sm_count} />
            <MetricCard title="Warp Size" value={threading.gpu_warp_size} />
            <MetricCard title="Max Threads/Block" value={threading.gpu_max_threads_per_block} />
          </div>
          <div className="card-row cols-4">
            <MetricCard title="Block" value={threading.block_dimensions ? threading.block_dimensions.slice(0, 2).join(" × ") : null} />
            <MetricCard title="Grid" value={threading.grid_dimensions ? threading.grid_dimensions.join(" × ") : null} />
            <MetricCard title="Threads/Block (configured)" value={threading.threads_per_block} />
            <MetricCard
              title="Total Logical Threads Launched (configured)"
              value={isMissing(threading.total_threads_launched) ? null : threading.total_threads_launched.toLocaleString("en-US")}
              subtitle={`for ${threading.representative_width}×${threading.representative_height}, batch=${threading.representative_batch_size}`}
            />
          </div>

          <h3>CPU vs. GPU thread count — this run</h3>
          <div className="card-row cols-3">
            <MetricCard title="CPU logical processors" kind="cpu" value={threading.cpu_logical_processors} />
            <MetricCard
              title="GPU logical threads launched" kind="enhanced"
              value={isMissing(threading.total_threads_launched) ? null : threading.total_threads_launched.toLocaleString("en-US")}
            />
            <MetricCard
              title="Ratio"
              value={!isMissing(threading.cpu_logical_processors) && !isMissing(threading.total_threads_launched)
                ? `${Math.round(threading.total_threads_launched / threading.cpu_logical_processors).toLocaleString("en-US")}×`
                : null}
              subtitle="GPU threads per CPU logical processor"
            />
          </div>
          <div className="card">
            <p style={{ margin: 0 }}>
              This run divided the workload into many GPU threads that were scheduled across the GPU's SMs — far
              more concurrent execution contexts than this CPU's {threading.cpu_logical_processors} logical
              processors, though (see below) that is a launch-configuration count, not a measurement of how many
              were truly resident on the GPU at once.
            </p>
          </div>
        </>
      ) : (
        <p className="data-unavailable" style={{ marginTop: "1rem" }}>
          Run something on the Live Processing page to see this run's actual threading summary here.
        </p>
      )}

      <h2>Interactive 3D Thread Block</h2>
      <ThreadBlock3D
        blockX={threading?.block_dimensions?.[0] ?? 16}
        blockY={threading?.block_dimensions?.[1] ?? 16}
        warpSize={threading?.gpu_warp_size ?? 32}
      />
      {threading ? (
        <p style={{ color: "var(--muted)", fontSize: "0.82rem" }}>Showing this run's actual launch geometry.</p>
      ) : (
        <p style={{ color: "var(--muted)", fontSize: "0.82rem" }}>
          Showing the production default (16×16, warp size 32) — run something on Live Processing to see your
          actual launch geometry here instead.
        </p>
      )}

      <h2>Interactive Launch Configuration Explorer</h2>
      <LaunchConfigExplorer width={224} height={224} />

      <h2>Compare Two Thread Configurations</h2>
      <ThreadComparison />

      <TechnicalDetail label="technical detail">
        <div className="card">
          <p style={{ marginTop: 0, fontWeight: 600 }}>Important distinction: configured vs. measured</p>
          <p style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
            "Total Logical Threads Launched" above is the launch <em>configuration</em> — grid × block, computed
            from the actual kernel launch parameters. It is never called "active threads": how many of those
            logical threads are truly concurrently resident on the GPU at any instant depends on occupancy,
            which this application does not claim to measure (see below).
          </p>
        </div>
        <div className="card-row cols-2" style={{ marginTop: "0.6rem" }}>
          <MetricCard title="Occupancy" value="Not measured" subtitle="Requires profiler data this app does not collect" />
          <MetricCard title="GPU Utilization" value="Not measured" subtitle="Requires a live monitoring tool this app does not run" />
        </div>
      </TechnicalDetail>
    </div>
  );
}
