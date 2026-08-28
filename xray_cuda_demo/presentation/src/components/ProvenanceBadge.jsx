const KIND_CLASS = { LIVE: "live", HISTORICAL: "historical", EXPERIMENTAL: "experimental", PRODUCTION: "production", FALLBACK: "fallback", ERROR: "error-kind" };

// Section 25 spec item 60: LIVE vs HISTORICAL vs EXPERIMENTAL vs PRODUCTION
// visual language -- every result-bearing card must carry one of these so
// a viewer can never mistake a stored artifact for a current execution.
export function ProvenanceBadge({ kind }) {
  return <span className={`pill provenance ${KIND_CLASS[kind] || "experimental"}`}>{kind}</span>;
}

// Spec item 59: metric provenance -- a small, real tooltip naming exactly
// where a displayed number came from (never just "trust me").
export function SourceTag({ source }) {
  return (
    <span className="source-tag" title={`Source: ${source}`} aria-label={`Source: ${source}`}>
      ⓘ
    </span>
  );
}
