// Shared by every LIVE and HISTORICAL data-rendering component: a `null`/
// `undefined` value renders as "Not available"/"Not measured" -- 0 is a
// real, present value, never treated as missing.
export function isMissing(value) {
  return value === null || value === undefined;
}
