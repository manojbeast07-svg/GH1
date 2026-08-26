export function Pill({ status = "experimental", children }) {
  return <span className={`pill ${status}`}>{children}</span>;
}
