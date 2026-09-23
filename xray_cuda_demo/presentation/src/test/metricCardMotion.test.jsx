import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { MetricCard } from "../components/MetricCard.jsx";

describe("MetricCard — implementation accent, motion only on change", () => {
  it("applies a colored accent class matching the given kind", () => {
    const { container } = render(<MetricCard title="X" value="5.0 ms" kind="enhanced" />);
    expect(container.querySelector(".metric-card-3d.enhanced")).toBeInTheDocument();
  });

  it("renders with no color class when no kind is given, never breaking existing callers", () => {
    const { container } = render(<MetricCard title="X" value="42" />);
    expect(container.querySelector(".metric-card-3d")).toBeInTheDocument();
    expect(container.querySelector(".cpu, .basic, .enhanced, .historical, .live")).not.toBeInTheDocument();
  });

  // Replaces an earlier test that asserted the card rotated in 3D under the
  // pointer. That behaviour was deliberately removed: a tile whose job is to
  // state a measured number must hold still while it is being read. This
  // test now pins the absence, so the tilt cannot quietly come back.
  it("does not move under the pointer — no inline transform is ever applied", () => {
    const { container } = render(<MetricCard title="X" value="42" />);
    const card = container.querySelector(".metric-card-3d");

    expect(card.style.transform).toBe("");
    fireEvent.mouseMove(card, { clientX: 90, clientY: 10 });
    expect(card.style.transform).toBe("");
    fireEvent.mouseLeave(card);
    expect(card.style.transform).toBe("");
  });

  it("pulses with real motion when its value changes, but not on first render", () => {
    const { container, rerender } = render(<MetricCard title="X" value="10.000 ms" />);
    expect(container.querySelector(".metric-pulse")).not.toBeInTheDocument();

    rerender(<MetricCard title="X" value="12.500 ms" />);
    expect(container.querySelector(".metric-pulse")).toBeInTheDocument();
  });

  it("does not pulse when the value is unchanged across a re-render", () => {
    const { container, rerender } = render(<MetricCard title="X" value="10.000 ms" subtitle="a" />);
    rerender(<MetricCard title="X" value="10.000 ms" subtitle="b" />);
    expect(container.querySelector(".metric-pulse")).not.toBeInTheDocument();
  });

  it("still shows 'Not available' for a missing value, unaffected by the motion/3D additions", () => {
    render(<MetricCard title="X" value={null} />);
    expect(screen.getByText("Not available")).toBeInTheDocument();
  });
});
