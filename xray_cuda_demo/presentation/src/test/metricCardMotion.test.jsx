import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { MetricCard } from "../components/MetricCard.jsx";

describe("MetricCard — 3D tilt, colorful accents, motion-on-change", () => {
  it("applies a colored accent class matching the given kind", () => {
    const { container } = render(<MetricCard title="X" value="5.0 ms" kind="enhanced" />);
    expect(container.querySelector(".metric-card-3d.enhanced")).toBeInTheDocument();
  });

  it("renders with no color class when no kind is given, never breaking existing callers", () => {
    const { container } = render(<MetricCard title="X" value="42" />);
    expect(container.querySelector(".metric-card-3d")).toBeInTheDocument();
    expect(container.querySelector(".cpu, .basic, .enhanced, .historical, .live")).not.toBeInTheDocument();
  });

  it("tilts in 3D as the mouse moves across the card, and resets on mouse leave", () => {
    const { container } = render(<MetricCard title="X" value="42" />);
    const card = container.querySelector(".metric-card-3d");
    vi.spyOn(card, "getBoundingClientRect").mockReturnValue({ left: 0, top: 0, width: 100, height: 100, right: 100, bottom: 100 });

    expect(card.style.transform).toContain("rotateX(0deg)");
    fireEvent.mouseMove(card, { clientX: 90, clientY: 10 });
    expect(card.style.transform).not.toContain("rotateX(0deg)");

    fireEvent.mouseLeave(card);
    expect(card.style.transform).toContain("rotateX(0deg)");
    expect(card.style.transform).toContain("rotateY(0deg)");
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
