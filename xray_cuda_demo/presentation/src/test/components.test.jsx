import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ModeProvider, useMode } from "../hooks/useModeContext.jsx";
import { TechnicalDetail } from "../components/TechnicalDetail.jsx";
import { BarChart } from "../components/BarChart.jsx";
import { MetricCard } from "../components/MetricCard.jsx";
import { Sidebar } from "../components/Sidebar.jsx";
import { ProvenanceBadge } from "../components/ProvenanceBadge.jsx";

const PAGES = [
  { id: "a", title: "Page A" },
  { id: "b", title: "Page B" },
  { id: "c", title: "Page C" },
];

function ModeReader() {
  const { mode } = useMode();
  return <div data-testid="mode">{mode}</div>;
}

describe("Simple/Technical mode", () => {
  it("defaults to simple mode, and technical detail starts collapsed", () => {
    render(
      <ModeProvider>
        <ModeReader />
        <TechnicalDetail label="detail"><div>secret technical content</div></TechnicalDetail>
      </ModeProvider>,
    );
    expect(screen.getByTestId("mode").textContent).toBe("simple");
    expect(screen.queryByText("secret technical content")).not.toBeInTheDocument();
  });

  it("clicking the Sidebar's Technical Mode button switches global mode", () => {
    render(
      <ModeProvider>
        <ModeReader />
        <Sidebar pages={PAGES} activeId="a" onSelect={() => {}} />
      </ModeProvider>,
    );
    fireEvent.click(screen.getByText("Technical Mode"));
    expect(screen.getByTestId("mode").textContent).toBe("technical");
  });

  it("technical detail can be expanded by clicking its own toggle regardless of mode", () => {
    render(
      <ModeProvider>
        <TechnicalDetail label="detail"><div>secret technical content</div></TechnicalDetail>
      </ModeProvider>,
    );
    fireEvent.click(screen.getByText(/Show detail/));
    expect(screen.getByText("secret technical content")).toBeInTheDocument();
  });
});

describe("performance bars", () => {
  it("computes bar widths proportional to the largest real value, not a fixed scale", () => {
    const { container } = render(
      <BarChart bars={[{ label: "A", value: 100, kind: "cpu" }, { label: "B", value: 50, kind: "basic" }]} formatValue={(v) => `${v}`} />,
    );
    const fills = container.querySelectorAll(".bar-fill");
    expect(fills[0].style.width).toBe("100%");
    expect(fills[1].style.width).toBe("50%");
  });

  it("shows 'Not available' for a missing bar value instead of a zero-width fabricated bar", () => {
    render(<BarChart bars={[{ label: "Missing", value: null, kind: "cpu" }]} />);
    expect(screen.getByText("Not available")).toBeInTheDocument();
  });
});

describe("MetricCard", () => {
  it("renders a real value", () => {
    render(<MetricCard title="X" value="42" />);
    expect(screen.getByText("42")).toBeInTheDocument();
  });
  it("renders 'Not available' for a null value, never fabricates a number", () => {
    render(<MetricCard title="X" value={null} />);
    expect(screen.getByText("Not available")).toBeInTheDocument();
  });
});

describe("navigation (Sidebar)", () => {
  it("lists every page and marks the active one", () => {
    render(<ModeProvider><Sidebar pages={PAGES} activeId="b" onSelect={() => {}} /></ModeProvider>);
    expect(screen.getByText("Page A")).toBeInTheDocument();
    expect(screen.getByText("Page B")).toBeInTheDocument();
    expect(screen.getByText("Page C")).toBeInTheDocument();
    expect(screen.getByText("Page B").closest("button").className).toContain("active");
  });

  it("calls onSelect with the page id when a link is clicked", () => {
    const onSelect = vi.fn();
    render(<ModeProvider><Sidebar pages={PAGES} activeId="a" onSelect={onSelect} /></ModeProvider>);
    fireEvent.click(screen.getByText("Page C"));
    expect(onSelect).toHaveBeenCalledWith("c");
  });

  it("shows an optional status label", () => {
    render(<ModeProvider><Sidebar pages={PAGES} activeId="a" onSelect={() => {}} statusLabel="Loading historical data…" /></ModeProvider>);
    expect(screen.getByText("Loading historical data…")).toBeInTheDocument();
  });
});

describe("ProvenanceBadge", () => {
  it("renders the LIVE/HISTORICAL/EXPERIMENTAL/PRODUCTION label text", () => {
    render(<ProvenanceBadge kind="LIVE" />);
    expect(screen.getByText("LIVE")).toBeInTheDocument();
  });
});
