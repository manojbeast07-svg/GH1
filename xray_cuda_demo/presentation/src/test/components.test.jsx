import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ModeProvider, useMode } from "../hooks/useModeContext.jsx";
import { TechnicalDetail } from "../components/TechnicalDetail.jsx";
import { BarChart } from "../components/BarChart.jsx";
import { MetricCard } from "../components/MetricCard.jsx";
import { Sidebar } from "../components/Sidebar.jsx";

const SLIDES = [
  { id: "a", title: "Section A" },
  { id: "b", title: "Section B" },
  { id: "c", title: "Section C" },
];

function ModeReader() {
  const { mode } = useMode();
  return <div data-testid="mode">{mode}</div>;
}

describe("6. Simple/Technical mode", () => {
  it("defaults to simple mode, and technical detail starts collapsed", () => {
    render(
      <ModeProvider>
        <ModeReader />
        <TechnicalDetail label="detail">
          <div>secret technical content</div>
        </TechnicalDetail>
      </ModeProvider>,
    );
    expect(screen.getByTestId("mode").textContent).toBe("simple");
    expect(screen.queryByText("secret technical content")).not.toBeInTheDocument();
  });

  it("clicking the Sidebar's Technical Mode button switches global mode", () => {
    render(
      <ModeProvider>
        <ModeReader />
        <Sidebar slides={SLIDES} activeId="a" onJump={() => {}} onFullscreen={() => {}} focusMode={false} onToggleFocus={() => {}} />
      </ModeProvider>,
    );
    fireEvent.click(screen.getByText("Technical Mode"));
    expect(screen.getByTestId("mode").textContent).toBe("technical");
  });

  it("technical detail can be expanded by clicking its own toggle regardless of mode", () => {
    render(
      <ModeProvider>
        <TechnicalDetail label="detail">
          <div>secret technical content</div>
        </TechnicalDetail>
      </ModeProvider>,
    );
    fireEvent.click(screen.getByText(/Show detail/));
    expect(screen.getByText("secret technical content")).toBeInTheDocument();
  });
});

describe("9. performance bars", () => {
  it("computes bar widths proportional to the largest real value, not a fixed scale", () => {
    const { container } = render(
      <BarChart
        bars={[
          { label: "A", value: 100, kind: "cpu" },
          { label: "B", value: 50, kind: "basic" },
        ]}
        formatValue={(v) => `${v}`}
      />,
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

describe("5. navigation (Sidebar)", () => {
  it("lists every section and marks the active one", () => {
    render(<ModeProvider><Sidebar slides={SLIDES} activeId="b" onJump={() => {}} onFullscreen={() => {}} focusMode={false} onToggleFocus={() => {}} /></ModeProvider>);
    expect(screen.getByText("Section A")).toBeInTheDocument();
    expect(screen.getByText("Section B")).toBeInTheDocument();
    expect(screen.getByText("Section C")).toBeInTheDocument();
    const activeButton = screen.getByText("Section B").closest("button");
    expect(activeButton.className).toContain("active");
  });

  it("calls onJump with the section id when a link is clicked", () => {
    const onJump = vi.fn();
    render(<ModeProvider><Sidebar slides={SLIDES} activeId="a" onJump={onJump} onFullscreen={() => {}} focusMode={false} onToggleFocus={() => {}} /></ModeProvider>);
    fireEvent.click(screen.getByText("Section C"));
    expect(onJump).toHaveBeenCalledWith("c");
  });
});

describe("sidebar scroll progress", () => {
  it("reflects the given progress percentage as the fill width", () => {
    render(<ModeProvider><Sidebar slides={SLIDES} activeId="a" onJump={() => {}} onFullscreen={() => {}} focusMode={false} onToggleFocus={() => {}} progress={42} /></ModeProvider>);
    const bar = screen.getByRole("progressbar", { name: "Presentation progress" });
    expect(bar).toHaveAttribute("aria-valuenow", "42");
    expect(bar.querySelector(".sidebar-progress-fill").style.width).toBe("42%");
  });

  it("defaults to 0 when no progress is passed", () => {
    render(<ModeProvider><Sidebar slides={SLIDES} activeId="a" onJump={() => {}} onFullscreen={() => {}} focusMode={false} onToggleFocus={() => {}} /></ModeProvider>);
    expect(screen.getByRole("progressbar", { name: "Presentation progress" })).toHaveAttribute("aria-valuenow", "0");
  });
});

describe("10. focus mode / fullscreen controls", () => {
  it("Focus Mode button calls the toggle handler and label reflects state", () => {
    const onToggleFocus = vi.fn();
    render(<ModeProvider><Sidebar slides={SLIDES} activeId="a" onJump={() => {}} onFullscreen={() => {}} focusMode={false} onToggleFocus={onToggleFocus} /></ModeProvider>);
    const btn = screen.getByRole("button", { name: "Toggle focus mode" });
    expect(btn.textContent).toBe("Focus Mode");
    fireEvent.click(btn);
    expect(onToggleFocus).toHaveBeenCalledTimes(1);
  });

  it("shows 'Show Sidebar' once focus mode is active", () => {
    render(<ModeProvider><Sidebar slides={SLIDES} activeId="a" onJump={() => {}} onFullscreen={() => {}} focusMode={true} onToggleFocus={() => {}} /></ModeProvider>);
    expect(screen.getByRole("button", { name: "Toggle focus mode" }).textContent).toBe("Show Sidebar");
  });

  it("Fullscreen button calls the fullscreen handler", () => {
    const onFullscreen = vi.fn();
    render(<ModeProvider><Sidebar slides={SLIDES} activeId="a" onJump={() => {}} onFullscreen={onFullscreen} focusMode={false} onToggleFocus={() => {}} /></ModeProvider>);
    fireEvent.click(screen.getByRole("button", { name: "Toggle fullscreen" }));
    expect(onFullscreen).toHaveBeenCalledTimes(1);
  });
});
