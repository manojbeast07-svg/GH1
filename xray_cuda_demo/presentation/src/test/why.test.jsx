import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { Why } from "../components/Why.jsx";

describe("Why (quick-explainer control)", () => {
  it("starts collapsed and reveals its answer on click", () => {
    render(<Why question="Why is the sky blue?"><p>Rayleigh scattering.</p></Why>);
    expect(screen.queryByText("Rayleigh scattering.")).not.toBeInTheDocument();
    fireEvent.click(screen.getByText(/Why is the sky blue\?/));
    expect(screen.getByText("Rayleigh scattering.")).toBeInTheDocument();
  });

  it("toggles closed again on a second click", () => {
    render(<Why question="Q"><p>Answer</p></Why>);
    const toggle = screen.getByText(/Q/);
    fireEvent.click(toggle);
    expect(screen.getByText("Answer")).toBeInTheDocument();
    fireEvent.click(toggle);
    expect(screen.queryByText("Answer")).not.toBeInTheDocument();
  });
});
