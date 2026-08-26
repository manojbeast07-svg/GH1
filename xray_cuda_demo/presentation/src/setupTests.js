import "@testing-library/jest-dom/vitest";

// jsdom implements neither IntersectionObserver (used by useScrollSpy)
// nor Element.scrollIntoView (used by the sidebar's jump-to-section
// navigation) -- both are provided by every real browser this app
// actually runs in, but need a minimal stub here so component tests
// don't crash. The stub never fires callbacks (scrollspy tests instead
// assert against the initial activeId), which is fine since scroll-
// triggered highlighting is a real-browser behavior, not something
// meaningfully testable in jsdom.
if (typeof window !== "undefined") {
  window.IntersectionObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
  if (!window.HTMLElement.prototype.scrollIntoView) {
    window.HTMLElement.prototype.scrollIntoView = function () {};
  }
}
