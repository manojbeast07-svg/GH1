import { useEffect } from "react";

// Adds a one-shot "is-visible" class to each section as it first scrolls
// into view, so theme.css can fade/slide it in. Purely a presentation
// effect -- it never affects section identity, active-link tracking
// (useScrollSpy owns that), or any measured data.
export function useRevealOnScroll(sectionIds, scrollRootRef) {
  useEffect(() => {
    if (!scrollRootRef.current) return undefined;
    const root = scrollRootRef.current;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { root, threshold: 0.12 },
    );

    const elements = sectionIds.map((id) => document.getElementById(id)).filter(Boolean);
    elements.forEach((el) => observer.observe(el));

    return () => observer.disconnect();
  }, [sectionIds, scrollRootRef]);
}
