import { useEffect, useState } from "react";

// Tracks which section is currently most visible in the scroll container,
// for sidebar active-link highlighting in the single-scroll layout.
export function useScrollSpy(sectionIds, scrollRootRef) {
  const [activeId, setActiveId] = useState(sectionIds[0]);

  // sectionIds is [] on the very first render (data hasn't loaded yet), so the
  // useState initializer above captures `undefined` -- and since a React state
  // initializer only runs once, activeId would otherwise stay `undefined`
  // forever unless a real IntersectionObserver callback happens to fire first.
  // Re-sync once real section ids exist, so keyboard/jump navigation has a
  // valid starting point even before the first intersection event arrives.
  useEffect(() => {
    setActiveId((prev) => (prev && sectionIds.includes(prev) ? prev : sectionIds[0]));
  }, [sectionIds]);

  useEffect(() => {
    if (!scrollRootRef.current) return undefined;
    const root = scrollRootRef.current;

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
        if (visible.length > 0) {
          setActiveId(visible[0].target.id);
        }
      },
      { root, threshold: [0.2, 0.5, 0.8], rootMargin: "-10% 0px -60% 0px" },
    );

    sectionIds.forEach((id) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });

    return () => observer.disconnect();
  }, [sectionIds, scrollRootRef]);

  return activeId;
}
