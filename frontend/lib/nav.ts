export type NavSection = {
  heading: string;
  items: { href: string; label: string }[];
};

/** Single sidebar model — see `docs/frontend/ROUTES.md`. */
export const navSections: NavSection[] = [
  {
    heading: "Overview",
    items: [
      { href: "/experiments", label: "Experiments" },
      { href: "/system", label: "System status" },
    ],
  },
  {
    heading: "Explore",
    items: [
      { href: "/knowledge-graph", label: "Knowledge graph" },
      { href: "/quantum", label: "Quantum logic" },
    ],
  },
  {
    heading: "Run",
    items: [
      { href: "/simulation", label: "Simulation" },
      { href: "/simulation/parameters", label: "Parameters" },
      { href: "/molecular-design", label: "Molecular design" },
      { href: "/hypotheses/new", label: "New hypothesis" },
    ],
  },
  {
    heading: "Analysis",
    items: [
      { href: "/visualization", label: "Visualizer" },
      { href: "/analysis/drug-delivery", label: "Drug delivery" },
      { href: "/analysis/next-steps", label: "Next steps" },
      { href: "/export", label: "Export" },
    ],
  },
];
