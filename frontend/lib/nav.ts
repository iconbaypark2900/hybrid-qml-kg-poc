export type NavSection = {
  heading: string;
  items: { href: string; label: string }[];
};

/**
 * Sidebar navigation — organised around user tasks, not engineering layers.
 *
 * Start    → context + the primary action (prediction)
 * Results  → what the pipeline produced
 * Run      → trigger new work
 * System   → deep-dive / technical
 */
export const navSections: NavSection[] = [
  {
    heading: "Start",
    items: [
      { href: "/", label: "Home" },
      { href: "/predict", label: "Predict treatment" },
    ],
  },
  {
    heading: "Results",
    items: [
      { href: "/experiments", label: "Experiments" },
      { href: "/analysis/drug-delivery", label: "Drug delivery" },
      { href: "/hypotheses/new", label: "Ranked candidates" },
      { href: "/visualization", label: "Visualizer" },
    ],
  },
  {
    heading: "Run",
    items: [
      { href: "/simulation", label: "Pipeline jobs" },
      { href: "/simulation/parameters", label: "New run" },
    ],
  },
  {
    heading: "System",
    items: [
      { href: "/system", label: "System status" },
      { href: "/knowledge-graph", label: "Knowledge graph" },
      { href: "/quantum", label: "Quantum config" },
      { href: "/export", label: "Export" },
    ],
  },
];
