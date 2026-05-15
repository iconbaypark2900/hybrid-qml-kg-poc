export type V2NavItem = {
  href: string;
  label: "Initialize" | "Experiment" | "Validate" | "Visualize" | "Operations";
  short: string;
  icon: string;
  description: string;
};

export const v2NavItems: V2NavItem[] = [
  {
    href: "/v2/start",
    label: "Initialize",
    short: "Init",
    icon: "home",
    description: "Define the research question and run path.",
  },
  {
    href: "/v2/experiment",
    label: "Experiment",
    short: "Exp",
    icon: "science",
    description: "Inspect runs, candidates, and model outputs.",
  },
  {
    href: "/v2/validation",
    label: "Validate",
    short: "Val",
    icon: "fact_check",
    description: "Check whether the evidence can be trusted.",
  },
  {
    href: "/v2/visual",
    label: "Visualize",
    short: "Viz",
    icon: "view_in_ar",
    description: "Explore graph, molecule, embedding, and circuit evidence.",
  },
  {
    href: "/v2/system",
    label: "Operations",
    short: "Ops",
    icon: "monitor_heart",
    description: "Monitor jobs, APIs, runtime credentials, and system health.",
  },
];
