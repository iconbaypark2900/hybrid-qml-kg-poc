"use client";

import Link from "next/link";

type ContextKey =
  | "predict"
  | "experiments"
  | "analysis"
  | "ranked"
  | "simulation"
  | "export"
  | "knowledge-graph"
  | "quantum"
  | "system";

interface Action {
  href: string;
  label: string;
  icon: string;
}

interface ResearchNextActionsProps {
  context: ContextKey;
  title?: string;
}

const ACTIONS: Record<ContextKey, Action[]> = {
  predict: [
    { href: "/experiments", label: "Latest run & models", icon: "monitoring" },
    { href: "/visualization?tab=predictions", label: "Open prediction charts", icon: "leaderboard" },
    { href: "/hypotheses/new", label: "Rank candidates for a disease", icon: "format_list_numbered" },
    { href: "/simulation/parameters", label: "Run a new experiment", icon: "tune" },
  ],
  experiments: [
    { href: "/visualization?tab=comparison", label: "Compare models in charts", icon: "bar_chart" },
    { href: "/predict", label: "Validate one drug-disease pair", icon: "biotech" },
    { href: "/export", label: "Export artifacts", icon: "download" },
    { href: "/simulation/parameters", label: "Launch follow-up run", icon: "play_circle" },
  ],
  analysis: [
    { href: "/simulation/parameters", label: "Apply changes in New run", icon: "tune" },
    { href: "/quantum", label: "Review simulator vs hardware config", icon: "blur_on" },
    { href: "/visualization?tab=comparison", label: "Inspect model comparison", icon: "bar_chart" },
    { href: "/hypotheses/new", label: "Generate ranked candidates", icon: "format_list_numbered" },
  ],
  ranked: [
    { href: "/predict", label: "Validate a single candidate", icon: "biotech" },
    { href: "/knowledge-graph", label: "Inspect candidate in KG", icon: "hub" },
    { href: "/visualization?tab=predictions", label: "Open ranked prediction view", icon: "leaderboard" },
    { href: "/simulation/parameters", label: "Design next run", icon: "tune" },
  ],
  simulation: [
    { href: "/simulation", label: "Monitor pipeline jobs", icon: "play_circle" },
    { href: "/experiments", label: "Open latest run summary", icon: "monitoring" },
    { href: "/visualization?tab=comparison", label: "Open comparison charts", icon: "bar_chart" },
    { href: "/quantum", label: "Check quantum readiness", icon: "blur_on" },
  ],
  export: [
    { href: "/experiments", label: "Review latest run first", icon: "monitoring" },
    { href: "/visualization", label: "Explore run visuals", icon: "scatter_plot" },
    { href: "/simulation/parameters", label: "Generate new artifacts", icon: "tune" },
    { href: "/system", label: "Check API readiness", icon: "dns" },
  ],
  "knowledge-graph": [
    { href: "/visualization?tab=kggraph", label: "Open KG tab in charts", icon: "hub" },
    { href: "/predict", label: "Validate graph-derived pair", icon: "biotech" },
    { href: "/hypotheses/new", label: "Rank candidates by hypothesis", icon: "format_list_numbered" },
    { href: "/simulation/parameters", label: "Run new graph experiment", icon: "tune" },
  ],
  quantum: [
    { href: "/simulation/parameters?preset=quantum-heavy", label: "Launch quantum-heavy run", icon: "memory" },
    { href: "/visualization?tab=circuit", label: "Open live circuit tab", icon: "blur_on" },
    { href: "/visualization?tab=comparison", label: "Compare quantum vs classical", icon: "bar_chart" },
    { href: "/system", label: "Check backend readiness", icon: "dns" },
  ],
  system: [
    { href: "/simulation/parameters", label: "Start a pipeline run", icon: "tune" },
    { href: "/simulation", label: "Monitor job status", icon: "play_circle" },
    { href: "/experiments", label: "Open latest run results", icon: "monitoring" },
    { href: "/quantum", label: "Verify IBM Quantum access", icon: "blur_on" },
  ],
};

export function ResearchNextActions({
  context,
  title = "Next research actions",
}: ResearchNextActionsProps) {
  const actions = ACTIONS[context] ?? [];
  if (actions.length === 0) return null;

  return (
    <section className="rounded-lg border border-outline-variant/20 bg-surface-container-lowest/60 p-4">
      <h2 className="font-label text-xs font-semibold uppercase tracking-wide text-on-surface-variant">
        {title}
      </h2>
      <div className="mt-3 flex flex-wrap gap-2">
        {actions.map((action) => (
          <Link
            key={`${context}-${action.href}`}
            href={action.href}
            className="inline-flex items-center gap-1.5 rounded-lg border border-outline/20 bg-surface-container-lowest px-3 py-2 text-xs font-medium text-on-surface transition-colors hover:bg-surface-container"
          >
            <span className="material-symbols-outlined text-sm" aria-hidden>
              {action.icon}
            </span>
            {action.label}
          </Link>
        ))}
      </div>
    </section>
  );
}
