import Link from "next/link";

interface NoPipelineResultsCtaProps {
  /** Optional reproducibility details (kept secondary to in-app actions). */
  hint?: React.ReactNode;
}

/**
 * Shared empty-state block for pages that depend on completed pipeline runs.
 * Primary CTA: start a new run in the app.
 * Secondary CTA: check system health.
 * Tertiary: reproducibility details (optional, passed as `hint`).
 */
export function NoPipelineResultsCta({ hint }: NoPipelineResultsCtaProps) {
  return (
    <div className="rounded-lg border border-outline-variant/15 bg-surface-container-high/60 p-5 space-y-4">
      <div>
        <p className="text-sm font-medium text-on-surface">No pipeline results yet</p>
        <p className="mt-1 text-xs text-on-surface-variant">
          Start a new pipeline job to generate results, then return here to explore them.
        </p>
      </div>

      <div className="flex flex-wrap gap-3">
        <Link
          href="/simulation/parameters"
          className="inline-flex items-center gap-1.5 rounded-lg bg-primary px-4 py-2 text-xs font-semibold text-on-primary transition-opacity hover:opacity-90"
        >
          <span className="material-symbols-outlined text-base" aria-hidden>play_arrow</span>
          New run
        </Link>
        <Link
          href="/system"
          className="inline-flex items-center gap-1.5 rounded-lg border border-outline/20 bg-surface-container-lowest px-4 py-2 text-xs font-medium text-on-surface transition-colors hover:bg-surface-container"
        >
          <span className="material-symbols-outlined text-base" aria-hidden>dns</span>
          Check system status
        </Link>
      </div>

      {hint ? (
        <details className="text-xs">
          <summary className="cursor-pointer select-none text-on-surface-variant hover:text-on-surface">
            Reproduce locally (optional)
          </summary>
          <div className="mt-2">{hint}</div>
        </details>
      ) : null}
    </div>
  );
}
