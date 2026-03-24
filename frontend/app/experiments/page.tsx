import { PagePlaceholder } from "@/components/page-placeholder";

export default function ExperimentsPage() {
  return (
    <PagePlaceholder
      title="Experiments"
      description="Latest pipeline runs and metrics (mockup: experiment_overview)."
    >
      <div className="space-y-3 text-sm text-on-surface-variant">
        <p>
          Planned: <code className="text-on-surface">GET /runs/latest</code>{" "}
          reading{" "}
          <code className="text-on-surface">results/optimized_results_*.json</code>{" "}
          and related CSVs. Until then, run the pipeline from the repo root and
          inspect <code className="text-on-surface">results/</code> locally.
        </p>
        <p>
          See{" "}
          <code className="text-on-surface">docs/frontend/MOCKUP_MAP.md</code>{" "}
          and{" "}
          <code className="text-on-surface">docs/frontend/PIPELINE_UI_FLOW.md</code>
          .
        </p>
      </div>
    </PagePlaceholder>
  );
}
