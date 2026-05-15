import { EvidenceProvenanceSummary } from "@/components/v2/evidence-provenance";
import { Card, RunSummaryCard } from "@/components/v2/v2-shell";
import type { AnalysisSummaryResponse, LatestRunResponse } from "@/lib/api";
import type { EvidenceState } from "@/lib/v2-live-data";

export function RunProvenancePanel({
  latestRun,
  analysis,
}: {
  latestRun: EvidenceState<LatestRunResponse | null>;
  analysis: EvidenceState<AnalysisSummaryResponse | null>;
}) {
  const latestJson = latestRun.data?.latest_json;
  const artifact =
    latestJson?.path ??
    latestRun.data?.latest_csv?.path ??
    latestRun.message ??
    "No live run artifact";

  return (
    <div className="space-y-4">
      <RunSummaryCard
        model={analysis.data?.best_model ?? "Latest run"}
        relation={analysis.data?.relation ?? latestJson?.relation ?? "CtD"}
        backend={backendLabel(latestJson?.config)}
        artifact={artifact}
      />
      <Card title="Evidence provenance" kicker={latestRun.source}>
        <EvidenceProvenanceSummary
          title="Latest run source"
          provenance={latestRun.provenance}
        />
        <div className="mt-3">
          <EvidenceProvenanceSummary
            title="Analysis source"
            provenance={analysis.provenance}
          />
        </div>
        <p className="mt-3 text-xs leading-relaxed text-on-surface-variant">
          {latestRun.source === "live"
            ? "This page is reading current run artifacts through the API."
            : "No live run artifact was available; fallback rows are labeled so they are not confused with fresh evidence."}
        </p>
      </Card>
    </div>
  );
}

function backendLabel(config: Record<string, unknown> | null | undefined) {
  if (!config) return "Not recorded";
  const quantumConfig = config.quantum_config_path;
  if (typeof quantumConfig === "string") return quantumConfig;
  const executionMode = config.execution_mode;
  return typeof executionMode === "string" ? executionMode : "Artifact-backed";
}
