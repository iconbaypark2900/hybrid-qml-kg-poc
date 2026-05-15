import { Card, StatusBadge } from "@/components/v2/v2-shell";
import type { V2Candidate } from "@/lib/v2-data";
import type { V2ExperimentEvidence } from "@/lib/v2-live-data";

export function ValidationEvidenceChecklist({
  candidate,
  evidence,
}: {
  candidate: V2Candidate;
  evidence: V2ExperimentEvidence;
}) {
  const checks = [
    {
      label: "Clinical support is explicit",
      passed: !candidate.decisionDetail.includes("No direct"),
      detail: candidate.supportItems[0]?.value ?? "Clinical support was not recorded.",
    },
    {
      label: "Mechanism remains visible",
      passed: candidate.supportItems.length > 1,
      detail: candidate.supportItems[1]?.value ?? candidate.reason,
    },
    {
      label: "Model signal has provenance",
      passed: evidence.predictions.provenance.length > 0,
      detail: evidence.predictions.message ?? "Candidate source should be traceable.",
    },
    {
      label: "Baseline/model comparison exists",
      passed: evidence.methods.data.length > 0,
      detail: evidence.methods.message ?? "Model rows are available for comparison.",
    },
    {
      label: "Weaknesses are documented",
      passed: candidate.weakItems.length > 0,
      detail: candidate.weakItems[0] ?? "No skeptical evidence was recorded.",
    },
  ];

  return (
    <Card
      title="Evidence checklist"
      kicker="Trust gates"
      help="Validation should make the reasons to keep, review, or reject the candidate explicit."
    >
      <div className="space-y-3">
        {checks.map((check) => (
          <div
            key={check.label}
            className="rounded-xl border border-outline-variant/25 bg-surface-container-high/60 p-4"
          >
            <div className="flex items-start justify-between gap-3">
              <p className="font-semibold text-on-surface">{check.label}</p>
              <StatusBadge tone={check.passed ? "success" : "warning"}>
                {check.passed ? "recorded" : "review"}
              </StatusBadge>
            </div>
            <p className="mt-2 text-xs leading-relaxed text-on-surface-variant">
              {check.detail}
            </p>
          </div>
        ))}
      </div>
    </Card>
  );
}
