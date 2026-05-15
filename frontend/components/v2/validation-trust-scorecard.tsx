import { EvidenceCard, StatusBadge } from "@/components/v2/v2-shell";
import type { V2Candidate } from "@/lib/v2-data";
import {
  validationSourceVocabulary,
  validationTrustLabels,
  type ValidationSource,
} from "@/lib/v2-validation-copy";
import type { V2ExperimentEvidence } from "@/lib/v2-live-data";

export function ValidationTrustScorecard({
  candidate,
  evidence,
}: {
  candidate: V2Candidate;
  evidence: V2ExperimentEvidence;
}) {
  const trustItems: Array<{
    label: string;
    source: ValidationSource;
    value: string;
    detail: string;
    tone: "success" | "warning" | "danger" | "quantum";
  }> = [
    {
      label: validationTrustLabels[0],
      source: validationSourceVocabulary[0],
      value: candidate.decisionDetail,
      detail: candidate.supportItems[0]?.value ?? "Clinical support is not recorded.",
      tone: candidate.decisionDetail.includes("No direct") ? "warning" : "success",
    },
    {
      label: validationTrustLabels[1],
      source: validationSourceVocabulary[1],
      value: candidate.supportItems[1]?.label ?? "Mechanism",
      detail: candidate.supportItems[1]?.value ?? candidate.reason,
      tone: "success",
    },
    {
      label: validationTrustLabels[2],
      source: validationSourceVocabulary[2],
      value: candidate.score,
      detail: "The score is a ranking signal and must travel with provenance.",
      tone: "quantum",
    },
    {
      label: validationTrustLabels[3],
      source: validationSourceVocabulary[3],
      value: evidence.methods.source,
      detail:
        evidence.methods.source === "live"
          ? "Model comparison is artifact-backed."
          : "Baseline comparison is using fallback model rows.",
      tone: evidence.methods.source === "live" ? "success" : "warning",
    },
    {
      label: validationTrustLabels[4],
      source: validationSourceVocabulary[4],
      value: evidence.predictions.source,
      detail:
        evidence.predictions.source === "live"
          ? "Candidate evidence came from live prediction artifacts."
          : "Candidate evidence is clearly labeled as fallback or unavailable.",
      tone: evidence.predictions.source === "live" ? "success" : "warning",
    },
  ];

  return (
    <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
      {trustItems.map((item) => (
        <div key={item.label} className="relative">
          <EvidenceCard
            title={item.label}
            value={item.value}
            detail={item.detail}
            tone={item.tone}
          />
          <div className="absolute right-3 top-3">
            <StatusBadge tone="quantum">{item.source}</StatusBadge>
          </div>
        </div>
      ))}
    </div>
  );
}
