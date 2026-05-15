import { ValidationSessionControls } from "@/components/v2/session-controls";
import { Card, StatusBadge } from "@/components/v2/v2-shell";
import type { V2Candidate, V2Session } from "@/lib/v2-data";
import type { V2EvidenceSnapshot } from "@/lib/v2-live-data";
import { validationDecisionLabels, type ValidationDecision } from "@/lib/v2-validation-copy";

export function ValidationDecisionPanel({
  session,
  candidate,
  evidenceSnapshot,
}: {
  session: V2Session;
  candidate: V2Candidate;
  evidenceSnapshot: V2EvidenceSnapshot;
}) {
  return (
    <Card
      title="Decision panel"
      kicker={session.sessionId ? `Session ${session.sessionId}` : "Reviewer decision"}
      help="Capture the researcher decision and notes so the evidence packet can be resumed or exported."
    >
      <div className="mb-4 grid gap-2 sm:grid-cols-3">
        {validationDecisionLabels.map((decision) => (
          <DecisionOption
            key={decision}
            decision={decision}
            active={candidate.decision === decision}
          />
        ))}
      </div>
      <ValidationSessionControls
        session={session}
        evidenceSnapshot={evidenceSnapshot}
      />
    </Card>
  );
}

function DecisionOption({
  decision,
  active,
}: {
  decision: ValidationDecision;
  active: boolean;
}) {
  const tone = decision === "Keep" ? "success" : decision === "Review" ? "warning" : "danger";
  return (
    <div
      className={`rounded-xl border p-3 ${
        active
          ? "border-primary/50 bg-primary/15"
          : "border-outline-variant/25 bg-surface-container-high/60"
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <p className="font-semibold text-on-surface">{decision}</p>
        <StatusBadge tone={tone}>{active ? "current" : "option"}</StatusBadge>
      </div>
      <p className="mt-2 text-xs leading-relaxed text-on-surface-variant">
        {copyFor(decision)}
      </p>
    </div>
  );
}

function copyFor(decision: ValidationDecision) {
  if (decision === "Keep") return "Retain as evidence-backed enough for follow-up.";
  if (decision === "Review") return "Keep visible, but require additional support.";
  return "Do not promote this candidate without new evidence.";
}
