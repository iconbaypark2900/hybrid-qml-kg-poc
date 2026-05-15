import { EvidenceProvenanceSummary } from "@/components/v2/evidence-provenance";
import { ActionLink, Card, StatusBadge } from "@/components/v2/v2-shell";
import { buildV2Params, type V2Candidate, type V2Session } from "@/lib/v2-data";
import type { EvidenceState } from "@/lib/v2-live-data";

export function ExperimentCandidatePanel({
  session,
  predictions,
}: {
  session: V2Session;
  predictions: EvidenceState<V2Candidate[]>;
}) {
  const candidateRows = [
    session.selectedCandidate,
    ...predictions.data.filter(
      (candidate) =>
        candidate.disease !== session.selectedCandidate.disease ||
        candidate.candidate !== session.selectedCandidate.candidate,
    ),
  ];

  return (
    <Card
      title="Candidate panel"
      kicker={predictions.source === "live" ? "Latest predictions" : "Paper-aligned candidates"}
      help="Primary candidate first, then other candidates from live artifacts or the QCE26 fallback set."
    >
      <EvidenceProvenanceSummary
        title="Candidate source"
        provenance={predictions.provenance}
      />
      {predictions.message ? (
        <p className="mt-3 text-xs leading-relaxed text-on-surface-variant">
          {predictions.message}
        </p>
      ) : null}
      <div className="mt-4 space-y-3">
        {candidateRows.map((row, index) => (
          <CandidateCard
            key={`${row.candidate}-${row.disease}`}
            row={row}
            session={session}
            primary={index === 0}
          />
        ))}
      </div>
    </Card>
  );
}

function CandidateCard({
  row,
  session,
  primary,
}: {
  row: V2Candidate;
  session: V2Session;
  primary: boolean;
}) {
  return (
    <div className="rounded-xl border border-outline-variant/25 bg-surface-container-high/60 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="font-headline text-lg font-semibold text-on-surface">
            {row.candidate} to {row.disease}
          </p>
          <p className="mt-1 text-xs leading-relaxed text-on-surface-variant">
            {row.reason}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {primary ? <StatusBadge tone="quantum">Selected</StatusBadge> : null}
          <StatusBadge tone={row.evidencePosture === "Validated" ? "success" : "warning"}>
            {row.evidencePosture}
          </StatusBadge>
        </div>
      </div>
      <div className="mt-4 grid gap-3 md:grid-cols-[120px_1fr_auto] md:items-center">
        <div>
          <p className="font-label text-[0.65rem] font-bold uppercase tracking-widest text-on-surface-variant">
            Score
          </p>
          <p className="mt-1 font-mono text-lg font-semibold text-primary">{row.score}</p>
        </div>
        <p className="text-xs leading-relaxed text-on-surface-variant">
          {row.workingConclusion}
        </p>
        <ActionLink
          href={`/v2/validation${buildV2Params({
            ...session,
            selectedCandidate: row,
          })}`}
          variant={primary ? "primary" : "secondary"}
        >
          Send to Validate
        </ActionLink>
      </div>
    </div>
  );
}
