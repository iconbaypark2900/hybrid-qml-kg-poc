import type { V2Candidate } from "@/lib/v2-data";

export function EvidenceAxis({ candidate }: { candidate: V2Candidate }) {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      <AxisCard
        title="Model score axis"
        value={candidate.score}
        badge="ranking signal"
        body="This is the model-produced score from the current evidence/run context. It ranks hypotheses; it is not clinical proof."
      />
      <AxisCard
        title="Clinical validation axis"
        value={candidate.decisionDetail}
        badge="separate evidence"
        body={`${candidate.supportItems[0]?.value ?? "Clinical support is not recorded."} This axis should be reviewed separately from model score.`}
      />
    </div>
  );
}

function AxisCard({
  title,
  value,
  badge,
  body,
}: {
  title: string;
  value: string;
  badge: string;
  body: string;
}) {
  return (
    <div className="rounded-lg border border-outline-variant/25 bg-surface-container-high/60 p-4">
      <div className="flex items-center justify-between gap-3">
        <p className="font-semibold text-on-surface">{title}</p>
        <span className="rounded-full border border-primary/30 bg-primary/10 px-2 py-1 text-xs font-semibold text-primary">
          {badge}
        </span>
      </div>
      <p className="mt-3 font-mono text-sm text-primary">{value}</p>
      <p className="mt-2 text-xs leading-relaxed text-on-surface-variant">
        {body}
      </p>
    </div>
  );
}
