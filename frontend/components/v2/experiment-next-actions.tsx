import { ActionLink, ActionRail, Card, HelpTooltip, StatusBadge } from "@/components/v2/v2-shell";
import { buildV2Params, type V2Session } from "@/lib/v2-data";
import type { EvidenceState, V2RunJob } from "@/lib/v2-live-data";

export function ExperimentNextActions({
  session,
  jobs,
}: {
  session: V2Session;
  jobs: EvidenceState<V2RunJob[]>;
}) {
  const routeParams = buildV2Params(session);

  return (
    <section className="grid gap-4 xl:grid-cols-[1fr_1fr]">
      <Card
        title="Next actions"
        kicker="Decision flow"
        help="Experiment should end by moving the researcher forward, not leaving them inside a static table."
      >
        <ActionRail>
          <ActionLink href={`/v2/validation${routeParams}`}>
            Send candidate to Validate
          </ActionLink>
          <ActionLink href={`/v2/visual${routeParams}`} variant="secondary">
            Visualize evidence
          </ActionLink>
          <ActionLink href="/simulation/parameters" variant="secondary">
            Launch or monitor run
          </ActionLink>
        </ActionRail>
        <div className="mt-4 space-y-3">
          <NextStep
            number="1"
            title="Validate top candidate"
            body={`Move ${session.selectedCandidate.candidate} to ${session.selectedCandidate.disease} into the trust workflow.`}
          />
          <NextStep
            number="2"
            title="Inspect visual evidence"
            body="Use graph, molecule, embedding, and circuit evidence to explain the ranking."
          />
          <NextStep
            number="3"
            title="Launch follow-up"
            body="Run a new pipeline or hardware check only after the evidence gap is clear."
          />
        </div>
      </Card>

      <Card
        title="Run monitor"
        kicker={jobs.source === "live" ? "Recent jobs" : "Representative states"}
        help="Shows whether recent follow-up work is ready, running, or failed."
      >
        <div className="overflow-x-auto">
          <table className="w-full min-w-[480px] text-sm">
            <thead>
              <tr className="border-b border-outline-variant/30 text-left font-label text-xs uppercase tracking-widest text-on-surface-variant">
                <th className="py-3 pr-4">ID</th>
                <th className="py-3 pr-4">Setup</th>
                <th className="py-3 pr-4">Status</th>
                <th className="py-3 pr-4">Duration</th>
                <th className="py-3 pr-4">
                  Next <HelpTooltip text="The next step once a job is ready. Most successful jobs feed Validate or Visualize." />
                </th>
              </tr>
            </thead>
            <tbody>
              {jobs.data.map(({ id, setup, status, duration, next }) => (
                <tr
                  key={id}
                  className="border-b border-outline-variant/20 last:border-b-0"
                >
                  <td className="py-3 pr-4 font-mono font-semibold text-on-surface">
                    {id}
                  </td>
                  <td className="py-3 pr-4 text-on-surface-variant">{setup}</td>
                  <td className="py-3 pr-4">
                    <StatusBadge tone={toneForJob(status)}>{status}</StatusBadge>
                  </td>
                  <td className="py-3 pr-4 font-mono text-on-surface-variant">
                    {duration}
                  </td>
                  <td className="py-3 pr-4 text-on-surface">{next}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </section>
  );
}

function NextStep({
  number,
  title,
  body,
}: {
  number: string;
  title: string;
  body: string;
}) {
  return (
    <div className="flex gap-3 rounded-xl border border-outline-variant/25 bg-surface-container-high/60 p-3">
      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/15 font-mono text-sm font-bold text-primary">
        {number}
      </span>
      <div>
        <p className="font-semibold text-on-surface">{title}</p>
        <p className="mt-1 text-xs leading-relaxed text-on-surface-variant">{body}</p>
      </div>
    </div>
  );
}

function toneForJob(status: string) {
  if (["completed", "success", "ready"].includes(status)) return "success";
  if (["queued", "running"].includes(status)) return "warning";
  return "danger";
}
