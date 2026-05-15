import { ActionLink, ActionRail, Card } from "@/components/v2/v2-shell";
import { buildV2Params, type V2Session } from "@/lib/v2-data";

export function ValidationNextActions({ session }: { session: V2Session }) {
  const routeParams = buildV2Params(session);
  return (
    <Card
      title="Next actions"
      kicker="After validation"
      help="Validation should end with a clear follow-up path."
    >
      <ActionRail>
        <ActionLink href={`/v2/visual${routeParams}`}>Visualize evidence</ActionLink>
        <ActionLink href={`/v2/experiment${routeParams}`} variant="secondary">
          Compare another setup
        </ActionLink>
        <ActionLink href="/export" variant="secondary">
          Export evidence
        </ActionLink>
      </ActionRail>
      <div className="mt-4 grid gap-3 md:grid-cols-3">
        <Step
          title="Visualize"
          body="Inspect graph, molecule, target, and circuit context."
        />
        <Step
          title="Rerun"
          body="Launch a follow-up experiment if the evidence gap is specific."
        />
        <Step
          title="Export"
          body="Save the evidence packet with reviewer notes and provenance."
        />
      </div>
    </Card>
  );
}

function Step({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-xl border border-outline-variant/25 bg-surface-container-high/60 p-3">
      <p className="font-semibold text-on-surface">{title}</p>
      <p className="mt-2 text-xs leading-relaxed text-on-surface-variant">{body}</p>
    </div>
  );
}
