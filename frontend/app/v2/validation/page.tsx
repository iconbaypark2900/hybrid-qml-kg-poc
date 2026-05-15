import {
  CandidateEvidenceLinks,
  EvidenceProvenanceSummary,
} from "@/components/v2/evidence-provenance";
import { EvidenceAxis } from "@/components/v2/evidence-axis";
import { ValidationDecisionPanel } from "@/components/v2/validation-decision-panel";
import { ValidationEvidenceChecklist } from "@/components/v2/validation-evidence-checklist";
import { ValidationNextActions } from "@/components/v2/validation-next-actions";
import { ValidationTrustScorecard } from "@/components/v2/validation-trust-scorecard";
import {
  ActionLink,
  Card,
  Metric,
  MetricStrip,
  PageHero,
  StatusBadge,
} from "@/components/v2/v2-shell";
import { buildV2Params, parseV2Session, type V2SearchParams } from "@/lib/v2-data";
import {
  buildV2EvidenceSnapshot,
  loadV2ExperimentEvidence,
  type EvidenceSource,
} from "@/lib/v2-live-data";

export default async function V2ValidationPage({
  searchParams,
}: {
  searchParams: Promise<V2SearchParams>;
}) {
  const session = parseV2Session(await searchParams);
  const evidence = await loadV2ExperimentEvidence(session);
  const candidate = session.selectedCandidate;
  const routeParams = buildV2Params(session);
  const evidenceSnapshot = buildV2EvidenceSnapshot(session, evidence);

  return (
    <div className="space-y-6">
      <PageHero
        eyebrow="Validate"
        title={`Decide whether ${candidate.candidate} to ${candidate.disease} should move forward.`}
        actions={
          <>
            <ActionLink href={`/v2/visual${routeParams}`}>
              Visualize evidence
            </ActionLink>
            <ActionLink href={`/v2/experiment${routeParams}`} variant="secondary">
              Back to Experiment
            </ActionLink>
          </>
        }
      >
        Validation is the trust layer. It separates clinical support, mechanism
        plausibility, model signal, baseline context, and artifact quality before
        the researcher chooses Keep, Review, or Reject.
      </PageHero>

      <MetricStrip>
        <Metric
          label="Candidate"
          value={candidate.candidate}
          detail={`${candidate.disease} hypothesis`}
          help="The compound-disease pair selected from Experiment."
        />
        <Metric
          label="Model score"
          value={candidate.score}
          detail="Ranking signal only"
          help="Scores help rank candidates; they are not clinical proof."
        />
        <Metric
          label="Evidence posture"
          value={candidate.evidencePosture}
          detail={candidate.decisionDetail}
          help="Plain-language posture based on the current evidence bundle."
        />
        <Metric
          label="Decision"
          value={candidate.decision}
          detail={session.sessionId ?? "Unsaved session"}
          help="The current recommended decision label before reviewer edits."
        />
      </MetricStrip>

      <ValidationTrustScorecard candidate={candidate} evidence={evidence} />

      <section className="grid gap-3 md:grid-cols-3">
        <SourceTile
          label="Candidate source"
          source={evidence.predictions.source}
          value={evidence.predictions.source === "live" ? "artifact-backed" : "fallback labeled"}
          message={evidence.predictions.message}
          provenance={evidence.predictions.provenance}
        />
        <SourceTile
          label="Model source"
          source={evidence.methods.source}
          value={evidence.methods.source === "live" ? "latest run metrics" : "paper fallback"}
          message={evidence.methods.message}
          provenance={evidence.methods.provenance}
        />
        <SourceTile
          label="System source"
          source={evidence.status.source}
          value={evidence.status.data?.status ?? "unavailable"}
          message={evidence.status.message}
          provenance={evidence.status.provenance}
        />
      </section>

      <Card
        title="Evidence axes"
        kicker="Score versus support"
        help="Separates model rank from clinical and mechanistic support so the score does not become the decision."
      >
        <EvidenceAxis candidate={candidate} />
      </Card>

      <section className="grid gap-4 xl:grid-cols-[1.05fr_0.95fr]">
        <ValidationEvidenceChecklist candidate={candidate} evidence={evidence} />
        <Card
          title="Skeptic view"
          kicker="What could weaken it"
          help="The page should make weak evidence as visible as supporting evidence."
        >
          <ul className="space-y-3">
            {candidate.weakItems.map((item) => (
              <li
                key={item}
                className="rounded-xl border border-outline-variant/25 bg-surface-container-high/60 p-3 text-sm text-on-surface-variant"
              >
                {item}
              </li>
            ))}
          </ul>
        </Card>
      </section>

      <Card
        title="Traceable evidence links"
        kicker="Model, clinical, mechanism, KG"
        help="Each validation statement should keep a source trail so review is reproducible."
      >
        <CandidateEvidenceLinks links={evidence.predictions.evidenceLinks} />
      </Card>

      <section className="grid gap-4 xl:grid-cols-[0.95fr_1.05fr]">
        <ValidationDecisionPanel
          session={session}
          candidate={candidate}
          evidenceSnapshot={evidenceSnapshot}
        />
        <Card
          title="Evidence notebook"
          kicker="Working conclusion"
          help="A concise explanation that travels with the candidate when it is saved or exported."
        >
          <div className="rounded-xl border border-outline-variant/40 bg-surface-container-high p-4">
            <p className="font-label text-xs font-bold uppercase tracking-widest text-on-surface-variant">
              Current conclusion
            </p>
            <p className="mt-3 text-sm leading-relaxed text-on-surface">
              {candidate.workingConclusion}
            </p>
          </div>
        </Card>
      </section>

      <ValidationNextActions session={session} />
    </div>
  );
}

function SourceTile({
  label,
  source,
  value,
  message,
  provenance,
}: {
  label: string;
  source: EvidenceSource;
  value: string;
  message: string | null | undefined;
  provenance: Parameters<typeof EvidenceProvenanceSummary>[0]["provenance"];
}) {
  return (
    <div className="rounded-xl border border-outline-variant/25 bg-surface-container-high/60 p-4">
      <div className="flex items-center justify-between gap-3">
        <p className="font-label text-xs font-bold uppercase tracking-widest text-on-surface-variant">
          {label}
        </p>
        <StatusBadge tone={toneForSource(source)}>{source}</StatusBadge>
      </div>
      <p className="mt-3 font-mono text-sm text-on-surface">{value}</p>
      {message ? (
        <p className="mt-2 text-xs leading-relaxed text-on-surface-variant">
          {message}
        </p>
      ) : null}
      <div className="mt-3">
        <EvidenceProvenanceSummary title="Trace" provenance={provenance} />
      </div>
    </div>
  );
}

function toneForSource(source: EvidenceSource) {
  if (source === "live") return "success";
  if (source === "error") return "danger";
  if (source === "empty") return "warning";
  return "quantum";
}
