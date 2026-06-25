import { ExperimentCandidatePanel } from "@/components/v2/experiment-candidate-panel";
import { ExperimentNextActions } from "@/components/v2/experiment-next-actions";
import { ModelLeaderboard } from "@/components/v2/model-leaderboard";
import { MoAExplanationPanel } from "@/components/v2/moa-explanation-panel";
import { RunProvenancePanel } from "@/components/v2/run-provenance-panel";
import { ScientificQualityPanel } from "@/components/v2/scientific-quality-panel";
import {
  ActionLink,
  Card,
  EvidenceCard,
  Metric,
  MetricStrip,
  PageHero,
} from "@/components/v2/v2-shell";
import {
  experimentMetricLabels,
  experimentSourceLabels,
} from "@/lib/v2-experiment-copy";
import { loadV2ExperimentEvidence, type EvidenceSource } from "@/lib/v2-live-data";
import { buildScientificQualityEvidence } from "@/lib/v2-quality-data";
import { buildV2Params, parseV2Session, type V2SearchParams } from "@/lib/v2-data";

export default async function V2ExperimentPage({
  searchParams,
}: {
  searchParams: Promise<V2SearchParams>;
}) {
  const session = parseV2Session(await searchParams);
  const evidence = await loadV2ExperimentEvidence(session);
  const quality = buildScientificQualityEvidence({
    latestRun: evidence.latestRun.data,
    analysis: evidence.analysis.data,
  });
  const routeParams = buildV2Params(session);

  return (
    <div className="space-y-6">
      <PageHero
        eyebrow="Experiment"
        title="Compare candidates, methods, and provenance for the active investigation."
        actions={
          <>
            <ActionLink href={`/v2/validation${routeParams}`}>
              Send candidate to Validate
            </ActionLink>
            <ActionLink href={`/v2/visual${routeParams}`} variant="secondary">
              Visualize evidence
            </ActionLink>
          </>
        }
      >
        Experiment turns the Initialize context into ranked candidates, model
        comparisons, run provenance, and explicit next actions. Every section
        labels whether it is live, fallback, empty, or error-derived.
      </PageHero>

      <MetricStrip>
        <Metric
          label={experimentMetricLabels[0]}
          value={session.selectedEntity.name}
          detail={session.selectedEntity.route}
          help="The selected entity determines the search direction inherited from Initialize."
        />
        <Metric
          label={experimentMetricLabels[1]}
          value={session.selectedCandidate.candidate}
          detail={`Against ${session.selectedCandidate.disease}`}
          help="The candidate currently queued for validation and visual evidence review."
        />
        <Metric
          label={experimentMetricLabels[2]}
          value={evidence.predictions.source}
          detail={evidence.predictions.source === "live" ? "Artifact-backed" : "Clearly labeled fallback"}
          help="Shows whether candidates came from live artifacts, fallback data, an empty response, or an API error."
        />
        <Metric
          label={experimentMetricLabels[3]}
          value={session.runMode}
          detail={`Score >= ${session.scoreThreshold}`}
          help="Run path and score filter selected during Initialize."
        />
      </MetricStrip>

      <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <EvidenceCard
          title={experimentSourceLabels[0]}
          value={evidence.status.source}
          detail={
            evidence.status.data?.status ??
            evidence.status.message ??
            "API health was not available."
          }
          tone={toneForSource(evidence.status.source)}
        />
        <EvidenceCard
          title={experimentSourceLabels[1]}
          value={evidence.latestRun.source}
          detail={
            evidence.latestRun.data?.message ??
            evidence.latestRun.message ??
            "Latest run artifact status was not recorded."
          }
          tone={toneForSource(evidence.latestRun.source)}
        />
        <EvidenceCard
          title={experimentSourceLabels[2]}
          value={evidence.predictions.source}
          detail={
            evidence.predictions.message ??
            `${evidence.predictions.data.length} candidate rows available.`
          }
          tone={toneForSource(evidence.predictions.source)}
        />
        <EvidenceCard
          title={experimentSourceLabels[3]}
          value={evidence.jobs.source}
          detail={evidence.jobs.message ?? `${evidence.jobs.data.length} jobs visible.`}
          tone={toneForSource(evidence.jobs.source)}
        />
      </section>

      <section className="grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
        <ModelLeaderboard methods={evidence.methods} />
        <RunProvenancePanel latestRun={evidence.latestRun} analysis={evidence.analysis} />
      </section>

      <MoAExplanationPanel />

      <ExperimentCandidatePanel session={session} predictions={evidence.predictions} />

      <Card
        title="Scientific quality controls"
        kicker="Trust layer"
        help="Quality controls are shown after the source trail so the reader can separate fresh evidence from fallback assertions."
      >
        <ScientificQualityPanel quality={quality} />
      </Card>

      <ExperimentNextActions session={session} jobs={evidence.jobs} />
    </div>
  );
}

function toneForSource(source: EvidenceSource) {
  if (source === "live") return "success";
  if (source === "error") return "danger";
  if (source === "empty") return "warning";
  return "quantum";
}
