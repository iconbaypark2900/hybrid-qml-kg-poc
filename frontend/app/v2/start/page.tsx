"use client";

import { useSearchParams } from "next/navigation";
import { useMemo, useState } from "react";
import {
  ActionLink,
  ActionRail,
  Card,
  Chip,
  EvidenceCard,
  HelpTooltip,
  Metric,
  MetricStrip,
  PageHero,
  RunSummaryCard,
} from "@/components/v2/v2-shell";
import {
  ResumeSessionsPanel,
  StartSessionPanel,
} from "@/components/v2/session-controls";
import {
  buildV2Params,
  entitySuggestions,
  getV2CandidateForEntity,
  parseV2Session,
  type V2EntitySuggestion,
  type V2RunMode,
} from "@/lib/v2-data";

const workflowSteps = [
  {
    title: "Initialize",
    body: "Set the biomedical question and lock the active investigation context.",
  },
  {
    title: "Experiment",
    body: "Compare candidates, models, commands, and run provenance.",
  },
  {
    title: "Validate",
    body: "Check trust signals before promoting a result into evidence.",
  },
  {
    title: "Visualize",
    body: "Inspect molecule, graph, embedding, pathway, and circuit evidence.",
  },
];

export default function V2StartPage() {
  const searchParams = useSearchParams();
  const initialSession = useMemo(
    () => parseV2Session(searchParams),
    [searchParams],
  );
  const [query, setQuery] = useState(initialSession.selectedEntity.name);
  const [selected, setSelected] = useState<V2EntitySuggestion>(
    initialSession.selectedEntity,
  );
  const [runMode, setRunMode] = useState<V2RunMode>(initialSession.runMode);

  const matches = useMemo(() => {
    const normalized = query.toLowerCase().trim();
    if (!normalized) return entitySuggestions;
    return entitySuggestions.filter(
      (item) =>
        item.name.toLowerCase().includes(normalized) ||
        item.type.toLowerCase().includes(normalized),
    );
  }, [query]);
  const selectedSession = useMemo(
    () =>
      parseV2Session({
        entity: selected.name,
        runMode,
        candidate: getV2CandidateForEntity(selected.name).disease,
      }),
    [runMode, selected.name],
  );
  const routeParams = buildV2Params(selectedSession);

  return (
    <div className="space-y-6">
      <PageHero
        eyebrow="Initialize"
        title="Frame the research question before the dashboard shows evidence."
        actions={
          <>
            <ActionLink href={`/v2/experiment${routeParams}`}>
              Continue to Experiment
            </ActionLink>
            <ActionLink href={`/v2/validation${routeParams}`} variant="secondary">
              Preview validation
            </ActionLink>
          </>
        }
      >
        Choose the starting entity, route, and run path. The rest of the cockpit
        follows this context so model outputs, validation checks, and visual
        evidence stay attached to one explicit hypothesis.
      </PageHero>

      <MetricStrip>
        <Metric
          label="Starting point"
          value={selected.name}
          detail={`${selected.type} selected`}
          help="The selected entity controls which candidate and evidence route the next pages emphasize."
        />
        <Metric
          label="Candidate"
          value={selectedSession.selectedCandidate.candidate}
          detail={`Against ${selectedSession.selectedCandidate.disease}`}
          help="This is the paper-aligned candidate queued for the active investigation."
        />
        <Metric
          label="Run path"
          value={runMode}
          detail="Controls the comparison posture"
          help="Classical, Hybrid, and Quantum hardware set different expectations for Experiment and Validate."
        />
        <Metric
          label="Evidence posture"
          value={selectedSession.selectedCandidate.evidencePosture}
          detail={selectedSession.selectedCandidate.decisionDetail}
          help="This is the starting evidence posture before deeper validation."
        />
      </MetricStrip>

      <section className="grid gap-4 xl:grid-cols-[1.35fr_0.9fr]">
        <Card
          title="Research intake"
          kicker="Context first"
          help="Initialize should read like a research intake form: one object, one candidate, one run path, and one next action."
          className="border-primary/30 bg-primary/5"
        >
          <div className="space-y-5">
            <div>
              <label className="mb-2 flex items-center gap-2 font-label text-xs font-bold uppercase tracking-widest text-on-surface-variant">
                Disease, compound, or target
                <HelpTooltip text="Choose from paper-aligned Hetionet-style entities. The selected type controls the investigation route." />
              </label>
              <div className="relative">
                <input
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  className="w-full rounded-xl border border-outline-variant/50 bg-surface-container-high px-4 py-3 text-sm text-on-surface outline-none focus:border-primary"
                  aria-label="Disease, compound, or target"
                />
                <span className="material-symbols-outlined pointer-events-none absolute right-3 top-3 text-base text-on-surface-variant">
                  search
                </span>
              </div>
              <div className="mt-3 grid gap-2 md:grid-cols-2">
                {matches.map((item) => (
                  <button
                    type="button"
                    key={item.name}
                    onClick={() => {
                      setSelected(item);
                      setQuery(item.name);
                    }}
                    className={`rounded-xl border px-3 py-3 text-left text-sm transition-colors ${
                      item.name === selected.name
                        ? "border-tertiary/50 bg-tertiary/15 text-on-surface"
                        : "border-outline-variant/30 bg-surface-container-high/50 text-on-surface-variant hover:border-primary/40 hover:text-on-surface"
                    }`}
                  >
                    <span className="block font-semibold">{item.name}</span>
                    <span className="mt-1 block text-xs">{item.type}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="rounded-xl border border-tertiary/25 bg-tertiary/10 p-4 text-sm leading-relaxed text-on-surface-variant">
              <strong className="text-on-surface">Investigation route:</strong>{" "}
              {selected.route}. Experiment will inspect{" "}
              {selectedSession.selectedCandidate.candidate} to{" "}
              {selectedSession.selectedCandidate.disease} and keep mechanism
              evidence visible beside model score.
            </div>

            <div>
              <p className="mb-2 flex items-center gap-2 font-label text-xs font-bold uppercase tracking-widest text-on-surface-variant">
                Run path
                <HelpTooltip text="Set the posture for the next pages. Hybrid is the default product path; Quantum hardware is reserved for explicit backend comparison." />
              </p>
              <div className="flex flex-wrap gap-2">
                {[
                  "Classical",
                  "Hybrid",
                  "Quantum hardware",
                ].map((item) => (
                  <Chip
                    key={item}
                    active={runMode === item}
                    onClick={() => setRunMode(item as V2RunMode)}
                  >
                    {item}
                  </Chip>
                ))}
              </div>
            </div>
          </div>
        </Card>

        <div className="space-y-4">
          <RunSummaryCard
            model={runMode === "Quantum hardware" ? "QSVC / IBM Runtime" : "Hybrid QML-KG"}
            relation="CtD"
            backend={runMode === "Quantum hardware" ? "IBM Runtime-ready" : "Artifact-backed"}
            artifact="Selected after Experiment"
          />
          <Card
            title="Save this setup"
            kicker="Resume-ready"
            help="Persist reviewer and session context so this investigation can be resumed and exported later."
          >
            <StartSessionPanel session={selectedSession} />
          </Card>
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-[1fr_1fr]">
        <Card title="Workflow contract" kicker="Canonical v2">
          <div className="grid gap-3 md:grid-cols-2">
            {workflowSteps.map((step) => (
              <EvidenceCard
                key={step.title}
                title={step.title}
                value="Required"
                detail={step.body}
                tone={step.title === "Validate" ? "warning" : "quantum"}
              />
            ))}
          </div>
        </Card>
        <Card
          title="Resume investigation"
          kicker="Recent sessions"
          help="Open a saved v2 research session and continue validation from its session id."
        >
          <ResumeSessionsPanel />
        </Card>
      </section>

      <ActionRail>
        <ActionLink href={`/v2/experiment${routeParams}`}>
          Open Experiment
        </ActionLink>
        <ActionLink href={`/v2/visual${routeParams}`} variant="secondary">
          Visualize Evidence
        </ActionLink>
        <ActionLink href="/v2/system" variant="secondary">
          Check Operations
        </ActionLink>
      </ActionRail>
    </div>
  );
}
