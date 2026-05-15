"use client";

import { Suspense, useMemo } from "react";
import { useSearchParams } from "next/navigation";
import {
  HelpTip,
  JourneyButton,
  JourneyCard,
  MetricTile,
  PageHero,
  TypeTag,
  journeyQuery,
  readJourneySetup,
  titleCase,
} from "@/components/journey-ui";

const methods = [
  {
    name: "ExtraTrees",
    type: "Classical",
    pr: "0.7987",
    meaning: "Best broad statistical judge for Compound-treats-Disease.",
  },
  {
    name: "Weighted ensemble",
    type: "Hybrid",
    pr: "0.7824",
    meaning: "Balances classical graph features with the quantum branch.",
  },
  {
    name: "QSVC Pauli",
    type: "Quantum",
    pr: "0.7210",
    meaning: "Comparison branch for feature-map behavior.",
  },
  {
    name: "Logistic regression",
    type: "Classical",
    pr: "0.6813",
    meaning: "Simple served predictor baseline.",
  },
];

const candidates = [
  ["Nilotinib", "Parkinson disease", "0.84", "Mechanism and target support"],
  ["Rapamycin", "Parkinson disease", "0.79", "Pathway evidence increased rank"],
  ["Ambroxol", "Parkinson disease", "0.72", "Known disease biology nearby"],
];

export default function ExperimentsPage() {
  return (
    <Suspense fallback={null}>
      <ExperimentsInner />
    </Suspense>
  );
}

function ExperimentsInner() {
  const searchParams = useSearchParams();
  const setup = useMemo(
    () => readJourneySetup(new URLSearchParams(searchParams.toString())),
    [searchParams],
  );
  const query = journeyQuery(setup);

  return (
    <div className="space-y-4">
      <PageHero
        eyebrow="Experiment"
        title="Change the setup and see how the evidence changes."
        body="Experiment complements Start by showing the consequences of your choices: which method judged the task best, which candidates surfaced, and what can be sent to validation."
        actions={
          <>
            <JourneyButton
              href={`/validation?${query}&drug=Nilotinib&disease=${encodeURIComponent(setup.disease)}`}
              tone="primary"
            >
              Send candidate to validation
            </JourneyButton>
            <JourneyButton href={`/visualization?${query}`}>Inspect visually</JourneyButton>
          </>
        }
      />

      <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <MetricTile
          label="Starting point"
          value={setup.topic}
          detail={`${setup.kind} route from Start`}
        />
        <MetricTile
          label="Run path"
          value={titleCase(setup.mode)}
          detail={setup.mode === "hybrid" ? "Classical plus quantum comparison" : "Single path comparison"}
        />
        <MetricTile
          label="Score filter"
          value={`>= ${setup.score}`}
          detail="Minimum prediction score"
        />
        <MetricTile
          label="Evidence rule"
          value={titleCase(setup.mechanism)}
          detail="Mechanism evidence weight"
        />
      </section>

      <section className="grid gap-4 xl:grid-cols-[minmax(0,1.2fr)_minmax(420px,0.8fr)]">
        <JourneyCard
          title="Method comparison"
          kicker="Which judge handled this setup best?"
          help="This is the model leaderboard. It answers which prediction method was most reliable for this task, not which drug is best."
        >
          <div className="overflow-x-auto">
            <table className="journey-table">
              <thead>
                <tr>
                  <th>Method</th>
                  <th>Type</th>
                  <th>PR-AUC</th>
                  <th>What it means</th>
                </tr>
              </thead>
              <tbody>
                {methods.map((row) => (
                  <tr key={row.name}>
                    <td className="font-semibold">{row.name}</td>
                    <td>
                      <TypeTag type={row.type} />
                    </td>
                    <td className="font-mono text-tertiary">{row.pr}</td>
                    <td className="text-on-surface-variant">{row.meaning}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </JourneyCard>

        <JourneyCard
          title="Candidate response"
          kicker="What the setup surfaced"
          help="Top predictions are candidate pairs. These are the leads to inspect, validate, rank, or use in a follow-up run."
        >
          <div className="overflow-x-auto">
            <table className="journey-table">
              <thead>
                <tr>
                  <th>Candidate</th>
                  <th>Score</th>
                  <th>Next</th>
                </tr>
              </thead>
              <tbody>
                {candidates.map(([drug, disease, score, reason]) => (
                  <tr key={drug}>
                    <td>
                      <strong>{drug}</strong>
                      <p className="mt-1 text-xs text-on-surface-variant">
                        {disease} - {reason}
                      </p>
                    </td>
                    <td className="font-mono text-tertiary">{score}</td>
                    <td>
                      <a
                        className="text-xs font-bold text-primary underline-offset-2 hover:underline"
                        href={`/validation?${query}&drug=${encodeURIComponent(drug)}&disease=${encodeURIComponent(disease)}`}
                      >
                        Validate
                      </a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </JourneyCard>
      </section>

      <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(380px,0.8fr)]">
        <JourneyCard
          title="Setup consequences"
          kicker="Readable experiment outcome"
          help="This section explains what changed because of the Start settings, so Experiment does not repeat the Start controls."
        >
          <div className="grid gap-3 md:grid-cols-3">
            <Outcome label="Classical signal" value="Broader" body="Graph and embedding features surface wider candidate coverage." />
            <Outcome label="Hybrid signal" value="Balanced" body="Stacking keeps candidates that hold up across model families." />
            <Outcome label="Quantum branch" value="Compare" body="QSVC branch is visible when feature-map behavior matters." />
          </div>
        </JourneyCard>

        <JourneyCard
          title="Launch follow-up"
          kicker="Next work"
          help="Follow-up turns a candidate and setup into the next focused run instead of leaving the user with a static report."
        >
          <div className="space-y-3 text-sm">
            {[
              "Validate the top candidate against evidence for and against it.",
              "Switch run path and compare whether the candidate survives.",
              "Launch a mechanism-first follow-up with the current filters.",
            ].map((item, index) => (
              <div
                className="grid grid-cols-[2rem_minmax(0,1fr)] gap-3 rounded-lg border border-outline/10 bg-surface-container-lowest/60 p-3"
                key={item}
              >
                <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10 text-xs font-black text-primary">
                  {index + 1}
                </span>
                <p className="leading-relaxed text-on-surface-variant">{item}</p>
              </div>
            ))}
          </div>
        </JourneyCard>
      </section>

      <div className="rounded-lg border border-primary/20 bg-primary/10 p-3 text-xs leading-relaxed text-on-surface-variant">
        <strong className="text-on-surface">Plain English:</strong> Start asks the
        question. Experiment shows what changed when that question became a run.{" "}
        <HelpTip text="This keeps Experiment from duplicating Start. It becomes the result comparison workspace." />
      </div>
    </div>
  );
}

function Outcome({ label, value, body }: { label: string; value: string; body: string }) {
  return (
    <div className="rounded-lg border border-outline/10 bg-surface-container-lowest/60 p-4">
      <p className="text-xs font-black uppercase tracking-[0.14em] text-on-surface-variant">
        {label}
      </p>
      <strong className="mt-2 block font-headline text-2xl text-on-surface">{value}</strong>
      <p className="mt-2 text-xs leading-relaxed text-on-surface-variant">{body}</p>
    </div>
  );
}
