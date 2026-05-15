"use client";

import { Suspense, useMemo } from "react";
import { useSearchParams } from "next/navigation";
import {
  HelpTip,
  JourneyButton,
  JourneyCard,
  MetricTile,
  PageHero,
  Pill,
  journeyQuery,
  readJourneySetup,
  titleCase,
} from "@/components/journey-ui";

export default function ValidationPage() {
  return (
    <Suspense fallback={null}>
      <ValidationInner />
    </Suspense>
  );
}

function ValidationInner() {
  const searchParams = useSearchParams();
  const setup = useMemo(
    () => readJourneySetup(new URLSearchParams(searchParams.toString())),
    [searchParams],
  );
  const query = journeyQuery(setup);

  return (
    <div className="space-y-4">
      <PageHero
        eyebrow="Validation"
        title="Build your own conclusion from the evidence."
        body="Validation is where a new user stops trusting the leaderboard blindly. Pick one candidate, check why it surfaced, weigh evidence for and against it, and decide what should happen next."
        actions={
          <>
            <JourneyButton href={`/visualization?${query}`} tone="primary">
              Open visual evidence
            </JourneyButton>
            <JourneyButton href={`/experiments?${query}`}>Back to experiment</JourneyButton>
          </>
        }
      />

      <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <MetricTile
          label="Candidate"
          value={setup.drug}
          detail={`For ${setup.disease}`}
          help="This is the candidate pair selected from Experiment."
        />
        <MetricTile
          label="Run path"
          value={titleCase(setup.mode)}
          detail={`Score filter ${setup.score}`}
        />
        <MetricTile
          label="Mechanism"
          value={titleCase(setup.mechanism)}
          detail="How strongly biology should affect belief"
        />
        <MetricTile
          label="Decision"
          value="Review"
          detail="Evidence still needs human interpretation"
        />
      </section>

      <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(420px,0.85fr)]">
        <JourneyCard
          title="Candidate under review"
          kicker="Focused belief check"
          help="The score is not the conclusion. This workspace helps the user decide whether the result is plausible enough to keep."
        >
          <div className="grid gap-4 md:grid-cols-2">
            <div>
              <label className="journey-label">
                Compound
                <HelpTip text="The proposed drug or compound. Change it to test a different pair with the same setup." />
              </label>
              <input className="journey-input" value={setup.drug} readOnly />
            </div>
            <div>
              <label className="journey-label">
                Disease
                <HelpTip text="The condition being tested for treatment or biological association." />
              </label>
              <input className="journey-input" value={setup.disease} readOnly />
            </div>
          </div>

          <div className="mt-4 grid gap-3 md:grid-cols-3">
            <ScoreCard label="Prediction score" value="0.84" body="High enough to inspect" />
            <ScoreCard label="Mechanism support" value="Strong" body="Target and pathway context present" />
            <ScoreCard label="Novelty" value="Medium" body="Some nearby known biology" />
          </div>

          <div className="mt-4 flex flex-wrap gap-2">
            <Pill active>Keep as plausible</Pill>
            <Pill>Needs more evidence</Pill>
            <Pill>Reject for now</Pill>
          </div>
        </JourneyCard>

        <JourneyCard
          title="Evidence checklist"
          kicker="For the user"
          help="This turns the research result into a decision-making checklist."
        >
          <div className="space-y-3">
            {[
              ["1", "Model agreement", "Does this candidate survive classical, hybrid, and quantum comparison?"],
              ["2", "Mechanism support", "Are genes, targets, or pathways connected in a believable way?"],
              ["3", "Disease relevance", "Does the evidence connect to the selected disease route?"],
              ["4", "Next step", "Should this move to visual inspection or a follow-up experiment?"],
            ].map(([number, title, body]) => (
              <div
                key={title}
                className="grid grid-cols-[2rem_minmax(0,1fr)] gap-3 rounded-lg border border-outline/10 bg-surface-container-lowest/60 p-3"
              >
                <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10 text-xs font-black text-primary">
                  {number}
                </span>
                <div>
                  <p className="font-semibold text-on-surface">{title}</p>
                  <p className="mt-1 text-xs leading-relaxed text-on-surface-variant">
                    {body}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </JourneyCard>
      </section>

      <section className="grid gap-4 xl:grid-cols-2">
        <JourneyCard
          title="Evidence for and against"
          kicker="Make the tradeoff visible"
          help="A candidate can have a high score and still be weak scientifically. This separates supporting and cautionary evidence."
        >
          <div className="grid gap-3 md:grid-cols-2">
            <EvidenceColumn
              title="For"
              items={[
                "Target neighborhood connects to Parkinson biology.",
                "Hybrid path keeps the candidate after score filtering.",
                "Mechanism-first lens increases confidence.",
              ]}
            />
            <EvidenceColumn
              title="Against"
              items={[
                "Evidence may depend on indirect graph paths.",
                "Quantum branch is a comparison signal, not standalone proof.",
                "Needs literature and assay follow-up before action.",
              ]}
            />
          </div>
        </JourneyCard>

        <JourneyCard
          title="Conclusion builder"
          kicker="User-owned decision"
          help="The goal is for a new user to leave with their own conclusion, not just a dashboard number."
        >
          <div className="space-y-3 text-sm text-on-surface-variant">
            <p>
              I would keep <strong className="text-on-surface">{setup.drug}</strong>{" "}
              for <strong className="text-on-surface">{setup.disease}</strong> as a
              plausible candidate because the score is above{" "}
              <strong className="text-on-surface">{setup.score}</strong> and the
              mechanism lens is set to{" "}
              <strong className="text-on-surface">{setup.mechanism}</strong>.
            </p>
            <div className="rounded-lg border border-primary/20 bg-primary/10 p-3">
              Next best action: open the Visual page and inspect molecule, graph,
              pathway, and quantum-circuit evidence before launching a follow-up.
            </div>
            <JourneyButton href={`/visualization?${query}`} tone="primary">
              Inspect visual evidence
            </JourneyButton>
          </div>
        </JourneyCard>
      </section>
    </div>
  );
}

function ScoreCard({ label, value, body }: { label: string; value: string; body: string }) {
  return (
    <div className="rounded-lg border border-outline/10 bg-surface-container-lowest/60 p-3">
      <p className="text-xs font-black uppercase tracking-[0.14em] text-on-surface-variant">
        {label}
      </p>
      <strong className="mt-2 block font-headline text-xl text-on-surface">{value}</strong>
      <p className="mt-1 text-xs text-on-surface-variant">{body}</p>
    </div>
  );
}

function EvidenceColumn({ title, items }: { title: string; items: string[] }) {
  return (
    <div className="rounded-lg border border-outline/10 bg-surface-container-lowest/60 p-3">
      <h3 className="font-headline text-base font-semibold text-on-surface">{title}</h3>
      <ul className="mt-3 space-y-2 text-sm text-on-surface-variant">
        {items.map((item) => (
          <li key={item} className="flex gap-2">
            <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-tertiary" />
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
