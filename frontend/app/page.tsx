"use client";

import { useEffect, useMemo, useState } from "react";
import type { KGSearchResult } from "@/lib/api";
import { searchKGEntities } from "@/lib/api";
import {
  HelpTip,
  JourneyButton,
  JourneyCard,
  MetricTile,
  MiniCircuit,
  PageHero,
  Pill,
  StageSteps,
  TypeTag,
  defaultSetup,
  journeyQuery,
  titleCase,
  type JourneySetup,
} from "@/components/journey-ui";

type TopicKind = "Disease" | "Compound" | "Gene / protein" | "Pathway";

const fallbackTopics: Array<KGSearchResult & { kind: TopicKind }> = [
  { id: "Disease::Parkinson disease", name: "Parkinson disease", kind: "Disease" },
  { id: "Disease::Hypertension", name: "Hypertension", kind: "Disease" },
  { id: "Compound::Metformin", name: "Metformin", kind: "Compound" },
  { id: "Gene::DRD2", name: "DRD2", kind: "Gene / protein" },
  { id: "Pathway::Autophagy", name: "Autophagy pathway", kind: "Pathway" },
];

function normalizeKind(kind: string): TopicKind {
  const value = kind.toLowerCase();
  if (value.includes("compound") || value.includes("drug")) return "Compound";
  if (value.includes("gene") || value.includes("protein")) return "Gene / protein";
  if (value.includes("pathway")) return "Pathway";
  return "Disease";
}

function routeExplanation(kind: TopicKind, topic: string) {
  if (kind === "Compound") {
    return `Compound route: Experiment asks which diseases and mechanisms ${topic} may connect to.`;
  }
  if (kind === "Gene / protein") {
    return `Target route: Experiment looks for compounds and diseases connected through ${topic}.`;
  }
  if (kind === "Pathway") {
    return `Pathway route: Experiment prioritizes compounds that may influence ${topic}.`;
  }
  return `Disease route: Experiment searches for compounds that may treat or mechanistically connect to ${topic}.`;
}

export default function StartPage() {
  const [topic, setTopic] = useState(defaultSetup.topic);
  const [kind, setKind] = useState<TopicKind>("Disease");
  const [mode, setMode] = useState<JourneySetup["mode"]>("hybrid");
  const [score, setScore] = useState(65);
  const [mechanism, setMechanism] = useState(82);
  const [targets, setTargets] = useState(true);
  const [pathways, setPathways] = useState(true);
  const [weakLinks, setWeakLinks] = useState(false);
  const [suggestions, setSuggestions] = useState(fallbackTopics);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const query = topic.trim();
    if (query.length < 2) return;
    let cancelled = false;
    const timer = window.setTimeout(() => {
      searchKGEntities(query, 6)
        .then((data) => {
          if (cancelled) return;
          const mapped = data.results.map((item) => ({
            ...item,
            kind: normalizeKind(item.kind),
          }));
          setSuggestions(mapped.length ? mapped : fallbackTopics);
        })
        .catch(() => {
          if (!cancelled) setSuggestions(fallbackTopics);
        });
    }, 180);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [topic]);

  const mechanismLabel = mechanism >= 75 ? "high" : mechanism >= 45 ? "medium" : "low";
  const setup = useMemo<JourneySetup>(
    () => ({
      ...defaultSetup,
      topic,
      kind,
      mode,
      score: (score / 100).toFixed(2),
      mechanism: mechanismLabel,
      targets,
      pathways,
      weakLinks,
      disease: kind === "Disease" ? topic : defaultSetup.disease,
    }),
    [kind, mechanismLabel, mode, pathways, score, targets, topic, weakLinks],
  );
  const query = journeyQuery(setup);

  return (
    <div className="space-y-4">
      <PageHero
        eyebrow="Start"
        title="Start with your own biomedical question and explore what the graph suggests."
        actions={
          <>
            <JourneyButton href={`/experiments?${query}`} tone="primary">
              Open experiment
            </JourneyButton>
            <JourneyButton href={`/visualization?${query}`}>Open visual evidence</JourneyButton>
          </>
        }
      />

      <StageSteps />

      <section className="grid gap-4 xl:grid-cols-[minmax(0,1.45fr)_minmax(360px,0.9fr)]">
        <JourneyCard
          title="Choose a starting point"
          kicker="User-controlled"
          help="Choose the biomedical entity that should drive the rest of the journey. The route changes depending on whether you pick a disease, compound, gene, or pathway."
        >
          <label className="journey-label">
            Disease or topic
            <HelpTip text="Disease starts drug repurposing. Compound asks which diseases it may treat. Gene/protein asks which compounds and diseases connect through that target. Pathway searches for compounds that influence a mechanism." />
          </label>
          <div className="relative">
            <input
              className="journey-input"
              value={topic}
              onChange={(event) => {
                setTopic(event.target.value);
                setOpen(true);
              }}
              onFocus={() => setOpen(true)}
              placeholder="Search disease, compound, gene, or pathway"
            />
            {open ? (
              <div className="absolute left-0 right-0 top-[calc(100%+0.45rem)] z-40 overflow-hidden rounded-lg border border-primary/25 bg-surface-container-lowest shadow-2xl">
                {suggestions.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onMouseDown={(event) => event.preventDefault()}
                    onClick={() => {
                      setTopic(item.name);
                      setKind(normalizeKind(item.kind));
                      setOpen(false);
                    }}
                    className="flex w-full items-center justify-between px-3 py-2.5 text-left text-sm text-on-surface hover:bg-primary/10"
                  >
                    <span>{item.name}</span>
                    <span className="text-xs text-on-surface-variant">
                      {normalizeKind(item.kind)}
                    </span>
                  </button>
                ))}
              </div>
            ) : null}
          </div>

          <div className="mt-3 rounded-lg border border-tertiary/25 bg-tertiary/10 p-3 text-xs leading-relaxed text-on-surface-variant">
            <strong className="text-on-surface">Current route:</strong>{" "}
            {routeExplanation(kind, topic)}
          </div>

          <div className="mt-4 flex flex-wrap gap-2">
            <Pill
              active={mode === "classical"}
              onClick={() => setMode("classical")}
              help="Runs the classical branch: RotatE features plus RandomForest, ExtraTrees, and LogisticRegression."
            >
              Classical
            </Pill>
            <Pill
              active={mode === "hybrid"}
              onClick={() => setMode("hybrid")}
              help="Runs the combined branch: classical results plus quantum comparison and stacking ensemble."
            >
              Hybrid
            </Pill>
            <Pill
              active={mode === "quantum"}
              onClick={() => setMode("quantum")}
              help="Keeps the quantum feature map, circuit, and QSVC comparison at the center of the workflow."
            >
              Quantum hardware
            </Pill>
          </div>
        </JourneyCard>

        <JourneyCard
          title="Control the Parameters"
          kicker="Make the graph answer differently"
          help="These settings reshape what Experiment will surface, without making the user choose model internals first."
        >
          <div className="space-y-5">
            <div>
              <label className="journey-label">
                Minimum prediction score: {(score / 100).toFixed(2)}
                <HelpTip text="Higher values show fewer, stronger candidates. Lower values reveal more speculative candidates for exploration." />
              </label>
              <input
                type="range"
                min="10"
                max="95"
                value={score}
                onChange={(event) => setScore(Number(event.target.value))}
                className="w-full accent-tertiary"
              />
            </div>
            <div>
              <label className="journey-label">
                Mechanism evidence weight: {mechanismLabel}
                <HelpTip text="Higher values favor candidates with target, pathway, and mechanism support before the score alone." />
              </label>
              <input
                type="range"
                min="0"
                max="100"
                value={mechanism}
                onChange={(event) => setMechanism(Number(event.target.value))}
                className="w-full accent-tertiary"
              />
            </div>
            <div className="flex flex-wrap gap-2">
              <Pill active={targets} onClick={() => setTargets((value) => !value)}>
                Include protein targets
              </Pill>
              <Pill active={pathways} onClick={() => setPathways((value) => !value)}>
                Show pathways
              </Pill>
              <Pill active={weakLinks} onClick={() => setWeakLinks((value) => !value)}>
                Show weak links
              </Pill>
            </div>
          </div>
        </JourneyCard>
      </section>

      <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <MetricTile
          label="Selected route"
          value={kind}
          detail={routeExplanation(kind, topic)}
          help="This tells Experiment how to repurpose the link-prediction task around your starting entity."
        />
        <MetricTile
          label="Run mode"
          value={mode === "quantum" ? "Quantum" : mode[0].toUpperCase() + mode.slice(1)}
          detail="Classical, hybrid, or quantum hardware path"
        />
        <MetricTile
          label="Score filter"
          value={`>= ${(score / 100).toFixed(2)}`}
          detail="Minimum candidate score"
        />
        <MetricTile
          label="Next action"
          value="Experiment"
          detail="Compare what this setup produces"
        />
      </section>

      <ExperimentSection setup={setup} />
      <ValidationSection setup={setup} />
      <VisualSection setup={setup} />
    </div>
  );
}

function ExperimentSection({ setup }: { setup: JourneySetup }) {
  const query = journeyQuery(setup);
  const methods = [
    ["ExtraTrees", "Classical", "0.7987", "Best whole-run CtD judge"],
    ["Weighted ensemble", "Hybrid", "0.7824", "Combines classical and quantum signals"],
    ["QSVC Pauli", "Quantum", "0.7210", "Useful comparison branch"],
    ["Logistic regression", "Classical", "0.6813", "Simple served predictor baseline"],
  ];
  const candidates = [
    ["Nilotinib", setup.disease, "0.84", "Validate"],
    ["Rapamycin", setup.disease, "0.79", "Compare"],
    ["Ambroxol", setup.disease, "0.72", "Open 3D"],
  ];

  return (
    <section className="journey-page" id="experiment">
      <PageHero
        eyebrow="Experiment"
        title="Compare what your setup produced."
        body="Experiment receives the choices from Start and shows the consequences: which method judged the task best, which candidates surfaced, what changed between run paths, and what follow-up work can be launched."
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

      <div className="grid gap-3 p-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricTile label="Starting point" value={setup.topic} detail={`${setup.kind} route from Start`} />
        <MetricTile label="Run path" value={titleCase(setup.mode)} detail="Classical plus quantum comparison" />
        <MetricTile label="Score filter" value={`>= ${setup.score}`} detail="Minimum prediction score" />
        <MetricTile label="Evidence rule" value={titleCase(setup.mechanism)} detail="Mechanism evidence weight" />
      </div>

      <div className="grid gap-4 p-4 pt-0 xl:grid-cols-[minmax(0,1.2fr)_minmax(420px,0.8fr)]">
        <JourneyCard title="Method comparison" kicker="Which judge handled this setup best?" help="This is the model leaderboard. It compares prediction methods, not drugs.">
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
              {methods.map(([name, type, pr, meaning]) => (
                <tr key={name}>
                  <td><strong>{name}</strong></td>
                  <td><TypeTag type={type} /></td>
                  <td className="font-mono text-tertiary">{pr}</td>
                  <td>{meaning}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </JourneyCard>

        <JourneyCard title="Candidate response" kicker="What the setup surfaced" help="These are the top prediction rows, which are the candidates to validate or inspect in 3D.">
          <table className="journey-table">
            <thead>
              <tr>
                <th>Candidate</th>
                <th>Score</th>
                <th>Next</th>
              </tr>
            </thead>
            <tbody>
              {candidates.map(([drug, disease, score, next]) => (
                <tr key={drug}>
                  <td><strong>{drug} -&gt; {disease}</strong></td>
                  <td className="font-mono text-tertiary">{score}</td>
                  <td>{next}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </JourneyCard>

        <JourneyCard title="Launch follow-up" kicker="Next action from this setup" help="Follow-up turns the selected setup into the next useful action.">
          <div className="space-y-2">
            {["Validate top candidate", "Compare another path", "Run follow-up"].map((item, index) => (
              <div className="grid grid-cols-[2rem_minmax(0,1fr)] gap-3 rounded-lg border border-outline/10 bg-surface-container-lowest/60 p-3" key={item}>
                <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10 text-xs font-black text-primary">{index + 1}</span>
                <div>
                  <strong className="text-sm text-on-surface">{item}</strong>
                  <p className="text-xs text-on-surface-variant">Move this setup into the next review step.</p>
                </div>
              </div>
            ))}
          </div>
        </JourneyCard>

        <JourneyCard title="Run monitor" kicker="Work generated from Experiment" help="This shows jobs created by experiments without turning the whole page into a job table.">
          <table className="journey-table">
            <tbody>
              {[
                ["job_0424_001", "ready", "18m 12s"],
                ["job_0424_002", "running", "4m 33s"],
                ["job_0423_017", "failed", "1m 09s"],
              ].map(([job, status, time]) => (
                <tr key={job}>
                  <td><strong>{job}</strong></td>
                  <td>{status}</td>
                  <td>{time}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </JourneyCard>
      </div>
    </section>
  );
}

function ValidationSection({ setup }: { setup: JourneySetup }) {
  const query = journeyQuery(setup);
  return (
    <section className="journey-page" id="validation">
      <PageHero
        eyebrow="Validation"
        title="Decide whether one candidate is worth believing."
        body="Validation is where the user moves from possible matches to provisional conclusions: inspect one candidate, weigh evidence for and against it, compare alternatives, and choose a next step."
        actions={<JourneyButton href={`/visualization?${query}`} tone="primary">Open visual evidence</JourneyButton>}
      />

      <div className="grid gap-4 p-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(420px,0.9fr)]">
        <JourneyCard title="Candidate under review" kicker="Selected from Experiment" help="This focuses on one drug-disease pair instead of the entire leaderboard.">
          <label className="journey-label">Candidate</label>
          <input className="journey-input mb-3" value={`${setup.drug} -> ${setup.disease}`} readOnly />
          <label className="journey-label">Why it was selected</label>
          <input className="journey-input mb-3" value="High hybrid score with target and pathway evidence" readOnly />
          <label className="journey-label">Active setup</label>
          <input className="journey-input" value={`${titleCase(setup.mode)} run, score >= ${setup.score}, mechanism evidence ${setup.mechanism}`} readOnly />
          <div className="mt-4 flex flex-wrap gap-2">
            <Pill active>Review evidence</Pill>
            <Pill>Compare alternatives</Pill>
            <Pill>Open 3D</Pill>
          </div>
        </JourneyCard>

        <JourneyCard title="Provisional confidence" kicker="76%" help="Confidence is not proof. It summarizes whether the candidate is worth keeping for follow-up.">
          <div className="flex items-center gap-5">
            <div className="grid h-28 w-28 place-items-center rounded-full border-[10px] border-tertiary bg-surface-container-lowest text-xl font-black text-on-surface">
              76%
            </div>
            <p className="text-sm leading-relaxed text-on-surface-variant">
              Strong enough to keep investigating, but still needs visual inspection,
              literature review, and comparison against alternatives.
            </p>
          </div>
        </JourneyCard>

        <JourneyCard title="Evidence checklist" kicker="What supports the candidate?" help="A readable checklist helps new users understand what they are validating.">
          <div className="space-y-2">
            {["Model agreement", "Mechanism support", "Literature gap"].map((item) => (
              <div className="rounded-lg border border-outline/10 bg-surface-container-lowest/60 p-3" key={item}>
                <strong className="text-sm text-on-surface">{item}</strong>
                <p className="text-xs text-on-surface-variant">Review this signal before deciding.</p>
              </div>
            ))}
          </div>
        </JourneyCard>

        <JourneyCard title="For and against" kicker="Make uncertainty visible" help="The UI should show why a candidate might fail, not just why it looks good.">
          <table className="journey-table">
            <tbody>
              <tr><td><TypeTag type="For" /></td><td>Target/pathway link</td><td>Biology gives the score a plausible story</td></tr>
              <tr><td><TypeTag type="For" /></td><td>Model consistency</td><td>Not only one model family supports it</td></tr>
              <tr><td><TypeTag type="Caution" /></td><td>Clinical distance</td><td>Could require more proof before follow-up</td></tr>
            </tbody>
          </table>
        </JourneyCard>
      </div>
    </section>
  );
}

function VisualSection({ setup }: { setup: JourneySetup }) {
  const query = journeyQuery(setup);
  return (
    <section className="journey-page" id="visual">
      <PageHero
        eyebrow="Visual"
        title="See the evidence from multiple angles."
        body="Visual is the understanding studio. Instead of only showing a graph, it lets the user move between molecule, target, binding pocket, pathway, quantum circuit, disease context, and the original Hetionet neighborhood."
        actions={<JourneyButton href={`/experiments?${query}`}>Back to experiment</JourneyButton>}
      />

      <div className="grid gap-3 p-4 md:grid-cols-3">
        <MetricTile label="3D molecule" value="Compound" detail={`${setup.drug}, atoms, bonds`} />
        <MetricTile label="Protein target" value="Gene" detail="Pocket and pathway context" />
        <MetricTile label="Quantum model" value="Circuit" detail="Feature-map branch" />
      </div>

      <div className="p-4 pt-0">
        <JourneyCard title="Evidence lenses" kicker="Switch how the same candidate is explained" help="These are the visual modes that are more useful than generic graph/embedding/neighborhood labels.">
          <div className="grid gap-2 md:grid-cols-3">
            {["Molecule shape", "Target biology", "Mechanism story", "Graph evidence", "Model disagreement", "Quantum circuit", "What changed?"].map((item) => (
              <button className="rounded-lg border border-outline/10 bg-surface-container-lowest/60 p-3 text-left" key={item} type="button">
                <strong className="text-sm text-on-surface">{item}</strong>
                <p className="mt-1 text-xs text-on-surface-variant">Explain one part of the evidence.</p>
              </button>
            ))}
          </div>
        </JourneyCard>
      </div>

      <div className="grid gap-4 p-4 pt-0 md:grid-cols-2 xl:grid-cols-3">
        <VisualMini title="Molecule" body="Compound conformer with atoms, bonds, charges, and scaffold." />
        <VisualMini title="Protein target" body="Pocket and active-site context for the candidate." />
        <VisualMini title="Binding pocket" body="Ligand and protein surface context." />
        <VisualMini title="Mechanism cascade" body="Drug to target to pathway to disease." />
        <VisualMini title="Disease context" body="Disease neighborhood and phenotype context." />
        <JourneyCard title="Quantum circuit" kicker="Feature-map view" help="Shows the quantum branch used for comparison.">
          <MiniCircuit />
        </JourneyCard>
        <VisualMini title="Evidence scene" body="Combined story: molecule, target, mechanism, disease, and model score." />
      </div>

      <div className="grid gap-4 p-4 pt-0 xl:grid-cols-[minmax(0,1.15fr)_minmax(360px,0.85fr)]">
        <JourneyCard title="What can become 3D?" kicker="Hetionet concept to visual" help="This maps dataset concepts to proper visual representations.">
          <table className="journey-table">
            <tbody>
              {[
                ["Compound / drug", "Molecular conformer", "SMILES, SDF, PubChem, DrugBank"],
                ["Gene / protein", "Protein structure", "UniProt, PDB, AlphaFold"],
                ["Drug target edge", "Binding pocket scene", "Known pocket or docking pose"],
                ["Pathway", "3D mechanism cascade", "Pathway membership and layout"],
                ["Quantum branch", "Quantum circuit scene", "Feature map, qubits, circuit layers"],
                ["Disease", "Tissue/translation landscape", "Disease system and tissue map"],
              ].map(([concept, visual, source]) => (
                <tr key={concept}><td><strong>{concept}</strong></td><td>{visual}</td><td>{source}</td></tr>
              ))}
            </tbody>
          </table>
        </JourneyCard>

        <JourneyCard title="Knowledge graph context" kicker="Still useful, but not the whole visual" help="The graph should explain relationships while richer 3D scenes make the evidence easier to inspect.">
          <div className="flex h-44 items-end gap-3">
            {[36, 58, 82, 70, 92, 75, 52, 95].map((height, index) => (
              <span
                className="flex-1 rounded-t bg-gradient-to-t from-tertiary/60 to-primary"
                key={index}
                style={{ height: `${height}%` }}
              />
            ))}
          </div>
        </JourneyCard>
      </div>
    </section>
  );
}

function VisualMini({ title, body }: { title: string; body: string }) {
  return (
    <JourneyCard title={title} help={body}>
      <div className="relative h-40 overflow-hidden rounded-lg border border-outline/10 bg-surface-container-lowest/60">
        <span className="absolute left-[20%] top-[55%] h-4 w-4 rounded-full bg-tertiary" />
        <span className="absolute left-[42%] top-[65%] h-4 w-4 rounded-full bg-primary" />
        <span className="absolute left-[62%] top-[42%] h-5 w-5 rounded-full bg-pink-300" />
        <span className="absolute left-[25%] top-[58%] h-[2px] w-[18%] rotate-[23deg] bg-slate-300/60" />
        <span className="absolute left-[45%] top-[58%] h-[2px] w-[18%] rotate-[-28deg] bg-slate-300/60" />
      </div>
      <p className="mt-3 text-xs leading-relaxed text-on-surface-variant">{body}</p>
    </JourneyCard>
  );
}
