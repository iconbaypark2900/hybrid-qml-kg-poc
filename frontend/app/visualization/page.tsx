"use client";

import { Suspense, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import {
  JourneyButton,
  JourneyCard,
  MetricTile,
  MiniCircuit,
  PageHero,
  Pill,
  journeyQuery,
  readJourneySetup,
  titleCase,
} from "@/components/journey-ui";

type VisualLens = "molecule" | "target" | "mechanism" | "graph" | "quantum" | "compare";

const lenses: Array<{ id: VisualLens; label: string; detail: string }> = [
  { id: "molecule", label: "Molecule", detail: "Compound shape and atoms" },
  { id: "target", label: "Target pocket", detail: "Protein and ligand context" },
  { id: "mechanism", label: "Mechanism", detail: "Drug to disease path" },
  { id: "graph", label: "Graph evidence", detail: "Hetionet relationship neighborhood" },
  { id: "quantum", label: "Quantum circuit", detail: "Feature-map branch" },
  { id: "compare", label: "Disagreement", detail: "Classical vs hybrid vs quantum" },
];

export default function VisualizationPage() {
  return (
    <Suspense fallback={null}>
      <VisualizationInner />
    </Suspense>
  );
}

function VisualizationInner() {
  const searchParams = useSearchParams();
  const setup = useMemo(
    () => readJourneySetup(new URLSearchParams(searchParams.toString())),
    [searchParams],
  );
  const [active, setActive] = useState<VisualLens>("molecule");
  const query = journeyQuery(setup);

  return (
    <div className="space-y-4">
      <PageHero
        eyebrow="Visual"
        title="Inspect the evidence from multiple scientific angles."
        body="Visual is the evidence studio. It should help a new user manipulate the knowledge, see why a candidate surfaced, and understand the model result without needing to read raw graph tables first."
        actions={
          <>
            <JourneyButton href={`/validation?${query}`} tone="primary">
              Back to validation
            </JourneyButton>
            <JourneyButton href={`/experiments?${query}`}>Compare setup</JourneyButton>
          </>
        }
      />

      <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <MetricTile
          label="Candidate"
          value={`${setup.drug} -> ${setup.disease}`}
          detail="The pair being inspected"
        />
        <MetricTile
          label="Run path"
          value={titleCase(setup.mode)}
          detail={`${titleCase(setup.mechanism)} mechanism evidence`}
        />
        <MetricTile
          label="Visual lens"
          value={lenses.find((lens) => lens.id === active)?.label ?? "Molecule"}
          detail="User-controlled evidence scene"
        />
        <MetricTile
          label="Quantum branch"
          value="Circuit"
          detail="Feature-map explanation available"
        />
      </section>

      <section className="grid gap-4 xl:grid-cols-[minmax(0,1.1fr)_420px]">
        <JourneyCard
          title="3D evidence studio"
          kicker="Interactive concept"
          help="This area is where the production app should mount real 3D molecule, target, pathway, KG, and quantum-circuit viewers."
        >
          <div className="mb-4 flex flex-wrap gap-2">
            {lenses.map((lens) => (
              <Pill
                key={lens.id}
                active={active === lens.id}
                onClick={() => setActive(lens.id)}
                help={lens.detail}
              >
                {lens.label}
              </Pill>
            ))}
          </div>
          <EvidenceScene active={active} drug={setup.drug} disease={setup.disease} />
        </JourneyCard>

        <JourneyCard
          title="What this visual answers"
          kicker="Plain language"
          help="Each lens should answer a different user question instead of showing graph data for its own sake."
        >
          <div className="space-y-3">
            {lenses.map((lens) => (
              <button
                key={lens.id}
                type="button"
                onClick={() => setActive(lens.id)}
                className={`w-full rounded-lg border p-3 text-left transition-colors ${
                  active === lens.id
                    ? "border-tertiary/40 bg-tertiary/10"
                    : "border-outline/10 bg-surface-container-lowest/50 hover:border-primary/30"
                }`}
              >
                <p className="font-semibold text-on-surface">{lens.label}</p>
                <p className="mt-1 text-xs leading-relaxed text-on-surface-variant">
                  {lens.detail}
                </p>
              </button>
            ))}
          </div>
        </JourneyCard>
      </section>

      <section className="grid gap-4 xl:grid-cols-3">
        <JourneyCard
          title="Molecule"
          kicker="Compound conformer"
          help="Use a real molecular viewer here so users can inspect atom layout and chemical structure."
        >
          <MiniMolecule />
        </JourneyCard>
        <JourneyCard
          title="Graph path"
          kicker="Hetionet context"
          help="This should show the specific relationship path that supports or weakens the candidate."
        >
          <MiniGraph drug={setup.drug} disease={setup.disease} />
        </JourneyCard>
        <JourneyCard
          title="Quantum circuit"
          kicker="Feature map"
          help="The circuit explains the quantum comparison branch without claiming it is the sole source of truth."
        >
          <MiniCircuit />
        </JourneyCard>
      </section>
    </div>
  );
}

function EvidenceScene({
  active,
  drug,
  disease,
}: {
  active: VisualLens;
  drug: string;
  disease: string;
}) {
  if (active === "quantum") {
    return (
      <div className="rounded-lg border border-outline/10 bg-surface-container-lowest/60 p-4">
        <MiniCircuit />
        <p className="mt-4 text-sm leading-relaxed text-on-surface-variant">
          The quantum branch maps selected graph and embedding features into a
          circuit representation, then compares whether that branch agrees with
          the classical models.
        </p>
      </div>
    );
  }

  if (active === "compare") {
    return (
      <div className="grid gap-3 md:grid-cols-3">
        <CompareBar label="Classical" value={82} />
        <CompareBar label="Hybrid" value={84} />
        <CompareBar label="Quantum" value={72} />
      </div>
    );
  }

  return (
    <div className="relative min-h-[440px] overflow-hidden rounded-lg border border-outline/10 bg-[#050b19]">
      <div className="absolute inset-0 opacity-80">
        <div className="absolute left-[12%] top-[20%] h-10 w-10 rounded-full bg-tertiary shadow-[0_0_32px_rgba(62,250,227,0.55)]" />
        <div className="absolute left-[25%] top-[34%] h-7 w-7 rounded-full bg-primary shadow-[0_0_24px_rgba(123,208,255,0.55)]" />
        <div className="absolute left-[37%] top-[22%] h-8 w-8 rounded-full bg-pink-300 shadow-[0_0_24px_rgba(249,168,212,0.45)]" />
        <div className="absolute left-[18%] top-[25%] h-[2px] w-[12%] origin-left rotate-[28deg] bg-slate-300/55" />
        <div className="absolute left-[29%] top-[31%] h-[2px] w-[10%] origin-left rotate-[-28deg] bg-slate-300/55" />

        <div className="absolute right-[17%] top-[18%] h-28 w-44 rounded-[50%] border-4 border-secondary/60" />
        <div className="absolute right-[28%] top-[29%] h-8 w-8 rounded-full bg-tertiary" />
        <div className="absolute right-[20%] top-[36%] h-7 w-7 rounded-full bg-pink-300" />

        <div className="absolute bottom-[20%] left-[43%] h-14 w-14 rounded-full bg-tertiary/70 blur-[1px]" />
        <div className="absolute bottom-[26%] left-[56%] h-11 w-11 rounded-full bg-yellow-300/60" />
        <div className="absolute bottom-[18%] left-[68%] h-12 w-12 rounded-full bg-pink-400/55" />
        <div className="absolute bottom-[25%] left-[47%] h-[2px] w-[12%] origin-left rotate-[-12deg] bg-slate-300/45" />
        <div className="absolute bottom-[27%] left-[59%] h-[2px] w-[10%] origin-left rotate-[18deg] bg-slate-300/45" />
      </div>
      <div className="relative z-10 flex h-full min-h-[440px] flex-col justify-between p-5">
        <div>
          <p className="journey-eyebrow">{active}</p>
          <h2 className="max-w-xl font-headline text-3xl font-bold text-on-surface">
            {sceneTitle(active, drug, disease)}
          </h2>
        </div>
        <div className="grid gap-3 md:grid-cols-3">
          {sceneFacts(active).map((fact) => (
            <div
              key={fact}
              className="rounded-lg border border-outline/15 bg-surface-container-lowest/75 p-3 text-xs leading-relaxed text-on-surface-variant"
            >
              {fact}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function sceneTitle(active: VisualLens, drug: string, disease: string) {
  if (active === "target") return `Where ${drug} may touch target biology.`;
  if (active === "mechanism") return `How ${drug} could connect to ${disease}.`;
  if (active === "graph") return "Which graph relationships support the route.";
  return `Molecular evidence for ${drug}.`;
}

function sceneFacts(active: VisualLens) {
  if (active === "target") {
    return ["Protein pocket context", "Ligand proximity", "Target confidence"];
  }
  if (active === "mechanism") {
    return ["Drug to target", "Target to pathway", "Pathway to disease"];
  }
  if (active === "graph") {
    return ["Compound nodes", "Gene/protein links", "Disease neighborhood"];
  }
  return ["Atoms and bonds", "Conformer shape", "Chemical plausibility"];
}

function MiniMolecule() {
  return (
    <div className="relative h-40 overflow-hidden rounded-lg border border-outline/10 bg-surface-container-lowest/60">
      <span className="absolute left-[18%] top-[52%] h-4 w-4 rounded-full bg-tertiary" />
      <span className="absolute left-[34%] top-[62%] h-4 w-4 rounded-full bg-primary" />
      <span className="absolute left-[51%] top-[47%] h-5 w-5 rounded-full bg-pink-300" />
      <span className="absolute left-[69%] top-[34%] h-5 w-5 rounded-full bg-tertiary" />
      <span className="absolute left-[23%] top-[57%] h-[2px] w-[14%] rotate-[25deg] bg-slate-300/60" />
      <span className="absolute left-[39%] top-[58%] h-[2px] w-[13%] rotate-[-25deg] bg-slate-300/60" />
      <span className="absolute left-[56%] top-[45%] h-[2px] w-[13%] rotate-[-20deg] bg-slate-300/60" />
    </div>
  );
}

function MiniGraph({ drug, disease }: { drug: string; disease: string }) {
  return (
    <div className="space-y-3">
      {[
        [drug, "Compound"],
        ["ABL1 / pathway context", "Target"],
        [disease, "Disease"],
      ].map(([name, type], index) => (
        <div
          key={name}
          className="flex items-center justify-between rounded-lg border border-outline/10 bg-surface-container-lowest/60 p-3"
        >
          <span className="font-semibold text-on-surface">{name}</span>
          <span className="text-xs text-on-surface-variant">{type}</span>
          {index < 2 ? <span className="text-primary">linked</span> : null}
        </div>
      ))}
    </div>
  );
}

function CompareBar({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-lg border border-outline/10 bg-surface-container-lowest/60 p-4">
      <div className="flex items-center justify-between">
        <p className="font-semibold text-on-surface">{label}</p>
        <span className="font-mono text-sm text-tertiary">{value / 100}</span>
      </div>
      <div className="mt-4 h-2 rounded-full bg-surface-container-high">
        <div
          className="h-2 rounded-full bg-tertiary"
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
}
