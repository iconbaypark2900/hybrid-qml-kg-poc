import {
  ActionLink,
  Card,
  Chip,
  PageHero,
  StatusBadge,
} from "@/components/v2/v2-shell";
import { CircuitEvidence } from "@/components/v2/circuit-evidence";
import {
  CandidateEvidenceLinks,
  EvidenceProvenanceSummary,
} from "@/components/v2/evidence-provenance";
import { EvidenceWarningBanner } from "@/components/v2/evidence-warning";
import { KGEvidenceGraph } from "@/components/v2/kg-evidence-graph";
import { MoleculeViewer } from "@/components/v2/molecule-viewer";
import { VisualOverlays } from "@/components/v2/visual-overlays";
import {
  buildV2Params,
  parseV2Session,
  type V2SearchParams,
} from "@/lib/v2-data";
import {
  loadV2ExperimentEvidence,
  loadV2VisualEvidence,
} from "@/lib/v2-live-data";
import {
  buildCircuitOverlays,
  buildGraphOverlays,
  buildMoleculeOverlays,
} from "@/lib/v2-quality-data";

export default async function V2VisualPage({
  searchParams,
}: {
  searchParams: Promise<V2SearchParams>;
}) {
  const session = parseV2Session(await searchParams);
  const candidate = session.selectedCandidate;
  const [evidence, experimentEvidence] = await Promise.all([
    loadV2VisualEvidence(session),
    loadV2ExperimentEvidence(session),
  ]);
  const routeParams = buildV2Params(session);
  const visualWarnings = [
    evidence.molecule.source !== "live"
      ? "Molecule geometry is fallback or unavailable; do not interpret spatial geometry."
      : null,
    evidence.graph.source !== "live"
      ? "KG graph is fallback or unavailable; inspect provenance before using graph evidence."
      : null,
    evidence.circuit.source !== "live"
      ? "Circuit panel is fallback or unavailable; treat circuit labels as configured defaults."
      : null,
    "Clinical validation evidence is curated/paper-aligned until clinical trial ingestion is implemented.",
  ].filter((warning): warning is string => Boolean(warning));

  return (
    <div className="space-y-6">
      <PageHero
        eyebrow="Visual"
        title="Inspect the evidence as a connected 3D studio."
        actions={
          <>
            <ActionLink href={`/v2/validation${routeParams}`}>
              Back to validation
            </ActionLink>
            <ActionLink href={`/v2/start${routeParams}`} variant="secondary">
              Change starting point
            </ActionLink>
          </>
        }
      >
        Visual keeps the candidate in view while showing the molecule,
        target/pocket, pathway context, Hetionet neighborhood, relation
        evidence, and the quantum circuit branch that contributed to the setup.
        {session.sessionId ? (
          <span className="mt-2 block font-mono text-xs text-primary">
            Saved session: {session.sessionId}
          </span>
        ) : null}
      </PageHero>

      <EvidenceWarningBanner warnings={visualWarnings} />

      <section className="grid gap-4 xl:grid-cols-[1.1fr_.9fr]">
        <Card
          title="Molecular scene"
          kicker={`${candidate.candidate} to ${candidate.disease}`}
          help="A spatial preview of the compound and the nearby evidence layers. The production version can replace this with a live molecular renderer."
        >
          <MoleculeViewer
            compoundName={evidence.molecule.compoundName}
            atoms={evidence.molecule.atoms}
            bonds={evidence.molecule.bonds}
            source={evidence.molecule.source}
            message={evidence.molecule.message}
          />
          <div className="mt-3">
            <VisualOverlays overlays={buildMoleculeOverlays(evidence.molecule)} />
          </div>
          <div className="mt-3">
            <EvidenceProvenanceSummary
              title="Molecule source"
              provenance={evidence.molecule.provenance}
            />
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            <Chip active>Compound</Chip>
            <Chip active>Protein target</Chip>
            <Chip active={candidate.evidencePosture === "Validated"}>
              Clinical support
            </Chip>
            <Chip>{candidate.disease}</Chip>
          </div>
        </Card>

        <Card
          title="Target pocket"
          kicker="Protein plus ligand"
          help="This panel shows how the candidate could be reviewed against target-level evidence before accepting it as plausible."
        >
          <div className="relative min-h-[360px] overflow-hidden rounded-lg border border-outline-variant/25 bg-surface-container-lowest p-5">
            <div className="absolute left-1/2 top-1/2 h-44 w-64 -translate-x-1/2 -translate-y-1/2 rounded-[45%] border-4 border-secondary/40" />
            <div className="absolute left-[42%] top-[45%] h-9 w-9 rounded-full bg-tertiary shadow-glow" />
            <div className="absolute left-[55%] top-[56%] h-7 w-7 rounded-full bg-error" />
            <div className="absolute left-[62%] top-[32%] h-5 w-5 rounded-full bg-primary" />
            <div className="absolute bottom-5 left-5 right-5 rounded-lg border border-outline-variant/25 bg-surface-container-high/80 p-3">
              <p className="font-label text-xs font-bold uppercase tracking-widest text-on-surface-variant">
                Pocket readout
              </p>
              <p className="mt-2 text-sm text-on-surface">
                {candidate.evidencePosture === "Validated"
                  ? "Clinical support is present, but the mechanism trail should stay attached to the decision."
                  : "Target context is present, but mechanism validation should decide whether the visual support is strong enough."}
              </p>
            </div>
          </div>
        </Card>
      </section>

      <section className="grid gap-4 xl:grid-cols-3">
        <Card
          title="Pathway context"
          kicker="Mechanism route"
          help="Shows the biological route that made the candidate more plausible under the mechanism-first lens."
        >
          <div className="space-y-3">
            {candidate.pathway.map((item, index) => (
                <div key={item} className="flex items-center gap-3">
                  <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/15 font-mono text-xs font-bold text-primary">
                    {index + 1}
                  </span>
                  <div className="flex-1 rounded-lg border border-outline-variant/25 bg-surface-container-high p-3">
                    <p className="text-sm font-semibold text-on-surface">{item}</p>
                  </div>
                </div>
              ))}
          </div>
        </Card>

        <Card
          title="Hetionet neighborhood"
          kicker="Graph evidence"
          help="A compact graph view for the relationships around the selected compound-disease candidate."
        >
          <KGEvidenceGraph evidence={evidence.graph} candidate={candidate} />
          <div className="mt-3">
            <VisualOverlays overlays={buildGraphOverlays(evidence.graph)} />
          </div>
          <div className="mt-3">
            <EvidenceProvenanceSummary
              title="KG source"
              provenance={evidence.graph.provenance}
            />
          </div>
        </Card>

        <Card
          title="Relation evidence"
          kicker="Why the graph believes"
          help="Turns raw graph links into readable evidence labels."
        >
          <div className="space-y-3">
            {candidate.relationEvidence.map(({ code, text, strength }) => (
              <div
                key={code}
                className="flex items-center justify-between gap-3 rounded-lg border border-outline-variant/25 bg-surface-container-high p-3"
              >
                <div>
                  <p className="font-mono text-sm font-bold text-primary">
                    {code}
                  </p>
                  <p className="mt-1 text-xs text-on-surface-variant">{text}</p>
                </div>
                <StatusBadge tone={strength === "check" ? "warning" : "success"}>
                  {strength}
                </StatusBadge>
              </div>
            ))}
          </div>
        </Card>
      </section>

      <Card
        title="Quantum circuit branch"
        kicker={`${session.runMode} feature-map comparison`}
        help="This shows the quantum branch from the hybrid paper flow: PCA-reduced pair features enter a 16-qubit Pauli feature map, then a fidelity-kernel score is compared with classical signals."
      >
        <CircuitEvidence evidence={evidence.circuit} />
        <div className="mt-3">
          <VisualOverlays overlays={buildCircuitOverlays(evidence.circuit)} />
        </div>
        <div className="mt-3">
          <EvidenceProvenanceSummary
            title="Circuit source"
            provenance={evidence.circuit.provenance}
          />
        </div>
        <div className="mt-5 grid gap-3 md:grid-cols-4">
          {[
            ["Input", "Pair embedding features"],
            ["PCA", "Reduce to quantum-ready dimensions"],
            ["16Q Pauli map", "Encode relationships as rotations"],
            ["Kernel score", "Add decorrelated signal to the stack"],
          ].map(([title, body]) => (
            <div
              key={title}
              className="rounded-lg border border-outline-variant/25 bg-surface-container-high p-3"
            >
              <p className="font-semibold text-on-surface">{title}</p>
              <p className="mt-1 text-xs leading-relaxed text-on-surface-variant">
                {body}
              </p>
            </div>
          ))}
          </div>
      </Card>

      <Card
        title="Candidate evidence links"
        kicker="Traceable support"
        help="Structured references connecting the candidate to model, KG, trial, and mechanism evidence."
      >
        <CandidateEvidenceLinks links={experimentEvidence.predictions.evidenceLinks} />
      </Card>
    </div>
  );
}

