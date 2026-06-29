import {
  ActionLink,
  Card,
  Chip,
  EvidenceCard,
  Metric,
  MetricStrip,
  PageHero,
  StatusBadge,
} from "@/components/v2/v2-shell";
import { EvidenceProvenanceSummary } from "@/components/v2/evidence-provenance";
import { EvidenceWarningBanner } from "@/components/v2/evidence-warning";
import { ProteinStructureViewer } from "@/components/v2/protein-structure-viewer";
import { buildV2Params, parseV2Session, type V2SearchParams } from "@/lib/v2-data";
import { loadV2RepurposingEvidence } from "@/lib/v2-live-data";
import { getRepurposingEvidenceBundleUrl } from "@/lib/api";
import type { RepurposingCandidate, RepurposingDisease } from "@/lib/api";

function readParam(
  params: V2SearchParams,
  key: string,
): string | undefined {
  if ("get" in params && typeof params.get === "function") {
    return params.get(key) ?? undefined;
  }
  const record = params as Partial<Record<string, string | string[]>>;
  const value = record[key];
  return Array.isArray(value) ? value[0] : value;
}

export default async function V2RepurposingPage({
  searchParams,
}: {
  searchParams: Promise<V2SearchParams>;
}) {
  const resolvedParams = await searchParams;
  const session = parseV2Session(resolvedParams);
  const selectedDiseaseId = readParam(resolvedParams, "disease_id") ?? "brca_external_validation";
  const evidence = await loadV2RepurposingEvidence(selectedDiseaseId);
  const response = evidence.candidates.data;
  const diseases = evidence.diseases.data?.diseases ?? fallbackDiseases;
  const selectedDisease = response?.disease ?? diseases.find((item) => item.id === selectedDiseaseId) ?? fallbackDiseases[0];
  const candidates = response?.candidates ?? [];
  const warnings = [
    response?.message ?? evidence.candidates.message,
    "This page ranks research hypotheses only; it does not produce cures, prescriptions, or clinical recommendations.",
    ...(candidates[0]?.audit.warnings ?? []),
  ].filter((item): item is string => Boolean(item));
  const routeParams = buildV2Params(session);
  const exportJsonUrl = getRepurposingEvidenceBundleUrl(selectedDisease.id, "json");
  const exportMarkdownUrl = getRepurposingEvidenceBundleUrl(selectedDisease.id, "markdown");

  return (
    <div className="space-y-6">
      <PageHero
        eyebrow="Repurposing"
        title="Rank disease-focused drug repurposing hypotheses with RNA-seq, KG, structure, and quantum evidence."
        actions={
          <>
            <ActionLink href={`/v2/visual${routeParams}`}>Open visual evidence</ActionLink>
            <ActionLink href={`/v2/validation${routeParams}`} variant="secondary">
              Validate current candidate
            </ActionLink>
            <ActionLink href={exportJsonUrl} variant="secondary">Export evidence JSON</ActionLink>
            <ActionLink href={exportMarkdownUrl} variant="secondary">Export report</ActionLink>
          </>
        }
      >
        The workbench starts from audited counts-level RNA-seq evidence, keeps local structure artifacts explicit,
        and shows quantum results as benchmark overlays unless audits support stronger claims.
      </PageHero>

      <EvidenceWarningBanner warnings={warnings} />

      <MetricStrip>
        <Metric
          label="Disease"
          value={selectedDisease.name}
          detail={selectedDisease.cohort}
          help="The first MVP is seeded from audited local cohorts and conservative fallback examples."
        />
        <Metric
          label="Samples"
          value={String(selectedDisease.sample_count)}
          detail={`Smallest class ${selectedDisease.smallest_class_count}`}
          help="Cohort scale for RNA-seq evidence. Fallback-only entries show zero because no local cohort is attached."
        />
        <Metric
          label="Candidates"
          value={String(candidates.length)}
          detail={evidence.candidates.source}
          help="Shows live API-backed candidates when available, otherwise an explicitly labeled fallback."
        />
        <Metric
          label="Policy"
          value="Hypotheses"
          detail="No cure or clinical efficacy claim"
          help="The audit layer blocks cure, treatment, and quantum-advantage language unless evidence gates allow it."
        />
      </MetricStrip>

      <section className="grid gap-4 xl:grid-cols-[0.75fr_1.25fr]">
        <Card
          title="Disease selector"
          kicker={evidence.diseases.source}
          help="Choose the disease evidence package used to build the candidate ranking."
        >
          <div className="space-y-3">
            {diseases.map((disease) => (
              <DiseaseRow key={disease.id} disease={disease} active={disease.id === selectedDisease.id} />
            ))}
          </div>
          <div className="mt-4">
            <EvidenceProvenanceSummary
              title="Disease evidence source"
              provenance={evidence.diseases.provenance}
            />
          </div>
        </Card>

        <Card
          title="Scoring modes"
          kicker="Ablation-ready"
          help="Modes are compared so structure and quantum evidence do not silently replace classical baselines."
        >
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            {(response?.scoring_modes ?? fallbackScoringModes).map((mode) => (
              <EvidenceCard
                key={mode}
                title={mode.replaceAll("_", " + ")}
                value={mode.includes("quantum") ? "Overlay" : "Input"}
                detail={detailForMode(mode)}
                tone={mode.includes("quantum") ? "quantum" : "success"}
              />
            ))}
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            <Chip active>Local only</Chip>
            <Chip active>Open source</Chip>
            <Chip active={response?.manifest.openfold_runner === "deferred"}>OpenFold artifact-first</Chip>
            <Chip active={response?.manifest.paid_or_hosted_services_required === false}>No paid service</Chip>
          </div>
        </Card>
      </section>

      <Card
        title="Ranked hypotheses"
        kicker={selectedDisease.evidence_status}
        help="Each row keeps the evidence split visible: KG, RNA-seq, structure artifacts, classical baseline, quantum benchmark, and audit guardrails."
      >
        <div className="space-y-4">
          {candidates.length > 0 ? (
            candidates.map((candidate) => <CandidateRow key={candidate.compound_id} candidate={candidate} />)
          ) : (
            <p className="rounded-lg border border-outline-variant/25 bg-surface-container-high p-4 text-sm text-on-surface-variant">
              No repurposing candidates were returned by the API. Check the backend route and local artifacts.
            </p>
          )}
        </div>
      </Card>

      <section className="grid gap-4 lg:grid-cols-2">
        <Card title="Run manifest" kicker="Traceability" help="Records the local-first policy and scoring setup for this workbench run.">
          <pre className="max-h-80 overflow-auto rounded-lg border border-outline-variant/25 bg-surface-container-high p-3 text-xs text-on-surface">
            {JSON.stringify(response?.manifest ?? { status: "unavailable" }, null, 2)}
          </pre>
        </Card>
        <Card title="Candidate provenance" kicker={evidence.candidates.source} help="The evidence trail includes RNA-seq artifacts and structure registries when present.">
          <EvidenceProvenanceSummary
            title="Repurposing source"
            provenance={evidence.candidates.provenance}
          />
        </Card>
      </section>
    </div>
  );
}

function DiseaseRow({ disease, active }: { disease: RepurposingDisease; active: boolean }) {
  const tone =
    disease.evidence_status === "review_ready"
      ? "success"
      : disease.evidence_status === "alternate_omics_policy"
        ? "quantum"
        : "warning";
  return (
    <a
      href={`/v2/repurposing?disease_id=${encodeURIComponent(disease.id)}`}
      className={`block rounded-lg border p-3 transition-colors ${
        active
          ? "border-primary/50 bg-primary/15"
          : "border-outline-variant/25 bg-surface-container-high hover:border-primary/40"
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="font-semibold text-on-surface">{disease.name}</p>
          <p className="mt-1 text-xs leading-relaxed text-on-surface-variant">{disease.cohort}</p>
        </div>
        <StatusBadge tone={tone}>
          {disease.evidence_status}
        </StatusBadge>
      </div>
    </a>
  );
}

function CandidateRow({ candidate }: { candidate: RepurposingCandidate }) {
  const quantumDelta = candidate.quantum_benchmark.delta_vs_classical;
  const structureTargets = candidate.structure_targets;
  const visibleTargets = structureTargets.target_ids.slice(0, 6);
  const hiddenTargetCount = Math.max(0, structureTargets.target_count - visibleTargets.length);
  const visibleProteinRows = candidate.protein_structures.slice(0, 4);
  const selectedProtein = candidate.protein_structures.find((protein) => protein.viewer.supports_3d) ?? null;
  const creedsStatus = String(
    (candidate.rnaseq_signature as Record<string, unknown>)?.creeds_match_status ?? "",
  );
  const creedsLabel =
    creedsStatus === "matched_human"
      ? "Human CREEDS profile"
      : creedsStatus === "matched_non_human"
        ? "Non-human CREEDS profile"
        : creedsStatus === "unmatched"
          ? "No CREEDS profile"
          : null;
  return (
    <article className="rounded-lg border border-outline-variant/25 bg-surface-container-high/70 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="font-label text-xs font-bold uppercase tracking-widest text-primary">
            Rank {candidate.rank} | {candidate.scoring_mode.replaceAll("_", " + ")}
          </p>
          <h2 className="mt-1 font-headline text-xl font-semibold text-on-surface">
            {candidate.compound_name} for {candidate.disease_name}
          </h2>
          <p className="mt-2 max-w-4xl text-sm leading-relaxed text-on-surface-variant">
            {candidate.summary}
          </p>
          {creedsLabel ? (
            <p className="mt-2">
              <Chip active={creedsStatus === "matched_human"}>
                {creedsLabel}
              </Chip>
            </p>
          ) : null}
        </div>
        <div className="text-right">
          <p className="font-label text-[0.65rem] font-bold uppercase tracking-widest text-on-surface-variant">
            Hypothesis score
          </p>
          <p className="mt-1 font-mono text-2xl font-semibold text-primary">
            {candidate.hypothesis_score.toFixed(3)}
          </p>
        </div>
      </div>

      <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-5">
        {candidate.evidence_components.map((item) => (
          <EvidenceCard
            key={item.label}
            title={item.label}
            value={item.value}
            detail={item.detail}
            tone={
              item.label === "CREEDS" && item.status === "missing"
                ? "warning"
                : item.status === "supporting"
                  ? "success"
                  : "warning"
            }
          />
        ))}
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-[1fr_1fr_1fr]">
        <div>
          <p className="font-label text-xs font-bold uppercase tracking-widest text-on-surface-variant">KG paths</p>
          <ul className="mt-2 space-y-2 text-xs text-on-surface-variant">
            {candidate.kg_paths.map((path) => <li key={path}>{path}</li>)}
          </ul>
        </div>
        <div>
          <p className="font-label text-xs font-bold uppercase tracking-widest text-on-surface-variant">Structure targets</p>
          <p className="mt-2 text-sm text-on-surface">
            {structureTargets.structure_artifact_target_count}/{structureTargets.target_count} mapped targets have local structure artifacts
          </p>
          <p className="mt-1 text-xs text-on-surface-variant">
            {structureTargets.mapping_status} | parsed {structureTargets.parsed_structure_count} | feature score {structureTargets.structure_feature_score.toFixed(3)}
          </p>
          <p className="mt-2 break-words text-xs text-on-surface-variant">
            {visibleTargets.join(", ") || "No KG targets mapped"}{hiddenTargetCount > 0 ? `, +${hiddenTargetCount} more` : ""}
          </p>
          {visibleProteinRows.length > 0 ? (
            <ul className="mt-3 space-y-2 text-xs text-on-surface-variant">
              {visibleProteinRows.map((protein) => (
                <li key={protein.target_id} className="flex items-center justify-between gap-3 rounded-md border border-outline-variant/20 px-2 py-1">
                  <span className="min-w-0 break-words">{protein.display_name}</span>
                  <span className={protein.viewer.supports_3d ? "text-primary" : "text-on-surface-variant"}>
                    {protein.viewer.supports_3d ? "3D ready" : "missing local artifact"}
                  </span>
                </li>
              ))}
            </ul>
          ) : null}
        </div>
        <div>
          <p className="font-label text-xs font-bold uppercase tracking-widest text-on-surface-variant">Quantum benchmark</p>
          <p className="mt-2 text-sm text-on-surface">
            Delta vs classical: {typeof quantumDelta === "number" ? quantumDelta.toFixed(4) : "not available"}
          </p>
          <p className="mt-1 text-xs text-on-surface-variant">
            Quantum advantage claim allowed: {candidate.audit.quantum_advantage_claim_allowed ? "yes" : "no"}
          </p>
        </div>
      </div>

      {selectedProtein ? (
        <div className="mt-4 rounded-lg border border-outline-variant/25 bg-surface-container-low p-3">
          <p className="font-label text-xs font-bold uppercase tracking-widest text-on-surface-variant">Local protein structure</p>
          <div className="mt-3">
            <ProteinStructureViewer protein={selectedProtein} />
          </div>
        </div>
      ) : null}

      <div className="mt-4 rounded-lg border border-outline-variant/25 bg-surface-container-low p-3">
        <p className="font-label text-xs font-bold uppercase tracking-widest text-on-surface-variant">Audit policy</p>
        <p className="mt-2 text-xs leading-relaxed text-on-surface-variant">{candidate.audit.claim_policy}</p>
      </div>
    </article>
  );
}

function detailForMode(mode: string): string {
  if (mode === "kg_only") return "Graph evidence baseline.";
  if (mode === "kg_plus_rnaseq") return "Adds audited disease signature evidence.";
  if (mode === "kg_plus_rnaseq_plus_structure") return "Adds local target structure features.";
  return "Quantum/classical comparison only.";
}

const fallbackScoringModes = [
  "kg_only",
  "kg_plus_rnaseq",
  "kg_plus_rnaseq_plus_structure",
  "quantum_benchmark_overlay",
];

const fallbackDiseases: RepurposingDisease[] = [
  {
    id: "brca_external_validation",
    name: "Breast cancer",
    cohort: "TCGA-BRCA to GEO GSE225846",
    source: "fallback",
    sample_count: 215,
    smallest_class_count: 30,
    evidence_status: "review_ready",
    notes: [],
  },
  {
    id: "brca_external_validation_organism_any",
    name: "Breast cancer (CREEDS any organism)",
    cohort: "TCGA-BRCA to GEO GSE225846",
    source: "fallback",
    sample_count: 215,
    smallest_class_count: 30,
    evidence_status: "alternate_omics_policy",
    notes: [],
  },
  {
    id: "all_pairs_kg_omics",
    name: "All diseases (200-pair KG+omics)",
    cohort: "Multiseed KG + CREEDS cosine",
    source: "fallback",
    sample_count: 0,
    smallest_class_count: 0,
    evidence_status: "exploratory_cross_disease",
    notes: [],
  },
];
