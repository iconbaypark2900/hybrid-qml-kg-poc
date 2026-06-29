#!/usr/bin/env python3
"""
Full Drug Repurposing Pipeline — orchestrates all layers.

Reads top-K candidate compound-disease pairs from existing KG+QML scoring
outputs (or generates a small demo set if none are present), enriches each
with single-cell, perturbation, and validation evidence, fuses them with
the evidence layer, and writes a ranked report to artifacts/predictions/.

Modes:
    --mode kg-only   omics features are zero-filled; preserves PR-AUC 0.7987 baseline
    --mode kg+omics  full stack (KG + QML + reversal + clinical)

Flags:
    --validate       run validation_layer (ClinicalTrials.gov + literature) on top-N
    --disease ID     run for a single disease (Hetionet Disease:: ID); default: demo set
    --top-n N        candidates to report (default 50)
    --output DIR     output directory (default artifacts/predictions/)

Usage:
    python scripts/run_full_repurposing_pipeline.py --mode kg-only
    python scripts/run_full_repurposing_pipeline.py --mode kg+omics --validate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Repo root on path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_full_repurposing_pipeline")


# ---------- Demo candidate generator (when no upstream KG scores are present) ----------

_DEMO_CANDIDATES: List[Tuple[str, str, str, str]] = [
    # (compound_name, compound_id, disease_name, disease_id)
    ("metformin", "Compound::DB00331", "type 2 diabetes mellitus", "Disease::DOID:9352"),
    ("aspirin", "Compound::DB00945", "diabetes mellitus", "Disease::DOID:9351"),
    ("rosuvastatin", "Compound::DB01098", "hyperlipidemia", "Disease::DOID:1387"),
    ("losartan", "Compound::DB00678", "hypertension", "Disease::DOID:10763"),
    ("sertraline", "Compound::DB01104", "major depressive disorder", "Disease::DOID:1470"),
    ("imatinib", "Compound::DB00619", "chronic myeloid leukemia", "Disease::DOID:8552"),
    ("tamoxifen", "Compound::DB00675", "breast cancer", "Disease::DOID:1612"),
    ("warfarin", "Compound::DB00682", "atrial fibrillation", "Disease::DOID:0060224"),
    ("levothyroxine", "Compound::DB00451", "hypothyroidism", "Disease::DOID:1459"),
    ("omeprazole", "Compound::DB00338", "gastroesophageal reflux disease", "Disease::DOID:8534"),
    ("simvastatin", "Compound::DB00641", "hypercholesterolemia", "Disease::DOID:2487"),
    ("amlodipine", "Compound::DB00381", "hypertension", "Disease::DOID:10763"),
]


def load_kg_qml_scores(scores_path: Optional[Path]) -> List[Dict]:
    """
    Load existing KG+QML candidate scores from a prior pipeline run.

    Expected JSON schema (list of objects):
        {compound, compound_hetionet_id, disease, disease_hetionet_id,
         kg_rotate_score, kg_complex_score, qsvc_score, classical_ensemble_score, ...}

    Returns [] if no upstream file is found — the caller will fall back to the
    demo candidate set so the pipeline always exits with a usable artifact.
    """
    if scores_path is None or not scores_path.exists():
        logger.info("No upstream KG+QML scores found; using demo candidate set.")
        return []
    try:
        with scores_path.open(encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} candidates from {scores_path}")
        return data
    except Exception as e:
        logger.warning(f"Failed to load {scores_path}: {e}; falling back to demo set.")
        return []


def build_demo_candidates(disease_filter: Optional[str] = None) -> List[Dict]:
    """Build a demo candidate set with seeded KG and QSVC scores."""
    import numpy as np
    rng = np.random.default_rng(42)
    out: List[Dict] = []
    for cname, cid, dname, did in _DEMO_CANDIDATES:
        if disease_filter and did != disease_filter:
            continue
        out.append({
            "compound": cname,
            "compound_hetionet_id": cid,
            "disease": dname,
            "disease_hetionet_id": did,
            "kg_rotate_score": float(rng.uniform(0.55, 0.92)),
            "kg_complex_score": float(rng.uniform(0.50, 0.88)),
            "qsvc_score": float(rng.uniform(0.45, 0.85)),
            "classical_ensemble_score": float(rng.uniform(0.50, 0.90)),
            "graph_topology_score": float(rng.uniform(0.40, 0.80)),
        })
    logger.info(f"Generated {len(out)} demo candidates")
    return out


# ---------- Omics enrichment (reversal scores) ----------

def _attach_synthetic_reversal_scores(candidates: List[Dict]) -> None:
    import numpy as np

    rng = np.random.default_rng(seed=1729)
    for c in candidates:
        base = (c.get("kg_rotate_score", 0.5) + c.get("qsvc_score", 0.5)) / 2.0
        c["signature_reversal_score"] = float(np.clip(base + rng.normal(0, 0.08), -1.0, 1.0))
        c["cell_type_reversal_score"] = 0.0
        c["pathway_reversal_score"] = 0.0


def enrich_with_omics(
    candidates: List[Dict],
    mode: str,
    *,
    creeds_path: Optional[str] = None,
    disease_signature_path: Optional[str] = None,
    gene_map_path: Optional[str] = None,
    creeds_organism: str = "human",
    creeds_reversal_method: str = "gene_overlap",
) -> Tuple[List[Dict], Dict]:
    """
    Attach reversal scores from CREEDS perturbation profiles vs a disease signature.

    In kg-only mode this is a no-op; the fields stay at 0.0 (zero-filled by
    the EvidenceFeatures dataclass) so the baseline PR-AUC is preserved.

    In kg+omics mode loads CREEDS JSON + disease signature when paths exist.
    Falls back to synthetic scores only if those artifacts are unavailable.
    """
    meta: Dict = {"omics_source": "none"}

    if mode == "kg-only":
        for c in candidates:
            c["signature_reversal_score"] = 0.0
            c["cell_type_reversal_score"] = 0.0
            c["pathway_reversal_score"] = 0.0
        logger.info("kg-only mode: omics features zero-filled (preserves baseline).")
        return candidates, meta

    from perturbation_layer.creeds_reversal import (
        enrich_candidates_with_creeds,
        load_creeds_reversal_context,
    )

    creeds_file = Path(creeds_path or "artifacts/external/creeds/single_drug_perturbations-v1.0.json")
    signature_file = Path(
        disease_signature_path or "artifacts/signatures/tcga_brca_60/disease_signature.json"
    )
    gene_map_file = Path(
        gene_map_path or "artifacts/external/gdc_tcga_brca/converted/tcga_brca_gene_map.csv"
    )

    try:
        context = load_creeds_reversal_context(
            creeds_path=creeds_file,
            disease_signature_path=signature_file,
            gene_map_path=gene_map_file,
            organism=creeds_organism,
            reversal_method=creeds_reversal_method,  # type: ignore[arg-type]
        )
        candidates, stats = enrich_candidates_with_creeds(
            candidates, context, filter_organism=creeds_organism
        )
        meta = {
            "omics_source": "creeds",
            "creeds_path": str(creeds_file),
            "disease_signature_path": str(signature_file),
            "gene_map_path": str(gene_map_file),
            "creeds_organism": creeds_organism,
            **stats,
        }
        logger.info(
            "CREEDS reversal scores attached: %d/%d candidates matched profiles.",
            stats["n_creeds_matched"],
            stats["n_candidates"],
        )
        return candidates, meta
    except Exception as exc:
        logger.warning(
            "CREEDS reversal unavailable (%s); using synthetic reversal scores.",
            exc,
        )
        _attach_synthetic_reversal_scores(candidates)
        meta = {"omics_source": "synthetic_fallback", "error": str(exc)}
        return candidates, meta


# ---------- Validation layer ----------

def enrich_with_validation(candidates: List[Dict], top_n: int) -> List[Dict]:
    """
    Run validation_layer on the top-N candidates (after fusion ranking).

    Adds:
        clinical_evidence_score: 1.0 if known indication, 0.5 if active trial, else 0.0
        known_indication: bool
        clinical_trial_phases: list of trial phases
    """
    try:
        from validation_layer.known_indications_validator import check_known_indication
        from validation_layer.clinical_trials_validator import query_clinical_trials
    except Exception as e:
        logger.warning(f"validation_layer import failed: {e}; skipping validation.")
        return candidates

    enriched = 0
    for c in candidates[:top_n]:
        cid = c.get("compound_hetionet_id", "")
        did = c.get("disease_hetionet_id", "")

        try:
            known = check_known_indication(cid, did)
        except Exception:
            known = False
        c["known_indication"] = known

        studies: List[Dict] = []
        try:
            studies = query_clinical_trials(c.get("compound", ""), c.get("disease", ""),
                                            max_results=3)
        except Exception as e:
            logger.debug(f"trials lookup failed for {c.get('compound')}: {e}")
        c["clinical_trial_phases"] = [s.get("phase", "Unknown") for s in studies]

        if known:
            c["clinical_evidence_score"] = 1.0
        elif studies:
            c["clinical_evidence_score"] = 0.5
        else:
            c["clinical_evidence_score"] = 0.0
        enriched += 1

    logger.info(f"Validation enriched {enriched} candidates.")
    return candidates


# ---------- Evidence fusion + report ----------

def fuse_and_rank(candidates: List[Dict], mode: str, *, config_path: str = "config/evidence_fusion_config.yaml") -> List:
    """Convert dicts → EvidenceFeatures, fuse, attach explanations, return sorted list."""
    from evidence_layer.evidence_schema import EvidenceFeatures
    from evidence_layer.feature_fusion import fuse_evidence
    from evidence_layer.explanation_builder import attach_explanations

    ev_list: List[EvidenceFeatures] = []
    match_statuses: List[str] = []
    for c in candidates:
        ef = EvidenceFeatures(
            compound=c.get("compound", ""),
            compound_hetionet_id=c.get("compound_hetionet_id"),
            disease=c.get("disease", ""),
            disease_hetionet_id=c.get("disease_hetionet_id"),
            kg_rotate_score=float(c.get("kg_rotate_score", 0.0)),
            kg_complex_score=float(c.get("kg_complex_score", 0.0)),
            graph_topology_score=float(c.get("graph_topology_score", 0.0)),
            qsvc_score=float(c.get("qsvc_score", 0.0)),
            classical_ensemble_score=float(c.get("classical_ensemble_score", 0.0)),
            signature_reversal_score=float(c.get("signature_reversal_score", 0.0)),
            cell_type_reversal_score=float(c.get("cell_type_reversal_score", 0.0)),
            pathway_reversal_score=float(c.get("pathway_reversal_score", 0.0)),
            clinical_evidence_score=float(c.get("clinical_evidence_score", 0.0)),
        )
        ev_list.append(ef)
        match_statuses.append(str(c.get("creeds_match_status", "unmatched")))

    fused = fuse_evidence(
        ev_list,
        mode=mode,
        omics_match_status=match_statuses if mode != "kg-only" else None,
        config_path=config_path,
    )
    attach_explanations(fused)
    return fused


def write_enriched_candidates(candidates: List[Dict], out_dir: Path) -> Path:
    """Persist full candidate dicts (CREEDS metadata + Hetionet IDs) before fusion."""
    path = out_dir / "candidates_enriched.json"
    path.write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    logger.info("Wrote %d enriched candidates to %s", len(candidates), path)
    return path


def write_outputs(
    fused: List,
    out_dir: Path,
    top_n: int,
    mode: str,
    disease_id: Optional[str],
    omics_meta: Optional[Dict] = None,
    enriched_candidates: Optional[List[Dict]] = None,
) -> Path:
    """Write CSV / JSON / Markdown report plus a run summary."""
    from evidence_layer.evidence_report import write_evidence_report
    csv_path = write_evidence_report(
        fused,
        out_dir=str(out_dir),
        top_n=top_n,
        disease_id=disease_id,
        extra_rows=enriched_candidates,
    )

    # Per-run summary
    summary = {
        "mode": mode,
        "disease_filter": disease_id,
        "n_candidates": len(fused),
        "top_n": top_n,
        "top_compound": fused[0].compound if fused else None,
        "top_score": fused[0].final_score if fused else None,
        "omics": omics_meta or {},
        "tier_distribution": {
            "tier_1": sum(1 for c in fused if c.confidence_tier == 1),
            "tier_2": sum(1 for c in fused if c.confidence_tier == 2),
            "tier_3": sum(1 for c in fused if c.confidence_tier == 3),
            "tier_4": sum(1 for c in fused if c.confidence_tier == 4),
        },
    }
    summary_path = out_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Run summary written to {summary_path}")
    return csv_path


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Full hybrid QML-KG drug repurposing pipeline.",
    )
    parser.add_argument("--mode", choices=["kg-only", "kg+omics"], default="kg+omics",
                        help="Whether to include omics reversal features.")
    parser.add_argument("--validate", action="store_true",
                        help="Run clinical/literature validation on top-N candidates.")
    parser.add_argument("--disease", default=None,
                        help="Restrict to a single Hetionet disease ID (Disease::DOID:XXXX).")
    parser.add_argument("--top-n", type=int, default=50,
                        help="Number of top candidates to report (default 50).")
    parser.add_argument("--output", default="artifacts/predictions",
                        help="Output directory (default artifacts/predictions/).")
    parser.add_argument("--kg-scores", default=None,
                        help="Optional path to upstream KG+QML scores JSON.")
    parser.add_argument(
        "--creeds-signatures",
        default="artifacts/external/creeds/single_drug_perturbations-v1.0.json",
        help="CREEDS drug perturbation JSON for reversal scoring.",
    )
    parser.add_argument(
        "--disease-signature",
        default="artifacts/signatures/tcga_brca_60/disease_signature.json",
        help="Disease RNA-seq signature JSON (up/down genes).",
    )
    parser.add_argument(
        "--gene-map",
        default="artifacts/external/gdc_tcga_brca/converted/tcga_brca_gene_map.csv",
        help="ENSG→symbol map for aligning signature genes with CREEDS.",
    )
    parser.add_argument("--creeds-organism", default="human",
                        help="Organism filter for CREEDS profiles (use 'any' for all).")
    parser.add_argument(
        "--creeds-reversal-method",
        choices=["gene_overlap", "cosine"],
        default="gene_overlap",
        help="CREEDS reversal scoring: gene_overlap (default) or cosine vs disease LFC.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Full repurposing pipeline ({args.mode}) ===")

    # 1. Get base candidates (KG+QML scored or demo)
    scores_path = Path(args.kg_scores) if args.kg_scores else None
    base = load_kg_qml_scores(scores_path)
    if not base:
        base = build_demo_candidates(disease_filter=args.disease)
    elif args.disease:
        base = [c for c in base if c.get("disease_hetionet_id") == args.disease]
        logger.info(f"Filtered kg-scores to disease {args.disease}: {len(base)} candidates")

    if not base:
        logger.error("No candidates available (empty after filter). Exiting.")
        return 1

    # 2. Omics enrichment (no-op in kg-only mode)
    base, omics_meta = enrich_with_omics(
        base,
        mode=args.mode,
        creeds_path=args.creeds_signatures,
        disease_signature_path=args.disease_signature,
        gene_map_path=args.gene_map,
        creeds_organism=args.creeds_organism,
        creeds_reversal_method=args.creeds_reversal_method,
    )

    # 3. Persist enriched candidates (CREEDS + KG metadata) before fusion
    write_enriched_candidates(base, out_dir)

    # 4. Fuse + rank
    fused = fuse_and_rank(base, mode=args.mode)

    # 5. Optional validation on top-N
    if args.validate:
        # Rebuild dict from fused to keep validation function generic, then
        # patch clinical_evidence_score back onto the EvidenceFeatures.
        dicts = [ef.to_dict() for ef in fused]
        dicts = enrich_with_validation(dicts, top_n=args.top_n)
        clin_map = {d["compound"]: d.get("clinical_evidence_score", 0.0) for d in dicts}
        match_map = {c.get("compound"): c.get("creeds_match_status", "unmatched") for c in base}
        for ef in fused:
            if ef.compound in clin_map:
                ef.clinical_evidence_score = clin_map[ef.compound]
        from evidence_layer.feature_fusion import fuse_evidence
        from evidence_layer.explanation_builder import attach_explanations
        match_statuses = [str(match_map.get(ef.compound, "unmatched")) for ef in fused]
        fused = fuse_evidence(
            fused,
            mode=args.mode,
            omics_match_status=match_statuses if args.mode != "kg-only" else None,
        )
        attach_explanations(fused)

    # 6. Write outputs
    csv_path = write_outputs(
        fused,
        out_dir,
        top_n=args.top_n,
        mode=args.mode,
        disease_id=args.disease,
        omics_meta=omics_meta,
        enriched_candidates=base,
    )
    logger.info(f"=== Pipeline complete. Top candidate: {fused[0].compound} "
                f"(score={fused[0].final_score:.4f}, tier={fused[0].confidence_tier}) ===")
    logger.info(f"Report: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
