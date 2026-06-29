#!/usr/bin/env bash
# Rebuild breast repurposing workbench artifacts (human + organism=any CREEDS).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PROJECT_ROOT}/.venv/bin/python"
KG_SCORES="results/rnaseq_repurposing_run/kg_qml_scores_256d_moa.json"
DISEASE="Disease::DOID:1612"

EVIDENCE_VERIFICATION="artifacts/benchmarks/rnaseq_quantum_tcga_brca_60_harmonized/evidence_bundle_verification.json"
EXTERNAL_VALIDATION="artifacts/benchmarks/rnaseq_quantum_tcga_brca_gse225846_external/external_validation.json"
BENCHMARK_VERDICT="artifacts/benchmarks/rnaseq_quantum_tcga_brca_60_harmonized/quantum_value_verdict.json"
EVIDENCE_AUDIT="artifacts/benchmarks/rnaseq_quantum_tcga_brca_60_harmonized/evidence_audit.json"
STRUCTURE_REGISTRY="artifacts/structures/alphafold/brca_anastrozole_targets/structure_registry.json"

run_pipeline_and_bundle() {
  local organism="$1"
  local run_dir="$2"
  local bundle_dir="$3"
  local disease_api_id="$4"
  local disease_name="$5"

  echo "=== Pipeline: CREEDS organism=${organism} -> ${run_dir} ==="
  "$PYTHON" scripts/run_full_repurposing_pipeline.py \
    --mode kg+omics \
    --kg-scores "$KG_SCORES" \
    --disease "$DISEASE" \
    --creeds-reversal-method cosine \
    --creeds-organism "$organism" \
    --output "$run_dir" \
    --top-n 50

  echo "=== Export ranking_comparison.csv ==="
  "$PYTHON" scripts/export_repurposing_ranking_comparison.py --input "$run_dir"

  local ranking_csv="${run_dir}/ranking_comparison.csv"
  local target_map="${bundle_dir}/candidate_target_map.csv"

  mkdir -p "$bundle_dir"
  echo "=== Build candidate target map ==="
  "$PYTHON" scripts/build_repurposing_candidate_target_map.py \
    --ranking-comparison "$ranking_csv" \
    --out "$target_map"

  echo "=== Build evidence bundle -> ${bundle_dir} ==="
  "$PYTHON" scripts/build_repurposing_evidence_bundle.py \
    --evidence-verification "$EVIDENCE_VERIFICATION" \
    --external-validation "$EXTERNAL_VALIDATION" \
    --benchmark-verdict "$BENCHMARK_VERDICT" \
    --evidence-audit "$EVIDENCE_AUDIT" \
    --ranking-comparison "$ranking_csv" \
    --structure-registry "$STRUCTURE_REGISTRY" \
    --candidate-target-map "$target_map" \
    --out-dir "$bundle_dir" \
    --disease-id "$disease_api_id" \
    --disease-name "$disease_name" \
    --top-k 25
}

echo "=== Repurposing workbench refresh started at $(date) ==="

echo "=== Ensure RNA-seq proof artifacts ==="
"$PYTHON" scripts/ensure_repurposing_rnaseq_proof.py

run_pipeline_and_bundle \
  human \
  "results/rnaseq_repurposing_run/repurposing_breast_bundle_human" \
  "artifacts/repurposing/brca_external_validation" \
  "brca_external_validation" \
  "Breast cancer"

run_pipeline_and_bundle \
  any \
  "results/rnaseq_repurposing_run/repurposing_breast_bundle_any" \
  "artifacts/repurposing/brca_external_validation_organism_any" \
  "brca_external_validation_organism_any" \
  "Breast cancer (CREEDS any organism)"

run_pipeline_and_bundle \
  any \
  "results/rnaseq_repurposing_run/repurposing_breast_bundle_any" \
  "artifacts/repurposing/brca_external_validation_organism_any" \
  "brca_external_validation_organism_any" \
  "Breast cancer (CREEDS any organism)"

echo "=== Pipeline: 200-pair all diseases (human CREEDS) ==="
"$PYTHON" scripts/run_full_repurposing_pipeline.py \
  --mode kg+omics \
  --kg-scores "$KG_SCORES" \
  --creeds-reversal-method cosine \
  --creeds-organism human \
  --output "results/rnaseq_repurposing_run/repurposing_full_200_cosine" \
  --top-n 200

"$PYTHON" scripts/export_repurposing_ranking_comparison.py \
  --input "results/rnaseq_repurposing_run/repurposing_full_200_cosine"

ALL_RANKING="results/rnaseq_repurposing_run/repurposing_full_200_cosine/ranking_comparison.csv"
ALL_BUNDLE="artifacts/repurposing/all_pairs_kg_omics"
mkdir -p "$ALL_BUNDLE"
"$PYTHON" scripts/build_repurposing_candidate_target_map.py \
  --ranking-comparison "$ALL_RANKING" \
  --out "$ALL_BUNDLE/candidate_target_map.csv"

"$PYTHON" scripts/build_repurposing_evidence_bundle.py \
  --evidence-verification "$EVIDENCE_VERIFICATION" \
  --external-validation "$EXTERNAL_VALIDATION" \
  --benchmark-verdict "$BENCHMARK_VERDICT" \
  --evidence-audit "$EVIDENCE_AUDIT" \
  --ranking-comparison "$ALL_RANKING" \
  --structure-registry "$STRUCTURE_REGISTRY" \
  --candidate-target-map "$ALL_BUNDLE/candidate_target_map.csv" \
  --out-dir "$ALL_BUNDLE" \
  --disease-id "all_pairs_kg_omics" \
  --disease-name "All diseases (200-pair KG+omics)" \
  --top-k 50

echo "=== Fusion weight ablation (breast human) ==="
"$PYTHON" scripts/ablate_evidence_fusion_weights.py

echo "=== Fig4 correlation ==="
"$PYTHON" figures/fig4_kg_rnaseq_correlation.py

echo "=== Verification ==="
"$PYTHON" - <<'PY'
import json
from pathlib import Path
from middleware.repurposing_workbench import build_repurposing_candidates

for disease_id in ("brca_external_validation", "brca_external_validation_organism_any", "all_pairs_kg_omics"):
    resp = build_repurposing_candidates(disease_id)
    src = resp.get("manifest", {}).get("source", "")
    n = len(resp.get("candidates", []))
    top = resp["candidates"][0]["compound_name"] if resp.get("candidates") else "none"
    print(f"{disease_id}: source={src} candidates={n} top={top}")
    assert src == "repurposing_evidence_bundle", f"expected bundle, got {src}"
PY

echo "=== Done at $(date) ==="
