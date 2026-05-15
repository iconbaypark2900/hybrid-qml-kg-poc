#!/usr/bin/env bash
# run_smoke_test.sh — End-to-end smoke test using synthetic data.
# Exits non-zero on any failure.
set -euo pipefail

echo "=== Smoke Test: hybrid-qml-kg-poc ==="
echo ""

echo "[1/5] Import check — all new layers"
python3 -c "
from entity_resolution.hetionet_resolver import HetionetResolver
from entity_resolution.gene_mapper import GeneMapper
from entity_resolution.disease_mapper import DiseaseMapper
from entity_resolution.compound_mapper import CompoundMapper
from single_cell_layer.loaders import load_single_cell_config
from single_cell_layer.qc import run_qc
from single_cell_layer.disease_signature import build_disease_signature
from perturbation_layer.reversal_score import compute_reversal_score
from evidence_layer.evidence_schema import EvidenceFeatures
from evidence_layer.feature_fusion import fuse_evidence
from evidence_layer.confidence_tiering import assign_tier
from validation_layer.known_indications_validator import check_known_indication
print('All imports OK')
"

echo ""
echo "[2/5] Entity resolution — HetionetResolver (nodes.tsv optional)"
python3 -c "
from entity_resolution.hetionet_resolver import HetionetResolver
r = HetionetResolver()
r.load()
print(f'Resolver loaded: {len(r._id_to_node)} nodes')
"

echo ""
echo "[3/5] Reversal score — unit check"
python3 -c "
from perturbation_layer.reversal_score import compute_reversal_score
# Perfect reversal: disease up = drug down and vice versa
score = compute_reversal_score(['A','B','C'], ['D','E'], ['D','E'], ['A','B','C'])
assert score > 0.9, f'Expected > 0.9, got {score}'
# No overlap
score2 = compute_reversal_score(['A'], ['B'], ['C'], ['D'])
assert score2 == 0.0, f'Expected 0.0, got {score2}'
print(f'Reversal scores: perfect={score:.3f}, no-overlap={score2:.3f}  OK')
"

echo ""
echo "[4/5] Evidence fusion — schema + scoring"
python3 -c "
from evidence_layer.evidence_schema import EvidenceFeatures
from evidence_layer.feature_fusion import fuse_evidence
from evidence_layer.explanation_builder import attach_explanations

candidates = [
    EvidenceFeatures(
        compound='aspirin', compound_hetionet_id='Compound::DB00945',
        disease='diabetes', disease_hetionet_id='Disease::DOID:9351',
        kg_rotate_score=0.85, qsvc_score=0.72, classical_ensemble_score=0.79,
        signature_reversal_score=0.68,
    ),
    EvidenceFeatures(
        compound='metformin', compound_hetionet_id='Compound::DB00331',
        disease='diabetes', disease_hetionet_id='Disease::DOID:9351',
        kg_rotate_score=0.91, qsvc_score=0.80, classical_ensemble_score=0.88,
        signature_reversal_score=0.75,
    ),
]
fused = fuse_evidence(candidates, mode='kg+omics')
attach_explanations(fused)
assert fused[0].final_score >= fused[1].final_score, 'Should be sorted descending'
assert fused[0].explanation != '', 'Explanation should be populated'
print(f'Top candidate: {fused[0].compound} (score={fused[0].final_score:.4f}, tier={fused[0].confidence_tier})  OK')
"

echo ""
echo "[5/5] Validation — known indication check"
python3 -c "
from validation_layer.known_indications_validator import check_known_indication
# metformin treats type 2 diabetes (seeded)
result = check_known_indication('Compound::DB00331', 'Disease::DOID:9352')
print(f'metformin→type2_diabetes known={result}')
"

echo ""
echo "=== All smoke tests passed ==="
