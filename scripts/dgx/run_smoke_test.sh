#!/usr/bin/env bash
# run_smoke_test.sh — End-to-end smoke test using synthetic / demo data.
# Exits non-zero on any failure. Intended as the M7 CI gate from a clean venv.
set -euo pipefail

echo "=== Smoke Test: hybrid-qml-kg-poc ==="
echo ""

echo "[1/6] Import check — all new layers"
python3 -c "
from entity_resolution.hetionet_resolver import HetionetResolver
from entity_resolution.gene_mapper import GeneMapper
from entity_resolution.disease_mapper import DiseaseMapper
from entity_resolution.compound_mapper import CompoundMapper
from single_cell_layer.loaders import load_single_cell_config
from single_cell_layer.qc import run_qc
from single_cell_layer.disease_signature import build_disease_signature
from single_cell_layer.cell_type_signature import build_per_cell_type_signatures
from perturbation_layer.reversal_score import compute_reversal_score
from perturbation_layer.cmap_loader import cmap_to_up_down
from evidence_layer.evidence_schema import EvidenceFeatures
from evidence_layer.feature_fusion import fuse_evidence
from evidence_layer.confidence_tiering import assign_tier
from validation_layer.known_indications_validator import check_known_indication
from validation_layer.drugbank_mapper import check_drugbank_indication
print('All imports OK')
"

echo ""
echo "[2/6] Entity resolution — HetionetResolver (nodes.tsv optional)"
python3 -c "
from entity_resolution.hetionet_resolver import HetionetResolver
r = HetionetResolver()
r.load()
print(f'Resolver loaded: {len(r._id_to_node)} nodes')
"

echo ""
echo "[3/6] Reversal score — unit check"
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
echo "[4/6] Evidence fusion — schema + scoring"
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
echo "[5/6] Validation — known indication + DrugBank seed"
python3 -c "
from validation_layer.known_indications_validator import check_known_indication
from validation_layer.drugbank_mapper import check_drugbank_indication, get_drugbank_indications
# metformin treats type 2 diabetes (seeded)
known = check_known_indication('Compound::DB00331', 'Disease::DOID:9352')
db_known = check_drugbank_indication('Compound::DB00331', 'type 2 diabetes mellitus')
print(f'metformin→T2D: known_indication={known}, drugbank={db_known}')
assert db_known, 'DrugBank seed lookup should hit'
assert not check_drugbank_indication('Compound::DB00331', 'breast cancer'), 'Should not falsely hit'
print('Validation checks OK')
"

echo ""
echo "[6/6] Full pipeline — end-to-end orchestration"
python3 scripts/run_full_repurposing_pipeline.py --mode kg+omics --top-n 5 2>&1 | tail -3
python3 -c "
import json
from pathlib import Path
summary = json.loads(Path('artifacts/predictions/run_summary.json').read_text())
assert summary['n_candidates'] > 0
assert summary['top_compound'] is not None
assert summary['mode'] == 'kg+omics'
print(f'Pipeline produced {summary[\"n_candidates\"]} candidates, top={summary[\"top_compound\"]}')
"

echo ""
echo "=== All smoke tests passed (6/6) ==="
