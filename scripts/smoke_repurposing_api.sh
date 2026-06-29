#!/usr/bin/env bash
# API smoke for Tier 3 repurposing workbench endpoints.
# Usage: API_PORT=8780 ./scripts/smoke_repurposing_api.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
API_PORT="${API_PORT:-8780}"
BASE="http://127.0.0.1:${API_PORT}"

if ! command -v curl >/dev/null 2>&1; then
  echo "[error] curl required" >&2
  exit 1
fi

echo "=== Repurposing API smoke @ ${BASE} ==="

curl -sf "${BASE}/status" >/dev/null || {
  echo "[error] API not reachable at ${BASE}/status (run ./scripts/dev_stack.sh or uvicorn)" >&2
  exit 1
}

"$ROOT/.venv/bin/python" -m pytest tests/test_repurposing_workbench.py -q

check_disease() {
  local disease_id="$1"
  local expected_top="$2"
  local body
  body="$(curl -sf "${BASE}/repurposing/candidates?disease_id=${disease_id}")"
  echo "$body" | "$ROOT/.venv/bin/python" -c "
import json, sys
d = json.load(sys.stdin)
assert d['manifest']['source'] == 'repurposing_evidence_bundle', d['manifest']
top = d['candidates'][0]['compound_name']
assert top == '${expected_top}', f'expected ${expected_top}, got {top}'
rnaseq = d['candidates'][0].get('rnaseq_signature') or {}
if '${disease_id}' == 'brca_external_validation':
    assert rnaseq.get('omics_cell_type_status') == 'not_computed'
    assert rnaseq.get('omics_pathway_status') == 'not_computed'
print('ok', '${disease_id}', top)
"
}

check_disease "brca_external_validation" "Vemurafenib"
check_disease "brca_external_validation_organism_any" "Prednisolone"
check_disease "all_pairs_kg_omics" "Fluticasone furoate"

echo "=== Repurposing API smoke passed ==="
