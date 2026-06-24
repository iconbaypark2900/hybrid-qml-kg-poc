from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = Path("scripts/fetch_alphafold_structures.py")
SPEC = importlib.util.spec_from_file_location("fetch_alphafold_structures", SCRIPT_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules["fetch_alphafold_structures"] = module
SPEC.loader.exec_module(module)


PDB_FIXTURE = (
    "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 91.00           C\n"
    "ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00 89.00           C\n"
    "END\n"
).encode("utf-8")


def test_fetch_alphafold_structures_writes_local_registry_without_network(tmp_path: Path) -> None:
    target_map = tmp_path / "targets.csv"
    target_map.write_text(
        "target_id,target_name,gene_symbol,uniprot_id\n"
        "Gene::1588,CYP19A1,CYP19A1,P11511\n",
        encoding="utf-8",
    )

    def fake_json(url: str):
        assert url.endswith("/P11511")
        return [
            {
                "modelEntityId": "AF-P11511-F1",
                "pdbUrl": "https://example.test/AF-P11511-F1-model_v6.pdb",
                "cifUrl": "https://example.test/AF-P11511-F1-model_v6.cif",
                "paeDocUrl": "https://example.test/pae.json",
                "latestVersion": 6,
                "modelCreatedDate": "2025-08-01T00:00:00Z",
                "globalMetricValue": 91.38,
                "fractionPlddtVeryHigh": 0.8,
                "fractionPlddtConfident": 0.1,
                "fractionPlddtLow": 0.05,
                "fractionPlddtVeryLow": 0.05,
                "sequenceChecksum": "abc123",
                "uniprotSequence": "AG",
                "organismScientificName": "Homo sapiens",
            }
        ]

    manifest = module.fetch_alphafold_structures(
        target_map=target_map,
        out_dir=tmp_path / "structures",
        fetch_json=fake_json,
        fetch_bytes=lambda url: PDB_FIXTURE,
    )

    assert manifest["status"] == "ready"
    assert manifest["artifact_count"] == 1
    registry = Path(manifest["registry"])
    assert registry.exists()
    assert (registry.parent / "Gene_1588_AF-P11511-F1.pdb").exists()
    text = registry.read_text(encoding="utf-8")
    assert "alphafold_db_cached" in text
    assert "P11511" in text
