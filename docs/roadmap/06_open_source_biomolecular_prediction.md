# Open-Source Biomolecular Prediction Plan

**Status:** Planned  
**Horizon:** 1-2 weeks for the first usable slice; 3-6 weeks for local model runners  
**Constraint:** Open-source, local-first tooling only. No required paid service,
hosted prediction API, external account, proprietary model endpoint, or QPU
runtime.

This plan turns the biomedical structure + hybrid quantum idea into a build path
that fits the existing Hybrid QML-KG system. The key design choice is to treat
structure biology as an upstream feature source and keep the predictive lift in
the existing KG/classical/quantum benchmark harness until evidence says
otherwise.

---

## 1. Decision

Default architecture:

```text
Local sequence / ligand / target artifacts
        |
        v
Open-source structure artifact layer
        |
        v
Structure and molecule feature extraction
        |
        v
Existing KG, embedding, graph, pathway, and MoA features
        |
        v
Existing classical + QSVC / VQC benchmark harness
        |
        v
Prediction, ranking, provenance, and ablation reporting
```

Quantum remains a constrained evaluation layer, not the whole predictive
backbone. The first scientific question is:

> Do local structure-derived features improve CtD-style ranking beyond the
> current KG + graph + embedding baseline, and do they improve or decorrelate
> the quantum ensemble signal?

---

## 2. Tooling Policy

### Allowed by default

Use these only through local installs or local artifacts:

| Area | Default candidates | Notes |
|------|--------------------|-------|
| Protein structure | OpenFold, ESMFold, Chai-1, Boltz | Prefer artifact ingestion first; add runners only after fixtures work. |
| Protein embeddings | ESM-2 / local protein language model weights | Use local weights/cache; do not call hosted APIs in normal runs. |
| Molecule features | RDKit | First target for SMILES/SDF descriptors and fingerprints. |
| Docking proxies | AutoDock Vina or other permissive local tools | Add only after structure artifact schema is stable. |
| KG / graph ML | Existing PyKEEN, NetworkX, scikit-learn path | Keep this as the main predictive backbone. |
| Quantum simulation | Existing Qiskit Aer CPU path | Already present in requirements and CI-friendly. |
| NVIDIA quantum prototyping | CUDA-Q, optional local GPU builds | Optional adapter, not a required dependency. |

### Restricted or optional

Do not make these required:

- AlphaFold Server, hosted BioNeMo, NIM endpoints, or other hosted structure APIs.
- IBM Quantum Runtime or any real QPU account.
- Any dependency that requires an account, click-through cloud workspace, paid
  quota, or non-open redistribution terms.
- cuQuantum as a hard dependency. Its public repository is primarily BSD-3-Clause
  but includes proprietary licensing exceptions, so any use must be optional and
  limited to reviewed components.

### License gate

Every new structure or quantum dependency must pass this gate before it is added
to a requirements file:

1. Code license is OSI-compatible or otherwise explicitly approved for this
   project.
2. Model weights license allows the intended research and redistribution story.
3. Normal tests and smoke runs do not require an account, API token, hosted
   endpoint, or paid quota.
4. Outputs are not used to train a competing structure-prediction model unless
   the upstream terms explicitly allow it.
5. The dependency is optional unless the repo can install it reproducibly in the
   existing local/DGX setup.

---

## 3. First Build Slice

Start with artifact ingestion rather than model execution. This avoids a large
dependency jump and creates a stable contract for OpenFold, ESMFold, Chai-1,
Boltz, and future local runners.

### 3.1 Add a structure artifact layer

Proposed files:

```text
structure_layer/
  __init__.py
  artifacts.py
  feature_extraction.py
  target_mapping.py
  provenance.py

config/
  structure_config.yaml

scripts/
  build_structure_features.py

tests/
  test_structure_layer.py
```

Core artifact schema:

```python
StructureArtifact(
    target_id: str,
    target_name: str,
    sequence_hash: str,
    source_tool: str,
    source_version: str,
    artifact_path: str,
    artifact_format: str,  # pdb, mmcif, json, npz
    license_note: str,
    confidence: dict,
    metadata: dict,
)
```

Acceptance criteria:

- A tiny local fixture can be loaded without network access.
- The artifact registry records source tool, version, file path, and license note.
- Missing optional fields degrade to explicit `null` or empty feature values, not
  silent zero-confidence claims.

### 3.2 Extract fixed-width structure features

Initial features should be cheap and deterministic:

- Sequence length and chain count.
- Residue coverage where available.
- Mean, median, and low-confidence fraction for pLDDT-like scores when present.
- Contact-density proxies from C-alpha distances.
- Radius of gyration and compactness proxies.
- Binding-site or pocket summary fields when supplied by an upstream artifact.
- Per-target missingness flags so models can learn when structure evidence is
  absent.

Output:

```text
results/structure_features/
  target_structure_features.csv
  target_structure_features.schema.json
  target_structure_features.provenance.json
```

Acceptance criteria:

- Fixture output is deterministic.
- Feature vector names and dimensions are documented in the schema JSON.
- A no-structure target produces a valid missingness vector.

### 3.3 Map structure features into CtD candidate pairs

Do not attach a random protein structure directly to a compound-disease pair.
Use an explicit aggregation rule:

1. Resolve candidate proteins through known compound-target, disease-gene, or
   short-path KG evidence.
2. Join available target-level structure features.
3. Aggregate target features per compound-disease pair using mean, max, count,
   and missingness rates.
4. Keep provenance for the target IDs that contributed to each pair.

Proposed integration points:

- `kg_layer/enhanced_features.py` for pair-level feature assembly.
- `evidence_layer/feature_fusion.py` for provenance-aware feature fusion.
- `scripts/run_optimized_pipeline.py` behind a new `--use_structure_features`
  flag.

Acceptance criteria:

- The default pipeline is unchanged unless `--use_structure_features` is set.
- The structure feature dimensionality is reported in the run manifest.
- Feature provenance can explain which targets contributed to a scored pair.

---

## 4. Benchmark Plan

Use the existing benchmark discipline. The first useful comparison is not a
large model demo; it is a controlled ablation:

| Condition | Description |
|-----------|-------------|
| A | Current best KG + graph + embedding feature set |
| B | A + structure features |
| C | A + quantum ensemble |
| D | A + structure features + quantum ensemble |

Required metrics:

- PR-AUC and ROC-AUC on the existing split.
- Bootstrap confidence interval if the run is promoted to paper evidence.
- Pair-level score correlation between classical and quantum predictions.
- Structure feature missingness rate.
- Runtime and dependency footprint.

Acceptance criteria:

- Structure features must beat or complement the existing baseline before any
  paper/UI claim is made.
- If the gain is only from missingness or leakage, the feature path is not
  accepted.
- Quantum claims remain ensemble/decorrelation claims unless a standalone
  quantum result improves materially.

---

## 5. Local Runner Roadmap

Only add model runners after artifact ingestion and feature extraction are
working.

### Runner 1: local OpenFold or ESMFold artifact generation

Scope:

- Input FASTA.
- Output local PDB/mmCIF plus provenance JSON.
- No hosted API calls.
- Heavy dependencies isolated in an optional requirements file, e.g.
  `requirements-structure.txt`.

### Runner 2: local Chai-1 or Boltz artifacts

Scope:

- Use for biomolecular complexes or binding-oriented features.
- Pin exact versions and record model weight source.
- Keep as optional because GPU memory requirements may be high.

### Runner 3: local molecule descriptors

Scope:

- Add RDKit descriptors and fingerprints for compound nodes.
- Join into the same pair-feature path as structure features.
- Keep docking separate until target mapping is validated.

---

## 6. Frontend and API Implications

First API surface should be read-only:

- `GET /runs/latest` includes whether structure features were used.
- Run manifests include `structure_feature_pipeline_id`.
- Candidate evidence includes a compact structure provenance summary only when
  available.

The Next.js v2 UI should show structure evidence as provenance, not as a
standalone promise that structure prediction caused the ranking. The right copy
is "structure evidence available" or "structure-derived features contributed",
not "AlphaFold-validated" unless that is literally true for the artifact.

---

## 7. Immediate Task List

1. Create `structure_layer` with artifact/provenance dataclasses and parser
   interfaces.
2. Add one tiny PDB/mmCIF fixture and one no-structure fixture.
3. Implement deterministic target-level feature extraction.
4. Add `scripts/build_structure_features.py`.
5. Add tests for fixture loading, missingness behavior, and schema stability.
6. Wire optional `--use_structure_features` into pair-feature construction.
7. Run A/B ablation on a fast CtD subset with
   `scripts/structure_feature_ablation.py`.
8. Promote to full CtD only if the fast subset has no leakage or obvious
   missingness artifact.

---

## 8. References Checked

Checked on 2026-06-16:

- OpenFold repository: Apache-2.0 code; notes pretrained parameters are CC BY
  4.0. <https://github.com/aqlaboratory/openfold>
- ESM repository: MIT license; archived on 2024-08-01, so pin carefully if used.
  <https://github.com/facebookresearch/esm>
- Chai-1 repository: Apache-2.0 code and weights according to its README.
  <https://github.com/chaidiscovery/chai-lab>
- Boltz repository: MIT code and weights according to its README.
  <https://github.com/jwohlwend/boltz>
- RDKit repository: BSD license. <https://github.com/rdkit/rdkit>
- Qiskit Aer repository: Apache-2.0 license.
  <https://github.com/Qiskit/qiskit-aer>
- CUDA-Q repository: Apache-2.0 license.
  <https://github.com/NVIDIA/cuda-quantum>
- cuQuantum repository: primarily BSD-3-Clause, with proprietary licensing
  exceptions called out in the README. <https://github.com/NVIDIA/cuQuantum>
