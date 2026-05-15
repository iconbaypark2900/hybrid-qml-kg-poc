# Assumptions and Deviations

Documents methodological choices that deviate from or extend the nominal paper description, plus hardware and dataset assumptions that affect reproducibility.

---

## Dataset Assumptions

| Assumption | Detail | Impact |
|---|---|---|
| Hetionet v1.0 only | We use the frozen v1.0 snapshot; Hetionet is not actively updated | Reproducible; no version drift |
| CtD edges only for training | CrC, CbG, DaG edges used as structural features but not as prediction targets | Consistent with paper §2.1 |
| Negative sampling ratio 1:10 | One positive CtD edge per 10 randomly sampled non-edges | Matches preregistration |
| Hard negatives sourced from PubChem neighbors | Compounds structurally similar to known treatments, not random | Documented in `scripts/hard_negatives_experiment.py` |
| Single-cell demo dataset | CELLxGENE public h5ad used for demonstration; paper results use [TBD] dataset | Demo ≠ paper dataset; add dataset DOI when confirmed |
| LINCS L1000 Phase II | Perturbation signatures sourced from LINCS L1000 Phase II (cp_genes.gctx) | License: public; registration required |

---

## Methodological Deviations

| Item | Nominal (paper) | Actual implementation | Reason |
|---|---|---|---|
| KG embedding training | PyKEEN RotatE | PyKEEN with fallback to seeded random if unavailable | Reproducibility on hardware without PyKEEN GPU |
| Quantum backend | IBM Quantum real hardware | Statevector simulator (ideal) as primary; noisy + hardware as optional | Hardware queues; ideal gives deterministic results for paper |
| Feature map depth | PauliFeatureMap reps=2 | Configurable via `config/quantum_config.yaml` | Allows ablation without code change |
| Stacking meta-learner | Logistic Regression | LR with C tuned per nested CV fold | Same family, tuned |
| Batch correction method | Harmony | harmonypy (Python port); results match R Harmony within tolerance | Python ecosystem |
| Reversal scoring method | CMap connectivity score | Rank correlation (Spearman); CMap score as supplementary | More transparent; CMap score requires GSEA licensing |

---

## Hardware Assumptions

| Assumption | Detail |
|---|---|
| Minimum CPU | 8-core; 16 GB RAM for full Hetionet graph |
| GPU (optional) | NVIDIA GPU with CUDA 12+ for RAPIDS backend and quantum GPU simulation |
| DGX Spark (target) | A100 80 GB; RAPIDS 23.x; CUDA 12.x |
| Quantum simulation | Qiskit Statevector: O(2^n) memory; n≤20 qubits practical on 16 GB RAM |
| Storage | ~5 GB for Hetionet + embeddings; ~50 GB for full LINCS L1000 |

---

## Known Limitations

- Entity resolution depends on Hetionet's internal ID scheme; compounds / diseases not in Hetionet v1.0 cannot receive KG scores (omics-only score available as fallback).
- Single-cell analysis is limited to datasets with clear disease/control metadata columns; studies without explicit condition labels require manual annotation.
- Literature validation (PubMed co-occurrence) is a proxy for support; co-occurrence ≠ causal evidence.
