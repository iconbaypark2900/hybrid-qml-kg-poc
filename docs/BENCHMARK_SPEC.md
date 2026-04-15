# Benchmark specification — version 1.0

Locked: 2026-04-12 | Relation: CtD

Do not change the settings in sections 1–7 without creating a new versioned spec.
Results generated under different settings are not directly comparable.

---

## 1. Dataset

| Parameter | Value |
|---|---|
| Source | Hetionet v1.0 |
| Relation | Compound-treats-Disease (CtD) |
| Positive edges | All 755 known CtD edges |
| Entity scope | All 464 compounds and diseases appearing in CtD edges |
| Embedding training graph | Full Hetionet (2.25 M edges, all 24 relation types) |
| Required entity coverage | ≥ 95 % of 464 CtD entities must have trained embeddings |

## 2. Split policy

| Parameter | Value |
|---|---|
| Method | Random edge split (not entity split) |
| Test fraction | 0.20 |
| Random state | 42 |
| Stratification | None (class balance maintained by 1:1 negative ratio) |
| Leakage control | Test edges excluded from embedding training graph |

## 3. Negative sampling

| Parameter | Value |
|---|---|
| Default strategy | `degree_corrupt` |
| Ratio | 1:1 (negatives : positives) |
| Avoidance | All 755 known CtD positive edges excluded from negative pool |
| Hard eval set | Separate test set generated with `type_aware` strategy for realism check |

**Rule:** Random-only negatives are not an acceptable sole evaluation condition.
Benchmarks must include at least one hard negative evaluation set.

The three supported strategies are implemented in `kg_layer/kg_loader.py`:

- `degree_corrupt` — degree-weighted KG corruption. Default for training.
- `type_aware` — same entity-type prefix corruption. Used for the hard eval set.
- `embedding_knn` — K-nearest-neighbour in embedding space. Hardest; requires trained embedder.

## 4. Feature regime

### Classical models (full feature set)
- Embedding method: RotatE
- Embedding dimension: 128
- Training epochs: 200
- Feature construction: `[h, t, |h−t|, h*t]` → 512-dim per pair

### Quantum models (reduced feature set)
- Pre-PCA: 512 → 24 dimensions (`--qml_pre_pca_dim 24`)
- Final QML dimension: 16 qubits (`--qml_dim 16`)
- Feature map: Pauli, reps=2 (`--qml_feature_map Pauli --qml_feature_map_reps 2`)
- Entanglement: full

### Fair comparison rule (blueprint §10)

QSVC (16-dim input) must always be compared against classical models on **both**:

1. Full 512-dim features — classical advantage condition
2. Same 16-dim PCA-reduced features — matched-regime condition

Comparing QSVC on 16-dim features against classical models on 512-dim features
**only** is not a valid benchmark comparison.

## 5. Model family

| Model | Type | Key config |
|---|---|---|
| LogisticRegression-L2 | classical | C=1.0, class_weight=balanced |
| ExtraTrees-Optimized | classical | n_estimators=600, max_features=sqrt, balanced |
| RandomForest-Optimized | classical | n_estimators=200, max_depth=10, balanced |
| HistGBDT | classical | max_depth=8, learning_rate=0.06, max_iter=400 |
| QSVC-Pauli | quantum | C=0.1, Pauli feature map, reps=2, 16 qubits |
| Stacking-Ensemble | ensemble | LR meta-learner trained on OOF predictions |

## 6. Primary evaluation metrics

| Metric | Role | Rationale |
|---|---|---|
| PR-AUC | **Primary** | Handles class imbalance correctly; preferred for biomedical ranking |
| ROC-AUC | Secondary | Standard AUC for context |
| F1 (threshold=0.5) | Tertiary | Classification quality |
| Top-10 hit rate | Discovery | Fraction of true CtD test edges appearing in top-10 candidates per disease |
| Mean rank of positives | Discovery | Average rank of true CtD edges in full ranked candidate list |

Top-10 hit rate and mean rank must be reported for any result claimed to demonstrate
biomedical discovery utility.

## 7. Quantum execution tiers — must not be mixed in any single comparison

| Tier | Backend label | Description |
|---|---|---|
| ideal | `simulator_statevector` | Noiseless statevector simulation |
| noisy | `simulator_noisy` | Aer with IBM-calibrated noise model |
| hardware | `ibm_torino` or `ibm_heron` | Real IBM Quantum device |

A claim of quantum advantage is only valid when classical and quantum models
are evaluated within the **same tier**. A quantum result on hardware must not
be compared against a classical result from ideal simulation, and vice versa.

## 8. Provenance requirements

Every benchmark run must call `scripts.benchmark_registry.register_run()`.
Each record must include:

- Embedding config: method, dim, epochs, full_graph flag
- Reduction config: method, input_dim, output_dim, explained_variance
- Model config: name, type, hyperparams, n_features_in
- Backend: name, execution_mode, shots, noise_model
- Circuit metadata (quantum runs): n_qubits, circuit_depth, feature_map, feature_map_reps, ansatz, ansatz_reps, optimizer, max_iter
- Negative sampling: strategy, ratio
- Split config: test_size, random_state, n_train_pos, n_test_pos, n_train_neg, n_test_neg
- All primary metrics from section 6

Registry file: `results/benchmark_registry.jsonl` (append-only, one JSON per line).

## 9. Current performance vs targets

| Model | PR-AUC target | Status (as of 2026-04-12) |
|---|---|---|
| Best classical | ≥ 0.75 | ✅ ExtraTrees, PR-AUC 0.81 |
| Best quantum (ideal sim) | ≥ 0.65 | in progress |
| Best ensemble | ≥ 0.78 | ✅ Stacking, PR-AUC ~0.80 |
| Hardware quantum | baseline TBD | blocked — `scripts/train_on_heron.py` empty |

## 10. What invalidates a result

- Comparing QML on reduced features against classical on full features without
  also running the matched-regime comparison
- Mixing execution tiers (e.g. ideal sim classical vs noisy sim quantum)
- Using random-only negatives as the sole evaluation condition
- Reporting PR-AUC without also reporting the negative sampling strategy used
- Missing provenance — any run not recorded in `benchmark_registry.jsonl`
