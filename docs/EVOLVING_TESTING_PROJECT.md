# Evolving Testing Project: Hybrid QML-KG Link Prediction

This document defines a repeatable experiment system for testing how far the Hybrid QML-KG theory can go.

The core theory is:

> Knowledge graph structure plus quantum-ready representations may improve biomedical link prediction, especially when the model must generalize from sparse treatment edges.

The testing project is not a single notebook. It is a continuously expanding experiment harness that runs controlled tests, records every result, keeps the best configurations, and pushes the boundary across larger datasets, richer embeddings, stronger baselines, and noisier quantum execution.

---

## 1. What we are testing

The current project predicts whether a `Compound` treats a `Disease` in Hetionet. The first target relation is `CtD`.

We test five levels of evidence:

| Level | Question | Evidence needed |
|---|---|---|
| L1: Reproducibility | Can we reproduce the current baseline results? | Same config + multiple seeds gives stable PR-AUC |
| L2: Classical strength | Which classical model is the real baseline? | Logistic Regression, SVM, Random Forest, and later GNN baselines |
| L3: Quantum signal | Does QSVC/VQC beat comparable classical models? | Mean PR-AUC improvement over seed variance |
| L4: Noise robustness | Does the quantum result survive noise? | Ideal vs noisy simulator gap is acceptable |
| L5: Scaling | Does performance improve or collapse as graph size grows? | PR-AUC/runtime curves across sample sizes |

Primary metric: **test PR-AUC**.

Secondary metrics:

- test ROC-AUC
- test F1
- recall at top-k
- precision at top-k
- runtime seconds
- memory footprint
- quantum shots
- circuit depth
- number of qubits
- noisy-vs-ideal degradation

---

## 2. The evolving loop

Each experiment cycle follows this loop:

```text
hypothesis -> config -> run -> score -> rank -> promote winners -> mutate configs -> repeat
```

### Step 1: Generate candidate configs

Examples:

- model type: LogisticRegression, SVM, RandomForest, QSVC, VQC
- relation: CtD first, then other Hetionet relation types
- max_entities: 300, 600, 1000, 2000, full subgraph
- embedding_dim: 32, 64, 128, 256
- qml_dim: 2, 3, 4, 5, 6, 8
- QML feature map: ZFeatureMap, ZZFeatureMap, PauliFeatureMap
- ansatz depth: 1, 2, 3
- optimizer: COBYLA, SPSA
- execution mode: ideal simulator, noisy simulator, hardware

### Step 2: Run the experiment

Each run must write:

- exact command
- git commit
- config hash
- seed
- start/end time
- metrics
- stdout/stderr tail
- artifact locations

### Step 3: Score the run

Default score:

```text
score = test_pr_auc - penalty(runtime_seconds) - penalty(noise_drop)
```

The scoring function should stay simple at first. Do not over-optimize until the raw metric collection is reliable.

### Step 4: Promote winners

A config is promoted when:

- it improves mean test PR-AUC over the current baseline
- the improvement is larger than seed variance
- it does not collapse under noisy simulation
- it has acceptable runtime for the next scale tier

### Step 5: Mutate the next search

Winners create the next generation:

- increase `max_entities`
- increase/decrease `qml_dim`
- change feature representation
- increase quantum circuit depth
- try a stronger embedding model
- add a harder negative sampling strategy

---

## 3. Phase ladder

### Phase A: Classical stability

Goal: establish the strongest non-quantum baseline.

Run:

- Logistic Regression
- SVM/RBF
- Random Forest
- 5+ seeds
- 300 to 1000 entities

Promotion criterion:

```text
best classical model = highest mean test PR-AUC with acceptable std
```

### Phase B: Quantum kernel comparison

Goal: compare QSVC against classical SVM.

Run:

- QSVC with 2 to 6 qml dimensions
- ZFeatureMap and ZZFeatureMap
- reps 1 to 3
- ideal simulator first

Promotion criterion:

```text
QSVC_mean_PR_AUC > SVM_mean_PR_AUC + practical_margin
```

A practical margin starts at 0.02 PR-AUC until there are enough seeds for tighter statistics.

### Phase C: Variational quantum classifier tuning

Goal: determine whether VQC is worth continuing.

Run:

- RealAmplitudes ansatz
- reps 1 to 3
- COBYLA and SPSA
- 50, 100, 300 max iterations
- multiple random initializations

Promotion criterion:

```text
VQC beats random and approaches QSVC/classical baselines without unstable variance
```

### Phase D: Noise and hardware readiness

Goal: check if ideal-simulator wins survive realistic execution.

Run:

- ideal simulator
- noisy simulator
- backend-informed noise model when available
- hardware only after the config is small and stable

Promotion criterion:

```text
noisy_PR_AUC_drop <= acceptable_noise_gap
```

Start with `acceptable_noise_gap = 0.05`.

### Phase E: Candidate ranking

Goal: generate useful biomedical hypotheses.

Run:

- top-k candidate compounds for a disease
- top-k candidate diseases for a compound
- KG neighbor evidence extraction
- rank stability across seeds

Promotion criterion:

```text
top candidates remain stable across seeds and have interpretable KG evidence
```

---

## 4. Directory structure

Recommended project layer:

```text
experiments/
├── README.md
├── evolving_harness.py
├── configs/
│   └── evolving_ctd.yaml
└── results/
    ├── runs.jsonl
    ├── leaderboard.csv
    └── promoted_configs.json
```

The `results/` folder can stay gitignored if outputs get large.

---

## 5. Minimum viable testing loop

Run the first evolving cycle:

```bash
python experiments/evolving_harness.py \
  --config experiments/configs/evolving_ctd.yaml \
  --phase classical_stability
```

Then inspect:

```bash
cat results/evolving/runs.jsonl | tail -n 5
cat results/evolving/leaderboard.csv
```

---

## 6. What counts as progress

This project progresses when it answers increasingly harder questions:

1. Can we reproduce the initial PoC?
2. What is the strongest classical baseline?
3. Does QSVC beat the best fair classical comparator?
4. Does the result survive seed variation?
5. Does the result survive noisy simulation?
6. Does it scale beyond 300 entities?
7. Does it produce stable top-k biomedical hypotheses?
8. Can the dashboard show each run as part of an evidence trail?

---

## 7. Guardrails

Avoid these failure modes:

- claiming quantum advantage from one seed
- comparing QSVC only against weak classical baselines
- using ROC-AUC alone on imbalanced data
- treating random negative samples as enough
- scaling quantum circuits before proving small circuits work
- running hardware before noisy simulation passes
- reporting candidate treatments without evidence context

---

## 8. Best next upgrades

After the minimum loop works, add:

- Optuna-driven search for classical and QML hyperparameters
- MLflow or Weights & Biases logging
- stronger negative sampling
- GNN baseline
- PyKEEN model comparison: TransE, DistMult, ComplEx, RotatE
- candidate ranking dashboard
- nightly scheduled local runs on DGX Spark
- hardware-readiness check for IBM Quantum backends
