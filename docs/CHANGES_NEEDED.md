# Changes needed — implementation checklist

Four areas, ordered by scientific impact. Each section states the file to change,
exactly what to add or replace, and why it matters.

---

## 1. Hard negative mining — `kg_layer/kg_loader.py`

**Status:** Three strategy functions are now fully written and present in the file
(`get_hard_negatives_degree_corrupt`, `get_hard_negatives_type_aware`,
`get_hard_negatives_embedding_knn`, `get_hard_negatives`). The import line at the
top of the file still uses `import numpy as np` via the old pattern — verify it is
not duplicated after the rewrite.

**One remaining wire-up:** `scripts/run_optimized_pipeline.py` contains an inline
`_sample_negs_hard()` function that reimplements degree corruption locally. It should
be replaced with a call to the canonical loader function so there is a single source
of truth.

### Change in `scripts/run_optimized_pipeline.py`

Find the internal `_sample_negs_hard` function (defined inside `main()`) and replace
the body with a delegation:

```python
# BEFORE — inline reimplementation inside main()
def _sample_negs_hard(n: int, seed: int) -> pd.DataFrame:
    ...  # ~30 lines of degree-weighted sampling logic

# AFTER — delegate to kg_loader
from kg_layer.kg_loader import get_hard_negatives

def _sample_negs_hard(n: int, seed: int) -> pd.DataFrame:
    return get_hard_negatives(
        pos_df,                     # the positive edges DataFrame in scope
        strategy="degree_corrupt",
        num_negatives=n,
        random_state=seed,
    )
```

Also update `_sample_negs_diverse` if it exists, to call:

```python
from kg_layer.kg_loader import get_hard_negatives

hard_part = get_hard_negatives(pos_df, strategy="degree_corrupt",
                                num_negatives=int(n * (1 - diversity_weight)),
                                random_state=seed)
random_part = _sample_negs_random(int(n * diversity_weight), seed + 1)
return pd.concat([hard_part, random_part], ignore_index=True)
```

**Why this matters:** Random negatives make every model look better than it is.
The benchmark's PR-AUC numbers are not credible for publication or comparison
until negatives are structurally hard. This is the single highest-leverage
scientific change remaining.

---

## 2. Heron hardware training — `scripts/train_on_heron.py`

**Status:** The file is empty. `quantum_layer/train_on_heron.py` has a stub body
but no CLI, no pre-flight checks, no cost estimate, and no provenance registration.

### Write `scripts/train_on_heron.py` with these sections

```
1. argparse block
   --relation, --max_entities, --embedding_dim
   --qubits, --model_type (QSVC|VQC), --feature_map, --feature_map_reps
   --ansatz_reps, --optimizer, --max_iter, --shots
   --backend (default: ibm_torino)
   --negative_sampling (default: degree_corrupt)
   --results_dir, --model_dir
   --dry_run  ← exits after pre-flight, no job submitted

2. _resolve_token()
   Read IBM_Q_TOKEN or IBM_QUANTUM_TOKEN from env.
   Strip whitespace/quotes. Exit if empty or placeholder.

3. _preflight(args)
   - Import qiskit_ibm_runtime; exit with instructions if missing.
   - Connect QiskitRuntimeService(channel, token).
   - List available backends; exit if args.backend not present.
   - Print cost estimate: shots × max_iter = O(N) total shots.
   - If --dry_run: sys.exit(0).

4. main()
   a. Call _preflight(args).
   b. Load Hetionet, extract CtD edges, split train/test.
   c. Generate hard negatives via get_hard_negatives(strategy=args.negative_sampling).
   d. Load/train embeddings (HetionetEmbedder, dim=args.embedding_dim).
   e. Write temporary quantum_config_heron_run.yaml pointing at args.backend.
   f. Instantiate QMLLinkPredictor with Heron config.
   g. model.fit(X_train_qml, y_train).
   h. Evaluate: pr_auc, roc_auc = average_precision_score, roc_auc_score.
   i. Print results table.
   j. Save JSON to results/heron_{model_type}_{stamp}.json.
   k. Call register_run() from scripts/benchmark_registry.py.
   l. Delete temporary config file.
```

**Key constraints to enforce in the script:**
- Default `--max_entities 200` — hardware is expensive.
- Default `--max_iter 25` — each iteration costs `shots` executions.
- Default `--model_type QSVC` — no variational loop, much cheaper than VQC.
- Default `--feature_map_reps 1` — limits circuit depth on NISQ hardware.
- SPSA is the recommended optimizer for hardware (gradient-free, shot-efficient).

**Why this matters:** Without this file the project cannot demonstrate any
hardware quantum execution. The Heron results tier is required by the benchmark
spec to be kept separate from simulator results.

---

## 3. Experiment registry — `scripts/benchmark_registry.py`

**Status:** Does not exist. The blueprint requires every run to store a versioned
provenance record.

### Create `scripts/benchmark_registry.py` with these exports

```python
def register_run(
    *,
    run_id: str,
    relation: str,
    embedding: dict,      # method, dim, epochs, full_graph
    reduction: dict,      # method, input_dim, output_dim, explained_variance
    model: dict,          # name, type, hyperparams, n_features
    backend: dict,        # name, execution_mode, shots, noise_model
    metrics: dict,        # pr_auc, roc_auc, f1, top10_hit_rate, mean_rank
    negative_sampling: dict = None,   # strategy, ratio
    circuit: dict = None,             # n_qubits, depth, feature_map, ansatz (None for classical)
    split: dict = None,               # test_size, random_state, n_train_pos, n_test_pos
    notes: str = "",
    registry_path: str = None,
) -> Path:
    """Append one JSON line to results/benchmark_registry.jsonl."""

def load_registry(registry_path=None) -> list[dict]:
    """Return all records from the registry as a list of dicts."""

def summarise_registry(registry_path=None) -> None:
    """Print a formatted table of all runs."""
```

Registry file: `results/benchmark_registry.jsonl` (one JSON object per line,
append-only, human-readable).

### Wire `register_run()` into the pipeline

At the end of `scripts/run_optimized_pipeline.py`, after the JSON payload is
written, add:

```python
from scripts.benchmark_registry import register_run

register_run(
    run_id=stamp,
    relation=args.relation,
    embedding={
        "method": args.embedding_method,
        "dim": args.embedding_dim,
        "epochs": args.embedding_epochs,
        "full_graph": args.full_graph_embeddings,
    },
    reduction={
        "method": "PCA",
        "output_dim": args.qml_dim,
        "pre_pca_dim": getattr(args, "qml_pre_pca_dim", None),
    },
    model={
        "name": best_overall["name"] if all_results else "none",
        "type": best_overall["type"] if all_results else "none",
        "pr_auc": best_overall["pr_auc"] if all_results else None,
    },
    backend={
        "name": "simulator_statevector",
        "execution_mode": "simulator",
        "shots": None,
        "noise_model": None,
    },
    metrics={
        "pr_auc": best_overall["pr_auc"] if all_results else None,
    },
    negative_sampling={
        "strategy": getattr(args, "negative_sampling", "random"),
    },
)
```

**Why this matters:** Without provenance records, runs cannot be compared
systematically. The registry is also what the benchmark spec (section 8) requires
for any result to be citable.

---

## 4. Benchmark spec — `docs/BENCHMARK_SPEC.md`

**Status:** Does not exist. Create it with the following sections.

```markdown
# Benchmark specification — version 1.0
Locked: 2026-04-12  |  Relation: CtD

## 1. Dataset
- Source: Hetionet v1.0
- Relation: Compound-treats-Disease (CtD)
- Positive edges: all 755 known CtD edges
- Entity scope: all 464 compounds and diseases in CtD
- Embedding training graph: full Hetionet (2.25 M edges, all relation types)
- Required entity coverage: ≥ 95 % of 464 CtD entities

## 2. Split policy
- Method: random edge split (not entity split)
- Test fraction: 0.20
- Random state: 42
- Leakage control: test edges excluded from embedding training graph

## 3. Negative sampling
- Default strategy: degree_corrupt
- Ratio: 1:1 (negatives : positives)
- Avoidance: all 755 known CtD edges excluded from negative pool
- Hard eval set: separate test set with type_aware negatives for realism check
- Rule: random-only negatives are not an acceptable sole evaluation condition

## 4. Feature regime
- Classical models: [h, t, |h−t|, h*t] on 128-dim RotatE → 512-dim
- Quantum models: PCA 512 → 24 (pre-PCA) → 16 (qml_dim), Pauli map, reps=2
- Fair comparison rule: QSVC (16-dim) must always be compared against
  classical models on BOTH full 512-dim AND same 16-dim PCA-reduced features.
  Comparing QSVC (16-dim) against classical (512-dim) only is not valid.

## 5. Model family
| Model                | Type      | Config                              |
|----------------------|-----------|-------------------------------------|
| LogisticRegression-L2| classical | C=1.0, balanced                     |
| ExtraTrees-Optimized | classical | n=600, sqrt features, balanced      |
| RandomForest-Optimized| classical| n=200, depth=10, balanced           |
| HistGBDT             | classical | depth=8, lr=0.06                    |
| QSVC-Pauli           | quantum   | C=0.1, Pauli map, reps=2, 16 qubits |
| Stacking-Ensemble    | ensemble  | LR meta-learner on OOF predictions  |

## 6. Primary evaluation metrics
| Metric              | Role      | Rationale                           |
|---------------------|-----------|-------------------------------------|
| PR-AUC              | Primary   | Handles class imbalance correctly   |
| ROC-AUC             | Secondary | Standard AUC for context            |
| F1 (threshold=0.5)  | Tertiary  | Classification quality              |
| Top-10 hit rate     | Discovery | Fraction of test positives in top-10 ranked candidates per disease |
| Mean rank           | Discovery | Average rank of true CtD edges in ranked candidate list |

## 7. Quantum execution tiers — must not be mixed
| Tier     | Label                   | Description                         |
|----------|-------------------------|-------------------------------------|
| ideal    | simulator_statevector   | Noiseless statevector simulation    |
| noisy    | simulator_noisy         | Aer with hardware-calibrated noise  |
| hardware | ibm_torino / ibm_heron  | Real IBM Quantum device             |

A claim of quantum advantage is valid only within a single tier.

## 8. Provenance requirements
Every benchmark run must call register_run() and store:
- embedding config (method, dim, epochs, full_graph)
- reduction config (method, input_dim, output_dim, explained_variance)
- model config (name, type, hyperparams, n_features)
- backend (name, execution_mode, shots, noise_model)
- circuit metadata for quantum runs (n_qubits, depth, feature_map, ansatz)
- negative sampling strategy and ratio
- split config (test_size, random_state)
- all primary metrics

## 9. Current performance vs targets
| Model              | PR-AUC target | Current status                    |
|--------------------|---------------|-----------------------------------|
| Best classical     | ≥ 0.75        | ✅ achieved (0.81, ExtraTrees)    |
| Best quantum (sim) | ≥ 0.65        | in progress                       |
| Best ensemble      | ≥ 0.78        | ✅ achieved (0.80, stacking)      |
| Hardware quantum   | baseline TBD  | blocked on train_on_heron.py      |
```

---

## 5. Embedding coverage gap — operational fix

**Status:** Cached embeddings cover ~249 of 464 CtD entities. This is a data
artifact, not a code bug.

**Fix:** Re-run the pipeline with `--full_graph_embeddings` and sufficient epochs
to cover all entities:

```bash
python scripts/run_optimized_pipeline.py \
  --relation CtD \
  --full_graph_embeddings \
  --embedding_method RotatE \
  --embedding_dim 128 \
  --embedding_epochs 200 \
  --negative_sampling hard \
  --model_dir models \
  --results_dir results
```

After this run:
- `data/entity_embeddings.npy` will cover all 464 entities
- `models/classical_best.joblib` will be updated
- `results/benchmark_registry.jsonl` will have a provenance record (once
  registry is wired in per section 3 above)

**Verify coverage:**

```python
import numpy as np, json
emb = np.load("data/entity_embeddings.npy")
ids = json.load(open("data/entity_ids.json"))
print(f"Embedding matrix: {emb.shape}")   # should be (464+, 128)
print(f"Entity count: {len(ids)}")         # should be ≥ 464
```

---

## Summary — order of operations

1. Wire `get_hard_negatives()` into `run_optimized_pipeline.py`
   replacing `_sample_negs_hard` — **30 min, highest scientific impact**
2. Create `scripts/benchmark_registry.py` and add `register_run()` call
   at the end of `run_optimized_pipeline.py` — **1 hr**
3. Create `docs/BENCHMARK_SPEC.md` — **30 min, copy from section 4 above**
4. Write `scripts/train_on_heron.py` — **2 hr, requires IBM token to test**
5. Re-run pipeline with `--full_graph_embeddings --negative_sampling hard`
   to close the coverage gap and generate the first registry entry — **pipeline runtime**
