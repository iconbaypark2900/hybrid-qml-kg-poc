# Next steps — hybrid QML-KG biomedical link prediction

This document reflects the state of the project as of April 2026. The deployment
stack is working end-to-end. The science layer has known gaps. The next steps
are ordered by what unblocks the most.

---

## Where things stand right now

### What is fully working
- Train → serialize → serve → display loop closed
- FastAPI prediction service (`/predict-link`, `/batch-predict`, `/ranked-mechanisms`)
- Entity name resolution across all 47,031 Hetionet nodes (INN names + DrugBank IDs)
- Next.js dashboard with home page, predict page, experiments, simulation control
- Docker deployment (`docker compose up --build`)
- Classical benchmark: ExtraTrees PR-AUC 0.81, stacking ensemble PR-AUC ~0.80
- Benchmark spec locked (`docs/BENCHMARK_SPEC.md`)

### What is incomplete
- Hard negative mining wired into pipeline (functions written, call not replaced)
- Experiment registry (`scripts/benchmark_registry.py` does not exist)
- Heron hardware training script (`scripts/train_on_heron.py` is empty)
- Embedding coverage: ~464 of 464 CtD entities, but full-graph (47,031) not yet run
- QSVC on simulator has not hit the ≥ 0.65 PR-AUC target
- Discovery metrics (top-10 hit rate, mean rank) not yet computed
- Noisy simulator tier not benchmarked
- No ablation matrix comparing matched vs mismatched feature regimes

---

## Immediate — unblock the science (1–3 days)

### 1. Wire hard negatives into the pipeline (30 min)

The three strategy functions exist in `kg_layer/kg_loader.py`. The pipeline's
`_sample_negs_hard()` inside `run_optimized_pipeline.py` is still an inline
reimplementation. Replace its body:

```python
from kg_layer.kg_loader import get_hard_negatives

def _sample_negs_hard(n: int, seed: int) -> pd.DataFrame:
    return get_hard_negatives(
        pos_df, strategy="degree_corrupt", num_negatives=n, random_state=seed
    )
```

Then rerun:

```bash
python scripts/run_optimized_pipeline.py \
  --relation CtD --full_graph_embeddings \
  --embedding_method RotatE --embedding_dim 128 --embedding_epochs 200 \
  --negative_sampling hard --model_dir models --results_dir results
```

This is the single highest-leverage change remaining. Every PR-AUC number
produced with random negatives is inflated. Hard negatives make the benchmark
credible.

### 2. Create the experiment registry (1 hr)

Create `scripts/benchmark_registry.py` per the spec in
`docs/BENCHMARK_REGISTRY_SPEC.md`. Then add a `register_run()` call at the
end of `run_optimized_pipeline.py`. Every run after that point will have a
provenance record in `results/benchmark_registry.jsonl`.

Without this, runs cannot be compared systematically and results are not
citable.

### 3. Run the full-graph embedding pipeline (pipeline runtime, ~30–60 min)

The current `entity_embeddings.npy` covers ~464 CtD entities. Running with
`--full_graph_embeddings` and no `--max_entities` cap will produce embeddings
for all 47,031 Hetionet entities. This closes the entity coverage gap and
enables name resolution to work for genes, pathways, anatomy, and side effects —
not just compounds and diseases.

Verify after the run:
```python
import numpy as np, json
emb = np.load("data/entity_embeddings.npy")
ids = json.load(open("data/entity_ids.json"))
print(emb.shape)   # target: (47031, 128)
print(len(ids))    # target: 47031
```

---

## Short-term — improve scientific credibility (1–2 weeks)

### 4. Build the ablation matrix

The benchmark spec requires comparing QSVC against classical on matched feature
regimes. Run these four conditions and record all results to the registry:

| Condition | Models | Feature dim |
|---|---|---|
| A | All classical | 512-dim (full) |
| B | All classical | 16-dim (PCA-reduced, matched to quantum) |
| C | QSVC | 16-dim |
| D | Stacking ensemble | A + C combined |

Condition B vs C is the scientifically honest quantum comparison. Condition A
gives the classical ceiling. Without B, any quantum-vs-classical claim is invalid
per the benchmark spec.

```bash
# Condition A — full classical
python scripts/run_optimized_pipeline.py --relation CtD \
  --negative_sampling hard --classical_only --results_dir results/ablation_A

# Condition B — classical on reduced features (matches quantum input dim)
python scripts/run_optimized_pipeline.py --relation CtD \
  --negative_sampling hard --classical_only \
  --restrict_classical_to_qml_dim --qml_dim 16 --results_dir results/ablation_B

# Condition C — quantum only
python scripts/run_optimized_pipeline.py --relation CtD \
  --negative_sampling hard --quantum_only \
  --qml_dim 16 --qml_feature_map Pauli --qml_feature_map_reps 2 \
  --qsvc_C 0.1 --results_dir results/ablation_C

# Condition D — ensemble
python scripts/run_optimized_pipeline.py --relation CtD \
  --negative_sampling hard --run_ensemble --ensemble_method stacking \
  --results_dir results/ablation_D
```

### 5. Add discovery metrics to the evaluation output

The current evaluation reports PR-AUC, ROC-AUC, and F1. The benchmark spec
also requires top-10 hit rate and mean rank of true positives. These are
discovery-oriented metrics that matter if the project is framed as a drug
repurposing tool rather than a pure classifier.

Add to `classical_baseline/evaluate_baseline.py`:

```python
def top_k_hit_rate(y_true, y_scores, k=10):
    """Fraction of test positives appearing in the top-k ranked predictions."""
    idx = np.argsort(y_scores)[::-1][:k]
    return y_true[idx].sum() / max(1, y_true.sum())

def mean_rank_of_positives(y_true, y_scores):
    """Average rank of true positive edges in the ranked candidate list."""
    ranked = np.argsort(y_scores)[::-1]
    ranks = np.where(y_true[ranked] == 1)[0] + 1  # 1-indexed
    return ranks.mean() if len(ranks) > 0 else float("nan")
```

### 6. Run the noisy simulator tier

The benchmark defines three tiers: ideal simulator, noisy simulator, hardware.
Only the ideal simulator has been run. Noisy simulation is cheap and provides a
more realistic picture of what hardware performance would look like.

```bash
# Edit config/quantum_config.yaml:
#   execution_mode: simulator
#   simulator:
#     noise_model: ibm_torino   # or any calibrated device name

python scripts/run_optimized_pipeline.py --relation CtD \
  --negative_sampling hard --quantum_only \
  --qml_dim 16 --qml_feature_map Pauli \
  --results_dir results/noisy_sim
```

Record to the registry with `backend.execution_mode = "simulator_noisy"` so
results stay separated from ideal-simulator runs.

---

## Medium-term — extend and harden (2–4 weeks)

### 7. Write `scripts/train_on_heron.py`

The full spec is in `docs/TRAIN_ON_HERON_SPEC.md`. Key sections:
- `_resolve_token()` — reads IBM_Q_TOKEN from env
- `_preflight(args)` — validates token, checks backend reachability, estimates cost
- `main()` — data → hard negatives → embeddings → QMLLinkPredictor → evaluate → register

Start with `--dry_run` to validate the token and backend without submitting jobs,
then run a real QSVC job with `--max_entities 200 --max_iter 25 --shots 2000`.

This is required to get any hardware results and to move the "Hardware quantum"
row in the benchmark spec from "blocked" to a real number.

### 8. Expand to additional relation types

The pipeline currently only benchmarks CtD (Compound-treats-Disease). Hetionet
contains 24 relation types. Two strong candidates for expansion:

- **DaG** (Disease-associates-Gene) — links disease phenotypes to genetic basis
- **CbG** (Compound-binds-Gene) — links drugs to molecular targets

Expanding to these validates that the pipeline generalises beyond CtD and
produces a more substantial paper narrative. The only change needed is passing
`--relation DaG` or `--relation CbG` to the pipeline — everything else is
already parameterised.

```bash
python scripts/run_optimized_pipeline.py --relation DaG \
  --full_graph_embeddings --negative_sampling hard \
  --embedding_method RotatE --embedding_dim 128
```

### 9. Add brand name aliases for common drugs

Currently "aspirin" does not resolve — the user must type "acetylsalicylic acid".
Add a `COMMON_NAME_ALIASES` dict to `middleware/orchestrator.py` covering the
most frequently queried drugs:

```python
COMMON_NAME_ALIASES = {
    "aspirin":        "Compound::DB00945",
    "ibuprofen":      "Compound::DB01050",
    "metformin":      "Compound::DB00331",
    "atorvastatin":   "Compound::DB01076",
    "dexamethasone":  "Compound::DB01234",
    "prednisone":     "Compound::DB00635",
    "warfarin":       "Compound::DB00682",
    "acetaminophen":  "Compound::DB00316",
    "paracetamol":    "Compound::DB00316",
}
```

For full coverage, download the DrugBank synonyms CSV (free account required)
and wire it into `_load_entity_mappings()` per `docs/FULL_KG_AND_NAME_RESOLUTION.md`.

### 10. Improve the API error message for name resolution failures

When a name does not resolve, the current error is generic. Replace it with a
message that tells the user exactly what to try:

```python
raise ValueError(
    f"'{name_or_id}' not found. Hetionet uses systematic/INN drug names. "
    f"Try: 'acetylsalicylic acid' (not 'aspirin'), 'DB00945' also works. "
    f"For diseases try: 'hypertension', 'type 2 diabetes mellitus', or 'DOID:10763'."
)
```

---

## Longer-term — towards publication (1–3 months)

### 11. Statistical validation

A single train/test split is not sufficient for a publishable result. Add:
- 5-fold cross-validation across all model families
- Multi-seed evaluation (seeds 42, 123, 456, 789, 1011)
- Wilcoxon signed-rank test for quantum vs classical significance (p < 0.05)

The cross-validation framework exists in `experiments/cross_validation_framework.py`.
It needs to be wired into the main pipeline and its results added to the registry.

### 12. Interpretability layer

The blueprint calls for adding explanations to top-ranked predictions — which
graph paths and features drove the score. This would make the system useful to
biomedical researchers who need to understand *why* a prediction was made, not
just what the score is.

Minimum viable version: for the top-K predictions, retrieve the 2-hop subgraph
in Hetionet connecting the compound to the disease and display the most
relevant intermediate nodes (shared genes, pathways, anatomy).

The `/viz/kg-subgraph` API endpoint already exists and can power this.

### 13. Paper outline

The project has the structure of a publishable workshop paper (QCE, QTML,
or a biomedical informatics venue). The narrative is:

1. Problem — biomedical link prediction is a graph structure problem
2. Method — KG embeddings + classical baselines + quantum kernel comparison
3. Honest evaluation — hard negatives, matched feature regimes, three tiers
4. Results — classical ceiling (0.81), quantum on simulator, quantum on hardware
5. Analysis — where quantum helps, where it does not, computational cost tradeoff

The gap between current state and paper-ready is primarily steps 4 (ablation
matrix) and 6 (noisy sim + hardware baseline). Everything else is already done
or in progress.

---

## Priority summary

| Priority | Item | Effort | Blocks |
|---|---|---|---|
| 1 | Wire hard negatives into pipeline | 30 min | Scientific credibility |
| 2 | Create benchmark registry | 1 hr | Provenance, citeability |
| 3 | Run full-graph embedding pipeline | ~60 min runtime | Entity coverage, name resolution |
| 4 | Run ablation matrix (conditions A–D) | 2 hr | Valid quantum comparison |
| 5 | Add discovery metrics (top-k, mean rank) | 2 hr | Biomedical framing |
| 6 | Run noisy simulator tier | 1 hr | Second benchmark tier |
| 7 | Write `scripts/train_on_heron.py` | 2 hr | Hardware results |
| 8 | Expand to DaG and CbG relations | 1 hr | Generalisation claim |
| 9 | Add brand name aliases | 30 min | User experience |
| 10 | Statistical validation (CV + multi-seed) | 3 hr | Publication readiness |
| 11 | Interpretability layer | 1 day | Research utility |
| 12 | Paper draft | ongoing | Dissemination |
