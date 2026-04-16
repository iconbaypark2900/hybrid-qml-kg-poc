# Platform Extensions — Longer-Term Build Items

**Status:** Not started  
**Horizon:** Weeks to months; some require significant infrastructure

These are features that would make this a genuinely differentiated
production drug-repurposing research platform rather than a research
proof-of-concept.

---

## 1. ClinicalTrials.gov Live Query Integration

### Problem

Clinical validation of novel predictions (§7 of the paper) is currently
done manually by searching ClinicalTrials.gov for each (compound, disease)
pair. This is slow, not reproducible, and can only be done for a handful
of top predictions.

### What to build

A module that automatically annotates every prediction with live
ClinicalTrials.gov data:

```python
# Proposed: kg_layer/clinical_trials_lookup.py

import requests

CTGOV_API = "https://clinicaltrials.gov/api/v2/studies"

def query_trials(compound: str, disease: str, max_results: int = 20) -> dict:
    """Return trial count, phases, and statuses for a compound-disease pair."""
    params = {
        "query.intr": compound,
        "query.cond": disease,
        "fields": "NCTId,Phase,OverallStatus",
        "pageSize": max_results,
        "format": "json",
    }
    resp = requests.get(CTGOV_API, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    studies = data.get("studies", [])
    return {
        "total": data.get("totalCount", 0),
        "phases": [s.get("Phase", "N/A") for s in studies],
        "statuses": [s.get("OverallStatus", "N/A") for s in studies],
    }
```

**Integration points:**
- Call after scoring top-N predictions in `scripts/run_optimized_pipeline.py`
- Expose via `GET /predict/validate` in `middleware/api.py`
- Display in the prediction results panel in the Next.js UI
- Add `trial_count` field to benchmark registry JSON per prediction

**Rate limiting:** ClinicalTrials.gov API is public and free. Cache results
to avoid hammering the API on repeated pipeline runs with the same pairs.

---

## 2. Multi-Relational Joint Training

### Problem

Currently CtD, CpD, and DrD are run as completely separate experiments
with separate embedding training runs. Each relation's QSVC sees only
its own (compound, disease) pairs and the embeddings are trained for
the same task each time.

### What to build

A shared quantum feature space that encodes signal from multiple relations
simultaneously:

**Option A — Shared embeddings, separate classifiers**

Train RotatE once over all 2.25M Hetionet edges (already done with
`--full_graph_embeddings`). Then train separate QSVC classifiers for
CtD, CpD, DrD on the same embedding space. This is already achievable
with the current pipeline — just needs the experiments to be run.

**Option B — Multi-task stacking**

Train classifiers for CtD, CpD, and DrD simultaneously. Use predictions
from CpD and DrD as additional features for the CtD classifier (i.e., a
compound that palliates a disease is a stronger CtD candidate). This
requires a new multi-task training loop.

**Option C — Relation-aware quantum kernel**

Encode the relation type as part of the quantum feature map input:
```
U(x_compound, x_disease, r_encoding) where r_encoding is a 2-bit one-hot
for {CtD, CpD, DrD}.
```
This allows the kernel to encode relation-type-specific similarity in
a single unified feature space. Relevant file:
`quantum_layer/quantum_enhanced_embeddings.py`

**Recommended starting point:** Option A (no code changes, just run the
three experiments). Document Option C as a research direction in the paper.

---

## 3. DRKG Extension

### What DRKG is

The Drug Repurposing Knowledge Graph (DRKG) integrates data from:
- DrugBank (drug–target, drug–drug interactions)
- STRING (protein–protein interactions)
- IntAct (molecular interactions)
- DGIdb (drug–gene interactions)
- Hetionet (all 24 relation types)
- GNBR (biomedical literature co-occurrence)

Scale: **4.4 million edges, 97,238 entities** across 107 relation types.

### Why this matters

Hetionet (2.25M edges, 47K entities) is a strong research graph but
was last updated in 2017. DRKG is more current and much larger. Extending
the pipeline to DRKG would:
- Produce embeddings richer in drug-target binding and protein interaction signal
- Enable validation of predictions against more diverse evidence
- Position the paper as a scalable approach rather than a graph-specific one

### What the extension requires

1. **Data loading:** DRKG is available as a TSV from
   `https://github.com/gnn4dr/DRKG`. The `kg_layer/kg_loader.py`
   `HetionetLoader` would need a `DRKGLoader` sibling class.

2. **Embedding training:** With 4.4M edges, full-graph RotatE training
   requires GPU (DGX). Use `./scripts/run_full_embedding_dgx.sh` as a
   template for a new `run_drkg_embedding.sh`.

3. **Nyström approximation:** With ~6,000+ CtD-equivalent edges in DRKG,
   the full quantum kernel is infeasible. Nyström with m=400–800 is mandatory.

4. **Feature construction:** The pair feature vector construction in
   `kg_layer/enhanced_features.py` is Hetionet-specific (references CbG,
   DaG, GpPW relation names). A DRKG-compatible feature constructor is needed.

---

## 4. Inference Robustness — Out-of-Distribution Pairs

### Problem

The `/predict` API currently calls `POST /predict` with a drug name and
disease name. If either entity is not in the embedding index:
- The embedding lookup silently fails or returns a zero vector
- The model scores the pair based on a zero-vector input
- The score is meaningless, but the user sees a number

### What to build

**Step 1: Embedding coverage check**

Before scoring, check whether the compound and disease are in the
embedding index:

```python
# In middleware/api.py
known_entities = set(json.load(open("data/entity_ids.json")).keys())

if compound not in known_entities:
    return {"error": f"'{compound}' not found in embedding index.",
            "suggestion": "Try a DrugBank ID (e.g. DB00960) or a canonical drug name."}
```

**Step 2: Nearest-neighbor fallback**

If the entity is not in the index, find the most similar entity by
name (fuzzy string match on entity names) and use its embedding as a
proxy, with a warning.

**Step 3: UI error state**

The `/predict` form in the Next.js UI should display a clear error
when an entity is not found, with suggestions from the known entity list.

---

## 5. GNN Baselines

### Current state

`kg_layer/gnn_baselines.py` exists. No GNN result has been reported.

### Why this matters

The paper compares against RotatE, ComplEx, and classical ML baselines
but not against GNN-based link predictors (GraphSAGE, R-GCN, or
CompGCN). A GNN baseline is increasingly expected by reviewers at
quantum-ML and bioinformatics venues.

### What to implement

- **R-GCN (Relational Graph Convolutional Network):** supports the 24
  heterogeneous Hetionet relation types natively
- **CompGCN:** composition-based message passing for multi-relational graphs

Train each on the same CtD train split, report PR-AUC on the test split
in a new Table 2 (GNN baselines) before the primary Table 3.

**Library:** PyKEEN supports R-GCN and CompGCN natively — use the same
PyKEEN training interface as RotatE/ComplEx.

---

## 6. Inference API Latency Optimization

### Problem

The current `/predict` endpoint rebuilds the full feature vector on every
request:
- Loads the embedding index
- Computes graph topology features (degree, Jaccard, common neighbors)
- Runs the classifier

For interactive use this is tolerable. For batch evaluation of thousands
of candidate pairs it is too slow.

### What to build

**Pre-computed feature cache:** At startup, compute and cache the graph
topology features for all known (compound, disease) pairs. Feature lookup
becomes O(1) instead of O(graph traversal).

**Batch endpoint:**

```python
# POST /predict/batch
# Body: {"pairs": [["pindolol", "hypertension"], ["ezetimibe", "gout"], ...]}
# Returns: [{"compound": ..., "disease": ..., "score": ..., "moa": {...}}, ...]
```

This enables researchers to score all ~1,552 × 137 = ~212K compound-disease
pairs in a single call for full-graph candidate ranking.

---

## 7. arXiv Submission Preparation

These are non-code tasks required before arXiv submission:

| Task | Owner | Status |
|------|-------|--------|
| Fix citation keys in `paper.tex` (all `[?]` refs) | Author | Pending |
| Render fig1, fig2, fig3 per Appendix A spec | Author | Pending (files in `figures/` untracked) |
| Add `figures/` to git | Author | Pending |
| Run `pdflatex` twice, verify no errors | Author | Pending |
| Add MoA benchmark results to Table 3 | Requires experiment first | Blocked |
| Add CpD results | Requires experiment first | Blocked |
| Add multi-seed mean ± std to Table 3 | Requires 5 seeds | Blocked |
| Add degree-heuristic baseline row | Requires computation | Pending |
| Add random baseline row (PR-AUC = 0.50) | Trivial | Pending |
| Update §8.5 Limitations to remove resolved items | After above experiments | Blocked |
| Submit to arXiv in categories: `quant-ph`, `cs.LG`, `q-bio.QM` | Author | Blocked |
