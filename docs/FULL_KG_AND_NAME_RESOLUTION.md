# Full knowledge graph + human-readable names — implementation spec

Two things are being requested:

1. **Full KG coverage** — embeddings that cover all 47,031 Hetionet entities, not just
   the ~464 compounds and diseases that appear in CtD edges.
2. **Human-readable names** — so users can type "aspirin" or "hypertension" instead of
   `Compound::DB00945` or `Disease::DOID:10763`.

Both are almost entirely already in place. This document describes the current state,
the exact gap, and the precise changes needed to close it.

---

## Current state

### Names — already working for all 47,031 entities

`data/hetionet-v1.0-nodes.tsv` (2.35 MB, columns: `id`, `name`, `kind`) contains the
authoritative human-readable name for every node in Hetionet. The enrichment step
already ran and produced `data/hetionet_nodes_metadata.csv` (3.85 MB, columns:
`node_id`, `name`, `namespace`, `external_url`).

The orchestrator's `_load_entity_mappings()` reads `hetionet_nodes_metadata.csv` and
indexes each node on three keys:
- Full ID: `Compound::DB00945`
- ID suffix: `DB00945`
- Lowercase human name: `acetylsalicylic acid`

**This already works for all entity types** — compounds, diseases, genes, pathways,
anatomy, side effects, symptoms. The name coverage is complete. The only limitation
is that Hetionet uses INN/systematic names, not brand names. "aspirin" will not
resolve; "acetylsalicylic acid" will. That is a data limitation, not a code one.

### Embeddings — partially there

| File | Entities covered | Dimensions | Source |
|---|---|---|---|
| `entity_embeddings.npy` + `entity_ids.json` | ~464 (CtD scope) | 128 | CtD-scoped pipeline run |
| `rotate_128d_entity_embeddings.npy` + ids | ~464 | 128 | Same |
| `rotate_256d_entity_embeddings.npy` + ids | ~464 | 256 | Same |
| `rotate_512d_entity_embeddings.npy` + ids | ~464 | 512 | Same |

All existing embedding files cover only the entities that appear in the CtD relation.
The `rotate_512d_entity_ids.json` is 11.28 KB — the same size as the others — which
confirms it is still CtD-scoped despite being trained with `--full_graph_embeddings`.

A true full-graph embedding over all 47,031 entities would produce an `entity_ids.json`
of approximately 1.5–2 MB and an `entity_embeddings.npy` of approximately 45 MB at
128 dimensions.

---

## What needs to change

### Change 1 — Run the pipeline with no entity cap

The `--full_graph_embeddings` flag trains the embedding model on all 2.25 M Hetionet
edges, but `extract_task_edges()` is still called with the default `max_entities` cap,
which limits which entities get IDs assigned and therefore which entities appear in the
final embedding matrix.

To embed all 47,031 entities, run with `--max_entities` unset or set to a value larger
than the full entity count:

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

Do **not** pass `--max_entities`. The default in the pipeline is `None` for the full
dataset. Confirm this by checking that `args.max_entities` is `None` (not 300) when
no flag is passed.

After this run, verify:

```python
import numpy as np, json

emb = np.load("data/entity_embeddings.npy")
ids = json.load(open("data/entity_ids.json"))

print(f"Embedding matrix shape: {emb.shape}")
# Expected: (47031, 128) for full graph
# Current:  (~464, 128) for CtD-scoped

print(f"Entity count: {len(ids)}")
# Expected: 47031
# Current:  ~464
```

### Change 2 — Cross-check that the CtD task still works after full embedding

After retraining on the full entity vocabulary, the orchestrator's feature builder
calls `embedder._get_vec(entity_id, reduced=True)`. This works correctly as long as
the entity ID is present in `entity_to_id`. With full-graph embeddings, all 47,031
entities will be present, so CtD prediction will continue to work — and prediction for
any other entity pair (e.g. compound-gene, disease-anatomy) will also become possible.

### Change 3 — Add a brand name alias layer (optional, for "aspirin" → works)

Hetionet uses INN/WHO International Nonproprietary Names for drugs. Common brand names
(Aspirin, Tylenol, Advil) are not in the dataset. The orchestrator's fuzzy fallback
helps for partial matches but cannot bridge brand → INN without an external mapping.

**Option A — DrugBank synonyms file (most complete)**

DrugBank provides a `drugbank_all_full_database.xml` or CSV export with full synonym
lists. The relevant column is `synonyms` or `international-brands`. Requires a free
DrugBank account.

Once downloaded, build a synonym → DrugBank ID mapping and extend
`_load_entity_mappings()`:

```python
# After loading hetionet_nodes_metadata.csv, optionally extend with synonyms
synonym_path = os.path.join(self.data_dir, "drugbank_synonyms.csv")
if os.path.exists(synonym_path):
    syn_df = pd.read_csv(synonym_path)
    # Columns expected: drugbank_id, synonym
    for _, row in syn_df.iterrows():
        het_id = f"Compound::{row['drugbank_id']}"
        synonym = str(row['synonym']).strip().lower()
        if het_id in self.id_to_name:
            self.name_to_id.setdefault(synonym, het_id)
    logger.info(f"Extended name lookup with {len(syn_df)} DrugBank synonyms.")
```

**Option B — Minimal hardcoded common-name map (fast, limited)**

Add a small lookup dict in the orchestrator for the most commonly used brand names:

```python
COMMON_NAME_ALIASES = {
    "aspirin":       "Compound::DB00945",
    "ibuprofen":     "Compound::DB01050",
    "metformin":     "Compound::DB00331",
    "atorvastatin":  "Compound::DB01076",
    "dexamethasone": "Compound::DB01234",
    "prednisone":    "Compound::DB00635",
    "warfarin":      "Compound::DB00682",
    "acetaminophen": "Compound::DB00316",
    "diabetes":      "Disease::DOID:9351",
    "hypertension":  "Disease::DOID:10763",
    "cancer":        "Disease::DOID:162",
}
```

Load these before the metadata CSV in `_load_entity_mappings()` so the CSV can
override them if a better name exists.

---

## What the resolution chain looks like after these changes

When a user types `"aspirin"` into the prediction form:

```
1. name_to_id.get("aspirin")
   → found via COMMON_NAME_ALIASES (Option B)
   → OR found via DrugBank synonyms (Option A)
   → OR not found → go to step 2

2. name_to_id.get("aspirin")  [case-insensitive — already handled]
   → not found (Hetionet calls it "acetylsalicylic acid")

3. fuzzy match: any key in name_to_id where "aspirin" in key.lower()
   → may find "acetylsalicylic acid" if partial match logic is extended

4. raise ValueError("Entity 'aspirin' not found in KG.")
   → API returns error with hint to use INN name
```

The API error message in `predict_link_probability` should include a hint:

```python
raise ValueError(
    f"Entity '{name_or_id}' not found in KG. "
    f"Hetionet uses systematic/INN names — try the INN name or DrugBank ID. "
    f"Example: 'acetylsalicylic acid' not 'aspirin', 'DB00945' also works."
)
```

---

## Embedding coverage for non-CtD entity types

After full-graph embedding, the orchestrator can resolve and embed any of the
11 Hetionet entity types:

| Kind | Count | Example name resolution |
|---|---|---|
| Compound | 1,552 | "acetylsalicylic acid" or "DB00945" |
| Disease | 137 | "hypertension" or "DOID:10763" |
| Gene | 20,945 | "BRCA1" or "1" (Entrez ID) |
| Anatomy | 402 | "liver" or "UBERON:0002107" |
| Biological Process | 11,381 | "apoptosis" or "GO:0006915" |
| Molecular Function | 2,884 | "kinase activity" or "GO:0016301" |
| Cellular Component | 1,391 | "nucleus" or "GO:0005634" |
| Pathway | 1,822 | "glycolysis" |
| Pharmacologic Class | 345 | "beta blocker" |
| Side Effect | 5,734 | "nausea" or "C0027497" |
| Symptom | 438 | "fever" or "D005334" |

All of these will resolve via `hetionet_nodes_metadata.csv` once the full-graph
embeddings are trained. Whether the model produces meaningful predictions for
non-CtD pairs depends on whether the relation type was seen during training —
for link prediction outside CtD, a relation-specific model would need to be trained.

---

## Summary — order of operations

| Step | Action | Time |
|---|---|---|
| 1 | Run pipeline with `--full_graph_embeddings` and no `--max_entities` cap | ~30 min |
| 2 | Verify `entity_embeddings.npy` shape is `(47031, 128)` | 1 min |
| 3 | Restart API — orchestrator picks up new embeddings on next init | 1 min |
| 4 | (Optional) Add `COMMON_NAME_ALIASES` dict to orchestrator for brand names | 20 min |
| 5 | (Optional) Download DrugBank synonyms CSV and wire into `_load_entity_mappings()` | 1 hr |
| 6 | Improve API error message to hint at INN naming convention | 10 min |
