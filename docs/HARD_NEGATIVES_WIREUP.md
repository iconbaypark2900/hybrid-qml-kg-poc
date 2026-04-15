# Hard negative mining — wire-up spec

The three strategy functions are now implemented in `kg_layer/kg_loader.py`.
This document describes the one remaining change: replacing the inline
`_sample_negs_hard` implementation in `run_optimized_pipeline.py` with a
call to the canonical loader function.

---

## Change in `scripts/run_optimized_pipeline.py`

### Step 1 — Add import at the top of the file

```python
from kg_layer.kg_loader import get_hard_negatives
```

### Step 2 — Replace `_sample_negs_hard`

Find the function `_sample_negs_hard` (defined inside `main()`). It currently
contains approximately 30 lines of inline degree-weighted sampling logic.

Replace the entire body with:

```python
def _sample_negs_hard(n: int, seed: int) -> pd.DataFrame:
    """Hard negatives via degree-weighted KG corruption.
    Delegates to the canonical implementation in kg_layer.kg_loader.
    """
    return get_hard_negatives(
        pos_df,
        strategy="degree_corrupt",
        num_negatives=n,
        random_state=seed,
    )
```

`pos_df` is the positive-edges DataFrame already in scope at the point where
`_sample_negs_hard` is defined inside `main()`. Confirm it uses columns
`source` and `target` (string entity IDs), not `source_id`/`target_id`
(integer IDs) — `get_hard_negatives` expects string IDs.

### Step 3 — Replace `_sample_negs_diverse`

If `_sample_negs_diverse` exists, replace its body with:

```python
def _sample_negs_diverse(n: int, seed: int) -> pd.DataFrame:
    """Mix of hard and random negatives."""
    n_hard   = int(n * (1.0 - diversity_weight))
    n_random = n - n_hard
    hard_df   = get_hard_negatives(pos_df, strategy="degree_corrupt",
                                   num_negatives=n_hard,  random_state=seed)
    random_df = _sample_negs_random(n_random, seed + 1)
    return pd.concat([hard_df, random_df], ignore_index=True)
```

### Step 4 — Verify the `--negative_sampling hard` flag still routes correctly

The pipeline already has a branch that calls `_sample_negs_hard` when
`args.negative_sampling == "hard"`. After the replacement in Step 2, this
branch will automatically use the canonical implementation.

Confirm the routing logic looks like:

```python
if neg_strategy == "hard":
    neg_train = _sample_negs_hard(len(pos_train), args.random_state)
    neg_test  = _sample_negs_hard(len(pos_test),  args.random_state + 1)
elif neg_strategy == "diverse":
    neg_train = _sample_negs_diverse(len(pos_train), args.random_state)
    neg_test  = _sample_negs_diverse(len(pos_test),  args.random_state + 1)
else:  # "random"
    neg_train = _sample_negs_random(len(pos_train), args.random_state)
    neg_test  = _sample_negs_random(len(pos_test),  args.random_state + 1)
```

---

## Verify the change works

```bash
# Short smoke test — fast mode with hard negatives
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --negative_sampling hard \
    --fast_mode \
    --classical_only \
    --max_entities 100

# Expected log output:
# INFO kg_layer.kg_loader: degree_corrupt: generated N hard negatives.
# (not "Generated N negative samples." which is the random path)
```

---

## What the three strategies produce for CtD

| Strategy | What gets corrupted | Replacement pool | Difficulty |
|---|---|---|---|
| `random` | tail or head uniformly | all entities | easy — trivially invalid pairs common |
| `degree_corrupt` | tail or head with prob 0.5 | degree-weighted entities | medium — high-degree nodes (promiscuous drugs/diseases) appear more |
| `type_aware` | tail or head with prob 0.5 | same entity-type only (Compound→Compound) | harder — structurally plausible, same entity family |
| `embedding_knn` | tail | K nearest neighbours in embedding space | hardest — maximally similar to true treatments in latent space |

For the benchmark, `degree_corrupt` is the training default and `type_aware`
is used for the hard evaluation set. `embedding_knn` is available for
ablation studies once the baseline is stable.
