# `scripts/benchmark_registry.py` — spec

Create this file. It does not exist yet.

## Purpose

Append-only provenance store for every pipeline run. The blueprint requires
every run to record embedding config, reduction config, model config, backend,
circuit metadata, and metrics. This file provides the canonical implementation.

## File location

`scripts/benchmark_registry.py`

## Registry file location

`results/benchmark_registry.jsonl` — one JSON object per line, append-only.

---

## Functions to implement

### `register_run(**kwargs) -> Path`

Appends one provenance record to the registry file.

**Required keyword arguments:**

| Argument | Type | Example |
|---|---|---|
| `run_id` | str | `"20260412-105405"` |
| `relation` | str | `"CtD"` |
| `embedding` | dict | `{"method": "RotatE", "dim": 128, "epochs": 200, "full_graph": True}` |
| `reduction` | dict | `{"method": "PCA", "pre_pca_dim": 24, "output_dim": 16}` |
| `model` | dict | `{"name": "ExtraTrees-Optimized", "type": "classical", "pr_auc": 0.81}` |
| `backend` | dict | `{"name": "simulator_statevector", "execution_mode": "simulator", "shots": None}` |
| `metrics` | dict | `{"pr_auc": 0.81, "roc_auc": 0.87}` |

**Optional keyword arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `negative_sampling` | dict | `{"strategy": "random"}` | Strategy and ratio |
| `circuit` | dict | `None` | Quantum circuit metadata; None for classical runs |
| `split` | dict | `{}` | test_size, random_state, n_train_pos, etc. |
| `notes` | str | `""` | Free-text annotation |
| `registry_path` | str | `None` | Override default path (useful for testing) |

**Record fields written automatically:**
- `timestamp` — UTC ISO 8601
- `host` — `socket.gethostname()`
- `python` — `sys.version`

**Returns:** `Path` to the registry file.

---

### `load_registry(registry_path=None) -> list[dict]`

Reads all records from the registry file and returns them as a list of dicts,
oldest first. Returns `[]` if the file does not exist.

---

### `summarise_registry(registry_path=None) -> None`

Prints a formatted table of all runs to stdout.

Columns: `run_id`, `model`, `type`, `backend`, `neg_strategy`, `PR-AUC`

---

## How to wire it into `run_optimized_pipeline.py`

Add this block at the end of `main()`, just after the JSON payload is written
to `out_path`:

```python
try:
    from scripts.benchmark_registry import register_run

    _best = all_results[0] if all_results else {}
    _best_classical = next((r for r in all_results if r.get("type") == "classical"), {})
    _best_quantum   = next((r for r in all_results if r.get("type") == "quantum"),   {})

    register_run(
        run_id=stamp,
        relation=args.relation,
        embedding={
            "method":     getattr(args, "embedding_method",  "unknown"),
            "dim":        getattr(args, "embedding_dim",     None),
            "epochs":     getattr(args, "embedding_epochs",  None),
            "full_graph": bool(getattr(args, "full_graph_embeddings", False)),
        },
        reduction={
            "method":      "PCA",
            "pre_pca_dim": getattr(args, "qml_pre_pca_dim", None),
            "output_dim":  getattr(args, "qml_dim",         None),
        },
        model={
            "name":    _best.get("name",   "none"),
            "type":    _best.get("type",   "none"),
            "pr_auc":  _best.get("pr_auc", None),
        },
        backend={
            "name":           "simulator_statevector",
            "execution_mode": "simulator",
            "shots":          None,
            "noise_model":    None,
        },
        metrics={
            "pr_auc":              _best.get("pr_auc"),
            "classical_pr_auc":    _best_classical.get("pr_auc"),
            "quantum_pr_auc":      _best_quantum.get("pr_auc"),
        },
        negative_sampling={
            "strategy": getattr(args, "negative_sampling", "random"),
            "ratio":    1.0,
        },
        split={
            "test_size":    0.20,
            "random_state": getattr(args, "random_state", 42),
        },
        notes=f"run_optimized_pipeline fast_mode={getattr(args, 'fast_mode', False)}",
    )
    logger.info("✅ Provenance registered → results/benchmark_registry.jsonl")
except Exception as _reg_err:
    logger.warning("Could not register run: %s", _reg_err)
```

The `try/except` wrapper ensures a registry failure never breaks the pipeline.

---

## CLI usage

Once created, the registry can be queried directly:

```bash
# List all runs
python scripts/benchmark_registry.py --list

# List runs from a custom registry path
python scripts/benchmark_registry.py --list --path results/benchmark_registry.jsonl

# Load in Python for analysis
import pandas as pd
df = pd.read_json("results/benchmark_registry.jsonl", lines=True)
df[["run_id", "model", "metrics"]].to_string()
```
