# `scripts/train_on_heron.py` — spec

Create this file. It currently exists but is **empty**.

Also update `quantum_layer/train_on_heron.py` as noted in section 3.

---

## Purpose

CLI entry point for training QSVC or VQC on IBM Quantum Heron hardware.
Includes token validation, backend reachability check, cost estimate,
hard negative generation, result persistence, and provenance registration.

---

## Argument groups

### Data
| Argument | Default | Notes |
|---|---|---|
| `--relation` | `CtD` | Hetionet relation type |
| `--max_entities` | `200` | Keep ≤ 300 for cost control |
| `--embedding_dim` | `16` | Small dim limits quantum feature cost |
| `--negative_sampling` | `degree_corrupt` | choices: random, degree_corrupt, type_aware |
| `--results_dir` | `results` | |
| `--model_dir` | `models` | |

### Circuit
| Argument | Default | Notes |
|---|---|---|
| `--qubits` | `4` | Equals QML feature dimension after PCA |
| `--model_type` | `QSVC` | QSVC = cheaper (no variational loop). VQC = expressive but costly |
| `--feature_map` | `ZZ` | choices: ZZ, Pauli, Z |
| `--feature_map_reps` | `1` | Keep at 1 on hardware to limit circuit depth |
| `--ansatz_reps` | `2` | VQC only |
| `--optimizer` | `SPSA` | SPSA recommended on hardware (gradient-free, shot-efficient) |
| `--max_iter` | `25` | Keep ≤ 50 on hardware — each iter costs shots |
| `--shots` | `2000` | |

### Backend
| Argument | Default | Notes |
|---|---|---|
| `--backend` | `ibm_torino` | Must be in the caller's IBM Quantum instance |

### Flags
| Argument | Default | Notes |
|---|---|---|
| `--dry_run` | False | Run pre-flight only; exit without submitting jobs |
| `--random_state` | `42` | |

---

## Implementation sections

### 1. `_resolve_token() -> str`

```python
for var in ("IBM_Q_TOKEN", "IBM_QUANTUM_TOKEN"):
    val = os.environ.get(var, "").strip().strip('"').strip("'")
    if val and val != "your_actual_token_here":
        return val
return ""
```

Exit with `sys.exit(1)` if token is empty, printing setup instructions.

### 2. `_preflight(args) -> None`

Steps in order:
1. Call `_resolve_token()` — exit if empty.
2. `from qiskit_ibm_runtime import QiskitRuntimeService` — exit with pip instruction if missing.
3. `load_dotenv()` if dotenv is available.
4. `QiskitRuntimeService(channel="ibm_quantum_platform", token=token)`.
5. `service.backends()` — get list of available backend names.
6. If `args.backend not in available_names` — print list and exit.
7. Print cost estimate:
   - VQC: `args.shots × args.max_iter` total shots (order of magnitude)
   - QSVC: one kernel matrix evaluation, not iterative
   - Trainable params (VQC): `args.qubits × args.ansatz_reps × 2`
8. If `args.dry_run`: `sys.exit(0)`.

### 3. `main() -> None`

```
a. argparse.parse_args()
b. _preflight(args)                              # exits on failure or --dry_run
c. load_hetionet_edges()
d. extract_task_edges(relation=args.relation, max_entities=args.max_entities)
e. train_test_split on positive edges (test_size=0.2, random_state=args.random_state)
f. get_hard_negatives(pos_train, strategy=args.negative_sampling, random_state=seed)
   get_hard_negatives(pos_test,  strategy=args.negative_sampling, random_state=seed+1)
g. pd.concat([pos_train, neg_train]) + shuffle → train_df
   pd.concat([pos_test,  neg_test])  + shuffle → test_df
h. HetionetEmbedder(embedding_dim=args.embedding_dim, qml_dim=args.qubits)
   embedder.load_saved_embeddings() or train_embeddings(task_edges)
i. X_train = embedder.prepare_link_features_qml(train_df)
   X_test  = embedder.prepare_link_features_qml(test_df)
j. Write temporary config/quantum_config_heron_run.yaml:
   - execution_mode: heron
   - heron.backend: args.backend
   - heron.shots: args.shots
k. QMLLinkPredictor(
       model_type=args.model_type,
       num_qubits=args.qubits,
       feature_map_type=args.feature_map,
       feature_map_reps=args.feature_map_reps,
       ansatz_reps=args.ansatz_reps,
       optimizer=args.optimizer,
       max_iter=args.max_iter,
       random_state=args.random_state,
       quantum_config_path=tmp_config,
   )
l. model.fit(X_train, y_train)
m. y_proba = model.predict_proba(X_test)[:, 1]
   pr_auc  = average_precision_score(y_test, y_proba)
   roc_auc = roc_auc_score(y_test, y_proba)
n. Print results table
o. stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
   payload = {backend, model_type, qubits, shots, pr_auc, roc_auc, ...}
   json.dump(payload, results/heron_{model_type}_{stamp}.json)
p. register_run(...) from scripts.benchmark_registry
   — set backend.execution_mode = "heron"
   — set circuit.n_qubits = args.qubits, feature_map = args.feature_map, etc.
q. Path(tmp_config).unlink()   # clean up temporary config
```

### 4. Guarding circuit depth on hardware

Add a warning if `feature_map_reps > 1` or `ansatz_reps > 3` and
`execution_mode == "heron"`:

```python
if args.feature_map_reps > 1:
    logger.warning(
        "feature_map_reps=%d may produce deep circuits on hardware. "
        "Consider --feature_map_reps 1 to reduce decoherence risk.",
        args.feature_map_reps,
    )
```

---

## Update `quantum_layer/train_on_heron.py`

The existing stub has a body but it:
- Uses the old `prepare_link_prediction_dataset()` which generates random negatives
- Does not call `register_run()`
- Does not write a temporary Heron config (it uses the global config directly)

Replace the body of `train_on_heron()` with a call to `scripts/train_on_heron.main()`
or update it to match the logic above. The simplest fix is to leave
`quantum_layer/train_on_heron.py` as a thin shim that delegates to the CLI script:

```python
# quantum_layer/train_on_heron.py
import subprocess, sys

def train_on_heron(**kwargs):
    """Thin shim — delegate to CLI script for full argument handling."""
    cmd = [sys.executable, "scripts/train_on_heron.py"]
    for k, v in kwargs.items():
        cmd += [f"--{k}", str(v)]
    return subprocess.run(cmd, check=True)
```

---

## Testing without hardware (dry run)

```bash
# Validate token and backend without submitting any jobs
IBM_Q_TOKEN=<token> python scripts/train_on_heron.py \
    --relation CtD --max_entities 100 \
    --qubits 4 --shots 100 --max_iter 5 \
    --dry_run

# Expected output:
# INFO: Connecting to IBM Quantum...
# INFO: Available backends (N): ibm_torino, ...
# INFO: ✅ Backend 'ibm_torino' reachable.
# INFO: Cost estimate: 100 shots × ~5 VQC iterations → O(500) total shots
# INFO: --dry_run: pre-flight passed. Exiting without submitting jobs.
```
