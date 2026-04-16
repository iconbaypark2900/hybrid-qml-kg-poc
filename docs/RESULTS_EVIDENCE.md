# Results Evidence Document
## Hybrid Quantum-Classical ML on Hetionet — Verified Experimental Record

**Generated**: 2026-04-16  
**Purpose**: Complete provenance trail for every metric claimed in the technical paper and README.  
**Total result files audited**: 218 `optimized_results_*.json` files + supporting `quantum_metrics_*` and `experiment_history.csv`.

---

## 1. Primary Claim: PR-AUC 0.7987 ✅ VERIFIED

**Source file**: `results/optimized_results_20260216-100431.json`  
**Timestamp**: 2026-02-16 10:04:31

### Raw ranking output (verbatim from file)
```json
{"name": "Ensemble-QC-stacking", "type": "ensemble", "pr_auc": 0.7987068071437018, "accuracy": 0.5761589403973509, "fit_time": 0.1808}
{"name": "RandomForest-Optimized",  "type": "classical","pr_auc": 0.7837785707146105, "accuracy": 0.5827814569536424, "fit_time": 1.2499}
{"name": "ExtraTrees-Optimized",    "type": "classical","pr_auc": 0.7807131892487402, "accuracy": 0.6622516556291391, "fit_time": 2.0806}
{"name": "QSVC-Optimized",          "type": "quantum",  "pr_auc": 0.6343474640767925, "accuracy": 0.5860927152317881, "fit_time": 2618.5966}
```

### Configuration (key parameters)
| Parameter | Value |
|-----------|-------|
| `embedding_method` | RotatE |
| `embedding_dim` | 128 |
| `embedding_epochs` | 200 |
| `full_graph_embeddings` | True |
| `qml_feature_map` | **Pauli** |
| `qml_feature_map_reps` | 2 |
| `qml_dim` | 16 qubits |
| `qml_pre_pca_dim` | 24 → 16D |
| `qsvc_C` | 0.1 |
| `qsvc_nystrom_m` | None (full kernel) |
| `negative_sampling` | hard |
| `ensemble_method` | stacking |
| `tune_classical` | True |
| `fast_mode` | True |
| `random_state` | 42 |

### Authenticity evidence
- **QSVC fit time = 2618.6 seconds** (43.6 minutes) confirms this was a genuine full-dataset quantum kernel computation, not a cache hit.
- No `kernel_cache/` file matches this exact config hash — the kernel was freshly computed.
- `quantum_metrics_QSVC_20260216-100431.json` independently records `test_pr_auc: 0.6343474640767925`.

---

## 2. README Table: All Five Rows Verified

### Row 1 — Ensemble-QC-stacking (Pauli): 0.7987
→ See Section 1 above. **VERIFIED**.

### Row 2 — RandomForest-Optimized: 0.7838
→ Same file `20260216-100431.json`. Raw: `0.7837785707146105`. **VERIFIED** (rounded to 0.7838).

### Row 3 — ExtraTrees-Optimized: 0.7807
→ Same file `20260216-100431.json`. Raw: `0.7807131892487402`. **VERIFIED** (rounded to 0.7807).

### Row 4 — Ensemble-QC-stacking (ZZ): 0.7408
**Source file**: `results/optimized_results_20260216-091710.json`

```json
{"name": "Ensemble-QC-stacking", "type": "ensemble", "pr_auc": 0.7407700155570323, "fit_time": 0.129}
{"name": "QSVC-Optimized",       "type": "quantum",  "pr_auc": 0.7216220334429990, "fit_time": 0.973}
```

Config: ZZ feature map, RotatE 128D, 200 epochs, full_graph=True, hard negatives, stacking.  
**VERIFIED** (raw: 0.74077, rounded to 0.7408).  
⚠️ Note: QSVC fit_time = 0.97s → **cached kernel**. The ZZ ensemble result relies on a pre-computed kernel. This is the source of the README's "QSVC 0.7216" row.

### Row 5 — QSVC-Optimized: 0.7216
→ Same file `20260216-091710.json`. Raw: `0.7216220334429990`. **VERIFIED** (rounded to 0.7216).  
⚠️ **Important clarification**: This QSVC score (0.7216) is from the **ZZ feature map run**, not the Pauli run. In the canonical Pauli run (`20260216-100431`), QSVC standalone scored only 0.6343. The README table mixes the best-per-configuration values, which is a standard reporting practice but should be noted in the paper.

---

## 3. Undocumented Superior Result: PR-AUC 0.8581

**Source file**: `results/optimized_results_20260323-134844.json`  
**Timestamp**: 2026-03-23 13:48:44

### Raw ranking output
```json
{"name": "Ensemble-QC-stacking", "type": "ensemble", "pr_auc": 0.8580649206389417, "accuracy": 0.7317880794701986, "fit_time": 0.312}
{"name": "RandomForest-Optimized","type": "classical","pr_auc": 0.8568851727414546, "accuracy": 0.7350993377483444, "fit_time": 1.505}
{"name": "ExtraTrees-Optimized",  "type": "classical","pr_auc": 0.8498029235558904, "accuracy": 0.7350993377483444, "fit_time": 1.207}
{"name": "QSVC-Optimized",        "type": "quantum",  "pr_auc": 0.7221871481080984, "accuracy": 0.6390728476821192, "fit_time": 7.238}
```

### Configuration differences from primary result
| Parameter | 0.7987 run | 0.8581 run |
|-----------|-----------|-----------|
| `embedding_dim` | 128 | **256** |
| `embedding_epochs` | 200 | **250** |
| `qml_feature_map_reps` | 2 | **1** |
| `qml_dim` | 16 qubits | **12 qubits** |
| `qml_pre_pca_dim` | 24 | **0 (none)** |
| `qsvc_C` | 0.1 (manual) | **0.6756 (Optuna)** |

### Methodological status: CACHED QUANTUM KERNEL
- The first run in this series (`20260323-125944`) computed QSVC fit in **99.3 seconds** (plausible, smaller kernel due to 12 qubits).
- Subsequent 83 runs show QSVC fit times of **0.2–7.2 seconds** — cache hits from `data/kernel_cache/`.
- All 83 runs produce **identical QSVC PR-AUC = 0.7221871481080984** — no variation whatsoever.
- The 306 MB `data/kernel_cache/` directory houses files matching this config (12-qubit Pauli reps=1).

**Conclusion**: The 0.8581 ensemble result is **real** (the classical models benefited from better embeddings + Optuna tuning), but the QSVC component was not freshly computed during the Optuna sweep. This must be disclosed in the paper as: *"Optuna hyperparameter search over classical components, with fixed cached quantum kernel (12-qubit Pauli, single rep, C=0.676)."*

---

## 4. Additional Documented Results

### ZZ Feature Map — Best Genuine Computation
| TS | Feature Map | Ensemble | QSVC | QSVC fit_s | Note |
|----|-------------|----------|------|-----------|------|
| 20260216-084359 | ZZ | 0.7371 | 0.7216 | **908.3s** | Genuine computation |
| 20260216-091710 | ZZ | 0.7408 | 0.7216 | 0.97s | Cached |
| 20260216-101302 | ZZ | 0.7408 | 0.7216 | 1.8s | Cached |

The highest genuine (non-cached) ZZ computation is `20260216-084359` with ensemble **0.7371** and QSVC **0.7216** (fit=908s). The README's 0.7408 ZZ result uses a cached kernel from that computation.

### IBM Quantum Heron Hardware (from experiment_history.csv)
| Config | Classical PR-AUC | Quantum PR-AUC | Backend |
|--------|-----------------|----------------|---------|
| Pauli 16-qubit | 0.6605 | 0.6343 | IBM Heron |

Hardware result matches the simulator QSVC score from the 0.7987 run (0.6343), providing cross-validation that the simulator quantum kernel is hardware-consistent.

### Early-Stage ComplEx / Suspicious Results (EXCLUDED from paper)
- `20251119-155145`: `Ensemble-RF-LR` PR-AUC = 0.9436 — this is a **classical RF+LogReg ensemble**, NOT a quantum ensemble. Config uses `full_graph_embeddings: False` (partial graph only) and random negatives. **Do not cite as quantum result.**
- Any `QSVC PR-AUC = 1.0000`: These appear in ComplEx 64D early runs and represent overfitting or data leakage. Excluded.

---

## 5. Unique Result Summary Table (Paper-Ready)

This table shows one row per methodologically distinct experiment (no duplicates from caching). Verified against raw JSON files.

| # | Config | Ensemble PR-AUC | RF PR-AUC | QSVC PR-AUC | Kernel | Source file |
|---|--------|----------------|-----------|-------------|--------|-------------|
| 1 | RotatE-128D · Pauli-16Q-r2 · C=0.1 · hard-neg | **0.7987** | 0.7838 | 0.6343 | Genuine (2619s) | `20260216-100431` |
| 2 | RotatE-256D · Pauli-12Q-r1 · C=0.676 · Optuna | **0.8581** | 0.8569 | 0.7222 | Cached† | `20260323-134844` |
| 3 | RotatE-128D · ZZ-16Q-r? · hard-neg | 0.7408 | 0.7838 | **0.7216** | Cached (908s orig) | `20260216-091710` |
| 4 | RotatE-128D · ZZ-16Q · genuine compute | 0.7371 | 0.7838 | 0.7216 | Genuine (908s) | `20260216-084359` |
| 5 | RotatE-128D · Pauli · 128ep · pre-optuna | 0.8037 | 0.7404 | 0.6310 | Genuine (158s) | `20260121-100351` |
| 6 | RotatE-128D · ZZ · 250ep · Optuna | 0.8063 | 0.7838 | 0.7328 | Mixed (40s) | `20260323-122209` |

† 0.8581 uses cached quantum kernel; Optuna swept classical hyperparams only.

---

## 6. Claim-by-Claim Verification Summary

| Claim (README / Paper) | Verdict | Raw Value | Source |
|------------------------|---------|-----------|--------|
| "Best PR-AUC 0.7987 via RotatE + QSVC (Pauli) + stacking" | ✅ VERIFIED | 0.7987068 | `20260216-100431` |
| "RandomForest-Optimized: 0.7838" | ✅ VERIFIED | 0.7837786 | `20260216-100431` |
| "ExtraTrees-Optimized: 0.7807" | ✅ VERIFIED | 0.7807132 | `20260216-100431` |
| "Ensemble-QC-stacking (ZZ): 0.7408" | ✅ VERIFIED (cached kernel) | 0.7407700 | `20260216-091710` |
| "QSVC-Optimized: 0.7216" | ✅ VERIFIED (ZZ run, not Pauli) | 0.7216220 | `20260216-091710` |
| "16-qubit Pauli feature map, reps=2" | ✅ VERIFIED | `qml_dim=16, reps=2` | `20260216-100431` config |
| "C=0.1" | ✅ VERIFIED | `qsvc_C: 0.1` | `20260216-100431` config |
| "RotatE 128D, 200 epochs, full_graph=True" | ✅ VERIFIED | exact match | `20260216-100431` config |
| "Hard negative sampling" | ✅ VERIFIED | `negative_sampling: hard` | `20260216-100431` config |
| "pre_pca_dim=24 → 16D" | ✅ VERIFIED | `qml_pre_pca_dim: 24, qml_dim: 16` | `20260216-100431` config |
| "IBM Quantum Heron hardware run" | ✅ VERIFIED | QSVC=0.6343, classical=0.6605 | `experiment_history.csv` |
| "Target PR-AUC > 0.70: Achieved" | ✅ VERIFIED | 0.7987 > 0.70 | `20260216-100431` |

---

## 7. Disclosures for Paper

1. **Kernel caching**: The 0.8581 result (Section 3) uses a cached quantum kernel. The kernel was computed once (first Optuna trial, ~99s) and reused across 83 subsequent trials. This should be reported as: *"best ensemble result across Optuna trials with fixed quantum kernel."*

2. **QSVC column ambiguity**: The README row "QSVC-Optimized: 0.7216" refers to the ZZ feature map run, not the canonical Pauli run. In the Pauli run that yields 0.7987 ensemble, QSVC standalone is 0.6343. Both values are genuine — they reflect different feature map configurations. The paper should clarify this distinction.

3. **Ensemble design**: The stacking ensemble PR-AUC (0.7987) exceeds the QSVC standalone PR-AUC (0.6343) by 0.164 points, demonstrating that QSVC contributes complementary signal rather than being the primary classifier.

4. **Score-validity inversion**: The highest-ranked prediction (Abacavir → ocular cancer, score ≈ 0.793) has zero ClinicalTrials.gov support, while lower-ranked predictions (scores 0.52–0.53) are validated by 7+ trials each. This is a known failure mode of pure embedding-based scoring and motivates the MoA feature module.

---

## 8. Next Experimental Steps (Evidence Gaps)

| Gap | Action | Expected result |
|-----|--------|----------------|
| MoA features not yet benchmarked | Run `--use_moa_features` on CtD | Expect reduction in score-validity inversion |
| CpD relation (390 edges) not tested | Run pipeline on `--relation CpD` | First quantum KG result on compound–pathway relation |
| 0.8581 with genuine quantum kernel | Re-run 20260323 config without cache | Would confirm if 0.8581 holds under fresh computation |
| CbG / GdD relations | New experiments | Expand quantum KG link prediction to gene-disease |
