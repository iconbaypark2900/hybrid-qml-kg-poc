# Paper v2 — Missing Experiments

**Status:** Not run  
**Blocks:** arXiv v2 submission  
**Source:** PAPER.md §9 Future Work, §8.5 Limitations

These are the experiments explicitly committed to in the paper's own text.
None have result files in `results/`. All are runnable with existing code.

---

## 1. MoA Feature Benchmark

**Why it matters:** The paper introduces a 10-feature mechanism-of-action module
in §4.5 and §7.3 as the fix for the score-validity inversion problem (Abacavir →
ocular cancer scores 0.793 with zero clinical support). The module is fully
implemented in `kg_layer/moa_features.py` and `kg_layer/enhanced_features.py`,
and the `--use_moa_features` flag exists in `scripts/run_optimized_pipeline.py`.
No PR-AUC result with this flag has ever been reported.

**What to run:**

```bash
# Baseline (already done, result in results/optimized_results_20260216-100431.json)
python scripts/run_optimized_pipeline.py \
  --relation CtD --full_graph_embeddings \
  --embedding_method RotatE --embedding_dim 128 --embedding_epochs 200 \
  --negative_sampling hard --qml_dim 16 --qml_feature_map Pauli \
  --qml_feature_map_reps 2 --qsvc_C 0.1 --qml_pre_pca_dim 24 \
  --run_ensemble --ensemble_method stacking --tune_classical --fast_mode

# With MoA features enabled (new run needed)
python scripts/run_optimized_pipeline.py \
  --relation CtD --full_graph_embeddings \
  --embedding_method RotatE --embedding_dim 128 --embedding_epochs 200 \
  --negative_sampling hard --qml_dim 16 --qml_feature_map Pauli \
  --qml_feature_map_reps 2 --qsvc_C 0.1 --qml_pre_pca_dim 24 \
  --run_ensemble --ensemble_method stacking --tune_classical --fast_mode \
  --use_moa_features
```

**What to record:**
- PR-AUC before vs after MoA features
- Change in rank of Abacavir → ocular cancer (expected: drops significantly)
- Change in rank of Losartan → atherosclerosis (expected: rises)
- Feature importance of MoA features vs embedding features

**Expected result location:** `results/optimized_results_<stamp>_moa.json`

---

## 2. CpD Relation Run

**Why it matters:** The paper claims the pipeline is designed for multi-relational
extension. CpD (Compound-palliates-Disease, 390 positive edges) is the natural
first extension requiring zero code changes.

**What to run:**

```bash
python scripts/run_optimized_pipeline.py \
  --relation CpD \
  --full_graph_embeddings \
  --embedding_method RotatE --embedding_dim 128 --embedding_epochs 200 \
  --negative_sampling hard --qml_dim 16 --qml_feature_map Pauli \
  --qml_feature_map_reps 2 --qsvc_C 0.1 --qml_pre_pca_dim 24 \
  --run_ensemble --ensemble_method stacking --tune_classical --fast_mode
```

**What to record:**
- PR-AUC for each model (CpD has fewer positives than CtD, expect lower absolute numbers)
- Whether the Pauli inversion effect (QSVC standalone < classical, ensemble > classical) holds on CpD
- Top novel CpD predictions not in training set

**Expected result location:** `results/optimized_results_<stamp>_cpd.json`

---

## 3. Multi-Seed Evaluation (5 seeds)

**Why it matters:** All reported results use `--random_state 42` exclusively.
A single seed run could be a lucky or unlucky split. No variance estimate
exists for any headline number in the paper.

**What to run:**

```bash
for seed in 42 7 13 99 2026; do
  python scripts/run_optimized_pipeline.py \
    --relation CtD --full_graph_embeddings \
    --embedding_method RotatE --embedding_dim 128 --embedding_epochs 200 \
    --negative_sampling hard --qml_dim 16 --qml_feature_map Pauli \
    --qml_feature_map_reps 2 --qsvc_C 0.1 --qml_pre_pca_dim 24 \
    --run_ensemble --ensemble_method stacking --tune_classical --fast_mode \
    --random_state $seed \
    --results_dir results/multiseed
done
```

**What to compute from output:**
- Mean ± std PR-AUC for each model across 5 seeds
- Update Table 3 in the paper with error bars
- Confirm 0.7987 is not an outlier seed

**Note on quantum computation time:** Each seed requires a fresh QSVC kernel
computation (~43 minutes on CPU). 5 seeds = ~3.5 hours. Use `--fast_mode` and
`--qsvc_nystrom_m 200` if time is a constraint; document if Nyström was used.

---

## 4. Fix LaTeX Citation Keys

**Why it matters:** The compiled `paper.pdf` currently shows `[?]` for every
in-text citation. The `\cite{himmelstein2017}` etc. in the body do not match
the `\bibitem` labels in the `thebibliography` block.

**What to do:**

Open `docs/paper.tex` and reconcile the cite keys. The body uses keys like:
- `himmelstein2017`
- `mayers2023`
- `qkdti2025`
- `kruger2023`
- `graphrag2025`
- `cim2026`
- `hybrid2024`
- `rotate2019`
- `pykeen2021`
- `schuld2021`
- `qiskit2023`

Find the `\begin{thebibliography}` block (line 578) and ensure each `\bibitem{key}`
matches exactly one `\cite{key}` in the body.

After fixing, run:
```bash
cd docs
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

---

## 5. Render and Commit the 3 Required Figures

**Why it matters:** `paper.tex` references three figures that must exist as PDFs
in the `figures/` directory. The directory is currently untracked (`??` in git status).
Appendix A of `PAPER.md` gives exact data and layout specs for all three.

**Figure 1 — Pipeline Architecture**  
File: `figures/fig1_pipeline.pdf`  
Layout: Two-column flowchart. Left: Hetionet → embeddings → pairs → features.
Right: classical path (blue) branching from quantum path (purple), merging at
stacking ensemble (green).

**Figure 2 — Pauli vs ZZ Tradeoff**  
File: `figures/fig2_pauli_zz.pdf`  
Data:
```
ZZ:    QSVC=0.7216, Ensemble=0.7408, RF_baseline=0.7838
Pauli: QSVC=0.6343, Ensemble=0.7987, RF_baseline=0.7838
```
Layout: Grouped bar chart with horizontal dashed baseline at 0.7838.

**Figure 3 — Score-Validity Scatter**  
File: `figures/fig3_clinical.pdf`  
Data: 6 predictions with scores and ClinicalTrials.gov trial counts.
Layout: Scatter with diagonal arrow annotated "score-validity inversion".

**Verify the renders match the spec in PAPER.md Appendix A exactly before committing.**

After generating, add to git:
```bash
git add figures/
git commit -m "add rendered paper figures (fig1, fig2, fig3)"
```

---

## 6. Add Missing Baselines to Table 3

**Why it matters:** Without a random baseline (PR-AUC ≈ 0.50 for balanced data)
and a degree-heuristic baseline, the paper cannot establish whether 0.7987 is
impressive or merely expected.

**Random baseline:** Always 0.50 for a perfectly balanced test set. Add as a row
in Table 3 labeled `Random (expected)` with PR-AUC = 0.50.

**Degree-heuristic baseline:** Rank all (compound, disease) test pairs by the
product of compound degree × disease degree in the training graph. Compute
PR-AUC of this ranking against test labels.

```python
# Compute degree-heuristic baseline PR-AUC
import pandas as pd, numpy as np
from sklearn.metrics import average_precision_score

# compound_degree[c] = number of training edges involving compound c
# disease_degree[d] = number of training edges involving disease d
scores = [compound_degree[c] * disease_degree[d] for c, d in test_pairs]
degree_pr_auc = average_precision_score(test_labels, scores)
print(f"Degree-heuristic PR-AUC: {degree_pr_auc:.4f}")
```

Add both rows to Table 3 in `PAPER.md` and `paper.tex`.
