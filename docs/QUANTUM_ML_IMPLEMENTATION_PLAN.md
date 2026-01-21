# 🚀 Quantum ML Implementation Plan
## Biomedical Link Prediction Project Roadmap

---

## 📋 Executive Summary

**Current State:**
- VQC underperforming (PR-AUC: 0.49 vs QSVC: 0.65)
- Classical baseline overfitting (Train: 0.86 → Test: 0.60)
- Dataset limited to 300 entities (proof-of-concept scale)
- No empirical validation of scalability claims

**Goal:** Close the 4-5% performance gap and establish quantum advantage

**Timeline:** 8-12 weeks

**Success Criteria:**
- VQC PR-AUC > 0.65 (match or exceed QSVC)
- Classical test PR-AUC > 0.70 (reduce overfitting)
- Empirical scaling validation (N > 2000 entities)
- Statistical significance (p < 0.05) for quantum advantage

---

## 🎯 Quick Wins (Week 1-2)

### Priority Actions
1. **VQC Loss Tracking** (2 hours)
   - Add loss callback to training loop
   - Plot loss curves to diagnose convergence
   - Files: `quantum_layer/qml_trainer.py`

2. **Classical Regularization Sweep** (4 hours)
   - Grid search over C values: [0.001, 0.01, 0.1, 1, 10, 100]
   - Plot regularization path
   - Files: `classical_baseline/train_baseline.py`

3. **Embedding Quality Check** (3 hours)
   - Compute drug-drug similarity within vs between classes
   - t-SNE visualization colored by entity type
   - Files: `kg_layer/kg_embedder.py`

4. **Multi-seed Validation** (6 hours)
   - Run pipeline with 10 different random seeds
   - Compute mean ± std for all metrics
   - Statistical significance test (Wilcoxon)
   - Files: `scripts/multi_seed_experiment.py` (create new)

**Deliverable:** Initial diagnostic report with loss curves, regularization plots, and statistical validation

---

## 📅 Phase 1: Performance Optimization (Week 3-4)

### 🎯 Goal: VQC PR-AUC > 0.60

#### Task 1.1: Optimizer Comparison
**Duration:** 3 days  
**Effort:** 8 hours

**Actions:**
- [ ] Implement optimizer comparison script
- [ ] Test: COBYLA, SPSA, NFT, GradientDescent
- [ ] Track convergence speed and final loss
- [ ] Document results in `experiments/optimizer_comparison.md`

**Code:**
```python
# scripts/compare_optimizers.py
optimizers = {
    'COBYLA': COBYLA(maxiter=200),
    'SPSA': SPSA(maxiter=200, learning_rate=0.01),
    'NFT': NFT(maxiter=200),
    'ADAM': GradientDescent(maxiter=200, learning_rate=0.01)
}

for name, opt in optimizers.items():
    vqc = VQC(feature_map=fm, ansatz=ans, optimizer=opt)
    losses = []
    vqc.fit(X_train, y_train, callback=track_loss)
    score = vqc.score(X_test, y_test)
    log_results(name, losses, score)
```

**Success Metric:** Find optimizer with 5%+ improvement over COBYLA

---

#### Task 1.2: Ansatz Architecture Search
**Duration:** 4 days  
**Effort:** 12 hours

**Actions:**
- [ ] Compare ansatzes: RealAmplitudes, EfficientSU2, TwoLocal, Custom
- [ ] Ablation study on reps: [1, 2, 3, 4, 5]
- [ ] Count parameters and circuit depth
- [ ] Create architecture comparison table

**Code:**
```python
# scripts/ansatz_search.py
ansatzes = {
    'RealAmplitudes': RealAmplitudes(5, reps=3),
    'EfficientSU2': EfficientSU2(5, reps=3),
    'TwoLocal': TwoLocal(5, rotation_blocks='ry', 
                         entanglement_blocks='cz', reps=3)
}

results = []
for name, ansatz in ansatzes.items():
    metrics = {
        'name': name,
        'params': ansatz.num_parameters,
        'depth': ansatz.depth(),
        'pr_auc': train_and_evaluate(ansatz)
    }
    results.append(metrics)
```

**Success Metric:** Identify ansatz with best PR-AUC (target > 0.60)

---

#### Task 1.3: Feature Engineering
**Duration:** 3 days  
**Effort:** 10 hours

**Actions:**
- [ ] Test feature combinations: diff, product, concat, polynomial
- [ ] Compare normalizations: l2, minmax, zscore, tanh
- [ ] Implement best-performing strategy
- [ ] Update `kg_embedder.py` with optimal encoding

**Code:**
```python
# kg_layer/feature_engineering.py
encodings = {
    'diff_only': lambda h, t: np.abs(h - t),
    'product_only': lambda h, t: h * t,
    'diff_prod': lambda h, t: np.concatenate([np.abs(h-t), h*t])[:5],
    'poly': lambda h, t: polynomial_features(h, t, degree=2)[:5]
}

# Test each encoding
for name, encoder in encodings.items():
    X_encoded = prepare_features(encoder)
    score = evaluate_model(X_encoded)
    print(f"{name}: {score:.4f}")
```

**Success Metric:** Feature encoding improves VQC by 3%+ over baseline

---

#### Task 1.4: Hyperparameter Tuning
**Duration:** 5 days  
**Effort:** 15 hours

**Actions:**
- [ ] Define parameter grid (ansatz_reps, feature_map_reps, optimizer, max_iter)
- [ ] Run grid search with 5-fold CV
- [ ] Analyze best parameters
- [ ] Document optimal configuration

**Code:**
```python
# scripts/hyperparameter_search.py
from sklearn.model_selection import ParameterGrid

param_grid = {
    'ansatz_reps': [2, 3, 4],
    'feature_map_reps': [1, 2, 3],
    'optimizer': ['COBYLA', 'SPSA'],
    'max_iter': [50, 100, 200]
}

best_score = 0
for params in ParameterGrid(param_grid):
    vqc = create_vqc(**params)
    score = cross_val_score(vqc, X, y, cv=5).mean()
    if score > best_score:
        best_score = score
        best_params = params
```

**Success Metric:** Optimized VQC achieves PR-AUC > 0.65

---

### Phase 1 Checkpoint

**Date:** End of Week 4  
**Review Criteria:**
- ✅ VQC PR-AUC ≥ 0.65
- ✅ Documented optimization journey
- ✅ Code committed with performance improvements

**Go/No-Go Decision:**
- **GO:** Proceed to Phase 2 (Classical optimization)
- **NO-GO:** Deep dive into Issue 1 pathways A-D, extend Phase 1 by 1 week

---

## 📅 Phase 2: Classical Baseline Improvement (Week 5-6)

### 🎯 Goal: Classical test PR-AUC > 0.70, reduce overfitting gap < 10%

#### Task 2.1: Regularization Analysis
**Duration:** 2 days  
**Effort:** 6 hours

**Actions:**
- [ ] Regularization path: C = [10^-4 to 10^4]
- [ ] Plot train vs test scores
- [ ] Identify optimal C value
- [ ] Compare L1 vs L2 penalty

**Success Metric:** Test PR-AUC > 0.70, train-test gap < 15%

---

#### Task 2.2: Cross-Validation Implementation
**Duration:** 3 days  
**Effort:** 10 hours

**Actions:**
- [ ] Implement nested CV (outer=5, inner=3)
- [ ] Run with optimal hyperparameters
- [ ] Compute 95% confidence intervals
- [ ] Document unbiased performance estimate

**Code:**
```python
# scripts/nested_cv.py
def nested_cv(X, y, model, param_grid):
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True)
    
    outer_scores = []
    for train_idx, test_idx in outer_cv.split(X, y):
        # Inner loop: hyperparameter tuning
        grid = GridSearchCV(model, param_grid, cv=inner_cv)
        grid.fit(X[train_idx], y[train_idx])
        
        # Outer loop: evaluation
        score = grid.score(X[test_idx], y[test_idx])
        outer_scores.append(score)
    
    return np.mean(outer_scores), np.std(outer_scores)
```

**Success Metric:** Robust estimate with narrow confidence interval (CI width < 0.10)

---

#### Task 2.3: Model Complexity Analysis
**Duration:** 3 days  
**Effort:** 8 hours

**Actions:**
- [ ] Generate learning curves
- [ ] Test alternative models: SVM-RBF, RandomForest, GradientBoosting, MLP
- [ ] Compare complexity vs performance
- [ ] Document best model choice

**Success Metric:** Identify if logistic regression is optimal or if non-linear model improves by 5%+

---

### Phase 2 Checkpoint

**Date:** End of Week 6  
**Review Criteria:**
- ✅ Classical PR-AUC > 0.70
- ✅ Overfitting gap < 10%
- ✅ Statistical validation complete

---

## 📅 Phase 3: Data Quality & Scale (Week 7-8)

### 🎯 Goal: Scale to 2000+ entities, improve embedding quality

#### Task 3.1: Embedding Quality Validation
**Duration:** 3 days  
**Effort:** 10 hours

**Actions:**
- [ ] Drug-drug similarity analysis (within-class vs between-class)
- [ ] Disease-disease similarity validation
- [ ] t-SNE visualization by entity type
- [ ] Correlation: embedding similarity vs true labels

**Success Metric:** Within-class similarity > between-class by 0.2+, correlation > 0.4

---

#### Task 3.2: Embedding Algorithm Comparison
**Duration:** 4 days  
**Effort:** 12 hours

**Actions:**
- [ ] Compare: TransE, DistMult, ComplEx, RotatE, TuckER
- [ ] Evaluate each on link prediction task
- [ ] Choose best-performing algorithm
- [ ] Re-train with optimal embeddings

**Code:**
```python
# scripts/embedding_comparison.py
from pykeen.pipeline import pipeline

algorithms = ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'TuckER']

for algo in algorithms:
    result = pipeline(
        dataset=hetionet_dataset,
        model=algo,
        embedding_dim=32,
        training_kwargs=dict(num_epochs=100)
    )
    
    hits_at_10 = result.metric_results['hits@10']
    downstream_score = evaluate_link_prediction(result.model)
    
    log_results(algo, hits_at_10, downstream_score)
```

**Success Metric:** Embedding choice improves downstream PR-AUC by 5%+

---

#### Task 3.3: Hard Negative Mining
**Duration:** 3 days  
**Effort:** 8 hours

**Actions:**
- [ ] Implement similarity-based hard negatives
- [ ] Implement adversarial hard negatives (high-confidence errors)
- [ ] Train model on hard negatives
- [ ] Evaluate on challenging test set

**Success Metric:** Model trained on hard negatives generalizes better (test PR-AUC +3%)

---

#### Task 3.4: Dataset Scaling
**Duration:** 5 days  
**Effort:** 15 hours

**Actions:**
- [ ] Scale to N = [300, 500, 1000, 2000, 5000]
- [ ] Track PR-AUC, training time, parameters
- [ ] Generate scaling curves
- [ ] Statistical power analysis at each scale

**Success Metric:** Achieve stable performance at N=2000+, variance < 0.05

---

### Phase 3 Checkpoint

**Date:** End of Week 8  
**Review Criteria:**
- ✅ Embeddings validated (similarity, clustering)
- ✅ Dataset scaled to 2000+ entities
- ✅ Hard negatives improve robustness

---

## 📅 Phase 4: Scalability Validation (Week 9-10)

### 🎯 Goal: Empirically validate O(N²) vs O(log N) claims

#### Task 4.1: Runtime Benchmarking
**Duration:** 4 days  
**Effort:** 12 hours

**Actions:**
- [ ] Measure runtimes for N = [100, 300, 500, 1000, 2000, 5000]
- [ ] Track: embedding training, model training, inference
- [ ] Fit empirical curves (quadratic vs logarithmic)
- [ ] Extrapolate to N=100K

**Code:**
```python
# benchmarking/empirical_scaling.py
import time

entity_counts = [100, 300, 500, 1000, 2000, 5000]
runtimes = {'classical': [], 'quantum': []}

for N in entity_counts:
    # Classical timing
    start = time.time()
    classical_model = train_classical(N)
    runtimes['classical'].append(time.time() - start)
    
    # Quantum timing
    start = time.time()
    quantum_model = train_quantum(N)
    runtimes['quantum'].append(time.time() - start)

# Fit curves
from scipy.optimize import curve_fit
popt_classical = curve_fit(lambda N, a, b: a*N**2 + b, 
                           entity_counts, runtimes['classical'])
popt_quantum = curve_fit(lambda N, a, b: a*np.log(N) + b, 
                         entity_counts, runtimes['quantum'])
```

**Success Metric:** Identify crossover point where quantum becomes faster

---

#### Task 4.2: Complexity Profiling
**Duration:** 2 days  
**Effort:** 6 hours

**Actions:**
- [ ] Profile code with cProfile
- [ ] Identify bottlenecks (top 20 functions)
- [ ] Measure memory usage
- [ ] Document complexity analysis

**Success Metric:** Understand where time/memory is spent, identify optimization opportunities

---

#### Task 4.3: Statistical Significance Testing
**Duration:** 3 days  
**Effort:** 8 hours

**Actions:**
- [ ] Bootstrap confidence intervals (n=1000 samples)
- [ ] Paired statistical tests (Wilcoxon, t-test)
- [ ] Effect size calculation (Cohen's d)
- [ ] Power analysis for sample size estimation

**Success Metric:** p < 0.05 for quantum vs classical difference, medium-large effect size

---

### Phase 4 Checkpoint

**Date:** End of Week 10  
**Review Criteria:**
- ✅ Empirical scaling curves validated
- ✅ Statistical significance established
- ✅ Complexity analysis documented

---

## 📅 Phase 5: Hardware Validation & Documentation (Week 11-12)

### 🎯 Goal: Real hardware comparison, complete documentation

#### Task 5.1: IBM Quantum Execution
**Duration:** 4 days  
**Effort:** 12 hours

**Actions:**
- [ ] Run on IBM Brisbane/Torino
- [ ] Compare: simulator vs real hardware
- [ ] Test error mitigation levels [0, 1, 2, 3]
- [ ] Cost-benefit analysis

**Success Metric:** Understand noise impact, determine if hardware is practical

---

#### Task 5.2: Documentation Creation
**Duration:** 5 days  
**Effort:** 20 hours

**Actions:**
- [ ] Create `THEORY.md` (mathematical foundations)
- [ ] Create `ARCHITECTURE.md` (system design)
- [ ] Create `TUTORIAL.md` (step-by-step guide)
- [ ] Create `WHEN_TO_USE_QUANTUM.md` (decision framework)
- [ ] Update README with complete setup instructions

**Deliverable:** Professional documentation suite

---

#### Task 5.3: Final Report
**Duration:** 3 days  
**Effort:** 10 hours

**Actions:**
- [ ] Compile all experimental results
- [ ] Create comparison tables and figures
- [ ] Write executive summary
- [ ] Recommendations for production deployment

**Deliverable:** Publication-ready report

---

## 📊 Success Metrics Summary

| Metric | Baseline | Target | Stretch Goal |
|--------|----------|--------|--------------|
| VQC PR-AUC | 0.49 | 0.65 | 0.70 |
| QSVC PR-AUC | 0.65 | 0.65 | 0.70 |
| Classical test PR-AUC | 0.60 | 0.70 | 0.75 |
| Overfitting gap | 26% | <10% | <5% |
| Dataset size | 300 | 2000 | 5000 |
| Statistical significance | N/A | p<0.05 | p<0.01 |
| Crossover point | Unknown | <10K | <5K |

---

## 🔗 Dependencies & Prerequisites

### Critical Path
```
Week 1-2: Quick Wins
   ↓
Week 3-4: Phase 1 (VQC Optimization)
   ↓
Week 5-6: Phase 2 (Classical Baseline)
   ↓
Week 7-8: Phase 3 (Data & Scale)
   ↓
Week 9-10: Phase 4 (Scalability)
   ↓
Week 11-12: Phase 5 (Hardware & Docs)
```

### Parallel Tracks
- **Track A:** VQC optimization (Weeks 3-4) + Classical optimization (Weeks 5-6)
- **Track B:** Embedding quality (Week 7) + Hard negatives (Week 7-8)
- **Track C:** Scaling experiments (Week 8-10) can run in parallel with other tasks

### External Dependencies
- IBM Quantum access (needed for Phase 5)
- PyKEEN library (needed for embedding comparison)
- Computational resources (GPU for training, quantum credits)

---

## 🚨 Risk Management

### High-Risk Items
1. **VQC not improving beyond 0.60**
   - Mitigation: Have backup quantum kernel methods (QSVC already works)
   - Fallback: Focus on hybrid quantum-classical approach

2. **Quantum hardware too noisy**
   - Mitigation: Document simulator results, propose future work
   - Fallback: Emphasis on algorithmic development for near-term

3. **Scaling experiments too expensive**
   - Mitigation: Use sampling strategies, distributed computing
   - Fallback: Theoretical analysis + limited empirical validation

4. **Dataset quality issues**
   - Mitigation: Explore alternative data sources (DrugBank, STITCH)
   - Fallback: Document data limitations, propose future datasets

---

## 📝 Weekly Deliverables

### Week 1-2
- [ ] Diagnostic report with loss curves
- [ ] Statistical validation (10 seeds)
- [ ] Initial embedding quality metrics

### Week 3-4 (Phase 1)
- [ ] Optimizer comparison results
- [ ] Ansatz architecture search findings
- [ ] Optimized VQC configuration

### Week 5-6 (Phase 2)
- [ ] Regularization analysis
- [ ] Nested CV results
- [ ] Classical baseline improvement

### Week 7-8 (Phase 3)
- [ ] Embedding validation report
- [ ] Hard negative strategy evaluation
- [ ] Scaled dataset (2000+ entities)

### Week 9-10 (Phase 4)
- [ ] Empirical scaling curves
- [ ] Statistical significance tests
- [ ] Complexity profiling report

### Week 11-12 (Phase 5)
- [ ] Hardware comparison results
- [ ] Complete documentation suite
- [ ] Final technical report

---

## 🎓 Learning Resources

### Essential Reading
1. **VQC Optimization:**
   - "Barren Plateaus in Quantum Neural Network Training" (McClean et al., 2018)
   - Qiskit VQC Tutorial

2. **Knowledge Graph Embeddings:**
   - "TransE: Translating Embeddings for Multi-relational Data" (Bordes et al., 2013)
   - PyKEEN Documentation

3. **Statistical Validation:**
   - "Statistical comparisons of classifiers over multiple data sets" (Demšar, 2006)
   - "Nested cross-validation when selecting classifiers" (Varma & Simon, 2006)

4. **Quantum Machine Learning:**
   - "The power of quantum neural networks" (Abbas et al., 2021)
   - "Quantum embeddings for machine learning" (Lloyd et al., 2020)

---

## 🔧 Implementation Checklist

### Setup (Before Week 1)
- [ ] IBM Quantum account activated
- [ ] Python environment configured (qiskit, pykeen, scikit-learn)
- [ ] Git repository initialized
- [ ] Experiment tracking setup (Weights & Biases or MLflow)
- [ ] Baseline code running successfully

### Code Organization
- [ ] Create `experiments/` directory for results
- [ ] Create `scripts/` directory for standalone scripts
- [ ] Create `benchmarking/` directory for performance tests
- [ ] Create `docs/` directory for documentation
- [ ] Update `.gitignore` for large files

### Version Control
- [ ] Branch strategy: `main`, `develop`, feature branches
- [ ] Commit after each completed task
- [ ] Tag releases at each phase checkpoint

---

## 📈 Progress Tracking

Use this table to track progress:

| Task | Status | Start Date | End Date | Notes |
|------|--------|------------|----------|-------|
| Quick Wins | ⬜ Not Started | | | |
| Phase 1: VQC Optimization | ⬜ Not Started | | | |
| Phase 2: Classical Baseline | ⬜ Not Started | | | |
| Phase 3: Data & Scale | ⬜ Not Started | | | |
| Phase 4: Scalability | ⬜ Not Started | | | |
| Phase 5: Hardware & Docs | ⬜ Not Started | | | |

**Status Legend:**
- ⬜ Not Started
- 🟡 In Progress
- ✅ Complete
- ⚠️ Blocked
- ❌ Cancelled

---

## 🎯 Final Checklist (End of Week 12)

- [ ] VQC PR-AUC ≥ 0.65
- [ ] Classical test PR-AUC ≥ 0.70
- [ ] Overfitting gap < 10%
- [ ] Dataset scaled to 2000+ entities
- [ ] Statistical significance established (p < 0.05)
- [ ] Empirical scaling curves generated
- [ ] Hardware comparison complete
- [ ] Complete documentation suite
- [ ] Final technical report
- [ ] Code clean, tested, and documented
- [ ] Results reproducible

---

## 🚀 Next Steps After Completion

1. **Publication:** Prepare paper for quantum computing conference (e.g., QCE, QTML)
2. **Production:** Deploy best model as API endpoint
3. **Expansion:** Scale to full Hetionet (47K entities)
4. **Collaboration:** Engage with quantum hardware providers for optimization
5. **Open Source:** Release code and documentation to community

---

**Document Version:** 1.0  
**Last Updated:** {{current_date}}  
**Author:** Implementation Team  
**Review Cycle:** Weekly during team meetings
