# QA Test Checklist: Hybrid QML-KG System

**Project**: Quantum-Classical Biomedical Link Prediction (Hetionet CtD)  
**Frontend**: Next.js 16 + React 19 + TypeScript  
**Backend**: Python (Quantum kernels, Classical ensemble, RotatE embeddings)  
**Target PR-AUC**: > 0.70  
**Created**: 2026-04-22

---

## Phase 1: Backend Core Functionality (Python Tests)

### 1.1 Data Pipeline & Embeddings ✓
- [ ] **Hetionet Data Loading**
  - [ ] Successfully loads 47,031 entities
  - [ ] Successfully loads 2.25M relationships
  - [ ] Compound-treats-Disease (CtD) subset correctly filtered
  - [ ] No data corruption or missing values

- [ ] **Embedding Generation (RotatE)**
  - [ ] RotatE model trains without errors
  - [ ] Embeddings generated at 128D
  - [ ] Training converges after 200 epochs
  - [ ] Embedding quality passes sanity checks (norm, variance)

- [ ] **Feature Construction**
  - [ ] Pairwise features created (concat, diff, Hadamard)
  - [ ] Feature dimensionality correct
  - [ ] No NaN or infinite values in features
  - [ ] Hard negative sampling works correctly

### 1.2 Quantum Improvements ✓
- [ ] **Quantum-Enhanced Embeddings** (test_quantum_enhanced_embeddings)
  - [ ] QuantumEnhancedEmbeddingOptimizer loads without errors
  - [ ] QuantumKernelAlignmentEmbedding initializes
  - [ ] Enhanced embeddings match expected shapes
  - [ ] Performance improvement > baseline

- [ ] **Quantum Transfer Learning** (test_quantum_transfer_learning)
  - [ ] Pre-trained quantum models load
  - [ ] Transfer learning improves convergence
  - [ ] Fine-tuning on CtD task works
  - [ ] Metrics show improvement over baseline

- [ ] **Quantum Error Mitigation** (test_quantum_error_mitigation)
  - [ ] Zero-noise extrapolation (ZNE) reduces errors
  - [ ] Readout error mitigation applies correctly
  - [ ] Mitigated results more stable than raw
  - [ ] No performance degradation from mitigation

- [ ] **Quantum Circuit Optimization** (test_quantum_circuit_optimization)
  - [ ] Circuit depth reduced by optimization
  - [ ] Gate count decreased
  - [ ] Optimized circuits produce valid outputs
  - [ ] Latency improvement measurable

- [ ] **Quantum Kernel Engineering** (test_quantum_kernel_engineering)
  - [ ] QSVC kernel computes without errors
  - [ ] Kernel matrix positive semi-definite
  - [ ] Kernel alignment > threshold
  - [ ] QSVC achieves PR-AUC ≥ 0.72

- [ ] **Quantum Variational Feature Selection** (test_quantum_variational_feature_selection)
  - [ ] VQC initializes with correct gates
  - [ ] Feature importance scores computed
  - [ ] Selected features reduce overfitting
  - [ ] Model converges on subset

### 1.3 Classical Models ✓
- [ ] **Random Forest Optimization**
  - [ ] GridSearchCV completes successfully
  - [ ] Optimal hyperparameters found
  - [ ] PR-AUC ≥ 0.7838 (baseline: 0.7838)
  - [ ] Cross-validation scores stable

- [ ] **Extra Trees Optimization**
  - [ ] ExtraTrees trains without errors
  - [ ] PR-AUC ≥ 0.7807 (baseline: 0.7807)
  - [ ] Feature importances computed
  - [ ] Out-of-bag estimates reasonable

- [ ] **Logistic Regression Baseline**
  - [ ] Logistic Regression trains quickly
  - [ ] Regularization prevents overfitting
  - [ ] Predictions in [0, 1] range
  - [ ] Interpretable coefficients

### 1.4 Ensemble & Stacking ✓
- [ ] **Stacking Ensemble (Pauli)**
  - [ ] Base learners train successfully
  - [ ] Meta-learner trained on cross-validated predictions
  - [ ] **PR-AUC = 0.7987** (target achieved ✓)
  - [ ] No data leakage in stacking process

- [ ] **Stacking Ensemble (ZZ)**
  - [ ] ZZ feature map trains
  - [ ] Pauli encoding alternative works
  - [ ] PR-AUC ≥ 0.7408 (acceptable alternative)
  - [ ] Produces valid predictions

- [ ] **GridSearchCV Integration**
  - [ ] Hyperparameter tuning completes
  - [ ] Best model selected
  - [ ] Cross-validation folds balanced
  - [ ] Results reproducible with random state

### 1.5 Metrics & Evaluation ✓
- [ ] **Discovery Metrics** (test_discovery_metrics.py)
  - [ ] `top_k_hit_rate(k=10)` computes correctly
  - [ ] `mean_rank_of_positives` ranks matches correctly
  - [ ] Handles edge cases (no positives, all zeros)
  - [ ] Metrics included in final reports

- [ ] **PR-AUC (Primary Metric)**
  - [ ] Calculated correctly from predictions
  - [ ] Threshold-independent metric
  - [ ] Best model achieves **0.7987** ✓
  - [ ] Surpasses target > 0.70 ✓

- [ ] **ROC-AUC (Secondary)**
  - [ ] Computed alongside PR-AUC
  - [ ] Reasonable correlation with PR-AUC
  - [ ] Balanced threshold selection

- [ ] **Precision, Recall, F1**
  - [ ] Threshold-dependent metrics computed
  - [ ] F1 optimized for balance
  - [ ] Precision-recall trade-off analyzed

---

## Phase 2: Frontend (Next.js) Functionality

### 2.1 Navigation & Routing ✓
- [ ] **Main Dashboard** (`/`)
  - [ ] Page loads without errors
  - [ ] Hero section renders correctly
  - [ ] Title: "Does this compound treat this disease?"
  - [ ] Quick-nav pills visible and clickable
  - [ ] Links navigate to correct routes

- [ ] **Predict Page** (`/predict`)
  - [ ] Page loads and renders layout
  - [ ] PredictForm component loads
  - [ ] Score interpretation sidebar visible
  - [ ] Instructions clear (systematic names, DrugBank IDs)

- [ ] **Experiments** (`/experiments`)
  - [ ] Experiment results table loads
  - [ ] Model comparison visible
  - [ ] PR-AUC scores displayed correctly
  - [ ] Results sortable and filterable

- [ ] **Quantum Page** (`/quantum`)
  - [ ] Quantum model details render
  - [ ] Feature map explanations clear
  - [ ] Circuit diagrams display (if present)
  - [ ] Hyperparameters shown

- [ ] **Knowledge Graph** (`/knowledge-graph`)
  - [ ] Graph visualization renders
  - [ ] Node/edge data loads
  - [ ] Interactive exploration works
  - [ ] Search functionality (if implemented)

- [ ] **Visualization** (`/visualization`)
  - [ ] Charts and plots render without errors
  - [ ] D3 visualizations load
  - [ ] Three.js 3D rendering works (if used)
  - [ ] Responsive to window resize

- [ ] **Molecular Design** (`/molecular-design`)
  - [ ] Design interface renders
  - [ ] Controls functional
  - [ ] Output display works

- [ ] **Analysis Pages** (`/analysis/drug-delivery`, `/analysis/next-steps`)
  - [ ] Load without errors
  - [ ] Content displays properly
  - [ ] Charts/tables render

- [ ] **Simulation Pages** (`/simulation`, `/simulation/parameters`)
  - [ ] Parameter input form works
  - [ ] Validation prevents invalid inputs
  - [ ] Submission successful

### 2.2 Core Predict Form ✓
- [ ] **Form Inputs**
  - [ ] Drug name/ID input accepts text
  - [ ] Disease name/ID input accepts text
  - [ ] Input field labels visible
  - [ ] Placeholders helpful (e.g., "pindolol", "DB00960")

- [ ] **Form Validation**
  - [ ] Required fields enforced
  - [ ] Error messages clear and actionable
  - [ ] Prevents submission with invalid data
  - [ ] Real-time validation feedback (if implemented)

- [ ] **Form Submission**
  - [ ] Submit button clickable
  - [ ] Loading state shown during submission
  - [ ] API call made to backend
  - [ ] Response received and parsed

- [ ] **Prediction Results Display**
  - [ ] Score displayed as percentage
  - [ ] Score in [0, 100] range
  - [ ] Color coding matches interpretation guide
  - [ ] **≥70%**: strong evidence (green)
  - [ ] **40-69%**: moderate evidence (yellow)
  - [ ] **<40%**: weak evidence (gray)

- [ ] **Score Interpretation Guide**
  - [ ] Sidebar visible and readable
  - [ ] Thresholds match documentation (70%, 40%)
  - [ ] Clinical relevance explanation present
  - [ ] Caveat visible (research tool, not clinical)

### 2.3 Visual Design & UX ✓
- [ ] **Typography**
  - [ ] Headings use Material Design 3 headline font
  - [ ] Body text readable at 1rem
  - [ ] Code blocks use monospace font
  - [ ] Font weights consistent (semibold, medium, regular)

- [ ] **Color & Contrast**
  - [ ] Text meets WCAG AA contrast ratio (4.5:1 for body)
  - [ ] Color scheme follows Material Design 3
  - [ ] Primary color used consistently
  - [ ] Error states visible (red/tertiary)
  - [ ] Success states visible (green/tertiary)

- [ ] **Spacing & Layout**
  - [ ] Sections have consistent spacing (space-y-8, space-y-4)
  - [ ] Grid layout responsive (lg:grid-cols-[1fr_320px])
  - [ ] Padding consistent (p-5, p-4)
  - [ ] Border radius applied (rounded-xl, rounded-full)

- [ ] **Interactive States**
  - [ ] Buttons show hover state (bg-surface-container)
  - [ ] Links show underline or color change
  - [ ] Form inputs show focus ring
  - [ ] Disabled state visually distinct

### 2.4 Responsiveness ✓
- [ ] **Mobile (375px)**
  - [ ] Content stacks vertically
  - [ ] Text readable at mobile sizes
  - [ ] Buttons/inputs large enough to tap (48px)
  - [ ] No horizontal scrolling

- [ ] **Tablet (768px)**
  - [ ] Grid layout adjusts appropriately
  - [ ] Sidebar may float beside content
  - [ ] Charts/graphs scale well

- [ ] **Desktop (1280px+)**
  - [ ] Full layout with sidebar
  - [ ] Multi-column grids display
  - [ ] Visualization detail visible

### 2.5 Charts & Visualizations ✓
- [ ] **Chart.js Charts**
  - [ ] Bar charts render correctly
  - [ ] Line charts show trends
  - [ ] Legend visible and accurate
  - [ ] Axes labeled

- [ ] **D3 Visualizations**
  - [ ] SVG elements render
  - [ ] Interactions (hover, click) work
  - [ ] Tooltips appear on hover
  - [ ] Zoom/pan functional (if implemented)

- [ ] **Three.js 3D (if used)**
  - [ ] Scene renders without WebGL errors
  - [ ] Lighting visible
  - [ ] Controls responsive (mouse, touch)
  - [ ] Performance acceptable (60 FPS target)

---

## Phase 3: Integration Testing (Backend ↔ Frontend)

### 3.1 API Communication ✓
- [ ] **Predict Endpoint**
  - [ ] Frontend sends valid request format
  - [ ] Backend receives and processes correctly
  - [ ] Response returned in <2 seconds
  - [ ] JSON serialization works both ways
  - [ ] Error responses have proper HTTP status

- [ ] **Error Handling**
  - [ ] Unknown drug returns 404 or error message
  - [ ] Unknown disease returns 404 or error message
  - [ ] Network timeout shown gracefully
  - [ ] Server error shown with retry option
  - [ ] Frontend shows helpful error message (not JSON)

- [ ] **Data Format**
  - [ ] Request body matches backend expectations
  - [ ] Response includes: score, model_name, timestamp
  - [ ] All fields have correct types
  - [ ] No unexpected fields that break parsing

### 3.2 End-to-End Workflows ✓

#### Journey 1: Quick Prediction
- [ ] User lands on home page
- [ ] Clicks "Drug-disease link prediction" link or uses inline form
- [ ] Enters known drug (e.g., "ibuprofen")
- [ ] Enters known disease (e.g., "headache")
- [ ] Clicks "Predict"
- [ ] Score appears with confidence level
- [ ] Color-coded interpretation matches score
- [ ] User can make another prediction immediately

#### Journey 2: Review Past Results
- [ ] User navigates to `/experiments`
- [ ] Experiment table loads with results
- [ ] Best model (Ensemble-QC-stacking Pauli: 0.7987) is highlighted
- [ ] Can sort by PR-AUC descending
- [ ] Can see model variants (Pauli vs ZZ)
- [ ] Detailed metrics expandable (if designed)

#### Journey 3: Explore Quantum Details
- [ ] User navigates to `/quantum`
- [ ] Feature map details visible (Pauli, 16-qubit, reps=2)
- [ ] Circuit composition explained
- [ ] Comparison with classical models available
- [ ] Can understand why quantum helps

#### Journey 4: Run Custom Pipeline
- [ ] User navigates to `/simulation/parameters`
- [ ] Can adjust hyperparameters (if exposed):
  - [ ] Feature map type (Pauli/ZZ)
  - [ ] Qubit count
  - [ ] QSVC regularization (C)
  - [ ] Training epochs
- [ ] Can submit job to queue
- [ ] Receives confirmation
- [ ] Can check status on `/simulation` page

#### Journey 5: Understand the System
- [ ] User reads home page description
- [ ] Understands it's research tool, not clinical advice
- [ ] Knows it uses Hetionet (47k entities, 2.25M relations)
- [ ] Understands RotatE embeddings + quantum kernel approach
- [ ] Score interpretation guide is clear
- [ ] Can find documentation/help (if present)

### 3.3 Performance & Load Times ✓
- [ ] **Frontend Build**
  - [ ] `npm run build` completes without errors
  - [ ] Build time < 60 seconds
  - [ ] No unused dependencies in bundle
  - [ ] TypeScript compilation clean (no errors/warnings)

- [ ] **Page Load Times**
  - [ ] Home page loads in < 2 seconds
  - [ ] Predict form interactive in < 1 second
  - [ ] Experiments table loads in < 3 seconds
  - [ ] Large visualizations (graph, 3D) load in < 5 seconds

- [ ] **API Response Times**
  - [ ] Prediction request responds in < 2 seconds
  - [ ] Error responses instant (< 500ms)
  - [ ] No timeout on legitimate requests

---

## Phase 4: Quality & Best Practices

### 4.1 Code Quality (Backend) ✓
- [ ] **Python Tests**
  - [ ] All pytest tests pass: `python -m pytest tests/ -v`
  - [ ] Test discovery metrics pass (see test_discovery_metrics.py)
  - [ ] `run_tests.py --mode terminal` completes without errors
  - [ ] No warnings in test output

- [ ] **Linting & Type Checking** (if available)
  - [ ] No flake8 errors
  - [ ] No mypy type errors
  - [ ] No unused imports
  - [ ] Docstrings present for public functions

- [ ] **Dependency Versions**
  - [ ] All requirements in requirements.txt pinned
  - [ ] No conflicting versions
  - [ ] Compatible with Python 3.10+

### 4.2 Code Quality (Frontend) ✓
- [ ] **ESLint**
  - [ ] `npm run lint` passes with 0 errors
  - [ ] No ESLint warnings in terminal
  - [ ] Consistent code style

- [ ] **TypeScript**
  - [ ] `npm run build` has no type errors
  - [ ] No `any` types without justification
  - [ ] Props properly typed
  - [ ] Return types specified

- [ ] **React Best Practices**
  - [ ] Hooks used correctly (deps arrays complete)
  - [ ] No missing keys in lists
  - [ ] No direct DOM manipulation
  - [ ] Components memoized if needed for performance

- [ ] **Accessibility (WCAG 2.1 AA)**
  - [ ] All form inputs have associated labels
  - [ ] Color not only way to convey meaning
  - [ ] Focus visible on all interactive elements
  - [ ] Keyboard navigation works
  - [ ] ARIA roles appropriate where used
  - [ ] Alt text on images (if any)

### 4.3 Security ✓
- [ ] **Frontend**
  - [ ] No hardcoded API keys in code
  - [ ] Environment variables used for secrets
  - [ ] XSS protections in place (React escapes by default)
  - [ ] CSRF tokens if applicable
  - [ ] Content-Security-Policy headers set (if possible)

- [ ] **Backend**
  - [ ] No SQL injection (if using SQL)
  - [ ] Input validation on all endpoints
  - [ ] Rate limiting (if needed)
  - [ ] Secrets in environment, not code

- [ ] **Data Privacy**
  - [ ] User inputs not logged (or logged securely)
  - [ ] Drug/disease names not exposed in URLs
  - [ ] Predictions not stored without consent
  - [ ] No PII leaked in error messages

---

## Phase 5: Documentation & Help

### 5.1 In-App Documentation ✓
- [ ] **Inline Help**
  - [ ] Score interpretation guide visible on `/predict`
  - [ ] Caveat about research-tool nature clear
  - [ ] Example inputs helpful (pindolol, DB00960)
  - [ ] Hetionet attribution present

- [ ] **Experiment Results**
  - [ ] Model names descriptive
  - [ ] PR-AUC metric explained
  - [ ] Comparison with baselines clear
  - [ ] Best model highlighted

### 5.2 External Documentation ✓
- [ ] **README**
  - [ ] Setup instructions clear
  - [ ] Architecture diagram present (if available)
  - [ ] Results table prominent
  - [ ] Citation information included

- [ ] **Comments & Docstrings** (Code)
  - [ ] Complex functions have docstrings
  - [ ] Quantum concepts briefly explained in comments
  - [ ] Assumptions documented (e.g., embedding dim)

---

## Phase 6: Regression Testing (Before Each Release)

### 6.1 Critical User Paths ✓
- [ ] **Known Good Prediction**
  - [ ] Drug: "ibuprofen", Disease: "headache"
  - [ ] Should score ≥ 0.60 (rough estimate)
  - [ ] Score should be consistent across runs

- [ ] **Edge Cases**
  - [ ] Unknown drug returns error gracefully
  - [ ] Empty inputs rejected with message
  - [ ] Special characters in names handled
  - [ ] Very long drug names handled

### 6.2 Metric Verification ✓
- [ ] **PR-AUC Remains ≥ 0.7987**
  - [ ] Run test set through stacking ensemble
  - [ ] PR-AUC reported as ≥ 0.7987
  - [ ] No degradation from refactoring
  - [ ] Results reproducible with seed

- [ ] **Ensemble Component Metrics**
  - [ ] RandomForest ≥ 0.7838
  - [ ] ExtraTrees ≥ 0.7807
  - [ ] QSVC ≥ 0.7216
  - [ ] All base models produce valid scores

---

## Test Execution Summary

### Running All Tests
```bash
# Backend tests
python -m pytest tests/ -v
python run_tests.py --mode terminal

# Frontend tests (manual + linting)
cd frontend
npm run lint
npm run build

# Development server for manual testing
npm run dev
# Navigate to http://localhost:3000
```

### Expected Outcomes
- ✅ All Python tests pass
- ✅ No ESLint errors
- ✅ TypeScript compilation clean
- ✅ All user journeys complete successfully
- ✅ PR-AUC ≥ 0.7987 verified
- ✅ No console errors in browser
- ✅ All pages render correctly
- ✅ API responses < 2 seconds

### Failure Handling
If any test fails:
1. Note the exact error message and step
2. Check recent changes (`git log`)
3. Isolate the failing component
4. Check dependencies (pip freeze, npm list)
5. Review test assumptions
6. File bug report with reproducible steps

---

## Appendix: Test Data & Known Values

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Ensemble-QC-stacking (Pauli) PR-AUC | > 0.70 | 0.7987 | ✅ PASS |
| RandomForest PR-AUC | — | 0.7838 | ✅ Baseline |
| ExtraTrees PR-AUC | — | 0.7807 | ✅ Baseline |
| QSVC PR-AUC | — | 0.7216 | ✅ Baseline |
| Hetionet entities | 47,031 | — | — |
| Hetionet relations | 2.25M | — | — |
| Embedding dimension | 128 | — | — |
| Training epochs | 200 | — | — |
| Quantum qubits | 16 | — | — |
| Top-k hit rate (k=10) | — | varies | — |
| Mean rank of positives | — | varies | — |

---

**Last Updated**: 2026-04-22  
**Prepared By**: Claude (QA Analysis)  
**Status**: Ready for QA Execution
