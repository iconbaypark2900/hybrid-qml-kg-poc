High‑impact structural fixes (edits to make)
Full‑graph embeddings (more signal)
Train embeddings on the full Hetionet (all relations), then filter to CtD only when building train/test features.
Where: kg_layer/advanced_embeddings.py (training call) and kg_layer/kg_loader.py (relation filtering order).
PCA/feature hygiene (remove NaNs, standardize, verify variance)
Standardize inputs, drop constant/NaN columns before PCA, and log explained variance; ensure no “nan%”.
Where: quantum_layer/advanced_qml_features.py.
Robust evaluation (reduce variance)
Add stratified k‑fold cross‑validation and aggregate PR‑AUC, not a single split.
Where: evaluation logic in scripts/run_optimized_pipeline.py (or a helper under classical_baseline/).
Better QSVC search and caching
Expand C grid and optionally kernel scaling; cache quantum kernel matrices across seeds/runs.
Where: quantum_layer/qml_trainer.py.
Negative sampling and calibration
Improve negative sampling diversity and add probability calibration (Platt/Isotonic) for PR‑AUC.
Where: kg_layer/kg_loader.py (sampling), classical_baseline/train_baseline.py and QML eval for calibration.
Guard against leakage
Ensure any feature computation that uses graph/global stats is fit on train only, applied to test.
Where: kg_layer/enhanced_features.py.
Stronger classical baselines
Wider hyperparameter grids and class_weight tuning for RF/SVM/LogReg.
Where: classical_baseline/train_baseline.py.
Reproducibility
Centralize and propagate --seed to all RNGs (numpy, torch, qiskit).
Where: shared utils used by all trainers.
Lightweight logging/artifacts
Always save PR curves and confusion matrices per model; summarize in the final JSON.
Where: results writing in scripts/run_optimized_pipeline.py and quantum_layer/qml_trainer.py.
Optional: dimensionality knobs
Expose CLI flags for feature-select k and PCA components for QML/classical branches.
Where: quantum_layer/advanced_qml_features.py, CLI in scripts/run_optimized_pipeline.py.