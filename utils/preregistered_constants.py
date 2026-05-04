"""Pre-registered methodological constants.

Locked by ``preregistration/osf_preregistration_v1.md``. Modifying any value
in this file requires a pre-registration amendment (Appendix §12) and a
deviation log entry in the manuscript reproducibility appendix.

Values reflect the configuration that produced PR-AUC 0.7987 on the Hetionet
Compound-treats-Disease (CtD) benchmark (PauliFeatureMap reps=2, RotatE 128D
embeddings, hard negative sampling, stacking ensemble). The reconciliation
between this configuration and the originally drafted preregistration
(ZZFeatureMap, metapath features, random negatives) is documented in
``preregistration/osf_preregistration_v1.md`` §12.
"""
from __future__ import annotations

# §3.2 — Edge types in scope (drug-repurposing inference subgraph)
HETIONET_EDGE_TYPES_IN_SCOPE: tuple[str, ...] = (
    "CtD",  # Compound-treats-Disease
    "CrC",  # Compound-resembles-Compound
    "CbG",  # Compound-binds-Gene
    "DaG",  # Disease-associates-Gene
)

# §4 — Pair feature construction (knowledge-graph node embeddings)
EMBEDDING_METHOD: str = "RotatE"
EMBEDDING_DIM: int = 128
EMBEDDING_EPOCHS: int = 200
EMBEDDING_FULL_GRAPH: bool = True  # Train on all 24 Hetionet relations
PAIR_FEATURE_OPS: tuple[str, ...] = ("concat", "diff", "hadamard")

# §5.1 — Quantum kernel feature map (PRIMARY)
QSVC_FEATURE_MAP_TYPE: str = "PauliFeatureMap"
QSVC_FEATURE_MAP_REPS: int = 2
QSVC_C_VALUES: tuple[float, ...] = (0.05, 0.1, 1.0, 10.0)
QSVC_C_BEST: float = 0.1

# §5.1.1 — Pre-PCA reduction before quantum kernel evaluation
QSVC_PRE_PCA_DIM: int = 24
QSVC_QML_DIM: int = 16  # Number of qubits

# §5.1.2 — Sensitivity comparison (SUPPLEMENTARY)
QSVC_FEATURE_MAP_SENSITIVITY: tuple[str, ...] = ("ZZFeatureMap",)

# §5.4 — Hardware experiment scope (forward-looking)
HARDWARE_BACKEND_PRIMARY: str = "ibm_torino"
HARDWARE_BACKEND_BACKUP: str = "ibm_brisbane"
HARDWARE_PROBLEM_SIZES: tuple[int, ...] = (10, 15, 20)
HARDWARE_BACKEND_SNAPSHOTS: int = 3
HARDWARE_SAMPLES_PER_CONDITION: int = 100

# §5.5 — ZNE error mitigation
ZNE_NOISE_SCALES: tuple[int, ...] = (1, 3, 5)
ZNE_EXTRAPOLATION_METHODS: tuple[str, ...] = ("linear", "richardson")

# §6 — Baselines (executed)
BASELINES_EXECUTED: tuple[str, ...] = (
    "LogisticRegression",
    "RandomForest",
    "ExtraTrees",
)

# §6.2 — Forward-looking baselines
BASELINES_FORWARD_LOOKING: tuple[str, ...] = ("R-GCN", "TransE")
RGCN_TUNING_TRIALS: int = 50
TRANSE_TUNING_TRIALS: int = 50

# §6.4 — Headline classifier (ensemble alongside QSVC-alone)
ENSEMBLE_METHOD: str = "stacking"
ENSEMBLE_TUNE_CLASSICAL: bool = True

# §7.1 — Splits
TRAIN_FRAC: float = 0.7
VAL_FRAC: float = 0.15
TEST_FRAC: float = 0.15
SPLIT_SEED: int = 20251015
NEGATIVE_SAMPLING: str = "hard"
NEGATIVE_SAMPLE_RATIO: int = 1

# §7.3 — Metrics
PRIMARY_METRIC: str = "PR-AUC"
SECONDARY_METRIC: str = "ROC-AUC"

# §8.4 — Bootstrap procedure (paired-bootstrap CI on per-fold PR-AUC differences)
BOOTSTRAP_N_RESAMPLES: int = 10_000
BOOTSTRAP_CONFIDENCE: float = 0.95
BOOTSTRAP_SEED: int = 20260504
BOOTSTRAP_CV_FOLDS: int = 5

# §8.2 — H2 threshold (hardware vs simulator)
H2_PR_AUC_DELTA_THRESHOLD: float = 0.05  # 5 percentage points

# §8.3 — H3 threshold (sub-quadratic scaling)
H3_SCALING_SLOPE_UPPER_BOUND: float = 2.0
