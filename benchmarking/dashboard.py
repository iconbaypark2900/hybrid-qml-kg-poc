# Hybrid QML-KG Biomedical Link Prediction Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
import sys
import logging
from pathlib import Path
from typing import Optional
import json
import time
import subprocess
import tempfile
import joblib
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _redact_token_from_message(message: str, token: Optional[str] = None) -> str:
    """Remove API tokens from a string so they are never shown in errors or logs."""
    import re
    out = message
    if token and len(token) > 8:
        out = out.replace(token, "[REDACTED]")
    # Redact any long alphanumeric token-like substring (e.g. 40+ chars)
    out = re.sub(r"[A-Za-z0-9_-]{40,}", "[REDACTED]", out)
    return out


# Page config
st.set_page_config(
    page_title="Hybrid QML-KG | Biomedical Link Prediction",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
/* Font stack */
html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
}
.block-container { padding-top: 1.5rem; }

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stRadio > div {
    gap: 2px;
}
section[data-testid="stSidebar"] .stRadio > div > label {
    background: transparent;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 0.88rem;
    font-weight: 500;
    transition: background 0.15s;
    cursor: pointer;
}
section[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
section[data-testid="stSidebar"] .stRadio > div [aria-checked="true"] ~ label {
    background: rgba(99,102,241,0.25);
    font-weight: 600;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1);
}
section[data-testid="stSidebar"] button {
    border-color: rgba(255,255,255,0.2) !important;
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] button:hover {
    background: rgba(255,255,255,0.08) !important;
}

/* ---- Metric cards ---- */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #6366f1;
    border-radius: 8px;
    padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
[data-testid="stMetricValue"] {
    font-size: 1.7rem;
    font-weight: 700;
    color: #1e293b;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
    font-weight: 600;
}
[data-testid="stMetricDelta"] {
    font-size: 0.78rem;
}

/* ---- Custom card component ---- */
.dash-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 20px 22px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    margin-bottom: 8px;
    height: 100%;
}
.dash-card-accent { border-top: 3px solid #6366f1; }
.dash-card-blue   { border-top: 3px solid #3b82f6; }
.dash-card-purple { border-top: 3px solid #8b5cf6; }
.dash-card-amber  { border-top: 3px solid #f59e0b; }
.dash-card-green  { border-top: 3px solid #10b981; }
.dash-card h4 {
    margin: 0 0 8px 0;
    font-size: 0.95rem;
    font-weight: 700;
    color: #1e293b;
}
.dash-card p {
    margin: 0;
    font-size: 0.84rem;
    color: #475569;
    line-height: 1.55;
}
.dash-card .card-stat {
    font-size: 1.6rem;
    font-weight: 700;
    color: #6366f1;
    margin: 4px 0;
}

/* ---- General polish ---- */
.stAlert { border-radius: 8px; }
.streamlit-expanderHeader { font-weight: 600; font-size: 0.9rem; }
hr { border: none; border-top: 1px solid #e2e8f0; margin: 1.5rem 0; }
.stDataFrame th {
    background-color: #f8fafc;
    font-weight: 600;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.stDataFrame td { font-size: 0.86rem; }

/* ---- Section header accent ---- */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6366f1;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)


def card(title: str, body: str, accent: str = "accent", stat: str = None) -> str:
    """Return an HTML card. accent: accent|blue|purple|amber|green."""
    stat_html = f'<div class="card-stat">{stat}</div>' if stat else ""
    return (
        f'<div class="dash-card dash-card-{accent}">'
        f"<h4>{title}</h4>"
        f"{stat_html}"
        f"<p>{body}</p>"
        f"</div>"
    )

# ---------- Glossary: layman + technical definitions for explainability ----------
GLOSSARY = {
    "link_prediction": (
        "**In plain terms:** Predicting whether a connection *should* exist between two things (e.g., “Does this drug treat this disease?”). "
        "**Technical:** A supervised or ranking task over graph edges: given (head, relation, tail), predict link existence or score.",
    ),
    "hetionet": (
        "**In plain terms:** A public “map” of biomedical facts—drugs, diseases, genes, and how they relate (e.g., “treats”, “associates with”). "
        "**Technical:** Heterogeneous knowledge graph (Hetionet) integrating multiple bio databases; we use the CtD (Compound–treats–Disease) relation.",
    ),
    "ctd": (
        "**Compound–treats–Disease (CtD):** A relation type meaning “this compound is known or predicted to treat this disease.” "
        "Used here as the target for link prediction and drug-repurposing style ranking.",
    ),
    "embedding": (
        "**In plain terms:** A list of numbers that represents an entity (e.g., a drug or disease) so a computer can compare and learn from it. "
        "**Technical:** Dense vector representation (e.g., from ComplEx/RotatE) in a continuous space; used as input to classical and quantum models.",
    ),
    "pr_auc": (
        "**In plain terms:** How good the model is at ranking positive links above negative ones, especially when positives are rare. Higher is better (max 1.0). "
        "**Technical:** Area under the Precision–Recall curve; preferred over ROC-AUC for imbalanced binary classification.",
    ),
    "accuracy": (
        "**In plain terms:** Fraction of predictions that are correct (right label). "
        "**Technical:** (TP + TN) / (TP + TN + FP + FN); can be misleading when classes are imbalanced.",
    ),
    "qubit": (
        "**In plain terms:** The basic unit of quantum information (like a “quantum bit”). More qubits = larger quantum state space. "
        "**Technical:** Two-level quantum system; our feature vectors are reduced to `num_qubits` dimensions for the quantum circuit.",
    ),
    "qsvc": (
        "**In plain terms:** A classifier that uses a “quantum similarity” between pairs of inputs instead of a classical kernel. "
        "**Technical:** Quantum Support Vector Classifier: SVM with a kernel computed by a quantum circuit (e.g., fidelity between feature-map states).",
    ),
    "vqc": (
        "**In plain terms:** A small quantum circuit whose “knobs” are tuned so its output predicts the label. "
        "**Technical:** Variational Quantum Classifier: parameterized circuit + classical optimizer (e.g., SPSA) for binary classification.",
    ),
    "ideal_vs_noisy": (
        "**In plain terms:** “Ideal” = perfect simulation; “noisy” = simulation that mimics real hardware errors. Comparing them shows how robust the model is. "
        "**Technical:** Ideal backend (statevector or noiseless); noisy backend uses a noise model (e.g., from real device calibration).",
    ),
    "feature_map": (
        "**In plain terms:** The recipe that turns your numbers into a quantum state so the quantum computer can “see” the input. "
        "**Technical:** Unitary that encodes classical features into qubits (e.g., ZZFeatureMap with a given number of repetitions).",
    ),
    "backend": (
        "**In plain terms:** Where the quantum (or classical) computation runs—e.g., “ideal simulator” vs “noisy simulator” vs real hardware. "
        "**Technical:** Execution target: statevector simulator, Aer with noise model, or IBM Runtime backend.",
    ),
    "parameters": (
        "**In plain terms:** Number of tunable numbers in the model. Fewer parameters can mean simpler, more scalable models. "
        "**Technical:** Trainable parameter count (e.g., classical logistic regression weights vs quantum circuit parameters).",
    ),
    "kernel_similarity": (
        "**In plain terms:** A single number saying how “similar” two inputs are in the model’s internal representation. Not a probability. "
        "**Technical:** Kernel evaluation k(x,y); e.g., statevector fidelity between two feature-map states.",
    ),
    "metaedge": (
        "**In plain terms:** A type of relationship in the graph (e.g., “treats”, “associates with”). "
        "**Technical:** Edge type in a heterogeneous graph; Hetionet uses abbreviations like CtD, DaG.",
    ),
    "hybrid": (
        "**In plain terms:** Combines classical and quantum predictions to get the best of both. "
        "**Technical:** Ensemble that averages or stacks classical model scores with quantum kernel-based scores.",
    ),
    "pca": (
        "**In plain terms:** A technique to shrink high-dimensional data to fewer dimensions while keeping important patterns. "
        "**Technical:** Principal Component Analysis: projects data onto top eigenvectors of the covariance matrix.",
    ),
    "negative_sampling": (
        "**In plain terms:** Creating fake non-links to train the model on what is *not* a valid connection. "
        "**Technical:** Sampling (head, tail) pairs with no known edge for the relation type; balances positive examples.",
    ),
    "encoding": (
        "**In plain terms:** How the pair of entity embeddings are combined into a single vector for the model. "
        "**Technical:** Strategies include concat (concatenation), hadamard (element-wise product), or hybrid (both).",
    ),
    "cross_validation": (
        "**In plain terms:** Testing the model on multiple different splits of the data to make sure it works well in general, not just on one particular test set. "
        "**Technical:** K-Fold cross-validation: splits data into k parts, trains on k-1 parts and tests on 1 part, repeated k times.",
    ),
    "overfitting": (
        "**In plain terms:** When a model learns the training data too well, including its random quirks, so it doesn't work well on new data. "
        "**Technical:** Model performs significantly better on training data than on unseen test data, indicating poor generalization.",
    ),
    "regularization": (
        "**In plain terms:** Techniques to prevent overfitting by discouraging overly complex models. "
        "**Technical:** Methods to constrain model complexity, such as limiting parameters or adding penalty terms to the loss function.",
    ),
    "anti_overfitting": (
        "**In plain terms:** Specific measures taken to ensure the model generalizes well to new data, not just memorizes the training set. "
        "**Technical:** Includes cross-validation, regularization, proper train/validation/test splits, and monitoring CV-test gaps.",
    ),
}


def _expander_for_term(key: str, title: str = None):
    """Render an expander explaining a glossary term."""
    if key not in GLOSSARY:
        return
    with st.expander(title or f"What is “{key.replace('_', ' ').title()}”?"):
        st.markdown(GLOSSARY[key])

# Paths (use PROJECT_ROOT so app works when run from HF Spaces or any cwd)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCALING_PLOT = PROJECT_ROOT / "docs" / "scaling_projection.png"


def _get_results_dir():
    """Use a writable results dir. On HF Spaces the repo is read-only, so use /tmp."""
    preferred = PROJECT_ROOT / "results"
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        (preferred / ".write_check").write_text("")
        (preferred / ".write_check").unlink()
        return preferred
    except (OSError, PermissionError):
        tmp = Path(tempfile.gettempdir()) / "hybrid_qml_kg_results"
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp


RESULTS_DIR = _get_results_dir()
LATEST_RUN = RESULTS_DIR / "latest_run.csv"
HISTORY_FILE = RESULTS_DIR / "experiment_history.csv"

# ---------- Best-run defaults (from docs/planning/NEXT_STEPS_TO_IMPROVE_PERFORMANCE.md) ----------
BEST_RUN_COMMAND = (
    "python scripts/run_optimized_pipeline.py --relation CtD \\\n"
    "  --full_graph_embeddings --embedding_method RotatE --embedding_dim 128 \\\n"
    "  --embedding_epochs 200 --negative_sampling hard --qml_dim 16 \\\n"
    "  --qml_feature_map Pauli --qml_feature_map_reps 2 --qsvc_C 0.1 \\\n"
    "  --optimize_feature_map_reps --run_ensemble --ensemble_method stacking \\\n"
    "  --tune_classical --qml_pre_pca_dim 24 --fast_mode"
)

BEST_RUN_CONFIG = {
    "relation": "CtD",
    "full_graph_embeddings": True,
    "embedding_method": "RotatE",
    "embedding_dim": 128,
    "embedding_epochs": 200,
    "negative_sampling": "hard",
    "qml_dim": 16,
    "qml_feature_map": "Pauli",
    "qml_feature_map_reps": 2,
    "qsvc_C": 0.1,
    "optimize_feature_map_reps": True,
    "run_ensemble": True,
    "ensemble_method": "stacking",
    "tune_classical": True,
    "qml_pre_pca_dim": 24,
    "fast_mode": True,
}

BEST_RUN_RANKING = [
    {"name": "Ensemble-QC-stacking (Pauli)", "type": "ensemble", "pr_auc": 0.7987, "accuracy": 0.7450, "fit_time": 0.0},
    {"name": "RandomForest-Optimized", "type": "classical", "pr_auc": 0.7838, "accuracy": 0.7320, "fit_time": 0.0},
    {"name": "ExtraTrees-Optimized", "type": "classical", "pr_auc": 0.7807, "accuracy": 0.7280, "fit_time": 0.0},
    {"name": "Ensemble-QC-stacking (ZZ)", "type": "ensemble", "pr_auc": 0.7408, "accuracy": 0.7050, "fit_time": 0.0},
    {"name": "QSVC-Optimized", "type": "quantum", "pr_auc": 0.7216, "accuracy": 0.6900, "fit_time": 0.0},
]

BEST_RUN_EXPERIMENT_LOG = [
    {"Variant": "Base (200 ep, stacking, tune_classical, pre_pca 24)", "RF": 0.7838, "ET": 0.7807, "QSVC": 0.7216, "Ensemble": 0.7408, "Notes": "Best classical"},
    {"Variant": "+ Pauli feature map (reps=2)", "RF": 0.7838, "ET": 0.7807, "QSVC": 0.6343, "Ensemble": 0.7987, "Notes": "Best ensemble"},
    {"Variant": "+ diverse negatives (dw=0.5)", "RF": 0.7144, "ET": 0.7298, "QSVC": 0.6689, "Ensemble": 0.6919, "Notes": "Lower; diverse hurts here"},
    {"Variant": "+ qsvc_C=0.05", "RF": 0.7838, "ET": 0.7807, "QSVC": 0.7216, "Ensemble": 0.7408, "Notes": "Same as C=0.1"},
    {"Variant": "+ ensemble_quantum_weight=0.4", "RF": 0.7838, "ET": 0.7807, "QSVC": 0.7216, "Ensemble": 0.7408, "Notes": "No effect (stacking learns)"},
]

QUICK_RUN_COMMAND = (
    "python scripts/run_optimized_pipeline.py --relation CtD \\\n"
    "  --full_graph_embeddings --embedding_method RotatE --embedding_dim 128 \\\n"
    "  --embedding_epochs 50 --negative_sampling hard --qml_dim 8 \\\n"
    "  --qml_feature_map_reps 2 --qsvc_C 0.1 \\\n"
    "  --tune_classical --fast_mode"
)

# Ensure local project modules (kg_layer/, quantum_layer/, etc.) are importable in Streamlit
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.ibm_runtime_verify import verify_ibm_quantum_runtime  # noqa: E402

# Load latest results (ttl=60 so Overview/Results charts update after new runs without requiring Refresh)
@st.cache_data(ttl=60)
def load_latest_results():
    """
    Loads the latest results from a CSV file.

    Returns
    -------
    pd.DataFrame or None
        A DataFrame containing the latest results, or None if the file does not exist.
    """
    if LATEST_RUN.exists():
        return pd.read_csv(LATEST_RUN)
    return None

@st.cache_data(ttl=60)
def load_history():
    """
    Loads the experiment history from a CSV file.

    Returns
    -------
    pd.DataFrame or None
        A DataFrame containing the experiment history, or None if the file does not exist.
    """
    if HISTORY_FILE.exists():
        return pd.read_csv(HISTORY_FILE)
    return None


def load_optimized_results():
    """Load the latest optimized_results_*.json. Returns dict or None."""
    try:
        dirs_to_scan = [RESULTS_DIR]
        if (PROJECT_ROOT / "results").resolve() != RESULTS_DIR.resolve():
            dirs_to_scan.append(PROJECT_ROOT / "results")
        files = []
        for d in dirs_to_scan:
            if d.exists():
                files.extend(d.glob("optimized_results_*.json"))
        if not files:
            return None
        latest = max(files, key=lambda p: p.stat().st_mtime)
        with open(latest, "r") as f:
            out = json.load(f)
        ranking = out.get("ranking") or []
        if not ranking and ("classical_results" in out or "quantum_results" in out):
            ranking = []
            for name, res in (out.get("classical_results") or {}).items():
                if isinstance(res, dict) and res.get("status") == "success":
                    tm = res.get("test_metrics") or {}
                    ranking.append({"name": name, "type": "classical", "pr_auc": tm.get("pr_auc", 0.0), "accuracy": tm.get("accuracy", 0.0), "fit_time": res.get("fit_seconds", 0.0)})
            for name, res in (out.get("quantum_results") or {}).items():
                if isinstance(res, dict) and res.get("status") == "success":
                    tm = res.get("test_metrics") or {}
                    ranking.append({"name": name, "type": "quantum", "pr_auc": tm.get("pr_auc", 0.0), "accuracy": tm.get("accuracy", 0.0), "fit_time": res.get("fit_seconds", 0.0)})
            for name, res in (out.get("ensemble_results") or {}).items():
                if isinstance(res, dict) and res.get("status") == "success":
                    tm = res.get("test_metrics") or {}
                    ranking.append({"name": name, "type": "ensemble", "pr_auc": tm.get("pr_auc", 0.0), "accuracy": tm.get("accuracy", 0.0), "fit_time": res.get("fit_seconds", 0.0)})
            ranking.sort(key=lambda x: x.get("pr_auc", 0.0), reverse=True)
            out = dict(out)
            out["ranking"] = ranking
        return out
    except Exception:
        return None


df_latest = load_latest_results()
df_history = load_history()

@st.cache_data
def load_entity_embeddings():
    candidates = [
        ("data/complex_128d_entity_embeddings.npy", "data/complex_128d_entity_ids.json", "complex_128d"),
        ("data/entity_embeddings.npy", "data/entity_ids.json", "entity_embeddings"),
        ("data/rotate_256d_entity_embeddings.npy", "data/rotate_256d_entity_ids.json", "rotate_256d"),
    ]
    for emb_path, id_path, _label in candidates:
        emb_file = (PROJECT_ROOT / emb_path).resolve()
        id_file = (PROJECT_ROOT / id_path).resolve()
        if emb_file.exists() and id_file.exists():
            embeddings = np.load(emb_file)
            with open(id_file, "r") as f:
                mapping = json.load(f)  # {entity: id}
            # Invert mapping to get entity by index
            idx_to_entity = [None] * len(mapping)
            for ent, idx in mapping.items():
                idx_to_entity[int(idx)] = str(ent)
            return embeddings, idx_to_entity, emb_file.name
    return None, None, None

@st.cache_resource
def fit_pca_reducer(embeddings: np.ndarray, reduced_dim: int):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=reduced_dim, random_state=42, svd_solver="full")
    pca.fit(embeddings)
    return pca

def safe_get(df: pd.DataFrame, col: str, default=None):
    if df is None or col not in df.columns or df.empty:
        return default
    return df[col].iloc[0]

def latest_execution_summary(df_hist: pd.DataFrame) -> pd.DataFrame:
    if df_hist is None or df_hist.empty:
        return pd.DataFrame()
    required = {"execution_mode", "noise_model", "backend_label"}
    if not required.issubset(df_hist.columns):
        return pd.DataFrame()
    df_comp = df_hist.copy()
    df_comp["run_index"] = df_comp.index
    group_cols = ["execution_mode", "noise_model", "backend_label"]
    latest = (
        df_comp.sort_values("run_index")
        .groupby(group_cols, dropna=False)
        .tail(1)
        .reset_index(drop=True)
    )
    display_cols = group_cols + ["run_index"]
    for metric in ["quantum_pr_auc", "classical_pr_auc", "quantum_accuracy", "classical_accuracy"]:
        if metric in latest.columns:
            display_cols.append(metric)
    return latest[display_cols].sort_values(group_cols)

@st.cache_resource
def load_classical_artifacts():
    _candidates = [
        PROJECT_ROOT / "models" / "classical_best.joblib",
        PROJECT_ROOT / "models" / "classical_logisticregression.joblib",
    ]
    model_path  = next((p for p in _candidates if p.exists()), _candidates[-1])
    scaler_path = PROJECT_ROOT / "models" / "scaler.joblib"
    if not model_path.exists() or not scaler_path.exists():
        return None, None, str(model_path), str(scaler_path)
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler, str(model_path), str(scaler_path)

def normalize_compound_id(drug_input: str) -> str:
    s = drug_input.strip()
    if s.startswith("Compound::"):
        return s
    return f"Compound::{s}"

def normalize_disease_id(disease_input: str) -> str:
    s = disease_input.strip()
    if s.startswith("Disease::"):
        return s
    # Accept DOID_9352 → Disease::DOID:9352
    if s.startswith("DOID_"):
        s = "DOID:" + s.split("_", 1)[1]
    if s.startswith("DOID:"):
        return f"Disease::{s}"
    return f"Disease::{s}"

def split_entities(entity_ids: list) -> tuple[list[str], list[str]]:
    compounds = [e for e in entity_ids if isinstance(e, str) and e.startswith("Compound::")]
    diseases = [e for e in entity_ids if isinstance(e, str) and e.startswith("Disease::")]
    return sorted(compounds), sorted(diseases)

def build_pair_features(reduced_h: np.ndarray, reduced_t: np.ndarray) -> np.ndarray:
    diff = np.abs(reduced_h - reduced_t)
    had = reduced_h * reduced_t
    return np.concatenate([reduced_h, reduced_t, diff, had], axis=0).reshape(1, -1)

@st.cache_resource
def load_quantum_kernel(
    qml_dim: int,
    reps: int = 2,
    feature_map_type: str = "ZZ",
    entanglement: str = "linear",
):
    """
    Return a statevector fidelity kernel for fast local 'quantum similarity' scoring.
    feature_map_type: "ZZ" or "Z". entanglement: "linear", "full", or "circular".
    """
    try:
        from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
        from qiskit_machine_learning.kernels import FidelityStatevectorKernel
    except Exception as e:
        return None, str(e)

    fm_type = (feature_map_type or "ZZ").strip().upper()
    ent = (entanglement or "linear").strip().lower()
    if fm_type == "Z":
        fm = ZFeatureMap(feature_dimension=qml_dim, reps=reps)
    else:
        fm = ZZFeatureMap(feature_dimension=qml_dim, reps=reps, entanglement=ent)
    return FidelityStatevectorKernel(feature_map=fm), None


def _build_feature_map(qml_dim: int, reps: int, feature_map_type: str, entanglement: str):
    """Build Qiskit feature map circuit (for use with statevector or shot-based kernel)."""
    from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
    fm_type = (feature_map_type or "ZZ").strip().upper()
    ent = (entanglement or "linear").strip().lower()
    if fm_type == "Z":
        return ZFeatureMap(feature_dimension=qml_dim, reps=reps)
    return ZZFeatureMap(feature_dimension=qml_dim, reps=reps, entanglement=ent)


def load_shot_based_kernel(
    qml_dim: int,
    reps: int,
    feature_map_type: str,
    entanglement: str,
    config_path: str,
    shots_override: Optional[int] = None,
):
    """
    Build a shot-based FidelityQuantumKernel using QuantumExecutor and the given config.
    config_path: path to quantum_config_ideal.yaml or quantum_config_noisy.yaml.
    shots_override: if set, write a temp config with this shots value and use it.
    Returns (kernel, error_msg).
    """
    try:
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        from qiskit_machine_learning.state_fidelities import ComputeUncompute
    except Exception as e:
        return None, str(e)
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = PROJECT_ROOT / config_path
    if not config_file.exists():
        return None, f"Config not found: {config_file}"
    use_path = str(config_file)
    if shots_override is not None and shots_override > 0:
        try:
            import yaml
            with open(config_file, "r") as f:
                cfg = yaml.safe_load(f) or {}
            sim = cfg.get("quantum", {}).get("simulator", {})
            if isinstance(sim, dict):
                sim["shots"] = int(shots_override)
                out_dir = PROJECT_ROOT / "results"
                out_dir.mkdir(parents=True, exist_ok=True)
                temp_config = out_dir / ".live_quantum_config_temp.yaml"
                with open(temp_config, "w") as f:
                    yaml.dump(cfg, f, default_flow_style=False)
                use_path = str(temp_config)
        except Exception as e:
            logger.warning("Could not override shots in temp config: %s", e)
    try:
        from quantum_layer.quantum_executor import QuantumExecutor
        qe = QuantumExecutor(use_path)
        sampler, exec_mode = qe.get_sampler()
        fm = _build_feature_map(qml_dim, reps, feature_map_type, entanglement)
        fm_exec = fm.decompose(reps=10)
        qk = FidelityQuantumKernel(feature_map=fm_exec, fidelity=ComputeUncompute(sampler=sampler))
        return qk, None
    except Exception as e:
        return None, str(e)


def evaluate_kernel_zne(
    x1: np.ndarray,
    x2: np.ndarray,
    qml_dim: int,
    reps: int,
    feature_map_type: str,
    entanglement: str,
    base_noise_p: float,
    shots: int,
    scales: list = None,
) -> tuple:
    """
    Evaluate k(x1,x2) at multiple noise scales and extrapolate to zero noise (linear ZNE).
    Returns (raw_value_at_scale_1, mitigated_value, dict with scales and values).
    """
    if scales is None:
        scales = [1.0, 1.5, 2.0]
    try:
        from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
        from qiskit_aer.noise import NoiseModel, depolarizing_error
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        from qiskit_machine_learning.state_fidelities import ComputeUncompute
    except Exception as e:
        return float("nan"), float("nan"), {"error": str(e)}

    def _noise_model(p: float):
        nm = NoiseModel()
        one_qubit = ["x", "y", "z", "h", "s", "t", "sx", "rz", "rx", "ry"]
        two_qubit = ["cx", "cz", "swap", "ecr"]
        nm.add_all_qubit_quantum_error(depolarizing_error(p, 1), one_qubit)
        nm.add_all_qubit_quantum_error(depolarizing_error(p, 2), two_qubit)
        return nm

    fm = _build_feature_map(qml_dim, reps, feature_map_type, entanglement)
    fm_exec = fm.decompose(reps=10)
    x1 = np.asarray(x1, dtype=np.float64).reshape(1, -1)
    x2 = np.asarray(x2, dtype=np.float64).reshape(1, -1)
    values = []
    for s in scales:
        p = min(1.0, base_noise_p * s)
        nm = _noise_model(p)
        sampler = AerSamplerV2(
            default_shots=shots,
            options={"backend_options": {"noise_model": nm}},
        )
        qk = FidelityQuantumKernel(feature_map=fm_exec, fidelity=ComputeUncompute(sampler=sampler))
        val = float(qk.evaluate(x1, x2)[0, 0])
        values.append(val)
    S = np.asarray(scales, dtype=float)
    V = np.asarray(values, dtype=float)
    A = np.stack([np.ones_like(S), S], axis=1)
    coef, *_ = np.linalg.lstsq(A, V, rcond=None)
    mitigated = float(np.clip(coef[0], 0.0, 1.0))
    raw_at_1 = float(values[0]) if values else float("nan")
    return raw_at_1, mitigated, {"scales": scales, "values": values, "mitigated": mitigated}


def to_quantum_input(x: np.ndarray) -> np.ndarray:
    """Map arbitrary real vectors into a safe bounded range for feature maps."""
    x = np.asarray(x, dtype=np.float32)
    return np.tanh(x)

def cosine_topk_indices(mat: np.ndarray, vec: np.ndarray, k: int) -> np.ndarray:
    vec = vec.astype(np.float32)
    denom = (np.linalg.norm(mat, axis=1) * (np.linalg.norm(vec) + 1e-9)) + 1e-9
    sims = (mat @ vec) / denom
    return np.argsort(sims)[-k:][::-1]

def suggest_available(entity_ids: list, prefix: str, needle: str, k: int = 10) -> list:
    needle = needle.strip()
    pool = [e for e in entity_ids if isinstance(e, str) and e.startswith(prefix)]
    # simple heuristic: same DB/DOID substring match
    matches = [e for e in pool if needle in e]
    if matches:
        return matches[:k]
    # fallback: show some options
    return pool[:k]

def scan_hetionet_context(entity: str, max_matches: int = 30) -> pd.DataFrame:
    sif_path = PROJECT_ROOT / "data" / "hetionet-v1.0-edges.sif"
    if not sif_path.exists():
        return pd.DataFrame()
    rows = []
    with open(sif_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                parts = line.strip().split()
            if len(parts) < 3:
                continue
            h, r, t = parts[0], parts[1], parts[2]
            if h == entity or t == entity:
                rows.append({"source": h, "relation": r, "target": t})
                if len(rows) >= max_matches:
                    break
    return pd.DataFrame(rows)

def run_command(cmd: list, log_container):
    """Run a command; return (returncode, output_text)."""
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        output = []
        for line in proc.stdout:
            output.append(line)
            log_container.code("".join(output[-200:]))
        proc.wait()
        return proc.returncode, "".join(output)
    except Exception as e:
        log_container.error(f"Failed to run command: {e}")
        return 1, str(e)


def run_user_script(script_content: str, timeout_sec: int):
    """
    Run user-provided Python code in a subprocess with the project as cwd.
    Returns (returncode, stdout, stderr). Raises on timeout or spawn failure.
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix="user_script_",
        delete=False,
        dir=str(PROJECT_ROOT),
    ) as f:
        f.write(script_content)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_sec)),
        )
        return result.returncode, result.stdout, result.stderr
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass



# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.markdown(
    '<div style="padding: 4px 0 12px 0;">'
    '<p style="font-size:1.25rem; font-weight:800; margin:0; letter-spacing:-0.02em;">Hybrid QML-KG</p>'
    '<p style="font-size:0.72rem; font-weight:500; margin:2px 0 0 0; opacity:0.6; letter-spacing:0.04em; text-transform:uppercase;">Biomedical Link Prediction</p>'
    '</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "nav",
    [
        "The Problem",
        "Our Approach",
        "Results",
        "What We Learned",
        "Try It",
        "Technical Reference",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<p style="font-size:0.7rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; opacity:0.5; margin-bottom:6px;">Tools</p>',
    unsafe_allow_html=True,
)
if st.sidebar.button("Refresh data", use_container_width=True):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

with st.sidebar.expander("IBM Quantum (BYOK test)", expanded=False):
    st.caption(
        "API token from quantum.ibm.com — used only in-memory for the check; not stored."
    )
    _byok_t = st.text_input("API token", type="password", key="ibm_byok_token")
    _byok_crn = st.text_input(
        "Instance CRN (optional)",
        type="default",
        key="ibm_byok_crn",
        placeholder="crn:v1:bluemix:...",
    )
    if st.button("Verify Runtime connection", key="ibm_byok_verify", use_container_width=True):
        if not (_byok_t or "").strip():
            st.warning("Enter your API token.")
        else:
            with st.spinner("Connecting to IBM Quantum Runtime…"):
                _vr = verify_ibm_quantum_runtime(
                    _byok_t.strip(),
                    instance_crn=(_byok_crn or "").strip() or None,
                )
            if _vr.get("status") == "ok":
                st.success(_vr.get("message", "OK"))
                st.metric("Total backends", _vr.get("backend_count", 0))
                if _vr.get("hardware_backend_names"):
                    st.caption(
                        "Sample hardware: "
                        + ", ".join(_vr["hardware_backend_names"][:8])
                    )
            else:
                st.error((_vr.get("message") or "Failed")[:800])

# ============================================================================
# PAGE 1: THE PROBLEM
# ============================================================================
if page == "The Problem":
    st.markdown('<div class="section-label">Introduction</div>', unsafe_allow_html=True)
    st.header("The Problem")

    col_intro, col_stats = st.columns([3, 1])
    with col_intro:
        st.markdown("""
Developing a new drug takes **10--15 years** and over **$2 billion** on average.
Drug repurposing -- finding new uses for existing drugs -- can dramatically shorten
this timeline. But the space of possible drug-disease pairs is enormous.

**Hetionet** is a public biomedical knowledge graph that encodes what we know:
47,031 entities connected by 2.25 million relationships. One type is especially
valuable: **Compound-treats-Disease (CtD)**.
""")
    with col_stats:
        st.metric("Entities", "47,031")
        st.metric("Relationships", "2.25M")
        st.metric("Target relation", "CtD")

    st.markdown("")
    st.info(
        "Given a drug and a disease, can we predict whether the drug treats that disease -- "
        "and can **quantum computing** improve these predictions?"
    )

    st.markdown("")
    st.markdown('<div class="section-label">Knowledge Graph</div>', unsafe_allow_html=True)
    st.subheader("What a knowledge graph looks like")
    st.caption("A simplified subgraph of Hetionet showing drugs, diseases, genes, and their relationships.")

    st.markdown("""
```mermaid
graph LR
    Aspirin["Aspirin<br/>(DB00945)"] -->|treats ?| T2D["Type 2 Diabetes<br/>(DOID:9352)"]
    Aspirin -->|targets| COX2["COX-2 (Gene)"]
    Aspirin -->|side effect| Bleeding
    Metformin["Metformin<br/>(DB00331)"] -->|treats| T2D
    T2D -->|associates| HNF4A["HNF4A (Gene)"]
    T2D -->|localizes| Pancreas
```
""")

    st.markdown("")
    st.markdown('<div class="section-label">Challenges</div>', unsafe_allow_html=True)
    st.subheader("Why this is hard")

    ch1, ch2, ch3 = st.columns(3)
    with ch1:
        st.markdown(card(
            "Imbalanced data",
            "Only a tiny fraction of drug-disease pairs are true treatments. A random classifier scores 0.50 PR-AUC.",
            accent="blue",
        ), unsafe_allow_html=True)
    with ch2:
        st.markdown(card(
            "High dimensionality",
            "Each entity is a 128-D embedding. Pair features combine two embeddings into vectors with hundreds of dimensions.",
            accent="purple",
        ), unsafe_allow_html=True)
    with ch3:
        st.markdown(card(
            "Non-linear structure",
            "Knowledge graph relationships have complex, non-linear patterns that simple models miss.",
            accent="amber",
        ), unsafe_allow_html=True)

    st.markdown("")
    st.markdown("---")
    res_col1, res_col2, res_col3 = st.columns([1, 1, 2])
    res_col1.metric("Target", "PR-AUC > 0.70")
    res_col2.metric("Best result", "0.7987", "+0.30 vs random")
    res_col3.success("**Target achieved.** A hybrid quantum-classical stacking ensemble surpassed 0.70 PR-AUC.")

# ============================================================================
# PAGE 2: OUR APPROACH
# ============================================================================
elif page == "Our Approach":
    st.markdown('<div class="section-label">Architecture</div>', unsafe_allow_html=True)
    st.header("Our Approach")
    st.markdown(
        "A **hybrid quantum-classical pipeline** that processes the knowledge graph "
        "end-to-end: from raw biomedical data to ranked treatment predictions."
    )

    st.markdown("""
```mermaid
flowchart TD
    A["Hetionet (CtD)"] --> B["Full-Graph Embeddings<br/>RotatE 128D, 200 epochs"]
    B --> C["Pair Features<br/>concat + diff + Hadamard"]
    C --> D["Hard Negative Sampling"]
    D --> E["Classical Path<br/>RF, ExtraTrees, LogReg<br/>GridSearchCV"]
    D --> F["Quantum Path<br/>PCA 24D to 16 qubits<br/>Pauli / ZZ feature maps<br/>QSVC (C=0.1)"]
    E --> G["Stacking Ensemble"]
    F --> G
    G --> H["PR-AUC 0.7987"]
```
""")

    st.markdown("")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Embed the graph",
        "Build pair features",
        "Classical models",
        "Quantum models",
        "Combine (stacking)",
    ])

    with tab1:
        col_t, col_s = st.columns([3, 1])
        with col_t:
            st.markdown("""
We train **RotatE** embeddings on the **full** Hetionet graph (all 2.25M edges, not just CtD).
Each entity gets a 128-dimensional vector that captures its position in the graph's relational
structure. Training on the full graph gives richer representations than using the target
relation alone -- entities "know about" all their relationships.
""")
        with col_s:
            st.metric("Method", "RotatE")
            st.metric("Dimensions", "128")
            st.metric("Epochs", "200")

    with tab2:
        col_t, col_s = st.columns([3, 1])
        with col_t:
            st.markdown("""
For each candidate drug-disease pair, we combine the two entity embeddings into a single
feature vector using three operations: **concatenation**, **absolute difference**, and
**element-wise product** (Hadamard).

We use **hard negative sampling** -- selecting difficult non-treatment pairs that are
structurally similar to real treatments -- to force the model to learn fine-grained
distinctions.
""")
        with col_s:
            st.metric("Feature ops", "3")
            st.metric("Negatives", "Hard")

    with tab3:
        col_t, col_s = st.columns([3, 1])
        with col_t:
            st.markdown("""
Three classical baselines see the full pair-feature space:
- **RandomForest** and **ExtraTrees** -- tree ensembles for non-linear patterns
- **LogisticRegression** -- a linear baseline

All are tuned with **GridSearchCV** when enabled. The best classical model alone reaches
PR-AUC **0.7838**.
""")
        with col_s:
            st.metric("Best classical", "0.7838")
            st.metric("Models", "3")

    with tab4:
        col_t, col_s = st.columns([3, 1])
        with col_t:
            st.markdown("""
Pair features are reduced via PCA: first to 24 dimensions, then to **16 dimensions** --
one per qubit. These 16-D vectors are encoded into a quantum state using a **Pauli
feature map** (2 reps), creating a quantum kernel used by a **Quantum Support Vector
Classifier (QSVC)** with regularization C=0.1.

The quantum kernel captures correlations that are exponentially expensive to compute
classically. Whether this helps depends on the data -- and in our case, it helps the ensemble.
""")
        with col_s:
            st.metric("Qubits", "16")
            st.metric("Feature map", "Pauli")
            st.metric("QSVC C", "0.1")

    with tab5:
        col_t, col_s = st.columns([3, 1])
        with col_t:
            st.markdown("""
A **stacking ensemble** trains a meta-learner on the predictions of all classical and quantum
models. It learns how much to trust each model for each type of input. The stacking ensemble
reaches **PR-AUC 0.7987** -- higher than any individual model.
""")
        with col_s:
            st.metric("Method", "Stacking")
            st.metric("Best PR-AUC", "0.7987")

# ============================================================================
# PAGE 3: RESULTS
# ============================================================================
elif page == "Results":
    st.markdown('<div class="section-label">Performance</div>', unsafe_allow_html=True)
    st.header("Results")

    opt = load_optimized_results()
    ranking = (opt.get("ranking") if opt else None) or BEST_RUN_RANKING

    best_ensemble = next((r for r in ranking if r.get("type") == "ensemble"), ranking[0])
    best_classical = next((r for r in ranking if r.get("type") == "classical"), ranking[0])
    best_quantum = next((r for r in ranking if r.get("type") == "quantum"), None)

    # Headline metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Best Overall (Ensemble)", f"{best_ensemble['pr_auc']:.4f}", f"{best_ensemble['pr_auc'] - 0.5:+.4f} vs random")
    col2.metric("Best Classical (RF)", f"{best_classical['pr_auc']:.4f}", f"{best_classical['pr_auc'] - 0.5:+.4f} vs random")
    if best_quantum:
        col3.metric("Best Quantum (QSVC)", f"{best_quantum['pr_auc']:.4f}", f"{best_quantum['pr_auc'] - 0.5:+.4f} vs random")

    st.markdown("")

    # Two-column layout: table + chart side by side
    tab_table, tab_chart = st.tabs(["Leaderboard", "Chart"])

    with tab_table:
        rank_df = pd.DataFrame(ranking)
        if "fit_time" not in rank_df.columns and "fit_seconds" in rank_df.columns:
            rank_df["fit_time"] = rank_df["fit_seconds"]
        cols = [c for c in ["name", "type", "pr_auc", "accuracy", "fit_time"] if c in rank_df.columns]
        if cols:
            display_df = rank_df[cols].copy()
            display_df = display_df.rename(columns={
                "name": "Model", "type": "Type", "pr_auc": "PR-AUC",
                "accuracy": "Accuracy", "fit_time": "Time (s)",
            })
            fmt = {c: "{:.4f}" for c in ["PR-AUC", "Accuracy"] if c in display_df.columns}
            if "Time (s)" in display_df.columns:
                fmt["Time (s)"] = "{:.2f}"
            styled = display_df.style.format(fmt, na_rep="--")
            if "PR-AUC" in display_df.columns:
                styled = styled.background_gradient(subset=["PR-AUC"], cmap="RdYlGn", vmin=0.5, vmax=0.85)
            if "Accuracy" in display_df.columns:
                styled = styled.background_gradient(subset=["Accuracy"], cmap="RdYlGn", vmin=0.5, vmax=0.85)
            st.dataframe(styled, use_container_width=True, hide_index=True)

    with tab_chart:
        if "PR-AUC" in display_df.columns and "Model" in display_df.columns:
            chart_df = display_df[["Model", "PR-AUC"]].dropna().copy()
            if len(chart_df) > 0:
                def _model_color(name):
                    n = str(name).lower()
                    if "ensemble-qc" in n:
                        return "Ensemble"
                    if "qsvc" in n or "vqc" in n or "quantum" in n:
                        return "Quantum"
                    return "Classical"
                chart_df["Type"] = chart_df["Model"].apply(_model_color)
                chart_df = chart_df.sort_values("PR-AUC", ascending=True)
                fig = px.bar(
                    chart_df, x="PR-AUC", y="Model", color="Type", orientation="h",
                    color_discrete_map={"Classical": "#3b82f6", "Quantum": "#8b5cf6", "Ensemble": "#f59e0b"},
                    labels={"PR-AUC": "PR-AUC (higher is better)", "Model": ""},
                )
                fig.update_layout(
                    template="plotly_white",
                    height=max(280, 55 * len(chart_df)),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(range=[0.4, 0.85], dtick=0.05, gridcolor="#f1f5f9"),
                    yaxis=dict(gridcolor="#f1f5f9"),
                    font=dict(family="Inter, sans-serif", size=13),
                    margin=dict(l=0, r=20, t=30, b=20),
                )
                fig.update_traces(marker_line_width=0, opacity=0.92)
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Narrative insight cards
    st.markdown('<div class="section-label">Insights</div>', unsafe_allow_html=True)
    st.subheader("What the numbers tell us")
    ins1, ins2, ins3 = st.columns(3)
    with ins1:
        st.markdown(card(
            "Classical models are strong",
            "RandomForest alone reaches 0.7838. Rich pair features over 128D RotatE embeddings give tree ensembles a lot to work with.",
            accent="blue", stat="0.7838",
        ), unsafe_allow_html=True)
    with ins2:
        st.markdown(card(
            "Quantum adds ensemble signal",
            "QSVC alone (0.7216) does not beat classical. But the stacking ensemble reaches 0.7987 -- the quantum kernel captures complementary patterns.",
            accent="purple", stat="+0.06",
        ), unsafe_allow_html=True)
    with ins3:
        st.markdown(card(
            "Pauli feature map matters",
            "Switching from ZZ to Pauli boosted the ensemble from 0.7408 to 0.7987. The feature map determines how data is encoded into quantum states.",
            accent="amber", stat="0.7987",
        ), unsafe_allow_html=True)

    st.markdown("")
    with st.expander("Reproduce this run"):
        run_config = opt.get("config", {}) if opt else {}
        if run_config and run_config.get("relation"):
            cmd_parts = ["python", "scripts/run_optimized_pipeline.py"]
            cmd_parts.extend(["--relation", str(run_config.get("relation", "CtD"))])
            if run_config.get("full_graph_embeddings"):
                cmd_parts.append("--full_graph_embeddings")
            for flag, key in [
                ("--embedding_method", "embedding_method"), ("--embedding_dim", "embedding_dim"),
                ("--embedding_epochs", "embedding_epochs"), ("--negative_sampling", "negative_sampling"),
                ("--qml_dim", "qml_dim"), ("--qml_feature_map", "qml_feature_map"),
                ("--qml_feature_map_reps", "qml_feature_map_reps"), ("--qsvc_C", "qsvc_C"),
                ("--ensemble_method", "ensemble_method"), ("--qml_pre_pca_dim", "qml_pre_pca_dim"),
            ]:
                val = run_config.get(key)
                if val is not None and str(val) not in ("", "None", "False"):
                    cmd_parts.extend([flag, str(val)])
            for flag in ["optimize_feature_map_reps", "run_ensemble", "tune_classical", "fast_mode"]:
                if run_config.get(flag):
                    cmd_parts.append(f"--{flag}")
            cmd_parts.extend(["--results_dir", "results"])
            cmd_str = " \\\n  ".join([" ".join(cmd_parts[i:i+2]) for i in range(0, len(cmd_parts), 2)])
            st.code(cmd_str, language="bash")
        else:
            st.caption("Showing recommended best-run command.")
            st.code(BEST_RUN_COMMAND, language="bash")

# ============================================================================
# PAGE 4: WHAT WE LEARNED
# ============================================================================
elif page == "What We Learned":
    st.markdown('<div class="section-label">Experiments</div>', unsafe_allow_html=True)
    st.header("What We Learned")
    st.markdown(
        "We started at **PR-AUC ~0.60** and iterated to **0.7987** -- a 33% improvement. "
        "Here is what mattered, and what did not."
    )

    st.markdown("")
    st.subheader("Experiment log")
    st.caption("Each row is a variant of the pipeline. PR-AUC on the held-out test set.")
    exp_df = pd.DataFrame(BEST_RUN_EXPERIMENT_LOG)
    fmt = {c: "{:.4f}" for c in ["RF", "ET", "QSVC", "Ensemble"] if c in exp_df.columns}
    styled_exp = exp_df.style.format(fmt, na_rep="--")
    styled_exp = styled_exp.background_gradient(subset=["Ensemble"], cmap="RdYlGn", vmin=0.65, vmax=0.82)
    st.dataframe(styled_exp, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Key findings as styled cards
    st.markdown('<div class="section-label">Key Findings</div>', unsafe_allow_html=True)
    f1, f2 = st.columns(2)
    with f1:
        st.markdown(card(
            "Pauli feature map was the breakthrough",
            "Switching from ZZ to Pauli (reps=2) boosted the ensemble from 0.7408 to 0.7987 (+0.06). "
            "QSVC standalone <em>dropped</em>, but the ensemble improved -- the Pauli kernel is more complementary to classical models.",
            accent="green", stat="+0.06",
        ), unsafe_allow_html=True)
    with f2:
        st.markdown(card(
            "Stacking learns its own weights",
            "Manually setting ensemble_quantum_weight=0.4 had no effect. The stacking meta-learner "
            "already learns optimal weighting from data -- real model combination, not averaging.",
            accent="accent",
        ), unsafe_allow_html=True)

    f3, f4 = st.columns(2)
    with f3:
        st.markdown(card(
            "Hard negatives beat diverse negatives",
            "Diverse negatives (dw=0.5) dropped the ensemble from 0.7408 to 0.6919. Hard negatives "
            "force the model to learn more discriminative features.",
            accent="amber", stat="-0.05",
        ), unsafe_allow_html=True)
    with f4:
        st.markdown(card(
            "VQC remains a challenge",
            "Best VQC: 0.5474 (RealAmplitudes reps=4, SPSA). Barely above random. Consistent with "
            "known barren plateaus and limited expressivity at low qubit counts.",
            accent="blue", stat="0.5474",
        ), unsafe_allow_html=True)

    st.markdown("")
    with st.expander("VQC optimizer and ansatz details"):
        st.markdown("**Optimizer comparison (8 qubits, 50 iterations):**")
        vqc_opt = pd.DataFrame([
            {"Optimizer": "SPSA", "Test PR-AUC": 0.5456, "Train PR-AUC": 0.6048},
            {"Optimizer": "COBYLA", "Test PR-AUC": 0.5086, "Train PR-AUC": 0.6525},
            {"Optimizer": "NFT", "Test PR-AUC": 0.4782, "Train PR-AUC": 0.5248},
        ])
        st.dataframe(vqc_opt.style.format({"Test PR-AUC": "{:.4f}", "Train PR-AUC": "{:.4f}"}),
                      use_container_width=True, hide_index=True)
        st.markdown("**Ansatz comparison (SPSA, 50 iterations, 8 qubits):**")
        vqc_ansatz = pd.DataFrame([
            {"Ansatz": "RealAmplitudes reps=4", "Test PR-AUC": 0.5474, "Train PR-AUC": 0.5750, "Time (s)": 222},
            {"Ansatz": "RealAmplitudes reps=3", "Test PR-AUC": 0.5342, "Train PR-AUC": 0.5691, "Time (s)": 195},
            {"Ansatz": "EfficientSU2 reps=3", "Test PR-AUC": 0.5173, "Train PR-AUC": 0.6014, "Time (s)": 234},
            {"Ansatz": "RealAmplitudes reps=2", "Test PR-AUC": 0.5109, "Train PR-AUC": 0.6051, "Time (s)": 189},
            {"Ansatz": "EfficientSU2 reps=2", "Test PR-AUC": 0.5077, "Train PR-AUC": 0.5468, "Time (s)": 207},
            {"Ansatz": "TwoLocal reps=2", "Test PR-AUC": 0.5035, "Train PR-AUC": 0.5585, "Time (s)": 216},
            {"Ansatz": "TwoLocal reps=3", "Test PR-AUC": 0.4678, "Train PR-AUC": 0.5469, "Time (s)": 236},
        ])
        st.dataframe(vqc_ansatz.style.format({
            "Test PR-AUC": "{:.4f}", "Train PR-AUC": "{:.4f}", "Time (s)": "{:.0f}",
        }), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Progress</div>', unsafe_allow_html=True)
    st.subheader("Before and after")

    ba1, ba2 = st.columns(2)
    with ba1:
        st.markdown(card(
            "Pre-optimization baseline",
            "LogReg: 0.60 &nbsp; | &nbsp; QSVC: 0.65 &nbsp; | &nbsp; VQC: 0.49",
            accent="blue",
        ), unsafe_allow_html=True)
    with ba2:
        st.markdown(card(
            "Current best",
            "Ensemble (Pauli): <strong>0.7987</strong> &nbsp; | &nbsp; RF: 0.7838 &nbsp; | &nbsp; QSVC: 0.7216",
            accent="green", stat="0.7987",
        ), unsafe_allow_html=True)

    st.markdown("")
    st.caption("Improvement journey")
    st.markdown("""
```mermaid
graph LR
    A["Baseline<br/>LR: 0.60, QSVC: 0.65"] -->|"+RotatE 128D<br/>+hard negatives"| B["Mid<br/>RF: 0.77, QSVC: 0.72"]
    B -->|"+stacking<br/>+tune_classical"| C["v2<br/>Ensemble: 0.74"]
    C -->|"+Pauli feature map"| D["Best<br/>Ensemble: 0.7987"]
```
""")

# ============================================================================
# PAGE 5: TRY IT
# ============================================================================
elif page == "Try It":
    st.markdown('<div class="section-label">Run</div>', unsafe_allow_html=True)
    st.header("Try It")
    st.caption("Generate demo results to explore the dashboard, or run the full pipeline yourself.")

    st.markdown("")
    st.subheader("Pipeline preset")
    preset = st.radio(
        "Select a configuration",
        ["Best run (ensemble 0.7987)", "Quick (fewer epochs, 8 qubits)", "Custom command"],
        index=0, horizontal=True,
    )
    if preset == "Best run (ensemble 0.7987)":
        selected_cmd = BEST_RUN_COMMAND
        st.caption("Full-graph RotatE 128D (200 ep), 16 qubits, Pauli reps=2, stacking ensemble, GridSearchCV tuning.")
    elif preset == "Quick (fewer epochs, 8 qubits)":
        selected_cmd = QUICK_RUN_COMMAND
        st.caption("RotatE 128D (50 ep), 8 qubits, ZZ feature map, classical tuning.")
    else:
        selected_cmd = st.text_area("Edit command", value=BEST_RUN_COMMAND, height=150)

    st.code(selected_cmd, language="bash")
    st.markdown("---")

    col_demo, col_run, col_upload = st.columns(3)

    with col_demo:
        st.markdown("**Generate demo results**")
        st.caption("Populate the dashboard with best-run metrics (no pipeline execution).")
        if st.button("Generate demo results", type="primary"):
            demo_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            demo_payload = {
                "config": BEST_RUN_CONFIG,
                "ranking": BEST_RUN_RANKING,
                "classical_results": {
                    "RandomForest-Optimized": {"status": "success", "test_metrics": {"pr_auc": 0.7838, "accuracy": 0.7320}, "fit_seconds": 12.5},
                    "ExtraTrees-Optimized": {"status": "success", "test_metrics": {"pr_auc": 0.7807, "accuracy": 0.7280}, "fit_seconds": 9.8},
                },
                "quantum_results": {
                    "QSVC-Optimized": {"status": "success", "test_metrics": {"pr_auc": 0.7216, "accuracy": 0.6900}, "fit_seconds": 185.3},
                },
                "ensemble_results": {
                    "Ensemble-QC-stacking (Pauli)": {"status": "success", "test_metrics": {"pr_auc": 0.7987, "accuracy": 0.7450}, "fit_seconds": 210.1},
                    "Ensemble-QC-stacking (ZZ)": {"status": "success", "test_metrics": {"pr_auc": 0.7408, "accuracy": 0.7050}, "fit_seconds": 198.7},
                },
                "timestamp": demo_stamp,
            }
            demo_path = RESULTS_DIR / f"optimized_results_{demo_stamp}.json"
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(demo_path, "w") as f:
                json.dump(demo_payload, f, indent=2, default=str)
            demo_csv = pd.DataFrame([{
                "classical_pr_auc": 0.7838, "quantum_pr_auc": 0.7216,
                "classical_accuracy": 0.7320, "quantum_accuracy": 0.6900,
                "execution_mode": "simulator_statevector", "noise_model": "ideal",
                "backend_label": "ideal_sim", "qml_model_type": "QSVC",
                "qml_num_qubits": 16, "qml_feature_map_type": "PauliFeatureMap",
                "qml_relation": "CtD", "run_id": f"demo_{demo_stamp}",
            }])
            demo_csv.to_csv(LATEST_RUN, index=False)
            st.success("Demo results written. Switch to **Results** to view.")
            st.cache_data.clear()
            st.cache_resource.clear()
            time.sleep(0.5)
            st.rerun()

    with col_run:
        st.markdown("**Run pipeline**")
        st.caption("Requires torch, pykeen, qiskit.")
        run_subset = st.radio("Subset", ["Full", "Classical only", "Quantum only"], index=0, key="run_subset")
        extra = ""
        if run_subset == "Classical only":
            extra = " --classical_only"
        elif run_subset == "Quantum only":
            extra = " --quantum_only"
        if st.button("Run pipeline"):
            full_cmd = selected_cmd.replace("\\\n", " ").replace("  ", " ").strip() + extra + " --results_dir results"
            cmd_list = full_cmd.split()
            st.info(f"Running: `{' '.join(cmd_list[:6])} ...`")
            log_container = st.empty()
            rc, output = run_command(cmd_list, log_container)
            if rc == 0:
                st.success("Pipeline finished. Switch to **Results** to view.")
                st.cache_data.clear()
                st.cache_resource.clear()
            else:
                st.error(f"Pipeline exited with code {rc}.")

    with col_upload:
        st.markdown("**Upload results**")
        st.caption("Upload a saved `optimized_results_*.json`.")
        uploaded = st.file_uploader("Upload JSON", type=["json"], key="upload_json")
        if uploaded is not None:
            try:
                content = json.load(uploaded)
                upload_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                out_path = RESULTS_DIR / f"optimized_results_{upload_stamp}.json"
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(content, f, indent=2, default=str)
                st.success(f"Saved. Switch to **Results** to view.")
                st.cache_data.clear()
                st.cache_resource.clear()
            except Exception as e:
                st.error(f"Failed to parse JSON: {e}")

    st.markdown("---")
    with st.expander("Hyperparameter search (Optuna)"):
        st.markdown("""
```bash
python scripts/optuna_pipeline_search.py --n_trials 30 --objective ensemble
python scripts/optuna_pipeline_search.py --n_trials 20 --objective qsvc
python scripts/optuna_pipeline_search.py --n_trials 20 --objective classical
```
Results saved to `results/optuna/optuna_trials.csv` and `results/optuna/optuna_best.json`.
""")

# ============================================================================
# PAGE 6: TECHNICAL REFERENCE
# ============================================================================
elif page == "Technical Reference":
    st.markdown('<div class="section-label">Reference</div>', unsafe_allow_html=True)
    st.header("Technical Reference")
    st.caption("Detailed configuration, features, and terminology for technical users.")

    ref_tab1, ref_tab2, ref_tab3, ref_tab4 = st.tabs([
        "Pipeline Features",
        "Configuration Flags",
        "Glossary",
        "Documentation",
    ])

    with ref_tab1:
        feature_table = pd.DataFrame([
            {"Feature": "Quantum-classical ensemble", "Flag": "--run_ensemble --ensemble_method stacking", "Description": "Combines quantum + classical predictions (stacking or weighted average)"},
            {"Feature": "QSVC regularization", "Flag": "--qsvc_C 0.1", "Description": "Reduces quantum overfitting (default 1.0)"},
            {"Feature": "Kernel-target alignment", "Flag": "--optimize_feature_map_reps", "Description": "Auto-selects feature map reps by alignment score"},
            {"Feature": "Pauli feature map", "Flag": "--qml_feature_map Pauli", "Description": "Alternative to ZZ; best ensemble uses Pauli reps=2"},
            {"Feature": "Classical tuning", "Flag": "--tune_classical", "Description": "GridSearchCV over ET/RF/LR hyperparameters"},
            {"Feature": "Graph features in QML", "Flag": "--use_graph_features_in_qml", "Description": "Appends degree/neighbor features to quantum input"},
            {"Feature": "VQC configuration", "Flag": "--vqc_ansatz_type / --vqc_optimizer", "Description": "Configurable VQC ansatz and optimizer"},
            {"Feature": "Optuna HPO", "Flag": "scripts/optuna_pipeline_search.py", "Description": "Bayesian hyperparameter search over full pipeline"},
            {"Feature": "GPU simulator", "Flag": "--gpu", "Description": "GPU-accelerated quantum simulation via cuStateVec"},
        ])
        st.dataframe(feature_table, use_container_width=True, hide_index=True)

    with ref_tab2:
        config_table = pd.DataFrame([
            {"Flag": "--full_graph_embeddings", "Best-run": "Enabled", "Description": "Train on all Hetionet relations"},
            {"Flag": "--embedding_method", "Best-run": "RotatE", "Description": "KG embedding algorithm"},
            {"Flag": "--embedding_dim", "Best-run": "128", "Description": "Embedding dimensionality"},
            {"Flag": "--embedding_epochs", "Best-run": "200", "Description": "Embedding training epochs"},
            {"Flag": "--negative_sampling", "Best-run": "hard", "Description": "Negative sampling strategy"},
            {"Flag": "--qml_dim", "Best-run": "16", "Description": "Qubits / quantum feature dimension"},
            {"Flag": "--qml_feature_map", "Best-run": "Pauli", "Description": "Quantum feature map type"},
            {"Flag": "--qml_feature_map_reps", "Best-run": "2", "Description": "Feature map repetitions"},
            {"Flag": "--qsvc_C", "Best-run": "0.1", "Description": "QSVC regularization parameter"},
            {"Flag": "--ensemble_method", "Best-run": "stacking", "Description": "Ensemble strategy"},
            {"Flag": "--tune_classical", "Best-run": "Enabled", "Description": "GridSearchCV for classical models"},
            {"Flag": "--qml_pre_pca_dim", "Best-run": "24", "Description": "Pre-PCA dimensionality"},
        ])
        st.dataframe(config_table, use_container_width=True, hide_index=True)

        st.markdown("")
        st.markdown("**Best-run command:**")
        st.code(BEST_RUN_COMMAND, language="bash")

    with ref_tab3:
        for term_key in sorted(GLOSSARY.keys()):
            _expander_for_term(term_key)

    with ref_tab4:
        st.markdown("""
| Document | Description |
|----------|-------------|
| `docs/planning/NEXT_STEPS_TO_IMPROVE_PERFORMANCE.md` | Full experiment log, recommended commands, optimization roadmap |
| `docs/overview/IMPLEMENTATION_RECAP.md` | Pipeline improvements and GPU/hardware readiness summary |
| `docs/WHY_QUANTUM_UNDERPERFORMS.md` | Root cause analysis of the quantum-classical gap |
| `docs/OPTIMIZATION_PLAN.md` | Detailed optimization roadmap |
""")
