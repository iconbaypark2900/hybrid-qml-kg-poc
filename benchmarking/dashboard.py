# benchmarking/dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    page_title="Hybrid QML-KG Benchmark Dashboard",
    page_icon=None,
    layout="wide"
)

# Title and description
st.title("Hybrid QML-KG Biomedical Link Prediction")
st.markdown("""
This dashboard summarizes a hybrid quantum–classical knowledge graph pipeline for
drug–disease link prediction on **Hetionet**. It highlights what was built,
what was tested, and how quantum models compare to classical baselines.
""")

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

# Ensure local project modules (kg_layer/, quantum_layer/, etc.) are importable in Streamlit
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    """
    Loads the latest optimized_results_*.json (full ranking: LogisticRegression, RF, Ensemble, QSVC, Hybrid).
    Returns dict with keys ranking, classical_results, quantum_results, config, timestamp or None.
    Not cached so the Results tab always shows the latest run (pipeline may write to RESULTS_DIR or project results/).
    """
    def _debug_log(msg, data):
        for _path in [Path("/home/roc/quantumGlobalGroup/hybrid-qml-kg-poc/.cursor/debug.log"), RESULTS_DIR / "debug.log"]:
            try:
                _path.parent.mkdir(parents=True, exist_ok=True)
                open(_path, "a").write(json.dumps({"location": msg.get("location", ""), "message": msg.get("message", ""), "data": data, "timestamp": time.time(), **{k: v for k, v in msg.items() if k in ("hypothesisId", "hypothesisId2")}}) + "\n")
                break
            except Exception:
                continue
    try:
        dirs_to_scan = [RESULTS_DIR]
        if (PROJECT_ROOT / "results").resolve() != RESULTS_DIR.resolve():
            dirs_to_scan.append(PROJECT_ROOT / "results")
        # #region agent log
        _debug_log({"hypothesisId": "H1", "location": "load_optimized_results:entry", "message": "dirs_to_scan"}, {"results_dir": str(RESULTS_DIR), "dirs": [str(d) for d in dirs_to_scan]})
        # #endregion
        files = []
        for d in dirs_to_scan:
            if d.exists():
                files.extend(d.glob("optimized_results_*.json"))
        # #region agent log
        _debug_log({"hypothesisId": "H1", "hypothesisId2": "H2", "location": "load_optimized_results:files", "message": "files_found"}, {"count": len(files), "paths": [str(p) for p in files[:5]]})
        # #endregion
        if not files:
            return None
        latest = max(files, key=lambda p: p.stat().st_mtime)
        with open(latest, "r") as f:
            out = json.load(f)
        ranking = out.get("ranking") or []
        # Fallback: build ranking from classical_results + quantum_results if missing or empty
        if not ranking and ("classical_results" in out or "quantum_results" in out):
            ranking = []
            for name, res in (out.get("classical_results") or {}).items():
                if isinstance(res, dict) and res.get("status") == "success":
                    tm = res.get("test_metrics") or {}
                    ranking.append({
                        "name": name,
                        "type": "classical",
                        "pr_auc": tm.get("pr_auc", 0.0),
                        "accuracy": tm.get("accuracy", 0.0),
                        "fit_time": res.get("fit_seconds", 0.0),
                    })
            for name, res in (out.get("quantum_results") or {}).items():
                if isinstance(res, dict) and res.get("status") == "success":
                    tm = res.get("test_metrics") or {}
                    ranking.append({
                        "name": name,
                        "type": "quantum",
                        "pr_auc": tm.get("pr_auc", 0.0),
                        "accuracy": tm.get("accuracy", 0.0),
                        "fit_time": res.get("fit_seconds", 0.0),
                    })
            ranking.sort(key=lambda x: x.get("pr_auc", 0.0), reverse=True)
            out = dict(out)
            out["ranking"] = ranking
        # #region agent log
        _debug_log({"hypothesisId": "H3", "location": "load_optimized_results:success", "message": "loaded"}, {"file": str(latest), "keys": list(out.keys()), "ranking_len": len(ranking)})
        # #endregion
        return out
    except Exception as e:
        # #region agent log
        _debug_log({"hypothesisId": "H5", "location": "load_optimized_results:exception", "message": "exception"}, {"type": type(e).__name__, "msg": str(e)})
        # #endregion
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
    model_path = PROJECT_ROOT / "models" / "classical_logisticregression.joblib"
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


# Sidebar: Navigation
st.sidebar.title("Navigation")

with st.sidebar.expander("How this dashboard works"):
    st.markdown("**Recommended workflow**")
    st.markdown("""
1. **1. Overview** — Read what the project does (link prediction on Hetionet). No buttons.
2. **2. Run benchmarks** — Get data: run the pipeline, **Generate demo results**, or **Upload** CSV. Toggle **Classical only** / **Quantum only** if you want a subset. Then click **Run** (or **Run ideal then noisy**).
3. **3. Results** — View the latest run: full model ranking table and **Metrics by model** for every model. Use **Refresh full model ranking** to reload; use **Refresh data** (below) so charts and numbers update after a new run.
4. **4. Live prediction** — Pick **Classical** or **Quantum kernel similarity**; enter compound and disease; optionally check **Use config from latest run**. Click **Score this pair** or **Rank candidates**.
5. **5. Experiments** — Browse history: set **Max rows**, **Hide quantum=0** if needed; use **Download filtered history (CSV)** to export.
6. **6. Comparison** — Classical vs quantum across runs: adjust **Bootstrap seed** / **Bootstrap samples** and sliders under **Cost-aware recommendation**.
7. **7. Findings** — Inspect top predicted links and **Generate evidence bundle**.
8. **8–10** — Knowledge graph inventory, Hardware status, Run your code (advanced).
""")
    st.markdown("**Buttons and toggles**")
    st.markdown("""
- **Refresh data** (sidebar, below): Clears cache and reloads all results/charts. Use after running a benchmark or uploading so Overview and Results show the latest numbers.
- **Refresh full model ranking** (Results tab): Reloads the all-models table from `optimized_results_*.json`.
- **Run benchmarks**: **Generate demo results** = instant sample data; **Upload** = use your own `latest_run.csv` / `experiment_history.csv`. **Run** = start pipeline (needs torch, pykeen, qiskit).
- **Live prediction**: **Use config from latest run** = use qubits/reps/feature map from last benchmark; uncheck to set **Qubits**, **Feature map reps**, **Feature map**, **Entanglement** yourself.
""")
    st.markdown("**Data**")
    st.caption("Results and charts are cached (refreshed every 60s). Click **Refresh data** to see new runs immediately.")

with st.sidebar.expander("Glossary (key terms)"):
    for term_key in sorted(GLOSSARY.keys()):
        text = GLOSSARY[term_key]
        st.markdown(f"**{term_key.replace('_', ' ').title()}**")
        st.caption(text[:140] + "…" if len(text) > 140 else text)
        st.markdown("---")
if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# Order for new users: 1) What is this? 2) How to run 3) See results 4) Explore
page = st.sidebar.radio(
    "Go to",
    [
        "1. Overview (what is this?)",
        "2. Run benchmarks (get results)",
        "3. Results (latest run)",
        "4. Live prediction (interactive)",
        "5. Experiments (history)",
        "6. Comparison (classical vs quantum)",
        "7. Findings (ranked hypotheses)",
        "8. Knowledge graph (inventory)",
        "9. Hardware readiness (backend status)",
        "10. Run your code (advanced)",
    ]
)

# ==============================
# PAGE 0: PROJECT STORY
# ==============================
if page == "1. Overview (what is this?)":
    st.header("Overview: hybrid link prediction over the Hetionet biomedical knowledge graph")

    st.info("**New here?** Open **How this dashboard works** in the sidebar (above the tab list) for a step-by-step workflow, which buttons to use, and when to refresh results.")

    st.markdown("""
This project is a **hybrid quantum–classical link prediction** pipeline over the **Hetionet** biomedical knowledge graph.
It predicts whether a given **Compound** is likely to **treat** a given **Disease** (the **CtD** relation).

This is a research/prototyping system: it produces **ranking signals** and **benchmarks**, not clinical guidance.
""")
    _expander_for_term("link_prediction", "What is link prediction?")
    _expander_for_term("hetionet", "What is Hetionet?")
    _expander_for_term("ctd", "What is CtD (Compound–treats–Disease)?")

    st.subheader("Problem statement and task definition")
    st.caption("This defines *what* the pipeline optimizes for: predicting whether a given (compound, disease) pair is a known or plausible CtD link.")
    st.markdown("""
- **Input**: Hetionet triples (edges) and a target relation (e.g., CtD)
- **Task**: binary link prediction (edge exists vs not)
- **Output**: a probability/score used for ranking candidate edges
""")

    st.subheader("Representations and model inputs")
    st.markdown("""
- **Classical**: **Embeddings** plus derived pairwise features (concatenation, difference, element-wise product).
- **Quantum**: Reduced, quantum-ready vectors (size = number of **qubits**), used by **QSVC** / **VQC**.
""")
    _expander_for_term("embedding", "What is an embedding?")
    _expander_for_term("qubit", "What is a qubit?")
    _expander_for_term("qsvc", "What is QSVC?")
    _expander_for_term("vqc", "What is VQC?")

    st.subheader("Implemented components (end-to-end)")
    st.caption("Artifacts and scripts that make up the pipeline; all are used by the **Run benchmarks** and **Live prediction** tabs.")
    st.markdown("""
- Embedding training artifacts in `data/` (multiple embedding families supported)
- Classical baseline training artifacts in `models/` (model + scaler)
- Quantum execution modes (ideal/noisy simulator + hardware option)
- Benchmark scripts and experiment history logging
""")

    st.subheader("Benchmarking context and current status")
    st.caption("Data is cached 60s. After running a benchmark, click **Refresh data** in the sidebar to update Overview and Results.")
    with st.expander("What do these fields mean?"):
        st.markdown("""
        - **classical_pr_auc / quantum_pr_auc**: Link-prediction quality (higher = better ranking). PR-AUC is preferred for imbalanced data.
        - **execution_mode**: How the quantum circuit was run (e.g. simulator vs hardware).
        - **noise_model**: Whether the run used an ideal (noiseless) or noisy simulation.
        - **backend_label**: The actual backend name (e.g. ideal_sim, noisy_sim, or a device name).
        - **qml_model_type**: Quantum model used (e.g. QSVC, VQC).
        - **qml_num_qubits**: Number of qubits used for the feature map; affects capacity and runtime.
        """)
    if df_latest is None:
        st.warning("No `results/latest_run.csv` found yet. Run a benchmark from “Run Benchmarks”.")
    else:
        st.json({
            "classical_pr_auc": float(df_latest["classical_pr_auc"].iloc[0]),
            "quantum_pr_auc": float(df_latest["quantum_pr_auc"].iloc[0]),
            "execution_mode": str(safe_get(df_latest, "execution_mode", "N/A")),
            "noise_model": str(safe_get(df_latest, "noise_model", "N/A")),
            "backend_label": str(safe_get(df_latest, "backend_label", "N/A")),
            "qml_model_type": str(safe_get(df_latest, "qml_model_type", "N/A")),
            "qml_num_qubits": str(safe_get(df_latest, "qml_num_qubits", "N/A")),
        })

    if df_history is not None and len(df_history) > 0:
        st.subheader("Ideal vs noisy snapshot (latest per mode)")
        st.caption(
            "One row per (execution_mode, noise_model, backend_label): run_index = experiment row; "
            "quantum_pr_auc / classical_pr_auc = latest scores for that mode. Use this to compare noiseless vs noisy performance."
        )
        exec_summary = latest_execution_summary(df_history)
        if not exec_summary.empty:
            st.dataframe(exec_summary)
        else:
            st.info("No execution metadata columns found in history yet.")

    st.subheader("How to interpret scores and claims")
    st.markdown("""
- **PR-AUC**: best for imbalanced link prediction; higher is better.
- **Ideal vs noisy**: shows sensitivity to noise model / backend.
- **A single score is not evidence**: use the “Evidence” section in Live Prediction.
""")
    _expander_for_term("pr_auc", "What is PR-AUC?")
    _expander_for_term("ideal_vs_noisy", "What does ideal vs noisy mean?")

    st.subheader("Limitations and next steps")
    st.caption("Planned improvements for future work (not current bugs).")
    st.markdown("""
- Add candidate ranking workflows (“top compounds for a disease”)
- Add evidence/interpretability (neighbors + KG context)
- Add robust evaluation (seeds/CV) and artifact linking per run
""")

# ==============================
# PAGE 1: RESULTS OVERVIEW
# ==============================
elif page == "3. Results (latest run)":
    st.header("Results: latest run summary")
    st.caption("This tab summarizes the **most recent** benchmark run and compares classical vs quantum metrics. If you see no data, run a benchmark from the **Run benchmarks** tab (or upload results). Data is cached for 60s—use **Refresh data** in the sidebar to see new runs immediately.")

    st.markdown("""
**What was built**
- End-to-end KG pipeline: data ingestion, embedding training, feature construction
- Quantum and classical link prediction models with reproducibility controls
- Benchmarking for ideal vs noisy simulators with run metadata logging
- Experiment history tracking and comparison utilities

**What was accomplished**
- Automated feature hygiene and PCA stability for quantum inputs
- Centralized seed control for consistent runs
- Full-graph embedding option and richer feature engineering
- Ideal vs noisy simulator runs with side-by-side comparisons
""")

    # Load full run once for snapshot and later for ranking table
    opt = load_optimized_results()

    st.subheader("Latest run snapshot")
    st.caption(
        "**QML Model**: quantum algorithm (QSVC or VQC). **Qubits**: feature-map width. "
        "**Feature Map**: encoding circuit (e.g. ZZFeatureMap). **Execution**: how the circuit was run (simulator/hardware)."
    )
    run_cols = st.columns(4)
    qml_model_type = safe_get(df_latest, "qml_model_type", "N/A")
    qml_num_qubits = safe_get(df_latest, "qml_num_qubits", "N/A")
    qml_feature_map = safe_get(df_latest, "qml_feature_map_type", "N/A")
    exec_mode = safe_get(df_latest, "execution_mode", "N/A")
    noise_model = safe_get(df_latest, "noise_model", "N/A")
    backend_label = safe_get(df_latest, "backend_label", "N/A")
    if pd.isna(noise_model) or noise_model == "" or str(noise_model).lower() == "nan":
        noise_model = "N/A"
    if pd.isna(backend_label) or backend_label == "" or str(backend_label).lower() == "nan":
        backend_label = "N/A"

    run_cols[0].metric("QML Model", qml_model_type)
    run_cols[1].metric("Qubits", qml_num_qubits)
    run_cols[2].metric("Feature Map", qml_feature_map)
    run_cols[3].metric("Execution", exec_mode)

    st.caption(f"**Backend:** {backend_label} · **Noise:** {noise_model}. "
               "Backend = where the quantum circuit runs; noise = whether we simulate real-device errors.")

    # Show full run: all classical, hybrid, and quantum models when we have full ranking
    if opt and opt.get("ranking"):
        ranking_list = opt["ranking"]
        classical_names = [r.get("name") for r in ranking_list if r.get("type") == "classical"]
        # Pipeline labels Hybrid as type "quantum"; treat name containing "Hybrid" as hybrid for display
        hybrid_names = [r.get("name") for r in ranking_list if "Hybrid" in (r.get("name") or "")]
        quantum_names = [r.get("name") for r in ranking_list if r.get("type") == "quantum" and r.get("name") not in hybrid_names]
        parts = []
        if classical_names:
            parts.append(f"**Classical:** {', '.join(classical_names)}")
        if quantum_names:
            parts.append(f"**Quantum:** {', '.join(quantum_names)}")
        if hybrid_names:
            parts.append(f"**Hybrid:** {', '.join(hybrid_names)}")
        if parts:
            st.markdown("**Models in this run:** " + "  |  ".join(parts))

    st.header("Model Performance Comparison")
    with st.expander("Understanding these metrics"):
        st.markdown(GLOSSARY["pr_auc"])
        st.markdown("---")
        st.markdown(GLOSSARY["accuracy"])
        st.markdown("---")
        st.markdown(GLOSSARY["parameters"])

    # Full model ranking (from optimized_results_*.json when pipeline ran classical + quantum; opt already loaded above)
    if st.button("Refresh full model ranking", key="refresh_optimized_results"):
        st.rerun()
    # #region agent log
    try:
        _lp = Path("/home/roc/quantumGlobalGroup/hybrid-qml-kg-poc/.cursor/debug.log")
        open(_lp, "a").write(json.dumps({"hypothesisId": "H3", "hypothesisId2": "H4", "location": "Results_tab:after_load", "message": "opt_state", "data": {"opt_is_none": opt is None, "ranking_len": len(opt.get("ranking", [])) if opt else 0}, "timestamp": time.time()}) + "\n")
    except Exception:
        pass
    # #endregion
    if opt and opt.get("ranking"):
        st.subheader("Full model ranking (latest run)")
        st.caption("All models from the last benchmark run. Same order as the pipeline terminal output.")
        rank_df = pd.DataFrame(opt["ranking"])
        # Prefer fit_time; fallback to fit_seconds (some payloads use either)
        if "fit_time" not in rank_df.columns and "fit_seconds" in rank_df.columns:
            rank_df = rank_df.copy()
            rank_df["fit_time"] = rank_df["fit_seconds"]
        cols = [c for c in ["name", "type", "pr_auc", "accuracy", "fit_time"] if c in rank_df.columns]
        if cols:
            display_df = rank_df[cols].copy()
            display_df = display_df.rename(columns={"name": "Model", "type": "Type", "pr_auc": "PR-AUC", "accuracy": "Accuracy", "fit_time": "Time (s)"})
            # Highlight numbers for audience: format decimals and color PR-AUC/Accuracy (higher = better)
            format_map = {c: "{:.4f}" for c in ["PR-AUC", "Accuracy"] if c in display_df.columns}
            if "Time (s)" in display_df.columns:
                format_map["Time (s)"] = "{:.2f}"
            styled = display_df.style.format(format_map, na_rep="—")
            if "PR-AUC" in display_df.columns:
                styled = styled.background_gradient(subset=["PR-AUC"], cmap="RdYlGn", vmin=0, vmax=1)
            if "Accuracy" in display_df.columns:
                styled = styled.background_gradient(subset=["Accuracy"], cmap="RdYlGn", vmin=0, vmax=1)
            st.dataframe(styled, use_container_width=True)
        best = opt.get("ranking", [])
        if best:
            best_overall = best[0]
            best_classical = next((r for r in best if r.get("type") == "classical"), best[0])
            best_quantum = next((r for r in best if r.get("type") == "quantum"), None)
            st.caption(
                f"**Best overall:** {best_overall.get('name', 'N/A')} — PR-AUC {best_overall.get('pr_auc', 0):.4f}  |  "
                f"**Best classical:** {best_classical.get('name', 'N/A')} — PR-AUC {best_classical.get('pr_auc', 0):.4f}"
                + (f"  |  **Best quantum:** {best_quantum.get('name', 'N/A')} — PR-AUC {best_quantum.get('pr_auc', 0):.4f}" if best_quantum else "")
            )
        if opt.get("timestamp"):
            st.caption(f"Run timestamp: {opt['timestamp']}")
        # Metrics by model: show PR-AUC, Accuracy, Time (s) for every model (not just classical/quantum buckets)
        st.subheader("Metrics by model")
        st.caption("PR-AUC, Accuracy, and Time for each model from the latest run. All models are shown so the audience can compare any model.")
        ranking_list = opt.get("ranking", [])
        n_models = len(ranking_list)
        if n_models > 0:
            n_cols = 3  # 3 columns, multiple rows so each model gets space
            for start in range(0, n_models, n_cols):
                chunk = ranking_list[start : start + n_cols]
                cols = st.columns(len(chunk))
                for idx, r in enumerate(chunk):
                    with cols[idx]:
                        name = r.get("name", "—")
                        pr_auc = r.get("pr_auc")
                        acc = r.get("accuracy")
                        fit_t = r.get("fit_time", r.get("fit_seconds"))
                        st.markdown(f"**{name}**")
                        st.metric("PR-AUC", f"{pr_auc:.4f}" if pr_auc is not None and not pd.isna(pr_auc) else "—", None)
                        st.metric("Accuracy", f"{acc:.4f}" if acc is not None and not pd.isna(acc) else "—", None)
                        st.metric("Time (s)", f"{fit_t:.2f}" if fit_t is not None and not pd.isna(fit_t) else "—", None)
                        st.markdown("---")
        st.markdown("---")
    else:
        st.caption("_Full model ranking (all 6 models) appears here after you run a benchmark with **Run benchmarks** (classical + quantum). The pipeline writes `optimized_results_*.json`; if you ran locally, upload that run's results or run from the dashboard._")
        st.markdown("---")

    if df_latest is not None:
        # Extract metrics (single classical / single quantum from latest_run.csv; kept for when no full ranking)
        classical_pr_auc = df_latest['classical_pr_auc'].iloc[0]
        quantum_pr_auc = df_latest['quantum_pr_auc'].iloc[0]
        classical_params = df_latest['classical_num_parameters'].iloc[0]
        quantum_params = df_latest['quantum_num_parameters'].iloc[0]
        classical_acc = df_latest['classical_accuracy'].iloc[0]
        quantum_acc = df_latest['quantum_accuracy'].iloc[0]
        
        # Show aggregate classical/quantum metrics only when we don't already show full per-model metrics above
        if not (opt and opt.get("ranking")):
            st.subheader("Latest run metrics (classical vs quantum)")
            st.caption("Single representative classical and quantum model from `latest_run.csv`. For all models, run a full benchmark to see the **Full model ranking** and **Metrics by model** above.")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="PR-AUC (Classical)",
                value=f"{classical_pr_auc:.4f}",
                delta=None
            )
            st.metric(
                label="Accuracy (Classical)",
                value=f"{classical_acc:.4f}",
                delta=None
            )
        
        with col2:
            st.metric(
                label="PR-AUC (Quantum)",
                value=f"{quantum_pr_auc:.4f}",
                delta=f"{quantum_pr_auc - classical_pr_auc:.4f}" if not pd.isna(quantum_pr_auc) else None,
                delta_color="normal"
            )
            st.metric(
                label="Accuracy (Quantum)",
                value=f"{quantum_acc:.4f}",
                delta=f"{quantum_acc - classical_acc:.4f}" if not pd.isna(quantum_acc) else None,
                delta_color="normal"
            )
        
        with col3:
            st.metric(
                label="Classical Parameters",
                value=int(classical_params) if not pd.isna(classical_params) else "N/A"
            )
            st.metric(
                label="Quantum Parameters",
                value=int(quantum_params) if not pd.isna(quantum_params) else "N/A",
                delta=int(quantum_params - classical_params) if (not pd.isna(quantum_params) and not pd.isna(classical_params)) else None,
                delta_color="inverse"  # Green if fewer params
            )
        
        # Outcome highlights
        st.subheader("Outcome Highlights")
        st.caption(
            "**PR-AUC Delta (Q − C)**: how much better (positive) or worse (negative) the quantum model is vs classical. "
            "**Param Ratio (Q/C)**: quantum vs classical parameter count; under 1 means a more parameter-efficient quantum model. "
            "**History Runs**: total experiments logged for comparison."
        )
        if not pd.isna(quantum_pr_auc) and not pd.isna(classical_pr_auc):
            delta_pr = quantum_pr_auc - classical_pr_auc
        else:
            delta_pr = None
        if not pd.isna(quantum_params) and not pd.isna(classical_params) and classical_params != 0:
            param_ratio = quantum_params / classical_params
        else:
            param_ratio = None
        highlight_cols = st.columns(3)
        highlight_cols[0].metric("PR-AUC Delta (Q - C)", f"{delta_pr:.4f}" if delta_pr is not None else "N/A")
        highlight_cols[1].metric("Param Ratio (Q/C)", f"{param_ratio:.2f}x" if param_ratio is not None else "N/A")
        highlight_cols[2].metric("History Runs", len(df_history) if df_history is not None else 0)
        if df_history is not None and "quantum_pr_auc" in df_history.columns:
            quantum_nonzero = int((df_history["quantum_pr_auc"] > 0).sum())
            st.caption(f"Quantum runs with PR-AUC > 0: {quantum_nonzero}/{len(df_history)}")

        st.subheader("Latest Run Details")
        st.caption("Exact configuration of the last benchmark: execution environment, quantum model type, qubits, feature map, and target relation (e.g. CtD).")
        st.json({
            "execution_mode": safe_get(df_latest, "execution_mode", "N/A"),
            "noise_model": safe_get(df_latest, "noise_model", "N/A"),
            "backend_label": safe_get(df_latest, "backend_label", "N/A"),
            "qml_model_type": safe_get(df_latest, "qml_model_type", "N/A"),
            "qml_num_qubits": safe_get(df_latest, "qml_num_qubits", "N/A"),
            "qml_feature_map_type": safe_get(df_latest, "qml_feature_map_type", "N/A"),
            "relation": safe_get(df_latest, "qml_relation", "N/A"),
        })

        # Parameter efficiency explanation
        st.info("""
        **Why Parameter Efficiency Matters**:  
        Quantum models use exponentially fewer parameters as the knowledge graph scales.  
        This suggests **superior scalability** even if classical models win on small datasets today.
        """)
        
        # Scaling projection
        st.subheader("Algorithmic Scaling Advantage")
        if SCALING_PLOT.exists():
            st.caption("Theoretical or simulated scaling: how runtime or cost grows as the knowledge graph (or embedding size) increases. Relevant for planning larger deployments.")
            st.image(str(SCALING_PLOT), caption="Projected runtime as KG size increases", use_column_width=True)
        else:
            st.warning("Scaling projection plot not found. Run benchmarking/scalability_sim.py to generate it.")
        
        # Ideal vs noisy summary (if available)
        st.subheader("Ideal vs Noisy Summary")
        st.caption("Latest score per (execution_mode, noise_model, backend). Compare rows to see how noise affects quantum PR-AUC.")
        exec_summary = latest_execution_summary(df_history)
        if not exec_summary.empty:
            st.dataframe(exec_summary)
        else:
            st.info("Run the benchmark script to populate ideal vs noisy comparisons.")

        # Model configuration
        st.subheader("Quantum Model Configuration")
        st.caption("All `qml_*` keys from the latest run: model type, qubits, feature map, relation, and any other logged quantum settings. These values are used by **Live prediction** when \"Use config from latest benchmark run\" is checked.")
        qml_config = {}
        for col in df_latest.columns:
            if col.startswith('qml_'):
                key = col.replace('qml_', '')
                value = df_latest[col].iloc[0]
                qml_config[key] = value
                st.text(f"{key}: {value}")
    
    else:
        st.error("No results found. Use the **Run benchmarks** tab to run the pipeline, or **Generate demo results** / **Upload** a CSV from a local run.")
        st.markdown("""
        To generate results from the terminal:
        1. `bash scripts/benchmark_ideal_noisy.sh CtD results --fast_mode --quantum_only`
        2. `python benchmarking/ideal_vs_noisy_compare.py --results_dir results`
        """)

# ==============================
# PAGE 2: LIVE PREDICTION
# ==============================
elif page == "4. Live prediction (interactive)":
    st.header("Live prediction: interactive scoring and candidate ranking")
    st.markdown("""
    Enter a **compound** (drug) and **disease** to get a **link prediction score**: how likely this pair is to be a “treats” link.
    You can use the **classical model** (trained logistic regression) or **quantum kernel similarity** (quantum circuit–based similarity, not a probability).
    Results include **candidate rankings** and **evidence** (nearest neighbors, KG context).

    This is a research demo and does **not** provide clinical or synthesis guidance.
    """)
    _expander_for_term("kernel_similarity", "What is quantum kernel similarity?")
    
    model, scaler, model_path, scaler_path = load_classical_artifacts()
    embeddings, entity_ids, emb_name = load_entity_embeddings()

    st.caption(
        f"Classical model: `{model_path}` | Scaler: `{scaler_path}` | Embeddings: `{emb_name}`. "
        "**Classical** uses the model from the last benchmark (run with **Run classical**). "
        "**Quantum** uses the same qubit count and feature map as the last run (run with **Run quantum**)."
    )

    # Initialize/update state from query params BEFORE widgets are instantiated.
    qp_drug = st.query_params.get("drug", "DB00688")
    qp_disease = st.query_params.get("disease", "DOID_0060048")
    if "drug_input" not in st.session_state:
        st.session_state["drug_input"] = qp_drug
    if "disease_input" not in st.session_state:
        st.session_state["disease_input"] = qp_disease
    # If query params changed (e.g., user clicked an example), update the widget state here.
    if st.session_state.get("drug_input") != qp_drug:
        st.session_state["drug_input"] = qp_drug
    if st.session_state.get("disease_input") != qp_disease:
        st.session_state["disease_input"] = qp_disease

    # Prefer selecting only valid IDs from the loaded embedding set.
    use_dropdown = st.checkbox("Select only valid IDs from embeddings (recommended)", value=True)

    # Build options for dropdowns if available
    compounds_list, diseases_list = ([], [])
    if entity_ids is not None:
        compounds_list, diseases_list = split_entities(entity_ids)

    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            if use_dropdown and compounds_list:
                default_compound = normalize_compound_id(st.session_state["drug_input"])
                if default_compound not in compounds_list:
                    default_compound = compounds_list[0]
                drug_input = st.selectbox(
                    "Compound (from embeddings)",
                    options=compounds_list,
                    index=compounds_list.index(default_compound),
                    key="drug_select"
                )
            else:
                drug_input = st.text_input("Drug", key="drug_input", help="e.g., DB00688 (Dexamethasone)")

        with col2:
            if use_dropdown and diseases_list:
                default_disease = normalize_disease_id(st.session_state["disease_input"])
                if default_disease not in diseases_list:
                    default_disease = diseases_list[0]
                disease_input = st.selectbox(
                    "Disease (from embeddings)",
                    options=diseases_list,
                    index=diseases_list.index(default_disease),
                    key="disease_select"
                )
            else:
                disease_input = st.text_input("Disease", key="disease_input", help="e.g., DOID_0060048 (COVID-19)")
        
        method = st.selectbox(
            "Scoring Method",
            ["classical_model", "quantum_kernel_similarity"],
            index=0,
            help="Classical = trained classifier from last benchmark. Quantum = kernel similarity; you can override qubits, feature map, and entanglement below.",
        )
        top_k = st.slider("Top-K candidates", min_value=5, max_value=50, value=15, step=5, help="How many top compounds (for this disease) and top diseases (for this compound) to show in the ranking tables after you predict.")

        with st.expander("What do the prediction flags do?"):
            st.markdown("""
            | Flag | Effect | When it applies |
            |------|--------|-----------------|
            | **Use config from latest run** | Use qubits, reps, and feature map from `results/latest_run.csv` so Live prediction matches the last benchmark. | Always; uncheck to use custom values below. |
            | **Qubits (feature dimension)** | PCA reduces embeddings to this many dimensions; same dim is used for pair features (and quantum feature map). Classical model expects the dimension it was trained with. | Quantum path uses your override when \"Use latest run\" is unchecked; classical always uses latest-run dim. |
            | **Feature map reps** | Number of repetition layers in the quantum feature map (more = deeper circuit). | Quantum only. |
            | **Feature map type** | **ZZ** = two-qubit rotations (entangling); **Z** = single-qubit only. | Quantum only. |
            | **Entanglement** | Qubit connectivity for ZZ map: **linear**, **full**, or **circular**. | Quantum only (ZZ feature map). |
            """)
        st.markdown("**Prediction flags (quantum and feature dimension)**")
        use_latest_config = st.checkbox(
            "Use config from latest benchmark run",
            value=True,
            help="When checked, qubits and feature map settings come from results/latest_run.csv. When unchecked, use the custom values below.",
        )
        # Defaults from latest run for display/fallback
        _latest_qubits = int(safe_get(df_latest, "qml_num_qubits", 12) or 12) if df_latest is not None else 12
        _latest_reps = int(safe_get(df_latest, "qml_feature_map_reps", 2) or 2) if df_latest is not None else 2
        override_qubits = st.number_input(
            "Qubits (feature dimension)",
            min_value=2,
            max_value=20,
            value=_latest_qubits,
            step=1,
            help="Number of qubits = PCA dimension for pair features. Classical model expects the dimension it was trained with (from latest run); override applies to quantum path and to PCA when using quantum.",
        )
        override_reps = st.number_input(
            "Feature map reps",
            min_value=1,
            max_value=6,
            value=_latest_reps,
            step=1,
            help="Feature map repetition layers (quantum only). More reps = deeper circuit, richer encoding.",
        )
        override_fm_type = st.selectbox(
            "Feature map type",
            ["ZZ", "Z"],
            index=0,
            help="ZZ = two-qubit rotations (entangling); Z = single-qubit only (quantum only).",
        )
        override_entanglement = st.selectbox(
            "Entanglement",
            ["linear", "full", "circular"],
            index=0,
            help="Qubit connectivity for ZZ feature map: linear, full, or circular (quantum only).",
        )
        if use_latest_config:
            st.caption("Using qubits and reps from latest run. Overrides above apply when you uncheck \"Use config from latest benchmark run\".")

        st.markdown("**Error mitigation (quantum only)**")
        with st.expander("What is error mitigation?", expanded=False):
            st.markdown("""
            - **Statevector**: Noiseless simulation (no shots, no mitigation). Fast; use for quick checks.
            - **Shot-based (ideal)**: Finite measurement shots, no noise model. Simulates sampling without device errors.
            - **Shot-based (noisy)**: Simulator with a noise model (e.g. depolarizing). Use **ZNE** to extrapolate to zero noise from several noise levels (slower, more accurate under noise).
            """)
        kernel_execution = st.radio(
            "Kernel execution",
            ["Statevector (noiseless, fast)", "Shot-based (ideal config)", "Shot-based (noisy config)"],
            index=0,
            help="Statevector = exact; Shot-based uses config/quantum_config_ideal.yaml or quantum_config_noisy.yaml and enables mitigation options.",
        )
        shots_override = st.number_input(
            "Shots (shot-based only)",
            min_value=256,
            max_value=8192,
            value=1024,
            step=256,
            help="Measurement shots per kernel evaluation when using shot-based execution.",
        )
        apply_zne = st.checkbox(
            "Apply ZNE to main prediction (noisy only)",
            value=False,
            help="Evaluate kernel at 3 noise scales and extrapolate to zero noise for the single drug–disease score. Slower; only when Shot-based (noisy) is selected.",
        )
        if apply_zne and not kernel_execution.startswith("Shot-based (noisy"):
            st.caption("ZNE applies only when **Shot-based (noisy config)** is selected.")

        submitted = st.form_submit_button("Predict")

    # If an example button was clicked, auto-run once without requiring an extra click.
    if st.session_state.get("auto_predict", False):
        submitted = True
        st.session_state["auto_predict"] = False
    
    if submitted:
        if embeddings is None or entity_ids is None:
            st.error("Missing embeddings. Run embedding training to generate files in `data/`.")
        else:
            with st.spinner("Scoring and building evidence..."):
                entity_to_idx = {eid: i for i, eid in enumerate(entity_ids) if isinstance(eid, str)}

                # When using dropdowns, values are already full IDs
                compound_id = drug_input if (use_dropdown and isinstance(drug_input, str) and drug_input.startswith("Compound::")) else normalize_compound_id(drug_input)
                disease_id = disease_input if (use_dropdown and isinstance(disease_input, str) and disease_input.startswith("Disease::")) else normalize_disease_id(disease_input)
                c_idx = entity_to_idx.get(compound_id)
                d_idx = entity_to_idx.get(disease_id)

                if c_idx is None:
                    st.error(f"Compound not found in embeddings: {compound_id}")
                    st.info("This embedding set contains a sampled subset of Hetionet entities. Try one of these available compounds:")
                    suggestions = suggest_available(entity_ids, "Compound::", compound_id.split("::")[-1], k=15)
                    st.code("\n".join(suggestions))
                    st.stop()
                elif d_idx is None:
                    st.error(f"Disease not found in embeddings: {disease_id}")
                    st.info("Try one of these available diseases:")
                    suggestions = suggest_available(entity_ids, "Disease::", disease_id.split("::")[-1], k=15)
                    st.code("\n".join(suggestions))
                    st.stop()
                else:
                    # Resolve flags: classical always uses latest-run dim (model expects it); quantum uses latest or overrides
                    latest_qubits = int(safe_get(df_latest, "qml_num_qubits", 12) or 12) if df_latest is not None else 12
                    latest_reps = int(safe_get(df_latest, "qml_feature_map_reps", 2) or 2) if df_latest is not None else 2
                    if method == "classical_model":
                        qml_dim = latest_qubits  # model was trained with this dimension
                        qml_reps = latest_reps
                        qml_fm_type = "ZZ"
                        qml_entanglement = "linear"
                    else:
                        qml_dim = latest_qubits if use_latest_config else int(override_qubits)
                        qml_reps = latest_reps if use_latest_config else int(override_reps)
                        if use_latest_config and df_latest is not None:
                            _fm = str(safe_get(df_latest, "qml_feature_map_type", "ZZ") or "ZZ").upper()
                            qml_fm_type = "Z" if _fm == "Z" else "ZZ"
                            _ent = str(safe_get(df_latest, "qml_entanglement", "linear") or "linear").lower()
                            qml_entanglement = _ent if _ent in ("linear", "full", "circular") else "linear"
                        else:
                            qml_fm_type = override_fm_type
                            qml_entanglement = override_entanglement
                    pca = fit_pca_reducer(embeddings, qml_dim)
                    reduced_all = pca.transform(embeddings).astype(np.float32)

                    c_red = reduced_all[c_idx]
                    d_red = reduced_all[d_idx]
                    payload = {
                        "compound_id": compound_id,
                        "disease_id": disease_id,
                        "embedding": emb_name,
                        "qml_dim_used_for_features": qml_dim,
                    }

                    qk_for_ranking = None  # set below for quantum path; used in candidate ranking
                    if method == "classical_model":
                        if model is None or scaler is None:
                            st.error("Missing trained classical model/scaler. Run the pipeline to generate `models/*.joblib`.")
                            st.stop()
                        X_pair = build_pair_features(c_red, d_red)  # (1, 48) if qml_dim=12
                        X_scaled = scaler.transform(X_pair)
                        prob = float(model.predict_proba(X_scaled)[0, 1])
                        st.success("Prediction computed (classical model)")
                        payload.update({
                            "score_type": "probability",
                            "link_probability": round(prob, 6),
                            "features_dim": int(X_pair.shape[1]),
                            "model_used": "classical_logisticregression",
                        })
                        st.json(payload)
                    else:
                        # Quantum path: statevector vs shot-based, optional ZNE
                        use_statevector = kernel_execution.startswith("Statevector")
                        use_shot_noisy = kernel_execution.startswith("Shot-based (noisy")
                        use_shot_ideal = kernel_execution.startswith("Shot-based (ideal")
                        x1_q = to_quantum_input(c_red).reshape(1, -1)
                        x2_q = to_quantum_input(d_red).reshape(1, -1)

                        if use_shot_noisy and apply_zne:
                            # ZNE for main prediction only: evaluate at 3 scales, extrapolate
                            base_noise_p = 0.01  # from config depolarizing:0.01
                            raw_at_1, mitigated, zne_info = evaluate_kernel_zne(
                                x1_q, x2_q, qml_dim, qml_reps, qml_fm_type, qml_entanglement,
                                base_noise_p, int(shots_override), scales=[1.0, 1.5, 2.0],
                            )
                            sim = mitigated if not np.isnan(mitigated) else raw_at_1
                            st.success("Quantum similarity computed (ZNE mitigated)")
                            fm_desc = f"{qml_fm_type}FeatureMap(reps={qml_reps}, entanglement={qml_entanglement})"
                            payload.update({
                                "score_type": "kernel_similarity",
                                "quantum_kernel_similarity": round(sim, 6),
                                "quantum_kernel_raw": round(raw_at_1, 6),
                                "zne_mitigated": round(mitigated, 6),
                                "zne_scales": zne_info.get("scales", []),
                                "zne_values": zne_info.get("values", []),
                                "feature_map": fm_desc,
                                "qml_dim_used": qml_dim,
                                "qml_reps_used": qml_reps,
                                "feature_map_type": qml_fm_type,
                                "entanglement": qml_entanglement,
                                "execution": "shot_noisy_zne",
                                "shots": int(shots_override),
                                "note": "Kernel similarity (ZNE extrapolated to zero noise). Not a calibrated probability.",
                            })
                            st.json(payload)
                            if "values" in zne_info:
                                st.caption(f"ZNE: raw at scale 1.0 = {raw_at_1:.4f}; at scales {zne_info['scales']} = {[round(v, 4) for v in zne_info['values']]}; mitigated (λ→0) = {mitigated:.4f}.")
                        else:
                            # Statevector or shot-based without ZNE for this single eval
                            if use_statevector:
                                qk, err = load_quantum_kernel(
                                    qml_dim=qml_dim, reps=qml_reps,
                                    feature_map_type=qml_fm_type, entanglement=qml_entanglement,
                                )
                            else:
                                config_path = "config/quantum_config_ideal.yaml" if use_shot_ideal else "config/quantum_config_noisy.yaml"
                                qk, err = load_shot_based_kernel(
                                    qml_dim, qml_reps, qml_fm_type, qml_entanglement,
                                    config_path, shots_override=int(shots_override),
                                )
                            if qk is None:
                                st.error(f"Quantum kernel unavailable: {err}")
                                st.stop()
                            sim = float(qk.evaluate(x1_q, x2_q)[0, 0])
                            st.success("Quantum similarity computed (kernel value)")
                            fm_desc = f"{qml_fm_type}FeatureMap(reps={qml_reps}, entanglement={qml_entanglement})"
                            exec_label = "statevector" if use_statevector else ("shot_ideal" if use_shot_ideal else "shot_noisy")
                            payload.update({
                                "score_type": "kernel_similarity",
                                "quantum_kernel_similarity": round(sim, 6),
                                "feature_map": fm_desc,
                                "qml_dim_used": qml_dim,
                                "qml_reps_used": qml_reps,
                                "feature_map_type": qml_fm_type,
                                "entanglement": qml_entanglement,
                                "execution": exec_label,
                                "shots": int(shots_override) if not use_statevector else None,
                                "note": "Kernel similarity is not a calibrated probability.",
                            })
                            st.json(payload)

                        # Kernel for candidate ranking (same execution as main prediction; no ZNE for ranking)
                        if method == "quantum_kernel_similarity":
                            if use_shot_noisy and apply_zne:
                                qk_for_ranking, _ = load_shot_based_kernel(
                                    qml_dim, qml_reps, qml_fm_type, qml_entanglement,
                                    "config/quantum_config_noisy.yaml", shots_override=int(shots_override),
                                )
                            elif use_statevector:
                                qk_for_ranking, _ = load_quantum_kernel(
                                    qml_dim=qml_dim, reps=qml_reps,
                                    feature_map_type=qml_fm_type, entanglement=qml_entanglement,
                                )
                            else:
                                cfg = "config/quantum_config_ideal.yaml" if use_shot_ideal else "config/quantum_config_noisy.yaml"
                                qk_for_ranking, _ = load_shot_based_kernel(
                                    qml_dim, qml_reps, qml_fm_type, qml_entanglement, cfg, shots_override=int(shots_override),
                                )
                        else:
                            qk_for_ranking = None

                    st.subheader("Candidate ranking")
                    st.caption(
                        "**Top compounds for this disease**: ranked by the chosen scoring method (probability or kernel similarity). "
                        "**Top diseases for this compound**: same, in the other direction. Use these for drug-repurposing or prioritization ideas."
                    )
                    compounds = [(eid, entity_to_idx[eid]) for eid in entity_to_idx.keys() if eid.startswith("Compound::")]
                    diseases = [(eid, entity_to_idx[eid]) for eid in entity_to_idx.keys() if eid.startswith("Disease::")]

                    colA, colB = st.columns(2)
                    with colA:
                        st.markdown("**Top compounds for this disease**")
                        hv = reduced_all[[idx for _, idx in compounds]]
                        if method == "classical_model":
                            if model is None or scaler is None:
                                st.error("Missing trained classical model/scaler.")
                                st.stop()
                            tv = np.repeat(d_red.reshape(1, -1), hv.shape[0], axis=0)
                            X = np.concatenate([hv, tv, np.abs(hv - tv), hv * tv], axis=1)
                            scores = model.predict_proba(scaler.transform(X))[:, 1]
                            top_idx = np.argsort(scores)[-int(top_k):][::-1]
                            out = pd.DataFrame({
                                "compound": [compounds[i][0] for i in top_idx],
                                "score": scores[top_idx]
                            })
                            st.dataframe(out)
                        else:
                            qk = qk_for_ranking
                            if qk is None:
                                st.error("Quantum kernel unavailable for ranking.")
                                st.stop()
                            # prefilter with cosine for speed
                            cand_idx = cosine_topk_indices(hv, d_red, k=min(200, hv.shape[0]))
                            hv_small = hv[cand_idx]
                            x2 = to_quantum_input(d_red).reshape(1, -1)
                            sims = []
                            for row in hv_small:
                                sims.append(float(qk.evaluate(to_quantum_input(row).reshape(1, -1), x2)[0, 0]))
                            sims = np.asarray(sims)
                            order = np.argsort(sims)[-int(top_k):][::-1]
                            out = pd.DataFrame({
                                "compound": [compounds[cand_idx[i]][0] for i in order],
                                "kernel_similarity": sims[order]
                            })
                            st.dataframe(out)

                    with colB:
                        st.markdown("**Top diseases for this compound**")
                        tv = reduced_all[[idx for _, idx in diseases]]
                        if method == "classical_model":
                            if model is None or scaler is None:
                                st.error("Missing trained classical model/scaler.")
                                st.stop()
                            hv_rep = np.repeat(c_red.reshape(1, -1), tv.shape[0], axis=0)
                            X = np.concatenate([hv_rep, tv, np.abs(hv_rep - tv), hv_rep * tv], axis=1)
                            scores = model.predict_proba(scaler.transform(X))[:, 1]
                            top_idx = np.argsort(scores)[-int(top_k):][::-1]
                            out = pd.DataFrame({
                                "disease": [diseases[i][0] for i in top_idx],
                                "score": scores[top_idx]
                            })
                            st.dataframe(out)
                        else:
                            qk = qk_for_ranking
                            if qk is None:
                                st.error("Quantum kernel unavailable for ranking.")
                                st.stop()
                            cand_idx = cosine_topk_indices(tv, c_red, k=min(200, tv.shape[0]))
                            tv_small = tv[cand_idx]
                            x1 = to_quantum_input(c_red).reshape(1, -1)
                            sims = []
                            for row in tv_small:
                                sims.append(float(qk.evaluate(x1, to_quantum_input(row).reshape(1, -1))[0, 0]))
                            sims = np.asarray(sims)
                            order = np.argsort(sims)[-int(top_k):][::-1]
                            out = pd.DataFrame({
                                "disease": [diseases[cand_idx[i]][0] for i in order],
                                "kernel_similarity": sims[order]
                            })
                            st.dataframe(out)

                    st.subheader("Representation: classical embedding vs quantum-ready reduced vector")
                    st.caption(
                        "Left: full embedding (first 20 dims shown)—used by classical models. "
                        "Right: PCA-reduced vector (same dims as qubits)—fed into the quantum feature map and classical pair features."
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"Classical embedding ({emb_name}) – first 20 dims")
                        vec = embeddings[c_idx]
                        fig, ax = plt.subplots(figsize=(6, 3))
                        ax.bar(range(min(20, vec.shape[0])), vec[:min(20, vec.shape[0])])
                        ax.set_xlabel("Dimension index")
                        ax.set_ylabel("Embedding value")
                        st.pyplot(fig)
                        plt.close(fig)
                    with col2:
                        st.caption(f"Reduced (PCA) vector – {qml_dim} dims (used by QML + classical features)")
                        fig2, ax2 = plt.subplots(figsize=(6, 3))
                        ax2.bar(range(qml_dim), c_red[:qml_dim])
                        ax2.set_xlabel("Dimension index")
                        ax2.set_ylabel("Reduced (PCA) value")
                        st.pyplot(fig2)
                        plt.close(fig2)

                    st.subheader("Evidence")
                    st.caption(
                        "**Nearest neighbors**: entities most similar in embedding space (cosine similarity)—helps interpret why the model might link this pair. "
                        "**KG neighborhood**: real Hetionet edges involving this compound or disease—gives biological context."
                    )
                    st.markdown("**Embedding nearest neighbors** (same type, cosine similarity)")
                    def top_neighbors(target_idx: int, prefix: str, k: int = 10):
                        vec = embeddings[target_idx]
                        mask = [i for i, eid in enumerate(entity_ids) if eid.startswith(prefix)]
                        mat = embeddings[mask]
                        denom = (np.linalg.norm(mat, axis=1) * (np.linalg.norm(vec) + 1e-9)) + 1e-9
                        sims = (mat @ vec) / denom
                        order = np.argsort(sims)[-k-1:][::-1]
                        rows = []
                        for j in order:
                            idx = mask[j]
                            if idx == target_idx:
                                continue
                            rows.append({"entity": entity_ids[idx], "cosine_similarity": float(sims[j])})
                            if len(rows) >= k:
                                break
                        return pd.DataFrame(rows)

                    nn_cols = st.columns(2)
                    with nn_cols[0]:
                        st.caption("Similar compounds")
                        st.dataframe(top_neighbors(c_idx, "Compound::", k=10))
                    with nn_cols[1]:
                        st.caption("Similar diseases")
                        st.dataframe(top_neighbors(d_idx, "Disease::", k=10))

                    st.markdown("**KG neighborhood snippet** (first matches in Hetionet edges file)")
                    ctx_cols = st.columns(2)
                    with ctx_cols[0]:
                        st.caption("Compound context edges")
                        st.dataframe(scan_hetionet_context(compound_id, max_matches=20))
                    with ctx_cols[1]:
                        st.caption("Disease context edges")
                        st.dataframe(scan_hetionet_context(disease_id, max_matches=20))
    
    # Example predictions
    st.subheader("Example Predictions")
    st.caption("Click a button to run a prediction for that drug–disease pair. Buttons only appear when both the compound and disease exist in the loaded embedding set.")
    examples = [
        ("DB00688", "DOID_0060048", "Dexamethasone for COVID-19"),
        ("DB00945", "DOID_9352", "Aspirin for Type 2 Diabetes"),
        ("DB00316", "DOID_2841", "Metformin for Type 2 Diabetes"),
    ]
    
    # Filter examples to those present in the current embedding set (avoid "not found" loops)
    available_compounds = set(compounds_list)
    available_diseases = set(diseases_list)

    shown = 0
    for drug, disease, desc in examples:
        c = normalize_compound_id(drug)
        d = normalize_disease_id(disease)
        if available_compounds and c not in available_compounds:
            continue
        if available_diseases and d not in available_diseases:
            continue
        shown += 1
        if st.button(f"Try: {desc}", key=f"example_{drug}_{disease}"):
            st.query_params.update({"drug": drug, "disease": disease})
            st.session_state["auto_predict"] = True
            st.rerun()

    if shown == 0 and compounds_list and diseases_list:
        st.info("No curated examples exist in the currently loaded embedding subset. Use a random valid example:")
        if st.button("Pick random valid example", key="random_valid_example"):
            rng = np.random.default_rng(42)
            rc = compounds_list[int(rng.integers(0, len(compounds_list)))]
            rd = diseases_list[int(rng.integers(0, len(diseases_list)))]
            st.query_params.update({"drug": rc.replace("Compound::", ""), "disease": rd.replace("Disease::", "")})
            st.session_state["auto_predict"] = True
            st.rerun()

# ==============================
# PAGE 3: EXPERIMENT HISTORY
# ==============================
elif page == "5. Experiments (history)":
    st.header("Experiments: history of benchmarked runs")
    st.markdown("""
    This tab shows **all recorded benchmark runs**: PR-AUC over time, quantum vs classical deltas, kernel observables, and ideal vs noisy comparison.
    Use **Filters** to narrow by model type, execution mode, or noise; **Performance Trends** and **Δ PR-AUC** show how metrics evolve across runs.
    """)
    if df_history is None or len(df_history) == 0:
        st.info("No experiment history yet. Run benchmarks from the **Run benchmarks** tab (or upload `experiment_history.csv`) to populate this view.")
    if df_history is not None and len(df_history) > 0:
        st.subheader("Filters")
        st.caption(
            "**Model Type**: quantum model (e.g. QSVC). **Execution Mode** / **Noise Model**: how and where the circuit ran. "
            "**Max rows**: limit table/chart length. **Hide quantum=0**: drop runs where quantum PR-AUC was 0."
        )
        cols = st.columns(5)
        with cols[0]:
            model_options = df_history.get("quantum_model_type", pd.Series(dtype=object)).dropna().unique().tolist()
            model_filter = st.multiselect("Model Type", model_options)
        with cols[1]:
            exec_options = df_history.get("execution_mode", pd.Series(dtype=object)).dropna().unique().tolist()
            exec_filter = st.multiselect("Execution Mode", exec_options)
        with cols[2]:
            noise_options = df_history.get("noise_model", pd.Series(dtype=object)).fillna("ideal").unique().tolist()
            noise_filter = st.multiselect("Noise Model", noise_options)
        with cols[3]:
            max_rows = st.number_input("Max rows", min_value=10, max_value=500, value=200, step=10)
        with cols[4]:
            hide_quantum_zero = st.checkbox("Hide quantum=0 rows", value=False)

        filtered = df_history.copy()
        if model_filter and "quantum_model_type" in filtered.columns:
            filtered = filtered[filtered["quantum_model_type"].isin(model_filter)]
        if exec_filter and "execution_mode" in filtered.columns:
            filtered = filtered[filtered["execution_mode"].isin(exec_filter)]
        if noise_filter and "noise_model" in filtered.columns:
            filtered = filtered[filtered["noise_model"].fillna("ideal").isin(noise_filter)]
        if hide_quantum_zero and "quantum_pr_auc" in filtered.columns:
            filtered = filtered[filtered["quantum_pr_auc"] > 0]
        filtered = filtered.tail(int(max_rows))

        st.subheader("Artifacts")
        st.caption("Download the filtered experiment log as CSV for analysis or reporting.")
        st.download_button(
            "Download filtered history (CSV)",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="experiment_history_filtered.csv",
            mime="text/csv"
        )

        # Metrics over time
        st.subheader("Performance Trends")
        st.caption("PR-AUC for classical (blue) and quantum (purple) across experiment index. Use this to spot trends or regressions.")
        # PR-AUC over experiments
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(filtered.index, filtered['classical_pr_auc'], 'o-', label='Classical PR-AUC', color='blue')
        ax.plot(filtered.index, filtered['quantum_pr_auc'], 's-', label='Quantum PR-AUC', color='purple')
        ax.set_xlabel('Experiment #')
        ax.set_ylabel('PR-AUC')
        ax.set_title('PR-AUC Over Experiments')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Quantum vs Classical PR-AUC Delta")
        st.caption("Bar chart: positive = quantum beat classical on that run; negative = classical won. Green/red by sign.")
        delta = filtered["quantum_pr_auc"] - filtered["classical_pr_auc"]
        fig3, ax3 = plt.subplots(figsize=(10, 3))
        ax3.bar(filtered.index, delta, color=["green" if v >= 0 else "red" for v in delta])
        ax3.axhline(0, color="black", linewidth=1)
        ax3.set_xlabel("Experiment #")
        ax3.set_ylabel("Δ PR-AUC (Q - C)")
        st.pyplot(fig3)
        plt.close(fig3)

        # Kernel observables (if present)
        obs_cols = [c for c in filtered.columns if c.startswith("obs_")]
        if obs_cols:
            st.subheader("Quantum kernel observables (fidelity-style)")
            st.caption(
                "Diagnostic metrics from the quantum kernel (e.g. kernel gap, posneg mean). "
                "Useful to see how kernel quality varies across runs or with mitigation."
            )
            selectable = [c for c in obs_cols if c in filtered.columns]
            metric = st.selectbox("Observable", selectable, index=selectable.index("obs_kernel_gap") if "obs_kernel_gap" in selectable else 0)
            fig_obs, ax_obs = plt.subplots(figsize=(10, 3))
            ax_obs.plot(filtered.index, filtered[metric], "o-", color="orange")
            ax_obs.set_xlabel("Experiment #")
            ax_obs.set_ylabel(metric)
            ax_obs.set_title(f"{metric} over experiments")
            ax_obs.grid(True)
            st.pyplot(fig_obs)
            plt.close(fig_obs)

            # If ZNE is present, show a quick raw vs mitigated comparison for the primary observable.
            if "obs_kernel_posneg_mean" in filtered.columns and "obs_zne_kernel_posneg_mean_C0" in filtered.columns:
                st.subheader("Mitigation comparison (kernel_posneg_mean)")
                st.caption(
                    "Raw vs ZNE (zero-noise extrapolation) and readout-mitigated kernel values. "
                    "Shows whether error mitigation improves the training kernel signal."
                )
                fig_zne, ax_zne = plt.subplots(figsize=(10, 3))
                ax_zne.plot(filtered.index, filtered["obs_kernel_posneg_mean"], "o-", label="raw kernel (training K)", color="gray")
                # Optional explicit compute-uncompute estimate at λ=1.0
                if "obs_kernel_posneg_mean_explicit_raw_lambda1" in filtered.columns:
                    ax_zne.plot(filtered.index, filtered["obs_kernel_posneg_mean_explicit_raw_lambda1"], "o--", label="explicit raw (λ=1.0)", color="black")
                if "obs_kernel_posneg_mean_explicit_readout_lambda1" in filtered.columns:
                    ax_zne.plot(filtered.index, filtered["obs_kernel_posneg_mean_explicit_readout_lambda1"], "o--", label="explicit + readout (λ=1.0)", color="blue")

                # ZNE outputs
                if "obs_zne_kernel_posneg_mean_C0_raw" in filtered.columns:
                    ax_zne.plot(filtered.index, filtered["obs_zne_kernel_posneg_mean_C0_raw"], "s-", label="ZNE C0 (raw)", color="green")
                if "obs_zne_kernel_posneg_mean_C0_readout" in filtered.columns:
                    ax_zne.plot(filtered.index, filtered["obs_zne_kernel_posneg_mean_C0_readout"], "s-", label="ZNE C0 (readout)", color="teal")
                ax_zne.plot(filtered.index, filtered["obs_zne_kernel_posneg_mean_C0"], "d-", label="ZNE C0 (primary)", color="purple")
                ax_zne.set_xlabel("Experiment #")
                ax_zne.set_ylabel("kernel_posneg_mean")
                ax_zne.grid(True)
                ax_zne.legend()
                st.pyplot(fig_zne)
                plt.close(fig_zne)

                # Quick table: latest mitigation diagnostics (if present)
                diag_cols = [
                    c for c in [
                        "obs_readout_mitigation_enabled",
                        "obs_readout_mitigation_calibration_shots",
                        "obs_readout_mitigation_ridge_lambda",
                        "obs_zne_enabled",
                        "obs_zne_C0_clipped",
                        "obs_zne_fit_error",
                        "obs_zne_base_noise_model",
                        "obs_zne_scales_json",
                    ]
                    if c in filtered.columns
                ]
                if diag_cols:
                    st.caption("Latest mitigation diagnostics: readout_mitigation_* = calibration settings; obs_zne_* = zero-noise extrapolation (scales, fit error, base noise).")
                    st.dataframe(filtered[diag_cols].tail(1).T, width="stretch")
        
        # Parameter count over experiments
        st.subheader("Model complexity over time (parameter count)")
        st.caption("Classical vs quantum parameter count per run. Fewer quantum parameters can mean better scalability for larger graphs.")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.bar(filtered.index - 0.2, filtered['classical_num_parameters'], 
                width=0.4, label='Classical Params', color='lightblue')
        ax2.bar(filtered.index + 0.2, filtered['quantum_num_parameters'], 
                width=0.4, label='Quantum Params', color='plum')
        ax2.set_xlabel('Experiment #')
        ax2.set_ylabel('Number of Parameters')
        ax2.set_title('Model Complexity Over Time')
        ax2.legend()
        ax2.grid(True, axis='y')
        st.pyplot(fig2)
        plt.close(fig2)

        # Ideal vs Noisy comparison (if metadata available)
        comparison_cols = {"execution_mode", "noise_model", "backend_label"}
        if comparison_cols.issubset(df_history.columns):
            st.subheader("Ideal vs Noisy Comparison")
            st.caption("Latest run per (execution_mode, noise_model, backend_label) with PR-AUC. Compare ideal vs noisy rows.")
            df_comp = filtered.copy()
            df_comp["run_index"] = df_comp.index
            group_cols = ["execution_mode", "noise_model", "backend_label"]
            latest = (
                df_comp.sort_values("run_index")
                .groupby(group_cols, dropna=False)
                .tail(1)
                .reset_index(drop=True)
            )
            display_cols = group_cols + ["run_index"]
            if "quantum_pr_auc" in latest.columns:
                display_cols.append("quantum_pr_auc")
            if "classical_pr_auc" in latest.columns:
                display_cols.append("classical_pr_auc")
            st.dataframe(latest[display_cols].sort_values(group_cols))

        # Full history table
        st.subheader("Full Experiment Log")
        st.caption("Full filtered table: all columns for each run. Scroll to see execution details, metrics, and observables.")
        st.dataframe(filtered)

        st.subheader("Export report bundle")
        st.caption("Saves PR-AUC, kernel gap, and mitigation plots plus the filtered CSV to a timestamped folder under `reports/` for slides or documentation.")

        if st.button("Generate report plots"):
            import pathlib
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            out_dir = pathlib.Path(PROJECT_ROOT) / "reports" / f"report_{ts}"
            out_dir.mkdir(parents=True, exist_ok=True)

            # CSV snapshot
            snapshot_path = out_dir / "experiment_history_filtered.csv"
            filtered.to_csv(snapshot_path, index=False)

            # PR-AUC plot (if present)
            if "quantum_pr_auc" in filtered.columns:
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(filtered.index, filtered.get("classical_pr_auc"), "o-", label="Classical PR-AUC", color="gray")
                ax.plot(filtered.index, filtered.get("quantum_pr_auc"), "s-", label="Quantum PR-AUC", color="purple")
                ax.set_title("PR-AUC over runs")
                ax.set_xlabel("Experiment #")
                ax.set_ylabel("PR-AUC")
                ax.grid(True)
                ax.legend()
                fig.tight_layout()
                fig.savefig(out_dir / "pr_auc.png", dpi=200)
                plt.close(fig)

            # Kernel gap (if present)
            if "obs_kernel_gap" in filtered.columns:
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(filtered.index, filtered["obs_kernel_gap"], "o-", color="orange")
                ax.set_title("Kernel gap (pospos/negneg vs posneg)")
                ax.set_xlabel("Experiment #")
                ax.set_ylabel("obs_kernel_gap")
                ax.grid(True)
                fig.tight_layout()
                fig.savefig(out_dir / "kernel_gap.png", dpi=200)
                plt.close(fig)

            # Mitigation plot (if present)
            if "obs_kernel_posneg_mean" in filtered.columns and "obs_zne_kernel_posneg_mean_C0" in filtered.columns:
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(filtered.index, filtered["obs_kernel_posneg_mean"], "o-", label="raw kernel (training K)", color="gray")
                if "obs_kernel_posneg_mean_explicit_raw_lambda1" in filtered.columns:
                    ax.plot(filtered.index, filtered["obs_kernel_posneg_mean_explicit_raw_lambda1"], "o--", label="explicit raw (λ=1.0)", color="black")
                if "obs_kernel_posneg_mean_explicit_readout_lambda1" in filtered.columns:
                    ax.plot(filtered.index, filtered["obs_kernel_posneg_mean_explicit_readout_lambda1"], "o--", label="explicit + readout (λ=1.0)", color="blue")
                if "obs_zne_kernel_posneg_mean_C0_raw" in filtered.columns:
                    ax.plot(filtered.index, filtered["obs_zne_kernel_posneg_mean_C0_raw"], "s-", label="ZNE C0 (raw)", color="green")
                if "obs_zne_kernel_posneg_mean_C0_readout" in filtered.columns:
                    ax.plot(filtered.index, filtered["obs_zne_kernel_posneg_mean_C0_readout"], "s-", label="ZNE C0 (readout)", color="teal")
                ax.plot(filtered.index, filtered["obs_zne_kernel_posneg_mean_C0"], "d-", label="ZNE C0 (primary)", color="purple")
                ax.set_title("Mitigation impact on kernel_posneg_mean")
                ax.set_xlabel("Experiment #")
                ax.set_ylabel("kernel_posneg_mean")
                ax.grid(True)
                ax.legend()
                fig.tight_layout()
                fig.savefig(out_dir / "mitigation_kernel_posneg_mean.png", dpi=200)
                plt.close(fig)

            st.success(f"Report bundle generated: `{out_dir}`")
    
    else:
        st.warning("No experiment history found. Use the **Run benchmarks** tab to run the pipeline one or more times, or upload `experiment_history.csv` from a local run.")

# ==============================
# PAGE: MODEL COMPARISON
# ==============================
elif page == "6. Comparison (classical vs quantum)":
    st.header("Model comparison: classical versus quantum (evidence from recorded runs)")
    st.markdown("""
    **In plain terms:** This page compares classical and quantum model performance across runs.
    Only runs where **both** classical and quantum were evaluated (paired) are used, so the comparison is fair.
    **Δ PR-AUC (Q − C)** = how much better (or worse) the quantum model did than the classical one.
    **Bootstrap** = resampling runs to estimate uncertainty (e.g. 95% CI for the average difference).
    **Win rate** = fraction of runs where quantum beat classical.
    """)
    if df_history is None or len(df_history) == 0:
        st.warning("No experiment history found. Run benchmarks (with both classical and quantum) from the **Run benchmarks** tab, or upload `experiment_history.csv`.")
    else:
        dfc = df_history.copy()
        for col in ["classical_pr_auc", "quantum_pr_auc", "obs_kernel_eval_seconds_total", "execution_shots"]:
            if col in dfc.columns:
                dfc[col] = pd.to_numeric(dfc[col], errors="coerce")

        # Only keep rows where both are present
        paired = dfc.dropna(subset=["classical_pr_auc", "quantum_pr_auc"]).copy()
        if len(paired) == 0:
            st.warning("No paired classical+quantum rows found yet (some runs are quantum-only or classical-only).")
        else:
            paired["delta_pr_auc"] = paired["quantum_pr_auc"] - paired["classical_pr_auc"]
            paired["quantum_variant"] = paired.get("obs_kernel_source", pd.Series(["unknown"] * len(paired))).fillna("unknown")
            paired["execution_bucket"] = (
                paired.get("execution_mode", "unknown").fillna("unknown").astype(str)
                + " | " + paired.get("backend_label", "unknown").fillna("unknown").astype(str)
                + " | " + paired.get("noise_model", "ideal").fillna("ideal").astype(str)
            )

            st.subheader("Summary")
            st.caption("**Paired runs**: experiments with both classical and quantum PR-AUC. **Mean Δ (Q − C)**: average advantage (or disadvantage) of quantum.")
            cols = st.columns(4)
            with cols[0]:
                st.metric("Paired runs", int(len(paired)))
            with cols[1]:
                st.metric("Mean Classical PR-AUC", float(paired["classical_pr_auc"].mean()))
            with cols[2]:
                st.metric("Mean Quantum PR-AUC", float(paired["quantum_pr_auc"].mean()))
            with cols[3]:
                st.metric("Mean Δ (Q - C)", float(paired["delta_pr_auc"].mean()))

            st.subheader("Statistical rigor (paired)")
            st.caption("Bootstrap is over paired runs (re-sampling runs with replacement). This quantifies uncertainty in the mean ΔPR-AUC.")

            rng_seed = st.number_input("Bootstrap seed", min_value=0, max_value=10_000_000, value=42, step=1, help="Random seed for reproducible bootstrap.")
            n_boot = st.number_input("Bootstrap samples", min_value=200, max_value=20_000, value=2000, step=200, help="Number of bootstrap samples; more = more stable 95% CI and P(mean Δ>0).")

            deltas = paired["delta_pr_auc"].dropna().values.astype(float)
            win_rate = float(np.mean(deltas > 0.0)) if len(deltas) else float("nan")
            median_delta = float(np.median(deltas)) if len(deltas) else float("nan")

            # Cohen's d (paired, using delta std)
            d_std = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else float("nan")
            cohens_d = float(np.mean(deltas) / d_std) if d_std and not np.isnan(d_std) and d_std > 0 else float("nan")

            # Bootstrap CI for mean delta
            ci_low = ci_high = p_mean_gt0 = float("nan")
            try:
                rng = np.random.default_rng(int(rng_seed))
                idx = rng.integers(0, len(deltas), size=(int(n_boot), len(deltas)))
                boot_means = np.mean(deltas[idx], axis=1)
                ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975]).tolist()
                p_mean_gt0 = float(np.mean(boot_means > 0.0))
            except Exception:
                pass

            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.metric("Win-rate (Δ>0)", f"{win_rate:.2%}" if not np.isnan(win_rate) else "N/A")
            with s2:
                st.metric("Median ΔPR-AUC", f"{median_delta:.4f}" if not np.isnan(median_delta) else "N/A")
            with s3:
                st.metric("Mean ΔPR-AUC 95% CI", f"[{ci_low:.4f}, {ci_high:.4f}]" if not np.isnan(ci_low) else "N/A")
            with s4:
                st.metric("P(mean Δ>0) (bootstrap)", f"{p_mean_gt0:.2%}" if not np.isnan(p_mean_gt0) else "N/A")

            if not np.isnan(cohens_d):
                st.caption(f"Effect size (Cohen’s d on paired deltas): **{cohens_d:.3f}** (roughly: 0.2 small, 0.5 medium, 0.8 large).")

            st.subheader("Distribution: PR-AUC")
            st.caption("Histogram of classical vs quantum PR-AUC across paired runs. Overlap or separation shows how consistent the advantage is.")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.hist(paired["classical_pr_auc"].values, bins=12, alpha=0.6, label="Classical", color="blue")
            ax.hist(paired["quantum_pr_auc"].values, bins=12, alpha=0.6, label="Quantum", color="purple")
            ax.set_xlabel("PR-AUC")
            ax.set_ylabel("Count")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Distribution: Δ PR-AUC (Quantum - Classical)")
            st.caption("Spread of the per-run difference. Centered above 0 = quantum tends to win; below 0 = classical tends to win.")
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            ax2.hist(paired["delta_pr_auc"].values, bins=16, color="gray")
            ax2.axvline(0, color="black", linewidth=1)
            ax2.set_xlabel("Δ PR-AUC")
            ax2.set_ylabel("Count")
            ax2.grid(True)
            st.pyplot(fig2)
            plt.close(fig2)

            st.subheader("By quantum kernel variant (full vs Nyström)")
            st.caption("Aggregates by kernel source: full kernel vs Nyström approximation. Compare mean_delta and mean_kernel_seconds to choose a variant.")
            grp = (
                paired.groupby(["quantum_variant"], dropna=False)
                .agg(
                    n=("delta_pr_auc", "count"),
                    mean_delta=("delta_pr_auc", "mean"),
                    mean_q=("quantum_pr_auc", "mean"),
                    mean_c=("classical_pr_auc", "mean"),
                    mean_kernel_seconds=("obs_kernel_eval_seconds_total", "mean"),
                    mean_shots=("execution_shots", "mean"),
                )
                .reset_index()
                .sort_values("mean_delta", ascending=False)
            )
            st.dataframe(grp, width="stretch")

            st.subheader("Cost-aware recommendation")
            st.caption("Uses a simple score: mean ΔPR-AUC minus a cost penalty (runtime + shots). Tune the penalty to match your budget.")

            cost_weight_seconds = st.slider("Penalty per second of kernel evaluation", min_value=0.0, max_value=0.01, value=0.001, step=0.0005, format="%.4f")
            cost_weight_shots = st.slider("Penalty per 1k shots", min_value=0.0, max_value=0.01, value=0.0005, step=0.0005, format="%.4f")

            g2 = grp.copy()
            g2["mean_kernel_seconds"] = pd.to_numeric(g2["mean_kernel_seconds"], errors="coerce")
            g2["mean_shots"] = pd.to_numeric(g2["mean_shots"], errors="coerce")
            g2["shots_k"] = g2["mean_shots"] / 1000.0
            g2["cost_score"] = g2["mean_delta"] - (cost_weight_seconds * g2["mean_kernel_seconds"].fillna(0.0)) - (cost_weight_shots * g2["shots_k"].fillna(0.0))
            g2 = g2.sort_values("cost_score", ascending=False)

            st.dataframe(
                g2[["quantum_variant", "n", "mean_delta", "mean_kernel_seconds", "mean_shots", "cost_score"]],
                width="stretch"
            )
            best = g2.iloc[0]
            st.success(
                f"Recommended default: **{best['quantum_variant']}** "
                f"(score={float(best['cost_score']):.4f}, mean Δ={float(best['mean_delta']):.4f}, "
                f"seconds={float(best['mean_kernel_seconds']):.2f}, shots={float(best['mean_shots']):.0f})."
            )

            st.subheader("Cost vs benefit scatter (paired runs)")
            st.caption("Each point = one run. X = kernel evaluation time (cost); Y = Δ PR-AUC (benefit). Green = quantum won; red = classical won.")
            if "obs_kernel_eval_seconds_total" in paired.columns:
                figc, axc = plt.subplots(figsize=(10, 3))
                axc.scatter(
                    paired["obs_kernel_eval_seconds_total"].values,
                    paired["delta_pr_auc"].values,
                    c=["green" if v >= 0 else "red" for v in paired["delta_pr_auc"].values],
                    alpha=0.7,
                )
                axc.axhline(0, color="black", linewidth=1)
                axc.set_xlabel("Kernel eval seconds (total)")
                axc.set_ylabel("Δ PR-AUC (Q - C)")
                axc.set_title("Benefit vs cost")
                axc.grid(True)
                st.pyplot(figc)
                plt.close(figc)

            st.subheader("Drilldown table (paired runs)")
            st.caption("Per-run details: timestamp, backend, shots, kernel time, model variant, and PR-AUC values for inspection or export.")
            show_cols = [c for c in [
                "run_timestamp_utc", "execution_mode", "backend_label", "noise_model",
                "execution_shots", "obs_kernel_eval_seconds_total",
                "obs_kernel_source", "obs_nystrom_m",
                "classical_pr_auc", "quantum_pr_auc", "delta_pr_auc",
            ] if c in paired.columns]
            st.dataframe(paired[show_cols].tail(200), width="stretch")

# ==============================
# PAGE: KG SUMMARY
# ==============================
elif page == "8. Knowledge graph (inventory)":
    st.header("Knowledge graph inventory: Hetionet")
    st.markdown("""
    **In plain terms:** A knowledge graph is a network of *things* (nodes) and *relationships* (edges).
    Here, nodes are drugs, diseases, genes, etc.; edges are typed (e.g. “treats”, “associates with”).
    **Metaedge** = the type of relationship (e.g. CtD = Compound treats Disease).
    """)
    _expander_for_term("metaedge", "What is a metaedge?")
    st.caption("This is the graph the pipeline trains on. Below: total edges, relation types, node types, and a sample of triples. Use this to understand the data scale and relation mix.")

    @st.cache_data(show_spinner=False)
    def _load_edges():
        from kg_layer.kg_loader import load_hetionet_edges
        return load_hetionet_edges()

    df_edges = _load_edges()
    st.caption("**Total edges**: number of (source, target, metaedge) triples. **Relation types**: distinct metaedges. **Unique nodes**: distinct entity IDs.")
    st.metric("Total edges", int(len(df_edges)))
    st.metric("Relation types (metaedge)", int(df_edges["metaedge"].nunique()))
    st.metric("Unique nodes (string IDs)", int(pd.concat([df_edges["source"], df_edges["target"]]).nunique()))

    # Node type breakdown (prefix before ::)
    def _node_type(x: str) -> str:
        try:
            return str(x).split("::", 1)[0]
        except Exception:
            return "Unknown"

    nodes = pd.concat([df_edges["source"], df_edges["target"]]).astype(str)
    node_types = nodes.map(_node_type).value_counts().reset_index()
    node_types.columns = ["node_type", "count"]
    st.subheader("Node types")
    st.caption("Entity type (prefix before ::), e.g. Compound, Disease, Gene. Count = how many times that type appears as source or target.")
    st.dataframe(node_types, width="stretch")

    rel_counts = df_edges["metaedge"].value_counts().reset_index()
    rel_counts.columns = ["metaedge", "count"]
    st.subheader("Top relations")
    st.caption("Most frequent relation types (metaedges). CtD = Compound–treats–Disease; others (e.g. DaG, GiG) describe different link types.")
    st.dataframe(rel_counts.head(20), width="stretch")

    fig, ax = plt.subplots(figsize=(10, 3))
    topk = rel_counts.head(20).iloc[::-1]
    ax.barh(topk["metaedge"], topk["count"], color="steelblue")
    ax.set_xlabel("Edge count")
    ax.set_ylabel("Relation type (metaedge)")
    ax.set_title("Top 20 metaedges")
    ax.grid(True, axis="x")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "Download KG relation counts (CSV)",
        data=rel_counts.to_csv(index=False).encode("utf-8"),
        file_name="hetionet_relation_counts.csv",
        mime="text/csv",
    )

    st.subheader("Sample edges")
    st.caption("Random sample of (source, target, metaedge) triples. Source/target are entity IDs (e.g. Compound::DB00123, Disease::DOID_1234).")
    st.dataframe(df_edges.sample(n=min(50, len(df_edges)), random_state=42), width="stretch")

# ==============================
# PAGE: FINDINGS
# ==============================
elif page == "7. Findings (ranked hypotheses)":
    st.header("Findings: ranked hypotheses from the latest model run")
    st.markdown("""
    **In plain terms:** The model scores every drug–disease pair. **Ranked hypotheses** = pairs ordered by that score (highest first).
    **Novel links** = high-scoring pairs that are *not* already in Hetionet—candidate “treats” links for validation.
    **y_score** = the model’s output (probability or kernel score) used for ranking.
    """)
    with st.expander("Column and term reference", expanded=False):
        st.markdown("""
        - **split**: train vs test (we show test-set predictions).
        - **source / target**: compound ID and disease ID (e.g. Compound::DB00123, Disease::DOID_1234).
        - **y_true**: 1 = known CtD link in the test set; 0 = non-link (negative).
        - **y_score**: model score (higher = more likely to be a link). Used to rank hypotheses.
        - **exists_in_hetionet_ctd**: whether this pair exists in the full Hetionet CtD relation (novel = False here).
        """)
    st.caption("These are **high-scoring predicted links** from the most recent QSVC run, with a novelty check against full Hetionet CtD edges.")

    pred_path = os.path.join(PROJECT_ROOT, "results", "predictions_latest.csv")
    if not os.path.exists(pred_path):
        st.warning("No `results/predictions_latest.csv` found yet. Run a benchmark with **Run quantum** (or both classical and quantum) from the **Run benchmarks** tab; the pipeline writes predictions for QSVC runs.")
    else:
        preds = pd.read_csv(pred_path)
        required = {"split", "source", "target", "y_true", "y_score"}
        if not required.issubset(set(preds.columns)):
            st.warning("Predictions file does not include endpoints yet. Re-run after the latest logging update.")
        else:
            preds["y_score"] = pd.to_numeric(preds["y_score"], errors="coerce")
            preds["y_true"] = pd.to_numeric(preds["y_true"], errors="coerce")

            test = preds[preds["split"] == "test"].dropna(subset=["y_score"]).copy()

            # Novelty check vs full Hetionet CtD edges
            @st.cache_data(show_spinner=False)
            def _load_ctd_pairs():
                from kg_layer.kg_loader import load_hetionet_edges
                df = load_hetionet_edges()
                ctd = df[df["metaedge"] == "CtD"][["source", "target"]].astype(str)
                return set(map(tuple, ctd.values.tolist()))

            try:
                ctd_pairs = _load_ctd_pairs()
                test["exists_in_hetionet_ctd"] = list(zip(test["source"].astype(str), test["target"].astype(str)))
                test["exists_in_hetionet_ctd"] = test["exists_in_hetionet_ctd"].apply(lambda x: x in ctd_pairs)
            except Exception:
                test["exists_in_hetionet_ctd"] = False

            st.subheader("Top predicted novel links (test negatives ranked high)")
            st.caption(
                "Pairs the model scored high but that are **not** in the test positives and **not** in full Hetionet CtD. "
                "**source** / **target** = compound / disease IDs; **y_score** = model score; **exists_in_hetionet_ctd** = already in KG (should be False here)."
            )
            k = st.slider("Top-K", min_value=5, max_value=100, value=25, step=5, help="Number of top novel (high-scoring, not-in-Hetionet) links to show in the table.")
            novel = test[(test["y_true"] == 0) & (~test["exists_in_hetionet_ctd"])].sort_values("y_score", ascending=False).head(int(k))
            st.dataframe(novel[["source", "target", "y_score", "exists_in_hetionet_ctd"]], width="stretch")

            st.subheader("Generate evidence bundle (neighbors + KG context)")
            st.caption(
                "Creates an exportable CSV with lightweight evidence: nearest neighbors in embedding space and top metaedges involving each node. "
                "**Neighbors per entity**: how many similar compounds/diseases to include. **Top metaedges per entity**: how many relation types per node."
            )

            ev_k = st.slider("Neighbors per entity", min_value=3, max_value=25, value=10, step=1, help="Number of nearest neighbors (embedding space) per compound/disease.")
            ctx_k = st.slider("Top metaedges per entity", min_value=3, max_value=25, value=10, step=1, help="Number of most frequent relation types to include per node.")

            @st.cache_resource(show_spinner=False)
            def _load_embeddings_for_evidence():
                # Reuse dashboard embedding loader (already handles repo paths)
                embs = load_entity_embeddings()
                return embs

            @st.cache_data(show_spinner=False)
            def _load_edges_for_context():
                from kg_layer.kg_loader import load_hetionet_edges
                return load_hetionet_edges()

            def _cosine_topk(vecs: np.ndarray, names: list[str], query_idx: int, topk: int) -> list[tuple[str, float]]:
                v = vecs[query_idx].astype(float)
                denom = (np.linalg.norm(vecs, axis=1) * (np.linalg.norm(v) + 1e-12)) + 1e-12
                sims = (vecs @ v) / denom
                sims[query_idx] = -np.inf
                idx = np.argpartition(-sims, kth=min(topk, len(sims) - 1))[:topk]
                idx = idx[np.argsort(-sims[idx])]
                return [(names[int(i)], float(sims[int(i)])) for i in idx]

            def _top_metaedges_for_entity(df_edges: pd.DataFrame, ent: str, topk: int) -> list[tuple[str, int]]:
                sub = df_edges[(df_edges["source"] == ent) | (df_edges["target"] == ent)]
                vc = sub["metaedge"].value_counts().head(topk)
                return [(str(k), int(v)) for k, v in vc.items()]

            def _evidence_bundle(rows: pd.DataFrame) -> pd.DataFrame:
                embs = _load_embeddings_for_evidence()
                df_edges = _load_edges_for_context()

                # Decide which embedding set we have for each entity type
                comp_vecs = embs.get("compound_embeddings")
                comp_names = embs.get("compounds_list") or []
                dis_vecs = embs.get("disease_embeddings")
                dis_names = embs.get("diseases_list") or []
                comp_index = {n: i for i, n in enumerate(comp_names)}
                dis_index = {n: i for i, n in enumerate(dis_names)}

                out = []
                for _, r in rows.iterrows():
                    c = str(r["source"])
                    d = str(r["target"])

                    c_neighbors = []
                    d_neighbors = []
                    try:
                        if comp_vecs is not None and c in comp_index:
                            c_neighbors = _cosine_topk(comp_vecs, comp_names, comp_index[c], int(ev_k))
                    except Exception:
                        c_neighbors = []
                    try:
                        if dis_vecs is not None and d in dis_index:
                            d_neighbors = _cosine_topk(dis_vecs, dis_names, dis_index[d], int(ev_k))
                    except Exception:
                        d_neighbors = []

                    c_ctx = []
                    d_ctx = []
                    try:
                        c_ctx = _top_metaedges_for_entity(df_edges, c, int(ctx_k))
                        d_ctx = _top_metaedges_for_entity(df_edges, d, int(ctx_k))
                    except Exception:
                        c_ctx = []
                        d_ctx = []

                    out.append({
                        "source": c,
                        "target": d,
                        "y_score": float(r["y_score"]) if not pd.isna(r["y_score"]) else None,
                        "exists_in_hetionet_ctd": bool(r.get("exists_in_hetionet_ctd", False)),
                        "compound_neighbors_json": json.dumps(c_neighbors),
                        "disease_neighbors_json": json.dumps(d_neighbors),
                        "compound_top_metaedges_json": json.dumps(c_ctx),
                        "disease_top_metaedges_json": json.dumps(d_ctx),
                    })
                return pd.DataFrame(out)

            if st.button("Build evidence bundle (CSV)"):
                bundle = _evidence_bundle(novel)
                st.dataframe(bundle, width="stretch")
                st.download_button(
                    "Download evidence bundle (CSV)",
                    data=bundle.to_csv(index=False).encode("utf-8"),
                    file_name="findings_evidence_bundle.csv",
                    mime="text/csv",
                )

            st.download_button(
                "Download top novel links (CSV)",
                data=novel[["source", "target", "y_score", "exists_in_hetionet_ctd"]].to_csv(index=False).encode("utf-8"),
                file_name="findings_top_novel_links.csv",
                mime="text/csv",
            )

            st.subheader("Sanity check: top-scoring true positives (test)")
            st.caption("Known CtD links (y_true=1) that the model also scored high. Confirms the model ranks real treatments well before trusting novel predictions.")
            top_tp = test[test["y_true"] == 1].sort_values("y_score", ascending=False).head(int(k))
            st.dataframe(top_tp[["source", "target", "y_score", "exists_in_hetionet_ctd"]], width="stretch")

# ==============================
# PAGE 4: RUN BENCHMARKS
# ==============================
elif page == "2. Run benchmarks (get results)":
    st.header("Run benchmarks")
    st.markdown("""
    Run the link-prediction pipeline from the dashboard and write results to `results/`. 
    Choose **what to run** (classical, quantum, or both) and **how** (single run vs Ideal + Noisy). 
    You can also **generate demo results** or **upload** CSVs if the pipeline cannot run in this environment.
    """)
    st.caption(
        "On Hugging Face Spaces, runs may hit time or memory limits. "
        "You can run benchmarks locally and upload results below."
    )

    # What is being tested — expandable reference
    with st.expander("What is being tested?", expanded=True):
        st.markdown("""
        | What you run | What the pipeline does | What gets measured |
        |--------------|------------------------|--------------------|
        | **Classical only** | Trains/evaluates **classical** models only: logistic regression (and optionally RF, SVM). Uses the same embeddings and pair features as the rest of the pipeline. | Classical PR-AUC, accuracy, parameter count. No quantum metrics. |
        | **Quantum only** | Trains/evaluates **quantum** models only (QSVC, optionally VQC). Uses PCA-reduced features and a quantum kernel (feature map). | Quantum PR-AUC, accuracy, parameter count, kernel observables. No classical metrics. |
        | **Both** (default) | Runs **classical and quantum** in one pipeline: same data, same split, so results are directly comparable. | Both classical and quantum PR-AUC, accuracy, parameters; ideal for the **Comparison** tab. |
        | **Ideal** config | Quantum circuits run on a **noiseless** simulator (statevector or ideal backend). | Best-case quantum performance. |
        | **Noisy** config | Quantum circuits run with a **noise model** (simulated device errors). | Robustness of quantum models to noise; compare with Ideal in **Results** and **Experiments**. |
        """)
        st.markdown("**Relation** (e.g. CtD) is the link type being predicted. **Fast mode** reduces epochs and model search for quicker runs.")
        st.caption("Results are written to `results/latest_run.csv` and `results/experiment_history.csv`.")

    with st.expander("Terminal commands reference (run from project root)"):
        st.markdown("**Run both classical and quantum (single run):**")
        st.code("python3 scripts/run_optimized_pipeline.py --relation CtD --results_dir results --fast_mode", language="bash")
        st.markdown("**Classical only:**")
        st.code("python3 scripts/run_optimized_pipeline.py --relation CtD --results_dir results --fast_mode --classical_only", language="bash")
        st.markdown("**Quantum only:**")
        st.code("python3 scripts/run_optimized_pipeline.py --relation CtD --results_dir results --fast_mode --quantum_only", language="bash")
        st.markdown("**Ideal then noisy (both classical and quantum):**")
        st.code("""# Ideal (noiseless)
python3 scripts/run_optimized_pipeline.py --relation CtD --results_dir results --fast_mode \\
  --quantum_config_path config/quantum_config_ideal.yaml
# Noisy (simulated device noise)
python3 scripts/run_optimized_pipeline.py --relation CtD --results_dir results --fast_mode \\
  --quantum_config_path config/quantum_config_noisy.yaml""", language="bash")
        st.caption("Or use: `bash scripts/benchmark_ideal_noisy.sh CtD results --fast_mode` (append `--classical_only` or `--quantum_only` to run only one).")

    # Environment check: show whether pipeline deps are available
    st.subheader("Environment check")
    st.caption("Checks that torch, pykeen, and qiskit_algorithms are importable. All three are needed to run the full pipeline; if any are missing, use **Generate demo results** or **Upload** instead.")
    def _check(name: str, fn) -> bool:
        try:
            fn()
            return True
        except Exception:
            return False
    has_torch = _check("torch", lambda: __import__("torch"))
    has_pykeen = _check("pykeen", lambda: __import__("pykeen"))
    has_qiskit_alg = _check("qiskit_algorithms", lambda: __import__("qiskit_algorithms"))
    pipeline_ready = has_torch and has_pykeen and has_qiskit_alg
    st.caption(
        f"torch: {'✓' if has_torch else '✗'}  |  "
        f"pykeen: {'✓' if has_pykeen else '✗'}  |  "
        f"qiskit_algorithms: {'✓' if has_qiskit_alg else '✗'}"
    )
    if not pipeline_ready:
        st.warning(
            "Pipeline dependencies (torch, pykeen, qiskit_algorithms) are not installed in this environment. "
            "**Run benchmark** will not work here. Use **Generate demo results** below to load sample data, or **Upload** a CSV from a local run."
        )
    else:
        st.success("Pipeline dependencies are available. You can run benchmarks below.")

    # Show writable results path when using /tmp (e.g. HF Spaces)
    if str(RESULTS_DIR).startswith("/tmp"):
        st.caption(f"Results are saved under a temporary directory (e.g. on Hugging Face Spaces). Data may not persist after the app restarts.")

    st.subheader("Configure run")
    with st.form("benchmark_form"):
        st.markdown("**What to run**")
        run_classical = st.checkbox(
            "Run classical",
            value=True,
            help="Train and evaluate classical models (logistic regression, optional RF/SVM). Measures classical PR-AUC and accuracy.",
        )
        run_quantum = st.checkbox(
            "Run quantum",
            value=True,
            help="Train and evaluate quantum model (QSVC). Measures quantum PR-AUC, kernel quality, and sensitivity to noise.",
        )
        if not run_classical and not run_quantum:
            st.caption("Both unchecked → pipeline will run **both** classical and quantum.")
        st.markdown("**Pipeline options**")
        relation = st.text_input(
            "Relation",
            value="CtD",
            help="Target relation type: CtD = Compound–treats–Disease; DaG = Disease–associates–Gene; etc.",
        )
        fast_mode = st.checkbox(
            "Fast mode",
            value=True,
            help="Fewer embedding epochs and smaller model search; good for quick checks. Turn off for full training.",
        )
        full_graph = st.checkbox(
            "Full-graph embeddings",
            value=False,
            help="Train embeddings on the full Hetionet graph (slower, potentially richer). Default: task-specific subset.",
        )
        benchmark_mode = st.radio(
            "Benchmark mode",
            ["Ideal + Noisy (two runs)", "Single run (ideal config only)"],
            index=0,
            help="Ideal + Noisy: run once with noiseless simulator, then once with noisy simulator (for quantum robustness). Single run: one pass with ideal config only.",
        )
        runs = st.number_input(
            "Repeats",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            help="Number of times to repeat the benchmark (each repeat = one run for Single run, or ideal+noisy for Ideal + Noisy).",
        )
        use_full_config = st.checkbox(
            "Use full config (classical + quantum, QSVC + hybrid, same as recommended terminal run)",
            value=False,
            help="Runs the exact pipeline: full-graph embeddings, contrastive learning, task-specific finetuning, calibration, 12 qubits, ZZ reps=1 linear, no pre-PCA. Slower but matches the full ranking (LR, RF, Ensemble, QSVC, Hybrid) in the Results tab.",
        )
        submitted = st.form_submit_button("Run benchmark")

    if submitted and not pipeline_ready:
        st.error("Cannot run: pipeline dependencies are missing. Use **Generate demo results** or **Upload** a CSV instead.")
    elif submitted:
        log_container = st.empty()
        # Use writable RESULTS_DIR so HF Spaces and any cwd work
        base_args = ["--relation", relation.strip() or "CtD", "--results_dir", str(RESULTS_DIR)]
        if use_full_config:
            base_args = [
                "--relation", relation.strip() or "CtD", "--results_dir", str(RESULTS_DIR),
                "--pos_edge_sample", "1500",
                "--full_graph_embeddings",
                "--embedding_method", "RotatE", "--embedding_dim", "128", "--embedding_epochs", "200",
                "--use_evidence_weighting", "--min_shared_genes", "1",
                "--use_contrastive_learning", "--contrastive_epochs", "75",
                "--use_task_specific_finetuning", "--task_specific_epochs", "100", "--task_specific_lr", "0.001",
                "--calibrate_probabilities", "--calibration_method", "isotonic",
                "--qml_dim", "12", "--qml_encoding", "hybrid", "--qml_reduction_method", "pca",
                "--qml_feature_selection_method", "f_classif", "--qml_feature_select_k_mult", "6.0",
                "--qml_pre_pca_dim", "0", "--qml_feature_map", "ZZ", "--qml_feature_map_reps", "1",
                "--qml_entanglement", "linear", "--negative_sampling", "hard", "--neg_ratio", "2.0",
                "--skip_svm_rbf", "--skip_svm_linear", "--skip_vqc", "--random_state", "42",
            ]
        else:
            if fast_mode:
                base_args.append("--fast_mode")
            if run_classical and not run_quantum:
                base_args.append("--classical_only")
            elif run_quantum and not run_classical:
                base_args.append("--quantum_only")
            if full_graph:
                base_args.append("--full_graph_embeddings")
        python_exe = sys.executable
        script = PROJECT_ROOT / "scripts" / "run_optimized_pipeline.py"
        run_ideal_noisy = benchmark_mode.startswith("Ideal + Noisy")

        # Show what will run (summary and equivalent terminal commands)
        what = []
        if run_classical and run_quantum:
            what.append("classical + quantum")
        elif run_classical:
            what.append("classical only")
        else:
            what.append("quantum only")
        if run_ideal_noisy:
            what.append("ideal then noisy config")
        else:
            what.append("single run (ideal config)")
        st.info(f"**This run:** {', '.join(what)} · **Repeats:** {int(runs)}")
        if run_quantum and run_ideal_noisy:
            st.caption("Ideal = noiseless simulator; Noisy = simulator with device-like noise. Compare results in Results and Experiments.")
        if not run_quantum and run_ideal_noisy:
            st.caption("Classical only: both ideal and noisy runs train the same classical models (quantum config has no effect).")

        # Build command list for display
        def cmd_str(cmd):
            return " ".join(cmd) if isinstance(cmd, list) else cmd

        commands_to_run = []
        for _ in range(int(runs)):
            commands_to_run.append([python_exe, str(script), "--quantum_config_path", "config/quantum_config_ideal.yaml"] + base_args)
            if run_ideal_noisy:
                commands_to_run.append([python_exe, str(script), "--quantum_config_path", "config/quantum_config_noisy.yaml"] + base_args)

        with st.expander("Equivalent terminal commands (copy to run locally)", expanded=False):
            for i, cmd in enumerate(commands_to_run):
                label = "ideal" if (i % 2 == 0) else "noisy" if run_ideal_noisy else "run"
                st.code(cmd_str(cmd), language="bash")
            st.caption("From project root; use python3 if python is not available.")

        any_failed = False
        last_output = ""

        for run_idx in range(int(runs)):
            st.write(f"Repeat {run_idx + 1}/{runs}")
            cmd_ideal = [python_exe, str(script), "--quantum_config_path", "config/quantum_config_ideal.yaml"] + base_args
            log_container.code("Running pipeline (ideal config)...")
            rc_ideal, out_ideal = run_command(cmd_ideal, log_container)
            if rc_ideal != 0:
                any_failed = True
                last_output = out_ideal
            if run_ideal_noisy:
                cmd_noisy = [python_exe, str(script), "--quantum_config_path", "config/quantum_config_noisy.yaml"] + base_args
                log_container.code("Running pipeline (noisy config)...")
                rc_noisy, out_noisy = run_command(cmd_noisy, log_container)
                if rc_noisy != 0:
                    any_failed = True
                    last_output = out_noisy

        st.cache_data.clear()
        st.cache_resource.clear()
        if any_failed:
            st.error(
                "Benchmark run failed. Expand **Last run output** below to see the exact error. "
                "If dependencies are missing, use **Generate demo results** or upload results from a local run."
            )
            if last_output:
                with st.expander("Last run output", expanded=True):
                    st.code(last_output[-4000:])
        else:
            st.success("Finished running benchmarks. Refresh the Overview or Results tab to see the latest metrics.")

    st.subheader("Generate demo results")
    st.markdown(
        "Create minimal result files so Overview and Results tabs show sample data (no pipeline run). "
        "Use this when the full pipeline cannot run (e.g. on Hugging Face Spaces) so you can still explore the dashboard."
    )
    if st.button("Generate demo results", key="gen_demo_results"):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        # latest_run.csv – one row with both classical and quantum (classical best, like real runs)
        latest = pd.DataFrame([{
            "classical_pr_auc": 0.8859,
            "quantum_pr_auc": 0.4159,
            "classical_accuracy": 0.8937,
            "quantum_accuracy": 0.5556,
            "classical_num_parameters": 49,
            "quantum_num_parameters": 24,
            "execution_mode": "simulator",
            "noise_model": "ideal",
            "backend_label": "ideal_sim",
            "qml_model_type": "QSVC",
            "qml_num_qubits": 12,
            "qml_feature_map_type": "ZZFeatureMap",
            "qml_relation": "CtD",
        }])
        latest.to_csv(LATEST_RUN, index=False)
        # experiment_history.csv – classical + quantum runs (ideal and noisy)
        history = pd.DataFrame([
            {"execution_mode": "simulator", "noise_model": "ideal", "backend_label": "ideal_sim", "classical_pr_auc": 0.8859, "quantum_pr_auc": 0.4159, "classical_accuracy": 0.8937, "quantum_accuracy": 0.5556, "classical_num_parameters": 49, "quantum_num_parameters": 24},
            {"execution_mode": "simulator", "noise_model": "noise", "backend_label": "noisy_sim", "classical_pr_auc": 0.8859, "quantum_pr_auc": 0.38, "classical_accuracy": 0.8937, "quantum_accuracy": 0.52, "classical_num_parameters": 49, "quantum_num_parameters": 24},
        ])
        history.to_csv(HISTORY_FILE, index=False)
        # optimized_results_*.json – full model ranking so Results tab shows all models (classical, quantum, hybrid)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        demo_ranking = [
            {"name": "LogisticRegression-L2", "type": "classical", "pr_auc": 0.8859, "accuracy": 0.8937, "fit_time": 0.35},
            {"name": "RandomForest-Optimized", "type": "classical", "pr_auc": 0.7586, "accuracy": 0.8406, "fit_time": 0.67},
            {"name": "Ensemble-RF-LR", "type": "classical", "pr_auc": 0.7498, "accuracy": 0.9082, "fit_time": 1.14},
            {"name": "Hybrid-Quantum-Classical", "type": "quantum", "pr_auc": 0.7421, "accuracy": 0.9034, "fit_time": 0.0},
            {"name": "QSVC-Optimized", "type": "quantum", "pr_auc": 0.4159, "accuracy": 0.5556, "fit_time": 0.94},
            {"name": "QSVC-Optimized-Calibrated", "type": "quantum", "pr_auc": 0.3344, "accuracy": 0.6280, "fit_time": 0.94},
        ]
        demo_opt_path = RESULTS_DIR / f"optimized_results_{stamp}.json"
        with open(demo_opt_path, "w") as f:
            json.dump({"config": {}, "ranking": demo_ranking, "timestamp": stamp}, f, indent=2)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Demo results written (latest_run, history, and full model ranking). Refreshing...")
        st.rerun()

    st.subheader("Upload results from a local run")
    st.markdown(
        "If you ran benchmarks locally, upload `latest_run.csv` or `experiment_history.csv` to view them here. "
        "Files are saved under `results/` and will appear in Overview, Results, Experiments, and Comparison."
    )
    uploaded = st.file_uploader("CSV file", type=["csv"], key="benchmark_upload")
    if uploaded is not None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / uploaded.name
        out_path.write_bytes(uploaded.getvalue())
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success(f"Saved to `{out_path}`. Refresh the Overview or Results tab.")

# ==============================
# PAGE 5: BACKEND STATUS
# ==============================
elif page == "9. Hardware readiness (backend status)":
    st.header("IBM Quantum backend status (access + queue)")
    st.markdown("""
    **In plain terms:** A **backend** is the machine (simulator or real quantum chip) that runs your circuits.
    This page checks whether IBM Quantum backends (e.g. Brisbane) are reachable with your token and shows **operational** status and **pending jobs** in the queue.
    """)
    _expander_for_term("backend", "What is a backend?")
    st.caption("Quick check: are IBM Quantum backends (e.g. ibm_brisbane) reachable with your credentials?")

    st.subheader("Credentials")
    with st.expander("Security: how your API key is handled", expanded=True):
        st.markdown("""
        - **Never stored:** Your API key is not written to disk, session state, or any database.
        - **Never logged:** The key is never sent to logs or error messages (we redact it if anything goes wrong).
        - **Used only for this check:** The token is sent to IBM Quantum only when you click **Check available backends**, solely to list backends and their status. We do not run jobs or change your account.
        - **You stay in control:** On a shared computer, clear the field or close the tab when done. The dashboard does not retain the key after the request.
        """)
    api_key = st.text_input(
        "IBM Quantum API key (optional)",
        value="",
        type="password",
        placeholder="Paste your token here (not stored)",
        help="Enter your IBM Quantum API token to test backend access. If left blank, the app uses IBM_Q_TOKEN or QISKIT_IBM_TOKEN from the environment. Your key is only used for this check and is not saved.",
    )
    st.caption(
        "Get a token at [quantum.ibm.com](https://quantum.ibm.com) → Account → API token. "
        "The key is sent only when you click **Check available backends** and is not stored by the dashboard."
    )

    # Dropdown: use list from last successful check if available, else known IBM backends
    if "hardware_backend_list" not in st.session_state or not st.session_state.get("hardware_backend_list"):
        _known_backends = [
            "ibm_brisbane", "ibm_kyoto", "ibm_osaka", "ibm_torino",
            "ibm_sherbrooke", "ibm_guadalupe", "ibm_nazca", "ibm_peekskill",
            "ibm_algiers", "ibm_cairo", "simulator_statevector", "simulator_extended_stabilizer",
        ]
    else:
        _known_backends = sorted(st.session_state["hardware_backend_list"])
    backend_options = ["All"] + _known_backends
    backend_choice = st.selectbox(
        "Backend to check",
        options=backend_options,
        index=0,
        help="Select a backend to verify or filter the list. 'All' shows every backend from your account after you run the check.",
    )
    backend_hint = "" if backend_choice == "All" else backend_choice
    instance_hint = st.text_input(
        "IBM instance",
        value=os.environ.get("IBM_INSTANCE", "ibm-q/open/main"),
        help="IBM Quantum instance path (channel/group/project). Default: ibm-q/open/main.",
    )

    st.caption("Click the button below to fetch the list of backends you can access and their operational status (and to refresh the **Backend to check** dropdown).")
    if st.button("Check available backends"):
        token = None
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService

            token = (api_key.strip() if api_key else None) or os.environ.get("IBM_Q_TOKEN") or os.environ.get("QISKIT_IBM_TOKEN")
            if not token:
                st.error(
                    "No IBM Quantum API key provided. Enter your token in **IBM Quantum API key** above, or set "
                    "`IBM_Q_TOKEN` (or `QISKIT_IBM_TOKEN`) in your environment."
                )
            else:
                token = token.strip()
                service = QiskitRuntimeService(channel="ibm_quantum", token=token, instance=instance_hint.strip())
                backends = service.backends()

                rows = []
                for b in backends:
                    name = getattr(b, "name", None) or getattr(b, "backend_name", None) or str(b)
                    operational = None
                    status_msg = None
                    pending = None
                    try:
                        stt = b.status()
                        operational = getattr(stt, "operational", None)
                        status_msg = getattr(stt, "status_msg", None)
                        pending = getattr(stt, "pending_jobs", None)
                    except Exception:
                        pass

                    rows.append({
                        "backend": name,
                        "operational": operational,
                        "status_msg": status_msg,
                        "pending_jobs": pending,
                        "num_qubits": getattr(b, "num_qubits", None),
                    })

                dfb = pd.DataFrame(rows).sort_values(by=["backend"])
                st.session_state["hardware_backend_list"] = dfb["backend"].astype(str).tolist()

                st.caption("**operational**: whether the backend is accepting jobs. **status_msg**: short status text. **pending_jobs**: number of jobs in the queue. **num_qubits**: backend size.")
                if backend_hint:
                    dfb_display = dfb[dfb["backend"].astype(str) == backend_hint]
                    if not dfb_display.empty:
                        st.success(f"`{backend_hint}` is accessible.")
                        st.dataframe(dfb_display, width="stretch")
                    else:
                        st.warning(f"`{backend_hint}` not found in your accessible backends list.")
                        st.dataframe(dfb, width="stretch")
                else:
                    st.dataframe(dfb, width="stretch")
        except Exception as e:
            safe_msg = _redact_token_from_message(str(e), token)
            st.error(f"Backend status check failed: {safe_msg}")

# ==============================
# PAGE: RUN YOUR CODE
# ==============================
elif page == "10. Run your code (advanced)":
    st.header("Upload & run your code")
    st.markdown("""
    Upload a **Python script** (or paste code) and run it here. Your code runs in a subprocess with the **project root** as the working directory,
    so you can import project modules (e.g. `kg_layer`, `quantum_layer`, `benchmarking`) and read/write under the project.
    Use this to try small experiments, run custom analyses, or reproduce results without leaving the dashboard.
    """)
    st.caption(
        "**Security:** Code runs in an isolated subprocess with a **time limit**. Do not upload code you do not trust. "
        "On shared or hosted environments, avoid secrets and long-running or resource-heavy jobs."
    )

    with st.expander("What can I do?", expanded=False):
        st.markdown("""
        - **Working directory:** Your script runs with `cwd` = project root (same as **Run benchmarks**).
        - **Imports:** You can `import kg_layer`, `quantum_layer`, and use `data/`, `results/`, `config/` paths.
        - **Output:** Anything you `print()` goes to **stdout**; exceptions and tracebacks go to **stderr**. Both are shown below.
        - **Results:** Write CSV or plots to `results/` and open them from **Results** or **Experiments**, or print summaries for quick checks.
        """)

    source = st.radio(
        "Code source",
        ["Upload a .py file", "Paste code"],
        index=0,
        help="Upload a Python file or type/paste code in the text area.",
    )

    script_content = None
    if source == "Upload a .py file":
        uploaded = st.file_uploader("Python script", type=["py"], key="run_code_upload")
        if uploaded is not None:
            script_content = uploaded.read().decode("utf-8", errors="replace")
            st.code(script_content, language="python")
    else:
        script_content = st.text_area(
            "Paste your Python code",
            height=280,
            placeholder="# Example: load project data and print a summary\nimport pandas as pd\nfrom pathlib import Path\npath = Path('results/latest_run.csv')\nif path.exists():\n    df = pd.read_csv(path)\n    print(df.to_string())\nelse:\n    print('No latest_run.csv yet')",
            help="Code is written to a temporary .py file and run with the project root as cwd.",
            key="run_code_paste",
        )

    timeout_sec = st.number_input(
        "Timeout (seconds)",
        min_value=5,
        max_value=300,
        value=60,
        step=5,
        help="Stop the script after this many seconds if it has not finished.",
    )

    if st.button("Run code", type="primary", key="run_code_btn"):
        if not script_content or not script_content.strip():
            st.warning("Please provide code: upload a .py file or paste code in the text area.")
        else:
            with st.spinner("Running your script..."):
                try:
                    returncode, stdout, stderr = run_user_script(script_content.strip(), timeout_sec)
                    st.subheader("Results")
                    col_ret, col_time = st.columns(2)
                    with col_ret:
                        st.metric("Return code", returncode)
                    with col_time:
                        st.caption("Stdout and stderr are shown below.")
                    if stdout:
                        st.text_area("Stdout", value=stdout, height=200, key="run_stdout", disabled=True)
                    else:
                        st.caption("*(no stdout)*")
                    if stderr:
                        st.text_area("Stderr", value=stderr, height=120, key="run_stderr", disabled=True)
                    if returncode == 0 and not stderr:
                        st.success("Script finished successfully.")
                    elif returncode != 0:
                        st.error("Script exited with a non-zero return code.")
                except subprocess.TimeoutExpired:
                    st.error(f"Script was stopped after {timeout_sec} seconds (timeout).")
                except Exception as e:
                    st.error(f"Failed to run script: {e}")

# Footer
st.markdown("---")
st.caption("Hybrid quantum–classical knowledge graph system for biomedical link prediction (proof of concept)")