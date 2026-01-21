# benchmarking/dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import shlex
import logging
from pathlib import Path
import json
import time
import subprocess
import joblib
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Hybrid QML-KG Benchmark Dashboard",
    page_icon=None,
    layout="wide"
)

# Title and description
st.title("Hybrid QML-KG Biomedical Link Prediction")
st.markdown("""
This dashboard summarizes a hybrid quantum-classical knowledge graph pipeline for
drug-disease link prediction on **Hetionet**. It highlights what was built,
what was tested, and how quantum models compare to classical baselines.
""")

# Paths
RESULTS_DIR = Path("results")
LATEST_RUN = RESULTS_DIR / "latest_run.csv"
HISTORY_FILE = RESULTS_DIR / "experiment_history.csv"
SCALING_PLOT = Path("docs/scaling_projection.png")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ensure local project modules (kg_layer/, quantum_layer/, etc.) are importable in Streamlit
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Optional: node metadata (human-readable names for node IDs)
NODE_METADATA_CSV = PROJECT_ROOT / "data" / "hetionet_nodes_metadata.csv"
HETIONET_NODES_TSV = PROJECT_ROOT / "data" / "hetionet-v1.0-nodes.tsv"

# ------------------------------
# Human-readable node labeling
# ------------------------------
@st.cache_resource
def load_node_metadata():
    try:
        from kg_layer.node_metadata import load_node_metadata_csv
        return load_node_metadata_csv(NODE_METADATA_CSV)
    except Exception:
        return {}

@st.cache_data(show_spinner=False)
def node_metadata_coverage() -> dict:
    """
    Returns coverage stats for metadata names, split by common Hetionet node types.
    """
    meta = load_node_metadata() or {}
    keys = list(meta.keys())
    def _named(k: str) -> bool:
        m = meta.get(k)
        return bool(getattr(m, "name", None))
    total = len(keys)
    named = sum(1 for k in keys if _named(k))
    comp = [k for k in keys if k.startswith("Compound::")]
    dis = [k for k in keys if k.startswith("Disease::")]
    cov = {
        "total_nodes": int(total),
        "named_nodes": int(named),
        "compound_total": int(len(comp)),
        "compound_named": int(sum(1 for k in comp if _named(k))),
        "disease_total": int(len(dis)),
        "disease_named": int(sum(1 for k in dis if _named(k))),
    }
    cov["named_frac"] = float(cov["named_nodes"] / cov["total_nodes"]) if cov["total_nodes"] else 0.0
    cov["compound_named_frac"] = float(cov["compound_named"] / cov["compound_total"]) if cov["compound_total"] else 0.0
    cov["disease_named_frac"] = float(cov["disease_named"] / cov["disease_total"]) if cov["disease_total"] else 0.0
    return cov

def render_names_status_panel():
    """
    Renders a global panel to explain and fix missing human-readable names.
    """
    cov = node_metadata_coverage()
    with st.sidebar:
        st.subheader("Names (metadata)")
        if cov["total_nodes"] == 0:
            st.warning("Node metadata not loaded yet.")
        else:
            st.caption(
                f"Named nodes: {cov['named_nodes']}/{cov['total_nodes']} "
                f"({cov['named_frac']*100:.1f}%)"
            )
            st.caption(
                f"Compounds named: {cov['compound_named']}/{cov['compound_total']} "
                f"({cov['compound_named_frac']*100:.1f}%)"
            )
            st.caption(
                f"Diseases named: {cov['disease_named']}/{cov['disease_total']} "
                f"({cov['disease_named_frac']*100:.1f}%)"
            )

        st.caption("If names are missing, the dashboard falls back to readable IDs (e.g., `Compound DB00688`).")
        can_local = HETIONET_NODES_TSV.exists()
        allow_download = st.checkbox("Allow downloading nodes table if missing", value=False)
        if st.button("Enrich names now"):
            log_container = st.empty()
            cmd = [sys.executable, "scripts/enrich_node_metadata.py"]
            if allow_download and not can_local:
                cmd.append("--download_if_missing")
            rc = run_command(cmd, log_container)
            if rc == 0:
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Name enrichment completed. Refresh pages/dropdowns.")
            else:
                st.error("Name enrichment failed. See logs above.")


# NOTE: render_names_status_panel() is invoked after run_command() is defined (further below)


def _parse_node_id(node_id: str) -> tuple[str | None, str | None]:
    try:
        if not isinstance(node_id, str):
            return None, None
        if "::" not in node_id:
            return None, node_id
        kind, rest = node_id.split("::", 1)
        return kind.strip() or None, rest.strip() or None
    except Exception:
        return None, None

def node_label(node_id: str, node_meta: dict) -> str:
    """
    Friendly display string for a Hetionet node ID.
    Falls back to a readable 'Type identifier' even if `name` is missing in metadata.
    """
    if not isinstance(node_id, str) or not node_id:
        return str(node_id)
    try:
        m = node_meta.get(node_id)
        if m is not None and getattr(m, "name", None):
            return f"{m.name} ({node_id})"
        kind, ident = _parse_node_id(node_id)
        if kind and ident:
            # Display without double-colons so non-technical audiences can read it quickly.
            ns = getattr(m, "namespace", None) if m is not None else None
            if ns:
                return f"{kind} {ident} ({ns})"
            return f"{kind} {ident}"
        return node_id
    except Exception:
        return node_id

def add_node_labels(df: pd.DataFrame, node_meta: dict, cols: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[f"{c}_label"] = out[c].apply(lambda x: node_label(str(x), node_meta))
    return out

# Load latest results
@st.cache_data
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

@st.cache_data
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
def load_quantum_kernel(qml_dim: int, reps: int = 2):
    """
    Return a statevector fidelity kernel for fast local 'quantum similarity' scoring.
    Note: this returns a kernel similarity, not a calibrated probability.
    """
    try:
        from qiskit.circuit.library import ZZFeatureMap
        from qiskit_machine_learning.kernels import FidelityStatevectorKernel
    except Exception as e:
        return None, str(e)

    fm = ZZFeatureMap(feature_dimension=qml_dim, reps=reps, entanglement="linear")
    return FidelityStatevectorKernel(feature_map=fm), None

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
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=PROJECT_ROOT
        )
        output = []
        for line in proc.stdout:
            output.append(line)
            log_container.code("".join(output[-200:]))
        proc.wait()
        return proc.returncode
    except Exception as e:
        log_container.error(f"Failed to run command: {e}")
        return 1

# Global metadata status + fix button (generic drug names, disease names, etc.)
render_names_status_panel()

# Sidebar: Navigation
st.sidebar.title("Navigation")
if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

page = st.sidebar.radio(
    "Go to",
    [
        "Overview (scientific summary)",
        "Results (latest run)",
        "Live prediction (interactive)",
        "Experiments (history)",
        "Comparison (classical vs quantum)",
        "Knowledge graph (inventory)",
        "Findings (ranked hypotheses)",
        "Run benchmarks",
        "Hardware readiness (backend status)",
    ]
)

# ==============================
# PAGE 0: PROJECT STORY
# ==============================
if page == "Overview (scientific summary)":
    st.header("Overview: hybrid link prediction over the Hetionet biomedical knowledge graph")

    st.markdown("""
This project is a **hybrid quantum–classical link prediction pipeline** over the **Hetionet** biomedical knowledge graph.
It predicts whether a given **Compound** is likely to **treat** a given **Disease** (e.g., the CtD relation).

This is a research/prototyping system: it produces **ranking signals** and **benchmarks**, not clinical guidance.
""")

    st.subheader("Goal and hypothesis")
    st.markdown("""
- **Goal**: rank candidate Compound→Disease links so that true treatments rise to the top.
- **Hypothesis**: graph embeddings + engineered features give strong classical baselines; quantum models can be competitive on the same reduced feature space when the signal is real and leakage-free.
""")

    st.subheader("Task definition")
    st.markdown("""
- **Input**: Hetionet triples (edges) and a target relation (e.g., CtD)
- **Task**: binary link prediction (edge exists vs not)
- **Output**: a probability/score used for ranking candidate edges
""")

    st.subheader("Baseline stack (what we compare)")
    st.markdown("""
- **Classical**: Logistic Regression / Random Forest / ExtraTrees / HistGradientBoosting over embedding-derived features
- **PyKEEN (direct)**: RotatE/ComplEx link prediction on a context graph (scoring triples directly)
- **GNN**: GraphSAGE/GIN-style link prediction with edge decoder
- **Quantum**: QSVC / VQC over reduced, qubit-sized features
""")

    st.subheader("Recommended run ladder (increasing cost, better evidence)")
    st.markdown("""
1. **Cheapest sanity**: tiny sample to verify the pipeline and UI wiring.
2. **Classical ceiling**: full classical baselines on larger data; validates signal.
3. **PyKEEN direct**: sanity-check link prediction with a KG-native baseline.
4. **GNN baseline**: compares a modern graph model to embedding+ML.
5. **Quantum vs classical**: run QSVC/VQC alongside classical for fairness.
6. **Learning curve**: sweep sample sizes to see how PR-AUC scales.
""")

    st.subheader("Representations and model inputs")
    st.markdown("""
- **Embeddings**: ComplEx / RotatE / DistMult with optional full-graph context.
- **Classical features**: [h, t, |h−t|, h⊙t] plus graph/domain features.
- **Quantum features**: reduced vectors sized to the number of qubits (PCA + feature selection).
""")

    st.subheader("Benchmarking context and current status")
    if df_latest is None:
        st.warning("No `results/latest_run.csv` found yet. Run a benchmark from 'Run Benchmarks'.")
    else:
        classical_pr = float(df_latest["classical_pr_auc"].iloc[0]) if "classical_pr_auc" in df_latest.columns else np.nan
        quantum_pr = float(df_latest["quantum_pr_auc"].iloc[0]) if "quantum_pr_auc" in df_latest.columns else np.nan
        pykeen_pr = float(df_latest["pykeen_pr_auc"].iloc[0]) if "pykeen_pr_auc" in df_latest.columns else np.nan
        gnn_pr = float(df_latest["gnn_pr_auc"].iloc[0]) if "gnn_pr_auc" in df_latest.columns else np.nan
        
        # Show best performance
        all_prs = [("Classical", classical_pr), ("Quantum", quantum_pr), ("PyKEEN", pykeen_pr), ("GNN", gnn_pr)]
        valid_prs = [(name, pr) for name, pr in all_prs if not pd.isna(pr)]
        if valid_prs:
            best_name, best_pr = max(valid_prs, key=lambda x: x[1])
            st.success(f"🏆 **Current Best**: {best_name} with PR-AUC = {best_pr:.4f}")
            if best_pr >= 0.80:
                st.balloons()
        
        st.json({
            "classical_pr_auc": classical_pr,
            "quantum_pr_auc": quantum_pr,
            "pykeen_pr_auc": pykeen_pr if not pd.isna(pykeen_pr) else None,
            "gnn_pr_auc": gnn_pr if not pd.isna(gnn_pr) else None,
            "execution_mode": str(safe_get(df_latest, "execution_mode", "N/A")),
            "noise_model": str(safe_get(df_latest, "noise_model", "N/A")),
            "backend_label": str(safe_get(df_latest, "backend_label", "N/A")),
            "qml_model_type": str(safe_get(df_latest, "qml_model_type", "N/A")),
            "qml_num_qubits": str(safe_get(df_latest, "qml_num_qubits", "N/A")),
        })

    if df_history is not None and len(df_history) > 0:
        st.subheader("Ideal vs noisy snapshot (latest per mode)")
        exec_summary = latest_execution_summary(df_history)
        if not exec_summary.empty:
            st.dataframe(exec_summary)
        else:
            st.info("No execution metadata columns found in history yet.")

    st.subheader("How to interpret scores and claims")
    st.markdown("""
- **PR-AUC**: best for imbalanced link prediction; higher is better.
- **Ideal vs noisy**: shows sensitivity to noise model / backend.
- **Leakage guard**: cached embeddings or graphs that include test positives can inflate scores; use leakage-safe settings for honest metrics.
- **A single score is not evidence**: use the “Evidence” section in Live Prediction.
""")

    st.subheader("Limitations and next steps")
    st.markdown("""
- Add candidate ranking workflows (“top compounds for a disease”)
- Add evidence/interpretability (neighbors + KG context)
- Add robust evaluation (seeds/CV) and artifact linking per run
""")

    st.subheader("Human-readable compound and disease names (metadata enrichment)")
    st.markdown("""
This dashboard can display **human-readable names** for compounds and diseases if `data/hetionet_nodes_metadata.csv`
contains a populated `name` column.

If names are missing, the dashboard falls back to readable IDs like `Compound DB00688` and `Disease DOID:0060048`.

To enrich names (best-effort; may require network access):
`python3 scripts/enrich_node_metadata.py --download_if_missing`
""")

    st.markdown("You can also run enrichment directly from this dashboard (recommended once, then reuse the cached file).")
    col_enrich_a, col_enrich_b = st.columns([2, 3])
    with col_enrich_a:
        allow_download = st.checkbox("Allow downloading Hetionet nodes table if missing", value=False)
        st.caption("Enable only if your environment allows outbound network access.")
    with col_enrich_b:
        if st.button("Enrich names now"):
            log_container = st.empty()
            cmd = [sys.executable, "scripts/enrich_node_metadata.py"]
            if allow_download:
                cmd.append("--download_if_missing")
            rc = run_command(cmd, log_container)
            if rc == 0:
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Metadata enrichment completed. Refresh dropdowns/tables or click Refresh data in the sidebar.")
            else:
                st.error("Metadata enrichment failed. See logs above; you may need to provide a local nodes table in `data/`.")

# ==============================
# PAGE 1: RESULTS OVERVIEW
# ==============================
elif page == "Results (latest run)":
    st.header("Results: latest run summary")

    # Recent improvements highlight
    if df_latest is not None:
        classical_pr = safe_get(df_latest, "classical_pr_auc", np.nan)
        quantum_pr = safe_get(df_latest, "quantum_pr_auc", np.nan)
        pykeen_pr = safe_get(df_latest, "pykeen_pr_auc", np.nan) if "pykeen_pr_auc" in df_latest.columns else np.nan
        gnn_pr = safe_get(df_latest, "gnn_pr_auc", np.nan) if "gnn_pr_auc" in df_latest.columns else np.nan
        
        max_pr = max([pr for pr in [classical_pr, quantum_pr, pykeen_pr, gnn_pr] if not pd.isna(pr)], default=np.nan)
        
        if not pd.isna(max_pr) and max_pr >= 0.75:
            st.info(f"""
            🎯 **Recent Achievement**: Models are achieving **PR-AUC ≥ 0.75**, with best performance at **{max_pr:.4f}**.
            This demonstrates strong signal extraction from the Hetionet knowledge graph.
            """)
        elif not pd.isna(max_pr) and max_pr >= 0.70:
            st.info(f"""
            📈 **Progress**: Models are achieving **PR-AUC ≥ 0.70**, with best performance at **{max_pr:.4f}**.
            Continue tuning to reach the 0.75-0.80 target range.
            """)

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
- **Recent improvements**: Evidence weighting, contrastive learning, LDA reduction, and advanced negative sampling
""")

    st.subheader("Latest run snapshot")
    run_cols = st.columns(4)
    qml_model_type = safe_get(df_latest, "qml_model_type", "N/A")
    qml_num_qubits = safe_get(df_latest, "qml_num_qubits", "N/A")
    qml_feature_map = safe_get(df_latest, "qml_feature_map_type", "N/A")
    exec_mode = safe_get(df_latest, "execution_mode", "N/A")
    noise_model = safe_get(df_latest, "noise_model", "N/A")
    backend_label = safe_get(df_latest, "backend_label", "N/A")

    run_cols[0].metric("QML Model", qml_model_type)
    run_cols[1].metric("Qubits", qml_num_qubits)
    run_cols[2].metric("Feature Map", qml_feature_map)
    run_cols[3].metric("Execution", exec_mode)

    st.caption(f"Backend: {backend_label} | Noise: {noise_model}")

    # Scoreboard: summarize all baselines present in this run.
    st.subheader("Scoreboard (PR-AUC)")
    rows = []
    if df_latest is not None:
        def _row(name: str, pr: str, acc: str | None = None, extra: dict | None = None):
            r = {"Model": name, "PR-AUC": safe_get(df_latest, pr, np.nan)}
            if acc:
                r["Accuracy"] = safe_get(df_latest, acc, np.nan)
            if extra:
                r.update(extra)
            rows.append(r)

        _row("Classical", "classical_pr_auc", "classical_accuracy", {"Type": safe_get(df_latest, "classical_model_type", "")})
        _row("Quantum", "quantum_pr_auc", "quantum_accuracy", {"Type": safe_get(df_latest, "quantum_model_type", "")})
        if "pykeen_pr_auc" in df_latest.columns:
            _row("PyKEEN (direct)", "pykeen_pr_auc", None, {"Type": safe_get(df_latest, "pykeen_model_type", "")})
        if "gnn_pr_auc" in df_latest.columns:
            _row("GNN", "gnn_pr_auc", None, {"Type": safe_get(df_latest, "gnn_model_type", "")})

    if rows:
        df_scoreboard = pd.DataFrame(rows)
        # Sort by PR-AUC descending, handling NaN
        df_scoreboard = df_scoreboard.sort_values("PR-AUC", ascending=False, na_last=True)
        
        # Add performance badges
        def _get_badge(pr_auc):
            if pd.isna(pr_auc):
                return ""
            if pr_auc >= 0.80:
                return "🏆 Excellent (≥0.80)"
            elif pr_auc >= 0.75:
                return "⭐ Strong (≥0.75)"
            elif pr_auc >= 0.70:
                return "✓ Good (≥0.70)"
            else:
                return ""
        
        df_scoreboard["Performance"] = df_scoreboard["PR-AUC"].apply(_get_badge)
        
        # Highlight best model
        best_idx = df_scoreboard["PR-AUC"].idxmax() if not df_scoreboard["PR-AUC"].isna().all() else None
        if best_idx is not None:
            best_model = df_scoreboard.loc[best_idx, "Model"]
            best_pr = df_scoreboard.loc[best_idx, "PR-AUC"]
            st.success(f"🏆 **Best Model**: {best_model} with PR-AUC = {best_pr:.4f}")
        
        st.dataframe(df_scoreboard, width="stretch", hide_index=True)

    # Dataset-size sanity (helps interpret PR-AUC stability)
    if df_latest is not None and "obs_data_test_n" in df_latest.columns and "obs_data_pos_edges_total" in df_latest.columns:
        try:
            tn = int(float(df_latest["obs_data_test_n"].iloc[0]))
            tp = int(float(df_latest["obs_data_test_pos"].iloc[0])) if "obs_data_test_pos" in df_latest.columns else None
            total_pos = int(float(df_latest["obs_data_pos_edges_total"].iloc[0]))
            st.caption(f"Dataset: test_n={tn}, test_pos={tp}, positives_total={total_pos}")
        except Exception:
            pass

    # Warn if metrics look suspiciously perfect on non-trivial test sets (often indicates leakage)
    try:
        tn = int(float(df_latest["obs_data_test_n"].iloc[0])) if df_latest is not None and "obs_data_test_n" in df_latest.columns else 0
        q_pr = float(df_latest["quantum_pr_auc"].iloc[0]) if df_latest is not None and "quantum_pr_auc" in df_latest.columns else None
        c_pr = float(df_latest["classical_pr_auc"].iloc[0]) if df_latest is not None and "classical_pr_auc" in df_latest.columns else None
        if tn >= 200 and ((q_pr is not None and abs(q_pr - 1.0) < 1e-12) or (c_pr is not None and abs(c_pr - 1.0) < 1e-12)):
            st.warning("Perfect PR-AUC on a large test set is unusual. If you recently changed representation learning, verify leakage guards are active (held-out positives excluded from representation training).")
    except Exception:
        pass

    # Optional: PyKEEN direct scoring baseline (if present in latest_run.csv)
    if df_latest is not None and "pykeen_pr_auc" in df_latest.columns:
        st.subheader("PyKEEN direct scoring baseline (link prediction model)")
        pk_cols = st.columns(3)
        pk_cols[0].metric("PyKEEN PR-AUC", f"{float(df_latest['pykeen_pr_auc'].iloc[0]):.4f}")
        if "pykeen_model_type" in df_latest.columns:
            pk_cols[1].metric("PyKEEN Model", str(df_latest["pykeen_model_type"].iloc[0]))
        if "pykeen_train_seconds" in df_latest.columns:
            try:
                pk_cols[2].metric("PyKEEN Train (s)", f"{float(df_latest['pykeen_train_seconds'].iloc[0]):.1f}")
            except Exception:
                pk_cols[2].metric("PyKEEN Train (s)", str(df_latest["pykeen_train_seconds"].iloc[0]))

    # Optional: GNN baseline (if present in latest_run.csv)
    if df_latest is not None and "gnn_pr_auc" in df_latest.columns:
        st.subheader("GNN baseline (GraphSAGE/GIN)")
        g_cols = st.columns(3)
        g_cols[0].metric("GNN PR-AUC", f"{float(df_latest['gnn_pr_auc'].iloc[0]):.4f}")
        if "gnn_model_type" in df_latest.columns:
            g_cols[1].metric("GNN Model", str(df_latest["gnn_model_type"].iloc[0]))
        if "gnn_train_seconds" in df_latest.columns:
            try:
                g_cols[2].metric("GNN Train (s)", f"{float(df_latest['gnn_train_seconds'].iloc[0]):.1f}")
            except Exception:
                g_cols[2].metric("GNN Train (s)", str(df_latest["gnn_train_seconds"].iloc[0]))

    st.header("Model Performance Comparison")
    
    if df_latest is not None:
        # Extract metrics
        classical_pr_auc = df_latest['classical_pr_auc'].iloc[0]
        quantum_pr_auc = df_latest['quantum_pr_auc'].iloc[0]
        classical_params = df_latest['classical_num_parameters'].iloc[0]
        quantum_params = df_latest['quantum_num_parameters'].iloc[0]
        classical_acc = df_latest['classical_accuracy'].iloc[0]
        quantum_acc = df_latest['quantum_accuracy'].iloc[0]
        
        # Metrics display
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
            st.image(str(SCALING_PLOT), caption="Projected runtime as KG size increases", use_column_width=True)
        else:
            st.warning("Scaling projection plot not found. Run benchmarking/scalability_sim.py to generate it.")
        
        # Ideal vs noisy summary (if available)
        st.subheader("Ideal vs Noisy Summary")
        exec_summary = latest_execution_summary(df_history)
        if not exec_summary.empty:
            st.dataframe(exec_summary)
        else:
            st.info("Run the benchmark script to populate ideal vs noisy comparisons.")

        # Model configuration
        st.subheader("Quantum Model Configuration")
        qml_config = {}
        for col in df_latest.columns:
            if col.startswith('qml_'):
                key = col.replace('qml_', '')
                value = df_latest[col].iloc[0]
                qml_config[key] = value
                st.text(f"{key}: {value}")
    
    else:
        st.error("No results found. Please run the training pipeline first.")
        st.markdown("""
        To generate results:
        1. `bash scripts/benchmark_ideal_noisy.sh CtD results --fast_mode --quantum_only`
        2. `python benchmarking/ideal_vs_noisy_compare.py --results_dir results`
        """)

# ==============================
# PAGE 2: LIVE PREDICTION
# ==============================
elif page == "Live prediction (interactive)":
    st.header("Live prediction: interactive scoring and candidate ranking")
    st.markdown("""
    Enter a compound and disease to see a **link prediction score** from the trained classical model,
    plus candidate rankings and evidence context.

    This is a research demo and does **not** provide clinical or synthesis guidance.
    """)
    
    model, scaler, model_path, scaler_path = load_classical_artifacts()
    embeddings, entity_ids, emb_name = load_entity_embeddings()
    node_meta = load_node_metadata()

    st.caption(f"Classical model: `{model_path}` | Scaler: `{scaler_path}` | Embeddings: `{emb_name}`")

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
                    format_func=lambda x: node_label(x, node_meta),
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
                    format_func=lambda x: node_label(x, node_meta),
                    key="disease_select"
                )
            else:
                disease_input = st.text_input("Disease", key="disease_input", help="e.g., DOID_0060048 (COVID-19)")
        
        method = st.selectbox("Scoring Method", ["classical_model", "quantum_kernel_similarity"], index=0)
        top_k = st.slider("Top-K candidates", min_value=5, max_value=50, value=15, step=5)
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
                    st.code("\n".join([node_label(s, node_meta) for s in suggestions]))
                    st.stop()
                elif d_idx is None:
                    st.error(f"Disease not found in embeddings: {disease_id}")
                    st.info("Try one of these available diseases:")
                    suggestions = suggest_available(entity_ids, "Disease::", disease_id.split("::")[-1], k=15)
                    st.code("\n".join([node_label(s, node_meta) for s in suggestions]))
                    st.stop()
                else:
                    qml_dim = int(safe_get(df_latest, "qml_num_qubits", 12) or 12)
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
                    st.caption(f"Selected: **{node_label(compound_id, node_meta)}** → **{node_label(disease_id, node_meta)}**")

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
                        qk, err = load_quantum_kernel(qml_dim=qml_dim, reps=2)
                        if qk is None:
                            st.error(f"Quantum kernel unavailable: {err}")
                            st.stop()
                        x1 = to_quantum_input(c_red)
                        x2 = to_quantum_input(d_red)
                        sim = float(qk.evaluate(x1.reshape(1, -1), x2.reshape(1, -1))[0, 0])
                        st.success("Quantum similarity computed (kernel value)")
                        payload.update({
                            "score_type": "kernel_similarity",
                            "quantum_kernel_similarity": round(sim, 6),
                            "feature_map": "ZZFeatureMap(reps=2, entanglement=linear)",
                            "note": "Kernel similarity is not a calibrated probability.",
                        })
                        st.json(payload)

                    st.subheader("Candidate ranking")
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
                            qk, err = load_quantum_kernel(qml_dim=qml_dim, reps=2)
                            if qk is None:
                                st.error(f"Quantum kernel unavailable: {err}")
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
                            qk, err = load_quantum_kernel(qml_dim=qml_dim, reps=2)
                            if qk is None:
                                st.error(f"Quantum kernel unavailable: {err}")
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
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"Classical embedding ({emb_name}) – first 20 dims")
                        vec = embeddings[c_idx]
                        fig, ax = plt.subplots(figsize=(6, 3))
                        ax.bar(range(min(20, vec.shape[0])), vec[:min(20, vec.shape[0])])
                        st.pyplot(fig)
                    with col2:
                        st.caption(f"Reduced (PCA) vector – {qml_dim} dims (used by QML + classical features)")
                        fig2, ax2 = plt.subplots(figsize=(6, 3))
                        ax2.bar(range(qml_dim), c_red[:qml_dim])
                        st.pyplot(fig2)

                    st.subheader("Evidence")
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
    # These are examples; availability depends on the sampled embedding set
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
elif page == "Experiments (history)":
    st.header("Experiments: history of benchmarked runs")
    
    if df_history is not None and len(df_history) > 0:
        st.caption(
            "Use filters to isolate comparable runs. For trustworthy trends, prefer leakage-safe runs "
            "(no cached embeddings when hold-outs exist) and consistent sampling."
        )
        st.subheader("Filters")
        cols = st.columns(6)
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
        with cols[5]:
            require_pykeen = st.checkbox("Require PyKEEN", value=False)

        filtered = df_history.copy()
        if model_filter and "quantum_model_type" in filtered.columns:
            filtered = filtered[filtered["quantum_model_type"].isin(model_filter)]
        if exec_filter and "execution_mode" in filtered.columns:
            filtered = filtered[filtered["execution_mode"].isin(exec_filter)]
        if noise_filter and "noise_model" in filtered.columns:
            filtered = filtered[filtered["noise_model"].fillna("ideal").isin(noise_filter)]
        if hide_quantum_zero and "quantum_pr_auc" in filtered.columns:
            filtered = filtered[filtered["quantum_pr_auc"] > 0]
        if require_pykeen and "pykeen_pr_auc" in filtered.columns:
            filtered = filtered[filtered["pykeen_pr_auc"].notna()]
        filtered = filtered.tail(int(max_rows))

        st.subheader("Quick summary")
        summary_cols = st.columns(5)
        with summary_cols[0]:
            st.metric("Runs (filtered)", int(len(filtered)))
        with summary_cols[1]:
            c_mean = float(pd.to_numeric(filtered.get("classical_pr_auc"), errors="coerce").mean())
            c_max = float(pd.to_numeric(filtered.get("classical_pr_auc"), errors="coerce").max())
            st.metric("Classical PR-AUC", f"{c_mean:.4f}", delta=f"Max: {c_max:.4f}" if not pd.isna(c_max) else None)
        with summary_cols[2]:
            if "quantum_pr_auc" in filtered.columns:
                q_mean = float(pd.to_numeric(filtered.get("quantum_pr_auc"), errors="coerce").mean())
                q_max = float(pd.to_numeric(filtered.get("quantum_pr_auc"), errors="coerce").max())
                st.metric("Quantum PR-AUC", f"{q_mean:.4f}", delta=f"Max: {q_max:.4f}" if not pd.isna(q_max) else None)
        with summary_cols[3]:
            if "pykeen_pr_auc" in filtered.columns:
                pk_mean = float(pd.to_numeric(filtered.get("pykeen_pr_auc"), errors="coerce").mean())
                pk_max = float(pd.to_numeric(filtered.get("pykeen_pr_auc"), errors="coerce").max())
                st.metric("PyKEEN PR-AUC", f"{pk_mean:.4f}", delta=f"Max: {pk_max:.4f}" if not pd.isna(pk_max) else None)
        with summary_cols[4]:
            if "gnn_pr_auc" in filtered.columns:
                g_mean = float(pd.to_numeric(filtered.get("gnn_pr_auc"), errors="coerce").mean())
                g_max = float(pd.to_numeric(filtered.get("gnn_pr_auc"), errors="coerce").max())
                st.metric("GNN PR-AUC", f"{g_mean:.4f}", delta=f"Max: {g_max:.4f}" if not pd.isna(g_max) else None)
        
        # Best overall performance highlight
        all_maxes = []
        if not pd.isna(c_max):
            all_maxes.append(("Classical", c_max))
        if "quantum_pr_auc" in filtered.columns and not pd.isna(q_max):
            all_maxes.append(("Quantum", q_max))
        if "pykeen_pr_auc" in filtered.columns and not pd.isna(pk_max):
            all_maxes.append(("PyKEEN", pk_max))
        if "gnn_pr_auc" in filtered.columns and not pd.isna(g_max):
            all_maxes.append(("GNN", g_max))
        
        if all_maxes:
            best_name, best_val = max(all_maxes, key=lambda x: x[1])
            if best_val >= 0.75:
                st.success(f"🏆 **Best Performance in History**: {best_name} achieved PR-AUC = {best_val:.4f}")

        st.subheader("Artifacts")
        st.download_button(
            "Download filtered history (CSV)",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="experiment_history_filtered.csv",
            mime="text/csv"
        )

        # Metrics over time
        st.subheader("Performance Trends")
        
        # PR-AUC over experiments
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(filtered.index, filtered['classical_pr_auc'], 'o-', label='Classical PR-AUC', color='blue')
        ax.plot(filtered.index, filtered['quantum_pr_auc'], 's-', label='Quantum PR-AUC', color='purple')
        if "pykeen_pr_auc" in filtered.columns:
            ax.plot(filtered.index, filtered["pykeen_pr_auc"], '^-', label='PyKEEN PR-AUC', color='teal')
        if "gnn_pr_auc" in filtered.columns:
            ax.plot(filtered.index, filtered["gnn_pr_auc"], 'd-', label='GNN PR-AUC', color='orange')
        ax.set_xlabel('Experiment #')
        ax.set_ylabel('PR-AUC')
        ax.set_title('PR-AUC Over Experiments')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        if "quantum_pr_auc" in filtered.columns:
            st.subheader("Quantum vs Classical PR-AUC Delta")
            delta = filtered["quantum_pr_auc"] - filtered["classical_pr_auc"]
            fig3, ax3 = plt.subplots(figsize=(10, 3))
            ax3.bar(filtered.index, delta, color=["green" if v >= 0 else "red" for v in delta])
            ax3.axhline(0, color="black", linewidth=1)
            ax3.set_ylabel("Δ PR-AUC (Q - C)")
            st.pyplot(fig3)

        # Kernel observables (if present)
        obs_cols = [c for c in filtered.columns if c.startswith("obs_")]
        if obs_cols:
            st.subheader("Quantum kernel observables (fidelity-style)")
            selectable = [c for c in obs_cols if c in filtered.columns]
            metric = st.selectbox("Observable", selectable, index=selectable.index("obs_kernel_gap") if "obs_kernel_gap" in selectable else 0)
            fig_obs, ax_obs = plt.subplots(figsize=(10, 3))
            ax_obs.plot(filtered.index, filtered[metric], "o-", color="orange")
            ax_obs.set_xlabel("Experiment #")
            ax_obs.set_ylabel(metric)
            ax_obs.set_title(f"{metric} over experiments")
            ax_obs.grid(True)
            st.pyplot(fig_obs)

            # If ZNE is present, show a quick raw vs mitigated comparison for the primary observable.
            if "obs_kernel_posneg_mean" in filtered.columns and "obs_zne_kernel_posneg_mean_C0" in filtered.columns:
                st.subheader("Mitigation comparison (kernel_posneg_mean)")
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
                    st.caption("Latest mitigation diagnostics")
                    st.dataframe(filtered[diag_cols].tail(1).T, width="stretch")
        
        # Parameter count over experiments
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
        
        # Ideal vs Noisy comparison (if metadata available)
        comparison_cols = {"execution_mode", "noise_model", "backend_label"}
        if comparison_cols.issubset(df_history.columns):
            st.subheader("Ideal vs Noisy Comparison")
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
        st.dataframe(filtered)

        st.subheader("Export report bundle")
        st.caption("Saves plots + a CSV snapshot to `reports/` for slides/docs.")

        if st.button("Generate report plots"):
            import pathlib
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            out_dir = pathlib.Path(PROJECT_ROOT) / "reports" / f"report_{ts}"
            out_dir.mkdir(parents=True, exist_ok=True)

            # CSV snapshot
            snapshot_path = out_dir / "experiment_history_filtered.csv"
            filtered.to_csv(snapshot_path, index=False)

            # PR-AUC plot (if present)
            if "classical_pr_auc" in filtered.columns:
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(filtered.index, filtered.get("classical_pr_auc"), "o-", label="Classical PR-AUC", color="gray")
                if "quantum_pr_auc" in filtered.columns:
                    ax.plot(filtered.index, filtered.get("quantum_pr_auc"), "s-", label="Quantum PR-AUC", color="purple")
                if "pykeen_pr_auc" in filtered.columns:
                    ax.plot(filtered.index, filtered.get("pykeen_pr_auc"), "^-", label="PyKEEN PR-AUC", color="teal")
                if "gnn_pr_auc" in filtered.columns:
                    ax.plot(filtered.index, filtered.get("gnn_pr_auc"), "d-", label="GNN PR-AUC", color="orange")
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
        st.warning("No experiment history found. Run multiple training experiments to populate this.")

# ==============================
# PAGE: MODEL COMPARISON
# ==============================
elif page == "Comparison (classical vs quantum)":
    st.header("Model comparison: classical versus quantum (evidence from recorded runs)")
    if df_history is None or len(df_history) == 0:
        st.warning("No experiment history found.")
    else:
        dfc = df_history.copy()
        for col in ["classical_pr_auc", "quantum_pr_auc", "obs_kernel_eval_seconds_total", "execution_shots"]:
            if col in dfc.columns:
                dfc[col] = pd.to_numeric(dfc[col], errors="coerce")

        # Allow comparing multiple baselines to the classical score recorded in the same run row.
        candidates = []
        if "quantum_pr_auc" in dfc.columns:
            candidates.append(("Quantum", "quantum_pr_auc"))
        if "pykeen_pr_auc" in dfc.columns:
            candidates.append(("PyKEEN (direct)", "pykeen_pr_auc"))
        if "gnn_pr_auc" in dfc.columns:
            candidates.append(("GNN", "gnn_pr_auc"))

        if not candidates:
            st.warning("No baselines found in history yet (need columns like quantum_pr_auc / pykeen_pr_auc / gnn_pr_auc).")
            st.stop()

        label_to_col = {lbl: col for lbl, col in candidates}
        baseline_choice = st.selectbox("Baseline to compare against Classical", [lbl for lbl, _ in candidates], index=0)
        bcol = label_to_col[baseline_choice]

        paired = dfc.dropna(subset=["classical_pr_auc", bcol]).copy()
        # For quantum, treat 0 as missing (older runs stored quantum_pr_auc=0 when not run)
        if baseline_choice == "Quantum" and "quantum_pr_auc" in paired.columns:
            paired.loc[paired["quantum_pr_auc"] == 0, "quantum_pr_auc"] = np.nan
            paired = paired.dropna(subset=["classical_pr_auc", "quantum_pr_auc"]).copy()

        if len(paired) == 0:
            st.warning(f"No paired rows found yet for Classical + {baseline_choice}.")
            st.stop()

        paired["delta_pr_auc"] = pd.to_numeric(paired[bcol], errors="coerce") - pd.to_numeric(paired["classical_pr_auc"], errors="coerce")
        paired["baseline"] = baseline_choice
        paired["execution_bucket"] = (
            paired.get("execution_mode", "unknown").fillna("unknown").astype(str)
            + " | " + paired.get("backend_label", "unknown").fillna("unknown").astype(str)
            + " | " + paired.get("noise_model", "ideal").fillna("ideal").astype(str)
        )
        paired["quantum_variant"] = paired.get("obs_kernel_source", pd.Series(["unknown"] * len(paired))).fillna("unknown")
        
        st.subheader("Summary")
        cols = st.columns(4)
        with cols[0]:
            st.metric("Paired runs", int(len(paired)))
        with cols[1]:
            st.metric("Mean Classical PR-AUC", float(paired["classical_pr_auc"].mean()))
        with cols[2]:
            st.metric(f"Mean {baseline_choice} PR-AUC", float(pd.to_numeric(paired[bcol], errors="coerce").mean()))
        with cols[3]:
            st.metric("Mean Δ (Q - C)", float(paired["delta_pr_auc"].mean()))

        st.subheader("Statistical rigor (paired)")
        st.caption("Bootstrap is over paired runs (re-sampling runs with replacement). This quantifies uncertainty in the mean ΔPR-AUC.")

        rng_seed = st.number_input("Bootstrap seed", min_value=0, max_value=10_000_000, value=42, step=1)
        n_boot = st.number_input("Bootstrap samples", min_value=200, max_value=20_000, value=2000, step=200)

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
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.hist(paired["classical_pr_auc"].values, bins=12, alpha=0.6, label="Classical", color="blue")
        ax.hist(pd.to_numeric(paired[bcol], errors="coerce").values, bins=12, alpha=0.6, label=baseline_choice, color="purple")
        ax.set_xlabel("PR-AUC")
        ax.set_ylabel("Count")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        st.subheader(f"Distribution: Δ PR-AUC ({baseline_choice} - Classical)")
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.hist(paired["delta_pr_auc"].values, bins=16, color="gray")
        ax2.axvline(0, color="black", linewidth=1)
        ax2.set_xlabel("Δ PR-AUC")
        ax2.set_ylabel("Count")
        ax2.grid(True)
        st.pyplot(fig2)

        if baseline_choice == "Quantum":
            st.subheader("By quantum kernel variant (full vs Nyström)")
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
                axc.set_ylabel("Δ PR-AUC (Quantum - Classical)")
                axc.set_title("Benefit vs cost")
                axc.grid(True)
                st.pyplot(figc)

        st.subheader("Drilldown table (paired runs)")
        show_cols = [c for c in [
            "run_timestamp_utc", "execution_mode", "backend_label", "noise_model",
            "classical_pr_auc", bcol, "delta_pr_auc",
            "obs_kernel_source", "obs_nystrom_m",
            "execution_shots", "obs_kernel_eval_seconds_total",
        ] if c in paired.columns]
        st.dataframe(paired[show_cols].tail(200), width="stretch")

# ==============================
# PAGE: KG SUMMARY
# ==============================
elif page == "Knowledge graph (inventory)":
    st.header("Knowledge graph inventory: Hetionet")
    st.caption("This is what the project is operating on (and what subset was sampled for the PoC).")

    @st.cache_data(show_spinner=False)
    def _load_edges():
        from kg_layer.kg_loader import load_hetionet_edges
        return load_hetionet_edges()

    df_edges = _load_edges()

    @st.cache_data(show_spinner=False)
    def _load_node_metadata():
        try:
            from kg_layer.node_metadata import load_node_metadata_csv
            return load_node_metadata_csv(NODE_METADATA_CSV)
        except Exception:
            return {}

    node_meta = _load_node_metadata()
    st.subheader("Node metadata (optional)")
    if node_meta:
        st.success(f"Loaded node metadata for {len(node_meta)} nodes from `{NODE_METADATA_CSV}`.")
    else:
        st.info(
            "No node metadata file found. To generate a stub (IDs + resolver links), run:\n"
            "`python3 scripts/build_node_metadata_stub.py --max_nodes 200000`.\n"
            "Then optionally fill the `name` column for the nodes you care about."
        )

    st.subheader("Entity identifiers (how to read node IDs)")
    st.markdown("""
Hetionet nodes are stored as **typed identifiers** of the form:

- `NodeType::Identifier`

These identifiers reference standard biomedical vocabularies. This dashboard does not ship full node-name metadata, so for compounds/diseases you typically resolve the identifier in its source registry (e.g., DrugBank, Disease Ontology).
""")

    id_guide = pd.DataFrame(
        [
            {
                "node_type": "Compound",
                "id_format": "Compound::DB####",
                "namespace": "DrugBank ID",
                "notes": "Drug/compound records; `DB...` identifiers are DrugBank.",
            },
            {
                "node_type": "Disease",
                "id_format": "Disease::DOID:####",
                "namespace": "Disease Ontology (DOID)",
                "notes": "Disease concepts; DOID is open and resolvable.",
            },
            {
                "node_type": "Gene",
                "id_format": "Gene::####",
                "namespace": "Entrez Gene ID",
                "notes": "Genes referenced by numeric Entrez Gene identifiers.",
            },
            {
                "node_type": "Anatomy",
                "id_format": "Anatomy::UBERON:#######",
                "namespace": "UBERON",
                "notes": "Anatomical structures.",
            },
            {
                "node_type": "Biological Process",
                "id_format": "Biological Process::GO:#######",
                "namespace": "Gene Ontology (GO)",
                "notes": "GO biological process terms.",
            },
            {
                "node_type": "Molecular Function",
                "id_format": "Molecular Function::GO:#######",
                "namespace": "Gene Ontology (GO)",
                "notes": "GO molecular function terms.",
            },
            {
                "node_type": "Cellular Component",
                "id_format": "Cellular Component::GO:#######",
                "namespace": "Gene Ontology (GO)",
                "notes": "GO cellular component terms.",
            },
            {
                "node_type": "Pathway",
                "id_format": "Pathway::PC7_#### or Pathway::WP###_r#####",
                "namespace": "Pathway Commons / WikiPathways identifiers (as used by Hetionet)",
                "notes": "Pathway concepts; prefixes reflect the source used by Hetionet.",
            },
            {
                "node_type": "Side Effect",
                "id_format": "Side Effect::C#######",
                "namespace": "UMLS CUI",
                "notes": "Side effects are UMLS CUIs (`C...`).",
            },
            {
                "node_type": "Symptom",
                "id_format": "Symptom::D######",
                "namespace": "MeSH Descriptor ID",
                "notes": "Symptoms are MeSH descriptor IDs (`D...`).",
            },
            {
                "node_type": "Pharmacologic Class",
                "id_format": "Pharmacologic Class::N##########",
                "namespace": "NDF-RT (NUI-style identifiers, as used by Hetionet)",
                "notes": "Pharmacologic classes using NDF-RT-style identifiers (`N...`).",
            },
        ]
    )
    st.dataframe(id_guide, width="stretch")

    st.subheader("Relations (metaedges)")
    st.markdown("Hetionet edges are `source  metaedge  target`, where `metaedge` is a short code (e.g., `CtD`).")
    try:
        from kg_layer.kg_loader import METAEDGES
        rel_guide = pd.DataFrame(
            [{"metaedge": k, "meaning": v} for k, v in sorted(METAEDGES.items(), key=lambda kv: kv[0])]
        )
        st.dataframe(rel_guide, width="stretch")
    except Exception:
        st.info("Metaedge dictionary unavailable.")

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
    st.dataframe(node_types, width="stretch")

    rel_counts = df_edges["metaedge"].value_counts().reset_index()
    rel_counts.columns = ["metaedge", "count"]
    st.subheader("Top relations")
    st.dataframe(rel_counts.head(20), width="stretch")

    fig, ax = plt.subplots(figsize=(10, 3))
    topk = rel_counts.head(20).iloc[::-1]
    ax.barh(topk["metaedge"], topk["count"], color="steelblue")
    ax.set_xlabel("Edge count")
    ax.set_title("Top 20 metaedges")
    ax.grid(True, axis="x")
    fig.tight_layout()
    st.pyplot(fig)

    st.download_button(
        "Download KG relation counts (CSV)",
        data=rel_counts.to_csv(index=False).encode("utf-8"),
        file_name="hetionet_relation_counts.csv",
        mime="text/csv",
    )

    st.subheader("Sample edges")
    sample_edges = df_edges.sample(n=min(50, len(df_edges)), random_state=42).copy()
    if node_meta:
        def _name_for(nid: str):
            m = node_meta.get(str(nid))
            return m.name if m and m.name else None
        sample_edges["source_name"] = sample_edges["source"].map(_name_for)
        sample_edges["target_name"] = sample_edges["target"].map(_name_for)
        sample_edges["source_url"] = sample_edges["source"].map(lambda nid: getattr(node_meta.get(str(nid)), "external_url", None))
        sample_edges["target_url"] = sample_edges["target"].map(lambda nid: getattr(node_meta.get(str(nid)), "external_url", None))
    st.dataframe(sample_edges, width="stretch")

# ==============================
# PAGE: FINDINGS
# ==============================
elif page == "Findings (ranked hypotheses)":
    st.header("Findings: ranked hypotheses from the latest model run")
    st.caption("These are **high-scoring predicted links** from the most recent QSVC run, with endpoint IDs included and a novelty check against full Hetionet CtD edges.")

    pred_path = os.path.join(PROJECT_ROOT, "results", "predictions_latest.csv")
    if not os.path.exists(pred_path):
        st.warning("No `results/predictions_latest.csv` found yet. Run a QSVC experiment first.")
    else:
        preds = pd.read_csv(pred_path)
        required = {"split", "source", "target", "y_true", "y_score"}
        if not required.issubset(set(preds.columns)):
            st.warning("Predictions file does not include endpoints yet. Re-run after the latest logging update.")
        else:
            preds["y_score"] = pd.to_numeric(preds["y_score"], errors="coerce")
            preds["y_true"] = pd.to_numeric(preds["y_true"], errors="coerce")

            test = preds[preds["split"] == "test"].dropna(subset=["y_score"]).copy()

            # Optional node name resolution
            @st.cache_data(show_spinner=False)
            def _load_node_metadata_for_findings():
                try:
                    from kg_layer.node_metadata import load_node_metadata_csv
                    return load_node_metadata_csv(NODE_METADATA_CSV)
                except Exception:
                    return {}

            node_meta = _load_node_metadata_for_findings()
            if node_meta:
                def _name_for(nid: str):
                    m = node_meta.get(str(nid))
                    return m.name if m and m.name else None
                test["source_name"] = test["source"].map(_name_for) if "source" in test.columns else None
                test["target_name"] = test["target"].map(_name_for) if "target" in test.columns else None

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
            k = st.slider("Top-K", min_value=5, max_value=100, value=25, step=5)
            novel = test[(test["y_true"] == 0) & (~test["exists_in_hetionet_ctd"])].sort_values("y_score", ascending=False).head(int(k))
            cols = ["source", "source_name", "target", "target_name", "y_score", "exists_in_hetionet_ctd"]
            cols = [c for c in cols if c in novel.columns]
            st.dataframe(novel[cols], width="stretch")

            st.subheader("Generate evidence bundle (neighbors + KG context)")
            st.caption("Creates an exportable CSV with lightweight evidence: nearest neighbors in embedding space and top metaedges involving each node.")

            ev_k = st.slider("Neighbors per entity", min_value=3, max_value=25, value=10, step=1)
            ctx_k = st.slider("Top metaedges per entity", min_value=3, max_value=25, value=10, step=1)

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
                data=novel[cols].to_csv(index=False).encode("utf-8"),
                file_name="findings_top_novel_links.csv",
                mime="text/csv",
            )

            st.subheader("Sanity check: top-scoring true positives (test)")
            top_tp = test[test["y_true"] == 1].sort_values("y_score", ascending=False).head(int(k))
            st.dataframe(top_tp[["source", "target", "y_score", "exists_in_hetionet_ctd"]], width="stretch")

# ==============================
# PAGE 4: RUN BENCHMARKS
# ==============================
elif page == "Run benchmarks":
    st.header("Run benchmarks")
    st.markdown("Run preset experiments and store results in `results/`.")
    node_meta = load_node_metadata()

    st.subheader("Recommended runs (toward PR-AUC 0.8–0.9)")
    st.markdown("""
- **First, measure the classical ceiling** with more data + CV (this tells you whether the task formulation can reach 0.8–0.9 at all).
- **Then, compare quantum vs classical** on the same data with repeated seeds (this tells you whether quantum helps beyond variance).
- If you want higher scores: increase the number of positive CtD edges and use more realistic/harder negatives.
""")

    preset_defs = {
        "Cheapest sanity (quantum-only, fast, noisy sim)": {
            "desc": "Fast end-to-end QSVC run to validate the quantum path; not a final metric.",
            "base_args": ["--fast_mode", "--quantum_only", "--cheap_mode"],
            "supports_cv": False,
            "recommended": True,
            "configs": ["config/quantum_config_noisy.yaml"],
            "defaults": {
                "relation": "CtD",
                "max_entities": 200,
                "qubits": [6],
                "seeds_csv": "42",
                "run_both": False,
            },
            "mode": "quantum_only",
        },
        "Recommended sweep (qubits 6/8/12 × seeds, ideal+noisy)": {
            "desc": "Primary sweep: qubits matrix + seed sweep across ideal+noisy configs (variance-aware comparison).",
            "base_args": ["--full_graph_embeddings", "--use_cached_embeddings"],
            "supports_cv": False,
            "recommended": True,
            "configs": ["config/quantum_config_ideal.yaml", "config/quantum_config_noisy.yaml"],
            "defaults": {
                "relation": "CtD",
                "max_entities": 0,
                "qubits": [6, 8, 12],
                "seeds_csv": "42,123,456",
                "run_both": True,
            },
            "mode": "quantum_and_classical",
        },
        "Learning curve (sample positives × seeds, classical-only)": {
            "desc": "Quickly estimate classical ceiling as positives increase (efficient sampling sweeps).",
            "base_args": ["--classical_only", "--full_graph_embeddings", "--use_cached_embeddings"],
            "supports_cv": False,
            "recommended": True,
            "configs": ["config/quantum_config_ideal.yaml"],
            "pos_edge_samples": [100, 200, 400, 700],
            "defaults": {
                "relation": "CtD",
                "max_entities": 0,
                "qubits": [12],
                "seeds_csv": "42,123,456",
                "run_both": False,
            },
            "mode": "classical_only",
        },
        "Classical ceiling (CV, bigger data, full-graph)": {
            "desc": "Best estimate of achievable PR-AUC before tuning quantum. Uses CV for stability.",
            "base_args": ["--classical_only", "--full_graph_embeddings", "--use_cached_embeddings", "--use_cv_evaluation", "--cv_folds", "5", "--negative_sampling", "hard", "--neg_ratio", "2.0", "--use_evidence_weighting", "--min_shared_genes", "1"],
            "supports_cv": True,
            "recommended": True,
            "configs": ["config/quantum_config_ideal.yaml"],
            "defaults": {
                "relation": "CtD",
                "max_entities": 0,
                "qubits": [12],
                "seeds_csv": "42,123,456",
                "run_both": False,
            },
            "mode": "classical_only",
        },
        "Quantum vs classical (same data, repeated seeds)": {
            "desc": "Compare QSVC vs classical on the same dataset. Run multiple seeds to quantify variance.",
            "base_args": ["--full_graph_embeddings", "--use_cached_embeddings", "--negative_sampling", "hard", "--neg_ratio", "2.0"],
            "supports_cv": False,
            "recommended": True,
            "configs": ["config/quantum_config_ideal.yaml", "config/quantum_config_noisy.yaml"],
            "defaults": {
                "relation": "CtD",
                "max_entities": 0,
                "qubits": [12],
                "seeds_csv": "42,123,456",
                "run_both": True,
            },
            "mode": "quantum_and_classical",
        },
        "Ablation: Nyström vs full kernel (small m vs none)": {
            "desc": "Compare faster Nyström kernel approximation vs full kernel on the same seed.",
            "base_args": ["--full_graph_embeddings"],
            "supports_cv": False,
            "recommended": False,
            "configs": ["config/quantum_config_noisy.yaml"],
            "defaults": {
                "relation": "CtD",
                "max_entities": 0,
                "qubits": [6],
                "seeds_csv": "42,123",
                "run_both": False,
            },
            "mode": "quantum_and_classical",
        },
    }

    # ---- preset selection (auto-fills defaults) ----
    preset_keys = list(preset_defs.keys())
    if "bench_selected_preset" not in st.session_state:
        st.session_state["bench_selected_preset"] = preset_keys[0]

    top_left, top_right = st.columns([2, 1])
    with top_left:
        preset_name = st.selectbox("Preset", preset_keys, index=preset_keys.index(st.session_state["bench_selected_preset"]), key="bench_preset_select")
        st.session_state["bench_selected_preset"] = preset_name
    base = preset_defs[preset_name]
    defaults = base.get("defaults", {}) or {}
    mode = str(base.get("mode", "quantum_and_classical"))

    # Ensure all expected session_state keys exist (important on first-load or after code updates).
    # This must happen BEFORE widgets access st.session_state[...] values.
    def _ensure(k: str, v):
        if k not in st.session_state:
            st.session_state[k] = v

    _ensure("bench_last_preset", None)
    _ensure("bench_relation", defaults.get("relation", "CtD"))
    _ensure("bench_max_entities", int(defaults.get("max_entities", 0)))
    _ensure("bench_qubits", defaults.get("qubits", [12]))
    _ensure("bench_seeds_csv", defaults.get("seeds_csv", "42,123,456"))
    _ensure("bench_run_both", bool(defaults.get("run_both", True)))
    _ensure("bench_opt_nystrom", "(preset default)")
    _ensure("bench_opt_pos_sample", "(no sampling)")
    _ensure("bench_opt_neg_sampling", "(preset default)")
    _ensure("bench_opt_neg_ratio", "(preset default)")
    _ensure("bench_opt_calibrate", False)
    _ensure("bench_extra_flags", "")

    # Apply defaults BEFORE widgets are created (only when preset changes)
    if st.session_state.get("bench_last_preset") != preset_name:
        st.session_state["bench_last_preset"] = preset_name
        st.session_state["bench_relation"] = defaults.get("relation", "CtD")
        st.session_state["bench_max_entities"] = int(defaults.get("max_entities", 0))
        st.session_state["bench_qubits"] = defaults.get("qubits", [12])
        st.session_state["bench_seeds_csv"] = defaults.get("seeds_csv", "42,123,456")
        st.session_state["bench_run_both"] = bool(defaults.get("run_both", True))
        # sensible dropdown defaults
        st.session_state["bench_opt_nystrom"] = "(preset default)"
        st.session_state["bench_opt_pos_sample"] = "(no sampling)"
        st.session_state["bench_opt_neg_sampling"] = "(preset default)"
        st.session_state["bench_opt_neg_ratio"] = "(preset default)"
        st.session_state["bench_opt_calibrate"] = False
        st.session_state["bench_extra_flags"] = ""

    with top_right:
        st.caption("Preset summary")
        st.write(base.get("desc", ""))
        st.caption("Included flags (fixed by preset)")
        st.code(" ".join(base.get("base_args", [])) or "(none)")

    st.divider()

    # ---- layout: tabs ----
    tab_data, tab_models, tab_exec, tab_adv = st.tabs(["Dataset", "Models", "Execution", "Advanced"])

    with tab_data:
        c1, c2, c3 = st.columns(3)
        with c1:
            relation = st.text_input("Relation", value=st.session_state["bench_relation"], key="bench_relation")
        with c2:
            max_entities = st.number_input("Max entities (0 = all)", min_value=0, max_value=20000, value=int(st.session_state["bench_max_entities"]), step=500, key="bench_max_entities")
            st.caption("For CtD, 0 uses all CtD edges/entities; higher caps are mainly for other relations.")
        with c3:
            # Only enable manual sampling if preset isn't already sweeping samples
            has_internal_sweep = isinstance(base.get("pos_edge_samples"), list) and len(base.get("pos_edge_samples")) > 0
            opt_pos_sample = st.selectbox(
                "Quick run: sample positive edges",
                options=["(no sampling)", "200", "400", "700"],
                index=["(no sampling)", "200", "400", "700"].index(st.session_state["bench_opt_pos_sample"]),
                disabled=has_internal_sweep,
                key="bench_opt_pos_sample",
            )
            if has_internal_sweep:
                st.info("This preset already runs an internal sampling sweep; the manual sampling control is disabled.")

    with tab_models:
        c1, c2, c3 = st.columns(3)
        with c1:
            qubits = st.multiselect(
                "Qubits to run",
                options=[6, 8, 12],
                default=st.session_state["bench_qubits"],
                disabled=(mode == "classical_only"),
                key="bench_qubits",
            )
            if mode == "classical_only":
                st.caption("Classical-only preset: qubits are ignored.")
        with c2:
            opt_nystrom = st.selectbox(
                "QSVC Nyström landmarks (m)",
                options=["(preset default)", "off (full kernel)", "m=24", "m=32", "m=64"],
                index=["(preset default)", "off (full kernel)", "m=24", "m=32", "m=64"].index(st.session_state["bench_opt_nystrom"]),
                disabled=("Nyström vs full kernel" in preset_name),
                key="bench_opt_nystrom",
            )
            if "Nyström vs full kernel" in preset_name:
                st.caption("This preset toggles Nyström vs full internally; the dropdown is disabled.")
        with c3:
            opt_neg_sampling = st.selectbox(
                "Negative sampling",
                options=["(preset default)", "random", "hard", "diverse"],
                index=["(preset default)", "random", "hard", "diverse"].index(st.session_state["bench_opt_neg_sampling"]),
                key="bench_opt_neg_sampling",
            )
            opt_neg_ratio = st.selectbox(
                "Negatives per positive (neg_ratio)",
                options=["(preset default)", "1.0", "2.0", "5.0"],
                index=["(preset default)", "1.0", "2.0", "5.0"].index(st.session_state["bench_opt_neg_ratio"]),
                key="bench_opt_neg_ratio",
            )
            opt_calibrate = st.checkbox("Calibrate probabilities (classical)", value=bool(st.session_state["bench_opt_calibrate"]), key="bench_opt_calibrate")

    with tab_exec:
        c1, c2 = st.columns([2, 1])
        with c1:
            seeds_csv = st.text_input("Seeds (comma-separated)", value=st.session_state["bench_seeds_csv"], key="bench_seeds_csv")
        with c2:
            run_both_ideal_noisy = st.checkbox(
                "Run both ideal + noisy (if supported)",
                value=bool(st.session_state["bench_run_both"]),
                disabled=(mode == "classical_only"),
                key="bench_run_both",
            )
            if mode == "classical_only":
                st.caption("Classical-only preset: ideal/noisy toggle is ignored.")

    with tab_adv:
        with st.expander("Advanced (custom CLI flags)"):
            extra_flags = st.text_input("Custom CLI flags", value=st.session_state["bench_extra_flags"], key="bench_extra_flags")
            st.caption("Example: `--enable_pykeen_direct --pykeen_model RotatE --enable_gnn_baseline --gnn_arch graphsage`")

    # ---- build run plan + preview ----
    errors = []
    try:
        seeds = [int(s.strip()) for s in str(seeds_csv).split(",") if s.strip()]
        if not seeds:
            raise ValueError("No seeds provided")
    except Exception:
        seeds = []
        errors.append("Seeds must be a comma-separated list of integers (e.g., 42,123,456).")

    cfgs = list(base.get("configs", []))
    if mode != "classical_only" and not run_both_ideal_noisy:
        cfgs = cfgs[:1]
    if mode == "classical_only":
        cfgs = cfgs[:1]

    # Optional: learning-curve sampling sweep (positive edge samples)
    pos_edge_samples = base.get("pos_edge_samples")
    pos_list = [0]
    if isinstance(pos_edge_samples, list) and len(pos_edge_samples) > 0:
        try:
            pos_list = [int(x) for x in pos_edge_samples if int(x) > 0]
        except Exception:
            pos_list = [0]
    else:
        # manual sampling
        if opt_pos_sample != "(no sampling)":
            try:
                pos_list = [int(opt_pos_sample)]
            except Exception:
                errors.append("Sample positives must be numeric (200/400/700).")

    # Nyström variants for ablation preset
    nystrom_variants = [{"label": "default", "args": []}]
    if "Nyström vs full kernel" in preset_name:
        nystrom_variants = [
            {"label": "full", "args": ["--qsvc_nystrom_m", "0"]},
            {"label": "nystrom", "args": ["--qsvc_nystrom_m", "32"]},
        ]

    # dropdown → flags
    dropdown_flags: list[str] = []
    if mode != "classical_only":
        if opt_nystrom == "off (full kernel)":
            dropdown_flags += ["--qsvc_nystrom_m", "0"]
        elif opt_nystrom == "m=24":
            dropdown_flags += ["--qsvc_nystrom_m", "24"]
        elif opt_nystrom == "m=32":
            dropdown_flags += ["--qsvc_nystrom_m", "32"]
        elif opt_nystrom == "m=64":
            dropdown_flags += ["--qsvc_nystrom_m", "64"]

    if opt_neg_sampling != "(preset default)":
        dropdown_flags += ["--negative_sampling", str(opt_neg_sampling)]
    if opt_neg_ratio != "(preset default)":
        dropdown_flags += ["--neg_ratio", str(opt_neg_ratio)]
    if opt_calibrate:
        dropdown_flags += ["--calibrate_probabilities"]

    extra = [t for t in (extra_flags or "").strip().split() if t.strip()]

    q_list = list(qubits) if (mode != "classical_only") else [12]
    if not q_list:
        errors.append("Select at least one qubit setting.")

    total_runs = len(q_list) * len(seeds) * len(cfgs) * len(pos_list) * len(nystrom_variants)
    st.subheader("Run plan")
    if errors:
        st.error(" / ".join(errors))
        st.stop()
    st.caption(f"Planned runs: {total_runs}")

    # show a compact preview table
    preview_rows = []
    for q in q_list[:3]:
        for seed in seeds[:3]:
            for cfg in cfgs[:2]:
                for pos_n in pos_list[:2]:
                    for v in nystrom_variants[:2]:
                        preview_rows.append(
                            {"qubits": q, "seed": seed, "config": cfg, "pos_edges": pos_n if pos_n else "", "variant": v["label"]}
                        )
    st.dataframe(pd.DataFrame(preview_rows), width="stretch")

    # command preview (first few)
    def _cmds_preview(limit: int = 5) -> list[list[str]]:
        out = []
        for q in q_list:
            for seed in seeds:
                for cfg in cfgs:
                    for pos_n in pos_list:
                        for v in nystrom_variants:
                            cmd = [
                                sys.executable,
                                "scripts/run_optimized_pipeline.py",
                                "--relation", str(relation),
                                "--qml_dim", str(int(q)),
                                "--random_state", str(int(seed)),
                                "--quantum_config_path", str(cfg),
                            ]
                            if int(max_entities) > 0:
                                cmd += ["--max_entities", str(int(max_entities))]
                            if pos_n and int(pos_n) > 0:
                                cmd += ["--pos_edge_sample", str(int(pos_n)), "--pos_edge_sample_strategy", "uniform"]
                            cmd += base.get("base_args", [])
                            cmd += dropdown_flags
                            cmd += v["args"]
                            cmd += extra
                            out.append(cmd)
                            if len(out) >= limit:
                                return out
        return out

    with st.expander("Command preview (first 5)"):
        lines = []
        for cmd in _cmds_preview(5):
            try:
                lines.append(shlex.join(cmd))
            except Exception:
                lines.append(" ".join(cmd))
        st.code("\n".join(lines) if lines else "(no commands)")

    # ---- run ----
    if st.button("Run planned benchmarks", type="primary"):
        log_container = st.empty()
        run_idx = 0
        for q in q_list:
            for seed in seeds:
                for cfg in cfgs:
                    for pos_n in pos_list:
                        for v in nystrom_variants:
                            run_idx += 1
                            suffix = f", pos_edges={pos_n}" if pos_n and int(pos_n) > 0 else ""
                            st.write(f"Run {run_idx}/{total_runs} — qubits={q}, seed={seed}, config={cfg}{suffix}")
                            cmd = [
                                sys.executable,
                                "scripts/run_optimized_pipeline.py",
                                "--relation", str(relation),
                                "--qml_dim", str(int(q)),
                                "--random_state", str(int(seed)),
                                "--quantum_config_path", str(cfg),
                            ]
                            if int(max_entities) > 0:
                                cmd += ["--max_entities", str(int(max_entities))]
                            if pos_n and int(pos_n) > 0:
                                cmd += ["--pos_edge_sample", str(int(pos_n)), "--pos_edge_sample_strategy", "uniform"]
                            cmd += base.get("base_args", [])
                            cmd += dropdown_flags
                            cmd += v["args"]
                            cmd += extra
                            run_command(cmd, log_container)

        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Finished running benchmarks. Check Results / Experiments / Comparison tabs.")

# ==============================
# PAGE 5: BACKEND STATUS
# ==============================
elif page == "Hardware readiness (backend status)":
    st.header("IBM Quantum backend status (access + queue)")
    st.caption("Quick sanity check for whether `ibm_fez` / `ibm_brisbane` are reachable from your credentials.")

    backend_hint = st.text_input("Backend name to check (optional)", value="ibm_brisbane")
    instance_hint = st.text_input("IBM instance", value=os.environ.get("IBM_INSTANCE", "ibm-q/open/main"))

    if st.button("Check available backends"):
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService

            token = os.environ.get("IBM_Q_TOKEN") or os.environ.get("QISKIT_IBM_TOKEN")
            if not token:
                st.error("Missing IBM token. Set `IBM_Q_TOKEN` (or `QISKIT_IBM_TOKEN`) in your environment and restart Streamlit.")
            else:
                service = QiskitRuntimeService(channel="ibm_quantum", token=token, instance=instance_hint)
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
                st.dataframe(dfb, width="stretch")

                if backend_hint:
                    if (dfb["backend"].astype(str) == backend_hint).any():
                        st.success(f"`{backend_hint}` is accessible.")
                    else:
                        st.warning(f"`{backend_hint}` not found in your accessible backends list.")
        except Exception as e:
            st.error(f"Backend status check failed: {e}")

# Footer
st.markdown("---")
st.caption("Hybrid quantum–classical knowledge graph system for biomedical link prediction (proof of concept)")