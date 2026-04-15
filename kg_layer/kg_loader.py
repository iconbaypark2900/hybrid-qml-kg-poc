# kg_layer/kg_loader.py

import os
import pandas as pd
import networkx as nx
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import yaml
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hetionet metaedge abbreviations (from https://het.io/edges)
METAEDGES = {
    'CtD': 'Compound treats Disease',
    'CpD': 'Compound palliates Disease',
    'PCiC': 'Pharmacologic Class includes Compound',
    'DaG': 'Disease associates Gene',
    'DdG': 'Disease downregulates Gene',
    'DuG': 'Disease upregulates Gene',
    'DpS': 'Disease presents Symptom',
    'DrD': 'Disease resembles Disease',
    'DlA': 'Disease localizes Anatomy',
    'GiG': 'Gene interacts Gene',
    'Gr>G': 'Gene expresses Gene',
    'GcG': 'Gene catalyzes Gene',
    'GpPW': 'Gene participates Pathway',
    'GpMF': 'Gene part Molecular Function',
    'GpBP': 'Gene part Biological Process',
    'GpCC': 'Gene part Cellular Component',
    'AdG': 'Anatomy expresses Gene',
    'AeG': 'Anatomy enriches Gene',
    'AuG': 'Anatomy upregulates Gene',
    'CdG': 'Compound downregulates Gene',
    'CuG': 'Compound upregulates Gene',
    'CbG': 'Compound binds Gene',
    'CrC': 'Compound resembles Compound',
    'CcSE': 'Compound causes Side Effect',
    'CpC': 'Compound palliates Compound',
}

def load_kg_config(config_path: str = "config/kg_layer_config.yaml") -> Dict:
    """
    Load KG layer configuration from YAML file.

    Args:
        config_path: Path to the KG layer config YAML file.

    Returns:
        Dictionary containing configuration parameters.
    """
    if not Path(config_path).exists():
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {
            "data_loading": {
                "data_dir": "data",
                "relation_type": "CtD",
                "max_entities": 300,
                "test_size": 0.2,
                "random_state": 42,
                "num_negatives": None
            }
        }

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def download_hetionet_if_missing(data_dir: str = "data") -> str:
    """
    Ensure Hetionet edge list is present locally.
    Tries multiple mirrors/new branch layout. Stores an uncompressed TSV at:
        {data_dir}/hetionet-v1.0-edges.sif

    Args:
        data_dir: The directory to store the data in.

    Returns:
        The path to the edge file.
    """
    os.makedirs(data_dir, exist_ok=True)
    edge_file = os.path.join(data_dir, "hetionet-v1.0-edges.sif")
    if os.path.exists(edge_file):
        logger.info(f"Hetionet edge file already exists at {edge_file}")
        return edge_file

    logger.info("Downloading Hetionet edge list...")

    # Candidate URLs: prefer .sif.gz on 'main', then fall back
    candidates = [
        ("https://raw.githubusercontent.com/hetio/hetionet/main/hetnet/tsv/hetionet-v1.0-edges.sif.gz", True),
        ("https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-edges.sif.gz", True),
        ("https://raw.githubusercontent.com/hetio/hetionet/master/hetnet/tsv/hetionet-v1.0-edges.sif.gz", True),
        ("https://github.com/hetio/hetionet/raw/master/hetnet/tsv/hetionet-v1.0-edges.sif.gz", True),
        ("https://raw.githubusercontent.com/hetio/hetionet/main/hetnet/tsv/hetionet-v1.0-edges.sif", False),
        ("https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-edges.sif", False),
    ]

    last_err = None
    for url, gz in candidates:
        try:
            logger.info(f"Trying {url}")
            df = pd.read_csv(
                url,
                sep="\t",
                names=["source", "metaedge", "target"],
                dtype=str,
                compression=("gzip" if gz else None),
            )
            # Basic sanity check
            if {"source", "metaedge", "target"}.issubset(df.columns) and len(df) > 0:
                df.to_csv(edge_file, sep="\t", index=False, header=False)
                logger.info(f"Saved Hetionet edges to {edge_file} ({len(df)} rows)")
                return edge_file
            else:
                raise ValueError(f"Downloaded file from {url} but schema/rows look invalid.")
        except Exception as e:
            last_err = e
            logger.warning(f"Failed to fetch from {url}: {e}")

    raise RuntimeError(
        "Could not download Hetionet edges from any known location. "
        "Manual workaround:\n"
        "  1) curl -L -o data/hetionet-v1.0-edges.sif.gz "
        "https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-edges.sif.gz\n"
        "  2) python - <<'PY'\n"
        "import gzip, shutil; "
        "shutil.copyfileobj(gzip.open('data/hetionet-v1.0-edges.sif.gz','rb'), open('data/hetionet-v1.0-edges.sif','wb'))\n"
        "PY\n"
        f"Last error was: {last_err}"
    )


def load_hetionet_edges(
    data_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Load Hetionet edges from local cache (auto-downloads if missing).

    Args:
        data_dir: The directory where the data is stored.

    Returns:
        A DataFrame with columns: ['source', 'metaedge', 'target'] (dtype=str).
    """
    # Load config if not provided
    if config is None:
        if config_path is None:
            config_path = "config/kg_layer_config.yaml"
        config = load_kg_config(config_path)

    # Use provided parameter or fall back to config
    if data_dir is None:
        data_dir = config["data_loading"]["data_dir"]

    edge_file = download_hetionet_if_missing(data_dir)
    df = pd.read_csv(
        edge_file,
        sep="\t",
        names=["source", "metaedge", "target"],
        dtype=str,
    )
    logger.info(f"Loaded {len(df)} edges from Hetionet.")
    return df

def extract_task_edges(
    df_edges: pd.DataFrame,
    relation_type: Optional[str] = None,
    max_entities: Optional[int] = None,
    config_path: Optional[str] = None,
    config: Optional[Dict] = None
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]:
    """
    Extract edges for a specific task (e.g., drug-disease treatment).

    Args:
        df_edges: Full Hetionet edge DataFrame
        relation_type: Metaedge code (e.g., 'CtD' for Compound-treats-Disease)
        max_entities: Optional cap on number of unique entities (for PoC scalability)

    Returns:
        filtered_edges: DataFrame with only the desired relation
        entity_to_id: Mapping from entity ID (e.g., 'Compound::DB00001') to int
        id_to_entity: Reverse mapping
    """
    # Load config if not provided
    if config is None:
        if config_path is None:
            config_path = "config/kg_layer_config.yaml"
        config = load_kg_config(config_path)

    # Use provided parameters or fall back to config
    if relation_type is None:
        relation_type = config["data_loading"]["relation_type"]
    if max_entities is None:
        max_entities = config["data_loading"].get("max_entities")

    logger.info(f"Filtering for relation: {relation_type} ({METAEDGES.get(relation_type, 'Unknown')})")

    task_edges = df_edges[df_edges["metaedge"] == relation_type].copy()
    logger.info(f"Found {len(task_edges)} '{relation_type}' edges.")

    # Get all unique entities
    all_entities = pd.concat([task_edges["source"], task_edges["target"]]).unique()

    if max_entities and len(all_entities) > max_entities:
        logger.info(f"Sampling {max_entities} entities for PoC scalability.")
        sampled_entities = pd.Series(all_entities).sample(n=max_entities, random_state=42)
        task_edges = task_edges[
            task_edges["source"].isin(sampled_entities) &
            task_edges["target"].isin(sampled_entities)
        ]
        all_entities = pd.concat([task_edges["source"], task_edges["target"]]).unique()

    # Create entity ID mapping
    entity_to_id = {entity: idx for idx, entity in enumerate(sorted(all_entities))}
    id_to_entity = {idx: entity for entity, idx in entity_to_id.items()}

    # Add integer IDs to edges
    task_edges["source_id"] = task_edges["source"].map(entity_to_id)
    task_edges["target_id"] = task_edges["target"].map(entity_to_id)

    logger.info(f"Final task graph: {len(task_edges)} edges, {len(all_entities)} unique entities.")
    return task_edges, entity_to_id, id_to_entity

def create_networkx_graph(task_edges: pd.DataFrame) -> nx.DiGraph:
    """
    Create a NetworkX graph from task-specific edges.

    Args:
        task_edges:

    Returns:
        G: A directed graph with edges from task_edges
    """
    G = nx.DiGraph()
    for _, row in task_edges.iterrows():
        G.add_edge(row["source_id"], row["target_id"], relation=row["metaedge"])
    return G

def get_negative_samples(
    task_edges: pd.DataFrame,
    num_negatives: Optional[int] = None,
    random_state: Optional[int] = None,
    config_path: Optional[str] = None,
    config: Optional[Dict] = None,
    strategy: str = "random",
    diversity_weight: float = 0.5,
) -> pd.DataFrame:
    """
    Generate negative samples (non-existing links) for training.
    Simple random sampling without replacement.

    Args:
        task_edges:
        num_negatives:
        random_state:

    Returns:
        DataFrame with negative samples: ['source_id', 'target_id', 'label' (0)]
    """
    # Load config if not provided
    if config is None:
        if config_path is None:
            config_path = "config/kg_layer_config.yaml"
        config = load_kg_config(config_path)

    # Use provided parameters or fall back to config
    if num_negatives is None:
        num_negatives = len(task_edges)  # 1:1 ratio

    # Support both string entity IDs (source/target) and integer IDs (source_id/target_id)
    if "source" in task_edges.columns:
        src_col, tgt_col = "source", "target"
    else:
        src_col, tgt_col = "source_id", "target_id"

    sources = task_edges[src_col].values
    targets = task_edges[tgt_col].values
    existing_pairs = set(zip(sources, targets))

    unique_sources = task_edges[src_col].unique()
    unique_targets = task_edges[tgt_col].unique()

    np.random.seed(random_state)

    neg_sources = []
    neg_targets = []
    attempts = 0
    max_attempts = num_negatives * 10

    while len(neg_sources) < num_negatives and attempts < max_attempts:
        s = np.random.choice(unique_sources)
        t = np.random.choice(unique_targets)
        if (s, t) not in existing_pairs and (s, t) not in zip(neg_sources, neg_targets):
            neg_sources.append(s)
            neg_targets.append(t)
        attempts += 1

    logger.info(f"Generated {len(neg_sources)} negative samples.")
    return pd.DataFrame({
        src_col: neg_sources,
        tgt_col: neg_targets,
        "label": 0
    })

def prepare_link_prediction_dataset(
    task_edges: pd.DataFrame,
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
    config_path: Optional[str] = None,
    config: Optional[Dict] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split edges into train/test and add negative samples.
    Returns train and test DataFrames with 'label' column.

    Args:
        task_edges:
        test_size:
        random_state:

    Returns:
        train_df: Training DataFrame with positive and negative samples
        test_df: Testing DataFrame with positive and negative samples
    """
    # Load config if not provided
    if config is None:
        if config_path is None:
            config_path = "config/kg_layer_config.yaml"
        config = load_kg_config(config_path)

    # Use provided parameters or fall back to config
    if test_size is None:
        test_size = config["data_loading"]["test_size"]
    if random_state is None:
        random_state = config["data_loading"]["random_state"]

    from sklearn.model_selection import train_test_split

    # Positive samples — use string entity IDs so the embedder can look them up
    pos_df = task_edges[["source", "target"]].copy()
    pos_df["label"] = 1

    # Train/test split on positive edges
    pos_train, pos_test = train_test_split(
        pos_df, test_size=test_size, random_state=random_state
    )

    # Generate negatives
    neg_train = get_negative_samples(pos_train, random_state=random_state)
    neg_test = get_negative_samples(pos_test, random_state=random_state + 1)

    # Combine
    train_df = pd.concat([pos_train, neg_train], ignore_index=True).sample(frac=1, random_state=random_state)
    test_df = pd.concat([pos_test, neg_test], ignore_index=True).sample(frac=1, random_state=random_state)

    logger.info(f"Train set: {len(train_df)} samples ({train_df['label'].sum()} positive)")
    logger.info(f"Test set: {len(test_df)} samples ({test_df['label'].sum()} positive)")

    return train_df, test_df


def prepare_full_graph_for_embeddings(
    df_edges: pd.DataFrame,
    task_entities: List[str]
) -> pd.DataFrame:
    """
    Prepare full-graph edges for embedding training.
    Returns ALL edges from Hetionet where at least one entity is in the task entity set.

    Args:
        df_edges: Full Hetionet edge DataFrame with columns ['source', 'metaedge', 'target']
        task_entities: List of entity strings (e.g., ['Compound::DB00001', 'Disease::DOID:1234'])

    Returns:
        DataFrame with all edges involving task entities, with columns ['source', 'metaedge', 'target']
    """
    task_entity_set = set(task_entities)

    full_graph_edges = df_edges[
        df_edges["source"].isin(task_entity_set) |
        df_edges["target"].isin(task_entity_set)
    ].copy()

    logger.info(f"Full graph scan: {len(df_edges)} total edges")
    logger.info(f"Filtered to task entities: {len(full_graph_edges)} edges "
               f"({full_graph_edges['metaedge'].nunique()} relation types)")

    return full_graph_edges


# =============================================================================
# HARD NEGATIVE MINING
#
# Three strategies with the same output contract:
#   DataFrame with columns [source, target, label=0]
#   — mirroring the positive edge schema used throughout the pipeline.
#
# Strategy 1 — degree_corrupt  (default "hard" in the pipeline)
#   For each positive (s, t) corrupt tail with probability 0.5, else corrupt
#   head. Replacement drawn proportional to node degree so high-degree
#   (biologically busy) nodes appear more often, making negatives harder than
#   uniform random. This is the standard KG corruption scheme (Bordes 2013)
#   with degree-weighted sampling.
#
# Strategy 2 — type_aware
#   Restrict corruption to entities of the same Hetionet type prefix
#   (e.g. only replace a Compound with another Compound). Prevents trivially
#   invalid pairs while remaining harder than random because type constraints
#   shrink the candidate pool.
#
# Strategy 3 — embedding_knn  (requires trained embedder + scipy)
#   Replace the tail with the K-nearest neighbour of the true tail in
#   embedding space, excluding known positives. These are the most
#   structurally similar to true treatments and therefore the hardest.
#   Falls back gracefully to degree_corrupt if embedder unavailable.
#
# All three avoid known positive pairs via an `existing_pairs` set and
# guarantee no duplicate negatives within a single generation call.
# =============================================================================


def _build_existing_pairs(pos_edges: pd.DataFrame) -> set:
    """Return set of (source, target) string pairs from positive edges."""
    if "source" in pos_edges.columns and "target" in pos_edges.columns:
        return set(zip(pos_edges["source"].astype(str).values,
                       pos_edges["target"].astype(str).values))
    return set()


def get_hard_negatives_degree_corrupt(
    pos_edges: pd.DataFrame,
    num_negatives: Optional[int] = None,
    random_state: int = 42,
    corrupt_tail_prob: float = 0.5,
) -> pd.DataFrame:
    """
    Hard negatives via degree-weighted KG corruption (Strategy 1).

    For each positive (s, t) we corrupt head or tail with probability
    `corrupt_tail_prob`. The replacement entity is drawn proportional to its
    degree (number of appearances as that role in pos_edges), making
    high-connectivity entities more likely — and resulting negatives
    structurally closer to real positives than uniform random.

    Args:
        pos_edges: DataFrame with columns [source, target] (positive examples).
        num_negatives: How many negatives to generate. Defaults to len(pos_edges).
        random_state: RNG seed for reproducibility.
        corrupt_tail_prob: Probability of corrupting the tail (vs head).

    Returns:
        DataFrame with columns [source, target, label=0].
    """
    if pos_edges.empty:
        return pd.DataFrame(columns=["source", "target", "label"])

    n = int(num_negatives or len(pos_edges))
    rng = np.random.default_rng(int(random_state))
    existing = _build_existing_pairs(pos_edges)

    sources = pos_edges["source"].astype(str).values
    targets = pos_edges["target"].astype(str).values

    # Degree-weighted pools
    src_counts = pd.Series(sources).value_counts()
    tgt_counts = pd.Series(targets).value_counts()
    src_ids = src_counts.index.values
    tgt_ids = tgt_counts.index.values
    src_p = src_counts.values.astype(float)
    src_p /= src_p.sum()
    tgt_p = tgt_counts.values.astype(float)
    tgt_p /= tgt_p.sum()

    pos_pairs = list(zip(sources, targets))

    neg_s, neg_t = [], []
    attempts = 0
    max_attempts = max(5_000, n * 30)

    while len(neg_s) < n and attempts < max_attempts:
        s_pos, t_pos = pos_pairs[attempts % len(pos_pairs)]
        if rng.random() < corrupt_tail_prob:
            s = str(s_pos)
            t = str(rng.choice(tgt_ids, p=tgt_p))
        else:
            s = str(rng.choice(src_ids, p=src_p))
            t = str(t_pos)
        if (s, t) not in existing:
            neg_s.append(s)
            neg_t.append(t)
            existing.add((s, t))
        attempts += 1

    if len(neg_s) < n:
        logger.warning(
            "degree_corrupt: requested %d negatives but only generated %d "
            "(exhausted %d attempts).",
            n, len(neg_s), max_attempts,
        )

    logger.info("degree_corrupt: generated %d hard negatives.", len(neg_s))
    return pd.DataFrame({"source": neg_s, "target": neg_t, "label": 0})


def get_hard_negatives_type_aware(
    pos_edges: pd.DataFrame,
    num_negatives: Optional[int] = None,
    random_state: int = 42,
    corrupt_tail_prob: float = 0.5,
) -> pd.DataFrame:
    """
    Hard negatives restricted to same Hetionet entity-type (Strategy 2).

    Hetionet entity IDs have the form ``TypeName::local_id``
    (e.g. ``Compound::DB00945``, ``Disease::DOID:9352``). This strategy
    replaces the corrupted entity with a random entity of the *same type*,
    preventing trivially invalid pairs while remaining harder than fully
    random because the type constraint shrinks the pool.

    Falls back to degree_corrupt if type prefix cannot be parsed.

    Args:
        pos_edges: DataFrame with columns [source, target].
        num_negatives: Number of negatives to generate.
        random_state: RNG seed.
        corrupt_tail_prob: Probability of corrupting the tail.

    Returns:
        DataFrame with columns [source, target, label=0].
    """
    if pos_edges.empty:
        return pd.DataFrame(columns=["source", "target", "label"])

    def _type_prefix(entity: str) -> str:
        return entity.split("::")[0] if "::" in str(entity) else "__unknown__"

    all_sources = pos_edges["source"].astype(str).values
    all_targets = pos_edges["target"].astype(str).values

    # Check that type prefixes are parseable
    sample_src_types = {_type_prefix(e) for e in all_sources[:20]}
    sample_tgt_types = {_type_prefix(e) for e in all_targets[:20]}
    if "__unknown__" in sample_src_types or "__unknown__" in sample_tgt_types:
        logger.warning(
            "type_aware: entity IDs don't contain '::' prefix — "
            "falling back to degree_corrupt."
        )
        return get_hard_negatives_degree_corrupt(
            pos_edges, num_negatives, random_state, corrupt_tail_prob
        )

    # Build per-type entity pools
    src_by_type: Dict[str, list] = {}
    for e in all_sources:
        src_by_type.setdefault(_type_prefix(e), []).append(str(e))
    tgt_by_type: Dict[str, list] = {}
    for e in all_targets:
        tgt_by_type.setdefault(_type_prefix(e), []).append(str(e))

    n = int(num_negatives or len(pos_edges))
    rng = np.random.default_rng(int(random_state))
    existing = _build_existing_pairs(pos_edges)
    pos_pairs = list(zip(all_sources, all_targets))

    neg_s, neg_t = [], []
    attempts = 0
    max_attempts = max(5_000, n * 30)

    while len(neg_s) < n and attempts < max_attempts:
        s_pos, t_pos = pos_pairs[attempts % len(pos_pairs)]
        if rng.random() < corrupt_tail_prob:
            s = str(s_pos)
            tgt_pool = tgt_by_type.get(
                _type_prefix(str(t_pos)), list(tgt_by_type.values())[0]
            )
            t = str(rng.choice(tgt_pool))
        else:
            src_pool = src_by_type.get(
                _type_prefix(str(s_pos)), list(src_by_type.values())[0]
            )
            s = str(rng.choice(src_pool))
            t = str(t_pos)
        if (s, t) not in existing:
            neg_s.append(s)
            neg_t.append(t)
            existing.add((s, t))
        attempts += 1

    if len(neg_s) < n:
        logger.warning(
            "type_aware: requested %d negatives but only generated %d.",
            n, len(neg_s),
        )

    logger.info("type_aware: generated %d hard negatives.", len(neg_s))
    return pd.DataFrame({"source": neg_s, "target": neg_t, "label": 0})


def get_hard_negatives_embedding_knn(
    pos_edges: pd.DataFrame,
    embedder,
    num_negatives: Optional[int] = None,
    random_state: int = 42,
    k_neighbors: int = 10,
    corrupt_tail_prob: float = 0.5,
) -> pd.DataFrame:
    """
    Hardest negatives via embedding K-nearest-neighbour (Strategy 3).

    For each positive pair, replace the tail (or head) with its nearest
    neighbour in embedding space, excluding known positives. These negatives
    sit closest to true treatments in the latent space and therefore provide
    the most informative training signal.

    Requires a trained HetionetEmbedder with loaded embeddings and scipy.
    Falls back to get_hard_negatives_degree_corrupt if unavailable.

    Args:
        pos_edges: DataFrame with columns [source, target].
        embedder: A HetionetEmbedder instance with loaded embeddings.
        num_negatives: Number of negatives to generate.
        random_state: RNG seed.
        k_neighbors: How many nearest neighbours to consider per entity.
        corrupt_tail_prob: Probability of corrupting the tail.

    Returns:
        DataFrame with columns [source, target, label=0].
    """
    if pos_edges.empty:
        return pd.DataFrame(columns=["source", "target", "label"])

    if embedder is None or not hasattr(embedder, "entity_embeddings") or \
            embedder.entity_embeddings is None:
        logger.warning(
            "embedding_knn: embedder not ready — falling back to degree_corrupt."
        )
        return get_hard_negatives_degree_corrupt(
            pos_edges, num_negatives, random_state, corrupt_tail_prob
        )

    try:
        from scipy.spatial import cKDTree as _KDTree
    except ImportError:
        logger.warning(
            "embedding_knn: scipy not available — falling back to degree_corrupt. "
            "Install with: pip install scipy"
        )
        return get_hard_negatives_degree_corrupt(
            pos_edges, num_negatives, random_state, corrupt_tail_prob
        )

    n = int(num_negatives or len(pos_edges))
    rng = np.random.default_rng(int(random_state))
    existing = _build_existing_pairs(pos_edges)

    entity_to_id: Dict[str, int] = getattr(embedder, "entity_to_id", {})
    id_to_entity: Dict[int, str] = getattr(embedder, "id_to_entity", {})
    emb_matrix = embedder.entity_embeddings  # shape (N, d)

    if len(entity_to_id) == 0 or emb_matrix is None:
        logger.warning(
            "embedding_knn: empty entity map — falling back to degree_corrupt."
        )
        return get_hard_negatives_degree_corrupt(
            pos_edges, num_negatives, random_state, corrupt_tail_prob
        )

    # Build KD-tree over all entity embeddings indexed in sorted order
    all_ids = sorted(entity_to_id.values())
    emb_order = np.array(all_ids, dtype=int)
    kd_tree = _KDTree(emb_matrix[emb_order])

    all_sources = pos_edges["source"].astype(str).values
    all_targets = pos_edges["target"].astype(str).values
    pos_pairs = list(zip(all_sources, all_targets))

    neg_s, neg_t = [], []
    attempts = 0
    max_attempts = max(5_000, n * 30)

    while len(neg_s) < n and attempts < max_attempts:
        s_pos, t_pos = pos_pairs[attempts % len(pos_pairs)]
        corrupt_tail = rng.random() < corrupt_tail_prob
        anchor_entity = str(t_pos) if corrupt_tail else str(s_pos)
        anchor_idx = entity_to_id.get(anchor_entity)

        if anchor_idx is None:
            attempts += 1
            continue

        anchor_emb = emb_matrix[anchor_idx].reshape(1, -1)
        _, nn_positions = kd_tree.query(
            anchor_emb, k=min(k_neighbors + 1, len(emb_order))
        )
        nn_ids = emb_order[nn_positions.ravel()]
        shuffled = rng.permutation(nn_ids)

        for cand_idx in shuffled:
            if cand_idx == anchor_idx:
                continue
            cand_entity = id_to_entity.get(int(cand_idx), "")
            if not cand_entity:
                continue
            s = str(s_pos) if corrupt_tail else cand_entity
            t = cand_entity if corrupt_tail else str(t_pos)
            if (s, t) not in existing:
                neg_s.append(s)
                neg_t.append(t)
                existing.add((s, t))
                break

        attempts += 1

    if len(neg_s) < n:
        logger.warning(
            "embedding_knn: requested %d but generated %d — "
            "consider increasing k_neighbors (current: %d).",
            n, len(neg_s), k_neighbors,
        )

    logger.info("embedding_knn: generated %d KNN hard negatives.", len(neg_s))
    return pd.DataFrame({"source": neg_s, "target": neg_t, "label": 0})


def get_hard_negatives(
    pos_edges: pd.DataFrame,
    strategy: str = "degree_corrupt",
    num_negatives: Optional[int] = None,
    random_state: int = 42,
    embedder=None,
    k_neighbors: int = 10,
    corrupt_tail_prob: float = 0.5,
) -> pd.DataFrame:
    """
    Unified entry point for hard negative generation.

    Dispatches to one of three strategies:
      - "degree_corrupt" (alias "hard"): degree-weighted KG corruption.
          Fast, no embedder required. Best default choice.
      - "type_aware": same-type-prefix corruption.
          Prevents trivially invalid pairs (e.g. Compound vs Disease).
      - "embedding_knn": K-nearest-neighbour in embedding space.
          Hardest negatives; requires trained embedder + scipy.

    Args:
        pos_edges: DataFrame of positive edges [source, target].
        strategy: One of "degree_corrupt" | "hard" | "type_aware" | "embedding_knn".
        num_negatives: How many negatives. Defaults to len(pos_edges).
        random_state: RNG seed for reproducibility.
        embedder: Required for "embedding_knn"; ignored otherwise.
        k_neighbors: KNN neighbourhood size (embedding_knn only).
        corrupt_tail_prob: Probability of corrupting tail vs head.

    Returns:
        DataFrame [source, target, label=0].
    """
    strategy = str(strategy).lower().strip()

    if strategy in ("degree_corrupt", "hard"):
        return get_hard_negatives_degree_corrupt(
            pos_edges, num_negatives, random_state, corrupt_tail_prob
        )
    elif strategy == "type_aware":
        return get_hard_negatives_type_aware(
            pos_edges, num_negatives, random_state, corrupt_tail_prob
        )
    elif strategy == "embedding_knn":
        return get_hard_negatives_embedding_knn(
            pos_edges, embedder, num_negatives, random_state,
            k_neighbors, corrupt_tail_prob
        )
    else:
        logger.warning(
            "Unknown hard negative strategy '%s'. "
            "Valid: degree_corrupt, type_aware, embedding_knn. "
            "Falling back to degree_corrupt.",
            strategy,
        )
        return get_hard_negatives_degree_corrupt(
            pos_edges, num_negatives, random_state, corrupt_tail_prob
        )
