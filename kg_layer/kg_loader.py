# kg_layer/kg_loader.py

import os
import pandas as pd
import networkx as nx
from typing import Tuple, List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hetionet metaedge abbreviations (from https://het.io/edges)
METAEDGES = {
    'CtD': 'Compound treats Disease',
    'CpD': 'Compound palliates Disease',
    'DaG': 'Disease associates Gene',
    'DdG': 'Disease downregulates Gene',
    'DuG': 'Disease upregulates Gene',
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
    'DrD': 'Disease resembles Disease',
    'DlA': 'Disease localizes Anatomy',
    'N1': 'Node type 1',  # placeholder; full list in Hetionet docs
}

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


def load_hetionet_edges(data_dir: str = "data") -> pd.DataFrame:
    """
    Load Hetionet edges from local cache (auto-downloads if missing).

    Args:
        data_dir: The directory where the data is stored.

    Returns:
        A DataFrame with columns: ['source', 'metaedge', 'target'] (dtype=str).
    """
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
    relation_type: str = "CtD",
    max_entities: Optional[int] = None
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
    random_state: int = 42
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
    if num_negatives is None:
        num_negatives = len(task_edges)  # 1:1 ratio

    sources = task_edges["source_id"].values
    targets = task_edges["target_id"].values
    existing_pairs = set(zip(sources, targets))

    unique_sources = task_edges["source_id"].unique()
    unique_targets = task_edges["target_id"].unique()

    import numpy as np
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
        "source_id": neg_sources,
        "target_id": neg_targets,
        "label": 0
    })

def prepare_link_prediction_dataset(
    task_edges: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
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
    from sklearn.model_selection import train_test_split

    # Positive samples
    pos_df = task_edges[["source_id", "target_id"]].copy()
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

# Example usage (uncomment for testing)
# if __name__ == "__main__":
#     df = load_hetionet_edges()
#     task_edges, ent2id, id2ent = extract_task_edges(df, relation_type="CtD", max_entities=500)
#     train, test = prepare_link_prediction_dataset(task_edges)