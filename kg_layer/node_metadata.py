from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


@dataclass(frozen=True)
class NodeMetadata:
    node_id: str
    name: Optional[str] = None
    namespace: Optional[str] = None
    external_url: Optional[str] = None


def load_node_metadata_csv(path: str | Path) -> Dict[str, NodeMetadata]:
    """
    Load node metadata from a CSV with columns:
      - node_id (required): e.g. "Compound::DB01048"
      - name (optional): human-readable label
      - namespace (optional): e.g. "DrugBank", "DOID"
      - external_url (optional): resolver URL
    """
    p = Path(path)
    if not p.exists():
        return {}

    df = pd.read_csv(p, dtype=str).fillna("")
    if "node_id" not in df.columns:
        raise ValueError(f"Node metadata file {p} missing required column 'node_id'")

    out: Dict[str, NodeMetadata] = {}
    for _, r in df.iterrows():
        node_id = str(r.get("node_id", "")).strip()
        if not node_id:
            continue
        out[node_id] = NodeMetadata(
            node_id=node_id,
            name=(str(r.get("name", "")).strip() or None),
            namespace=(str(r.get("namespace", "")).strip() or None),
            external_url=(str(r.get("external_url", "")).strip() or None),
        )
    return out

