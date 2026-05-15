from __future__ import annotations

"""
On-disk cache for external API responses.

Used to make `validation_layer.{clinical_trials_validator,literature_validator,
opentargets_mapper}` deterministic for paper-table builds and CI runs that
must not hit live external services.

Cache key: SHA256 of `(provider, query_dict)`. Cache value: JSON.

Enable per-call:
    from validation_layer.api_cache import cached_call
    studies = cached_call("clinicaltrials.gov", {"q": "metformin"},
                          fetch_fn=lambda: query_clinical_trials(...))

Or globally via env var:
    CT_API_CACHE=1   → use cache for ClinicalTrials.gov queries
    PUBMED_CACHE=1   → use cache for PubMed
    OT_CACHE=1       → use cache for Open Targets
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path(os.environ.get("VALIDATION_API_CACHE_DIR",
                                         "artifacts/api_cache"))


def _key(provider: str, params: Dict) -> str:
    payload = json.dumps({"provider": provider, "params": params},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _path(provider: str, params: Dict, cache_dir: Optional[Path] = None) -> Path:
    base = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{provider.replace('/', '_')}_{_key(provider, params)}.json"


def cache_get(provider: str, params: Dict,
              cache_dir: Optional[Path] = None) -> Optional[Any]:
    """Return cached value or None if not present / corrupt."""
    p = _path(provider, params, cache_dir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"Corrupt cache entry {p}: {e}; ignoring.")
        return None


def cache_put(provider: str, params: Dict, value: Any,
              cache_dir: Optional[Path] = None) -> Path:
    """Persist value under (provider, params). Returns path."""
    p = _path(provider, params, cache_dir)
    p.write_text(json.dumps(value, indent=2, sort_keys=False), encoding="utf-8")
    return p


def cached_call(
    provider: str,
    params: Dict,
    fetch_fn: Callable[[], Any],
    *,
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False,
) -> Any:
    """
    Read-through cache: try cache → call fetch_fn → store result.

    Args:
        provider: short label, e.g. "clinicaltrials.gov" or "open_targets"
        params: dict that uniquely identifies the request
        fetch_fn: zero-arg callable that returns a JSON-serialisable value
        force_refresh: if True, skip cache and refresh

    Returns the value (cached or freshly fetched). Falls back to the empty
    fetch result on cache write failure (logged but not raised).
    """
    if not force_refresh:
        hit = cache_get(provider, params, cache_dir)
        if hit is not None:
            logger.debug(f"Cache hit: {provider} {_key(provider, params)}")
            return hit

    logger.debug(f"Cache miss: {provider} {_key(provider, params)}")
    value = fetch_fn()
    try:
        cache_put(provider, params, value, cache_dir)
    except Exception as e:
        logger.warning(f"Failed to write cache: {e}; returning value anyway.")
    return value


def clear_cache(provider: Optional[str] = None,
                cache_dir: Optional[Path] = None) -> int:
    """Delete cached entries; returns count removed. Provider=None = all."""
    base = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
    if not base.exists():
        return 0
    pattern = f"{provider}_*.json" if provider else "*.json"
    files = list(base.glob(pattern))
    for p in files:
        try:
            p.unlink()
        except Exception as e:
            logger.warning(f"Failed to remove {p}: {e}")
    return len(files)


def env_flag(provider: str) -> bool:
    """
    Convenience: check if caching is enabled for `provider` via env var.

    Recognised flags (any truthy value enables):
        CT_API_CACHE     → "clinicaltrials.gov"
        PUBMED_CACHE     → "pubmed"
        OT_CACHE         → "open_targets"
        DRUGBANK_CACHE   → "drugbank"
        VALIDATION_API_CACHE_ALL  → enable for every provider
    """
    if os.environ.get("VALIDATION_API_CACHE_ALL"):
        return True
    flag_map = {
        "clinicaltrials.gov": "CT_API_CACHE",
        "pubmed": "PUBMED_CACHE",
        "open_targets": "OT_CACHE",
        "drugbank": "DRUGBANK_CACHE",
    }
    flag = flag_map.get(provider)
    return bool(flag and os.environ.get(flag))
