"""Sealed test-set wrapper enforcing the §7.2 unsealing-events convention.

Per ``preregistration/osf_preregistration_v1.md`` §7.2, the manuscript
test partition must only be accessed at two pre-registered unsealing
events:

  1. After all baseline implementations and simulator QSVC + ensemble
     evaluation, before hardware experiments — produces simulator-only
     test results.
  2. After hardware experiments — produces hardware-validated test
     results.

This module provides an enforcement mechanism for that convention.
``SealedTestSet`` wraps ``(features, labels)`` as private attributes;
any access requires an explicit ``unseal(reason, methodology_version)``
call that is logged permanently to
``results/test_set_unseal_log.jsonl``.

Each unseal record captures: ISO timestamp, reason, methodology version
(typically a git commit SHA), the SHA-256 of the test set itself (so
the audit log proves which test set was unsealed), and the row count.

Usage
-----

    from utils.sealed_test_set import SealedTestSet

    sealed = SealedTestSet(X_test, y_test)

    # ... earlier in the run, train/eval on train+val only ...

    # Pre-registered unsealing event 1:
    X, y = sealed.unseal(
        reason="simulator-only test results, pre-hardware",
        methodology_version="<git commit sha>",
    )

    # ... compute final test metrics ...

The wrapper does NOT prevent loading the underlying data from disk
directly — it's an enforced convention, not a security boundary. Its
value is the audit log: any unseal event is permanently recorded with
context, which strengthens the manuscript's reproducibility claim
without requiring trust in the methodology's narrative.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_LOG_PATH = os.path.join(REPO_ROOT, "results", "test_set_unseal_log.jsonl")


def _sha256_arrays(*arrays: np.ndarray) -> str:
    """Hash a sequence of numpy arrays in a content- and shape-aware way."""
    h = hashlib.sha256()
    for arr in arrays:
        a = np.ascontiguousarray(arr)
        h.update(str(a.shape).encode("utf-8"))
        h.update(str(a.dtype).encode("utf-8"))
        h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class UnsealEvent:
    """Audit record for one unsealing call. Serialized to the log as JSONL."""

    timestamp_utc: str
    reason: str
    methodology_version: str
    test_set_sha256: str
    n_rows: int

    def to_jsonl(self) -> str:
        return json.dumps(
            {
                "timestamp_utc": self.timestamp_utc,
                "reason": self.reason,
                "methodology_version": self.methodology_version,
                "test_set_sha256": self.test_set_sha256,
                "n_rows": self.n_rows,
            },
            sort_keys=True,
        )


class SealedTestSet:
    """Wrap a held-out test set so access is gated by a logged unseal call.

    The features and labels are stored as private attributes
    (``_features``, ``_labels``). Accessing them requires
    :meth:`unseal`, which records the access to the audit log.

    A non-mutating ``sha256()`` accessor is exposed so callers can
    confirm the wrapped data without unsealing it (e.g., the bootstrap
    driver writes the hash into its report metadata).
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        log_path: str | None = None,
    ) -> None:
        if features.ndim < 1:
            raise ValueError(f"features must have at least 1 dim, got {features.ndim}")
        if labels.ndim != 1:
            raise ValueError(f"labels must be 1-D, got shape {labels.shape}")
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"features and labels must agree on the first axis "
                f"({features.shape[0]} vs {labels.shape[0]})"
            )
        self._features = features
        self._labels = labels
        self._log_path = log_path or DEFAULT_LOG_PATH
        self._sha256 = _sha256_arrays(features, labels)

    @property
    def n_rows(self) -> int:
        """Row count is non-sensitive and can be inspected without unsealing."""
        return int(self._labels.shape[0])

    def sha256(self) -> str:
        """SHA-256 of the wrapped (features, labels) tuple. Non-unsealing."""
        return self._sha256

    def unseal(self, *, reason: str, methodology_version: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(features, labels)`` and append an audit record.

        Args:
            reason: human-readable description of why the test set is being
                unsealed (e.g., "simulator-only test results, pre-hardware").
                Must be non-empty.
            methodology_version: stable identifier for the code+config that
                will use the unsealed data. Typically a git commit SHA.
                Must be non-empty.

        Returns:
            ``(features, labels)`` — the same arrays the wrapper was
            constructed with. They are NOT copied; do not mutate.
        """
        if not reason or not reason.strip():
            raise ValueError("unseal reason must be a non-empty string")
        if not methodology_version or not methodology_version.strip():
            raise ValueError("methodology_version must be a non-empty string")

        event = UnsealEvent(
            timestamp_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            reason=reason.strip(),
            methodology_version=methodology_version.strip(),
            test_set_sha256=self._sha256,
            n_rows=self.n_rows,
        )
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(event.to_jsonl() + "\n")
        return self._features, self._labels


__all__ = ["SealedTestSet", "UnsealEvent", "DEFAULT_LOG_PATH"]
