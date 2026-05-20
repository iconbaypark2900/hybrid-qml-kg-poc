#!/usr/bin/env python3
"""
Kernel-target alignment (KTA) pre-screen for ZZ vs Pauli feature maps.

Implements the workflow sketched in ``docs/roadmap/02_scientific_gaps.md`` §6:
evaluate ``kernel_target_alignment`` on a small training subset before committing
to a full QSVC kernel run.

Circuit construction mirrors ``quantum_layer/qml_encoder.py`` (ZZ linear;
Pauli with paulis ["Z", "ZZ"] defaults).

Examples:
    python scripts/compute_kta_zz_vs_pauli_subset.py --qml_dim 8 --n_samples 40
    python scripts/export_kta_xy_npz.py --out results/kta_train_subset.npz --subset 100 --qml_dim 16
    python scripts/compute_kta_zz_vs_pauli_subset.py --npz results/kta_train_subset.npz --qml_dim 16
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from qiskit.circuit.library import PauliFeatureMap, ZZFeatureMap  # noqa: E402
from qiskit_machine_learning.kernels import FidelityStatevectorKernel  # noqa: E402

from quantum_layer.quantum_kernel_alignment import kernel_target_alignment  # noqa: E402


def _rng_features(n_samples: int, qml_dim: int, rng: np.random.Generator) -> np.ndarray:
    """Random angles in [-pi, pi] — stable range for kernel feature-map inputs."""
    return rng.uniform(-np.pi, np.pi, size=(n_samples, qml_dim))


def load_xy_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    blob = np.load(path, allow_pickle=False)
    if "X" not in blob or "y" not in blob:
        raise ValueError("npz must contain arrays 'X' and 'y'")
    X = np.asarray(blob["X"], dtype=float)
    y = np.asarray(blob["y"]).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    if len(X) != len(y):
        raise ValueError("X and y row counts must match")
    y_cat = np.unique(y)
    if not np.all(np.isin(y_cat, [0, 1])):
        raise ValueError("labels must be binary in {0, 1}")
    return X.astype(float), y.astype(int)


def main() -> int:
    p = argparse.ArgumentParser(description="KTA scores for ZZ vs Pauli (subset / smoke).")
    p.add_argument("--qml_dim", type=int, default=16, help="Feature-map dimension (= qubits).")
    p.add_argument("--reps", type=int, default=2, help="Feature-map repetitions (matches pipeline default).")
    p.add_argument("--n_samples", type=int, default=100, help="Subset size when using synthetic data.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for synthetic features / subsampling.")
    p.add_argument(
        "--npz",
        type=str,
        default=None,
        help="Optional .npz with X (n, qml_dim) and y binary; first n_used rows after shuffle.",
    )
    p.add_argument("--n_used", type=int, default=None, help="Truncate npz subset to this many rows (default: min(n_samples,len)).")
    args = p.parse_args()

    if args.qml_dim < 2:
        raise SystemExit("--qml_dim must be >= 2")
    rng = np.random.default_rng(args.seed)

    if args.npz:
        X, y = load_xy_npz(args.npz)
        if X.shape[1] != args.qml_dim:
            raise SystemExit(f"npz X has shape {X.shape[1]} features; expected {args.qml_dim} (match --qml_dim)")
        n_cap = args.n_used or min(args.n_samples, len(X))
        if len(X) > n_cap:
            idx = rng.permutation(len(X))[:n_cap]
            X, y = X[idx], y[idx]
        elif args.n_used is not None:
            raise SystemExit(f"--n_used={args.n_used} but dataset has only {len(X)} rows")
    else:
        if args.n_samples < 10:
            raise SystemExit("--n_samples should be at least ~10 for a meaningful KTA")
        if args.n_samples > 240:
            print(
                f"WARNING: computing full {args.n_samples}×{args.n_samples} kernels can be slow; "
                "consider --n_samples 100.",
                file=sys.stderr,
            )
        X = _rng_features(args.n_samples, args.qml_dim, rng)
        y = rng.integers(0, 2, size=args.n_samples, endpoint=False).astype(int)
        pos = np.count_nonzero(y == 1)
        neg = len(y) - pos
        if pos == 0 or neg == 0:
            y[0] = 1
            y[-1] = 0

    fm_zz = ZZFeatureMap(
        feature_dimension=args.qml_dim,
        reps=args.reps,
        entanglement="linear",
    )
    fm_pauli = PauliFeatureMap(
        feature_dimension=args.qml_dim,
        reps=args.reps,
        paulis=["Z", "ZZ"],
    )

    k_zz = FidelityStatevectorKernel(feature_map=fm_zz).evaluate(X)
    k_pauli = FidelityStatevectorKernel(feature_map=fm_pauli).evaluate(X)

    score_zz = kernel_target_alignment(k_zz, y)
    score_pauli = kernel_target_alignment(k_pauli, y)

    print(f"n={len(X)} qml_dim={args.qml_dim} reps={args.reps} seed={args.seed}")
    print(f"ZZ KTA     : {score_zz:.6f}")
    print(f"Pauli KTA  : {score_pauli:.6f}")
    print(f"higher_alignment : {'Pauli' if score_pauli >= score_zz else 'ZZ'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
