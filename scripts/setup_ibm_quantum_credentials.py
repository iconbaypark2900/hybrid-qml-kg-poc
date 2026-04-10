#!/usr/bin/env python3
"""
One-time helper: save IBM Quantum credentials for Qiskit Runtime CLI usage.

1. Copy .env.example to .env and set IBM_Q_TOKEN (and optionally IBM_QUANTUM_INSTANCE).
2. Run from repo root:  python scripts/setup_ibm_quantum_credentials.py

This calls QiskitRuntimeService.save_account() so `QiskitRuntimeService()` works without
passing a token each time. Optional; QuantumExecutor also reads .env directly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Repo root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


def main() -> int:
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        print("Install qiskit-ibm-runtime: pip install qiskit-ibm-runtime")
        return 1

    token = (os.environ.get("IBM_Q_TOKEN") or os.environ.get("IBM_QUANTUM_TOKEN") or "").strip()
    if not token or token == "your_ibm_quantum_token_here":
        print("Set IBM_Q_TOKEN or IBM_QUANTUM_TOKEN in .env (see .env.example).")
        return 1

    channel = (os.environ.get("IBM_QUANTUM_CHANNEL") or "ibm_quantum_platform").strip()
    instance = (os.environ.get("IBM_QUANTUM_INSTANCE") or "").strip() or None

    kwargs = {"channel": channel, "token": token, "overwrite": True}
    if instance:
        kwargs["instance"] = instance

    QiskitRuntimeService.save_account(**kwargs)
    print("Saved IBM Quantum account for channel=%s%s." % (channel, " with instance CRN" if instance else ""))
    svc = QiskitRuntimeService()
    n = len(svc.backends())
    print("Connection OK: %d backend(s) visible." % n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
