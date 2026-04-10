#!/usr/bin/env python3
"""
Verify IBM Quantum API token and optional instance CRN from the environment.

Usage (from repo root, with .env loaded):
  python scripts/test/test_token.py

Requires IBM_Q_TOKEN or IBM_QUANTUM_TOKEN in .env (never commit real tokens).
Optional: IBM_QUANTUM_INSTANCE (CRN), IBM_QUANTUM_CHANNEL.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


def main() -> int:
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        print("Install: pip install qiskit-ibm-runtime")
        return 1

    token = (os.environ.get("IBM_Q_TOKEN") or os.environ.get("IBM_QUANTUM_TOKEN") or "").strip()
    if not token:
        print("❌ Set IBM_Q_TOKEN or IBM_QUANTUM_TOKEN in .env")
        return 1

    channel = (os.environ.get("IBM_QUANTUM_CHANNEL") or "ibm_quantum_platform").strip()
    instance = (os.environ.get("IBM_QUANTUM_INSTANCE") or "").strip() or None

    print("Testing IBM Quantum Runtime...")
    print(f"  channel: {channel}")
    print(f"  instance: {'(default)' if not instance else '(CRN set)'}")

    try:
        kwargs = {"channel": channel, "token": token, "overwrite": True}
        if instance:
            kwargs["instance"] = instance
        QiskitRuntimeService.save_account(**kwargs)
        service = QiskitRuntimeService()
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return 1

    try:
        instances = service.instances()
        print(f"  instances: {len(instances) if instances else 0}")
    except Exception as e:
        print(f"  (instances list unavailable: {e})")

    backends = service.backends()
    print(f"  backends: {len(backends)}")
    real_devices = [b for b in backends if "simulator" not in b.name.lower()]
    if real_devices:
        print("  sample hardware:")
        for device in real_devices[:5]:
            op = "✓" if device.status().operational else "✗"
            print(f"    {op} {device.name}")

    print("✅ Token works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
