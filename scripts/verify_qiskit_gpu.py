#!/usr/bin/env python3
"""DGX-side verification: confirm qiskit-aer-gpu / cuStateVec is wired up.

Run this on the DGX BEFORE launching ``scripts/run_bootstrap_ci.py --gpu``.
Reports:
  1. CUDA driver / nvidia-smi (informational; doesn't gate anything)
  2. qiskit + qiskit-aer + qiskit-aer-gpu versions
  3. Available Aer backends (looking for `aer_simulator_statevector_gpu`)
  4. cuStateVec registration via Aer's `available_devices()`
  5. A 1-shot statevector simulation on the GPU device to confirm it actually runs
  6. The project's QuantumExecutor.gpu_available() result, which is what
     scripts/run_bootstrap_ci.py keys off

Exit code: 0 if everything is wired up; non-zero with a descriptive
message if anything is missing.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _hr(label: str) -> None:
    print(f"\n=== {label} ===")


def main() -> int:
    failures: list[str] = []

    _hr("1. nvidia-smi (informational)")
    nvsmi = shutil.which("nvidia-smi")
    if nvsmi:
        try:
            out = subprocess.check_output(
                [nvsmi, "--query-gpu=name,driver_version,memory.total",
                 "--format=csv,noheader"],
                text=True, stderr=subprocess.STDOUT, timeout=10,
            ).strip()
            print(out)
        except Exception as e:
            print(f"nvidia-smi failed: {e}")
    else:
        print("nvidia-smi not on PATH (informational; not fatal)")

    _hr("2. Qiskit / Aer versions")
    try:
        import qiskit
        print(f"qiskit         {qiskit.__version__}")
    except Exception as e:
        failures.append(f"qiskit not importable: {e}")
        print(f"FAIL: {e}")
    try:
        import qiskit_aer
        print(f"qiskit-aer     {qiskit_aer.__version__}")
    except Exception as e:
        failures.append(f"qiskit-aer not importable: {e}")
        print(f"FAIL: {e}")
    try:
        import qiskit_machine_learning
        print(f"qiskit-ml      {qiskit_machine_learning.__version__}")
    except Exception as e:
        # Not strictly required for the GPU check, but the bootstrap driver needs it
        print(f"qiskit-machine-learning missing: {e}")

    _hr("3. AerSimulator available_devices()")
    try:
        from qiskit_aer import AerSimulator
        sim = AerSimulator()
        devices = sim.available_devices()
        print(f"available_devices = {devices}")
        if "GPU" not in devices:
            failures.append(
                "AerSimulator.available_devices() does not include 'GPU' — "
                "qiskit-aer-gpu either not installed or not seeing CUDA. "
                "Install with `pip install qiskit-aer-gpu` against your CUDA version."
            )
    except Exception as e:
        failures.append(f"AerSimulator probe failed: {e}")
        print(f"FAIL: {e}")

    _hr("4. AerSimulator backends list")
    try:
        from qiskit_aer import Aer
        backends = [b.name for b in Aer.backends()]
        print("\n".join(f"  - {b}" for b in backends))
        gpu_backends = [b for b in backends if "gpu" in b.lower()]
        if gpu_backends:
            print(f"\nGPU-flavored backends: {gpu_backends}")
        else:
            print("\n(no backends with 'gpu' in the name)")
    except Exception as e:
        print(f"backends listing failed: {e}")

    _hr("5. 1-shot statevector run on GPU")
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.save_statevector()
        sim_gpu = AerSimulator(method="statevector", device="GPU")
        result = sim_gpu.run(qc).result()
        sv = result.get_statevector()
        print(f"GPU statevector amplitudes: {sv}")
    except Exception as e:
        failures.append(
            f"GPU statevector simulation failed: {e}. "
            "This usually means cuStateVec is not registered or the driver/CUDA "
            "version mismatches the qiskit-aer-gpu wheel."
        )
        print(f"FAIL: {e}")

    _hr("6. Project's QuantumExecutor.gpu_available()")
    try:
        from quantum_layer.quantum_executor import QuantumExecutor
        avail = QuantumExecutor.gpu_available()
        print(f"QuantumExecutor.gpu_available() = {avail}")
        if not avail:
            failures.append(
                "QuantumExecutor.gpu_available() returned False — "
                "scripts/run_bootstrap_ci.py --gpu will NOT use the GPU. "
                "Investigate the GPU-detection logic in quantum_layer/quantum_executor.py."
            )
    except Exception as e:
        failures.append(f"QuantumExecutor.gpu_available() probe failed: {e}")
        print(f"FAIL: {e}")

    _hr("Summary")
    if failures:
        print("FAIL — issues to resolve:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("OK — qiskit-aer-gpu is wired up; you can launch:")
    print("  python scripts/run_bootstrap_ci.py --gpu")
    return 0


if __name__ == "__main__":
    sys.exit(main())
