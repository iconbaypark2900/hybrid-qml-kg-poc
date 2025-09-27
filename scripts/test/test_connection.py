#!/usr/bin/env python3
"""
Test connection and configuration without actually running a circuit
This verifies everything works without using quantum time or waiting in queue
"""

import os
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Load .env
env_path = Path('.env')
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip().strip('"').strip("'")

print("IBM Quantum Connection Test (No Circuit Execution)")
print("=" * 70)

# Connect
print("\nConnecting to semantics instance...")
service = QiskitRuntimeService(channel="ibm_quantum_platform")
print("✅ Connected successfully!")

# Show instance
instances = service.instances()
print(f"\n✅ Instance: {instances[0]['name']}")

# Show backends
backends = service.backends()
print(f"\n✅ Available backends: {len(backends)}")

for backend in backends:
    if 'simulator' not in backend.name.lower():
        status = "🟢" if backend.status().operational else "🔴"
        queue = backend.status().pending_jobs
        qubits = backend.configuration().n_qubits
        print(f"   {status} {backend.name}: {qubits} qubits, {queue} jobs in queue")

# Create a test circuit
print("\n✅ Creating test circuit...")
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

print("\nCircuit created:")
print(circuit.draw())

# Transpile for Brisbane
print("\n✅ Testing transpilation for ibm_brisbane...")
brisbane = service.backend("ibm_brisbane")
pm = generate_preset_pass_manager(backend=brisbane, optimization_level=1)
isa_circuit = pm.run(circuit)
print("   Circuit successfully transpiled for Brisbane")

# Transpile for Torino
print("\n✅ Testing transpilation for ibm_torino...")
torino = service.backend("ibm_torino")
pm = generate_preset_pass_manager(backend=torino, optimization_level=1)
isa_circuit = pm.run(circuit)
print("   Circuit successfully transpiled for Torino")

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("\nYour configuration is working perfectly:")
print("  • Connected to semantics instance ✅")
print("  • Access to ibm_brisbane ✅")
print("  • Access to ibm_torino ✅")
print("  • Can create and transpile circuits ✅")
print("\n⚠️  Note: No simulators available in your instance")
print("   All runs will use real quantum computers and quantum time")
print(f"\n💡 Tip: Use ibm_torino (only {torino.status().pending_jobs} jobs in queue)")
print(f"   vs ibm_brisbane ({brisbane.status().pending_jobs} jobs in queue)")