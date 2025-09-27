#!/usr/bin/env python3
"""
Test quantum circuit on your semantics instance
This will prove you can run circuits on IBM Brisbane or Torino
"""

import os
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def load_env():
    """Load .env file"""
    env_path = Path('.env')
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        value = value.strip().strip('"').strip("'")
                        if value and value != 'your_token_here':
                            os.environ[key.strip()] = value

def main():
    print("=" * 70)
    print("IBM Quantum Circuit Test - semantics instance")
    print("=" * 70)
    
    # Load environment
    load_env()
    
    # Connect using the CORRECT configuration
    print("\nConnecting to Quantum Global Group...")
    try:
        service = QiskitRuntimeService(channel="ibm_quantum_platform")
        print("✅ Connected to semantics instance!")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("Run setup_verified.py first!")
        return
    
    # Verify instance
    instances = service.instances()
    print(f"Instance: {instances[0]['name']}")
    
    # Show available backends
    print("\n" + "-" * 70)
    print("YOUR QUANTUM COMPUTERS:")
    print("-" * 70)
    
    brisbane = service.backend("ibm_brisbane")
    torino = service.backend("ibm_torino")
    
    print(f"\n1. IBM Brisbane (127 qubits)")
    print(f"   Status: {'🟢 Operational' if brisbane.status().operational else '🔴 Offline'}")
    print(f"   Queue: {brisbane.status().pending_jobs} jobs")
    
    print(f"\n2. IBM Torino (133 qubits)")
    print(f"   Status: {'🟢 Operational' if torino.status().operational else '🔴 Offline'}")
    print(f"   Queue: {torino.status().pending_jobs} jobs")
    
    # Select backend
    print("\n" + "-" * 70)
    print("BACKEND SELECTION:")
    print("-" * 70)
    
    print("\nOptions:")
    print("1. IBM Brisbane (real quantum computer)")
    print("2. IBM Torino (real quantum computer)")
    print("3. Simulator (for testing, doesn't use quantum time)")
    
    choice = input("\nSelect backend (1/2/3, default=3): ").strip()
    
    if choice == "1":
        backend = brisbane
        print(f"✓ Selected: IBM Brisbane")
    elif choice == "2":
        backend = torino
        print(f"✓ Selected: IBM Torino")
    else:
        backend = service.backend("simulator_mps")
        print(f"✓ Selected: Simulator (saving quantum time)")
    
    # Create quantum circuit
    print("\n" + "-" * 70)
    print("CREATING QUANTUM CIRCUIT:")
    print("-" * 70)
    
    # Bell state circuit
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)        # Hadamard on qubit 0
    circuit.cx(0, 1)    # CNOT from qubit 0 to 1
    circuit.measure([0, 1], [0, 1])
    
    print("\nBell State Circuit:")
    print(circuit.draw())
    print("\nThis creates quantum entanglement between two qubits.")
    print("Expected: ~50% |00⟩ and ~50% |11⟩ (no |01⟩ or |10⟩)")
    
    # Transpile circuit
    print("\n" + "-" * 70)
    print("PREPARING FOR QUANTUM EXECUTION:")
    print("-" * 70)
    
    print(f"\nTranspiling circuit for {backend.name}...")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(circuit)
    print("✅ Circuit optimized for quantum hardware")
    
    # Confirm before running on real hardware
    if 'simulator' not in backend.name.lower():
        print(f"\n⚠️  This will use your quantum time (10 minutes available)")
        print(f"   Current queue: {backend.status().pending_jobs} jobs")
        confirm = input("   Proceed? (y/n): ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return
    
    # Run circuit
    print("\n" + "-" * 70)
    print("RUNNING ON QUANTUM SYSTEM:")
    print("-" * 70)
    
    shots = 1024
    print(f"\nSubmitting job to {backend.name}...")
    print(f"Shots: {shots}")
    
    sampler = Sampler(backend=backend)
    job = sampler.run([isa_circuit], shots=shots)
    
    print(f"✅ Job submitted!")
    print(f"Job ID: {job.job_id()}")
    
    if 'simulator' in backend.name.lower():
        print("⏳ Running on simulator...")
    else:
        print("⏳ Waiting in queue for quantum computer...")
        print("   (This may take several minutes)")
    
    # Get results
    result = job.result()
    print("\n✅ Quantum computation complete!")
    
    # Display results
    print("\n" + "=" * 70)
    print("QUANTUM MEASUREMENT RESULTS:")
    print("=" * 70)
    
    counts = result[0].data.c.get_counts()  # Note: using 'c' for classical register
    total = sum(counts.values())
    
    print(f"\nMeasured {total} times on {backend.name}:")
    print("-" * 40)
    
    for state, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        probability = count / total
        bar = "█" * int(probability * 40)
        print(f"|{state}⟩: {count:4d} ({probability:6.1%}) {bar}")
    
    # Verify entanglement
    bell_states = counts.get('00', 0) + counts.get('11', 0)
    other_states = counts.get('01', 0) + counts.get('10', 0)
    bell_percentage = (bell_states / total) * 100
    
    print("\n" + "-" * 70)
    print("ENTANGLEMENT VERIFICATION:")
    print("-" * 70)
    print(f"Bell states (|00⟩ + |11⟩): {bell_states} ({bell_percentage:.1f}%)")
    print(f"Other states (|01⟩ + |10⟩): {other_states} ({(other_states/total)*100:.1f}%)")
    
    if bell_percentage > 90:
        print("✅ EXCELLENT: Strong quantum entanglement!")
    elif bell_percentage > 75:
        print("✅ GOOD: Clear entanglement with some noise")
    elif bell_percentage > 60:
        print("⚠️  NOISY: Entanglement visible but significant noise")
    else:
        print("❌ Very noisy - typical for real quantum hardware")
    
    print("\n" + "=" * 70)
    print("✨ SUCCESS! You ran a quantum circuit on the semantics instance!")
    print("=" * 70)

if __name__ == "__main__":
    main()