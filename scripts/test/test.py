#!/usr/bin/env python3
"""
Test quantum circuit on IBM Brisbane or Torino
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
                        os.environ[key.strip()] = value.strip().strip('"').strip("'")

def main():
    print("IBM Quantum Test - semantics instance")
    print("=" * 60)
    
    # Load environment
    load_env()
    
    # Your specific instance
    instance = "crn:v1:bluemix:public:quantum-computing:us-east:a/556fb1e727034afe9a87f957315e6078:43e69c42-13c1-414f-9210-299452c18749::"
    
    try:
        # Connect to your instance
        print("Connecting to semantics instance...")
        service = QiskitRuntimeService(channel="ibm_cloud", instance=instance)
        print("✅ Connected successfully!")
        
        # Get your backends
        backends = service.backends()
        
        # Check status of your quantum computers
        print("\nYour quantum computers:")
        print("-" * 40)
        brisbane = service.backend("ibm_brisbane")
        torino = service.backend("ibm_torino")
        
        print(f"ibm_brisbane (127 qubits):")
        print(f"  Status: {brisbane.status().status_msg}")
        print(f"  Queue: {brisbane.status().pending_jobs} jobs")
        print(f"  Operational: {brisbane.status().operational}")
        
        print(f"\nibm_torino (133 qubits):")
        print(f"  Status: {torino.status().status_msg}")
        print(f"  Queue: {torino.status().pending_jobs} jobs")
        print(f"  Operational: {torino.status().operational}")
        
        # Select least busy
        if brisbane.status().pending_jobs <= torino.status().pending_jobs:
            backend = brisbane
            print(f"\n✅ Selected: ibm_brisbane (shorter queue)")
        else:
            backend = torino
            print(f"\n✅ Selected: ibm_torino (shorter queue)")
        
        # Create Bell state circuit
        print("\n" + "=" * 60)
        print("CREATING QUANTUM CIRCUIT:")
        print("=" * 60)
        
        circuit = QuantumCircuit(2)
        circuit.h(0)  # Superposition
        circuit.cx(0, 1)  # Entangle
        circuit.measure_all()
        
        print("\nBell State Circuit:")
        print(circuit.draw())
        
        # Transpile
        print("\nTranspiling for", backend.name, "...")
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        isa_circuit = pm.run(circuit)
        print("✅ Circuit optimized for quantum hardware")
        
        # Submit job
        print("\n🚀 SUBMITTING TO QUANTUM COMPUTER")
        print("-" * 40)
        shots = 1024
        print(f"Backend: {backend.name}")
        print(f"Shots: {shots}")
        print("Note: You have 10 minutes of quantum time available")
        
        confirm = input("\nSubmit job? (y/n): ").lower()
        if confirm != 'y':
            print("Job cancelled.")
            return
        
        sampler = Sampler(backend=backend)
        job = sampler.run([isa_circuit], shots=shots)
        print(f"✅ Job submitted!")
        print(f"   Job ID: {job.job_id()}")
        print("   ⏳ Waiting for quantum computer...")
        print("   (This may take a few minutes)")
        
        # Get results
        result = job.result()
        print("\n✅ Quantum computation complete!")
        
        # Show results
        print("\n" + "=" * 60)
        print("RESULTS FROM", backend.name.upper())
        print("=" * 60)
        
        counts = result[0].data.meas.get_counts()
        total = sum(counts.values())
        
        for state, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            prob = count / total
            bar = "█" * int(prob * 30)
            print(f"|{state}⟩: {count:4d} ({prob:6.2%}) {bar}")
        
        # Analyze entanglement
        bell_counts = counts.get('00', 0) + counts.get('11', 0)
        bell_prob = bell_counts / total
        
        print(f"\n✨ Entanglement Analysis:")
        print(f"   Bell states (|00⟩ + |11⟩): {bell_prob:.1%}")
        print(f"   Other states: {(1-bell_prob):.1%}")
        
        if bell_prob > 0.8:
            print("   ✅ Strong quantum entanglement observed!")
        elif bell_prob > 0.6:
            print("   ✓ Quantum entanglement detected (with noise)")
        else:
            print("   ⚠ High noise level (typical for real quantum hardware)")
        
        print("\n🎉 Successfully ran on a real quantum computer!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Added your token to .env file")
        print("2. Run setup.py first")

if __name__ == "__main__":
    main()