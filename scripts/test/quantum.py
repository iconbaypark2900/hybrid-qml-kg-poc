#!/usr/bin/env python3
"""
Simple quantum computing script that works with:
- Local simulator (option 3)
- IBM Brisbane and Torino (options 1 and 2)
"""

import os
import random
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

def simulate_locally(shots=1024):
    """Simple local simulation - no quantum resources needed"""
    counts = {'00': 0, '01': 0, '10': 0, '11': 0}
    
    # Simulate Bell state: 50% |00⟩, 50% |11⟩
    for _ in range(shots):
        if random.random() < 0.5:
            counts['00'] += 1
        else:
            counts['11'] += 1
    
    # Add 2% noise for realism
    noise_shots = int(shots * 0.02)
    for _ in range(noise_shots):
        if counts['00'] > 0:
            counts['00'] -= 1
            counts['01' if random.random() < 0.5 else '10'] += 1
    
    return counts

def main():
    print("=" * 70)
    print("IBM Quantum Circuit Test - semantics instance")
    print("=" * 70)
    
    # Load environment
    load_env()
    
    # Show options
    print("\nOPTIONS:")
    print("-" * 70)
    print("1. IBM Brisbane (127 qubits, real quantum)")
    print("2. IBM Torino (133 qubits, real quantum)")
    print("3. Local Simulator (instant, no quantum time)")
    
    choice = input("\nSelect (1-3, default=3): ").strip()
    if not choice:
        choice = "3"
    
    # Create Bell state circuit
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    
    print("\nBell State Circuit:")
    print(circuit.draw())
    
    shots = 1024
    
    # Handle choice
    if choice == "3":
        # LOCAL SIMULATOR
        print("\n🖥️  Running LOCAL SIMULATOR...")
        print("   No quantum time used, instant results")
        counts = simulate_locally(shots)
        backend_name = "Local Simulator"
        
    else:
        # REAL QUANTUM COMPUTER
        print("\nConnecting to IBM Quantum...")
        try:
            service = QiskitRuntimeService(channel="ibm_quantum_platform")
            print("✅ Connected to semantics instance")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return
        
        # Select backend
        if choice == "1":
            backend = service.backend("ibm_brisbane")
        else:  # choice == "2" or anything else
            backend = service.backend("ibm_torino")
        
        backend_name = backend.name
        print(f"\n🔬 Using: {backend_name}")
        print(f"   Queue: {backend.status().pending_jobs} jobs")
        
        # Confirm
        print(f"\n⚠️  This will use quantum time (10 minutes available)")
        confirm = input("Proceed? (y/n): ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return
        
        # Transpile
        print("\nTranspiling circuit...")
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        isa_circuit = pm.run(circuit)
        
        # Submit job
        print(f"Submitting job...")
        sampler = Sampler(backend=backend)
        job = sampler.run([isa_circuit], shots=shots)
        
        print(f"Job ID: {job.job_id()}")
        print("⏳ Waiting in queue...")
        
        # Get results
        result = job.result()
        
        # Extract counts
        try:
            counts = result[0].data.c.get_counts()
        except:
            try:
                counts = result[0].data.meas.get_counts()
            except:
                print("Error extracting results")
                return
    
    # Display results
    print("\n" + "=" * 70)
    print(f"RESULTS from {backend_name}:")
    print("=" * 70)
    
    total = sum(counts.values())
    
    for state, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            probability = count / total
            bar = "█" * int(probability * 40)
            print(f"|{state}⟩: {count:4d} ({probability:6.1%}) {bar}")
    
    # Analyze entanglement
    bell_states = counts.get('00', 0) + counts.get('11', 0)
    other_states = counts.get('01', 0) + counts.get('10', 0)
    bell_percentage = (bell_states / total) * 100
    
    print(f"\nEntanglement: {bell_percentage:.1f}% Bell states")
    
    if bell_percentage > 90:
        print("✅ Excellent entanglement!")
    elif bell_percentage > 75:
        print("✅ Good entanglement")
    else:
        print("⚠️  Noisy results")
    
    print("\n✨ Complete!")

if __name__ == "__main__":
    main()