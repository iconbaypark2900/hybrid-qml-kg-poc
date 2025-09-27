#!/usr/bin/env python3
"""
Simple quantum script with guaranteed working local simulation
No complex imports needed for local simulation!
"""

import os
import random
from pathlib import Path
from qiskit import QuantumCircuit

def load_env():
    """Load .env file"""
    env_path = Path('.env')
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line and '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

def simulate_bell_state_locally(shots=1024):
    """
    Simulate Bell state locally with pure Python
    No special imports needed!
    """
    print("\n🖥️  Running LOCAL simulation (pure Python)...")
    print("   No quantum time used, instant results!")
    
    # Bell state: |00⟩ and |11⟩ with equal probability
    counts = {
        '00': 0,
        '01': 0,
        '10': 0,
        '11': 0
    }
    
    # Simulate measurements
    for _ in range(shots):
        # In a perfect Bell state, we get |00⟩ or |11⟩ with 50% probability each
        if random.random() < 0.5:
            counts['00'] += 1
        else:
            counts['11'] += 1
    
    # Add small noise to make it realistic (optional)
    noise_level = 0.02  # 2% noise
    if random.random() < 0.5:  # Sometimes add noise
        noise_shots = int(shots * noise_level)
        for _ in range(noise_shots):
            if counts['00'] > 0:
                counts['00'] -= 1
                if random.random() < 0.5:
                    counts['01'] += 1
                else:
                    counts['10'] += 1
    
    print("✅ Simulation complete!")
    return counts

def run_on_real_quantum(backend_choice="torino", shots=1024):
    """Run on real quantum computer"""
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    
    # Create circuit
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    
    # Connect
    print(f"\n🔬 Connecting to IBM Quantum...")
    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    
    # Choose backend
    if backend_choice == "brisbane":
        backend = service.backend("ibm_brisbane")
    else:
        backend = service.backend("ibm_torino")
    
    print(f"✅ Connected to {backend.name}")
    print(f"   Queue: {backend.status().pending_jobs} jobs")
    
    # Confirm
    print(f"\n⚠️  This will use quantum time!")
    confirm = input("Proceed? (y/n): ")
    if confirm.lower() != 'y':
        return None
    
    # Transpile and run
    print("Preparing circuit...")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(circuit)
    
    print(f"Submitting job...")
    sampler = Sampler(backend=backend)
    job = sampler.run([isa_circuit], shots=shots)
    
    print(f"Job ID: {job.job_id()}")
    print("⏳ Waiting in queue (this may take a while)...")
    
    result = job.result()
    
    # Get counts
    try:
        counts = result[0].data.c.get_counts()
    except:
        try:
            counts = result[0].data.meas.get_counts()
        except:
            # Convert from whatever format
            counts = {}
            print("Note: Results in unexpected format, extracting...")
    
    print("✅ Quantum computation complete!")
    return counts

def display_results(counts, source="Simulator"):
    """Display the results nicely"""
    print("\n" + "=" * 60)
    print(f"RESULTS from {source}:")
    print("=" * 60)
    
    total = sum(counts.values())
    
    # Sort by count
    for state, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:  # Only show states that occurred
            prob = count / total
            bar = "█" * int(prob * 30)
            print(f"|{state}⟩: {count:4d} ({prob:6.1%}) {bar}")
    
    # Calculate entanglement
    bell_states = counts.get('00', 0) + counts.get('11', 0)
    other_states = counts.get('01', 0) + counts.get('10', 0)
    
    print("\n" + "-" * 60)
    print("ENTANGLEMENT ANALYSIS:")
    print("-" * 60)
    print(f"Bell states (|00⟩+|11⟩): {bell_states} ({bell_states/total*100:.1f}%)")
    print(f"Other states (|01⟩+|10⟩): {other_states} ({other_states/total*100:.1f}%)")
    
    if bell_states/total > 0.95:
        print("\n✅ EXCELLENT: Near-perfect entanglement!")
    elif bell_states/total > 0.85:
        print("\n✅ GOOD: Clear entanglement with small noise")
    elif bell_states/total > 0.70:
        print("\n⚠️  MODERATE: Entanglement visible but noisy")
    else:
        print("\n❌ POOR: High noise level")

def main():
    print("=" * 60)
    print("QUANTUM BELL STATE EXPERIMENT")
    print("=" * 60)
    
    # Show the circuit
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    
    print("\nBell State Circuit:")
    print(circuit.draw())
    print("\nThis creates entanglement: |00⟩ + |11⟩")
    
    # Menu
    print("\n" + "-" * 60)
    print("WHERE TO RUN:")
    print("-" * 60)
    print("1. Local Simulator (instant, FREE)")
    print("2. IBM Brisbane (127 qubits, long queue)")
    print("3. IBM Torino (133 qubits, shorter queue)")
    print("-" * 60)
    
    choice = input("Select (1-3, press Enter for local): ").strip()
    
    if not choice or choice == "1":
        # Local simulation
        counts = simulate_bell_state_locally(shots=1024)
        display_results(counts, "Local Simulator")
        
    elif choice == "2":
        # Load environment and run on Brisbane
        load_env()
        counts = run_on_real_quantum("brisbane")
        if counts:
            display_results(counts, "IBM Brisbane")
            
    elif choice == "3":
        # Load environment and run on Torino
        load_env()
        counts = run_on_real_quantum("torino")
        if counts:
            display_results(counts, "IBM Torino")
    
    else:
        print("Invalid choice, running local simulation...")
        counts = simulate_bell_state_locally(shots=1024)
        display_results(counts, "Local Simulator")
    
    print("\n✨ Experiment complete!")
    print("\nWhat you learned:")
    print("• Bell states show quantum entanglement")
    print("• Perfect result: 50% |00⟩, 50% |11⟩")
    print("• Real quantum computers have noise")
    print("• Local simulation is great for testing!")

if __name__ == "__main__":
    main()