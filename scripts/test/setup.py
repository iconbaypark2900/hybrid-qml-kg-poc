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
    
    # Get all backends
    all_backends = service.backends()
    real_devices = [b for b in all_backends if 'simulator' not in b.name.lower()]
    simulators = [b for b in all_backends if 'simulator' in b.name.lower()]
    
    # Show available backends
    print("\n" + "-" * 70)
    print("YOUR QUANTUM COMPUTERS:")
    print("-" * 70)
    
    # Show real devices
    for i, device in enumerate(real_devices, 1):
        operational = "🟢" if device.status().operational else "🔴"
        queue = device.status().pending_jobs
        qubits = device.configuration().n_qubits
        print(f"\n{i}. {device.name} ({qubits} qubits)")
        print(f"   Status: {operational} {'Operational' if device.status().operational else 'Offline'}")
        print(f"   Queue: {queue} jobs")
    
    # Check for simulators
    if simulators:
        print(f"\nSimulators available:")
        for sim in simulators:
            print(f"   - {sim.name}")
    else:
        print(f"\n⚠️  Note: No simulators available in this instance")
    
    # Select backend
    print("\n" + "-" * 70)
    print("BACKEND SELECTION:")
    print("-" * 70)
    
    # Build options based on what's available
    print("\nOptions:")
    for i, device in enumerate(real_devices, 1):
        print(f"{i}. {device.name} (real quantum computer, {device.configuration().n_qubits} qubits)")
    
    if simulators:
        print(f"{len(real_devices)+1}. Use simulator (doesn't use quantum time)")
        default_choice = str(len(real_devices)+1)
    else:
        print("\n(No simulators available - must use real quantum computer)")
        default_choice = "2" if len(real_devices) > 1 else "1"
    
    choice = input(f"\nSelect backend (default={default_choice}): ").strip()
    
    # Handle choice
    if not choice:
        choice = default_choice
    
    try:
        choice_num = int(choice)
        if 1 <= choice_num <= len(real_devices):
            backend = real_devices[choice_num - 1]
            print(f"✓ Selected: {backend.name}")
        elif simulators and choice_num == len(real_devices) + 1:
            backend = simulators[0]
            print(f"✓ Selected: {backend.name} (simulator)")
        else:
            # Default to least busy
            backend = service.least_busy(operational=True)
            print(f"✓ Using least busy: {backend.name}")
    except:
        # Default to least busy
        backend = service.least_busy(operational=True)
        print(f"✓ Using least busy: {backend.name}")
    
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
        print(f"   Estimated wait time: varies based on queue")
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
        print("   (This may take several minutes or hours depending on queue)")
    
    # Get results
    result = job.result()
    print("\n✅ Quantum computation complete!")
    
    # Display results
    print("\n" + "=" * 70)
    print("QUANTUM MEASUREMENT RESULTS:")
    print("=" * 70)
    
    # Get counts - try different attribute names
    try:
        counts = result[0].data.c.get_counts()
    except:
        try:
            counts = result[0].data.meas.get_counts()
        except:
            # Generic approach
            counts = {}
            for key in dir(result[0].data):
                if not key.startswith('_'):
                    try:
                        data_attr = getattr(result[0].data, key)
                        if hasattr(data_attr, 'get_counts'):
                            counts = data_attr.get_counts()
                            break
                    except:
                        pass
    
    if not counts:
        print("Could not extract counts from results")
        print(f"Result structure: {result}")
        return
    
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