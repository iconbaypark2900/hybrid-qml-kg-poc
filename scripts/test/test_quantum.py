#!/usr/bin/env python3
"""
IBM Quantum test script
Tests quantum circuit execution on available backends
"""

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import os

def load_config():
    """Load saved configuration"""
    try:
        with open(".quantum_config", "r") as f:
            lines = f.readlines()
            channel = lines[0].strip()
            instance = lines[1].strip() if len(lines) > 1 else None
            return channel, instance
    except FileNotFoundError:
        return None, None

def connect_to_service():
    """Connect to IBM Quantum service"""
    channel, instance = load_config()
    
    if not channel:
        print("❌ No configuration found. Please run save_token.py first.")
        return None
    
    try:
        if instance:
            service = QiskitRuntimeService(channel=channel, instance=instance)
            print(f"✅ Connected using {channel} channel with instance {instance}")
        else:
            service = QiskitRuntimeService(channel=channel)
            print(f"✅ Connected using {channel} channel")
        return service
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        print("Please run save_token.py to reconfigure.")
        return None

def main():
    print("IBM Quantum Circuit Test")
    print("=" * 60)
    
    # Connect to service
    service = connect_to_service()
    if not service:
        return
    
    # Get backends
    print("\nFetching available backends...")
    backends = service.backends()
    
    if not backends:
        print("❌ No backends available")
        return
    
    print(f"Found {len(backends)} backends")
    
    # Separate simulators and real devices
    simulators = [b for b in backends if 'simulator' in b.name.lower()]
    real_devices = [b for b in backends if 'simulator' not in b.name.lower()]
    
    # Display available backends
    print("\n" + "=" * 60)
    print("AVAILABLE QUANTUM SYSTEMS:")
    print("=" * 60)
    
    if simulators:
        print(f"\n📊 SIMULATORS ({len(simulators)}):")
        for sim in simulators[:3]:
            status = "✓" if sim.status().operational else "✗"
            print(f"  {status} {sim.name}")
            print(f"     Queue: {sim.status().pending_jobs} jobs")
    
    if real_devices:
        print(f"\n🔬 REAL QUANTUM COMPUTERS ({len(real_devices)}):")
        operational_devices = [d for d in real_devices if d.status().operational]
        print(f"   Operational: {len(operational_devices)}/{len(real_devices)}")
        
        for device in operational_devices[:5]:
            print(f"\n  • {device.name}")
            print(f"     Qubits: {device.configuration().n_qubits}")
            print(f"     Queue: {device.status().pending_jobs} jobs")
            print(f"     Status: {device.status().status_msg}")
    
    # Select backend for test
    print("\n" + "=" * 60)
    print("SELECTING BACKEND FOR TEST:")
    print("=" * 60)
    
    try:
        # Try to get least busy operational backend
        backend = service.least_busy(operational=True)
        print(f"✅ Selected: {backend.name}")
        print(f"   Type: {'Simulator' if 'simulator' in backend.name.lower() else 'Real quantum computer'}")
        print(f"   Queue: {backend.status().pending_jobs} jobs pending")
    except Exception as e:
        print(f"Could not select least busy: {e}")
        # Fall back to first available
        backend = backends[0]
        print(f"✅ Using: {backend.name}")
    
    # Create quantum circuit
    print("\n" + "=" * 60)
    print("QUANTUM CIRCUIT TEST:")
    print("=" * 60)
    
    # Create Bell state circuit
    circuit = QuantumCircuit(2)
    circuit.h(0)  # Put qubit 0 in superposition
    circuit.cx(0, 1)  # Entangle qubits 0 and 1
    circuit.measure_all()
    
    print("\nCircuit (Bell State):")
    print(circuit.draw())
    print("\nThis circuit creates an entangled Bell state.")
    print("Expected results: ~50% |00⟩ and ~50% |11⟩")
    
    # Transpile for backend
    print("\n🔧 Transpiling circuit for backend...")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(circuit)
    print("✅ Circuit transpiled")
    
    # Run on quantum system
    print("\n🚀 Submitting job to quantum system...")
    print(f"   Backend: {backend.name}")
    print(f"   Shots: 1024")
    
    try:
        sampler = Sampler(backend=backend)
        job = sampler.run([isa_circuit], shots=1024)
        print(f"   Job ID: {job.job_id()}")
        print("   ⏳ Waiting for results...")
        
        result = job.result()
        print("   ✅ Job completed!")
        
        # Display results
        print("\n" + "=" * 60)
        print("RESULTS:")
        print("=" * 60)
        
        counts = result[0].data.meas.get_counts()
        total_shots = sum(counts.values())
        
        print(f"\nMeasurement outcomes from {total_shots} shots:")
        print("-" * 40)
        
        for state, count in sorted(counts.items()):
            probability = count / total_shots
            bar_length = int(probability * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"|{state}⟩: {count:4d} ({probability:6.2%}) {bar}")
        
        # Verify Bell state
        if '00' in counts and '11' in counts:
            bell_probability = (counts.get('00', 0) + counts.get('11', 0)) / total_shots
            print(f"\n✨ Bell state verification:")
            print(f"   |00⟩ + |11⟩ probability: {bell_probability:.1%}")
            if bell_probability > 0.9:
                print("   ✅ Excellent Bell state!")
            elif bell_probability > 0.7:
                print("   ✓ Good Bell state (some noise present)")
            else:
                print("   ⚠ Noisy results (expected on real hardware)")
        
    except Exception as e:
        print(f"\n❌ Error running circuit: {e}")
        print("\nTroubleshooting:")
        print("1. Check your account has quantum time remaining")
        print("2. Try again - the backend might be temporarily unavailable")
        print("3. Run save_token.py to reconfigure if needed")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()