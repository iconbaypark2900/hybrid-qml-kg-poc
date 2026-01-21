#!/usr/bin/env python3
"""
IBM Quantum Computer Test - Open Plan Compatible
Uses direct backend execution instead of sessions
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService

def load_env():
    """Load environment variables"""
    env_path = Path('.env')
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        value = value.strip().strip('"').strip("'").strip('{').strip('}')
                        os.environ[key.strip()] = value

def main():
    print("🚀 IBM Quantum Computer Test - Open Plan")
    print("=" * 50)
    
    load_env()
    
    token = os.getenv('IBM_Q_TOKEN')
    if not token:
        print("❌ No token found in .env file")
        return
    
    # Create a simple Bell state circuit
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)  # Superposition
    circuit.cx(0, 1)  # Entangle
    circuit.measure_all()
    
    print("\nBell State Circuit:")
    print(circuit.draw())
    
    try:
        # Connect to IBM Quantum Platform
        print("\n🌐 Connecting to IBM Quantum Platform...")
        service = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=token
        )
        
        # List available backends
        backends = service.backends()
        print(f"✅ Connected! Found {len(backends)} backends")
        
        # Show available backends
        for backend in backends:
            status = backend.status()
            print(f"  🔹 {backend.name} ({backend.configuration().n_qubits} qubits)")
            print(f"     Status: {status.status_msg}")
            print(f"     Operational: {status.operational}")
            print(f"     Queue: {status.pending_jobs} jobs")
            print()
        
        # Choose backend
        print("Available backends:")
        for i, backend in enumerate(backends):
            print(f"  {i+1}. {backend.name}")
        
        choice = input("\nSelect backend (1-2, default=1): ").strip()
        if not choice:
            choice = "1"
        
        try:
            backend_index = int(choice) - 1
            backend = backends[backend_index]
        except (ValueError, IndexError):
            backend = backends[0]
        
        print(f"\n🔬 Using: {backend.name}")
        
        # Check backend status
        status = backend.status()
        print(f"   Status: {status.status_msg}")
        print(f"   Operational: {status.operational}")
        print(f"   Queue: {status.pending_jobs} jobs")
        
        if not status.operational:
            print("⚠️  Backend is not operational")
            return
        
        # Transpile circuit for the backend
        print(f"\n🔧 Transpiling circuit for {backend.name}...")
        transpiled_circuit = transpile(circuit, backend=backend, optimization_level=1)
        
        print("Transpiled circuit:")
        print(transpiled_circuit.draw())
        
        # Execute on quantum computer using direct backend.run()
        print(f"\n🚀 Running on {backend.name}...")
        print("⚠️  This will use quantum credits and may take time due to queue")
        
        confirm = input("Proceed? (y/n): ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return
        
        # Use direct backend execution (open plan compatible)
        print("📤 Submitting job...")
        job = backend.run(transpiled_circuit, shots=1024)
        
        print(f"Job ID: {job.job_id()}")
        print("⏳ Waiting for results...")
        print("   (This may take several minutes due to queue)")
        
        # Monitor job status
        import time
        while job.status().name not in ['DONE', 'CANCELLED', 'ERROR']:
            status = job.status()
            print(f"   Status: {status.name}")
            time.sleep(30)  # Check every 30 seconds
        
        if job.status().name == 'DONE':
            # Get results
            result = job.result()
            counts = result.get_counts()
            
            # Display results
            print("\n📊 Quantum Results:")
            print("=" * 30)
            total = sum(counts.values())
            
            for state, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
                probability = count / total
                bar = "█" * int(probability * 30)
                print(f"|{state}⟩: {count:4d} ({probability:6.1%}) {bar}")
            
            # Analyze entanglement
            bell_states = counts.get('00', 0) + counts.get('11', 0)
            bell_percentage = (bell_states / total) * 100
            
            print(f"\n🔗 Entanglement Analysis:")
            print(f"   Bell states: {bell_percentage:.1f}%")
            
            if bell_percentage > 90:
                print("   ✅ Excellent quantum entanglement!")
            elif bell_percentage > 75:
                print("   ✅ Good quantum entanglement")
            else:
                print("   ⚠️  Noisy results (expected on real hardware)")
            
            print(f"\n🎉 Successfully ran on {backend.name}!")
            
        else:
            print(f"❌ Job failed with status: {job.status().name}")
            if hasattr(job, 'error_message'):
                print(f"   Error: {job.error_message()}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  - Check if your token is valid")
        print("  - Verify you have access to quantum backends")
        print("  - Make sure you're using the open plan correctly")

if __name__ == "__main__":
    main()
