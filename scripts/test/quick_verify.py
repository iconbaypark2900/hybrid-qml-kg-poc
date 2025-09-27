#!/usr/bin/env python3
"""
Quick verification that IBM Quantum connection is working
Run this after fixing and re-running save_token.py
"""

from qiskit_ibm_runtime import QiskitRuntimeService

try:
    # Connect with the correct channel for free tier
    service = QiskitRuntimeService(channel="ibm_quantum")
    
    # Get list of backends
    backends = service.backends()
    
    print("✅ SUCCESS! Connected to IBM Quantum")
    print(f"✅ You have access to {len(backends)} quantum systems\n")
    
    # Separate simulators and real devices
    simulators = [b.name for b in backends if 'simulator' in b.name.lower()]
    real_devices = [b.name for b in backends if 'simulator' not in b.name.lower()]
    
    print(f"📊 Simulators ({len(simulators)}):")
    for sim in simulators:
        print(f"   - {sim}")
    
    print(f"\n🔬 Real Quantum Computers ({len(real_devices)}):")
    for i, device in enumerate(real_devices[:10], 1):  # Show first 10
        print(f"   {i}. {device}")
    
    if len(real_devices) > 10:
        print(f"   ... and {len(real_devices)-10} more")
    
    print("\n✨ Everything is working! You can now run quantum circuits.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTroubleshooting steps:")
    print("1. Make sure you ran the fixed save_token.py")
    print("2. Verify your token is correct")
    print("3. Check that your IBM Quantum account is active at:")
    print("   https://quantum.ibm.com/account")