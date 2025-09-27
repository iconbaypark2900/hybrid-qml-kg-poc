#!/usr/bin/env python3
"""
IBM Cloud Quantum Trial Account Setup
For Quantum Global Group trial accounts
"""

import os
import getpass
from qiskit_ibm_runtime import QiskitRuntimeService

def main():
    print("IBM Cloud Quantum Trial Account Setup")
    print("Account: Quantum Global Group")
    print("=" * 50)
    
    # Clear any existing accounts
    try:
        QiskitRuntimeService.delete_account()
        print("✓ Cleared existing configuration")
    except:
        pass
    
    # Get token
    token = os.environ.get('IBM_QUANTUM_TOKEN')
    if not token:
        print("\nNo IBM_QUANTUM_TOKEN environment variable found.")
        print("Please enter your IBM Quantum token:")
        print("(Get it from: https://quantum.ibm.com/account)")
        token = getpass.getpass("Token: ")
    
    if not token:
        print("❌ No token provided")
        return
    
    print("\nConfiguring IBM Cloud trial account...")
    
    # For IBM Cloud trial accounts, use ibm_cloud channel
    try:
        # Save account - let it auto-detect the instance
        QiskitRuntimeService.save_account(
            channel="ibm_cloud",
            token=token,
            overwrite=True
        )
        print("✅ Token saved successfully!")
        
        # Verify connection
        print("\nVerifying connection...")
        service = QiskitRuntimeService(channel="ibm_cloud")
        backends = service.backends()
        
        print(f"✅ Successfully authenticated!")
        print(f"✅ You have access to {len(backends)} backends")
        
        # Show available backends
        simulators = [b for b in backends if 'simulator' in b.name.lower()]
        real_devices = [b for b in backends if 'simulator' not in b.name.lower()]
        
        if simulators:
            print(f"\n📊 Simulators available: {len(simulators)}")
            for sim in simulators:
                print(f"   - {sim.name}")
        
        if real_devices:
            print(f"\n🔬 Real quantum computers available: {len(real_devices)}")
            for i, device in enumerate(real_devices[:10], 1):
                status = "🟢" if device.status().operational else "🔴"
                qubits = device.configuration().n_qubits if hasattr(device.configuration(), 'n_qubits') else "?"
                print(f"   {i:2d}. {status} {device.name} ({qubits} qubits)")
            if len(real_devices) > 10:
                print(f"   ... and {len(real_devices)-10} more")
        
        print("\n✨ Setup complete!")
        print("   Your IBM Cloud trial account is configured.")
        print("   You have 29 days remaining in your trial.")
        print("\n   You can now run: python test_quantum.py")
        
    except Exception as e:
        print(f"\n❌ Configuration failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure your token is correct")
        print("2. Regenerate your token at https://quantum.ibm.com/account")
        print("3. Check that your trial account is still active")

if __name__ == "__main__":
    main()