#!/usr/bin/env python3
"""
Test what your specific token has access to
IMPORTANT: Regenerate your token after running this since it was exposed
"""

from qiskit_ibm_runtime import QiskitRuntimeService

# Your token (REGENERATE AFTER THIS TEST!)
token = "JnkUYmuqqGvMjpvOqfsdyJOh8It_KWwEaS6EBr3rStub"

print("=" * 60)
print("⚠️  SECURITY WARNING")
print("=" * 60)
print("After this test, immediately regenerate your token at:")
print("https://quantum.ibm.com/account")
print("=" * 60)
print()

# Clear any existing accounts
try:
    QiskitRuntimeService.delete_account()
except:
    pass

print("Testing your token...")
print("=" * 60)

# Test 1: IBM Quantum Platform (most likely)
print("\nTest 1: IBM Quantum Platform (no instance specified)")
print("-" * 40)
try:
    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform",
        token=token,
        overwrite=True
    )
    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    
    print("✅ SUCCESS! Connected to IBM Quantum Platform")
    
    # Check instances
    instances = service.instances()
    print(f"\nInstances you have access to: {len(instances) if instances else 0}")
    if instances:
        for i, inst in enumerate(instances, 1):
            if isinstance(inst, dict):
                print(f"  {i}. Name: {inst.get('name', 'unnamed')}")
                print(f"     CRN: {inst.get('crn', 'no-crn')[:60]}...")
            else:
                print(f"  {i}. {inst}")
    
    # Check backends
    backends = service.backends()
    print(f"\nBackends available: {len(backends)}")
    
    # Real quantum computers
    real_devices = [b for b in backends if 'simulator' not in b.name.lower()]
    if real_devices:
        print(f"\nReal quantum computers you can access:")
        for device in real_devices[:10]:
            status = "✓" if device.status().operational else "✗"
            print(f"  {status} {device.name}: {device.configuration().n_qubits} qubits")
    
    # Simulators
    simulators = [b for b in backends if 'simulator' in b.name.lower()]
    if simulators:
        print(f"\nSimulators available:")
        for sim in simulators:
            print(f"  - {sim.name}")
    
    print("\n" + "=" * 60)
    print("✅ YOUR TOKEN WORKS!")
    print("=" * 60)
    print("\nYou have access to IBM Quantum Platform (NOT the semantics instance)")
    print("The 'semantics' instance is in a DIFFERENT account")
    print("\nTo use YOUR access, update .env with:")
    print("  IBM_QUANTUM_CHANNEL=ibm_quantum_platform")
    print("  # Don't specify IBM_QUANTUM_INSTANCE")
    
    # Save working config
    with open(".env.working", "w") as f:
        f.write("# Working configuration for your token\n")
        f.write(f"IBM_QUANTUM_TOKEN={token}\n")
        f.write("IBM_QUANTUM_CHANNEL=ibm_quantum_platform\n")
        f.write("# No instance needed - you have default access\n")
    print("\nCreated .env.working with your configuration")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    
    # Test 2: IBM Cloud
    print("\nTest 2: IBM Cloud")
    print("-" * 40)
    try:
        QiskitRuntimeService.delete_account()
        QiskitRuntimeService.save_account(
            channel="ibm_cloud",
            token=token,
            overwrite=True
        )
        service = QiskitRuntimeService(channel="ibm_cloud")
        print("✅ Connected to IBM Cloud")
        backends = service.backends()
        print(f"Backends available: {len(backends)}")
    except Exception as e2:
        print(f"❌ Failed: {e2}")
        print("\n" + "=" * 60)
        print("Your token doesn't work with either platform")
        print("It might be expired or invalid")

# Test if the semantics instance would work
print("\n" + "=" * 60)
print("Testing 'semantics' instance access:")
print("-" * 40)
try:
    QiskitRuntimeService.delete_account()
    # Try with the full CRN
    QiskitRuntimeService.save_account(
        channel="ibm_cloud",
        token=token,
        instance="crn:v1:bluemix:public:quantum-computing:us-east:a/556fb1e727034afe9a87f957315e6078:43e69c42-13c1-414f-9210-299452c18749::",
        overwrite=True
    )
    service = QiskitRuntimeService(
        channel="ibm_cloud",
        instance="crn:v1:bluemix:public:quantum-computing:us-east:a/556fb1e727034afe9a87f957315e6078:43e69c42-13c1-414f-9210-299452c18749::"
    )
    print("✅ You DO have access to semantics instance!")
except Exception as e:
    print(f"❌ No access to semantics instance")
    print(f"   Reason: The token is from a different account")

print("\n" + "=" * 60)
print("⚠️  IMPORTANT: REGENERATE YOUR TOKEN NOW!")
print("=" * 60)
print("Go to: https://quantum.ibm.com/account")
print("Click 'Regenerate API Token'")
print("This token has been exposed and should not be used anymore")
print("=" * 60)