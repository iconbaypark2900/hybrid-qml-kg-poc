#!/usr/bin/env python3
"""
Test if the token is clean and can connect
"""

import os
from pathlib import Path
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
    print("🔍 Testing Token Cleanliness")
    print("=" * 40)
    
    load_env()
    
    token = os.getenv('IBM_Q_TOKEN')
    instance = os.getenv('IBM_QUANTUM_INSTANCE')
    
    print(f"Token: {token}")
    print(f"Instance: {instance[:50]}...")
    
    # Check for problematic characters
    if token:
        if token.startswith('{') or token.startswith('"'):
            print("❌ Token has problematic characters!")
            print("   Clean token:", token.strip('{"').strip('}"'))
        else:
            print("✅ Token looks clean")
    
    # Test connection
    try:
        print("\n🌐 Testing connection...")
        service = QiskitRuntimeService(
            channel="ibm_cloud",
            token=token,
            instance=instance
        )
        print("✅ Connection successful!")
        
        backends = service.backends()
        print(f"📋 Found {len(backends)} backends")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    main()
