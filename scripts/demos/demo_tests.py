#!/usr/bin/env python3
"""
Quantum Improvements Testing Suite - Demo Script

This script demonstrates how to run both terminal and dashboard testing interfaces.
"""

import subprocess
import sys
import os
import time
import threading
import webbrowser
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent

def run_terminal_tests():
    """Run terminal tests."""
    print("Running terminal tests...")
    result = subprocess.run(
        [sys.executable, "tests/test_quantum_improvements_terminal.py"],
        capture_output=True,
        text=True,
        cwd=_ROOT,
    )
    print("Terminal test output:")
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    return result.returncode

def run_dashboard():
    """Run the Streamlit dashboard."""
    print("Starting dashboard on port 8501...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "tests/test_quantum_improvements_dashboard.py",
            "--server.port",
            "8501",
        ],
        cwd=_ROOT,
    )

def main():
    print("Quantum Improvements Testing Suite - Demo")
    print("="*50)
    
    # Option 1: Run terminal tests only
    print("\n1. Running terminal tests...")
    run_terminal_tests()
    
    # Option 2: Launch dashboard
    print("\n2. Opening dashboard interface...")
    print("The dashboard will be available at http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard server.")
    
    # Open browser after a delay
    def open_browser():
        time.sleep(3)  # Wait for Streamlit to start
        webbrowser.open("http://localhost:8501")
    
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.start()
    
    # Run Streamlit
    try:
        run_dashboard()
    except KeyboardInterrupt:
        print("\nDashboard stopped by user.")

if __name__ == "__main__":
    main()