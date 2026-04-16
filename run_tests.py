#!/usr/bin/env python3
"""
Quantum Improvements Testing Suite

This script provides both terminal and dashboard interfaces for testing
the quantum improvements in the hybrid QML-KG system.
"""

import sys
import os
import argparse
from pathlib import Path

# Project root (parent of tests/)
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.test_quantum_improvements_terminal import run_terminal_tests
import subprocess
import threading
import time
import webbrowser

def run_dashboard():
    """Run the Streamlit dashboard."""
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "tests/test_quantum_improvements_dashboard.py", "--server.port", "8501"],
        cwd=_ROOT,
    )

def main():
    parser = argparse.ArgumentParser(description="Quantum Improvements Testing Suite")
    parser.add_argument("--mode", choices=["terminal", "dashboard", "both"], default="both",
                        help="Run mode: terminal only, dashboard only, or both")
    parser.add_argument("--port", type=int, default=8501,
                        help="Port for dashboard (default: 8501)")
    
    args = parser.parse_args()
    
    if args.mode in ["terminal", "both"]:
        print("Running terminal tests...")
        run_terminal_tests()
        print("Running discovery metrics tests (tests/test_discovery_metrics.py)...")
        r = subprocess.run(
            [sys.executable, str(_ROOT / "tests" / "test_discovery_metrics.py")],
            cwd=_ROOT,
        )
        if r.returncode != 0:
            sys.exit(r.returncode)
    
    if args.mode in ["dashboard", "both"]:
        print(f"Starting dashboard on port {args.port}...")
        print("Dashboard will be available at: http://localhost:8501")
        
        # Open browser after a delay
        def open_browser():
            time.sleep(3)  # Wait for Streamlit to start
            webbrowser.open(f"http://localhost:{args.port}")
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.start()
        
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "tests/test_quantum_improvements_dashboard.py",
                "--server.port",
                str(args.port),
            ],
            cwd=_ROOT,
        )

if __name__ == "__main__":
    main()