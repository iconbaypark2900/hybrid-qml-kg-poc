# Root entry point for Hugging Face Spaces (Streamlit looks for app.py by default).
# Runs the full dashboard so the Space shows the app even without README app_file.

import runpy
from pathlib import Path

if __name__ == "__main__":
    dashboard_path = Path(__file__).resolve().parent / "benchmarking" / "dashboard.py"
    runpy.run_path(str(dashboard_path), run_name="__main__")
