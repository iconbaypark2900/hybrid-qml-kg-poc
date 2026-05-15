"""hetqml service shell — typed, async, multi-tenant FastAPI on top of the
existing kg_layer / classical_baseline / quantum_layer / utils modules.

See SERVICE_DESIGN.md for the design pressure-tested before this was built.
"""
from .app import create_app

__all__ = ["create_app"]
__version__ = "0.1.0"
