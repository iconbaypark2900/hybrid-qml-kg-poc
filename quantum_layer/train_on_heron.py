"""
Thin shim — delegates to the CLI script ``scripts/train_on_heron.py``
for full argument handling, hard-negative generation, and provenance.
"""

import subprocess
import sys
import logging

logger = logging.getLogger(__name__)


def train_on_heron(**kwargs):
    """Train QML model on IBM Heron processor.

    All keyword arguments are forwarded as CLI flags to
    ``scripts/train_on_heron.py``.  For example::

        train_on_heron(relation="CtD", max_entities=200, qubits=4, dry_run=True)
    """
    cmd = [sys.executable, "scripts/train_on_heron.py"]
    for k, v in kwargs.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd += [flag, str(v)]

    logger.info("Delegating to CLI: %s", " ".join(cmd))
    return subprocess.run(cmd, check=True)


if __name__ == "__main__":
    train_on_heron()
