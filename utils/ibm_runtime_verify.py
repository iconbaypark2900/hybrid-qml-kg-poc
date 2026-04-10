"""
Ephemeral IBM Quantum Runtime verification (BYOK).

Credentials are used only in-memory for the duration of the call — never persisted,
never written to logs by this module. Callers must not log request bodies.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional


def sanitize_quantum_config_for_client(config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Return a deep copy safe to send to browsers: redact API tokens and raw CRNs.
    """
    if not config:
        return None
    out = copy.deepcopy(config)
    iq = out.get("quantum", {}).get("ibm_quantum")
    if isinstance(iq, dict):
        if "token" in iq:
            t = iq.get("token")
            if isinstance(t, str) and t and not t.startswith("${"):
                iq["token"] = "[REDACTED]"
            elif isinstance(t, str) and t.startswith("${"):
                iq["token"] = "(from environment)"
        inst = iq.get("instance")
        if isinstance(inst, str) and inst and not inst.startswith("${") and len(inst) > 12:
            iq["instance"] = "[REDACTED — set in server .env or use BYOK form]"
    return out


def _redact_secrets_in_error_message(msg: str) -> str:
    return re.sub(r"[A-Za-z0-9_-]{24,}", "[REDACTED]", msg)


def verify_ibm_quantum_runtime(
    api_token: str,
    *,
    instance_crn: Optional[str] = None,
    channel: str = "ibm_quantum_platform",
) -> Dict[str, Any]:
    """
    Connect to IBM Quantum Runtime with user-supplied credentials (in-memory only).

    Returns only non-sensitive fields suitable for JSON responses.
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        return {
            "status": "error",
            "message": "qiskit-ibm-runtime is not installed on this server.",
            "backend_count": 0,
            "hardware_backend_names": [],
            "simulator_count": 0,
            "instances_count": None,
        }

    token = (api_token or "").strip()
    if not token:
        return {
            "status": "error",
            "message": "API token is empty.",
            "backend_count": 0,
            "hardware_backend_names": [],
            "simulator_count": 0,
            "instances_count": None,
        }

    kwargs: Dict[str, Any] = {"channel": channel.strip() or "ibm_quantum_platform", "token": token}
    inst = (instance_crn or "").strip()
    if inst:
        kwargs["instance"] = inst

    try:
        service = QiskitRuntimeService(**kwargs)
        backends = service.backends()
    except Exception as e:
        safe = _redact_secrets_in_error_message(str(e))
        return {
            "status": "error",
            "message": f"Could not connect to IBM Quantum Runtime: {safe[:500]}",
            "backend_count": 0,
            "hardware_backend_names": [],
            "simulator_count": 0,
            "instances_count": None,
        }

    hardware_names: List[str] = []
    sim_count = 0
    for b in backends:
        try:
            name = b.name
            if "simulator" in name.lower():
                sim_count += 1
            else:
                if len(hardware_names) < 15:
                    hardware_names.append(name)
        except Exception:
            continue

    instances_count: Optional[int] = None
    try:
        inst_list = service.instances()
        if inst_list is not None:
            instances_count = len(inst_list)
    except Exception:
        pass

    return {
        "status": "ok",
        "message": "Connected successfully. Credentials were not stored.",
        "backend_count": len(backends),
        "hardware_backend_names": hardware_names,
        "simulator_count": sim_count,
        "instances_count": instances_count,
    }
