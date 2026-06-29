#!/usr/bin/env python3
"""HTTP-level smoke checks for critical v2 dashboard routes."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request


BASE_URL = os.environ.get("V2_BASE_URL", "http://localhost:3780").rstrip("/")
ROUTES = [
    ("/v2/start", "Start with your own biomedical question"),
    (
        "/v2/experiment?entity=Atherosclerosis&runMode=Hybrid&candidate=Atherosclerosis",
        "Scientific quality controls",
    ),
    (
        "/v2/validation?entity=Atherosclerosis&runMode=Hybrid&candidate=Atherosclerosis",
        "Evidence axes",
    ),
    (
        "/v2/visual?entity=Atherosclerosis&runMode=Hybrid&candidate=Atherosclerosis",
        "Visual",
    ),
    ("/v2/repurposing?disease_id=brca_external_validation", "Vemurafenib"),
    (
        "/v2/repurposing?disease_id=brca_external_validation_organism_any",
        "Prednisolone",
    ),
    ("/v2/repurposing?disease_id=all_pairs_kg_omics", "Fluticasone furoate"),
]


def fetch(path: str) -> tuple[int, str, float]:
    url = f"{BASE_URL}{path}"
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status, body, (time.perf_counter() - started) * 1000
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, body, (time.perf_counter() - started) * 1000


def main() -> int:
    results = []
    ok = True
    for path, marker in ROUTES:
        status, body, duration_ms = fetch(path)
        marker_present = marker in body
        route_ok = status == 200 and marker_present
        ok = ok and route_ok
        results.append(
            {
                "path": path,
                "status": status,
                "duration_ms": round(duration_ms, 2),
                "marker": marker,
                "marker_present": marker_present,
                "ok": route_ok,
            }
        )

    print(json.dumps({"base_url": BASE_URL, "ok": ok, "routes": results}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
