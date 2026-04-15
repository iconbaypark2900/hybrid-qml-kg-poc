"""
Hugging Face lite UI: Gradio shell over the same FastAPI app (middleware.api).

Run locally from repository root:
  ./scripts/run_hf_lite.sh
or:
  cd /path/to/hybrid-qml-kg-poc && PYTHONPATH=. python hf_space/app.py

Tier A: /status, /runs/latest, /analysis/summary, /quantum/config, /quantum/runtime/verify
Tier B: /viz/run-predictions, /viz/model-metrics, /viz/circuit-params (read-only JSON)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _bootstrap() -> None:
    root = _repo_root()
    os.chdir(root)
    rs = str(root)
    if rs not in sys.path:
        sys.path.insert(0, rs)


_bootstrap()

import gradio as gr  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

_api_app = None
_client: TestClient | None = None


def get_client() -> TestClient:
    global _api_app, _client
    if _client is None:
        from middleware.api import app as api_app

        _api_app = api_app
        _client = TestClient(_api_app, raise_server_exceptions=True)
    return _client


def _fmt(data: object) -> str:
    try:
        return json.dumps(data, indent=2, default=str)
    except Exception as e:
        return f"(could not serialize: {e})\n{data!r}"


def refresh_overview() -> tuple[str, str, str]:
    """Status, latest run, analysis summary."""
    c = get_client()
    status = c.get("/status")
    latest = c.get("/runs/latest")
    analysis = c.get("/analysis/summary")
    return (
        _fmt(status.json() if status.status_code == 200 else {"error": status.text}),
        _fmt(latest.json() if latest.status_code == 200 else {"error": latest.text}),
        _fmt(analysis.json() if analysis.status_code == 200 else {"error": analysis.text}),
    )


def refresh_quantum_config() -> str:
    c = get_client()
    r = c.get("/quantum/config")
    return _fmt(r.json() if r.status_code == 200 else {"error": r.text})


def verify_runtime(api_token: str, instance_crn: str, channel: str) -> str:
    """BYOK verify — token is only sent to IBM for this request (not logged by our API)."""
    token = (api_token or "").strip()
    if not token:
        token = (os.environ.get("IBM_Q_TOKEN") or os.environ.get("IBM_QUANTUM_TOKEN") or "").strip()
    if not token:
        return _fmt(
            {
                "status": "error",
                "message": "No token: paste one above or set Space secret IBM_Q_TOKEN / IBM_QUANTUM_TOKEN.",
            }
        )
    crn = (instance_crn or "").strip() or (os.environ.get("IBM_QUANTUM_INSTANCE") or "").strip() or None
    c = get_client()
    body = {
        "api_token": token,
        "instance_crn": crn,
        "channel": (channel or "ibm_quantum_platform").strip(),
    }
    r = c.post("/quantum/runtime/verify", json=body)
    return _fmt(r.json() if r.status_code == 200 else {"error": r.text, "status_code": r.status_code})


def refresh_viz_slice(top_k: int) -> tuple[str, str, str]:
    c = get_client()
    k = max(1, min(int(top_k or 20), 200))
    rp = c.get("/viz/run-predictions", params={"top_k": k})
    mm = c.get("/viz/model-metrics")
    cp = c.get("/viz/circuit-params")
    return (
        _fmt(rp.json() if rp.status_code == 200 else {"error": rp.text}),
        _fmt(mm.json() if mm.status_code == 200 else {"error": mm.text}),
        _fmt(cp.json() if cp.status_code == 200 else {"error": cp.text}),
    )


def build_demo():
    with gr.Blocks(title="Hybrid QML-KG (lite)") as demo:
        gr.Markdown(
            "### Hybrid QML-KG — Hugging Face lite\n"
            "Read-only views over the same **FastAPI** app as the full product. "
            "No pipeline jobs here — upload `data/`, `models/`, `results/` or run locally for full parity."
        )
        with gr.Tabs():
            with gr.Tab("System & results"):
                b = gr.Button("Refresh status / latest / analysis")
                out_status = gr.Textbox(label="/status", lines=12, max_lines=40)
                out_latest = gr.Textbox(label="/runs/latest", lines=12, max_lines=40)
                out_analysis = gr.Textbox(label="/analysis/summary", lines=12, max_lines=40)
                b.click(fn=refresh_overview, inputs=[], outputs=[out_status, out_latest, out_analysis])
            with gr.Tab("Quantum"):
                gr.Markdown("Configuration is read from `config/quantum_config.yaml`. IBM verify uses your token **in memory only** (see API docs).")
                q_refresh = gr.Button("Refresh quantum config")
                q_cfg = gr.Textbox(label="/quantum/config", lines=16, max_lines=50)
                q_refresh.click(fn=refresh_quantum_config, inputs=[], outputs=[q_cfg])
                gr.Markdown("#### IBM Quantum Runtime (BYOK)")
                tok = gr.Textbox(
                    label="API token",
                    type="password",
                    placeholder="Optional if Space secret IBM_Q_TOKEN is set",
                )
                crn = gr.Textbox(
                    label="Instance CRN (optional)",
                    placeholder="Or set IBM_QUANTUM_INSTANCE secret",
                )
                ch = gr.Textbox(label="Channel", value="ibm_quantum_platform")
                vbtn = gr.Button("Verify connection")
                vout = gr.Textbox(label="/quantum/runtime/verify response", lines=14, max_lines=50)
                vbtn.click(fn=verify_runtime, inputs=[tok, crn, ch], outputs=[vout])
            with gr.Tab("Visualizer (JSON)"):
                gr.Markdown("Subset of **Visualizer** data: run predictions, model metrics, circuit params.")
                topk = gr.Number(label="top_k (run predictions)", value=20, minimum=1, maximum=200, step=1)
                vb = gr.Button("Load /viz endpoints")
                v1 = gr.Textbox(label="/viz/run-predictions", lines=14, max_lines=60)
                v2 = gr.Textbox(label="/viz/model-metrics", lines=14, max_lines=60)
                v3 = gr.Textbox(label="/viz/circuit-params", lines=14, max_lines=60)
                vb.click(fn=refresh_viz_slice, inputs=[topk], outputs=[v1, v2, v3])

        demo.load(fn=refresh_overview, inputs=[], outputs=[out_status, out_latest, out_analysis])

    return demo


# Hugging Face Gradio runtime convention
demo = build_demo()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=port)
