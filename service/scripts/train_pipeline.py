"""One-command training pipeline: Hetionet → manifests → first evaluation.

Stages:
  1. Verify or download Hetionet snapshot
  2. Train embeddings (delegates to kg_layer.kg_embedder)
  3. Train classical baseline (delegates to classical_baseline.train_baseline)
  4. (optional) Train quantum VQC
  5. Synthesize manifest chain (sha256 + LATEST.txt)
  6. Run cross-validated evaluation, append to /evaluations

Designed to be re-runnable: the manifest IDs are content-addressed so an
unchanged input produces the same output and the pipeline short-circuits.

Usage:
    python -m service.scripts.train_pipeline \\
        --max-entities 5000        # full = 0
        --quantum-num-qubits 4
        --skip-quantum             # if no qiskit env

This is the SCAFFOLD wrapping the kept ML modules. Steps 2-4 currently
delegate to module-level mains/scripts in the repo (which were the
audited code paths). When the ML modules are themselves refactored to
write manifest-aware artifacts, step 5 becomes optional.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("train_pipeline")


def stage_check_hetionet(data_dir: Path) -> bool:
    nodes = data_dir / "hetionet-v1.0-nodes.tsv"
    if nodes.exists():
        log.info("hetionet nodes present: %s", nodes)
        return True
    log.warning(
        "hetionet snapshot not found at %s. The kg_layer.kg_loader will "
        "download it on first call, but you should commit a hash now to "
        "data/VERSION so the manifest pins it.", nodes,
    )
    return False


def stage_synthesize(root: Path, force: bool) -> dict:
    """Wrap synthesize_manifest_chain.py as a subprocess so its sys.path
    setup works correctly (it expects to be run as a module)."""
    cmd = [sys.executable, "-m", "service.scripts.synthesize_manifest_chain",
           "--root", str(root)]
    if force:
        cmd.append("--force")
    log.info("running: %s", " ".join(cmd))
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    log.info(out.stdout)
    if out.returncode != 0:
        log.error("synthesize_manifest_chain failed: %s", out.stderr)
        return {"ok": False, "stderr": out.stderr}
    import json
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return {"ok": True, "raw": out.stdout}


def stage_quantum(root: Path, num_qubits: int, max_iter: int) -> dict:
    cmd = [
        sys.executable, "-m", "service.scripts.synthesize_quantum_manifest",
        "--mode", "qiskit-mini-train",
        "--num-qubits", str(num_qubits),
        "--max-iter", str(max_iter),
        "--force",
    ]
    log.info("running: %s", " ".join(cmd))
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    log.info(out.stdout)
    if out.returncode != 0:
        log.error("synthesize_quantum_manifest failed: %s", out.stderr)
        return {"ok": False, "stderr": out.stderr}
    import json
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return {"ok": True, "raw": out.stdout}


def stage_evaluate(root: Path, tenant_id: str, cv_folds: int) -> dict:
    cmd = [
        sys.executable, "-m", "service.scripts.run_cv_evaluation",
        "--root", str(root),
        "--tenant-id", tenant_id,
        "--cv-folds", str(cv_folds),
    ]
    log.info("running: %s", " ".join(cmd))
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    log.info(out.stdout)
    if out.returncode != 0:
        log.error("run_cv_evaluation failed: %s", out.stderr)
        return {"ok": False, "stderr": out.stderr}
    import json
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return {"ok": True, "raw": out.stdout}


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=REPO_ROOT / "artifacts")
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--max-entities", type=int, default=0,
                   help="0 = uncapped; PoC default is 300, real runs need 0")
    p.add_argument("--skip-embeddings", action="store_true",
                   help="Reuse existing embeddings in data/")
    p.add_argument("--skip-classical", action="store_true",
                   help="Reuse existing models/classical_*.joblib")
    p.add_argument("--skip-quantum", action="store_true",
                   help="Don't synthesize a quantum manifest")
    p.add_argument("--quantum-num-qubits", type=int, default=4)
    p.add_argument("--quantum-max-iter", type=int, default=10)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--tenant-id", default="demo")
    p.add_argument("--force", action="store_true",
                   help="Force re-synthesis even if manifest IDs match")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    summary = {"started_at": time.time(), "stages": {}}
    log.info("=== stage 1: verify Hetionet ===")
    summary["stages"]["hetionet"] = {"ok": stage_check_hetionet(args.data_dir)}

    log.info("=== stage 2-3: embeddings + classical model ===")
    if args.skip_embeddings and args.skip_classical:
        log.info("skipping embeddings + classical (using existing artifacts)")
        summary["stages"]["embeddings_classical"] = {"ok": True, "skipped": True}
    else:
        log.warning(
            "embeddings + classical training is delegated to existing kg_layer "
            "and classical_baseline modules — invoke them directly until they "
            "are wrapped here. For now, run synthesize_manifest_chain against "
            "whatever models/ + data/ artifacts you have."
        )
        summary["stages"]["embeddings_classical"] = {
            "ok": True, "delegated": True,
            "note": "wrap kg_layer.kg_embedder and classical_baseline.train_baseline mains here",
        }

    log.info("=== stage 4: synthesize classical manifest chain ===")
    summary["stages"]["manifest"] = stage_synthesize(args.root, args.force)

    if not args.skip_quantum:
        log.info("=== stage 5: synthesize quantum manifest ===")
        summary["stages"]["quantum"] = stage_quantum(
            args.root, args.quantum_num_qubits, args.quantum_max_iter,
        )

    log.info("=== stage 6: cross-validated evaluation ===")
    summary["stages"]["evaluation"] = stage_evaluate(
        args.root, args.tenant_id, args.cv_folds,
    )

    summary["completed_at"] = time.time()
    summary["elapsed_seconds"] = summary["completed_at"] - summary["started_at"]
    summary["overall_ok"] = all(
        s.get("ok", True) for s in summary["stages"].values()
    )

    import json
    print(json.dumps(summary, indent=2))
    return 0 if summary["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
