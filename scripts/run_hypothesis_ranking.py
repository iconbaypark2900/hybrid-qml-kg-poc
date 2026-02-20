#!/usr/bin/env python3
"""
Run hypothesis-based candidate ranking and ablation comparison.

Compares baseline CtD model vs mechanism-informed ranking.
Logs to MetricsTracker and generates ablation_comparison.md report.

Usage:
    python scripts/run_hypothesis_ranking.py --hypothesis H-001 --disease DOID:9352
    python scripts/run_hypothesis_ranking.py --hypothesis H-002 --disease DOID:9352 --top_k 20
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from benchmarking.metrics_tracker import MetricsTracker
from benchmarking.report_generator import generate_ablation_report
from benchmarking.negative_controls import (
    get_random_controls,
    get_mock_mechanism_controls,
)
from kg_layer.kg_loader import load_hetionet_edges
from kg_layer.hypothesis_graph import build_mechanism_subgraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ensure_data_dirs():
    """Create data/hypothesis directory for future data."""
    os.makedirs("data/hypothesis", exist_ok=True)


def run_ranking(
    hypothesis_id: str,
    disease_id: str,
    top_k: int = 50,
    use_orchestrator: bool = True,
) -> dict:
    """
    Run mechanism-informed ranking via orchestrator or fallback.

    Returns dict with ranked_candidates, model_used, scores for ablation.
    """
    if use_orchestrator:
        try:
            from middleware.orchestrator import LinkPredictionOrchestrator
            orch = LinkPredictionOrchestrator(use_quantum=False)
            result = orch.rank_mechanism_candidates(
                hypothesis_id=hypothesis_id,
                disease_id=disease_id,
                top_k=top_k,
            )
            if result.get("status") == "success":
                scores = [c["score"] for c in result.get("ranked_candidates", [])]
                return {
                    "ranked_candidates": result.get("ranked_candidates", []),
                    "model_used": result.get("model_used", "classical"),
                    "scores": scores,
                    "mean_score": np.mean(scores) if scores else 0.0,
                }
        except Exception as e:
            logger.warning(f"Orchestrator not available: {e}. Using fallback.")
            use_orchestrator = False

    if not use_orchestrator:
        return _fallback_ranking(hypothesis_id, disease_id, top_k)
    return {"ranked_candidates": [], "model_used": "error", "scores": [], "mean_score": 0.0}


def _fallback_ranking(hypothesis_id: str, disease_id: str, top_k: int) -> dict:
    """Fallback when orchestrator/embeddings unavailable: return placeholder."""
    logger.info("Using fallback (no rankings - run pipeline first to train models)")
    return {
        "ranked_candidates": [],
        "model_used": "fallback_unavailable",
        "scores": [],
        "mean_score": 0.0,
    }


def run_ablation(
    hypothesis_id: str,
    disease_id: str,
    top_k: int = 50,
) -> None:
    """
    Run baseline vs mechanism-informed ranking, log metrics, generate report.
    """
    _ensure_data_dirs()
    tracker = MetricsTracker(results_dir="results")
    tracker.start_run({
        "hypothesis_id": hypothesis_id,
        "disease_id": disease_id,
        "top_k": top_k,
        "task": "hypothesis_ranking_ablation",
    })

    # Mechanism-informed ranking
    mech = run_ranking(hypothesis_id, disease_id, top_k, use_orchestrator=True)
    mechanism_scores = mech.get("scores", [])
    mechanism_mean = mech.get("mean_score", 0.0)

    # Baseline: random controls ranked by same model (or placeholder)
    try:
        df_edges = load_hetionet_edges()
        subgraph = build_mechanism_subgraph(df_edges, hypothesis_id)
        all_entities = list(set(df_edges["source"].tolist() + df_edges["target"].tolist()))
        controls = get_random_controls(
            all_entities, subgraph, entity_type="Compound", n_controls=min(50, top_k * 2)
        )
    except Exception as e:
        logger.warning(f"Could not generate controls: {e}")
        controls = []

    baseline_scores = []
    if controls and mech.get("model_used") not in ("error", "fallback_unavailable"):
        try:
            from middleware.orchestrator import LinkPredictionOrchestrator
            orch = LinkPredictionOrchestrator(use_quantum=False)
            for comp_id in controls[:top_k]:
                res = orch.predict_link_probability(comp_id, disease_id, method="classical")
                if res["status"] == "success":
                    baseline_scores.append(res["link_probability"])
        except Exception:
            pass

    baseline_mean = np.mean(baseline_scores) if baseline_scores else 0.0
    rank_diff = mechanism_mean - baseline_mean
    better_than_baseline = rank_diff > 0

    tracker.log_control_ranking(mechanism_scores, baseline_scores)
    tracker.log_metrics({
        "mechanism_mean_score": mechanism_mean,
        "baseline_mean_score": baseline_mean,
        "ablation_rank_diff": rank_diff,
        "better_than_baseline": better_than_baseline,
    })
    tracker.save_run()

    # Generate ablation report
    baseline_metrics = {"mean_score": baseline_mean, "n_candidates": len(baseline_scores)}
    mechanism_metrics = {"mean_score": mechanism_mean, "n_candidates": len(mechanism_scores)}
    report = generate_ablation_report(baseline_metrics, mechanism_metrics)
    logger.info(report)
    logger.info(f"Run saved to results/run_{tracker.run_id}.json")


def main():
    parser = argparse.ArgumentParser(description="Run hypothesis ranking and ablation")
    parser.add_argument("--hypothesis", "-H", default="H-001", help="Hypothesis ID (H-001, H-002, H-003)")
    parser.add_argument("--disease", "-d", default="DOID:9352", help="Disease ID (e.g., DOID:9352)")
    parser.add_argument("--top_k", "-k", type=int, default=50, help="Number of top candidates")
    args = parser.parse_args()

    run_ablation(
        hypothesis_id=args.hypothesis,
        disease_id=args.disease,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
