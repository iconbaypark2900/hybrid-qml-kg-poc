"""Unit tests for top_k_hit_rate and mean_rank_of_positives."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scripts.run_optimized_pipeline import top_k_hit_rate, mean_rank_of_positives


def test_top_k_perfect_ranking():
    y_true = np.array([1, 1, 0, 0, 0])
    y_scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    assert top_k_hit_rate(y_true, y_scores, k=2) == 1.0


def test_top_k_worst_ranking():
    y_true = np.array([0, 0, 0, 1, 1])
    y_scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    assert top_k_hit_rate(y_true, y_scores, k=3) == 0.0


def test_top_k_partial():
    y_true = np.array([1, 0, 1, 0, 0])
    y_scores = np.array([0.9, 0.85, 0.3, 0.2, 0.1])
    hit = top_k_hit_rate(y_true, y_scores, k=2)
    assert hit == 0.5  # 1 of 2 positives in top-2


def test_mean_rank_perfect():
    y_true = np.array([1, 1, 0, 0, 0])
    y_scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    mr = mean_rank_of_positives(y_true, y_scores)
    assert mr == 1.5  # ranks 1 and 2


def test_mean_rank_worst():
    y_true = np.array([0, 0, 0, 1, 1])
    y_scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    mr = mean_rank_of_positives(y_true, y_scores)
    assert mr == 4.5  # ranks 4 and 5


def test_mean_rank_no_positives():
    y_true = np.array([0, 0, 0])
    y_scores = np.array([0.9, 0.5, 0.1])
    mr = mean_rank_of_positives(y_true, y_scores)
    assert np.isnan(mr)


def test_compute_metrics_includes_discovery():
    from scripts.run_optimized_pipeline import compute_metrics
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])
    y_score = np.array([0.9, 0.4, 0.6, 0.2])
    m = compute_metrics(y_true, y_pred, y_score)
    assert "top_10_hit_rate" in m
    assert "mean_rank" in m
    assert isinstance(m["top_10_hit_rate"], float)
    assert isinstance(m["mean_rank"], float)


if __name__ == "__main__":
    test_top_k_perfect_ranking()
    test_top_k_worst_ranking()
    test_top_k_partial()
    test_mean_rank_perfect()
    test_mean_rank_worst()
    test_mean_rank_no_positives()
    test_compute_metrics_includes_discovery()
    print("All discovery metric tests passed.")
