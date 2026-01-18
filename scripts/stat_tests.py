#!/usr/bin/env python3
"""Statistical significance tests from saved results"""

import sys
import os
# Add project root to path (for potential future imports)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
from scipy.stats import wilcoxon, ttest_rel, bootstrap
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Statistical significance tests")
    parser.add_argument("--results_file", type=str, required=True,
                       help="Path to JSON file with results (e.g., multi_seed_summary)")
    parser.add_argument("--metric", type=str, default="pr_auc",
                       choices=["pr_auc", "accuracy", "f1"],
                       help="Metric to test")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    
    # Load results
    with open(args.results_file, "r") as f:
        data = json.load(f)
    
    # Extract metrics
    if 'all_results' in data:
        # Multi-seed format
        classical_scores = [
            r['classical'].get(f'test_{args.metric}', r['classical'].get(args.metric, 0.0))
            for r in data['all_results']
        ]
        quantum_scores = [
            r['quantum'].get(args.metric, 0.0)
            for r in data['all_results']
        ]
    elif 'summary' in data:
        # Summary format
        classical_scores = data['summary']['classical'][args.metric]['values']
        quantum_scores = data['summary']['quantum'][args.metric]['values']
    else:
        raise ValueError("Unknown results format")
    
    classical_scores = np.array(classical_scores)
    quantum_scores = np.array(quantum_scores)
    
    if len(classical_scores) != len(quantum_scores):
        print("⚠️ Warning: Unequal sample sizes")
        return
    
    print(f"\n{'='*60}")
    print("STATISTICAL SIGNIFICANCE TESTS")
    print(f"{'='*60}\n")
    
    print(f"Metric: {args.metric}")
    print(f"Sample size: {len(classical_scores)}")
    print(f"\nClassical: {np.mean(classical_scores):.4f} ± {np.std(classical_scores):.4f}")
    print(f"Quantum:   {np.mean(quantum_scores):.4f} ± {np.std(quantum_scores):.4f}")
    
    results = {}
    
    # Paired t-test
    try:
        t_stat, t_pval = ttest_rel(quantum_scores, classical_scores)
        results['paired_t_test'] = {
            'statistic': float(t_stat),
            'p_value': float(t_pval),
            'significant': t_pval < 0.05
        }
        print(f"\nPaired t-test:")
        print(f"  t = {t_stat:.4f}, p = {t_pval:.4f}")
        print(f"  {'✅ Significant' if t_pval < 0.05 else '❌ Not significant'} (α=0.05)")
    except Exception as e:
        print(f"T-test failed: {e}")
    
    # Wilcoxon signed-rank test
    try:
        w_stat, w_pval = wilcoxon(quantum_scores, classical_scores)
        results['wilcoxon'] = {
            'statistic': float(w_stat),
            'p_value': float(w_pval),
            'significant': w_pval < 0.05
        }
        print(f"\nWilcoxon signed-rank test:")
        print(f"  W = {w_stat:.4f}, p = {w_pval:.4f}")
        print(f"  {'✅ Significant' if w_pval < 0.05 else '❌ Not significant'} (α=0.05)")
    except Exception as e:
        print(f"Wilcoxon test failed: {e}")
    
    # Effect size (Cohen's d)
    try:
        pooled_std = np.sqrt((np.var(classical_scores) + np.var(quantum_scores)) / 2)
        cohens_d = (np.mean(quantum_scores) - np.mean(classical_scores)) / (pooled_std + 1e-9)
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect_size = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_size = "small"
        elif abs(cohens_d) < 0.8:
            effect_size = "medium"
        else:
            effect_size = "large"
        
        results['cohens_d'] = {
            'value': float(cohens_d),
            'interpretation': effect_size
        }
        print(f"\nEffect size (Cohen's d):")
        print(f"  d = {cohens_d:.4f} ({effect_size})")
    except Exception as e:
        print(f"Effect size calculation failed: {e}")
    
    # Bootstrap confidence intervals
    try:
        def statistic(x):
            return np.mean(x)
        
        classical_ci = bootstrap(
            (classical_scores,), statistic, n_resamples=1000, random_state=42
        ).confidence_interval
        quantum_ci = bootstrap(
            (quantum_scores,), statistic, n_resamples=1000, random_state=42
        ).confidence_interval
        
        results['bootstrap_ci'] = {
            'classical': {
                'low': float(classical_ci.low),
                'high': float(classical_ci.high)
            },
            'quantum': {
                'low': float(quantum_ci.low),
                'high': float(quantum_ci.high)
            }
        }
        print(f"\nBootstrap 95% CI:")
        print(f"  Classical: [{classical_ci.low:.4f}, {classical_ci.high:.4f}]")
        print(f"  Quantum:   [{quantum_ci.low:.4f}, {quantum_ci.high:.4f}]")
    except Exception as e:
        print(f"Bootstrap CI failed: {e}")
    
    # Save results
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(args.results_dir, f"stat_tests_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump({
            'source_file': args.results_file,
            'metric': args.metric,
            'results': results,
            'timestamp': stamp
        }, f, indent=2)
    print(f"\n✅ Saved results → {json_path}")

if __name__ == "__main__":
    main()

