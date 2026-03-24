#!/usr/bin/env python3
"""
Demonstrate Multi-Model Fusion

This script demonstrates the multi-model fusion capabilities
implemented in quantum_layer/multi_model_fusion.py

Usage:
    python scripts/test_multi_model_fusion.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from quantum_layer.multi_model_fusion import (
    MultiModelFusion,
    create_fusion_ensemble
)

def demo_fusion_methods():
    """Demonstrate different fusion methods on synthetic data."""
    
    print("="*70)
    print("MULTI-MODEL FUSION DEMONSTRATION")
    print("="*70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic ground truth
    y_true = np.random.randint(0, 2, n_samples)
    
    # Generate synthetic predictions from 4 models with different performance
    # Model 1: Good performance (PR-AUC ~0.75)
    pred_model1 = np.clip(
        np.random.normal(loc=y_true * 0.6 + 0.2, scale=0.2),
        0, 1
    )
    
    # Model 2: Moderate performance (PR-AUC ~0.65)
    pred_model2 = np.clip(
        np.random.normal(loc=y_true * 0.4 + 0.3, scale=0.25),
        0, 1
    )
    
    # Model 3: Decent performance (PR-AUC ~0.70)
    pred_model3 = np.clip(
        np.random.normal(loc=y_true * 0.5 + 0.25, scale=0.22),
        0, 1
    )
    
    # Model 4: Lower performance (PR-AUC ~0.60)
    pred_model4 = np.clip(
        np.random.normal(loc=y_true * 0.3 + 0.35, scale=0.28),
        0, 1
    )
    
    # Create predictions dict
    predictions = {
        'quantum_model': pred_model1,
        'random_forest': pred_model2,
        'extra_trees': pred_model3,
        'gradient_boosting': pred_model4
    }
    
    # Evaluate individual models
    print("\n1. INDIVIDUAL MODEL PERFORMANCE")
    print("-" * 70)
    individual_metrics = {}
    for name, pred in predictions.items():
        pr_auc = average_precision_score(y_true, pred)
        try:
            roc_auc = roc_auc_score(y_true, pred)
        except Exception:
            roc_auc = float('nan')
        individual_metrics[name] = {'pr_auc': pr_auc, 'roc_auc': roc_auc}
        print(f"  {name:20s}: PR-AUC = {pr_auc:.4f}, ROC-AUC = {roc_auc:.4f}")
    
    # Test different fusion methods
    fusion_methods = [
        'weighted_average',
        'optimized_weights',
        'bayesian_averaging',
        'rank_fusion',
        'confidence_weighted'
    ]
    
    print("\n2. FUSION METHOD COMPARISON")
    print("-" * 70)
    
    fusion_results = {}
    
    for method in fusion_methods:
        try:
            # Create and fit fusion system
            fusion, metrics = create_fusion_ensemble(
                predictions,
                y_true,
                fusion_method=method,
                random_state=42
            )
            
            # Store results
            fusion_results[method] = metrics
            
            # Print results
            print(f"\n  {method.upper():25s}")
            print(f"    Fused PR-AUC:  {metrics['fused_pr_auc']:.4f}")
            print(f"    Fused ROC-AUC: {metrics['fused_roc_auc']:.4f}")
            print(f"    Improvement:   {metrics['improvement_over_mean']:+.4f} over mean")
            
            # Print weights if available
            if hasattr(fusion, 'weights') and fusion.weights:
                print(f"    Weights: {fusion.weights}")
            elif hasattr(fusion, 'optimized_weights') and fusion.optimized_weights:
                print(f"    Optimized Weights: {fusion.optimized_weights}")
            elif hasattr(fusion, 'bma_weights') and fusion.bma_weights:
                print(f"    BMA Weights: {fusion.bma_weights}")
                
        except Exception as e:
            print(f"\n  {method.upper():25s}: FAILED - {str(e)}")
    
    # Summary table
    print("\n3. SUMMARY TABLE")
    print("-" * 70)
    print(f"  {'Method':25s} | {'PR-AUC':>10s} | {'ROC-AUC':>10s} | {'Improvement':>12s}")
    print(f"  {'-'*25}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")
    
    mean_individual = np.mean([m['pr_auc'] for m in individual_metrics.values()])
    
    for method, metrics in fusion_results.items():
        print(f"  {method:25s} | {metrics['fused_pr_auc']:10.4f} | {metrics['fused_roc_auc']:10.4f} | {metrics['improvement_over_mean']:>+12.4f}")
    
    print(f"\n  Mean Individual: {mean_individual:.4f}")
    
    # Find best method
    best_method = max(fusion_results.keys(), key=lambda k: fusion_results[k]['fused_pr_auc'])
    best_fused = fusion_results[best_method]['fused_pr_auc']
    
    print(f"\n4. RECOMMENDATION")
    print("-" * 70)
    print(f"  Best fusion method: {best_method}")
    print(f"  Best fused PR-AUC:  {best_fused:.4f}")
    print(f"  Improvement over best individual: {best_fused - max(m['pr_auc'] for m in individual_metrics.values()):+.4f}")
    print(f"  Improvement over mean: {best_fused - mean_individual:+.4f}")
    
    # Usage example
    print("\n5. USAGE EXAMPLE")
    print("-" * 70)
    print("""
    from quantum_layer.multi_model_fusion import create_fusion_ensemble
    
    # Prepare predictions from your models
    model_predictions = {
        'quantum': quantum_pred_proba,
        'random_forest': rf_pred_proba,
        'extra_trees': et_pred_proba,
        'gradient_boosting': gb_pred_proba
    }
    
    # Create fusion ensemble
    fusion, metrics = create_fusion_ensemble(
        model_predictions,
        y_train,
        fusion_method='optimized_weights'  # or 'bayesian_averaging'
    )
    
    # Get fused predictions for test set
    fused_pred = fusion.predict(test_predictions)
    
    # Evaluate
    from sklearn.metrics import average_precision_score
    pr_auc = average_precision_score(y_test, fused_pred)
    """)
    
    print("="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    
    return fusion_results


def demo_neural_metalearner():
    """Demonstrate neural network meta-learner fusion."""
    
    print("\n" + "="*70)
    print("NEURAL META-LEARNER DEMONSTRATION")
    print("="*70)
    
    np.random.seed(123)
    n_samples = 2000
    
    # Generate synthetic data
    y_true = np.random.randint(0, 2, n_samples)
    
    # Create correlated predictions (simulate model ensemble)
    pred1 = np.clip(y_true * 0.5 + np.random.normal(0, 0.3, n_samples) + 0.25, 0, 1)
    pred2 = np.clip(y_true * 0.45 + np.random.normal(0, 0.32, n_samples) + 0.27, 0, 1)
    pred3 = np.clip(y_true * 0.48 + np.random.normal(0, 0.28, n_samples) + 0.26, 0, 1)
    
    predictions = {
        'model1': pred1,
        'model2': pred2,
        'model3': pred3
    }
    
    print("\nTraining neural meta-learner...")
    
    fusion, metrics = create_fusion_ensemble(
        predictions,
        y_true,
        fusion_method='neural_metalearner',
        cv_folds=3,
        random_state=123
    )
    
    print(f"\nResults:")
    print(f"  Fused PR-AUC:  {metrics['fused_pr_auc']:.4f}")
    print(f"  Fused ROC-AUC: {metrics['fused_roc_auc']:.4f}")
    print(f"  Improvement:   {metrics['improvement_over_mean']:+.4f} over mean")
    
    print("\nNeural meta-learner architecture:")
    print(f"  Hidden layers: 64 -> 32 -> 16")
    print(f"  Activation: ReLU")
    print(f"  Solver: Adam")
    print(f"  Max iterations: 500")
    
    print("="*70)
    
    return metrics


if __name__ == "__main__":
    # Run demonstrations
    fusion_results = demo_fusion_methods()
    neural_results = demo_neural_metalearner()
    
    print("\n" + "="*70)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Review results above")
    print("2. Select best fusion method for your use case")
    print("3. Integrate into your pipeline using the usage example")
    print("4. See quantum_layer/multi_model_fusion.py for advanced options")
    print("="*70 + "\n")
