#!/usr/bin/env python3
"""
Comprehensive analysis of pipeline results.
Dissects scores, metrics, and predictions to understand model performance.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from sklearn.metrics import confusion_matrix, classification_report

def load_results(results_file):
    """Load results JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)

def analyze_metrics(results_data):
    """Analyze and print detailed metrics."""
    print("="*80)
    print("DETAILED METRICS ANALYSIS")
    print("="*80)
    
    # Classical results
    print("\n📊 CLASSICAL MODELS:")
    print("-"*80)
    for model_name, model_data in results_data.get('classical_results', {}).items():
        print(f"\n{model_name}:")
        if model_data.get('status') == 'success':
            train_metrics = model_data.get('train_metrics', {})
            test_metrics = model_data.get('test_metrics', {})
            
            print(f"  Train Metrics:")
            for metric, value in train_metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    print(f"    {metric:20s}: {value:.4f}")
            
            print(f"  Test Metrics:")
            for metric, value in test_metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    print(f"    {metric:20s}: {value:.4f}")
            
            print(f"  Training Time: {model_data.get('fit_seconds', 0):.2f}s")
        else:
            print(f"  Status: {model_data.get('status')}")
    
    # Quantum results
    print("\n⚛️  QUANTUM MODELS:")
    print("-"*80)
    for model_name, model_data in results_data.get('quantum_results', {}).items():
        print(f"\n{model_name}:")
        if model_data.get('status') == 'success':
            train_metrics = model_data.get('train_metrics', {})
            test_metrics = model_data.get('test_metrics', {})
            
            print(f"  Train Metrics:")
            for metric, value in train_metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    print(f"    {metric:20s}: {value:.4f}")
            
            print(f"  Test Metrics:")
            for metric, value in test_metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    print(f"    {metric:20s}: {value:.4f}")
            
            print(f"  Training Time: {model_data.get('fit_seconds', 0):.2f}s")
        else:
            print(f"  Status: {model_data.get('status')}")

def analyze_predictions(predictions_file):
    """Analyze prediction distributions and patterns."""
    print("\n" + "="*80)
    print("PREDICTION ANALYSIS")
    print("="*80)
    
    df = pd.read_csv(predictions_file)
    
    # Split by train/test
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    
    print(f"\n📈 Dataset Sizes:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    for split_name, split_df in [("Train", train_df), ("Test", test_df)]:
        print(f"\n{split_name} Set Analysis:")
        print("-"*80)
        
        # Class distribution
        print(f"  Class Distribution (y_true):")
        class_counts = split_df['y_true'].value_counts().sort_index()
        for label, count in class_counts.items():
            pct = 100 * count / len(split_df)
            print(f"    Class {label}: {count:4d} ({pct:5.2f}%)")
        
        # Prediction distribution
        print(f"  Prediction Distribution (y_pred):")
        pred_counts = split_df['y_pred'].value_counts().sort_index()
        for label, count in pred_counts.items():
            pct = 100 * count / len(split_df)
            print(f"    Class {label}: {count:4d} ({pct:5.2f}%)")
        
        # Score distribution
        print(f"  Score Distribution (y_score):")
        scores = split_df['y_score']
        print(f"    Mean:   {scores.mean():.4f}")
        print(f"    Std:    {scores.std():.4f}")
        print(f"    Min:    {scores.min():.4f}")
        print(f"    Max:    {scores.max():.4f}")
        print(f"    Median: {scores.median():.4f}")
        
        # Score distribution by class
        print(f"  Score Distribution by True Class:")
        for label in sorted(split_df['y_true'].unique()):
            class_scores = split_df[split_df['y_true'] == label]['y_score']
            print(f"    Class {label}: mean={class_scores.mean():.4f}, std={class_scores.std():.4f}, "
                  f"min={class_scores.min():.4f}, max={class_scores.max():.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(split_df['y_true'], split_df['y_pred'])
        print(f"\n  Confusion Matrix:")
        print(f"    {'':>10} {'Pred 0':>10} {'Pred 1':>10}")
        print(f"    {'True 0':>10} {cm[0,0]:>10} {cm[0,1]:>10}")
        print(f"    {'True 1':>10} {cm[1,0]:>10} {cm[1,1]:>10}")
        
        # Classification report
        print(f"\n  Classification Report:")
        report = classification_report(split_df['y_true'], split_df['y_pred'], 
                                     target_names=['Class 0', 'Class 1'],
                                     output_dict=True)
        for class_name in ['Class 0', 'Class 1', 'macro avg', 'weighted avg']:
            if class_name in report:
                metrics = report[class_name]
                print(f"    {class_name:15s}: precision={metrics['precision']:.4f}, "
                      f"recall={metrics['recall']:.4f}, f1={metrics['f1-score']:.4f}")
        
        # Error analysis
        errors = split_df[split_df['y_true'] != split_df['y_pred']]
        print(f"\n  Error Analysis:")
        print(f"    Total Errors: {len(errors)} ({100*len(errors)/len(split_df):.2f}%)")
        if len(errors) > 0:
            print(f"    False Positives (pred=1, true=0): {len(errors[(errors['y_pred']==1) & (errors['y_true']==0)])}")
            print(f"    False Negatives (pred=0, true=1): {len(errors[(errors['y_pred']==0) & (errors['y_true']==1)])}")
            
            # Score distribution for errors
            error_scores = errors['y_score']
            print(f"    Error Score Stats:")
            print(f"      Mean: {error_scores.mean():.4f}, Std: {error_scores.std():.4f}")
            print(f"      Min: {error_scores.min():.4f}, Max: {error_scores.max():.4f}")

def compare_models(results_data):
    """Compare models side by side."""
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    all_models = []
    
    # Collect classical models
    for model_name, model_data in results_data.get('classical_results', {}).items():
        if model_data.get('status') == 'success':
            test_metrics = model_data.get('test_metrics', {})
            all_models.append({
                'name': model_name,
                'type': 'classical',
                'pr_auc': test_metrics.get('pr_auc', np.nan),
                'roc_auc': test_metrics.get('roc_auc', np.nan),
                'accuracy': test_metrics.get('accuracy', np.nan),
                'precision': test_metrics.get('precision', np.nan),
                'recall': test_metrics.get('recall', np.nan),
                'f1': test_metrics.get('f1', np.nan),
                'time': model_data.get('fit_seconds', np.nan)
            })
    
    # Collect quantum models
    for model_name, model_data in results_data.get('quantum_results', {}).items():
        if model_data.get('status') == 'success':
            test_metrics = model_data.get('test_metrics', {})
            all_models.append({
                'name': model_name,
                'type': 'quantum',
                'pr_auc': test_metrics.get('pr_auc', np.nan),
                'roc_auc': test_metrics.get('roc_auc', np.nan),
                'accuracy': test_metrics.get('accuracy', np.nan),
                'precision': test_metrics.get('precision', np.nan),
                'recall': test_metrics.get('recall', np.nan),
                'f1': test_metrics.get('f1', np.nan),
                'time': model_data.get('fit_seconds', np.nan)
            })
    
    if not all_models:
        print("No successful models found.")
        return
    
    df = pd.DataFrame(all_models)
    
    # Sort by PR-AUC
    df = df.sort_values('pr_auc', ascending=False)
    
    print("\n📊 Test Set Performance (sorted by PR-AUC):")
    print("-"*80)
    print(f"{'Model':<25} {'Type':<10} {'PR-AUC':>8} {'ROC-AUC':>8} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Time(s)':>8}")
    print("-"*80)
    for _, row in df.iterrows():
        print(f"{row['name']:<25} {row['type']:<10} "
              f"{row['pr_auc']:>8.4f} {row['roc_auc']:>8.4f} "
              f"{row['accuracy']:>6.4f} {row['precision']:>6.4f} "
              f"{row['recall']:>6.4f} {row['f1']:>6.4f} {row['time']:>8.2f}")
    
    # Summary statistics
    print("\n📈 Summary Statistics:")
    print("-"*80)
    print(f"  Best PR-AUC: {df['pr_auc'].max():.4f} ({df.loc[df['pr_auc'].idxmax(), 'name']})")
    print(f"  Worst PR-AUC: {df['pr_auc'].min():.4f} ({df.loc[df['pr_auc'].idxmin(), 'name']})")
    print(f"  Mean PR-AUC: {df['pr_auc'].mean():.4f}")
    print(f"  Std PR-AUC: {df['pr_auc'].std():.4f}")
    
    # Classical vs Quantum
    classical_df = df[df['type'] == 'classical']
    quantum_df = df[df['type'] == 'quantum']
    
    if len(classical_df) > 0 and len(quantum_df) > 0:
        print(f"\n  Classical Models:")
        print(f"    Mean PR-AUC: {classical_df['pr_auc'].mean():.4f}")
        print(f"    Mean Time: {classical_df['time'].mean():.2f}s")
        
        print(f"\n  Quantum Models:")
        print(f"    Mean PR-AUC: {quantum_df['pr_auc'].mean():.4f}")
        print(f"    Mean Time: {quantum_df['time'].mean():.2f}s")
        
        print(f"\n  Performance Gap:")
        gap = classical_df['pr_auc'].mean() - quantum_df['pr_auc'].mean()
        print(f"    PR-AUC Difference: {gap:+.4f} ({'Classical' if gap > 0 else 'Quantum'} leads)")

def analyze_config(results_data):
    """Analyze configuration settings."""
    print("\n" + "="*80)
    print("CONFIGURATION ANALYSIS")
    print("="*80)
    
    config = results_data.get('config', {})
    
    print("\n🔧 Key Configuration:")
    print("-"*80)
    important_keys = [
        'relation', 'embedding_method', 'embedding_dim', 'embedding_epochs',
        'full_graph_embeddings', 'qml_encoding', 'qml_dim',
        'use_graph_features', 'use_domain_features', 'use_feature_selection',
        'negative_sampling', 'random_state'
    ]
    
    for key in important_keys:
        if key in config:
            value = config[key]
            print(f"  {key:25s}: {value}")

def main():
    parser = argparse.ArgumentParser(description="Analyze pipeline results")
    parser.add_argument("--results", type=str, required=True,
                       help="Path to results JSON file")
    parser.add_argument("--predictions", type=str, default=None,
                       help="Path to predictions CSV file (optional)")
    
    args = parser.parse_args()
    
    # Load results
    print("Loading results...")
    results_data = load_results(args.results)
    
    # Analyze metrics
    analyze_metrics(results_data)
    
    # Analyze predictions if provided
    if args.predictions and Path(args.predictions).exists():
        analyze_predictions(args.predictions)
    else:
        # Try to find predictions file automatically
        results_dir = Path(args.results).parent
        timestamp = Path(args.results).stem.split('_')[-1]
        predictions_file = results_dir / f"predictions_QSVC_{timestamp}.csv"
        if predictions_file.exists():
            print(f"\nFound predictions file: {predictions_file}")
            analyze_predictions(str(predictions_file))
        else:
            print("\n⚠️  No predictions file found. Skipping prediction analysis.")
    
    # Compare models
    compare_models(results_data)
    
    # Analyze config
    analyze_config(results_data)
    
    print("\n" + "="*80)
    print("✅ Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()

