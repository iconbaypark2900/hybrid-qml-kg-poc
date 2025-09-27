# classical_baseline/evaluate_baseline.py

"""
Comprehensive evaluation utilities for classical baseline models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score, roc_auc_score
)
import logging

logger = logging.getLogger(__name__)

def plot_confusion_matrix(y_true, y_pred, classes=['Negative', 'Positive'], title="Confusion Matrix"):
    """
    Plot confusion matrix with proper labeling.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    return plt.gcf()

def plot_roc_curve(y_true, y_proba, title="ROC Curve"):
    """
    Plot ROC curve with AUC score.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def plot_precision_recall_curve(y_true, y_proba, title="Precision-Recall Curve"):
    """
    Plot Precision-Recall curve with AP score.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap_score = average_precision_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkgreen', lw=2,
             label=f'PR curve (AP = {ap_score:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def calculate_comprehensive_metrics(y_true, y_pred, y_proba):
    """
    Calculate comprehensive set of evaluation metrics.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        matthews_corrcoef, balanced_accuracy_score
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'average_precision': average_precision_score(y_true, y_proba)
    }
    
    return metrics

def compare_models(results_dict, metric='average_precision'):
    """
    Compare multiple models on a specific metric.
    
    Args:
        results_dict: Dict with model names as keys and results as values
        metric: Metric to compare (e.g., 'average_precision', 'roc_auc', 'f1_score')
    """
    models = list(results_dict.keys())
    scores = [results_dict[model][metric] for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, scores, color=['#2E86AB', '#A23B72', '#F24236', '#4ECDC4'])
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'Model Comparison: {metric.replace("_", " ").title()}')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{score:.4f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return plt.gcf()

def generate_evaluation_report(y_true, y_pred, y_proba, model_name="Model"):
    """
    Generate a comprehensive evaluation report.
    """
    metrics = calculate_comprehensive_metrics(y_true, y_pred, y_proba)
    
    report = f"""
    {'='*60}
    EVALUATION REPORT: {model_name}
    {'='*60}
    
    Accuracy:           {metrics['accuracy']:.4f}
    Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}
    Precision:          {metrics['precision']:.4f}
    Recall:             {metrics['recall']:.4f}
    F1-Score:           {metrics['f1_score']:.4f}
    Matthews Corr:      {metrics['matthews_corrcoef']:.4f}
    ROC-AUC:            {metrics['roc_auc']:.4f}
    Average Precision:  {metrics['average_precision']:.4f}
    
    {'='*60}
    """
    
    logger.info(report)
    return metrics, report

def save_evaluation_results(metrics_dict, filename="evaluation_results.csv"):
    """
    Save evaluation results to CSV file.
    """
    df = pd.DataFrame([metrics_dict])
    df.to_csv(filename, index=False)
    logger.info(f"Saved evaluation results to {filename}")

# Example usage function
def evaluate_classical_model(model, X_test, y_test, model_name="Classical Model"):
    """
    Complete evaluation pipeline for a classical model.
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Generate report
    metrics, report = generate_evaluation_report(y_test, y_pred, y_proba, model_name)
    
    # Create plots
    fig1 = plot_confusion_matrix(y_test, y_pred, title=f"{model_name} - Confusion Matrix")
    fig2 = plot_roc_curve(y_test, y_proba, title=f"{model_name} - ROC Curve")
    fig3 = plot_precision_recall_curve(y_test, y_proba, title=f"{model_name} - PR Curve")
    
    # Save results
    save_evaluation_results(metrics, f"{model_name.lower().replace(' ', '_')}_evaluation.csv")
    
    return metrics, [fig1, fig2, fig3]