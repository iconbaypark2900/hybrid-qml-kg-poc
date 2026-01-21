"""
Evaluation utilities for robust model assessment.

This module provides K-Fold Cross-Validation utilities for more reliable
performance evaluation compared to single train/test splits.
"""

import logging
from typing import Dict, List, Tuple, Callable, Optional, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

logger = logging.getLogger(__name__)


def stratified_kfold_cv(
    task_edges: pd.DataFrame,
    entity_to_id: Dict[str, int],
    n_folds: int = 5,
    random_state: int = 42
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create stratified K-Fold splits for link prediction.

    Each fold maintains class balance (positive/negative ratio) and generates
    negative samples independently to avoid data leakage.

    Args:
        task_edges: DataFrame with positive edges (source_id, target_id)
        entity_to_id: Entity to ID mapping
        n_folds: Number of CV folds
        random_state: Random seed

    Returns:
        List of (train_df, test_df) tuples, one per fold
    """
    from kg_layer.kg_loader import get_negative_samples

    logger.info(f"Creating {n_folds}-fold stratified CV splits...")

    # Positive samples
    pos_df = task_edges[["source_id", "target_id"]].copy()
    pos_df["label"] = 1

    # Stratified K-Fold on positive edges
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Create dummy stratification variable (all same class for positive edges)
    # We'll manually balance with negatives
    y_dummy = np.ones(len(pos_df))

    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(pos_df, y_dummy)):
        # Split positive edges
        pos_train = pos_df.iloc[train_idx].copy()
        pos_test = pos_df.iloc[test_idx].copy()

        # Generate negative samples for this fold (independent per fold)
        # This prevents leakage - negatives are sampled fresh each time
        neg_train = get_negative_samples(
            pd.DataFrame(pos_train),
            num_negatives=len(pos_train),
            random_state=random_state + fold_idx * 1000  # Different seed per fold
        )
        neg_test = get_negative_samples(
            pd.DataFrame(pos_test),
            num_negatives=len(pos_test),
            random_state=random_state + fold_idx * 1000 + 1
        )

        # Combine and shuffle
        train_df = pd.concat([pos_train, neg_train], ignore_index=True).sample(
            frac=1, random_state=random_state + fold_idx
        )
        test_df = pd.concat([pos_test, neg_test], ignore_index=True).sample(
            frac=1, random_state=random_state + fold_idx
        )

        logger.info(f"Fold {fold_idx + 1}/{n_folds}: "
                   f"Train={len(train_df)} ({train_df['label'].sum()} pos), "
                   f"Test={len(test_df)} ({test_df['label'].sum()} pos)")

        folds.append((train_df, test_df))

    return folds


def evaluate_model_cv(
    model_fn: Callable,
    folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
    embeddings: Dict[str, np.ndarray],
    model_name: str = "Model",
    id_to_entity: Optional[Dict[int, str]] = None,  # ADD THIS
    **model_kwargs
) -> Dict[str, Any]:
    """
    Evaluate a model using K-Fold Cross-Validation.

    Args:
        model_fn: Function that trains and returns a model
                 Signature: model_fn(X_train, y_train, **kwargs) -> model
        folds: List of (train_df, test_df) tuples from stratified_kfold_cv
        embeddings: Dict mapping entity_id -> embedding vector
        model_name: Name for logging
        **model_kwargs: Additional arguments passed to model_fn

    Returns:
        Dict with aggregated metrics:
        {
            'pr_aucs': List[float],
            'roc_aucs': List[float],
            'accuracies': List[float],
            'f1_scores': List[float],
            'mean_pr_auc': float,
            'std_pr_auc': float,
            'mean_roc_auc': float,
            'std_roc_auc': float,
            ...
        }
    """
    logger.info(f"Evaluating {model_name} with {len(folds)}-fold CV...")

    pr_aucs = []
    roc_aucs = []
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []

    for fold_idx, (train_df, test_df) in enumerate(folds):
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold_idx + 1}/{len(folds)}")
        logger.info(f"{'='*60}")

        # Prepare features
        embedding_dim = next(iter(embeddings.values())).shape[0]
        
        # FIX: Convert integer IDs to entity strings for embedding lookup
        if id_to_entity is not None:
            # Convert source_id and target_id to entity strings
            train_source_entities = train_df['source_id'].map(id_to_entity)
            train_target_entities = train_df['target_id'].map(id_to_entity)
            test_source_entities = test_df['source_id'].map(id_to_entity)
            test_target_entities = test_df['target_id'].map(id_to_entity)
        else:
            # Fallback: assume IDs are already strings
            train_source_entities = train_df['source_id'].astype(str)
            train_target_entities = train_df['target_id'].astype(str)
            test_source_entities = test_df['source_id'].astype(str)
            test_target_entities = test_df['target_id'].astype(str)
        
        # Training data - use entity strings as keys
        train_h_embs = np.array([
            embeddings.get(str(h), np.zeros(embedding_dim))
            for h in train_source_entities.values
        ])
        train_t_embs = np.array([
            embeddings.get(str(t), np.zeros(embedding_dim))
            for t in train_target_entities.values
        ])
        X_train = np.concatenate([train_h_embs, train_t_embs], axis=1)
        y_train = train_df['label'].values
        
        # Test data - use entity strings as keys
        test_h_embs = np.array([
            embeddings.get(str(h), np.zeros(embedding_dim))
            for h in test_source_entities.values
        ])
        test_t_embs = np.array([
            embeddings.get(str(t), np.zeros(embedding_dim))
            for t in test_target_entities.values
        ])
        X_test = np.concatenate([test_h_embs, test_t_embs], axis=1)
        y_test = test_df['label'].values
        
        # Train model
        try:
            model = model_fn(X_train, y_train, **model_kwargs)

            # Predict
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_pred_proba = model.decision_function(X_test)
            else:
                raise ValueError(f"{model_name} doesn't have predict_proba or decision_function")

            y_pred = (y_pred_proba > 0.5).astype(int)

            # Calculate metrics
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall, precision)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)

            pr_aucs.append(pr_auc)
            roc_aucs.append(roc_auc)
            accuracies.append(accuracy)
            f1_scores.append(f1)
            precisions.append(prec)
            recalls.append(rec)

            logger.info(f"Fold {fold_idx + 1} - PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}, "
                       f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        except Exception as e:
            logger.error(f"Error in fold {fold_idx + 1}: {e}")
            # Append NaN for failed folds
            pr_aucs.append(np.nan)
            roc_aucs.append(np.nan)
            accuracies.append(np.nan)
            f1_scores.append(np.nan)
            precisions.append(np.nan)
            recalls.append(np.nan)

    # Filter out NaN values for aggregation
    valid_pr_aucs = [x for x in pr_aucs if not np.isnan(x)]
    valid_roc_aucs = [x for x in roc_aucs if not np.isnan(x)]
    valid_accuracies = [x for x in accuracies if not np.isnan(x)]
    valid_f1_scores = [x for x in f1_scores if not np.isnan(x)]

    results = {
        'pr_aucs': pr_aucs,
        'roc_aucs': roc_aucs,
        'accuracies': accuracies,
        'f1_scores': f1_scores,
        'precisions': precisions,
        'recalls': recalls,
        'mean_pr_auc': np.mean(valid_pr_aucs) if valid_pr_aucs else np.nan,
        'std_pr_auc': np.std(valid_pr_aucs) if valid_pr_aucs else np.nan,
        'mean_roc_auc': np.mean(valid_roc_aucs) if valid_roc_aucs else np.nan,
        'std_roc_auc': np.std(valid_roc_aucs) if valid_roc_aucs else np.nan,
        'mean_accuracy': np.mean(valid_accuracies) if valid_accuracies else np.nan,
        'std_accuracy': np.std(valid_accuracies) if valid_accuracies else np.nan,
        'mean_f1': np.mean(valid_f1_scores) if valid_f1_scores else np.nan,
        'std_f1': np.std(valid_f1_scores) if valid_f1_scores else np.nan,
        'n_successful_folds': len(valid_pr_aucs)
    }

    return results


def print_cv_results(results: Dict[str, Any], model_name: str = "Model"):
    """
    Pretty-print cross-validation results.

    Args:
        results: Output from evaluate_model_cv
        model_name: Name for display
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"{model_name} - Cross-Validation Results")
    logger.info(f"{'='*70}")
    logger.info(f"Successful folds: {results['n_successful_folds']}/{len(results['pr_aucs'])}")
    logger.info(f"\nPR-AUC:     {results['mean_pr_auc']:.4f} ± {results['std_pr_auc']:.4f}")
    logger.info(f"ROC-AUC:    {results['mean_roc_auc']:.4f} ± {results['std_roc_auc']:.4f}")
    logger.info(f"Accuracy:   {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    logger.info(f"F1-Score:   {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
    logger.info(f"\nPer-fold PR-AUCs: {[f'{x:.4f}' for x in results['pr_aucs']]}")
    logger.info(f"Per-fold ROC-AUCs: {[f'{x:.4f}' for x in results['roc_aucs']]}")
    logger.info(f"{'='*70}\n")


def compare_models_cv(
    results_dict: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Compare multiple models' CV results in a table.

    Args:
        results_dict: Dict mapping model_name -> cv_results

    Returns:
        DataFrame with comparison table
    """
    comparison_data = []

    for model_name, results in results_dict.items():
        comparison_data.append({
            'Model': model_name,
            'PR-AUC': f"{results['mean_pr_auc']:.4f} ± {results['std_pr_auc']:.4f}",
            'ROC-AUC': f"{results['mean_roc_auc']:.4f} ± {results['std_roc_auc']:.4f}",
            'Accuracy': f"{results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}",
            'F1-Score': f"{results['mean_f1']:.4f} ± {results['std_f1']:.4f}",
            'Successful Folds': f"{results['n_successful_folds']}/{len(results['pr_aucs'])}"
        })

    df = pd.DataFrame(comparison_data)
    return df


# Example usage for classical models
def train_random_forest(X_train, y_train, **kwargs):
    """Example: Train RandomForest classifier."""
    from sklearn.ensemble import RandomForestClassifier

    n_estimators = kwargs.get('n_estimators', 100)
    max_depth = kwargs.get('max_depth', 10)
    random_state = kwargs.get('random_state', 42)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train, **kwargs):
    """Example: Train LogisticRegression classifier."""
    from sklearn.linear_model import LogisticRegression

    C = kwargs.get('C', 1.0)
    random_state = kwargs.get('random_state', 42)

    model = LogisticRegression(
        C=C,
        max_iter=1000,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_rbf_svm(X_train, y_train, **kwargs):
    """Example: Train RBF SVM classifier."""
    from sklearn.svm import SVC

    C = kwargs.get('C', 1.0)
    gamma = kwargs.get('gamma', 'scale')
    random_state = kwargs.get('random_state', 42)

    model = SVC(
        C=C,
        gamma=gamma,
        kernel='rbf',
        probability=True,
        random_state=random_state,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model
