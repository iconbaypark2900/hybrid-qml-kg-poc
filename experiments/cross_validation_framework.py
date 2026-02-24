# experiments/cross_validation_framework.py
"""
Cross-Validation Framework - Issue 6 Deep Dive

This script addresses Issue 6: Missing Cross-Validation by implementing:
1. Nested cross-validation (outer for evaluation, inner for hyperparameter tuning)
2. Repeated k-fold cross-validation
3. Multi-seed evaluation
4. Statistical significance testing

Addresses:
- Issue 6: Missing Cross-Validation
- Issue 2: Classical Baseline Overfitting (via regularization tuning in inner CV)
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, RepeatedStratifiedKFold,
    GridSearchCV, cross_val_score
)
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.linear_model import LogisticRegression
from scipy.stats import wilcoxon, ttest_rel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_layer.qml_model import QMLLinkPredictor
from classical_baseline.train_baseline import ClassicalLinkPredictor
from kg_layer.kg_loader import load_hetionet_edges, extract_task_edges, prepare_link_prediction_dataset
from kg_layer.kg_embedder import HetionetEmbedder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom scorer for PR-AUC
pr_auc_scorer = make_scorer(average_precision_score, needs_proba=True)


class CrossValidationFramework:
    """Comprehensive cross-validation framework for model evaluation."""
    
    def __init__(self, results_dir: str = "results/cv", random_state: int = 42):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
    
    def nested_cv_classical(
        self,
        X: np.ndarray,
        y: np.ndarray,
        outer_cv: int = 5,
        inner_cv: int = 3,
        param_grid: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Nested cross-validation for classical models.
        
        Outer loop: Unbiased performance estimation
        Inner loop: Hyperparameter tuning
        """
        if param_grid is None:
            param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'lbfgs']
            }
        
        outer_scores = []
        best_params_per_fold = []
        
        outer_cv_splitter = StratifiedKFold(
            n_splits=outer_cv,
            shuffle=True,
            random_state=self.random_state
        )
        
        for fold, (train_idx, test_idx) in enumerate(outer_cv_splitter.split(X, y)):
            logger.info(f"\n{'='*60}")
            logger.info(f"Outer Fold {fold+1}/{outer_cv}")
            logger.info(f"{'='*60}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Inner CV: Tune hyperparameters
            inner_cv_splitter = StratifiedKFold(
                n_splits=inner_cv,
                shuffle=True,
                random_state=self.random_state
            )
            
            # Create base model
            base_model = LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=self.random_state
            )
            
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=inner_cv_splitter,
                scoring=pr_auc_scorer,
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            best_params_per_fold.append(grid_search.best_params_)
            logger.info(f"  Best params: {grid_search.best_params_}")
            logger.info(f"  Best inner CV score: {grid_search.best_score_:.4f}")
            
            # Evaluate on held-out test fold (unbiased!)
            best_model = grid_search.best_estimator_
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            score = average_precision_score(y_test, y_pred_proba)
            outer_scores.append(score)
            
            logger.info(f"  Test PR-AUC: {score:.4f}")
        
        # Report unbiased estimate
        mean_score = np.mean(outer_scores)
        std_score = np.std(outer_scores)
        ci_low = np.percentile(outer_scores, 2.5)
        ci_high = np.percentile(outer_scores, 97.5)
        
        results = {
            'mean_pr_auc': float(mean_score),
            'std_pr_auc': float(std_score),
            'ci_95_low': float(ci_low),
            'ci_95_high': float(ci_high),
            'all_scores': [float(s) for s in outer_scores],
            'best_params_per_fold': best_params_per_fold
        }
        
        logger.info(f"\n{'='*60}")
        logger.info("NESTED CV RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Mean PR-AUC: {mean_score:.4f} ± {std_score:.4f}")
        logger.info(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
        
        return results
    
    def repeated_kfold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model,
        n_splits: int = 5,
        n_repeats: int = 10
    ) -> Dict[str, Any]:
        """Repeated k-fold cross-validation."""
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=self.random_state
        )
        
        scores = cross_val_score(
            model,
            X, y,
            cv=cv,
            scoring=pr_auc_scorer,
            n_jobs=-1
        )
        
        mean_score = scores.mean()
        std_score = scores.std()
        ci_low = np.percentile(scores, 2.5)
        ci_high = np.percentile(scores, 97.5)
        
        results = {
            'mean_pr_auc': float(mean_score),
            'std_pr_auc': float(std_score),
            'ci_95_low': float(ci_low),
            'ci_95_high': float(ci_high),
            'all_scores': [float(s) for s in scores]
        }
        
        logger.info(f"\n{'='*60}")
        logger.info("REPEATED K-FOLD RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Mean PR-AUC: {mean_score:.4f} ± {std_score:.4f}")
        logger.info(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
        logger.info(f"Min: {scores.min():.4f}, Max: {scores.max():.4f}")
        
        return results
    
    def multi_seed_evaluation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_fn,
        seeds: List[int],
        train_test_split_func
    ) -> Dict[str, Any]:
        """
        Evaluate model across multiple random seeds.
        
        Args:
            model_fn: Function that creates and trains a model
            seeds: List of random seeds
            train_test_split_func: Function that splits data and returns (X_train, X_test, y_train, y_test)
        """
        all_scores = []
        
        for seed in seeds:
            logger.info(f"\n{'='*60}")
            logger.info(f"Seed {seed}")
            logger.info(f"{'='*60}")
            
            np.random.seed(seed)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split_func(X, y, seed)
            
            # Train and evaluate
            model = model_fn(X_train, y_train, seed)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            score = average_precision_score(y_test, y_pred_proba)
            
            all_scores.append(score)
            logger.info(f"PR-AUC: {score:.4f}")
        
        mean_score = np.mean(all_scores)
        std_score = np.std(all_scores)
        
        results = {
            'mean_pr_auc': float(mean_score),
            'std_pr_auc': float(std_score),
            'all_scores': [float(s) for s in all_scores],
            'seeds': seeds
        }
        
        logger.info(f"\n{'='*60}")
        logger.info("MULTI-SEED RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Mean PR-AUC: {mean_score:.4f} ± {std_score:.4f}")
        logger.info(f"Min: {min(all_scores):.4f}, Max: {max(all_scores):.4f}")
        
        return results
    
    def compare_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models: Dict[str, Any],
        cv: int = 5
    ) -> pd.DataFrame:
        """Compare multiple models using cross-validation."""
        results = []
        
        for name, model in models.items():
            logger.info(f"\nEvaluating: {name}")
            
            scores = cross_val_score(
                model,
                X, y,
                cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
                scoring=pr_auc_scorer,
                n_jobs=-1
            )
            
            results.append({
                'model': name,
                'mean_pr_auc': scores.mean(),
                'std_pr_auc': scores.std(),
                'min_pr_auc': scores.min(),
                'max_pr_auc': scores.max()
            })
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('mean_pr_auc', ascending=False)
        
        logger.info(f"\n{'='*60}")
        logger.info("MODEL COMPARISON")
        logger.info(f"{'='*60}")
        logger.info(df_results.to_string(index=False))
        
        return df_results
    
    def statistical_significance_test(
        self,
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        model_a_name: str = "Model A",
        model_b_name: str = "Model B"
    ) -> Dict[str, Any]:
        """Perform statistical significance tests between two models."""
        # Paired t-test
        t_stat, t_pval = ttest_rel(scores_a, scores_b)
        
        # Wilcoxon signed-rank test (non-parametric)
        w_stat, w_pval = wilcoxon(scores_a, scores_b)
        
        # Effect size (Cohen's d)
        diff = scores_a - scores_b
        cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0
        
        results = {
            'model_a': model_a_name,
            'model_b': model_b_name,
            'mean_a': float(scores_a.mean()),
            'mean_b': float(scores_b.mean()),
            'mean_diff': float(diff.mean()),
            't_statistic': float(t_stat),
            't_pvalue': float(t_pval),
            'wilcoxon_statistic': float(w_stat),
            'wilcoxon_pvalue': float(w_pval),
            'cohens_d': float(cohens_d),
            'significant': t_pval < 0.05
        }
        
        logger.info(f"\n{'='*60}")
        logger.info("STATISTICAL SIGNIFICANCE TEST")
        logger.info(f"{'='*60}")
        logger.info(f"{model_a_name} mean: {scores_a.mean():.4f}")
        logger.info(f"{model_b_name} mean: {scores_b.mean():.4f}")
        logger.info(f"Mean difference: {diff.mean():.4f}")
        logger.info(f"\nPaired t-test: t={t_stat:.3f}, p={t_pval:.4f}")
        logger.info(f"Wilcoxon test: W={w_stat:.3f}, p={w_pval:.4f}")
        logger.info(f"Cohen's d: {cohens_d:.3f}")
        
        if t_pval < 0.05:
            logger.info(f"✅ Significant difference (p < 0.05)")
        else:
            logger.info(f"⚠️ No significant difference (p >= 0.05)")
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save results to JSON."""
        output_path = self.results_dir / filename
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {output_path}")


def create_classical_model_fn(X_train, y_train, seed):
    """Factory function for classical model."""
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=seed
    )
    model.fit(X_train_scaled, y_train)
    
    # Store scaler for prediction
    model.scaler = scaler
    return model


def create_qml_model_fn(X_train, y_train, seed, qml_config):
    """Factory function for QML model."""
    model = QMLLinkPredictor(**qml_config, random_state=seed)
    model.fit(X_train, y_train)
    return model


def main():
    parser = argparse.ArgumentParser(description="Cross-Validation Framework")
    parser.add_argument("--relation", type=str, default="CtD")
    parser.add_argument("--max_entities", type=int, default=300)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--qml_dim", type=int, default=5)
    parser.add_argument("--qml_features", type=str, default="diff", choices=["diff", "hadamard", "both"])
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "nested_cv", "repeated_kfold", "multi_seed", "compare"])
    parser.add_argument("--results_dir", type=str, default="results/cv")
    parser.add_argument("--random_state", type=int, default=42)
    
    args = parser.parse_args()
    
    # Load data
    logger.info("Loading Hetionet data...")
    df = load_hetionet_edges()
    task_edges, _, _ = extract_task_edges(
        df,
        relation_type=args.relation,
        max_entities=args.max_entities
    )
    train_df, test_df = prepare_link_prediction_dataset(task_edges)
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    embedder = HetionetEmbedder(
        embedding_dim=args.embedding_dim,
        qml_dim=args.qml_dim
    )
    if not embedder.load_saved_embeddings():
        embedder.train_embeddings(task_edges)
        embedder.reduce_to_qml_dim()
    else:
        embedder.reduce_to_qml_dim()
    
    # Prepare features
    X_classical = embedder.prepare_link_features(train_df)
    X_qml = embedder.prepare_link_features_qml(train_df, mode=args.qml_features)
    y = train_df['label'].values
    
    # Scale classical features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_classical_scaled = scaler.fit_transform(X_classical)
    
    logger.info(f"Classical features: {X_classical_scaled.shape}")
    logger.info(f"QML features: {X_qml.shape}")
    
    # Initialize framework
    cv_framework = CrossValidationFramework(
        results_dir=args.results_dir,
        random_state=args.random_state
    )
    
    all_results = {}
    
    if args.experiment in ["all", "nested_cv"]:
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT: Nested Cross-Validation (Classical)")
        logger.info("="*60)
        
        nested_results = cv_framework.nested_cv_classical(
            X_classical_scaled, y,
            outer_cv=5,
            inner_cv=3
        )
        all_results['nested_cv_classical'] = nested_results
        cv_framework.save_results(nested_results, "nested_cv_classical.json")
    
    if args.experiment in ["all", "repeated_kfold"]:
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT: Repeated K-Fold (Classical)")
        logger.info("="*60)
        
        model = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=args.random_state
        )
        
        repeated_results = cv_framework.repeated_kfold(
            X_classical_scaled, y, model,
            n_splits=5,
            n_repeats=10
        )
        all_results['repeated_kfold_classical'] = repeated_results
        cv_framework.save_results(repeated_results, "repeated_kfold_classical.json")
    
    if args.experiment in ["all", "multi_seed"]:
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT: Multi-Seed Evaluation")
        logger.info("="*60)
        
        seeds = [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]
        
        def train_test_split_fn(X, y, seed):
            from sklearn.model_selection import train_test_split
            return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        
        # Classical
        def classical_model_fn(X_train, y_train, seed):
            return create_classical_model_fn(X_train, y_train, seed)
        
        classical_scores = []
        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split_fn(X_classical_scaled, y, seed)
            model = classical_model_fn(X_train, y_train, seed)
            y_pred_proba = model.predict_proba(model.scaler.transform(X_test))[:, 1]
            score = average_precision_score(y_test, y_pred_proba)
            classical_scores.append(score)
        
        # QML
        qml_config = {
            'model_type': 'VQC',
            'num_qubits': args.qml_dim,
            'ansatz_type': 'RealAmplitudes',
            'ansatz_reps': 3,
            'optimizer': 'COBYLA',
            'max_iter': 100
        }
        
        qml_scores = []
        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split_fn(X_qml, y, seed)
            model = create_qml_model_fn(X_train, y_train, seed, qml_config)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            score = average_precision_score(y_test, y_pred_proba)
            qml_scores.append(score)
        
        multi_seed_results = {
            'classical': {
                'mean_pr_auc': float(np.mean(classical_scores)),
                'std_pr_auc': float(np.std(classical_scores)),
                'all_scores': [float(s) for s in classical_scores]
            },
            'quantum': {
                'mean_pr_auc': float(np.mean(qml_scores)),
                'std_pr_auc': float(np.std(qml_scores)),
                'all_scores': [float(s) for s in qml_scores]
            }
        }
        
        # Statistical test
        sig_test = cv_framework.statistical_significance_test(
            np.array(qml_scores),
            np.array(classical_scores),
            "Quantum", "Classical"
        )
        multi_seed_results['significance_test'] = sig_test
        
        all_results['multi_seed'] = multi_seed_results
        cv_framework.save_results(multi_seed_results, "multi_seed_results.json")
    
    if args.experiment in ["all", "compare"]:
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT: Model Comparison")
        logger.info("="*60)
        
        models = {
            'LogisticRegression_C1': LogisticRegression(
                C=1.0, class_weight='balanced', max_iter=1000, random_state=args.random_state
            ),
            'LogisticRegression_C10': LogisticRegression(
                C=10.0, class_weight='balanced', max_iter=1000, random_state=args.random_state
            ),
            'LogisticRegression_C0.1': LogisticRegression(
                C=0.1, class_weight='balanced', max_iter=1000, random_state=args.random_state
            ),
        }
        
        comparison_results = cv_framework.compare_models(
            X_classical_scaled, y, models, cv=5
        )
        all_results['model_comparison'] = comparison_results.to_dict('records')
        comparison_results.to_csv(cv_framework.results_dir / "model_comparison.csv", index=False)
    
    # Save all results
    cv_framework.save_results(all_results, "all_cv_results.json")
    
    logger.info("\n" + "="*60)
    logger.info("CROSS-VALIDATION ANALYSIS COMPLETE")
    logger.info("="*60)
    logger.info(f"Results saved to: {cv_framework.results_dir}")


if __name__ == "__main__":
    main()

