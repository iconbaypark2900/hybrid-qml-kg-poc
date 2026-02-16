#!/usr/bin/env python3
"""
Simplified cure analysis that focuses on classical ML approaches without quantum dependencies.

This script implements a streamlined pipeline:
1. Loads the Hetionet knowledge graph
2. Trains classical embeddings
3. Builds link prediction models (classical only)
4. Identifies potential compound-disease treatments
5. Ranks compounds by their potential to treat diseases
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kg_layer.kg_loader import load_hetionet_edges, extract_task_edges, prepare_link_prediction_dataset
from kg_layer.kg_embedder import HetionetEmbedder
from kg_layer.kg_visualizer import KGVisualizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred, y_score=None) -> Dict[str, float]:
    """Compute comprehensive metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_score is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            metrics["roc_auc"] = float("nan")
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        except Exception:
            metrics["pr_auc"] = float("nan")

    return metrics


class SimplifiedCurePredictionFramework:
    """
    Simplified framework for predicting potential cures using knowledge graph embeddings
    and classical machine learning models only.
    """

    def __init__(self, data_dir: str = "data", results_dir: str = "results"):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.task_edges = None
        self.entity_to_id = {}
        self.id_to_entity = {}
        self.embeddings = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
        os.makedirs(results_dir, exist_ok=True)

    def load_knowledge_graph(self, relation_type: str = "CtD", max_entities: Optional[int] = None):
        """
        Load the knowledge graph for cure prediction.

        Args:
            relation_type: The relation type to focus on (e.g., "CtD" for compound-treats-disease)
            max_entities: Maximum number of entities to include (for performance)
        """
        logger.info(f"Loading knowledge graph for relation: {relation_type}")
        
        # Load Hetionet edges
        df_edges = load_hetionet_edges(data_dir=self.data_dir)
        
        # Extract task-specific edges
        self.task_edges, self.entity_to_id, self.id_to_entity = extract_task_edges(
            df_edges, 
            relation_type=relation_type, 
            max_entities=max_entities
        )
        
        logger.info(f"Loaded {len(self.task_edges)} {relation_type} edges with {len(self.entity_to_id)} entities")

    def train_embeddings(self, embedding_dim: int = 64, method: str = "random"):
        """
        Train knowledge graph embeddings using the fallback method (deterministic random).

        Args:
            embedding_dim: Dimension of embeddings
            method: Embedding method ("random" for deterministic fallback)
        """
        logger.info(f"Training {method} embeddings (dim={embedding_dim})")
        
        # Use the HetionetEmbedder which has fallback functionality
        embedder = HetionetEmbedder(
            embedding_dim=embedding_dim,
            work_dir=self.data_dir
        )
        
        # Prepare training data - just use the task edges
        train_data = self.task_edges[['source', 'target']].copy()
        train_data.columns = ['source', 'target']
        
        # Add dummy relation column to match expected format
        train_data['metaedge'] = 'treats'
        
        # Train embeddings (this will use deterministic fallback if PyKEEN not available)
        embedder.train_embeddings(train_data)
        
        self.embeddings = embedder.entity_embeddings
        self.entity_to_id = embedder.entity_to_id
        self.id_to_entity = embedder.id_to_entity
        
        logger.info(f"Embedding training completed. Shape: {self.embeddings.shape}")

    def prepare_link_prediction_data(self, test_size: float = 0.2, random_state: int = 42):
        """
        Prepare link prediction dataset with positive and negative samples.

        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        logger.info(f"Preparing link prediction dataset (test_size={test_size})")
        
        # Prepare training dataset with negative samples
        train_df, test_df = prepare_link_prediction_dataset(
            self.task_edges,
            test_size=test_size,
            random_state=random_state
        )
        
        logger.info(f"Dataset prepared: train={len(train_df)}, test={len(test_df)}")
        return train_df, test_df

    def extract_pair_features(self, pairs_df: pd.DataFrame, feature_type: str = "combined"):
        """
        Extract features for (compound, disease) pairs.

        Args:
            pairs_df: DataFrame with 'source_id' and 'target_id' columns
            feature_type: Type of features to extract ("concat", "diff", "hadamard", "combined")

        Returns:
            Feature matrix X
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not trained. Call train_embeddings() first.")
        
        features = []
        
        for _, row in pairs_df.iterrows():
            source_id = int(row['source_id'])
            target_id = int(row['target_id'])
            
            # Get embeddings for source and target
            source_emb = self.embeddings[source_id]
            target_emb = self.embeddings[target_id]
            
            # Extract features based on type
            if feature_type == "concat":
                # Concatenate source and target embeddings
                feat = np.concatenate([source_emb, target_emb])
            elif feature_type == "diff":
                # Difference between embeddings
                feat = np.abs(source_emb - target_emb)
            elif feature_type == "hadamard":
                # Hadamard (element-wise) product
                feat = source_emb * target_emb
            elif feature_type == "combined":
                # Combined features: [concat, diff, hadamard]
                concat_feat = np.concatenate([source_emb, target_emb])
                diff_feat = np.abs(source_emb - target_emb)
                hadamard_feat = source_emb * target_emb
                feat = np.concatenate([concat_feat, diff_feat, hadamard_feat])
            else:
                raise ValueError(f"Unknown feature_type: {feature_type}")
            
            features.append(feat)
        
        return np.array(features)

    def train_classical_models(self, X_train, y_train, X_test, y_test):
        """
        Train classical machine learning models.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
        """
        logger.info("Training classical models...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        classical_models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        for name, model in classical_models.items():
            logger.info(f"Training {name}...")
            
            # Fit model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Compute metrics
            metrics = compute_metrics(y_test, y_pred, y_pred_proba)
            
            # Store results
            self.models[name] = model
            self.results[f'classical_{name}'] = {
                'model': model,
                'test_metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            logger.info(f"{name} - Test PR-AUC: {metrics['pr_auc']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

    def run_simulation(self, 
                      relation_type: str = "CtD", 
                      max_entities: Optional[int] = None,
                      embedding_dim: int = 64,
                      test_size: float = 0.2):
        """
        Run the complete simulation for cure prediction.

        Args:
            relation_type: Relation type to predict (e.g., "CtD")
            max_entities: Max entities to include (None for all)
            embedding_dim: Embedding dimension
            test_size: Test set proportion
        """
        logger.info("Starting simplified cure prediction simulation...")
        
        # Step 1: Load knowledge graph
        self.load_knowledge_graph(relation_type=relation_type, max_entities=max_entities)
        
        # Step 2: Train embeddings
        self.train_embeddings(embedding_dim=embedding_dim)
        
        # Step 3: Prepare data
        train_df, test_df = self.prepare_link_prediction_data(test_size=test_size)
        
        # Step 4: Extract features
        X_train = self.extract_pair_features(train_df, feature_type="combined")
        X_test = self.extract_pair_features(test_df, feature_type="combined")
        y_train = train_df['label'].values
        y_test = test_df['label'].values
        
        logger.info(f"Feature extraction completed: X_train shape = {X_train.shape}")
        
        # Step 5: Train classical models
        self.train_classical_models(X_train, y_train, X_test, y_test)
        
        # Step 6: Compile results
        self.compile_results()
        
        logger.info("Simulation completed successfully!")
        return self.results

    def compile_results(self):
        """
        Compile and summarize all results.
        """
        logger.info("Compiling results...")
        
        summary = {}
        
        for key, result in self.results.items():
            if 'test_metrics' in result and result['test_metrics']:
                summary[key] = {
                    'pr_auc': result['test_metrics'].get('pr_auc', 0.0),
                    'accuracy': result['test_metrics'].get('accuracy', 0.0),
                    'precision': result['test_metrics'].get('precision', 0.0),
                    'recall': result['test_metrics'].get('recall', 0.0),
                    'f1': result['test_metrics'].get('f1', 0.0)
                }
        
        # Sort by PR-AUC
        sorted_results = sorted(summary.items(), key=lambda x: x[1]['pr_auc'], reverse=True)
        
        logger.info("Top performing models by PR-AUC:")
        for model_name, metrics in sorted_results[:5]:
            logger.info(f"  {model_name}: PR-AUC = {metrics['pr_auc']:.4f}, Accuracy = {metrics['accuracy']:.4f}")
        
        self.results['summary'] = summary
        self.results['sorted_by_pr_auc'] = sorted_results

    def predict_cures(self, compounds: List[str], diseases: List[str], top_k: int = 10):
        """
        Predict potential cures for given compounds and diseases.

        Args:
            compounds: List of compound IDs (e.g., ["Compound::DB00001", ...])
            diseases: List of disease IDs (e.g., ["Disease::DOID:1234", ...])
            top_k: Number of top predictions to return

        Returns:
            DataFrame with predictions
        """
        if not self.models:
            raise ValueError("No models trained. Run simulation first.")
        
        # Get the best performing model
        best_model_name = self.results['sorted_by_pr_auc'][0][0] if self.results.get('sorted_by_pr_auc') else None
        
        if not best_model_name:
            raise ValueError("No trained models found in results.")
        
        best_model = self.results[best_model_name]['model']
        
        # Create all possible pairs
        pairs = []
        for comp in compounds:
            for disease in diseases:
                if comp in self.entity_to_id and disease in self.entity_to_id:
                    pairs.append({
                        'compound': comp,
                        'disease': disease,
                        'compound_id': self.entity_to_id[comp],
                        'disease_id': self.entity_to_id[disease]
                    })
        
        if not pairs:
            raise ValueError("No valid compound-disease pairs found.")
        
        # Create DataFrame
        pairs_df = pd.DataFrame(pairs)
        pairs_df = pairs_df.rename(columns={'compound_id': 'source_id', 'disease_id': 'target_id'})
        
        # Extract features
        X = self.extract_pair_features(pairs_df, feature_type="combined")
        
        # Make predictions using the best model
        X_scaled = self.scaler.transform(X)
        probabilities = best_model.predict_proba(X_scaled)[:, 1]
        
        # Add predictions to DataFrame
        pairs_df['prediction_probability'] = probabilities
        pairs_df['prediction_score'] = probabilities  # Same for now, but could be different
        
        # Sort by prediction score
        pairs_df = pairs_df.sort_values('prediction_score', ascending=False)
        
        logger.info(f"Predicted top {top_k} potential cures:")
        for i, (_, row) in enumerate(pairs_df.head(top_k).iterrows()):
            logger.info(f"  {i+1}. {row['compound']} -> {row['disease']}: {row['prediction_score']:.4f}")
        
        return pairs_df.head(top_k)

    def get_compound_disease_interactions(self, compound_id: str, top_k: int = 10):
        """
        Get diseases most associated with a specific compound.

        Args:
            compound_id: The compound ID to analyze
            top_k: Number of top diseases to return

        Returns:
            DataFrame with compound-disease interactions
        """
        if self.task_edges is None:
            raise ValueError("Knowledge graph not loaded.")
        
        if compound_id not in self.entity_to_id:
            available_compounds = [eid for eid in self.entity_to_id.keys() if eid.startswith("Compound::")]
            raise ValueError(f"Compound {compound_id} not found. Available: {available_compounds[:10]}...")
        
        # Find all edges involving the compound
        compound_idx = self.entity_to_id[compound_id]
        
        # Get all edges where this compound is the source
        related_edges = self.task_edges[
            (self.task_edges['source_id'] == compound_idx) |
            (self.task_edges['target_id'] == compound_idx)
        ].copy()
        
        # Determine which column contains the disease
        if related_edges.empty:
            return pd.DataFrame(columns=['disease', 'relation', 'score'])
        
        # Add entity names
        related_edges['source_entity'] = related_edges['source_id'].map(self.id_to_entity)
        related_edges['target_entity'] = related_edges['target_id'].map(self.id_to_entity)
        
        # Identify diseases (could be in source or target)
        diseases = []
        for _, row in related_edges.iterrows():
            if row['source_entity'] == compound_id and row['target_entity'].startswith('Disease::'):
                diseases.append({
                    'disease': row['target_entity'],
                    'relation': row['metaedge'],
                    'score': 1.0  # Known association
                })
            elif row['target_entity'] == compound_id and row['source_entity'].startswith('Disease::'):
                diseases.append({
                    'disease': row['source_entity'],
                    'relation': row['metaedge'],
                    'score': 1.0  # Known association
                })
        
        df_diseases = pd.DataFrame(diseases)
        
        if not df_diseases.empty:
            df_diseases = df_diseases.sort_values('score', ascending=False).head(top_k)
        
        return df_diseases


def run_simplified_cure_analysis(
    relation_type: str = "CtD",
    max_entities: Optional[int] = 200,  # Limit for faster processing
    embedding_dim: int = 64,
    test_size: float = 0.2,
    top_compounds_to_analyze: int = 5,
    top_predictions_per_compound: int = 3,
    data_dir: str = "data",
    results_dir: str = "results"
):
    """
    Run simplified analysis on the knowledge graph to find potential cures.

    Args:
        relation_type: The relation type to analyze (e.g., "CtD" for compound-treats-disease)
        max_entities: Maximum entities to include (limited for faster processing)
        embedding_dim: Dimension of embeddings
        test_size: Proportion of data for testing
        top_compounds_to_analyze: Number of top compounds to analyze for potential cures
        top_predictions_per_compound: Number of top predictions per compound
        data_dir: Directory for data files
        results_dir: Directory for results
    """
    logger.info("="*80)
    logger.info("RUNNING SIMPLIFIED CURE ANALYSIS ON KNOWLEDGE GRAPH")
    logger.info("="*80)
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize the cure prediction framework
    framework = SimplifiedCurePredictionFramework(data_dir=data_dir, results_dir=results_dir)
    
    # Run the complete simulation
    logger.info(f"Starting simulation with parameters:")
    logger.info(f"  - Relation type: {relation_type}")
    logger.info(f"  - Max entities: {'No limit' if max_entities is None else max_entities}")
    logger.info(f"  - Embedding dimension: {embedding_dim}")
    logger.info(f"  - Test size: {test_size}")
    
    results = framework.run_simulation(
        relation_type=relation_type,
        max_entities=max_entities,
        embedding_dim=embedding_dim,
        test_size=test_size
    )
    
    # Load the graph to identify all compounds and diseases
    logger.info("Loading knowledge graph to identify compounds and diseases...")
    df_edges = load_hetionet_edges(data_dir=data_dir)
    
    # Extract all unique compounds and diseases from the task edges
    all_compounds = []
    all_diseases = []
    
    for _, row in framework.task_edges.iterrows():
        source_entity = framework.id_to_entity[row['source_id']]
        target_entity = framework.id_to_entity[row['target_id']]
        
        if source_entity.startswith('Compound::'):
            all_compounds.append(source_entity)
        elif source_entity.startswith('Disease::'):
            all_diseases.append(source_entity)
            
        if target_entity.startswith('Compound::'):
            all_compounds.append(target_entity)
        elif target_entity.startswith('Disease::'):
            all_diseases.append(target_entity)
    
    # Remove duplicates
    all_compounds = list(set(all_compounds))
    all_diseases = list(set(all_diseases))
    
    logger.info(f"Identified {len(all_compounds)} unique compounds and {len(all_diseases)} unique diseases")
    
    # Limit to top compounds for analysis (to manage computational complexity)
    compounds_to_analyze = all_compounds[:top_compounds_to_analyze]
    diseases_to_consider = all_diseases[:20]  # Limit diseases too for performance
    
    logger.info(f"Analyzing top {len(compounds_to_analyze)} compounds against {len(diseases_to_consider)} diseases")
    
    # Find potential cures for each compound
    all_predictions = []
    
    for i, compound in enumerate(compounds_to_analyze):
        logger.info(f"Analyzing compound {i+1}/{len(compounds_to_analyze)}: {compound}")
        
        try:
            # Find potential cures for this compound
            predictions = framework.predict_cures(
                compounds=[compound],
                diseases=diseases_to_consider,
                top_k=top_predictions_per_compound
            )
            
            # Add compound identifier to predictions
            predictions['analyzed_compound'] = compound
            
            # Add to all predictions
            all_predictions.append(predictions)
            
        except Exception as e:
            logger.error(f"Error analyzing compound {compound}: {e}")
            continue
    
    # Combine all predictions
    if all_predictions:
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        
        # Sort by prediction score
        combined_predictions = combined_predictions.sort_values('prediction_score', ascending=False)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        predictions_file = os.path.join(results_dir, f"simplified_potential_cures_predictions_{timestamp}.csv")
        combined_predictions.to_csv(predictions_file, index=False)
        
        logger.info(f"Potential cures predictions saved to: {predictions_file}")
        
        # Print top predictions
        logger.info("\nTOP POTENTIAL CURES:")
        logger.info("-" * 80)
        for i, (_, row) in enumerate(combined_predictions.head(20).iterrows()):
            logger.info(f"{i+1:2d}. {row['compound']} → {row['disease']}: {row['prediction_score']:.4f}")
        
        # Also save top predictions by compound
        if not combined_predictions.empty:
            top_by_compound = combined_predictions.groupby('analyzed_compound').first().reset_index()
            top_by_compound_file = os.path.join(results_dir, f"top_cures_by_compound_{timestamp}.csv")
            top_by_compound.to_csv(top_by_compound_file, index=False)
            
            logger.info(f"\nTop cure per compound saved to: {top_by_compound_file}")
        
        # Create summary statistics
        summary_stats = {
            'total_compounds_analyzed': len(compounds_to_analyze),
            'total_diseases_considered': len(diseases_to_consider),
            'total_predictions_made': len(combined_predictions),
            'top_prediction': f"{combined_predictions.iloc[0]['compound']} → {combined_predictions.iloc[0]['disease']}" if not combined_predictions.empty else "N/A",
            'top_prediction_score': combined_predictions.iloc[0]['prediction_score'] if not combined_predictions.empty else 0.0,
            'timestamp': timestamp
        }
        
        # Save summary
        summary_file = os.path.join(results_dir, f"analysis_summary_{timestamp}.json")
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        logger.info(f"\nAnalysis summary saved to: {summary_file}")
        
        return combined_predictions, summary_stats
    else:
        logger.warning("No predictions were generated due to errors in analysis.")
        return None, {}


def main():
    parser = argparse.ArgumentParser(description="Run simplified cure analysis on knowledge graph")
    
    parser.add_argument("--relation", type=str, default="CtD", 
                       help="Relation type to analyze (default: CtD for compound-treats-disease)")
    parser.add_argument("--max_entities", type=int, default=200,
                       help="Maximum entities to include (default: 200 for faster processing)")
    parser.add_argument("--embedding_dim", type=int, default=64,
                       help="Dimension of embeddings (default: 64)")
    parser.add_argument("--test_size", type=float, default=0.2,
                       help="Proportion of data for testing (default: 0.2)")
    parser.add_argument("--top_compounds", type=int, default=5,
                       help="Number of top compounds to analyze (default: 5)")
    parser.add_argument("--top_predictions", type=int, default=3,
                       help="Number of top predictions per compound (default: 3)")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Directory for data files (default: data)")
    parser.add_argument("--results_dir", type=str, default="results",
                       help="Directory for results (default: results)")
    
    args = parser.parse_args()
    
    # Run the simplified analysis
    predictions, summary = run_simplified_cure_analysis(
        relation_type=args.relation,
        max_entities=args.max_entities,
        embedding_dim=args.embedding_dim,
        test_size=args.test_size,
        top_compounds_to_analyze=args.top_compounds,
        top_predictions_per_compound=args.top_predictions,
        data_dir=args.data_dir,
        results_dir=args.results_dir
    )
    
    logger.info("\nSimplified cure analysis completed successfully!")


if __name__ == "__main__":
    main()