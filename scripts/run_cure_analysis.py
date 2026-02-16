#!/usr/bin/env python3
"""
Run comprehensive tests on the full knowledge graph to find potential cures.

This script implements the complete pipeline:
1. Loads the full Hetionet knowledge graph
2. Trains advanced embeddings
3. Builds link prediction models (classical and quantum)
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

from kg_layer.kg_loader import load_hetionet_edges, extract_task_edges
from kg_layer.advanced_embeddings import AdvancedKGEmbedder
from cure_prediction.simulation_framework import CurePredictionFramework, find_potential_cures
from kg_layer.kg_visualizer import KGVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_full_kg_cure_analysis(
    relation_type: str = "CtD",
    max_entities: Optional[int] = None,  # None means no limit
    embedding_method: str = "ComplEx",
    embedding_dim: int = 128,
    test_size: float = 0.2,
    num_qubits: int = 10,
    top_compounds_to_analyze: int = 20,
    top_predictions_per_compound: int = 10,
    data_dir: str = "data",
    results_dir: str = "results"
):
    """
    Run comprehensive analysis on the full knowledge graph to find potential cures.

    Args:
        relation_type: The relation type to analyze (e.g., "CtD" for compound-treats-disease)
        max_entities: Maximum entities to include (None for full graph)
        embedding_method: KG embedding method to use
        embedding_dim: Dimension of embeddings
        test_size: Proportion of data for testing
        num_qubits: Number of qubits for quantum models
        top_compounds_to_analyze: Number of top compounds to analyze for potential cures
        top_predictions_per_compound: Number of top predictions per compound
        data_dir: Directory for data files
        results_dir: Directory for results
    """
    logger.info("="*80)
    logger.info("RUNNING COMPREHENSIVE CURE ANALYSIS ON FULL KNOWLEDGE GRAPH")
    logger.info("="*80)
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize the cure prediction framework
    framework = CurePredictionFramework(data_dir=data_dir, results_dir=results_dir)
    
    # Run the complete simulation
    logger.info(f"Starting simulation with parameters:")
    logger.info(f"  - Relation type: {relation_type}")
    logger.info(f"  - Max entities: {'No limit' if max_entities is None else max_entities}")
    logger.info(f"  - Embedding method: {embedding_method}")
    logger.info(f"  - Embedding dimension: {embedding_dim}")
    logger.info(f"  - Test size: {test_size}")
    logger.info(f"  - Quantum qubits: {num_qubits}")
    
    results = framework.run_simulation(
        relation_type=relation_type,
        max_entities=max_entities,
        embedding_method=embedding_method,
        embedding_dim=embedding_dim,
        test_size=test_size,
        num_qubits=num_qubits
    )
    
    # Load the full graph to identify all compounds and diseases
    logger.info("Loading full knowledge graph to identify compounds and diseases...")
    df_edges = load_hetionet_edges(data_dir=data_dir)
    
    # Extract all unique compounds and diseases
    all_compounds = df_edges[
        df_edges['source'].str.startswith('Compound::') | 
        df_edges['target'].str.startswith('Compound::')
    ]['source'].append(
        df_edges[
            df_edges['source'].str.startswith('Compound::') | 
            df_edges['target'].str.startswith('Compound::')
        ]['target']
    ).unique()
    
    all_compounds = [c for c in all_compounds if str(c).startswith('Compound::')]
    all_compounds = list(set(all_compounds))  # Remove duplicates
    
    all_diseases = df_edges[
        df_edges['source'].str.startswith('Disease::') | 
        df_edges['target'].str.startswith('Disease::')
    ]['source'].append(
        df_edges[
            df_edges['source'].str.startswith('Disease::') | 
            df_edges['target'].str.startswith('Disease::')
        ]['target']
    ).unique()
    
    all_diseases = [d for d in all_diseases if str(d).startswith('Disease::')]
    all_diseases = list(set(all_diseases))  # Remove duplicates
    
    logger.info(f"Identified {len(all_compounds)} unique compounds and {len(all_diseases)} unique diseases")
    
    # Limit to top compounds for analysis (to manage computational complexity)
    compounds_to_analyze = all_compounds[:top_compounds_to_analyze]
    diseases_to_consider = all_diseases[:100]  # Limit diseases too for performance
    
    logger.info(f"Analyzing top {len(compounds_to_analyze)} compounds against {len(diseases_to_consider)} diseases")
    
    # Find potential cures for each compound
    all_predictions = []
    
    for i, compound in enumerate(compounds_to_analyze):
        logger.info(f"Analyzing compound {i+1}/{len(compounds_to_analyze)}: {compound}")
        
        try:
            # Find potential cures for this compound
            predictions = find_potential_cures(
                compound_list=[compound],
                disease_list=diseases_to_consider,
                framework=framework,
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
        predictions_file = os.path.join(results_dir, f"potential_cures_predictions_{timestamp}.csv")
        combined_predictions.to_csv(predictions_file, index=False)
        
        logger.info(f"Potential cures predictions saved to: {predictions_file}")
        
        # Print top predictions
        logger.info("\nTOP POTENTIAL CURES:")
        logger.info("-" * 80)
        for i, (_, row) in enumerate(combined_predictions.head(20).iterrows()):
            logger.info(f"{i+1:2d}. {row['compound']} → {row['disease']}: {row['prediction_score']:.4f}")
        
        # Also save top predictions by compound
        top_by_compound = combined_predictions.groupby('analyzed_compound').first().reset_index()
        top_by_compound_file = os.path.join(results_dir, f"top_cures_by_compound_{timestamp}.csv")
        top_by_compound.to_csv(top_by_compound_file, index=False)
        
        logger.info(f"\nTop cure per compound saved to: {top_by_compound_file}")
        
        # Create summary statistics
        summary_stats = {
            'total_compounds_analyzed': len(compounds_to_analyze),
            'total_diseases_considered': len(diseases_to_consider),
            'total_predictions_made': len(combined_predictions),
            'top_prediction': f"{combined_predictions.iloc[0]['compound']} → {combined_predictions.iloc[0]['disease']}",
            'top_prediction_score': combined_predictions.iloc[0]['prediction_score'],
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


def run_detailed_compound_analysis(compound_id: str, framework: CurePredictionFramework, top_k: int = 10):
    """
    Run detailed analysis for a specific compound.

    Args:
        compound_id: The compound ID to analyze
        framework: Trained CurePredictionFramework instance
        top_k: Number of top predictions to return
    """
    logger.info(f"Running detailed analysis for compound: {compound_id}")
    
    # Get known interactions from the knowledge graph
    known_interactions = framework.get_compound_disease_interactions(compound_id, top_k=20)
    
    logger.info(f"Known interactions for {compound_id}:")
    for i, (_, row) in enumerate(known_interactions.iterrows()):
        logger.info(f"  {i+1}. {row['disease']} ({row['relation']})")
    
    # Get all diseases in the knowledge graph
    all_diseases = [eid for eid in framework.entity_to_id.keys() if eid.startswith('Disease::')]
    
    # Predict potential new interactions
    predictions = find_potential_cures(
        compound_list=[compound_id],
        disease_list=all_diseases,
        framework=framework,
        top_k=top_k
    )
    
    logger.info(f"\nTop potential new interactions for {compound_id}:")
    for i, (_, row) in enumerate(predictions.head(top_k).iterrows()):
        logger.info(f"  {i+1}. {row['disease']}: {row['prediction_score']:.4f}")
    
    return known_interactions, predictions


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive cure analysis on full knowledge graph")
    
    parser.add_argument("--relation", type=str, default="CtD", 
                       help="Relation type to analyze (default: CtD for compound-treats-disease)")
    parser.add_argument("--max_entities", type=int, default=None,
                       help="Maximum entities to include (default: None = no limit)")
    parser.add_argument("--embedding_method", type=str, default="ComplEx",
                       choices=["ComplEx", "RotatE", "DistMult", "TransE"],
                       help="KG embedding method to use (default: ComplEx)")
    parser.add_argument("--embedding_dim", type=int, default=128,
                       help="Dimension of embeddings (default: 128)")
    parser.add_argument("--test_size", type=float, default=0.2,
                       help="Proportion of data for testing (default: 0.2)")
    parser.add_argument("--num_qubits", type=int, default=10,
                       help="Number of qubits for quantum models (default: 10)")
    parser.add_argument("--top_compounds", type=int, default=20,
                       help="Number of top compounds to analyze (default: 20)")
    parser.add_argument("--top_predictions", type=int, default=10,
                       help="Number of top predictions per compound (default: 10)")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Directory for data files (default: data)")
    parser.add_argument("--results_dir", type=str, default="results",
                       help="Directory for results (default: results)")
    parser.add_argument("--analyze_compound", type=str, default=None,
                       help="Specific compound ID to run detailed analysis on")
    
    args = parser.parse_args()
    
    # Run the full analysis
    predictions, summary = run_full_kg_cure_analysis(
        relation_type=args.relation,
        max_entities=args.max_entities,
        embedding_method=args.embedding_method,
        embedding_dim=args.embedding_dim,
        test_size=args.test_size,
        num_qubits=args.num_qubits,
        top_compounds_to_analyze=args.top_compounds,
        top_predictions_per_compound=args.top_predictions,
        data_dir=args.data_dir,
        results_dir=args.results_dir
    )
    
    # If a specific compound was requested for detailed analysis
    if args.analyze_compound:
        # We need to create a framework instance for detailed analysis
        # Since the full analysis already created one, we'll run it separately
        logger.info(f"\nRunning detailed analysis for: {args.analyze_compound}")
        
        # For detailed analysis, we need to run a focused analysis
        framework = CurePredictionFramework(data_dir=args.data_dir, results_dir=args.results_dir)
        
        # Load the knowledge graph
        framework.load_knowledge_graph(relation_type=args.relation, max_entities=args.max_entities)
        
        # Train embeddings
        framework.train_embeddings(
            method=args.embedding_method,
            embedding_dim=args.embedding_dim
        )
        
        # Detailed analysis
        known, predicted = run_detailed_compound_analysis(
            compound_id=args.analyze_compound,
            framework=framework,
            top_k=args.top_predictions
        )
    
    logger.info("\nCure analysis completed successfully!")


if __name__ == "__main__":
    main()