#!/usr/bin/env python3
"""
Quantum-Enhanced Cure Analysis Script

This script runs the quantum-enhanced cure prediction framework with quantum machine learning models.
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

# Try to import quantum libraries
try:
    from quantum_cure_prediction.quantum_enhanced_framework import QuantumCurePredictionFramework, run_quantum_cure_prediction_pipeline
    # Check if quantum libraries are available by importing the global variable
    from quantum_cure_prediction.quantum_enhanced_framework import QUANTUM_AVAILABLE
except ImportError as e:
    print(f"Quantum libraries not available: {e}")
    QUANTUM_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_quantum_cure_analysis(
    relation_type: str = "CtD",
    max_entities: Optional[int] = 100,
    embedding_method: str = "ComplEx",
    embedding_dim: int = 64,
    test_size: float = 0.2,
    num_qubits: int = 10,
    top_compounds_to_analyze: int = 3,
    top_predictions_per_compound: int = 5,
    data_dir: str = "data",
    results_dir: str = "results"
):
    """
    Run quantum-enhanced analysis on the knowledge graph to find potential cures.
    """
    if not QUANTUM_AVAILABLE:
        logger.error("Quantum libraries not available. Install qiskit and related packages.")
        return None, {}
    
    logger.info("="*80)
    logger.info("RUNNING QUANTUM-ENHANCED CURE ANALYSIS ON KNOWLEDGE GRAPH")
    logger.info("="*80)
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize the quantum-enhanced cure prediction framework
    framework = QuantumCurePredictionFramework(data_dir=data_dir, results_dir=results_dir)
    
    # Run the quantum-enhanced simulation
    logger.info(f"Starting quantum-enhanced simulation with parameters:")
    logger.info(f"  - Relation type: {relation_type}")
    logger.info(f"  - Max entities: {'No limit' if max_entities is None else max_entities}")
    logger.info(f"  - Embedding method: {embedding_method}")
    logger.info(f"  - Embedding dimension: {embedding_dim}")
    logger.info(f"  - Test size: {test_size}")
    logger.info(f"  - Quantum qubits: {num_qubits}")
    
    results = framework.run_quantum_simulation(
        relation_type=relation_type,
        max_entities=max_entities,
        embedding_method=embedding_method,
        embedding_dim=embedding_dim,
        test_size=test_size,
        num_qubits=num_qubits
    )
    
    # Load the graph to identify all compounds and diseases
    from kg_layer.kg_loader import load_hetionet_edges, extract_task_edges
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
    
    # Limit to top compounds for analysis
    compounds_to_analyze = all_compounds[:top_compounds_to_analyze]
    diseases_to_consider = all_diseases[:20]  # Increased from previous run
    
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
            if not predictions.empty:
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
        predictions_file = os.path.join(results_dir, f"quantum_enhanced_potential_cures_predictions_{timestamp}.csv")
        combined_predictions.to_csv(predictions_file, index=False)
        
        logger.info(f"Quantum-enhanced potential cures predictions saved to: {predictions_file}")
        
        # Print top predictions
        logger.info("\nTOP QUANTUM-ENHANCED POTENTIAL CURES:")
        logger.info("-" * 80)
        for i, (_, row) in enumerate(combined_predictions.head(20).iterrows()):
            logger.info(f"{i+1:2d}. {row['compound']} → {row['disease']}: {row['prediction_score']:.4f}")
        
        # Also save top predictions by compound
        if not combined_predictions.empty:
            top_by_compound = combined_predictions.groupby('analyzed_compound').first().reset_index()
            top_by_compound_file = os.path.join(results_dir, f"quantum_enhanced_top_cures_by_compound_{timestamp}.csv")
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
        summary_file = os.path.join(results_dir, f"quantum_enhanced_analysis_summary_{timestamp}.json")
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        logger.info(f"\nQuantum-enhanced analysis summary saved to: {summary_file}")
        
        # Print model performance comparison
        logger.info("\nQUANTUM MODEL PERFORMANCE COMPARISON:")
        logger.info("-" * 80)
        if 'sorted_by_pr_auc' in framework.results:
            for model_name, metrics in framework.results['sorted_by_pr_auc']:
                logger.info(f"{model_name:<35} PR-AUC: {metrics['pr_auc']:.4f}, Acc: {metrics['accuracy']:.4f}")
        
        return combined_predictions, summary_stats
    else:
        logger.warning("No predictions were generated due to errors in analysis.")
        return None, {}


def main():
    if not QUANTUM_AVAILABLE:
        print("Quantum libraries not available. Install qiskit and related packages.")
        print("Required packages: qiskit, qiskit-machine-learning, qiskit-algorithms")
        return
    
    parser = argparse.ArgumentParser(description="Run quantum-enhanced cure analysis on knowledge graph")
    
    parser.add_argument("--relation", type=str, default="CtD", 
                       help="Relation type to analyze (default: CtD for compound-treats-disease)")
    parser.add_argument("--max_entities", type=int, default=100,
                       help="Maximum entities to include (default: 100 for faster processing)")
    parser.add_argument("--embedding_method", type=str, default="ComplEx",
                       choices=["ComplEx", "RotatE", "DistMult", "TransE"],
                       help="KG embedding method to use (default: ComplEx)")
    parser.add_argument("--embedding_dim", type=int, default=64,
                       help="Dimension of embeddings (default: 64)")
    parser.add_argument("--test_size", type=float, default=0.2,
                       help="Proportion of data for testing (default: 0.2)")
    parser.add_argument("--num_qubits", type=int, default=10,
                       help="Number of qubits for quantum models (default: 10)")
    parser.add_argument("--top_compounds", type=int, default=3,
                       help="Number of top compounds to analyze (default: 3)")
    parser.add_argument("--top_predictions", type=int, default=5,
                       help="Number of top predictions per compound (default: 5)")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Directory for data files (default: data)")
    parser.add_argument("--results_dir", type=str, default="results",
                       help="Directory for results (default: results)")
    
    args = parser.parse_args()
    
    # Run the quantum-enhanced analysis
    predictions, summary = run_quantum_cure_analysis(
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
    
    logger.info("\nQuantum-enhanced cure analysis completed successfully!")


if __name__ == "__main__":
    main()