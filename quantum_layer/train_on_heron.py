# quantum_layer/train_on_heron.py

"""
Training script specifically optimized for IBM Heron quantum processor.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .qml_trainer import QMLTrainer
from kg_layer.kg_loader import load_hetionet_edges, extract_task_edges, prepare_link_prediction_dataset
from kg_layer.kg_embedder import HetionetEmbedder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_on_heron():
    """Train QML model on IBM Heron processor."""
    
    # Load smaller dataset for Heron (cost and time constraints)
    logger.info("Loading Hetionet data for Heron training...")
    df = load_hetionet_edges()
    task_edges, _, _ = extract_task_edges(df, relation_type="CtD", max_entities=200)
    train_df, test_df = prepare_link_prediction_dataset(task_edges)
    
    # Load embeddings (reduced dimension for Heron)
    logger.info("Generating embeddings for Heron...")
    embedder = HetionetEmbedder(embedding_dim=16, qml_dim=4)
    if not embedder.load_saved_embeddings():
        embedder.train_embeddings(train_df)
        embedder.reduce_to_qml_dim()
    
    # Heron-optimized QML config
    qml_config = {
        "model_type": "VQC",
        "encoding_method": "feature_map",
        "num_qubits": 4,
        "feature_map_type": "ZZ",
        "feature_map_reps": 1,
        "ansatz_type": "RealAmplitudes",
        "ansatz_reps": 2,
        "optimizer": "SPSA",
        "max_iter": 25,
        "random_state": 42
    }
    
    # Train with Heron configuration
    logger.info("Starting Heron training...")
    trainer = QMLTrainer()
    results = trainer.train_and_evaluate(
        train_df, test_df, embedder, qml_config,
        classical_model_type="LogisticRegression",
        quantum_config_path="config/quantum_config.yaml"
    )
    
    print("\n" + "="*50)
    print("HERON TRAINING RESULTS")
    print("="*50)
    print(f"Quantum PR-AUC:   {results['quantum']['pr_auc']:.4f}")
    print(f"Classical PR-AUC: {results['classical']['pr_auc']:.4f}")
    print(f"Quantum Params:   {results['quantum']['num_parameters']}")
    print(f"Classical Params: {results['classical']['num_parameters']}")
    
    # Cost estimation
    if 'quantum_executor' in trainer.__dict__:
        logger.info("Cost estimation completed during training")

if __name__ == "__main__":
    train_on_heron()