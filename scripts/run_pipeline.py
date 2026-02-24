# scripts/run_pipeline.py
#!/usr/bin/env python3
"""Complete pipeline execution"""

from kg_layer.kg_loader import load_hetionet_edges, extract_task_edges, prepare_link_prediction_dataset
from kg_layer.kg_embedder import HetionetEmbedder
from classical_baseline.train_baseline import ClassicalLinkPredictor
from quantum_layer.qml_trainer import QMLTrainer
from benchmarking.scalability_sim import run_scalability_simulation
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_complete_pipeline():
    """Execute the complete 45-minute journey"""
    
    logger.info("🚀 Starting Hybrid QML-KG Pipeline")
    
    # Phase 1: Load KG
    logger.info("📊 Loading Hetionet...")
    df = load_hetionet_edges()
    task_edges, _, _ = extract_task_edges(df, relation_type="CtD", max_entities=300)
    train_df, test_df = prepare_link_prediction_dataset(task_edges)
    
    # Phase 2: Generate Embeddings
    logger.info("🧠 Generating embeddings...")
    embedder = HetionetEmbedder(embedding_dim=32, qml_dim=5)
    if not embedder.load_saved_embeddings():
        embedder.train_embeddings(train_df)
        embedder.reduce_to_qml_dim()
    
    # Phase 3: Train Classical
    logger.info("📈 Training classical baseline...")
    classical_predictor = ClassicalLinkPredictor()
    classical_predictor.train(train_df, embedder, test_df)
    
    # Phase 4: Train Quantum
    logger.info("⚛️ Training quantum model...")
    qml_config = {
        "model_type": "VQC",
        "encoding_method": "feature_map",
        "num_qubits": 5,
        "feature_map_type": "ZZ",
        "feature_map_reps": 2,
        "ansatz_type": "RealAmplitudes", 
        "ansatz_reps": 3,
        "optimizer": "COBYLA",
        "max_iter": 50,
        "random_state": 42
    }
    
    trainer = QMLTrainer()
    results = trainer.train_and_evaluate(train_df, test_df, embedder, qml_config)
    
    # Phase 5: Generate Scaling Plot
    logger.info("📊 Generating scaling projection...")
    run_scalability_simulation()
    
    # Results Summary
    logger.info("✅ Pipeline Complete!")
    logger.info(f"Classical PR-AUC: {results['classical']['pr_auc']:.4f}")
    logger.info(f"Quantum PR-AUC: {results['quantum']['pr_auc']:.4f}")
    logger.info(f"🚀 Ready for API/Dashboard demo!")
    
    return results

if __name__ == "__main__":
    run_complete_pipeline()