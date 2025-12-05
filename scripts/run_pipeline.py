# scripts/run_pipeline.py
#!/usr/bin/env python3
"""Complete pipeline execution"""

from kg_layer.kg_loader import (
    load_hetionet_edges,
    extract_task_edges,
    prepare_link_prediction_dataset,
    load_kg_config
)
from kg_layer.kg_embedder import HetionetEmbedder
from classical_baseline.train_baseline import ClassicalLinkPredictor, load_classical_config
from quantum_layer.qml_trainer import QMLTrainer, load_quantum_config
from benchmarking.scalability_sim import run_scalability_simulation
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_complete_pipeline(
    kg_config_path: str = "config/kg_layer_config.yaml",
    classical_config_path: str = "config/classical_layer_config.yaml",
    quantum_config_path: str = "config/quantum_layer_config.yaml"
):
    """
    Execute the complete pipeline using configuration files.

    Args:
        kg_config_path: Path to KG layer config YAML file
        classical_config_path: Path to classical layer config YAML file
        quantum_config_path: Path to quantum layer config YAML file
    """
    logger.info("🚀 Starting Hybrid QML-KG Pipeline")

    # Load configurations
    logger.info("📋 Loading configurations...")
    kg_config = load_kg_config(kg_config_path)
    classical_config = load_classical_config(classical_config_path)
    quantum_config = load_quantum_config(quantum_config_path)

    # Phase 1: Load KG
    logger.info("📊 Loading Hetionet...")
    df = load_hetionet_edges(data_dir=kg_config["data_loading"]["data_dir"])
    task_edges, _, _ = extract_task_edges(
        df,
        relation_type=kg_config["data_loading"]["relation_type"],
        max_entities=kg_config["data_loading"]["max_entities"],
        config=kg_config
    )
    train_df, test_df = prepare_link_prediction_dataset(
        task_edges,
        test_size=kg_config["data_loading"]["test_size"],
        random_state=kg_config["data_loading"]["random_state"],
        config=kg_config
    )

    # Phase 2: Generate Embeddings
    logger.info("🧠 Generating embeddings...")
    embedder = HetionetEmbedder(
        embedding_dim=kg_config["embedding"]["embedding_dim"],
        qml_dim=kg_config["embedding"]["qml_dim"],
        work_dir=kg_config["embedding"]["work_dir"],
        config=kg_config
    )
    if not embedder.load_saved_embeddings():
        embedder.train_embeddings(task_edges)
        embedder.reduce_to_qml_dim()

    # Phase 3: Train Classical
    logger.info("📈 Training classical baseline...")
    classical_predictor = ClassicalLinkPredictor(config=classical_config)
    classical_predictor.train(train_df, embedder, test_df)

    # Phase 4: Train Quantum
    logger.info("⚛️ Training quantum model...")
    trainer = QMLTrainer(config=quantum_config)
    results = trainer.train_and_evaluate(
        train_df,
        test_df,
        embedder,
        qml_config=None,  # Use config from file
        classical_model_type=classical_config["model"]["model_type"],
        quantum_config_path=quantum_config["quantum_executor"]["quantum_config_path"],
        config=quantum_config
    )

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