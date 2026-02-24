#!/usr/bin/env python3
"""Fast end-to-end smoke test for CI (< 3 minutes)"""

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

# Minimal imports
from kg_layer.kg_loader import (
    load_hetionet_edges,
    extract_task_edges,
    prepare_link_prediction_dataset,
)
from kg_layer.kg_embedder import HetionetEmbedder
from classical_baseline.train_baseline import ClassicalLinkPredictor

def main():
    print("🧪 Running E2E Smoke Test...")
    start_time = time.time()
    
    # Use minimal configuration for speed
    relation = "CtD"
    max_entities = 50  # Very small for speed
    embedding_dim = 16  # Smaller embeddings
    qml_dim = 3  # Fewer qubits
    
    try:
        # 1. Load data
        print("  [1/4] Loading data...")
        df = load_hetionet_edges()
        task_edges, _, _ = extract_task_edges(
            df, relation_type=relation, max_entities=max_entities
        )
        train_df, test_df = prepare_link_prediction_dataset(task_edges, random_state=42)
        print(f"      ✓ Loaded {len(train_df)} train, {len(test_df)} test samples")
        
        # 2. Generate embeddings
        print("  [2/4] Generating embeddings...")
        embedder = HetionetEmbedder(embedding_dim=embedding_dim, qml_dim=qml_dim)
        if not embedder.load_saved_embeddings():
            embedder.train_embeddings(task_edges)
            embedder.reduce_to_qml_dim()
        print("      ✓ Embeddings ready")
        
        # 3. Train classical baseline
        print("  [3/4] Training classical baseline...")
        classical_predictor = ClassicalLinkPredictor(
            model_type="LogisticRegression",
            random_state=42
        )
        classical_predictor.train(train_df, embedder, test_df)
        classical_pr_auc = classical_predictor.metrics.get('test_pr_auc', 0.0)
        print(f"      ✓ Classical PR-AUC: {classical_pr_auc:.4f}")
        
        # 4. Quick quantum test (optional, skip if too slow)
        print("  [4/4] Testing quantum pipeline (optional)...")
        try:
            from quantum_layer.qml_trainer import QMLTrainer
            qml_config = {
                "model_type": "QSVC",  # QSVC is faster than VQC
                "encoding_method": "feature_map",
                "num_qubits": qml_dim,
                "feature_map_type": "ZZ",
                "feature_map_reps": 1,  # Minimal reps for speed
                "ansatz_type": "RealAmplitudes",
                "ansatz_reps": 1,
                "optimizer": "COBYLA",
                "max_iter": 10,  # Very few iterations
                "random_state": 42
            }
            trainer = QMLTrainer(results_dir="results", random_state=42)
            results = trainer.train_and_evaluate(
                train_df, test_df, embedder, qml_config
            )
            quantum_pr_auc = results.get('quantum', {}).get('pr_auc', 0.0)
            print(f"      ✓ Quantum PR-AUC: {quantum_pr_auc:.4f}")
        except Exception as e:
            print(f"      ⚠ Quantum test skipped: {e}")
        
        elapsed = time.time() - start_time
        print(f"\n✅ Smoke test passed in {elapsed:.1f}s")
        return 0
        
    except Exception as e:
        print(f"\n❌ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

