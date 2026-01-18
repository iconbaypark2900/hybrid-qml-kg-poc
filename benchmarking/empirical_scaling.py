#!/usr/bin/env python3
"""Empirical runtime measurement and scaling curve fitting"""

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import numpy as np
from datetime import datetime
from scipy.optimize import curve_fit

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

from kg_layer.kg_loader import (
    load_hetionet_edges,
    extract_task_edges,
    prepare_link_prediction_dataset,
)
from kg_layer.kg_embedder import HetionetEmbedder
from classical_baseline.train_baseline import ClassicalLinkPredictor
from quantum_layer.qml_trainer import QMLTrainer

def quadratic(N, a, b, c):
    """Quadratic model: a*N^2 + b*N + c"""
    return a * N**2 + b * N + c

def logarithmic(N, a, b):
    """Logarithmic model: a*log(N) + b"""
    return a * np.log(N) + b

def measure_runtime(N, relation, embedding_dim, qml_dim, random_state):
    """Measure runtime for classical and quantum training."""
    print(f"Measuring N={N}...")
    
    df = load_hetionet_edges()
    task_edges, _, _ = extract_task_edges(
        df, relation_type=relation, max_entities=N
    )
    train_df, test_df = prepare_link_prediction_dataset(
        task_edges, random_state=random_state
    )
    
    embedder = HetionetEmbedder(embedding_dim=embedding_dim, qml_dim=qml_dim)
    if not embedder.load_saved_embeddings():
        embedder.train_embeddings(task_edges)
        embedder.reduce_to_qml_dim()
    
    runtimes = {}
    
    # Classical timing
    try:
        t0 = time.time()
        classical_predictor = ClassicalLinkPredictor(random_state=random_state)
        classical_predictor.train(train_df, embedder, test_df)
        runtimes['classical_train'] = time.time() - t0
        
        # Inference timing
        X_test = embedder.prepare_link_features(test_df, reduced=False)
        y_test = test_df["label"].values
        t0 = time.time()
        _ = classical_predictor.model.predict(X_test[:min(100, len(X_test))])
        runtimes['classical_predict'] = time.time() - t0
    except Exception as e:
        print(f"  Classical failed: {e}")
        runtimes['classical_train'] = None
        runtimes['classical_predict'] = None
    
    # Quantum timing (only for smaller sizes)
    if N <= 2000:
        try:
            qml_config = {
                "model_type": "QSVC",
                "encoding_method": "feature_map",
                "num_qubits": qml_dim,
                "feature_map_type": "ZZ",
                "feature_map_reps": 2,
                "ansatz_type": "RealAmplitudes",
                "ansatz_reps": 2,
                "optimizer": "COBYLA",
                "max_iter": 30,
                "random_state": random_state
            }
            
            t0 = time.time()
            trainer = QMLTrainer(random_state=random_state)
            trainer.train_and_evaluate(train_df, test_df, embedder, qml_config)
            runtimes['quantum_train'] = time.time() - t0
            
            # Inference would require model persistence, skip for now
            runtimes['quantum_predict'] = None
        except Exception as e:
            print(f"  Quantum failed: {e}")
            runtimes['quantum_train'] = None
            runtimes['quantum_predict'] = None
    else:
        runtimes['quantum_train'] = None
        runtimes['quantum_predict'] = None
    
    return runtimes

def main():
    parser = argparse.ArgumentParser(description="Empirical scaling analysis")
    parser.add_argument("--relation", type=str, default="CtD")
    parser.add_argument("--entity_counts", type=int, nargs='+',
                       default=[100, 200, 300, 500, 1000, 2000],
                       help="Entity counts to measure")
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--qml_dim", type=int, default=5)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Measure runtimes
    all_runtimes = {}
    for N in args.entity_counts:
        runtimes = measure_runtime(
            N, args.relation, args.embedding_dim, args.qml_dim, args.random_state
        )
        all_runtimes[N] = runtimes
    
    # Extract data for fitting
    entity_counts = np.array(args.entity_counts)
    classical_times = np.array([
        all_runtimes[N]['classical_train'] for N in args.entity_counts
        if all_runtimes[N]['classical_train'] is not None
    ])
    classical_N = np.array([
        N for N in args.entity_counts
        if all_runtimes[N]['classical_train'] is not None
    ])
    
    quantum_times = np.array([
        all_runtimes[N]['quantum_train'] for N in args.entity_counts
        if all_runtimes[N]['quantum_train'] is not None
    ])
    quantum_N = np.array([
        N for N in args.entity_counts
        if all_runtimes[N]['quantum_train'] is not None
    ])
    
    results = {
        'measurements': all_runtimes,
        'fits': {}
    }
    
    # Fit classical (quadratic)
    if len(classical_N) >= 3:
        try:
            popt_classical, _ = curve_fit(
                quadratic, classical_N, classical_times,
                p0=[1e-6, 1e-3, 0.1]
            )
            results['fits']['classical'] = {
                'model': 'quadratic',
                'params': popt_classical.tolist(),
                'formula': f"{popt_classical[0]:.2e}*N^2 + {popt_classical[1]:.2e}*N + {popt_classical[2]:.2e}"
            }
            print(f"\nClassical fit: {results['fits']['classical']['formula']}")
        except Exception as e:
            print(f"Classical fit failed: {e}")
    
    # Fit quantum (logarithmic)
    if len(quantum_N) >= 3:
        try:
            popt_quantum, _ = curve_fit(
                logarithmic, quantum_N, quantum_times,
                p0=[1.0, 0.1]
            )
            results['fits']['quantum'] = {
                'model': 'logarithmic',
                'params': popt_quantum.tolist(),
                'formula': f"{popt_quantum[0]:.2e}*log(N) + {popt_quantum[1]:.2e}"
            }
            print(f"Quantum fit: {results['fits']['quantum']['formula']}")
        except Exception as e:
            print(f"Quantum fit failed: {e}")
    
    # Extrapolate to larger N
    if 'classical' in results['fits'] and 'quantum' in results['fits']:
        N_large = np.logspace(2, 5, 50)  # 100 to 100,000
        classical_extrapolate = quadratic(N_large, *results['fits']['classical']['params'])
        quantum_extrapolate = logarithmic(N_large, *results['fits']['quantum']['params'])
        
        # Find crossover point
        crossover_idx = np.where(quantum_extrapolate < classical_extrapolate)[0]
        if len(crossover_idx) > 0:
            crossover_N = N_large[crossover_idx[0]]
            results['crossover_point'] = float(crossover_N)
            print(f"\nCrossover point: {crossover_N:.0f} entities")
        else:
            results['crossover_point'] = None
            print("\nNo crossover detected in range")
    
    # Plot
    if MATPLOTLIB_AVAILABLE and classical_N.size > 0:
        try:
            plt.figure(figsize=(12, 6))
            
            # Measured points
            plt.loglog(classical_N, classical_times, 'o', label='Classical (measured)', markersize=8)
            if quantum_N.size > 0:
                plt.loglog(quantum_N, quantum_times, 's', label='Quantum (measured)', markersize=8)
            
            # Extrapolated curves
            if 'classical' in results['fits']:
                N_extrap = np.logspace(np.log10(classical_N.min()), 5, 100)
                classical_extrap = quadratic(N_extrap, *results['fits']['classical']['params'])
                plt.loglog(N_extrap, classical_extrap, '--', label='Classical (extrapolated)', alpha=0.7)
            
            if 'quantum' in results['fits'] and quantum_N.size > 0:
                N_extrap = np.logspace(np.log10(quantum_N.min()), 5, 100)
                quantum_extrap = logarithmic(N_extrap, *results['fits']['quantum']['params'])
                plt.loglog(N_extrap, quantum_extrap, '--', label='Quantum (extrapolated)', alpha=0.7)
            
            # Crossover point
            if 'crossover_point' in results and results['crossover_point']:
                crossover_time = logarithmic(results['crossover_point'], *results['fits']['quantum']['params'])
                plt.plot(results['crossover_point'], crossover_time, 'r*', markersize=15, label='Crossover')
            
            plt.xlabel('Number of Entities', fontsize=12)
            plt.ylabel('Training Time (seconds)', fontsize=12)
            plt.title('Empirical Scaling: Classical vs Quantum', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            plot_path = os.path.join(args.results_dir, f"empirical_scaling_{stamp}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\n✅ Saved plot → {plot_path}")
        except Exception as e:
            print(f"Plotting failed: {e}")
    
    # Save results
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(args.results_dir, f"empirical_scaling_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump({
            'args': vars(args),
            'results': results,
            'timestamp': stamp
        }, f, indent=2)
    print(f"✅ Saved results → {json_path}")

if __name__ == "__main__":
    main()

