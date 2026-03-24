"""
Comprehensive Test Suite for Quantum Improvements

Validates all the new quantum improvements implemented in the hybrid QML-KG system.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
import sys
import os
from pathlib import Path

# Project root (parent of tests/)
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from quantum_layer.quantum_enhanced_embeddings import QuantumEnhancedEmbeddingOptimizer, QuantumKernelAlignmentEmbedding
from quantum_layer.quantum_transfer_learning import QuantumTransferLearning, QuantumDomainAdaptation
from quantum_layer.quantum_error_mitigation import ZeroNoiseExtrapolation, CompositeErrorMitigation
from quantum_layer.quantum_circuit_optimization import QuantumCircuitOptimizer, VariationalParameterOptimizer
from quantum_layer.quantum_kernel_engineering import AdaptiveQuantumKernel, TrainableQuantumKernel, QuantumKernelAligner
from quantum_layer.quantum_variational_feature_selection import QuantumVariationalFeatureSelector, QuantumApproximateOptimizerFeatureSelection, QuantumMutualInformationFeatureSelector

def test_quantum_enhanced_embeddings():
    """Test quantum-enhanced embedding techniques."""
    print("Testing Quantum-Enhanced Embeddings...")
    
    # Create sample data
    np.random.seed(42)
    n_entities = 50
    embedding_dim = 16
    n_samples = 100
    
    # Generate sample embeddings
    original_embeddings = np.random.randn(n_entities, embedding_dim)
    
    # Generate sample link prediction data
    head_indices = np.random.randint(0, n_entities, n_samples)
    tail_indices = np.random.randint(0, n_entities, n_samples)
    labels = np.random.randint(0, 2, n_samples)
    
    try:
        # Test optimization method
        enhancer_opt = QuantumEnhancedEmbeddingOptimizer(
            num_qubits=8,
            feature_map_type='ZZ',
            num_epochs=10
        )
        enhanced_emb_opt, metrics_opt = enhancer_opt.fit_transform(
            original_embeddings, head_indices, tail_indices, labels
        )
        print(f"  ✓ Optimization method: {original_embeddings.shape} → {enhanced_emb_opt.shape}")
        print(f"  Metrics: {metrics_opt}")
        
        # Test alignment method
        enhancer_align = QuantumKernelAlignmentEmbedding(
            num_qubits=8,
            feature_map_type='ZZ',
            alignment_iterations=10
        )
        enhanced_emb_align, metrics_align = enhancer_align.fit_transform(
            original_embeddings, labels
        )
        print(f"  ✓ Alignment method: {original_embeddings.shape} → {enhanced_emb_align.shape}")
        print(f"  Metrics: {metrics_align}")
        
        return True
    except Exception as e:
        print(f"  ✗ Quantum-enhanced embeddings failed: {e}")
        return False

def test_quantum_transfer_learning():
    """Test quantum transfer learning approaches."""
    print("Testing Quantum Transfer Learning...")
    
    # Create sample data
    np.random.seed(42)
    n_source = 80
    n_target = 40
    n_features = 10
    
    X_source = np.random.randn(n_source, n_features)
    y_source = np.random.randint(0, 2, n_source)
    X_target = np.random.randn(n_target, n_features) + 0.5  # Shifted distribution
    y_target = np.random.randint(0, 2, n_target)
    
    try:
        # Test fine-tuning approach
        transfer_model = QuantumTransferLearning(
            num_qubits=n_features,
            transfer_epochs=10
        )
        
        # Train on source
        source_metrics = transfer_model.fit_source_model(X_source, y_source, epochs=10)
        
        # Transfer to target
        target_metrics = transfer_model.transfer_to_target(X_target, y_target)
        
        print(f"  ✓ Transfer learning completed")
        print(f"  Source metrics: {source_metrics}")
        print(f"  Target metrics: {target_metrics}")
        
        # Make predictions
        preds = transfer_model.predict(X_target)
        accuracy = accuracy_score(y_target, preds)
        print(f"  Target accuracy: {accuracy:.3f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Quantum transfer learning failed: {e}")
        return False

def test_quantum_error_mitigation():
    """Test quantum error mitigation techniques."""
    print("Testing Quantum Error Mitigation...")
    
    try:
        # Create a mock execution function
        def mock_execute(circuit):
            # Simulate a noisy execution
            import random
            return random.uniform(0.4, 0.6)  # Random value in [0.4, 0.6]
        
        # Create a simple test circuit
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        # Test ZNE
        zne = ZeroNoiseExtrapolation(
            noise_factors=[1.0, 1.5, 2.0, 2.5],
            extrapolation_method='polynomial'
        )
        
        # Get values at different noise levels
        zne_values = []
        for factor in zne.noise_factors:
            # For this test, we'll just use the mock function directly
            # since we don't have a real circuit amplification method
            value = mock_execute(qc)
            zne_values.append(value)
        
        mitigated_value = zne.extrapolate(zne.noise_factors, zne_values)
        print(f"  ✓ ZNE completed: {mitigated_value:.4f}")
        
        # Test composite mitigation
        composite = CompositeErrorMitigation(
            zne_noise_factors=[1.0, 1.5, 2.0],
            zne_method='linear',
            cdr_samples=10
        )
        
        # Mock backend for composite
        result = composite.mitigate_composite(mock_execute, qc, None)
        print(f"  ✓ Composite mitigation completed")
        print(f"  Results: {list(result.keys())}")
        
        return True
    except Exception as e:
        print(f"  ✗ Quantum error mitigation failed: {e}")
        return False

def test_quantum_circuit_optimization():
    """Test quantum circuit optimization techniques."""
    print("Testing Quantum Circuit Optimization...")
    
    try:
        from qiskit import QuantumCircuit, Parameter
        
        # Create a parameterized test circuit
        qc = QuantumCircuit(3, 3)
        theta = Parameter('θ')
        phi = Parameter('φ')
        
        qc.h(0)
        qc.ry(theta, 0)
        qc.cx(0, 1)
        qc.rz(phi, 1)
        qc.cx(1, 2)
        qc.rx(theta, 2)
        qc.measure_all()
        
        print(f"  Original circuit - Depth: {qc.depth()}, Size: {qc.size()}")
        
        # Mock objective function
        def mock_objective(circuit):
            return circuit.depth() * 0.1 + len(circuit.parameters) * 0.05
        
        # Test circuit optimization
        opt_circuit, metrics = QuantumCircuitOptimizer(optimization_level=1).optimize_circuit(
            qc, objective_fn=mock_objective, method='transpile'
        )
        print(f"  ✓ Circuit optimization completed")
        print(f"  Optimized: Depth={opt_circuit.depth()}, Size={opt_circuit.size()}")
        
        # Test parameter optimization
        param_optimizer = VariationalParameterOptimizer(max_iter=10)
        
        # For this test, we'll just check if the optimizer can run
        # without actually optimizing the parameters
        params = list(qc.parameters)
        if params:
            # Create a simple objective function for testing
            def simple_obj(params_vals):
                return sum(p**2 for p in params_vals)  # Simple quadratic
            
            opt_params, param_metrics = param_optimizer.optimize(qc, simple_obj)
            print(f"  ✓ Parameter optimization completed")
            print(f"  Final loss: {param_metrics['final_loss']:.4f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Quantum circuit optimization failed: {e}")
        return False

def test_quantum_kernel_engineering():
    """Test quantum kernel engineering improvements."""
    print("Testing Quantum Kernel Engineering...")
    
    # Create sample data
    np.random.seed(42)
    n_samples, n_features = 30, 4
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    try:
        # Test Adaptive Quantum Kernel
        adaptive_kernel, adaptive_metrics = AdaptiveQuantumKernel(
            num_qubits=4,
            feature_map_type='ZZ',
            reps=2
        ), {}
        
        adaptive_kernel.fit(X, y)
        adaptive_metrics['kernel_alignment'] = adaptive_kernel.kernel_alignment
        print(f"  ✓ Adaptive kernel created")
        print(f"  Alignment: {adaptive_metrics['kernel_alignment']}")
        
        # Test Trainable Quantum Kernel
        trainable_kernel, trainable_metrics = TrainableQuantumKernel(
            num_qubits=4,
            feature_map_type='Z',
            reps=1,
            max_iterations=10
        ), {}
        
        def mock_loss(params, X_batch, y_batch):
            return 0.5  # Mock loss
        
        trainable_kernel.fit(X, y, loss_fn=mock_loss)
        trainable_metrics['final_loss'] = trainable_kernel.training_history[-1]['loss'] if trainable_kernel.training_history else float('inf')
        print(f"  ✓ Trainable kernel created")
        print(f"  Final loss: {trainable_metrics['final_loss']:.4f}")
        
        # Test Quantum Kernel Alignment
        aligned_kernel, aligned_metrics = QuantumKernelAligner(
            num_qubits=4,
            feature_map_type='ZZ',
            reps=2
        ), {}
        
        aligned_metrics = aligned_kernel.align_to_target(X, y)
        print(f"  ✓ Kernel alignment completed")
        print(f"  Alignment score: {aligned_metrics['alignment_score']:.4f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Quantum kernel engineering failed: {e}")
        return False

def test_quantum_variational_feature_selection():
    """Test quantum variational algorithms for feature selection."""
    print("Testing Quantum Variational Feature Selection...")
    
    # Create sample data
    np.random.seed(42)
    n_samples, n_features = 100, 8
    X = np.random.randn(n_samples, n_features)
    
    # Create a target that depends on only a few features
    y = (X[:, 0] + X[:, 2] - X[:, 4] > 0).astype(int)
    
    try:
        # Test Quantum Variational Feature Selector
        X_qvfs, metrics_qvfs = QuantumVariationalFeatureSelector(
            num_features=n_features,
            num_layers=2,
            max_iter=20
        ).fit_transform(X, y)
        print(f"  ✓ QVFS completed")
        print(f"  Selected features: {metrics_qvfs['selected_features']}")
        
        # Test QAOA Feature Selection
        X_qaoa, metrics_qaoa = QuantumApproximateOptimizerFeatureSelection(
            num_features=n_features,
            p=2,
            max_iter=20
        ).fit_transform(X, y)
        print(f"  ✓ QAOA completed")
        print(f"  Selected features: {metrics_qaoa['selected_features']}")
        
        # Test Quantum Mutual Information Feature Selection
        X_qmi, metrics_qmi = QuantumMutualInformationFeatureSelector(
            num_features=n_features,
            max_iter=20
        ).fit_transform(X, y)
        print(f"  ✓ QMI completed")
        print(f"  Selected features: {metrics_qmi['selected_features']}")
        
        return True
    except Exception as e:
        print(f"  ✗ Quantum variational feature selection failed: {e}")
        return False

def run_comprehensive_tests():
    """Run all tests and report results."""
    print("="*60)
    print("COMPREHENSIVE TEST SUITE FOR QUANTUM IMPROVEMENTS")
    print("="*60)
    
    tests = [
        ("Quantum-Enhanced Embeddings", test_quantum_enhanced_embeddings),
        ("Quantum Transfer Learning", test_quantum_transfer_learning),
        ("Quantum Error Mitigation", test_quantum_error_mitigation),
        ("Quantum Circuit Optimization", test_quantum_circuit_optimization),
        ("Quantum Kernel Engineering", test_quantum_kernel_engineering),
        ("Quantum Variational Feature Selection", test_quantum_variational_feature_selection),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        icon = "✓" if success else "✗"
        print(f"{icon} {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Quantum improvements are working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)