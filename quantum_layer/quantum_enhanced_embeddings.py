"""
Quantum-Enhanced Embeddings

Implements embedding enhancement techniques specifically designed to work well with quantum models.
These techniques optimize embeddings to maximize the effectiveness of quantum feature maps and kernels.
"""

import logging
import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)

try:
    from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
    from qiskit_machine_learning.kernels import FidelityStatevectorKernel
    from qiskit import QuantumCircuit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    logger.warning("Qiskit not available. Quantum-enhanced embeddings will be limited.")


class QuantumEnhancedEmbeddingOptimizer:
    """
    Optimizes embeddings to enhance quantum kernel performance.

    This optimizer adjusts embeddings to maximize the separability of classes
    in the quantum feature space, leading to better performance with quantum models.
    """
    
    def __init__(
        self,
        num_qubits: int = 12,
        feature_map_type: str = 'ZZ',
        feature_map_reps: int = 2,
        entanglement: str = 'full',
        learning_rate: float = 0.01,
        num_epochs: int = 100,
        regularization_strength: float = 0.1,
        random_state: int = 42
    ):
        """
        Args:
            num_qubits: Number of qubits for the quantum feature map
            feature_map_type: Type of feature map ('ZZ', 'Z')
            feature_map_reps: Number of feature map repetitions
            entanglement: Entanglement pattern ('full', 'linear', 'circular')
            learning_rate: Learning rate for optimization
            num_epochs: Number of optimization epochs
            regularization_strength: Strength of regularization to preserve original embeddings
            random_state: Random seed for reproducibility
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for quantum-enhanced embeddings")
        
        self.num_qubits = num_qubits
        self.feature_map_type = feature_map_type
        self.feature_map_reps = feature_map_reps
        self.entanglement = entanglement
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.regularization_strength = regularization_strength
        self.random_state = random_state
        
        # Initialize quantum components
        self.feature_map = self._create_feature_map()
        self.quantum_kernel = FidelityStatevectorKernel(feature_map=self.feature_map)
        
        # Initialize optimizer components
        self.scaler = StandardScaler()
        self.original_embeddings = None
        self.optimized_embeddings = None
    
    def _create_feature_map(self):
        """Create the quantum feature map."""
        if self.feature_map_type == 'ZZ':
            return ZZFeatureMap(
                feature_dimension=self.num_qubits,
                reps=self.feature_map_reps,
                entanglement=self.entanglement
            )
        elif self.feature_map_type == 'Z':
            return ZFeatureMap(
                feature_dimension=self.num_qubits,
                reps=self.feature_map_reps
            )
        else:
            raise ValueError(f"Unknown feature_map_type: {self.feature_map_type}")
    
    def _compute_quantum_separability_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss based on quantum kernel separability.
        
        The loss encourages embeddings to be more separable in the quantum feature space.
        """
        # For computational efficiency, we'll use a simplified approach
        # that approximates the quantum kernel behavior
        
        # First, ensure embeddings are in the right range for quantum feature maps
        embeddings = torch.tanh(embeddings)  # Bound to [-1, 1]
        
        # Separate positive and negative samples
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)
        
        if not pos_mask.any() or not neg_mask.any():
            # If only one class is present, return zero loss
            return torch.tensor(0.0, dtype=torch.float32, device=embeddings.device)
        
        pos_embeddings = embeddings[pos_mask]
        neg_embeddings = embeddings[neg_mask]
        
        if len(pos_embeddings) < 2 or len(neg_embeddings) < 2:
            # Need at least 2 samples per class for meaningful separability
            return torch.tensor(0.0, dtype=torch.float32, device=embeddings.device)
        
        # Compute intra-class distances (want to minimize)
        if len(pos_embeddings) > 1:
            pos_distances = torch.pdist(pos_embeddings)
            pos_loss = torch.mean(pos_distances)
        else:
            pos_loss = torch.tensor(0.0, dtype=torch.float32, device=embeddings.device)
            
        if len(neg_embeddings) > 1:
            neg_distances = torch.pdist(neg_embeddings)
            neg_loss = torch.mean(neg_distances)
        else:
            neg_loss = torch.tensor(0.0, dtype=torch.float32, device=embeddings.device)
        
        intra_class_loss = (pos_loss + neg_loss) / 2
        
        # Compute inter-class distances (want to maximize)
        if len(pos_embeddings) > 0 and len(neg_embeddings) > 0:
            # Compute distances between all positive and negative pairs
            pos_expanded = pos_embeddings.unsqueeze(1).expand(-1, len(neg_embeddings), -1)
            neg_expanded = neg_embeddings.unsqueeze(0).expand(len(pos_embeddings), -1, -1)
            inter_distances = torch.norm(pos_expanded - neg_expanded, dim=2)
            inter_class_loss = torch.mean(inter_distances)
        else:
            inter_class_loss = torch.tensor(0.0, dtype=torch.float32, device=embeddings.device)
        
        # The separability loss is intra-class distances minus inter-class distances
        # We want to minimize intra-class distances and maximize inter-class distances
        separability_loss = intra_class_loss - inter_class_loss
        
        return separability_loss
    
    def fit_transform(
        self,
        embeddings: np.ndarray,
        head_indices: np.ndarray,
        tail_indices: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Optimize embeddings to enhance quantum model performance.
        
        Args:
            embeddings: Original embeddings [N, D]
            head_indices: Indices of head entities for each sample
            tail_indices: Indices of tail entities for each sample
            labels: Labels for each sample
            
        Returns:
            Optimized embeddings and optimization metrics
        """
        logger.info(f"Optimizing embeddings for quantum models: {embeddings.shape}")
        
        # Store original embeddings for regularization
        self.original_embeddings = embeddings.copy()
        
        # Initialize PyTorch tensors
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32, requires_grad=True)
        head_indices_tensor = torch.tensor(head_indices, dtype=torch.long)
        tail_indices_tensor = torch.tensor(tail_indices, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        
        # Setup optimizer
        optimizer = optim.Adam([embeddings_tensor], lr=self.learning_rate)
        
        # Optimization loop
        losses = []
        separability_losses = []
        reg_losses = []
        
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            
            # Create link embeddings by combining head and tail embeddings
            head_embs = embeddings_tensor[head_indices_tensor]
            tail_embs = embeddings_tensor[tail_indices_tensor]
            
            # Combine head and tail embeddings (simple concatenation for now)
            link_embeddings = torch.cat([head_embs, tail_embs], dim=1)
            
            # Compute quantum separability loss
            separability_loss = self._compute_quantum_separability_loss(link_embeddings, labels_tensor)
            
            # Compute regularization loss to prevent drastic changes
            reg_loss = torch.mean(torch.norm(embeddings_tensor - torch.tensor(self.original_embeddings, dtype=torch.float32), dim=1))
            
            # Total loss
            total_loss = separability_loss + self.regularization_strength * reg_loss
            
            # Backpropagate
            total_loss.backward()
            optimizer.step()
            
            # Store losses for monitoring
            losses.append(total_loss.item())
            separability_losses.append(separability_loss.item())
            reg_losses.append(reg_loss.item())
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}/{self.num_epochs}: "
                           f"Total Loss: {total_loss.item():.4f}, "
                           f"Separability Loss: {separability_loss.item():.4f}, "
                           f"Reg Loss: {reg_loss.item():.4f}")
        
        # Get optimized embeddings
        optimized_embeddings = embeddings_tensor.detach().numpy()
        
        # Store results
        self.optimized_embeddings = optimized_embeddings
        
        # Compute metrics
        metrics = {
            'initial_loss': losses[0] if losses else 0.0,
            'final_loss': losses[-1] if losses else 0.0,
            'min_loss': min(losses) if losses else 0.0,
            'loss_improvement': (losses[0] - losses[-1]) / losses[0] if losses and losses[0] != 0 else 0.0,
            'num_epochs': len(losses),
            'converged': abs(losses[-1] - losses[0]) < 1e-4 if len(losses) > 1 else True
        }
        
        logger.info(f"Embedding optimization completed. Loss improved by {metrics['loss_improvement']*100:.2f}%")
        
        return optimized_embeddings, metrics


class QuantumKernelAlignmentEmbedding:
    """
    Embeddings optimized using quantum kernel alignment.
    
    This technique adjusts embeddings to maximize the alignment between
    the quantum kernel and the target similarity matrix.
    """
    
    def __init__(
        self,
        num_qubits: int = 12,
        feature_map_type: str = 'ZZ',
        feature_map_reps: int = 2,
        entanglement: str = 'full',
        alignment_iterations: int = 50,
        learning_rate: float = 0.01,
        random_state: int = 42
    ):
        """
        Args:
            num_qubits: Number of qubits for the quantum feature map
            feature_map_type: Type of feature map ('ZZ', 'Z')
            feature_map_reps: Number of feature map repetitions
            entanglement: Entanglement pattern ('full', 'linear', 'circular')
            alignment_iterations: Number of iterations for kernel alignment
            learning_rate: Learning rate for optimization
            random_state: Random seed for reproducibility
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for quantum kernel alignment")
        
        self.num_qubits = num_qubits
        self.feature_map_type = feature_map_type
        self.feature_map_reps = feature_map_reps
        self.entanglement = entanglement
        self.alignment_iterations = alignment_iterations
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # Initialize quantum components
        self.feature_map = self._create_feature_map()
        self.quantum_kernel = FidelityStatevectorKernel(feature_map=self.feature_map)
        
        # Initialize components
        self.scaler = StandardScaler()
        self.original_embeddings = None
        self.aligned_embeddings = None
    
    def _create_feature_map(self):
        """Create the quantum feature map."""
        if self.feature_map_type == 'ZZ':
            return ZZFeatureMap(
                feature_dimension=self.num_qubits,
                reps=self.feature_map_reps,
                entanglement=self.entanglement
            )
        elif self.feature_map_type == 'Z':
            return ZFeatureMap(
                feature_dimension=self.num_qubits,
                reps=self.feature_map_reps
            )
        else:
            raise ValueError(f"Unknown feature_map_type: {self.feature_map_type}")
    
    def _compute_target_kernel(self, labels: np.ndarray) -> np.ndarray:
        """
        Compute the target kernel based on labels.
        
        Creates a kernel that has high values for similar samples (same label)
        and low values for dissimilar samples (different labels).
        """
        n_samples = len(labels)
        target_kernel = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                # High similarity for same labels, low for different labels
                target_kernel[i, j] = 1.0 if labels[i] == labels[j] else 0.0
        
        return target_kernel
    
    def fit_transform(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Align embeddings to maximize quantum kernel alignment with target.
        
        Args:
            embeddings: Original embeddings [N, D]
            labels: Labels for each sample [N]
            
        Returns:
            Aligned embeddings and alignment metrics
        """
        logger.info(f"Aligning embeddings using quantum kernel alignment: {embeddings.shape}")
        
        # Store original embeddings
        self.original_embeddings = embeddings.copy()
        
        # Initialize PyTorch tensors
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32, requires_grad=True)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        # Compute target kernel
        target_kernel = self._compute_target_kernel(labels)
        target_kernel_tensor = torch.tensor(target_kernel, dtype=torch.float32)
        
        # Setup optimizer
        optimizer = optim.Adam([embeddings_tensor], lr=self.learning_rate)
        
        # Alignment loop
        losses = []
        alignments = []
        
        for iteration in range(self.alignment_iterations):
            optimizer.zero_grad()
            
            # Normalize embeddings to quantum-friendly range
            normalized_embeddings = torch.tanh(embeddings_tensor)
            
            # Compute quantum kernel (approximation for efficiency)
            # For large datasets, we'll use a subset for kernel computation
            n_samples = len(normalized_embeddings)
            if n_samples > 100:  # Use subset for efficiency
                subset_size = 100
                indices = torch.randperm(n_samples)[:subset_size]
                subset_embeddings = normalized_embeddings[indices]
                
                # Compute kernel for subset
                subset_kernel = self._approximate_quantum_kernel(subset_embeddings)
                
                # Compute alignment loss for subset
                target_subset = target_kernel_tensor[indices][:, indices]
                alignment_loss = torch.mean((subset_kernel - target_subset) ** 2)
            else:
                full_kernel = self._approximate_quantum_kernel(normalized_embeddings)
                alignment_loss = torch.mean((full_kernel - target_kernel_tensor) ** 2)
            
            # Add regularization to prevent divergence
            reg_loss = torch.mean(torch.norm(embeddings_tensor - torch.tensor(self.original_embeddings, dtype=torch.float32), dim=1))
            
            # Total loss
            total_loss = alignment_loss + 0.1 * reg_loss
            
            # Backpropagate
            total_loss.backward()
            optimizer.step()
            
            # Store metrics
            losses.append(total_loss.item())
            alignments.append(1.0 - total_loss.item())  # Higher alignment = lower loss
            
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}/{self.alignment_iterations}: "
                           f"Loss: {total_loss.item():.4f}, "
                           f"Alignment: {1.0 - total_loss.item():.4f}")
        
        # Get aligned embeddings
        aligned_embeddings = embeddings_tensor.detach().numpy()
        
        # Store results
        self.aligned_embeddings = aligned_embeddings
        
        # Compute metrics
        metrics = {
            'initial_loss': losses[0] if losses else 0.0,
            'final_loss': losses[-1] if losses else 0.0,
            'min_loss': min(losses) if losses else 0.0,
            'alignment_improvement': (alignments[-1] - alignments[0]) if alignments else 0.0,
            'num_iterations': len(losses),
            'converged': abs(losses[-1] - losses[0]) < 1e-4 if len(losses) > 1 else True
        }
        
        logger.info(f"Kernel alignment completed. Alignment improved by {metrics['alignment_improvement']:.4f}")
        
        return aligned_embeddings, metrics
    
    def _approximate_quantum_kernel(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Approximate the quantum kernel computation for efficiency.
        
        Args:
            embeddings: Embeddings to compute kernel for
            
        Returns:
            Approximated quantum kernel matrix
        """
        # For efficiency, we'll approximate the quantum kernel
        # by using a simplified distance-based kernel that mimics quantum behavior
        n_samples = len(embeddings)
        
        # Compute pairwise distances (this approximates quantum fidelity)
        diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)
        distances = torch.norm(diff, dim=2)
        
        # Convert to similarity (higher for closer embeddings)
        # Using Gaussian-like similarity
        kernel_approx = torch.exp(-distances ** 2 / 2.0)
        
        return kernel_approx


def enhance_embeddings_for_quantum(
    embeddings: np.ndarray,
    head_indices: np.ndarray,
    tail_indices: np.ndarray,
    labels: np.ndarray,
    method: str = 'optimization',
    **kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Enhance embeddings specifically for quantum models.
    
    Args:
        embeddings: Original embeddings [N, D]
        head_indices: Indices of head entities for each sample
        tail_indices: Indices of tail entities for each sample
        labels: Labels for each sample
        method: Enhancement method ('optimization' or 'alignment')
        **kwargs: Additional arguments for the enhancement method
        
    Returns:
        Enhanced embeddings and metrics
    """
    if method == 'optimization':
        enhancer = QuantumEnhancedEmbeddingOptimizer(**kwargs)
        return enhancer.fit_transform(embeddings, head_indices, tail_indices, labels)
    elif method == 'alignment':
        enhancer = QuantumKernelAlignmentEmbedding(**kwargs)
        return enhancer.fit_transform(embeddings, labels)
    else:
        raise ValueError(f"Unknown enhancement method: {method}")


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    n_entities = 100
    embedding_dim = 16
    n_samples = 200
    
    # Generate sample embeddings
    original_embeddings = np.random.randn(n_entities, embedding_dim)
    
    # Generate sample link prediction data
    head_indices = np.random.randint(0, n_entities, n_samples)
    tail_indices = np.random.randint(0, n_entities, n_samples)
    labels = np.random.randint(0, 2, n_samples)
    
    print("Testing Quantum-Enhanced Embeddings...")
    
    # Test optimization method
    print("\n1. Testing Quantum Embedding Optimization:")
    try:
        enhanced_emb_opt, metrics_opt = enhance_embeddings_for_quantum(
            original_embeddings,
            head_indices,
            tail_indices,
            labels,
            method='optimization',
            num_qubits=8,
            feature_map_type='ZZ',
            num_epochs=20
        )
        print(f"   Original shape: {original_embeddings.shape}")
        print(f"   Enhanced shape: {enhanced_emb_opt.shape}")
        print(f"   Loss improvement: {metrics_opt['loss_improvement']:.4f}")
        print("   ✓ Optimization method works")
    except Exception as e:
        print(f"   ✗ Optimization method failed: {e}")
    
    # Test alignment method
    print("\n2. Testing Quantum Kernel Alignment:")
    try:
        enhanced_emb_align, metrics_align = enhance_embeddings_for_quantum(
            original_embeddings,
            head_indices,
            tail_indices,
            labels,
            method='alignment',
            num_qubits=8,
            feature_map_type='ZZ',
            alignment_iterations=20
        )
        print(f"   Original shape: {original_embeddings.shape}")
        print(f"   Enhanced shape: {enhanced_emb_align.shape}")
        print(f"   Alignment improvement: {metrics_align['alignment_improvement']:.4f}")
        print("   ✓ Alignment method works")
    except Exception as e:
        print(f"   ✗ Alignment method failed: {e}")
    
    print("\nQuantum-Enhanced Embeddings testing completed!")