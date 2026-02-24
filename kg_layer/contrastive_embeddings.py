"""
Contrastive Learning for Knowledge Graph Embeddings

Fine-tunes embeddings using contrastive learning to maximize separation
between positive and negative pairs. Uses triplet loss and margin-based
objectives to improve class separability.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class TripletDataset(Dataset):
    """Dataset for triplet loss training."""
    
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray, margin: float = 1.0):
        """
        Args:
            embeddings: Entity embeddings [n_entities, dim]
            labels: Binary labels [n_samples] (1=positive, 0=negative)
            margin: Margin for triplet loss
        """
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = labels
        self.margin = margin
        
        # Create triplets: (anchor, positive, negative)
        self.triplets = self._create_triplets()
        
    def _create_triplets(self):
        """Create triplets for contrastive learning."""
        triplets = []
        pos_indices = np.where(self.labels == 1)[0]
        neg_indices = np.where(self.labels == 0)[0]
        
        # For each positive sample, find a negative sample
        for pos_idx in pos_indices:
            # Random negative sample
            neg_idx = np.random.choice(neg_indices)
            triplets.append((pos_idx, pos_idx, neg_idx))
        
        # Also create negative anchors
        for neg_idx in neg_indices[:len(pos_indices)]:  # Balance
            pos_idx = np.random.choice(pos_indices)
            triplets.append((neg_idx, pos_idx, neg_idx))
        
        return triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        anchor_idx, positive_idx, negative_idx = self.triplets[idx]
        return (
            self.embeddings[anchor_idx],
            self.embeddings[positive_idx],
            self.embeddings[negative_idx]
        )


class TripletLoss(nn.Module):
    """Triplet loss for contrastive learning."""
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings [batch_size, dim]
            positive: Positive embeddings [batch_size, dim]
            negative: Negative embeddings [batch_size, dim]
        
        Returns:
            Loss value
        """
        # Compute distances
        pos_dist = torch.nn.functional.pairwise_distance(anchor, positive, p=2)
        neg_dist = torch.nn.functional.pairwise_distance(anchor, negative, p=2)
        
        # Triplet loss: max(0, margin + pos_dist - neg_dist)
        loss = torch.clamp(self.margin + pos_dist - neg_dist, min=0.0)
        return loss.mean()


class ContrastiveEmbeddingFineTuner:
    """
    Fine-tune embeddings using contrastive learning to improve class separability.
    
    Takes pre-trained embeddings and fine-tunes them using triplet loss to
    maximize distance between positive and negative pairs.
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        learning_rate: float = 0.001,
        num_epochs: int = 50,
        batch_size: int = 64,
        device: Optional[str] = None,
        random_state: int = 42
    ):
        """
        Args:
            margin: Margin for triplet loss (larger = more separation)
            learning_rate: Learning rate for fine-tuning
            num_epochs: Number of fine-tuning epochs
            batch_size: Batch size for training
            device: Device to use ('cuda' or 'cpu')
            random_state: Random seed
        """
        self.margin = margin
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Contrastive fine-tuning: margin={margin}, device={self.device}")
    
    def fine_tune(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        head_indices: np.ndarray,
        tail_indices: np.ndarray
    ) -> np.ndarray:
        """
        Fine-tune embeddings using contrastive learning.
        
        Uses triplet loss on link embeddings (concatenated head+tail) to maximize
        separation between positive and negative pairs.
        
        Args:
            embeddings: Pre-trained entity embeddings [n_entities, dim]
            labels: Binary labels for each sample [n_samples]
            head_indices: Head entity indices [n_samples]
            tail_indices: Tail entity indices [n_samples]
        
        Returns:
            Fine-tuned embeddings [n_entities, dim]
        """
        logger.info(f"Fine-tuning embeddings with contrastive learning...")
        logger.info(f"  Original embeddings: {embeddings.shape}")
        logger.info(f"  Positive samples: {np.sum(labels == 1)}, Negative: {np.sum(labels == 0)}")
        
        # Normalize embeddings
        scaler = StandardScaler()
        embeddings_normalized = scaler.fit_transform(embeddings)
        
        # Create link embeddings (concatenate head and tail)
        # This is what we'll optimize for better separability
        head_embs = embeddings_normalized[head_indices]
        tail_embs = embeddings_normalized[tail_indices]
        link_embeddings = np.concatenate([head_embs, tail_embs], axis=1)
        
        # Create learnable entity embedding layer
        embedding_layer = nn.Embedding(
            num_embeddings=embeddings.shape[0],
            embedding_dim=embeddings.shape[1]
        )
        embedding_layer.weight.data = torch.FloatTensor(embeddings_normalized)
        embedding_layer = embedding_layer.to(self.device)
        
        # Convert indices to tensors
        head_indices_tensor = torch.LongTensor(head_indices).to(self.device)
        tail_indices_tensor = torch.LongTensor(tail_indices).to(self.device)
        labels_tensor = torch.FloatTensor(labels).to(self.device)
        
        # Loss and optimizer
        optimizer = optim.Adam(embedding_layer.parameters(), lr=self.learning_rate)
        
        # Training loop with margin-based loss
        embedding_layer.train()
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            n_batches = 0
            
            # Create batches
            batch_indices = np.arange(len(labels))
            np.random.shuffle(batch_indices)
            
            for i in range(0, len(batch_indices), self.batch_size):
                batch_idx = batch_indices[i:i+self.batch_size]
                if len(batch_idx) == 0:
                    continue
                
                # Get embeddings for this batch
                batch_head_embs = embedding_layer(head_indices_tensor[batch_idx])
                batch_tail_embs = embedding_layer(tail_indices_tensor[batch_idx])
                batch_labels = labels_tensor[batch_idx]
                
                # Create link embeddings (concatenate)
                batch_link_embs = torch.cat([batch_head_embs, batch_tail_embs], dim=1)
                
                # Compute margin-based loss: maximize distance between positive and negative pairs
                pos_mask = batch_labels == 1
                neg_mask = batch_labels == 0
                
                if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                    pos_links = batch_link_embs[pos_mask]
                    neg_links = batch_link_embs[neg_mask]
                    
                    # Within-class distance (should be small)
                    if pos_links.shape[0] > 1:
                        pos_within = torch.nn.functional.pairwise_distance(
                            pos_links[0:1], pos_links[1:min(2, pos_links.shape[0])], p=2
                        ).mean()
                    else:
                        pos_within = torch.tensor(0.0).to(self.device)
                    
                    # Between-class distance (should be large)
                    pos_neg_dist = torch.nn.functional.pairwise_distance(
                        pos_links[:min(10, pos_links.shape[0])],
                        neg_links[:min(10, neg_links.shape[0])],
                        p=2
                    ).mean()
                    
                    # Margin loss: maximize (pos_neg_dist - pos_within)
                    loss = torch.clamp(self.margin - (pos_neg_dist - pos_within), min=0.0)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    n_batches += 1
            
            if (epoch + 1) % 10 == 0 and n_batches > 0:
                avg_loss = total_loss / n_batches
                logger.info(f"  Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.6f}")
        
        # Extract fine-tuned embeddings
        fine_tuned = embedding_layer.weight.data.cpu().numpy()
        
        # Denormalize
        fine_tuned = scaler.inverse_transform(fine_tuned)
        
        logger.info(f"✓ Fine-tuning complete")
        
        return fine_tuned


def compute_contrastive_loss(
    embeddings: np.ndarray,
    head_indices: np.ndarray,
    tail_indices: np.ndarray,
    labels: np.ndarray,
    margin: float = 1.0
) -> float:
    """
    Compute contrastive loss for embeddings.
    
    Args:
        embeddings: Entity embeddings [n_entities, dim]
        head_indices: Head entity indices [n_samples]
        tail_indices: Tail entity indices [n_samples]
        labels: Binary labels [n_samples]
        margin: Margin for contrastive loss
    
    Returns:
        Contrastive loss value
    """
    pos_mask = labels == 1
    neg_mask = labels == 0
    
    # Get head and tail embeddings
    head_embs = embeddings[head_indices]
    tail_embs = embeddings[tail_indices]
    
    # Compute link embeddings (concatenate)
    link_embs = np.concatenate([head_embs, tail_embs], axis=1)
    
    # Positive pairs
    pos_links = link_embs[pos_mask]
    # Negative pairs
    neg_links = link_embs[neg_mask]
    
    if len(pos_links) == 0 or len(neg_links) == 0:
        return 0.0
    
    # Compute distances within positive class
    pos_distances = []
    for i in range(min(100, len(pos_links))):  # Sample for efficiency
        for j in range(i+1, min(100, len(pos_links))):
            pos_distances.append(np.linalg.norm(pos_links[i] - pos_links[j]))
    
    # Compute distances between positive and negative
    pos_neg_distances = []
    for p in pos_links[:min(50, len(pos_links))]:
        for n in neg_links[:min(50, len(neg_links))]:
            pos_neg_distances.append(np.linalg.norm(p - n))
    
    if len(pos_distances) == 0 or len(pos_neg_distances) == 0:
        return 0.0
    
    mean_pos_dist = np.mean(pos_distances)
    mean_pos_neg_dist = np.mean(pos_neg_distances)
    
    # Contrastive loss: maximize pos_neg_dist - pos_dist
    loss = max(0, margin - (mean_pos_neg_dist - mean_pos_dist))
    
    return loss

