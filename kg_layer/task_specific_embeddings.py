"""
Task-Specific Embedding Fine-Tuning

Fine-tunes embeddings specifically for the CtD classification task using
supervised learning with classification loss. This addresses the core issue
that embeddings trained for link prediction (ranking) don't learn class
boundaries needed for classification.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)


class ClassificationEmbeddingFineTuner:
    """
    Fine-tune embeddings using classification loss on the target task.
    
    Takes pre-trained embeddings and fine-tunes them using a classification
    objective (cross-entropy loss) to learn class boundaries directly.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        num_epochs: int = 100,
        batch_size: int = 64,
        weight_decay: float = 1e-5,
        device: Optional[str] = None,
        random_state: int = 42,
        early_stopping_patience: int = 10
    ):
        """
        Args:
            learning_rate: Learning rate for fine-tuning
            num_epochs: Number of fine-tuning epochs
            batch_size: Batch size for training
            weight_decay: L2 regularization strength
            device: Device to use ('cuda' or 'cpu')
            random_state: Random seed
            early_stopping_patience: Patience for early stopping
        """
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.early_stopping_patience = early_stopping_patience
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Classification fine-tuning: device={self.device}, epochs={num_epochs}")
    
    def fine_tune(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        head_indices: np.ndarray,
        tail_indices: np.ndarray,
        val_head_indices: Optional[np.ndarray] = None,
        val_tail_indices: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Fine-tune embeddings using classification loss.
        
        Args:
            embeddings: Pre-trained entity embeddings [n_entities, dim]
            labels: Binary labels for each sample [n_samples]
            head_indices: Head entity indices [n_samples]
            tail_indices: Tail entity indices [n_samples]
            val_head_indices: Optional validation head indices
            val_tail_indices: Optional validation tail indices
            val_labels: Optional validation labels
        
        Returns:
            Fine-tuned embeddings and training history
        """
        logger.info(f"Fine-tuning embeddings for classification task...")
        logger.info(f"  Original embeddings: {embeddings.shape}")
        logger.info(f"  Training samples: {len(labels)} (pos: {np.sum(labels==1)}, neg: {np.sum(labels==0)})")
        
        # Normalize embeddings
        scaler = StandardScaler()
        embeddings_normalized = scaler.fit_transform(embeddings)
        
        # Create learnable entity embedding layer
        embedding_layer = nn.Embedding(
            num_embeddings=embeddings.shape[0],
            embedding_dim=embeddings.shape[1]
        )
        embedding_layer.weight.data = torch.FloatTensor(embeddings_normalized)
        embedding_layer = embedding_layer.to(self.device)
        
        # Create classification head (maps concatenated head+tail embeddings to binary prediction)
        embedding_dim = embeddings.shape[1]
        classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        # Convert indices to tensors
        head_indices_tensor = torch.LongTensor(head_indices).to(self.device)
        tail_indices_tensor = torch.LongTensor(tail_indices).to(self.device)
        labels_tensor = torch.FloatTensor(labels).to(self.device)
        
        # Validation tensors if provided
        has_validation = val_head_indices is not None and val_tail_indices is not None and val_labels is not None
        if has_validation:
            val_head_tensor = torch.LongTensor(val_head_indices).to(self.device)
            val_tail_tensor = torch.LongTensor(val_tail_indices).to(self.device)
            val_labels_tensor = torch.FloatTensor(val_labels).to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            list(embedding_layer.parameters()) + list(classifier.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Training history
        history = {
            'train_losses': [],
            'train_aucs': [],
            'val_losses': [],
            'val_aucs': [],
            'best_val_auc': 0.0,
            'best_epoch': 0
        }
        
        # Early stopping
        best_val_auc = 0.0
        patience_counter = 0
        best_embedding_state = None
        
        embedding_layer.train()
        classifier.train()
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        
        for epoch in range(self.num_epochs):
            # Training
            total_loss = 0.0
            n_batches = 0
            
            # Create batches
            batch_indices = np.arange(len(labels))
            np.random.shuffle(batch_indices)
            
            for i in range(0, len(batch_indices), self.batch_size):
                batch_idx = batch_indices[i:i+self.batch_size]
                if len(batch_idx) == 0:
                    continue
                
                optimizer.zero_grad()
                
                # Get embeddings for this batch
                batch_head_embs = embedding_layer(head_indices_tensor[batch_idx])
                batch_tail_embs = embedding_layer(tail_indices_tensor[batch_idx])
                batch_labels = labels_tensor[batch_idx]
                
                # Create link embeddings (concatenate)
                batch_link_embs = torch.cat([batch_head_embs, batch_tail_embs], dim=1)
                
                # Forward pass through classifier
                predictions = classifier(batch_link_embs).squeeze()
                
                # Compute loss
                loss = criterion(predictions, batch_labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / max(n_batches, 1)
            
            # Compute training AUC
            with torch.no_grad():
                embedding_layer.eval()
                classifier.eval()
                
                # Full training set predictions
                train_head_embs = embedding_layer(head_indices_tensor)
                train_tail_embs = embedding_layer(tail_indices_tensor)
                train_link_embs = torch.cat([train_head_embs, train_tail_embs], dim=1)
                train_preds = classifier(train_link_embs).squeeze().cpu().numpy()
                train_auc = roc_auc_score(labels, train_preds)
                
                # Validation if available
                val_auc = None
                val_loss = None
                if has_validation:
                    val_head_embs = embedding_layer(val_head_tensor)
                    val_tail_embs = embedding_layer(val_tail_tensor)
                    val_link_embs = torch.cat([val_head_embs, val_tail_embs], dim=1)
                    val_preds = classifier(val_link_embs).squeeze().cpu().numpy()
                    val_auc = roc_auc_score(val_labels, val_preds)
                    val_loss = criterion(
                        torch.FloatTensor(val_preds).to(self.device),
                        val_labels_tensor
                    ).item()
                
                embedding_layer.train()
                classifier.train()
            
            history['train_losses'].append(avg_loss)
            history['train_aucs'].append(train_auc)
            
            if has_validation:
                history['val_losses'].append(val_loss)
                history['val_aucs'].append(val_auc)
                
                # Early stopping
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                    history['best_val_auc'] = best_val_auc
                    history['best_epoch'] = epoch
                    # Save best embedding state
                    best_embedding_state = embedding_layer.weight.data.clone()
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        logger.info(f"  Early stopping at epoch {epoch+1} (patience: {self.early_stopping_patience})")
                        # Restore best embedding state
                        embedding_layer.weight.data = best_embedding_state
                        break
            
            if (epoch + 1) % 10 == 0:
                log_msg = f"  Epoch {epoch+1}/{self.num_epochs}, Train Loss: {avg_loss:.6f}, Train AUC: {train_auc:.4f}"
                if has_validation:
                    log_msg += f", Val Loss: {val_loss:.6f}, Val AUC: {val_auc:.4f}"
                logger.info(log_msg)
        
        # Extract fine-tuned embeddings
        with torch.no_grad():
            fine_tuned = embedding_layer.weight.data.cpu().numpy()
        
        # Denormalize
        fine_tuned = scaler.inverse_transform(fine_tuned)
        
        logger.info(f"✓ Classification fine-tuning complete")
        if has_validation:
            logger.info(f"  Best validation AUC: {best_val_auc:.4f} at epoch {history['best_epoch']+1}")
        
        return fine_tuned, history


def compute_classification_metrics(
    embeddings: np.ndarray,
    head_indices: np.ndarray,
    tail_indices: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Compute classification metrics using embeddings.
    
    Args:
        embeddings: Entity embeddings [n_entities, dim]
        head_indices: Head entity indices [n_samples]
        tail_indices: Tail entity indices [n_samples]
        labels: Binary labels [n_samples]
    
    Returns:
        Dictionary with metrics
    """
    # Create link embeddings
    head_embs = embeddings[head_indices]
    tail_embs = embeddings[tail_indices]
    link_embs = np.concatenate([head_embs, tail_embs], axis=1)
    
    # Train simple classifier
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    link_embs_scaled = scaler.fit_transform(link_embs)
    
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(link_embs_scaled, labels)
    
    # Predictions
    pred_proba = clf.predict_proba(link_embs_scaled)[:, 1]
    
    # Metrics
    auc = roc_auc_score(labels, pred_proba)
    pr_auc = average_precision_score(labels, pred_proba)
    
    return {
        'roc_auc': auc,
        'pr_auc': pr_auc
    }

