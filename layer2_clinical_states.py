"""
layer2_clinical_states.py - Layer 2: Clinical State Attention Network
Implements GRU, multi-head attention, and TCN for temporal state modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances
import logging
from typing import Tuple, List, Optional, Dict

logger = logging.getLogger(__name__)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network with dilated causal convolutions.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 n_layers: int = 3,
                 kernel_size: int = 3,
                 dropout: float = 0.2):
        """
        Initialize TCN.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            n_layers: Number of dilated layers
            kernel_size: Convolution kernel size
            dropout: Dropout rate
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        for i in range(n_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            
            layer = nn.Sequential(
                nn.Conv1d(input_dim if i == 0 else hidden_dim, 
                         hidden_dim, 
                         kernel_size,
                         padding=padding,
                         dilation=dilation),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.layers.append(layer)
            
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TCN.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_dim]
        """
        # Transpose for Conv1d: [batch_size, input_dim, seq_len]
        x = x.transpose(1, 2)
        
        for layer in self.layers:
            x = layer(x)
            # Causal masking: remove future information
            x = x[:, :, :-layer[0].padding[0]] if layer[0].padding[0] > 0 else x
            
        # Transpose back: [batch_size, seq_len, hidden_dim]
        x = x.transpose(1, 2)
        x = self.output_proj(x)
        
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    """
    
    def __init__(self, hidden_dim: int, n_heads: int = 8, dropout: float = 0.1):
        """
        Initialize multi-head attention.
        
        Args:
            hidden_dim: Hidden dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through multi-head attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        # Split into heads
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)
        
        # Final linear projection
        output = self.out_linear(attn_output)
        
        return output


class ClinicalStateEncoder(nn.Module):
    """
    Encodes clinical state sequences using GRU, Attention, and TCN.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 n_heads: int = 8,
                 n_tcn_layers: int = 3,
                 dropout: float = 0.2):
        """
        Initialize clinical state encoder.
        
        Args:
            input_dim: Input feature dimension (symptoms + cognition + functional)
            hidden_dim: Hidden dimension
            n_heads: Number of attention heads
            n_tcn_layers: Number of TCN layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GRU for sequential encoding
        self.gru = nn.GRU(
            hidden_dim, 
            hidden_dim, 
            num_layers=2, 
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
            bidirectional=False
        )
        
        # Multi-head attention
        self.attention = MultiHeadAttention(hidden_dim, n_heads, dropout)
        
        # Temporal Convolutional Network
        self.tcn = TemporalConvNet(hidden_dim, hidden_dim, n_tcn_layers, dropout=dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            x: Input sequences [batch_size, max_seq_len, input_dim]
            lengths: Sequence lengths [batch_size]
            
        Returns:
            Encoded representations [batch_size, max_seq_len, hidden_dim]
        """
        # Project input
        x = self.input_proj(x)
        
        # GRU encoding
        # Pack sequences for efficiency
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        gru_out, _ = self.gru(packed_x)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        
        # Multi-head attention
        attn_out = self.attention(gru_out)
        
        # Temporal convolution
        tcn_out = self.tcn(gru_out)
        
        # Combine via residual connections
        combined = self.layer_norm(gru_out + attn_out + tcn_out)
        
        # Final projection
        output = self.output_proj(combined)
        
        return output


class TransitionModel(nn.Module):
    """
    Models transitions between clinical states.
    """
    
    def __init__(self, n_states: int, embedding_dim: int, hidden_dim: int = 128):
        """
        Initialize transition model.
        
        Args:
            n_states: Number of clinical states (K_2)
            embedding_dim: State embedding dimension
            hidden_dim: Hidden dimension
        """
        super().__init__()
        
        self.n_states = n_states
        
        # Learnable state embeddings
        self.state_embeddings = nn.Embedding(n_states, embedding_dim)
        
        # Transition probability network
        # Input: [current_state_emb, next_state_emb, patient_emb, time_delta]
        input_dim = 2 * embedding_dim + embedding_dim + 1
        
        self.transition_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, 
                current_state: torch.Tensor, 
                next_state: torch.Tensor,
                patient_emb: torch.Tensor,
                time_delta: torch.Tensor) -> torch.Tensor:
        """
        Compute transition probabilities.
        
        Args:
            current_state: Current state indices [batch_size]
            next_state: Next state indices [batch_size]
            patient_emb: Patient embeddings [batch_size, emb_dim]
            time_delta: Time difference [batch_size, 1]
            
        Returns:
            Transition probabilities [batch_size, n_states]
        """
        current_emb = self.state_embeddings(current_state)  # [batch_size, emb_dim]
        next_emb = self.state_embeddings(next_state)  # [batch_size, emb_dim]
        
        # Concatenate features
        features = torch.cat([current_emb, next_emb, patient_emb, time_delta], dim=-1)
        
        # Compute transition score
        logits = self.transition_net(features)
        
        return logits


class Layer2ClinicalStates:
    """
    Layer 2: Clinical State Attention Network with temporal clustering.
    """
    
    def __init__(self,
                 n_states: int = 15,
                 hidden_dim: int = 128,
                 n_heads: int = 8,
                 n_tcn_layers: int = 3,
                 sigma_temporal: float = 1.0,
                 tau_decay: float = 1.0,
                 device: str = 'cpu'):
        """
        Initialize Layer 2.
        
        Args:
            n_states: Number of clinical states (K_2)
            hidden_dim: Hidden dimension
            n_heads: Number of attention heads
            n_tcn_layers: Number of TCN layers
            sigma_temporal: Temporal similarity kernel bandwidth
            tau_decay: Temporal decay parameter
            device: Device to use
        """
        self.n_states = n_states
        self.hidden_dim = hidden_dim
        self.sigma_temporal = sigma_temporal
        self.tau_decay = tau_decay
        self.device = device
        
        self.encoder = None
        self.transition_model = None
        self.state_labels = None
        self.state_centroids = None
        
    def train_encoder(self,
                     sequences: Dict,
                     input_dim: int,
                     n_epochs: int = 50,
                     batch_size: int = 32,
                     learning_rate: float = 1e-3) -> None:
        """
        Train temporal encoder.
        
        Args:
            sequences: Dictionary mapping patient IDs to sequence data
            input_dim: Input feature dimension
            n_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        logger.info(f"Training temporal encoder on {len(sequences)} sequences...")
        
        # Initialize encoder
        self.encoder = ClinicalStateEncoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            n_heads=8,
            n_tcn_layers=3
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=learning_rate)
        
        # Prepare data
        patient_ids = list(sequences.keys())
        
        for epoch in range(n_epochs):
            self.encoder.train()
            epoch_loss = 0.0
            n_batches = 0
            
            # Shuffle patients
            np.random.shuffle(patient_ids)
            
            for i in range(0, len(patient_ids), batch_size):
                batch_ids = patient_ids[i:i+batch_size]
                
                # Prepare batch
                batch_sequences = []
                batch_lengths = []
                
                for pid in batch_ids:
                    seq = sequences[pid]['features']
                    batch_sequences.append(torch.FloatTensor(seq))
                    batch_lengths.append(len(seq))
                    
                # Pad sequences
                max_len = max(batch_lengths)
                padded_sequences = []
                for seq in batch_sequences:
                    if len(seq) < max_len:
                        padding = torch.zeros(max_len - len(seq), seq.shape[1])
                        seq = torch.cat([seq, padding], dim=0)
                    padded_sequences.append(seq)
                    
                X_batch = torch.stack(padded_sequences).to(self.device)
                lengths_batch = torch.LongTensor(batch_lengths)
                
                # Forward pass
                optimizer.zero_grad()
                encoded = self.encoder(X_batch, lengths_batch)
                
                # Prediction loss: predict next state
                # Simple autoregressive loss
                loss = 0.0
                for j, (seq, length) in enumerate(zip(encoded, batch_lengths)):
                    if length > 1:
                        # Predict next state from current
                        # Use actual length to avoid size mismatch with padding
                        predictions = seq[:length-1]  # First (length-1) elements
                        targets = seq[1:length]       # Elements from index 1 to length
                        loss += F.mse_loss(predictions, targets)
                        
                loss = loss / len(batch_ids)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
                
            avg_loss = epoch_loss / n_batches
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
                
        logger.info("Encoder training completed.")
        
    def extract_state_representations(self, sequences: Dict) -> Tuple[np.ndarray, List]:
        """
        Extract state representations from sequences.
        
        Args:
            sequences: Dictionary mapping patient IDs to sequence data
            
        Returns:
            Tuple of (representations, metadata)
        """
        self.encoder.eval()
        
        all_representations = []
        all_metadata = []
        
        with torch.no_grad():
            for pid, data in sequences.items():
                seq = torch.FloatTensor(data['features']).unsqueeze(0).to(self.device)
                length = torch.LongTensor([len(data['features'])])
                
                encoded = self.encoder(seq, length)
                encoded = encoded.squeeze(0).cpu().numpy()
                
                # Store each time point with metadata
                for t in range(len(encoded)):
                    all_representations.append(encoded[t])
                    all_metadata.append({
                        'patient_id': pid,
                        'time_index': t,
                        'time_point': data['time_points'][t] if t < len(data['time_points']) else 0
                    })
                    
        return np.array(all_representations), all_metadata
        
    def perform_temporal_clustering(self, 
                                   representations: np.ndarray,
                                   metadata: List) -> np.ndarray:
        """
        Perform spectral clustering on temporal state representations.
        
        Args:
            representations: State representations [n_visits, hidden_dim]
            metadata: List of metadata dictionaries
            
        Returns:
            State labels for each visit
        """
        logger.info(f"Performing temporal clustering into {self.n_states} states...")
        
        # Construct temporal similarity matrix
        # Spatial similarity
        spatial_dist = pairwise_distances(representations, metric='euclidean')
        spatial_sim = np.exp(-spatial_dist**2 / (2 * self.sigma_temporal**2))
        
        # Temporal decay
        n_visits = len(metadata)
        temporal_decay = np.ones((n_visits, n_visits))
        
        for i in range(n_visits):
            for j in range(n_visits):
                time_diff = abs(metadata[i]['time_point'] - metadata[j]['time_point'])
                temporal_decay[i, j] = np.exp(-time_diff / self.tau_decay)
                
        # Combined affinity
        affinity = spatial_sim * temporal_decay
        
        # Spectral clustering
        clustering = SpectralClustering(
            n_clusters=self.n_states,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42
        )
        
        labels = clustering.fit_predict(affinity)
        
        # Compute state centroids
        centroids = []
        for k in range(self.n_states):
            mask = labels == k
            if mask.sum() > 0:
                centroid = representations[mask].mean(axis=0)
                centroids.append(centroid)
            else:
                centroids.append(np.zeros(representations.shape[1]))
                
        self.state_labels = labels
        self.state_centroids = np.array(centroids)
        
        # Log state sizes
        for k in range(self.n_states):
            count = (labels == k).sum()
            logger.info(f"Clinical State {k}: {count} visits ({count/len(labels)*100:.1f}%)")
            
        return labels
        
    def fit(self,
            sequences: Dict,
            input_dim: int,
            n_epochs: int = 50,
            batch_size: int = 32) -> np.ndarray:
        """
        Complete Layer 2 pipeline.
        
        Args:
            sequences: Sequence data
            input_dim: Input feature dimension
            n_epochs: Training epochs
            batch_size: Batch size
            
        Returns:
            State labels
        """
        # Train encoder
        self.train_encoder(sequences, input_dim, n_epochs, batch_size)
        
        # Extract representations
        representations, metadata = self.extract_state_representations(sequences)
        
        # Perform clustering
        labels = self.perform_temporal_clustering(representations, metadata)
        
        return labels, metadata