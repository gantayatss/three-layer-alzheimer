"""
layer1_hypergraph.py - Layer 1: Dynamic Hypergraph of Patient Trajectories
Implements hypergraph construction with DTW and hypergraph neural network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import logging
from typing import List, Tuple, Dict, Optional, Set

logger = logging.getLogger(__name__)


def compute_dtw_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """
    Compute Dynamic Time Warping distance between two trajectories.
    
    Args:
        traj1: First trajectory
        traj2: Second trajectory
        
    Returns:
        DTW distance
    """
    distance, _ = fastdtw(traj1, traj2, dist=euclidean)
    return distance


def compute_progression_rate(trajectory: np.ndarray, times: np.ndarray) -> float:
    """
    Compute rate of progression for a trajectory.
    
    Args:
        trajectory: Feature trajectory over time
        times: Time points
        
    Returns:
        Progression rate
    """
    if len(trajectory) < 2:
        return 0.0
        
    total_change = np.linalg.norm(trajectory[-1] - trajectory[0])
    total_time = times[-1] - times[0]
    
    if total_time == 0:
        return 0.0
        
    return total_change / total_time


class HypergraphConvolution(nn.Module):
    """
    Hypergraph convolution layer.
    """
    
    def __init__(self, in_dim: int, out_dim: int):
        """
        Initialize hypergraph convolution.
        
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
        """
        super().__init__()
        
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        
    def forward(self, 
                X: torch.Tensor, 
                H: torch.Tensor, 
                W: torch.Tensor,
                D_v: torch.Tensor,
                D_e: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hypergraph convolution.
        
        Args:
            X: Node features [n_nodes, in_dim]
            H: Incidence matrix [n_nodes, n_edges]
            W: Edge weight matrix [n_edges, n_edges]
            D_v: Vertex degree matrix [n_nodes, n_nodes]
            D_e: Edge degree matrix [n_edges, n_edges]
            
        Returns:
            Updated node features [n_nodes, out_dim]
        """
        # Apply linear transformation
        X_transformed = self.linear(X)
        
        # Hypergraph propagation: D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} X
        # D_v and D_e are passed as diagonal matrices, extract diagonals first
        if D_v.dim() == 2:
            d_v = torch.diag(D_v)  # Extract diagonal to 1D
        else:
            d_v = D_v
            
        if D_e.dim() == 2:
            d_e = torch.diag(D_e)  # Extract diagonal to 1D
        else:
            d_e = D_e
        
        # Compute D_v^{-1/2}
        d_v_sqrt_inv = torch.pow(d_v + 1e-10, -0.5)
        D_v_sqrt_inv = torch.diag(d_v_sqrt_inv)
        
        # Compute D_e^{-1}
        d_e_inv = torch.pow(d_e + 1e-10, -1.0)
        D_e_inv = torch.diag(d_e_inv)
        
        # Propagation
        out = D_v_sqrt_inv @ H @ W @ D_e_inv @ H.T @ D_v_sqrt_inv @ X_transformed
        
        # Batch normalization
        if out.shape[0] > 1:
            out = self.bn(out)
        
        return F.relu(out)


class HypergraphNN(nn.Module):
    """
    Hypergraph Neural Network with multiple layers.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dims: List[int], 
                 output_dim: int,
                 dropout: float = 0.2):
        """
        Initialize HGNN.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden dimensions
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        dims = [input_dim] + hidden_dims + [output_dim]
        
        self.layers = nn.ModuleList([
            HypergraphConvolution(dims[i], dims[i+1])
            for i in range(len(dims) - 1)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                X: torch.Tensor, 
                H: torch.Tensor, 
                W: torch.Tensor,
                D_v: torch.Tensor,
                D_e: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through HGNN.
        
        Args:
            X: Node features
            H: Incidence matrix
            W: Edge weights
            D_v: Vertex degrees
            D_e: Edge degrees
            
        Returns:
            Node embeddings
        """
        for i, layer in enumerate(self.layers[:-1]):
            X = layer(X, H, W, D_v, D_e)
            X = self.dropout(X)
            
        # Final layer without dropout
        X = self.layers[-1](X, H, W, D_v, D_e)
        
        return X


class SymptomEmergenceModel(nn.Module):
    """
    Temporal point process model for symptom emergence.
    """
    
    def __init__(self, embedding_dim: int, n_symptoms: int = 12):
        """
        Initialize symptom emergence model.
        
        Args:
            embedding_dim: Patient embedding dimension
            n_symptoms: Number of symptoms to model
        """
        super().__init__()
        
        self.n_symptoms = n_symptoms
        
        # Intensity function parameters for each symptom
        self.alpha = nn.Parameter(torch.zeros(n_symptoms))
        self.beta = nn.ModuleList([
            nn.Linear(embedding_dim, 1) for _ in range(n_symptoms)
        ])
        
        # RNN for temporal history
        self.rnn = nn.GRU(n_symptoms, embedding_dim, batch_first=True)
        self.gamma = nn.Linear(embedding_dim, n_symptoms)
        
    def compute_intensity(self, 
                         embedding: torch.Tensor, 
                         history: torch.Tensor) -> torch.Tensor:
        """
        Compute intensity function for symptom emergence.
        
        Args:
            embedding: Patient embedding [batch_size, embedding_dim]
            history: Symptom history [batch_size, seq_len, n_symptoms]
            
        Returns:
            Intensity values [batch_size, n_symptoms]
        """
        batch_size = embedding.shape[0]
        
        # Base intensity
        alpha = self.alpha.unsqueeze(0).expand(batch_size, -1)
        
        # Patient-specific effect
        beta_contrib = torch.zeros(batch_size, self.n_symptoms).to(embedding.device)
        for k in range(self.n_symptoms):
            beta_contrib[:, k] = self.beta[k](embedding).squeeze()
            
        # Temporal history effect
        if history.shape[1] > 0:
            _, h_last = self.rnn(history)
            h_last = h_last.squeeze(0)
            gamma_contrib = self.gamma(h_last)
        else:
            gamma_contrib = torch.zeros(batch_size, self.n_symptoms).to(embedding.device)
            
        # Intensity: exp(alpha + beta^T z + gamma * h)
        intensity = torch.exp(alpha + beta_contrib + gamma_contrib)
        
        return intensity


class Layer1Hypergraph:
    """
    Layer 1: Dynamic Hypergraph of Patient Trajectories.
    """
    
    def __init__(self,
                 epsilon_traj: float = 100.0,  # INCREASED: More lenient DTW threshold
                 epsilon_speed: float = 1.0,    # INCREASED: More lenient speed threshold
                 min_hyperedge_size: int = 2,   # REDUCED: Allow smaller hyperedges
                 max_hyperedge_size: int = 15,  # INCREASED: Allow larger hyperedges
                 hidden_dims: List[int] = [128, 64],
                 sigma: float = 1.0,
                 device: str = 'cpu'):
        """
        Initialize Layer 1.
        
        Args:
            epsilon_traj: DTW distance threshold
            epsilon_speed: Progression speed threshold
            min_hyperedge_size: Minimum hyperedge size
            max_hyperedge_size: Maximum hyperedge size
            hidden_dims: Hidden dimensions for HGNN
            sigma: Hyperedge weight sensitivity
            device: Device to use
        """
        self.epsilon_traj = epsilon_traj
        self.epsilon_speed = epsilon_speed
        self.min_hyperedge_size = min_hyperedge_size
        self.max_hyperedge_size = max_hyperedge_size
        self.hidden_dims = hidden_dims
        self.sigma = sigma
        self.device = device
        
        self.hgnn = None
        self.symptom_model = None
        self.hypergraph_structure = None
        
    def construct_hypergraph(self,
                            sequences: Dict,
                            patient_embeddings: np.ndarray,
                            patient_ids: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct hypergraph structure.
        
        Args:
            sequences: Patient sequences
            patient_embeddings: Patient embeddings from Layer 3
            patient_ids: List of patient IDs
            
        Returns:
            Tuple of (incidence_matrix, edge_weights, progression_rates)
        """
        logger.info("Constructing hypergraph structure...")
        
        n_patients = len(patient_ids)
        
        # Compute DTW distances and progression rates
        logger.info("Computing pairwise DTW distances...")
        dtw_distances = np.zeros((n_patients, n_patients))
        progression_rates = np.zeros(n_patients)
        
        trajectories = {}
        for i, pid in enumerate(patient_ids):
            if pid in sequences:
                seq_data = sequences[pid]
                trajectories[i] = {
                    'features': seq_data['features'],
                    'times': seq_data['time_points']
                }
                progression_rates[i] = compute_progression_rate(
                    seq_data['features'], 
                    np.array(seq_data['time_points'])
                )
            else:
                trajectories[i] = None
                
        # Compute DTW distances (only for patients with sequences)
        valid_pairs = []
        for i in range(n_patients):
            if trajectories[i] is None:
                continue
            for j in range(i+1, n_patients):
                if trajectories[j] is None:
                    continue
                    
                dist = compute_dtw_distance(
                    trajectories[i]['features'],
                    trajectories[j]['features']
                )
                dtw_distances[i, j] = dist
                dtw_distances[j, i] = dist
                
                if dist < self.epsilon_traj:
                    valid_pairs.append((i, j))
                    
        logger.info(f"Found {len(valid_pairs)} similar patient pairs")
        
        # Form hyperedges using greedy approach
        logger.info("Forming hyperedges...")
        hyperedges = []
        
        # Group patients by risk factors (APOE status for example)
        risk_groups = {}
        for i, pid in enumerate(patient_ids):
            # Simplified: use cluster from patient embedding as "risk group"
            risk_group = int(np.argmax(patient_embeddings[i][:10]))  # Top 10 dims
            if risk_group not in risk_groups:
                risk_groups[risk_group] = []
            risk_groups[risk_group].append(i)
            
        # Form hyperedges from similar patients within risk groups
        for risk_group, members in risk_groups.items():
            if len(members) < self.min_hyperedge_size:
                continue
                
            # Find connected components based on similarity
            for seed_idx in members:
                if trajectories[seed_idx] is None:
                    continue
                    
                # Find similar patients
                hyperedge = [seed_idx]
                
                for other_idx in members:
                    if other_idx == seed_idx or trajectories[other_idx] is None:
                        continue
                        
                    # Check all criteria
                    dtw_ok = dtw_distances[seed_idx, other_idx] < self.epsilon_traj
                    speed_ok = abs(progression_rates[seed_idx] - progression_rates[other_idx]) < self.epsilon_speed
                    
                    if dtw_ok and speed_ok and len(hyperedge) < self.max_hyperedge_size:
                        hyperedge.append(other_idx)
                        
                if len(hyperedge) >= self.min_hyperedge_size:
                    hyperedges.append(hyperedge)
                    
        # Remove duplicate hyperedges
        hyperedges_set = set()
        unique_hyperedges = []
        for he in hyperedges:
            he_tuple = tuple(sorted(he))
            if he_tuple not in hyperedges_set:
                hyperedges_set.add(he_tuple)
                unique_hyperedges.append(he)
                
        logger.info(f"Formed {len(unique_hyperedges)} hyperedges")
        
        # Construct incidence matrix
        n_edges = len(unique_hyperedges)
        H = np.zeros((n_patients, n_edges))
        
        for j, hyperedge in enumerate(unique_hyperedges):
            for i in hyperedge:
                H[i, j] = 1
                
        # Compute hyperedge weights based on within-edge variance
        edge_weights = np.zeros(n_edges)
        for j, hyperedge in enumerate(unique_hyperedges):
            if len(hyperedge) > 1:
                embeddings_in_edge = patient_embeddings[hyperedge]
                variance = np.var(embeddings_in_edge, axis=0).mean()
                edge_weights[j] = np.exp(-self.sigma**2 * variance)
            else:
                edge_weights[j] = 1.0
                
        self.hypergraph_structure = {
            'incidence': H,
            'weights': edge_weights,
            'hyperedges': unique_hyperedges,
            'progression_rates': progression_rates
        }
        
        logger.info(f"Hypergraph constructed: {n_patients} nodes, {n_edges} edges")
        
        return H, edge_weights, progression_rates
        
    def train_hgnn(self,
                   node_features: np.ndarray,
                   incidence_matrix: np.ndarray,
                   edge_weights: np.ndarray,
                   n_epochs: int = 50,
                   learning_rate: float = 1e-3) -> None:
        """
        Train Hypergraph Neural Network.
        
        Args:
            node_features: Initial node features
            incidence_matrix: Hypergraph incidence matrix
            edge_weights: Hyperedge weights
            n_epochs: Number of training epochs
            learning_rate: Learning rate
        """
        logger.info("Training Hypergraph Neural Network...")
        
        n_nodes, n_edges = incidence_matrix.shape
        input_dim = node_features.shape[1]
        output_dim = self.hidden_dims[-1]
        
        # Initialize HGNN
        self.hgnn = HypergraphNN(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims[:-1],
            output_dim=output_dim,
            dropout=0.2
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.hgnn.parameters(), lr=learning_rate)
        
        # Convert to tensors (ensure all are float32)
        X = torch.FloatTensor(node_features).to(self.device)
        H = torch.FloatTensor(incidence_matrix).to(self.device)
        edge_weights_tensor = torch.FloatTensor(edge_weights).to(self.device)
        W = torch.diag(edge_weights_tensor).to(self.device)
        
        # Compute degree matrices (ensure float32)
        D_v = torch.diag(torch.sum(H * edge_weights_tensor, dim=1)).float()
        D_e = torch.diag(torch.sum(H, dim=0)).float()
        
        # Training loop (self-supervised: reconstruct features)
        for epoch in range(n_epochs):
            self.hgnn.train()
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = self.hgnn(X, H, W, D_v, D_e)
            
            # Reconstruction loss
            # Use simple MSE to reconstruct original features
            reconstructed = embeddings @ embeddings.T @ X
            loss = F.mse_loss(reconstructed, X)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
                
        logger.info("HGNN training completed.")
        
    def get_embeddings(self, 
                      node_features: np.ndarray,
                      incidence_matrix: np.ndarray,
                      edge_weights: np.ndarray) -> np.ndarray:
        """
        Get node embeddings from trained HGNN.
        
        Args:
            node_features: Node features
            incidence_matrix: Incidence matrix
            edge_weights: Edge weights
            
        Returns:
            Node embeddings
        """
        self.hgnn.eval()
        
        with torch.no_grad():
            X = torch.FloatTensor(node_features).to(self.device)
            H = torch.FloatTensor(incidence_matrix).to(self.device)
            edge_weights_tensor = torch.FloatTensor(edge_weights).to(self.device)
            W = torch.diag(edge_weights_tensor).to(self.device)
            
            D_v = torch.diag(torch.sum(H * edge_weights_tensor, dim=1)).float()
            D_e = torch.diag(torch.sum(H, dim=0)).float()
            
            embeddings = self.hgnn(X, H, W, D_v, D_e)
            
            return embeddings.cpu().numpy()
            
    def fit(self,
            sequences: Dict,
            patient_embeddings: np.ndarray,
            patient_ids: List,
            n_epochs: int = 50) -> np.ndarray:
        """
        Complete Layer 1 pipeline.
        
        Args:
            sequences: Patient sequences
            patient_embeddings: Embeddings from Layer 3
            patient_ids: Patient IDs
            n_epochs: Training epochs
            
        Returns:
            Final node embeddings
        """
        # Construct hypergraph
        H, edge_weights, _ = self.construct_hypergraph(
            sequences, patient_embeddings, patient_ids
        )
        
        # Train HGNN
        self.train_hgnn(patient_embeddings, H, edge_weights, n_epochs)
        
        # Get final embeddings
        embeddings = self.get_embeddings(patient_embeddings, H, edge_weights)
        
        return embeddings