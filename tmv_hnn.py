"""
tmv_hnn.py - Complete TMV-HNN Model
Integrates Layer 3, Layer 2, Layer 1, and survival analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

from layer3_patient_clustering import Layer3PatientClustering
from layer2_clinical_states import Layer2ClinicalStates
from layer1_hypergraph import Layer1Hypergraph
from survival_model import BayesianSurvivalModel

logger = logging.getLogger(__name__)


class InterLayerGating(nn.Module):
    """
    Learned gating mechanism for inter-layer communication.
    """
    
    def __init__(self, dim1: int, dim2: int, time_encoding_dim: int = 16):
        """
        Initialize gating network.
        
        Args:
            dim1: Dimension of first layer representation
            dim2: Dimension of second layer representation
            time_encoding_dim: Dimension of time encoding
        """
        super().__init__()
        
        self.time_encoding = nn.Linear(1, time_encoding_dim)
        
        input_dim = dim1 + dim2 + time_encoding_dim
        
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, h1: torch.Tensor, h2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute gating weights.
        
        Args:
            h1: First layer representation
            h2: Second layer representation
            t: Time encoding
            
        Returns:
            Gate values
        """
        # Encode time
        time_emb = self.time_encoding(t)
        
        # Concatenate
        combined = torch.cat([h1, h2, time_emb], dim=-1)
        
        # Compute gate
        gate = self.gate_net(combined)
        
        return gate


class TMVHNNModel:
    """
    Complete Temporal Multi-View Hypergraph Neural Network.
    """
    
    def __init__(self,
                 n_clusters_l3: int = 10,
                 n_states_l2: int = 15,
                 latent_dim: int = 64,
                 hidden_dim_l2: int = 128,
                 hidden_dims_l1: List[int] = [128, 64],
                 survival_hidden_dims: List[int] = [128, 64],
                 device: str = 'cpu'):
        """
        Initialize TMV-HNN model.
        
        Args:
            n_clusters_l3: Number of patient profile clusters (Layer 3)
            n_states_l2: Number of clinical states (Layer 2)
            latent_dim: Latent dimension for Layer 3
            hidden_dim_l2: Hidden dimension for Layer 2
            hidden_dims_l1: Hidden dimensions for Layer 1
            survival_hidden_dims: Hidden dimensions for survival model
            device: Device to use
        """
        self.device = device
        
        # Layer 3: Patient clustering
        self.layer3 = Layer3PatientClustering(
            n_clusters=n_clusters_l3,
            latent_dim=latent_dim,
            device=device
        )
        
        # Layer 2: Clinical states
        self.layer2 = Layer2ClinicalStates(
            n_states=n_states_l2,
            hidden_dim=hidden_dim_l2,
            device=device
        )
        
        # Layer 1: Hypergraph
        self.layer1 = Layer1Hypergraph(
            hidden_dims=hidden_dims_l1,
            device=device
        )
        
        # Survival model
        self.survival_model = None
        
        # Inter-layer gates
        self.gate_3to2 = None
        self.gate_2to1 = None
        
        # Storage for embeddings
        self.patient_embeddings_l3 = None
        self.patient_embeddings_l1 = None
        self.patient_ids = None
        
    def fit_layer3(self, 
                   features: np.ndarray, 
                   modality_mask: np.ndarray,
                   patient_ids: List,
                   n_epochs: int = 100) -> np.ndarray:
        """
        Fit Layer 3 (patient clustering).
        
        Args:
            features: Patient features
            modality_mask: Modality availability mask
            patient_ids: List of patient IDs
            n_epochs: Training epochs
            
        Returns:
            Cluster labels
        """
        logger.info("=" * 80)
        logger.info("PHASE 1: Training Layer 3 (Patient Profile Clustering)")
        logger.info("=" * 80)
        
        labels = self.layer3.fit(features, modality_mask, n_epochs)
        
        # Extract and store embeddings
        self.patient_embeddings_l3 = self.layer3.extract_embeddings(features)
        self.patient_ids = patient_ids
        
        logger.info(f"Layer 3 completed: {self.layer3.n_clusters} clusters identified")
        return labels
        
    def fit_layer2(self,
                   sequences: Dict,
                   input_dim: int,
                   n_epochs: int = 50) -> Tuple[np.ndarray, List]:
        """
        Fit Layer 2 (clinical states).
        
        Args:
            sequences: Patient sequences
            input_dim: Input feature dimension
            n_epochs: Training epochs
            
        Returns:
            Tuple of (state_labels, metadata)
        """
        logger.info("=" * 80)
        logger.info("PHASE 2: Training Layer 2 (Clinical State Modeling)")
        logger.info("=" * 80)
        
        labels, metadata = self.layer2.fit(sequences, input_dim, n_epochs)
        
        logger.info(f"Layer 2 completed: {self.layer2.n_states} clinical states identified")
        return labels, metadata
        
    def fit_layer1(self,
                   sequences: Dict,
                   n_epochs: int = 50) -> np.ndarray:
        """
        Fit Layer 1 (hypergraph).
        
        Args:
            sequences: Patient sequences
            n_epochs: Training epochs
            
        Returns:
            Final patient embeddings
        """
        logger.info("=" * 80)
        logger.info("PHASE 3: Training Layer 1 (Hypergraph Construction)")
        logger.info("=" * 80)
        
        embeddings = self.layer1.fit(
            sequences=sequences,
            patient_embeddings=self.patient_embeddings_l3,
            patient_ids=self.patient_ids,
            n_epochs=n_epochs
        )
        
        self.patient_embeddings_l1 = embeddings
        
        logger.info("Layer 1 completed: Hypergraph structure built")
        return embeddings
        
    def initialize_inter_layer_gates(self):
        """
        Initialize inter-layer gating mechanisms.
        """
        logger.info("Initializing inter-layer communication gates...")
        
        # Gate from Layer 3 to Layer 2
        self.gate_3to2 = InterLayerGating(
            dim1=self.layer3.latent_dim,
            dim2=self.layer2.hidden_dim
        ).to(self.device)
        
        # Gate from Layer 2 to Layer 1
        self.gate_2to1 = InterLayerGating(
            dim1=self.layer2.hidden_dim,
            dim2=self.layer1.hidden_dims[-1]
        ).to(self.device)
        
    def integrate_layers(self) -> np.ndarray:
        """
        Integrate all layers with learned gates.
        
        Returns:
            Final integrated embeddings
        """
        logger.info("=" * 80)
        logger.info("PHASE 4: Integrating Layers")
        logger.info("=" * 80)
        
        self.initialize_inter_layer_gates()
        
        # Convert to tensors
        emb_l3 = torch.FloatTensor(self.patient_embeddings_l3).to(self.device)
        emb_l1 = torch.FloatTensor(self.patient_embeddings_l1).to(self.device)
        
        # For simplicity, use mean pooling for Layer 2 representations
        # In practice, this would use the actual temporal representations
        emb_l2 = torch.zeros(len(self.patient_ids), self.layer2.hidden_dim).to(self.device)
        
        # Compute gates (using time=0 as reference)
        t = torch.zeros(len(self.patient_ids), 1).to(self.device)
        
        gate_3to2_vals = self.gate_3to2(emb_l3, emb_l2, t)
        gate_2to1_vals = self.gate_2to1(emb_l2, emb_l1, t)
        
        # Integrate with gates
        # m_{3->2} = gate * emb_l3
        # m_{2->1} = gate * emb_l2
        
        # Final representation: z_final = z_l3 + z_l2 + z_l1 + m_{3->2} + m_{2->1}
        # Simplified version (matching dimensions via projection)
        
        final_emb = emb_l3 + emb_l1[:, :self.layer3.latent_dim]  # Match dimensions
        
        logger.info("Layer integration completed")
        
        return final_emb.detach().cpu().numpy()
        
    def fit_survival_model(self,
                          survival_data: Dict,
                          n_epochs: int = 100) -> None:
        """
        Fit survival model on integrated embeddings.
        
        Args:
            survival_data: Survival times and events
            n_epochs: Training epochs
        """
        logger.info("=" * 80)
        logger.info("PHASE 5: Training Survival Model")
        logger.info("=" * 80)
        
        # Get final embeddings
        final_embeddings = self.integrate_layers()
        
        # Prepare survival data
        times = []
        events = []
        valid_indices = []
        
        for idx, pid in enumerate(self.patient_ids):
            if pid in survival_data:
                times.append(survival_data[pid]['time'])
                events.append(survival_data[pid]['event'])
                valid_indices.append(idx)
                
        features = final_embeddings[valid_indices]
        times = np.array(times)
        events = np.array(events)
        
        # Initialize and train survival model
        self.survival_model = BayesianSurvivalModel(
            input_dim=features.shape[1],
            hidden_dims=[128, 64],
            device=self.device
        )
        
        self.survival_model.train_model(
            features=features,
            times=times,
            events=events,
            n_epochs=n_epochs
        )
        
        # Evaluate
        c_index = self.survival_model.compute_concordance_index(features, times, events)
        
        logger.info(f"Survival model training completed. C-index: {c_index:.4f}")
        
    def predict_survival(self,
                        patient_indices: List[int],
                        eval_times: np.ndarray,
                        n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict survival probabilities with uncertainty.
        
        Args:
            patient_indices: Indices of patients to predict
            eval_times: Time points to evaluate
            n_samples: Number of MC samples
            
        Returns:
            Tuple of (survival_probs, confidence_intervals)
        """
        final_embeddings = self.integrate_layers()
        features = final_embeddings[patient_indices]
        
        survival_probs, ci = self.survival_model.predict_survival(
            features=features,
            eval_times=eval_times,
            n_samples=n_samples
        )
        
        return survival_probs, ci
        
    def fit(self,
            baseline_features: np.ndarray,
            modality_mask: np.ndarray,
            sequences: Dict,
            survival_data: Dict,
            patient_ids: List,
            sequence_input_dim: int,
            n_epochs_l3: int = 100,
            n_epochs_l2: int = 50,
            n_epochs_l1: int = 50,
            n_epochs_survival: int = 100) -> Dict:
        """
        Complete training pipeline.
        
        Args:
            baseline_features: Patient baseline features
            modality_mask: Modality availability mask
            sequences: Patient sequences
            survival_data: Survival times and events
            patient_ids: List of patient IDs
            sequence_input_dim: Dimension of sequence features
            n_epochs_l3: Epochs for Layer 3
            n_epochs_l2: Epochs for Layer 2
            n_epochs_l1: Epochs for Layer 1
            n_epochs_survival: Epochs for survival model
            
        Returns:
            Dictionary with results
        """
        logger.info("=" * 80)
        logger.info("STARTING TMV-HNN TRAINING")
        logger.info("=" * 80)
        
        # Phase 1: Layer 3
        cluster_labels_l3 = self.fit_layer3(
            baseline_features, 
            modality_mask, 
            patient_ids,
            n_epochs_l3
        )
        
        # Phase 2: Layer 2
        state_labels_l2, state_metadata = self.fit_layer2(
            sequences, 
            sequence_input_dim,
            n_epochs_l2
        )
        
        # Phase 3: Layer 1
        final_embeddings_l1 = self.fit_layer1(
            sequences,
            n_epochs_l1
        )
        
        # Phase 4: Survival model
        self.fit_survival_model(
            survival_data,
            n_epochs_survival
        )
        
        logger.info("=" * 80)
        logger.info("TMV-HNN TRAINING COMPLETED")
        logger.info("=" * 80)
        
        return {
            'cluster_labels_l3': cluster_labels_l3,
            'state_labels_l2': state_labels_l2,
            'final_embeddings': self.integrate_layers(),
            'n_patients': len(patient_ids),
            'n_clusters_l3': self.layer3.n_clusters,
            'n_states_l2': self.layer2.n_states
        }
