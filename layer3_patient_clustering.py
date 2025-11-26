"""
layer3_patient_clustering.py - Layer 3: Multi-Modal Patient Profile Clustering
Implements VAE-based embedding learning and spectral clustering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import pairwise_distances
import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)


class MultiModalVAE(nn.Module):
    """
    Variational Autoencoder for multi-modal patient data integration.
    """
    
    def __init__(self, 
                 demo_dim: int, 
                 gene_dim: int, 
                 cog_dim: int, 
                 mri_dim: int,
                 latent_dim: int = 64,
                 hidden_dims: List[int] = [256, 128]):
        """
        Initialize Multi-Modal VAE.
        
        Args:
            demo_dim: Dimension of demographic features
            gene_dim: Dimension of genetic features
            cog_dim: Dimension of cognitive features
            mri_dim: Dimension of MRI features
            latent_dim: Dimension of latent space
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        
        self.demo_dim = demo_dim
        self.gene_dim = gene_dim
        self.cog_dim = cog_dim
        self.mri_dim = mri_dim
        self.latent_dim = latent_dim
        
        total_dim = demo_dim + gene_dim + cog_dim + mri_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = total_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoders for each modality
        self.decoder_shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.decoder_demo = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, demo_dim)
        )
        
        self.decoder_gene = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, gene_dim)
        )
        
        self.decoder_cog = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, cog_dim)
        )
        
        self.decoder_mri = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, mri_dim)
        )
        
        # Learnable importance weights for modalities
        self.modality_weights = nn.Parameter(torch.ones(4))
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input features [batch_size, total_dim]
            
        Returns:
            Tuple of (mu, logvar)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode latent vector to reconstructed modalities.
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            
        Returns:
            Tuple of (demo_recon, gene_recon, cog_recon, mri_recon)
        """
        h = self.decoder_shared(z)
        
        demo_recon = self.decoder_demo(h)
        gene_recon = self.decoder_gene(h)
        cog_recon = self.decoder_cog(h)
        mri_recon = self.decoder_mri(h)
        
        return demo_recon, gene_recon, cog_recon, mri_recon
        
    def forward(self, x: torch.Tensor) -> Tuple:
        """
        Forward pass through VAE.
        
        Args:
            x: Input features
            
        Returns:
            Tuple of (reconstructions, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructions = self.decode(z)
        return reconstructions, mu, logvar
        
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation (mean of distribution).
        
        Args:
            x: Input features
            
        Returns:
            Latent representation
        """
        mu, _ = self.encode(x)
        return mu


def vae_loss(reconstructions: Tuple, 
             x: torch.Tensor, 
             mu: torch.Tensor, 
             logvar: torch.Tensor, 
             modality_mask: torch.Tensor,
             modality_weights: torch.Tensor,
             beta: float = 1.0) -> torch.Tensor:
    """
    Compute VAE loss with multi-modal reconstruction and KL divergence.
    
    Args:
        reconstructions: Tuple of reconstructed modalities
        x: Original input features
        mu: Latent mean
        logvar: Latent log variance
        modality_mask: Binary mask indicating available modalities [batch_size, 4]
        modality_weights: Learned importance weights for modalities
        beta: KL divergence weight
        
    Returns:
        Total loss
    """
    demo_recon, gene_recon, cog_recon, mri_recon = reconstructions
    
    # Split input into modalities
    demo_dim = demo_recon.shape[1]
    gene_dim = gene_recon.shape[1]
    cog_dim = cog_recon.shape[1]
    mri_dim = mri_recon.shape[1]
    
    demo_orig = x[:, :demo_dim]
    gene_orig = x[:, demo_dim:demo_dim+gene_dim]
    cog_orig = x[:, demo_dim+gene_dim:demo_dim+gene_dim+cog_dim]
    mri_orig = x[:, demo_dim+gene_dim+cog_dim:]
    
    # Normalize modality weights
    weights = F.softmax(modality_weights, dim=0)
    
    # Reconstruction losses (weighted by availability)
    recon_loss = 0.0
    
    if modality_mask[:, 0].any():
        demo_loss = F.mse_loss(demo_recon[modality_mask[:, 0]], 
                               demo_orig[modality_mask[:, 0]], reduction='mean')
        recon_loss += weights[0] * demo_loss
        
    if modality_mask[:, 1].any():
        gene_loss = F.mse_loss(gene_recon[modality_mask[:, 1]], 
                               gene_orig[modality_mask[:, 1]], reduction='mean')
        recon_loss += weights[1] * gene_loss
        
    if modality_mask[:, 2].any():
        cog_loss = F.mse_loss(cog_recon[modality_mask[:, 2]], 
                              cog_orig[modality_mask[:, 2]], reduction='mean')
        recon_loss += weights[2] * cog_loss
        
    if modality_mask[:, 3].any():
        mri_loss = F.mse_loss(mri_recon[modality_mask[:, 3]], 
                              mri_orig[modality_mask[:, 3]], reduction='mean')
        recon_loss += weights[3] * mri_loss
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    # ADDED: Diversity loss to prevent collapse
    # Encourage embeddings to be different from each other
    if mu.shape[0] > 1:
        # Compute pairwise distances in latent space
        mu_expanded = mu.unsqueeze(1)  # [batch, 1, latent_dim]
        mu_tiled = mu.unsqueeze(0)     # [1, batch, latent_dim]
        pairwise_dist = torch.sum((mu_expanded - mu_tiled) ** 2, dim=2)  # [batch, batch]
        
        # Diversity loss: penalize if embeddings are too similar
        # Target: embeddings should be at least distance d_min apart
        # EXTREME: 5.0 (was 2.0) - force very large separation
        d_min = 5.0  # Minimum desired distance
        diversity_loss = torch.mean(torch.relu(d_min - pairwise_dist))
    else:
        diversity_loss = 0.0
    
    # Total loss
    # EXTREME: diversity weight = 10.0 (was 1.0) to prevent any collapse
    total_loss = recon_loss + beta * kl_loss + 10.0 * diversity_loss
    
    return total_loss, recon_loss, kl_loss


class Layer3PatientClustering:
    """
    Layer 3: Multi-Modal Patient Profile Clustering using VAE and Spectral Clustering.
    """
    
    def __init__(self, 
                 n_clusters: int = 10,
                 latent_dim: int = 128,  # INCREASED: 64→128 for more space
                 hidden_dims: List[int] = [256, 128],
                 beta: float = 0.001,  # EXTREME: 0.05→0.001 (100x less than original)
                 sigma: float = 1.0,
                 device: str = 'cpu'):
        """
        Initialize Layer 3.
        
        Args:
            n_clusters: Number of patient profile clusters (K_3)
            latent_dim: Latent space dimension
            hidden_dims: Hidden layer dimensions for VAE
            beta: KL divergence weight
            sigma: Kernel bandwidth for affinity matrix
            device: Device to use ('cpu' or 'cuda')
        """
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.beta = beta
        self.sigma = sigma
        self.device = device
        
        self.vae = None
        self.cluster_labels = None
        self.cluster_centroids = None
        
    def train_vae(self, 
                  features: np.ndarray, 
                  modality_mask: np.ndarray,
                  n_epochs: int = 100,
                  batch_size: int = 64,
                  learning_rate: float = 1e-3) -> None:
        """
        Train VAE on patient features.
        
        Args:
            features: Feature matrix [n_patients, feature_dim]
            modality_mask: Binary mask [n_patients, 4] indicating available modalities
            n_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        logger.info(f"Training VAE on {features.shape[0]} patients...")
        
        # Infer dimensions from data
        demo_dim = 6  # Fixed from data_loader
        gene_dim = 2
        # Estimate remaining dimensions
        remaining_dim = features.shape[1] - demo_dim - gene_dim
        cog_dim = remaining_dim // 2
        mri_dim = remaining_dim - cog_dim
        
        # Initialize VAE
        self.vae = MultiModalVAE(
            demo_dim=demo_dim,
            gene_dim=gene_dim,
            cog_dim=cog_dim,
            mri_dim=mri_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=learning_rate)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(features).to(self.device)
        mask_tensor = torch.BoolTensor(modality_mask).to(self.device)
        
        # Training loop
        n_samples = features.shape[0]
        best_loss = float('inf')
        
        for epoch in range(n_epochs):
            self.vae.train()
            epoch_loss = 0.0
            
            # Mini-batch training
            indices = np.random.permutation(n_samples)
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_x = X_tensor[batch_indices]
                batch_mask = mask_tensor[batch_indices]
                
                optimizer.zero_grad()
                
                # Forward pass
                reconstructions, mu, logvar = self.vae(batch_x)
                
                # Compute loss
                loss, recon_loss, kl_loss = vae_loss(
                    reconstructions, batch_x, mu, logvar, 
                    batch_mask, self.vae.modality_weights, self.beta
                )
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / (n_samples / batch_size)
            
            if (epoch + 1) % 10 == 0:
                # ADDED: Detailed diagnostics
                self.vae.eval()
                with torch.no_grad():
                    sample_embeddings = self.vae.get_latent(X_tensor[:100])
                    emb_std = sample_embeddings.std(dim=0).mean().item()
                    emb_mean_norm = torch.norm(sample_embeddings.mean(dim=0)).item()
                logger.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}, "
                           f"Emb_std: {emb_std:.4f}, Emb_mean_norm: {emb_mean_norm:.4f}")
                
                # WARNING: Check if embeddings are collapsing
                if emb_std < 0.1 and epoch > 20:
                    logger.error(f"❌ VAE COLLAPSE DETECTED: Emb_std={emb_std:.4f} is too low!")
                    logger.error("Embeddings are becoming too similar. Consider:")
                    logger.error("  1. Reduce beta further (current beta may be too high)")
                    logger.error("  2. Increase diversity loss weight")
                    logger.error("  3. Restart training with adjusted hyperparameters")
                elif emb_std < 0.3:
                    logger.warning(f"⚠️ Embedding std={emb_std:.4f} is getting low (target: >0.5)")
                
                self.vae.train()
                
            if avg_loss < best_loss:
                best_loss = avg_loss
                
        logger.info(f"VAE training completed. Best loss: {best_loss:.4f}")
        
    def extract_embeddings(self, features: np.ndarray) -> np.ndarray:
        """
        Extract latent embeddings using trained VAE.
        
        Args:
            features: Feature matrix
            
        Returns:
            Latent embeddings
        """
        self.vae.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(features).to(self.device)
            embeddings = self.vae.get_latent(X_tensor)
            return embeddings.cpu().numpy()
            
    def perform_spectral_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Perform clustering on latent embeddings.
        
        Args:
            embeddings: Latent embeddings [n_patients, latent_dim]
            
        Returns:
            Cluster labels
        """
        logger.info(f"Performing clustering into {self.n_clusters} clusters...")
        
        # FIXED: Use k-means instead of spectral clustering
        # K-means is more robust and doesn't suffer from degeneracy issues
        clustering = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=20,  # Multiple initializations for robustness
            max_iter=300
        )
        
        labels = clustering.fit_predict(embeddings)
        
        logger.info(f"K-means clustering completed with {self.n_clusters} clusters")
        
        # Compute cluster centroids
        centroids = []
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                centroid = embeddings[mask].mean(axis=0)
                centroids.append(centroid)
            else:
                centroids.append(np.zeros(embeddings.shape[1]))
                
        self.cluster_labels = labels
        self.cluster_centroids = np.array(centroids)
        
        # Log cluster sizes
        for k in range(self.n_clusters):
            count = (labels == k).sum()
            logger.info(f"Cluster {k}: {count} patients ({count/len(labels)*100:.1f}%)")
        
        # ADDED: Validate clustering quality
        cluster_sizes = [(labels == k).sum() for k in range(self.n_clusters)]
        max_cluster_size = max(cluster_sizes)
        max_cluster_pct = max_cluster_size / len(labels) * 100
        
        if max_cluster_pct > 40:
            logger.warning(f"âš ï¸ Clustering may be degenerate: largest cluster has {max_cluster_pct:.1f}% of patients")
            logger.warning("âš ï¸ Consider adjusting sigma or using different clustering method")
        
        # Compute clustering quality metrics
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        try:
            silhouette = silhouette_score(embeddings, labels, sample_size=min(1000, len(embeddings)))
            davies_bouldin = davies_bouldin_score(embeddings, labels)
            logger.info(f"Clustering quality - Silhouette: {silhouette:.3f}, Davies-Bouldin: {davies_bouldin:.3f}")
            
            if silhouette < 0.1:
                logger.warning(f"âš ï¸ Low silhouette score ({silhouette:.3f}) indicates poor cluster separation")
        except Exception as e:
            logger.warning(f"Could not compute clustering metrics: {e}")
            
        return labels
        
    def fit(self, 
            features: np.ndarray, 
            modality_mask: np.ndarray,
            n_epochs: int = 100,
            batch_size: int = 64) -> np.ndarray:
        """
        Complete Layer 3 pipeline: train VAE and perform clustering.
        
        Args:
            features: Feature matrix
            modality_mask: Modality availability mask
            n_epochs: Training epochs
            batch_size: Batch size
            
        Returns:
            Cluster labels
        """
        # Train VAE
        self.train_vae(features, modality_mask, n_epochs, batch_size)
        
        # Extract embeddings
        embeddings = self.extract_embeddings(features)
        
        # Perform clustering
        labels = self.perform_spectral_clustering(embeddings)
        
        return labels
        
    def get_cluster_embeddings(self) -> np.ndarray:
        """
        Get cluster centroid embeddings.
        
        Returns:
            Cluster centroids [n_clusters, latent_dim]
        """
        return self.cluster_centroids
    
