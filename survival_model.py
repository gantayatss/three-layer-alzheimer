"""
survival_model.py - Bayesian Neural Survival Analysis
Implements Cox proportional hazards with neural networks and uncertainty quantification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)


class MonotonicNN(nn.Module):
    """
    Monotonic neural network for baseline hazard estimation.
    """
    
    def __init__(self, n_basis: int = 10):
        """
        Initialize monotonic NN.
        
        Args:
            n_basis: Number of basis functions
        """
        super().__init__()
        
        self.n_basis = n_basis
        
        # IMPROVED: Initialize weights to give reasonable baseline hazard
        # Start with small positive values to avoid extreme hazards
        self.weights = nn.Parameter(torch.ones(n_basis) * 0.1)
        
        # Basis function centers - spread across reasonable time range (0-10 years)
        self.register_buffer('centers', torch.linspace(0, 10, n_basis))
        # Wider widths for smoother baseline
        self.register_buffer('widths', torch.ones(n_basis) * 1.0)
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute baseline log hazard.
        
        Args:
            t: Time points [batch_size, 1]
            
        Returns:
            Log baseline hazard [batch_size, 1]
        """
        # Ensure positive weights (but keep them small)
        pos_weights = F.softplus(self.weights) * 0.1
        
        # RBF basis functions
        basis = torch.exp(-((t - self.centers.unsqueeze(0)) ** 2) / (2 * self.widths.unsqueeze(0) ** 2))
        
        # Weighted sum
        log_hazard = torch.sum(pos_weights.unsqueeze(0) * basis, dim=1, keepdim=True)
        
        # Ensure baseline hazard is in reasonable range
        # Baseline should be relatively flat and moderate
        log_hazard = torch.clamp(log_hazard, min=-4, max=-1)
        
        return log_hazard


class BayesianNeuralCox(nn.Module):
    """
    Bayesian Neural Network for Cox Proportional Hazards.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dims: List[int] = [128, 64],
                 n_basis: int = 10):
        """
        Initialize Bayesian Neural Cox model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            n_basis: Number of basis functions for baseline hazard
        """
        super().__init__()
        
        # Hazard function network
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
            
        layers.append(nn.Linear(prev_dim, 1))  # Output: log relative hazard
        
        self.hazard_net = nn.Sequential(*layers)
        
        # Baseline hazard
        self.baseline_hazard = MonotonicNN(n_basis)
        
        # Variational parameters for Bayesian inference
        self.log_sigma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute hazard function.
        
        Args:
            x: Patient features [batch_size, input_dim]
            t: Time points [batch_size, 1]
            
        Returns:
            Hazard values [batch_size, 1]
        """
        # Relative hazard
        log_relative_hazard = self.hazard_net(x)
        
        # Baseline hazard
        log_baseline = self.baseline_hazard(t)
        
        # Total log hazard: log(lambda(t|x)) = log(lambda_0(t)) + f(x)
        log_hazard = log_baseline + log_relative_hazard
        
        # FIXED: Adjust clipping range for more realistic Alzheimer's hazards
        # For Alzheimer's: annual event rate typically 5-20%
        # This corresponds to hazard rates of ~0.05-0.25 per year
        # In log space: log(0.05) = -3.0, log(0.25) = -1.4
        # Allow slightly wider range: [-4, -0.5] gives hazard [0.018, 0.61]
        log_hazard = torch.clamp(log_hazard, min=-4, max=-0.5)
        
        hazard = torch.exp(log_hazard)
        
        return hazard
        
    def forward_with_uncertainty(self, 
                                x: torch.Tensor, 
                                t: torch.Tensor, 
                                n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty estimation via MC dropout.
        
        Args:
            x: Patient features
            t: Time points
            n_samples: Number of MC samples
            
        Returns:
            Tuple of (mean_hazard, std_hazard)
        """
        self.train()  # Enable dropout for MC sampling
        
        hazards = []
        for _ in range(n_samples):
            h = self.forward(x, t)
            hazards.append(h)
            
        hazards = torch.stack(hazards, dim=0)
        
        mean_hazard = hazards.mean(dim=0)
        std_hazard = hazards.std(dim=0)
        
        return mean_hazard, std_hazard


def cox_partial_likelihood(hazards: torch.Tensor,
                           events: torch.Tensor,
                           times: torch.Tensor) -> torch.Tensor:
    """
    Compute Cox partial likelihood loss.
    
    Args:
        hazards: Hazard predictions [batch_size, 1]
        events: Event indicators [batch_size] (1=event, 0=censored)
        times: Event/censoring times [batch_size]
        
    Returns:
        Negative log partial likelihood
    """
    # Sort by time
    sorted_indices = torch.argsort(times, descending=True)
    hazards = hazards[sorted_indices].squeeze()
    events = events[sorted_indices]
    
    # Log hazards
    log_hazards = torch.log(hazards + 1e-10)
    
    # Risk set calculation
    # For each time point, sum hazards of all patients still at risk
    log_risk = torch.logcumsumexp(log_hazards, dim=0)
    
    # Partial likelihood for observed events
    uncensored_likelihood = (log_hazards - log_risk) * events
    
    # Negative log likelihood
    loss = -uncensored_likelihood.sum() / (events.sum() + 1e-10)
    
    return loss


def survival_function(hazards: torch.Tensor, 
                     times: torch.Tensor,
                     eval_times: torch.Tensor) -> torch.Tensor:
    """
    Compute survival function S(t) = exp(-cumulative hazard).
    
    Args:
        hazards: Hazard function values
        times: Time points where hazard is evaluated
        eval_times: Times to evaluate survival function
        
    Returns:
        Survival probabilities
    """
    # Simple numerical integration: trapezoidal rule
    # Cumulative hazard = integral of hazard from 0 to t
    
    survival_probs = []
    
    for t_eval in eval_times:
        mask = times <= t_eval
        if mask.sum() == 0:
            survival_probs.append(torch.ones_like(hazards[0]))
            continue
            
        hazards_subset = hazards[mask]
        times_subset = times[mask]
        
        # Trapezoidal integration
        if len(hazards_subset) > 1:
            cumulative_hazard = torch.trapz(hazards_subset, times_subset)
        else:
            cumulative_hazard = hazards_subset[0] * times_subset[0]
            
        survival = torch.exp(-cumulative_hazard)
        survival_probs.append(survival)
        
    return torch.stack(survival_probs)


class BayesianSurvivalModel:
    """
    Bayesian Neural Survival Analysis with uncertainty quantification.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [128, 64],
                 n_basis: int = 10,
                 device: str = 'cpu'):
        """
        Initialize survival model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            n_basis: Number of basis functions
            device: Device to use
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_basis = n_basis
        self.device = device
        
        self.model = BayesianNeuralCox(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            n_basis=n_basis
        ).to(device)
        
    def train_model(self,
                   features: np.ndarray,
                   times: np.ndarray,
                   events: np.ndarray,
                   n_epochs: int = 100,
                   batch_size: int = 64,
                   learning_rate: float = 1e-3,
                   l2_reg: float = 1e-4) -> None:
        """
        Train survival model.
        
        Args:
            features: Patient features [n_patients, input_dim]
            times: Survival times [n_patients]
            events: Event indicators [n_patients]
            n_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            l2_reg: L2 regularization weight
        """
        logger.info(f"Training survival model on {len(features)} patients...")
        
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                    lr=learning_rate, 
                                    weight_decay=l2_reg)
        
        # Convert to tensors
        X = torch.FloatTensor(features).to(self.device)
        T = torch.FloatTensor(times).to(self.device)
        E = torch.FloatTensor(events).to(self.device)
        
        n_samples = len(features)
        best_loss = float('inf')
        
        for epoch in range(n_epochs):
            self.model.train()
            epoch_loss = 0.0
            
            # Mini-batch training
            indices = np.random.permutation(n_samples)
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                
                batch_x = X[batch_indices]
                batch_t = T[batch_indices].unsqueeze(1)
                batch_e = E[batch_indices]
                
                optimizer.zero_grad()
                
                # Compute hazards
                hazards = self.model(batch_x, batch_t)
                
                # Cox partial likelihood loss
                loss = cox_partial_likelihood(hazards, batch_e, batch_t.squeeze())
                
                # Add KL divergence for Bayesian uncertainty (simplified)
                kl_loss = 0.5 * torch.sum(self.model.log_sigma ** 2)
                total_loss = loss + 0.01 * kl_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                
            avg_loss = epoch_loss / (n_samples / batch_size)
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
                
            if avg_loss < best_loss:
                best_loss = avg_loss
                
        logger.info(f"Training completed. Best loss: {best_loss:.4f}")
        
    def predict_survival(self,
                        features: np.ndarray,
                        eval_times: np.ndarray,
                        n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict survival probabilities with uncertainty.
        
        Args:
            features: Patient features [n_patients, input_dim]
            eval_times: Time points to evaluate [n_timepoints]
            n_samples: Number of MC samples for uncertainty
            
        Returns:
            Tuple of (survival_probs, confidence_intervals)
        """
        self.model.eval()
        
        X = torch.FloatTensor(features).to(self.device)
        
        all_survival_curves = []
        
        # MC sampling for uncertainty
        for _ in range(n_samples):
            self.model.train()  # Enable dropout
            
            with torch.no_grad():
                survival_probs = []
                
                for t_eval in eval_times:
                    # FIXED: Proper integration of hazard function
                    # Compute cumulative hazard: Î›(t) = âˆ«â‚€áµ— Î»(s) ds
                    
                    # Create time grid for numerical integration
                    n_grid = 50  # Number of integration points
                    t_grid = torch.linspace(0, t_eval, n_grid).to(self.device)
                    
                    # Evaluate hazard at each time point
                    hazards_over_time = []
                    for t_point in t_grid:
                        t_tensor = torch.full((len(features), 1), t_point).to(self.device)
                        h = self.model(X, t_tensor)
                        hazards_over_time.append(h.squeeze())
                    
                    # Stack: [n_patients, n_grid]
                    hazards_over_time = torch.stack(hazards_over_time, dim=1)
                    
                    # Trapezoidal integration to get cumulative hazard
                    # torch.trapz integrates over dimension 1 (time dimension)
                    cumulative_hazard = torch.trapz(hazards_over_time, t_grid, dim=1)
                    
                    # Survival function: S(t) = exp(-Î›(t))
                    survival = torch.exp(-cumulative_hazard)
                    
                    # Ensure survival is in valid range [0, 1]
                    survival = torch.clamp(survival, min=0.0, max=1.0)
                    
                    survival_probs.append(survival.cpu().numpy())
                    
                survival_probs = np.array(survival_probs).T
                all_survival_curves.append(survival_probs)
                
        all_survival_curves = np.array(all_survival_curves)
        
        # Compute mean and confidence intervals
        mean_survival = all_survival_curves.mean(axis=0)
        
        ci_lower = np.percentile(all_survival_curves, 2.5, axis=0)
        ci_upper = np.percentile(all_survival_curves, 97.5, axis=0)
        
        confidence_intervals = np.stack([ci_lower, ci_upper], axis=-1)
        
        # ADDED: Log sample predictions for validation
        logger.info(f"Sample survival predictions at t={eval_times[0]}yr: "
                   f"mean={mean_survival[:, 0].mean():.3f}, "
                   f"range=[{mean_survival[:, 0].min():.3f}, {mean_survival[:, 0].max():.3f}]")
        
        # ADDED: Validate predictions are reasonable
        if mean_survival[:, 0].mean() < 0.50:
            logger.error(f"❌ UNREALISTIC: Mean 1-year survival is {mean_survival[:, 0].mean():.3f}")
            logger.error("This suggests the model is predicting extremely high risk!")
            logger.error("Possible causes:")
            logger.error("  1. Hazard function is too high")
            logger.error("  2. Baseline hazard initialization is wrong")
            logger.error("  3. Features are causing extreme predictions")
            logger.error("Consider: reducing learning rate, checking feature scaling, or adjusting hazard clipping")
        elif mean_survival[:, 0].mean() < 0.70:
            logger.warning(f"⚠️ LOW: Mean 1-year survival is {mean_survival[:, 0].mean():.3f}")
            logger.warning("Expected >0.9 for Alzheimer's patients")
        else:
            logger.info(f"✓ Survival predictions look reasonable (mean 1yr survival: {mean_survival[:, 0].mean():.3f})")
        
        return mean_survival, confidence_intervals
        
    def predict_risk(self, features: np.ndarray) -> np.ndarray:
        """
        Predict relative risk scores.
        
        Args:
            features: Patient features
            
        Returns:
            Risk scores
        """
        self.model.eval()
        
        X = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            # Use a reference time (e.g., t=1)
            t = torch.ones(len(features), 1).to(self.device)
            hazards = self.model(X, t)
            
            return hazards.cpu().numpy().squeeze()
            
    def compute_concordance_index(self,
                                 features: np.ndarray,
                                 times: np.ndarray,
                                 events: np.ndarray) -> float:
        """
        Compute Harrell's C-index.
        
        Args:
            features: Patient features
            times: Survival times
            events: Event indicators
            
        Returns:
            C-index value
        """
        risk_scores = self.predict_risk(features)
        
        # Compute concordance
        concordant = 0
        total = 0
        
        for i in range(len(times)):
            if events[i] == 0:
                continue
                
            for j in range(len(times)):
                if times[j] > times[i]:
                    total += 1
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                        
        if total == 0:
            logger.warning("âš ï¸ No valid pairs for C-index computation")
            return 0.5
            
        c_index = concordant / total
        
        logger.info(f"C-index: {c_index:.4f}")
        logger.info(f"  Concordant pairs: {concordant}/{total} ({concordant/total*100:.1f}%)")
        
        # ADDED: Validation checks
        if c_index < 0.4:
            logger.error(f"âŒ C-index ({c_index:.4f}) is extremely poor - model predictions are anti-correlated with outcomes!")
            logger.error("This indicates a serious bug in the survival model.")
        elif c_index < 0.5:
            logger.warning(f"âš ï¸ C-index ({c_index:.4f}) is worse than random (0.5)")
            logger.warning("Model is not learning meaningful patterns.")
        elif c_index < 0.6:
            logger.warning(f"âš ï¸ C-index ({c_index:.4f}) is only slightly better than random")
            logger.warning("Consider: more training epochs, better features, or hyperparameter tuning")
        elif c_index < 0.7:
            logger.info(f"âœ“ C-index ({c_index:.4f}) shows moderate discrimination")
        else:
            logger.info(f"âœ“âœ“ C-index ({c_index:.4f}) shows good discrimination")
        
        return c_index