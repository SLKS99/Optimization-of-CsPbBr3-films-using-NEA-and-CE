import gpax
import numpy as np
import jax.numpy as jnp
import jax.random as jra
from typing import Tuple, List

# Enable x64 for better precision
gpax.utils.enable_x64()

def run_GP(xtrain: np.ndarray, ytrain: np.ndarray, xtest: np.ndarray) -> Tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs Exact Gaussian Process Regression using GPax.
    
    Args:
        xtrain: Training features (N, D)
        ytrain: Training targets (N,)
        xtest: Test/Candidate features (M, D)
        
    Returns:
        gp_model: Trained GPax model
        acq_values: Acquisition function values (UE - Uncertainty Sampling by default in notebook, 
                    but we usually want UCB for optimization. Notebook used UE? Let's check.)
                    Notebook used: obj = gpax.acquisition.UE(...) -> Uncertainty Exploitation? 
                    Usually for BO we want UCB or EI. The notebook comment said "UE".
                    Let's stick to UCB for general optimization, or EI.
                    However, if the user wants *active learning* (better model), UE is fine.
                    If the user wants *best material*, UCB is better.
                    Let's use UCB as it balances both.
        posterior_mean: Mean prediction
        posterior_samples: Samples from posterior
    """
    # Ensure inputs are JAX arrays
    xtrain = jnp.array(xtrain)
    ytrain = jnp.array(ytrain)
    xtest = jnp.array(xtest)
    
    rng_key, rng_key_predict = gpax.utils.get_keys(1)
    
    # Standard Matern Kernel often used in BO
    # Notebook used: kernel='Matern', mean_fn=None
    gp_model = gpax.ExactGP(input_dim=xtrain.shape[1], kernel='Matern')
    gp_model.fit(rng_key, xtrain, ytrain, num_warmup=2000, num_samples=2000)
    
    # Notebook used UE (Uncertainty Exploitation?), let's switch to UCB for "Optimization"
    # UCB = Mean + beta * Std
    acq_values = gpax.acquisition.UCB(rng_key_predict, gp_model, xtest, beta=2.0)
    
    # Get GP prediction
    posterior_mean, posterior_samples = gp_model.predict(rng_key_predict, xtest, n=200)
    
    return gp_model, acq_values, posterior_mean, posterior_samples

def transition_constraint(x_candidates: np.ndarray, x_transition: np.ndarray, constraint_factor: float = 5.0) -> np.ndarray:
    """
    Computes a penalty factor based on distance to experiments currently "in transition".
    Returns a value between 0 (bad, too close) and 1 (good, far away).
    
    Args:
        x_candidates: Candidate points to score (M, D)
        x_transition: Points currently running (K, D)
        constraint_factor: Strength of penalty
        
    Returns:
        constraint: (M,) array of weights
    """
    if len(x_transition) == 0:
        return np.ones(len(x_candidates))
        
    # Compute distance matrix between candidates and transition points
    # We need min distance for each candidate to ANY transition point
    
    # Simple Euclidean distance in feature space (features should be normalized!)
    # x_candidates: (M, D), x_transition: (K, D)
    
    # Manual broadcast for distance
    # d[i, j] = dist(cand[i], trans[j])
    
    constraints = []
    for cand in x_candidates:
        # Distance to all transition points
        dists = np.linalg.norm(x_transition - cand, axis=1)
        min_dist = np.min(dists)
        
        # Gaussian penalty: exp(-alpha * d^2) -> 1 at d=0 (bad)
        # We want 1 - penalty
        # Notebook: constraint = np.exp(-constraint_factor*10 * d**2)
        #           return 1 - constraint
        
        penalty = np.exp(-constraint_factor * 10 * (min_dist**2))
        constraints.append(1.0 - penalty)
        
    return np.array(constraints)

