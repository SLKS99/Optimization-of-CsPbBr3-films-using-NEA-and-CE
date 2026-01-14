"""
Gaussian Process models and acquisition functions
"""

import numpy as np
import jax.numpy as jnp
import jax.random as jra
import gpax
import numpyro
import numpyro.distributions as dist
from typing import Tuple, Optional


def setup_kernel_prior():
    """
    Define custom prior on GP kernel parameters.
    
    Returns:
    --------
    function
        Kernel prior function
    """
    def kernel_prior():
        k_length = numpyro.sample("k_length", dist.Gamma(0.5, 1))
        k_scale = numpyro.sample("k_scale", dist.LogNormal(0, 2))
        return {"k_length": k_length, "k_scale": k_scale}
    
    return kernel_prior


def normalize_data(x: np.ndarray, ranges: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Normalize data to [0, 1] range.
    If ranges provided, normalize each component independently using its range.
    Otherwise, normalize globally.
    
    Parameters:
    -----------
    x : np.ndarray
        Input data (N x D)
    ranges : np.ndarray, optional
        Array of shape (D, 2) with [min, max] for each dimension.
        If None, uses global min/max normalization.
        
    Returns:
    --------
    np.ndarray
        Normalized data
    """
    if ranges is not None:
        # Normalize each component independently
        x_norm = np.zeros_like(x)
        for d in range(x.shape[1]):
            x_min, x_max = ranges[d, 0], ranges[d, 1]
            x_range = x_max - x_min
            if x_range == 0:
                x_norm[:, d] = 0
            else:
                x_norm[:, d] = (x[:, d] - x_min) / x_range
        return x_norm
    else:
        # Global normalization (original behavior)
        x_min = x.min()
        x_ptp = np.ptp(x)
        if x_ptp == 0:
            return np.zeros_like(x)
        return (x - x_min) / x_ptp


def train_intensity_gp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel_prior_func,
    noise_prior_dist,
    jitter: float = 1e-5
) -> gpax.viGP:
    """
    Train a GP model for intensity prediction.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training input features (compositions)
    y_train : np.ndarray
        Training targets (intensities)
    kernel_prior_func : function
        Kernel prior function
    noise_prior_dist : distribution
        Noise prior distribution
    jitter : float
        Jitter value for numerical stability
        
    Returns:
    --------
    gpax.viGP
        Trained GP model
    """
    rng_key1, _ = gpax.utils.get_keys()
    
    # Determine input dimension from training data
    input_dim = X_train.shape[1]
    
    gp_model = gpax.viGP(
        input_dim=input_dim,  # Now supports 2D or 3D input
        kernel='RBF',
        kernel_prior=kernel_prior_func,
        noise_prior_dist=noise_prior_dist
    )
    
    gp_model.fit(rng_key1, X_train, y_train, jitter=jitter)
    
    return gp_model


def create_search_grid(
    cs_min: float = 0.1,
    cs_max: float = 0.2,
    nea_min: float = 0.0,
    nea_max: float = 0.4,
    ce_min: float = 0.0,
    ce_max: float = 0.4,
    grid_resolution: float = 0.01,
    normalize_func: Optional[callable] = None,
    ranges: Optional[np.ndarray] = None
) -> jnp.ndarray:
    """
    Create a search grid for independent concentration ranges.
    PbBr is fixed at 0.2, so we optimize Cs, NEA, and CE independently.
    
    Parameters:
    -----------
    cs_min, cs_max : float
        Cs concentration range [0.1, 0.2]
    nea_min, nea_max : float
        NEA concentration range [0.0, 0.4]
    ce_min, ce_max : float
        CE concentration range [0.0, 0.4]
    grid_resolution : float
        Step size for the grid
    normalize_func : callable, optional
        Normalization function to apply
    ranges : np.ndarray, optional
        Array of shape (3, 2) with [min, max] for each component (Cs, NEA, CE)
        Used for normalization if normalize_func is provided
        
    Returns:
    --------
    jnp.ndarray
        Grid of composition points (N x 3) with columns [Cs, NEA, CE]
    """
    # Create independent grids for each component
    cs_grid = jnp.arange(cs_min, cs_max + grid_resolution, grid_resolution)
    nea_grid = jnp.arange(nea_min, nea_max + grid_resolution, grid_resolution)
    ce_grid = jnp.arange(ce_min, ce_max + grid_resolution, grid_resolution)
    
    # Create meshgrid for all combinations
    CS, NEA, CE = jnp.meshgrid(cs_grid, nea_grid, ce_grid, indexing='ij')
    
    # Reshape to (N, 3) array
    n_points = CS.size
    Xnew = jnp.stack([
        CS.ravel(),
        NEA.ravel(),
        CE.ravel()
    ], axis=1)
    
    # Apply normalization if requested
    if normalize_func is not None:
        Xnew_np = np.array(Xnew)
        # Check if normalize_func accepts ranges parameter
        import inspect
        sig = inspect.signature(normalize_func)
        if 'ranges' in sig.parameters and ranges is not None:
            Xnew_norm = normalize_func(Xnew_np, ranges=ranges)
        else:
            # Function already has ranges captured in closure
            Xnew_norm = normalize_func(Xnew_np)
        Xnew = jnp.array(Xnew_norm)
    
    return Xnew


def calculate_acquisition(
    gp_model: gpax.viGP,
    X_search: jnp.ndarray,
    beta: float = 5,
    maximize: bool = True,
    jitter: float = 1e-4
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Calculate acquisition function and predictions.
    
    Parameters:
    -----------
    gp_model : gpax.viGP
        Trained GP model
    X_search : jnp.ndarray
        Search grid points
    beta : float
        UCB beta parameter
    maximize : bool
        Whether to maximize acquisition
    jitter : float
        Jitter value
        
    Returns:
    --------
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        Acquisition values, predictions, sampled predictions
    """
    rng_key2, _ = gpax.utils.get_keys()
    
    # Calculate acquisition
    acq = gpax.acquisition.UCB(
        rng_key2, gp_model, X_search,
        maximize=maximize, beta=beta,
        noiseless=False, jitter=jitter
    )
    
    # Get predictions with uncertainty
    y_pred, y_sampled = gp_model.predict_in_batches(
        rng_key2, X_search, noiseless=False, jitter=jitter
    )
    
    # Calculate uncertainty (standard deviation) from sampled predictions
    # y_sampled shape: (n_samples, n_points) or (n_points,) if single sample
    y_sampled_array = np.array(y_sampled)
    if y_sampled_array.ndim > 1:
        y_std = np.std(y_sampled_array, axis=0)
    else:
        # If single sample, use a small constant uncertainty or calculate from variance
        # For now, use a small fraction of the prediction range as uncertainty estimate
        y_std = np.ones_like(y_pred) * (np.max(y_pred) - np.min(y_pred)) * 0.05
    
    return acq, y_pred, y_sampled, y_std


def tune_gp(
    X_measured: np.ndarray,
    y_tune_score: np.ndarray,
    X_unmeasured: jnp.ndarray,
    kernel_prior_func,
    noise_prior_dist,
    thre_percent: float = 0.5,
    jitter: float = 1e-4
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Train a GP model to predict stability scores and adjust acquisition.
    
    Parameters:
    -----------
    X_measured : np.ndarray
        Measured composition points
    y_tune_score : np.ndarray
        Stability scores for measured points
    X_unmeasured : jnp.ndarray
        Unmeasured composition points
    kernel_prior_func : function
        Kernel prior function
    noise_prior_dist : distribution
        Noise prior distribution
    thre_percent : float
        Threshold percentile for adjustment
    jitter : float
        Jitter value
        
    Returns:
    --------
    Tuple[jnp.ndarray, jnp.ndarray]
        Predicted scores and adjusted scores
    """
    rng_key1, rng_key2 = gpax.utils.get_keys()
    
    # Determine input dimension from training data
    input_dim = X_measured.shape[1]
    
    gp_model = gpax.viGP(
        input_dim, kernel='RBF',
        kernel_prior=kernel_prior_func,
        noise_prior_dist=noise_prior_dist
    )
    
    gp_model.fit(rng_key1, X_measured, y_tune_score, jitter=1e-5)
    y_pred, y_sampled = gp_model.predict_in_batches(
        rng_key2, X_unmeasured, noiseless=False, jitter=jitter
    )
    
    thre_value = np.quantile(y_pred, thre_percent)
    adjust_tune_score = jnp.where(y_pred > thre_value, 0, y_pred)
    
    return y_pred, adjust_tune_score
