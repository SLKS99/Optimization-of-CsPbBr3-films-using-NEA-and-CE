"""
Plotting functions for composition GP and decision tree models
=============================================================
Visualizes how the composition GP and process decision tree are learning.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, List
from src.models import ExperimentParams


def plot_composition_gp(ax, comp_gp, history_df: Optional[pd.DataFrame] = None):
    """
    Plot composition GP learning: FA ratio vs quality predictions.
    
    Shows:
    - Training data points (if available)
    - GP mean prediction
    - Uncertainty bands (±2σ)
    - UCB acquisition function
    """
    if comp_gp is None or not comp_gp.is_trained or comp_gp.gp_model is None:
        ax.text(0.5, 0.5, 'Composition GP Not Trained', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Composition GP: FA Ratio → Quality', fontsize=12)
        return
    
    # Create FA ratio grid for prediction - ensure full 0-1 range
    fa_range = np.linspace(0, 1, 200)
    X_query = fa_range.reshape(-1, 1)
    
    # Check training data range - if it doesn't cover 0-1, we might have issues
    if comp_gp.X_min is not None and comp_gp.X_max is not None:
        train_min = comp_gp.X_min[0] if hasattr(comp_gp.X_min, '__len__') else comp_gp.X_min
        train_max = comp_gp.X_max[0] if hasattr(comp_gp.X_max, '__len__') else comp_gp.X_max
        print(f"GP Training range: {train_min:.3f} to {train_max:.3f}, Querying: 0.0 to 1.0")
        
        # If training data doesn't cover full range, we'll still query but expect higher uncertainty
        if train_max < 0.5:
            print(f"Warning: GP training data only covers up to {train_max:.3f}, but querying full 0-1 range")
    
    X_query_norm = comp_gp._normalize(X_query)
    
    # Debug: Check normalized query range
    print(f"Normalized query range: {X_query_norm.min():.3f} to {X_query_norm.max():.3f}")
    print(f"  (Training data was normalized to 0.0 to 1.0)")
    
    # Check normalization - ensure it's valid
    if np.any(np.isnan(X_query_norm)) or np.any(np.isinf(X_query_norm)):
        nan_mask = np.isnan(X_query_norm) | np.isinf(X_query_norm)
        fa_nan = fa_range[nan_mask]
        print(f"ERROR: Normalization produced NaN/Inf for FA range: {fa_nan.min():.3f} to {fa_nan.max():.3f}")
        ax.text(0.5, 0.5, f'GP Normalization Error\n\nTraining range: {train_min:.3f}-{train_max:.3f}\nCheck training data', 
               ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_title('Composition GP: FA Ratio → Quality', fontsize=12)
        return
    
    import gpax
    import jax.numpy as jnp
    rng_key = gpax.utils.get_keys(1)[0]
    X_query_jax = jnp.array(X_query_norm)
    
    try:
        # Get GP predictions - use fewer samples to avoid memory issues
        n_samples = 50  # Reduced from 200 to avoid huge arrays
        posterior_mean, posterior_samples = comp_gp.gp_model.predict(
            rng_key, X_query_jax, n=n_samples
        )
        
        # Debug: Check if predictions cover full range
        if len(posterior_mean) != len(fa_range):
            print(f"Warning: GP predictions length ({len(posterior_mean)}) != FA range length ({len(fa_range)})")
        posterior_mean_np = np.array(posterior_mean)
        posterior_samples_np = np.array(posterior_samples)
        
        # Debug: Check raw GP predictions
        print(f"Raw GP predictions shape: mean={posterior_mean_np.shape}, samples={posterior_samples_np.shape}")
        if len(posterior_mean_np) > 0:
            print(f"Raw mean range: {np.nanmin(posterior_mean_np):.3f} to {np.nanmax(posterior_mean_np):.3f}")
            print(f"  NaN count: {np.sum(np.isnan(posterior_mean_np))}, Inf count: {np.sum(np.isinf(posterior_mean_np))}")
        
        # Handle different shapes of posterior_mean
        if posterior_mean_np.ndim > 1:
            posterior_mean_np = posterior_mean_np.ravel()
        else:
            posterior_mean_np = np.atleast_1d(posterior_mean_np)
        
        # Calculate uncertainty (std dev) - handle different shapes more carefully
        original_shape = posterior_samples_np.shape
        n_query = len(fa_range)
        
        print(f"Processing posterior_samples: original shape={original_shape}, expected (n_samples={n_samples}, n_query={n_query})")
        
        if posterior_samples_np.ndim == 3:
            # Handle 3D shapes - common cases:
            # 1. (n_chains, n_samples, n_query) - e.g., (1000, 50, 200) from MCMC
            # 2. (n_samples, n_query, output_dim) - standard case
            # 3. (n_query, n_samples, output_dim) - transposed
            
            # Check for (n_chains, n_samples, n_query) first - this is the most common MCMC case
            if (posterior_samples_np.shape[1] == n_samples and 
                posterior_samples_np.shape[2] == n_query and 
                posterior_samples_np.shape[0] > n_samples):
                # Shape: (n_chains, n_samples, n_query) - take mean across chains
                print(f"Detected MCMC chains shape: (n_chains={posterior_samples_np.shape[0]}, n_samples={n_samples}, n_query={n_query})")
                posterior_samples_np = np.mean(posterior_samples_np, axis=0)
            # Check for (n_samples, n_query, output_dim)
            elif (posterior_samples_np.shape[0] == n_samples and 
                  posterior_samples_np.shape[1] == n_query):
                # Shape: (n_samples, n_query, output_dim) - squeeze or mean last dim
                if posterior_samples_np.shape[2] == 1:
                    posterior_samples_np = posterior_samples_np[:, :, 0]
                else:
                    posterior_samples_np = np.mean(posterior_samples_np, axis=2)
            # Check for (n_query, n_samples, output_dim)
            elif (posterior_samples_np.shape[0] == n_query and 
                  posterior_samples_np.shape[1] == n_samples):
                # Shape: (n_query, n_samples, output_dim) - transpose and squeeze
                if posterior_samples_np.shape[2] == 1:
                    posterior_samples_np = posterior_samples_np[:, :, 0].T
                else:
                    posterior_samples_np = np.mean(posterior_samples_np, axis=2).T
            # Check for (n_chains, n_samples, n_query) as fallback (if first check didn't match)
            elif (posterior_samples_np.shape[2] == n_query and 
                  posterior_samples_np.shape[1] == n_samples):
                # Shape: (n_chains, n_samples, n_query) - take mean across chains
                print(f"Detected MCMC chains shape (fallback): (n_chains={posterior_samples_np.shape[0]}, n_samples={n_samples}, n_query={n_query})")
                posterior_samples_np = np.mean(posterior_samples_np, axis=0)
            else:
                # Last resort: try to find matching dimensions
                print(f"Warning: Unexpected 3D shape {original_shape}, attempting to find matching dimensions")
                found_match = False
                for i in range(3):
                    for j in range(3):
                        if i != j:
                            if (posterior_samples_np.shape[i] == n_samples and 
                                posterior_samples_np.shape[j] == n_query):
                                # Found matching dimensions, extract them
                                k = [k for k in range(3) if k != i and k != j][0]
                                # Take mean across the third dimension
                                posterior_samples_np = np.mean(posterior_samples_np, axis=k)
                                # Now should be 2D with (n_samples, n_query) or (n_query, n_samples)
                                if posterior_samples_np.shape[0] == n_query:
                                    posterior_samples_np = posterior_samples_np.T
                                found_match = True
                                break
                    if found_match:
                        break
                
                if not found_match:
                    raise ValueError(f"Cannot determine correct shape from 3D array: {original_shape}, expected n_samples={n_samples}, n_query={n_query}")
        elif posterior_samples_np.ndim == 2:
            # Already in correct shape (n_samples, n_query_points)
            # But check if it's actually (n_query_points, n_samples) - transpose if needed
            if posterior_samples_np.shape[0] == n_query and posterior_samples_np.shape[1] == n_samples:
                # Shape is (n_query, n_samples) - transpose to (n_samples, n_query)
                posterior_samples_np = posterior_samples_np.T
            elif posterior_samples_np.shape[1] == n_query and posterior_samples_np.shape[0] == n_samples:
                # Already correct: (n_samples, n_query)
                pass
            else:
                # Try to infer correct shape
                if posterior_samples_np.size == n_samples * n_query:
                    posterior_samples_np = posterior_samples_np.reshape(n_samples, n_query)
                else:
                    # Flatten and reshape
                    posterior_samples_np = posterior_samples_np.flatten()[:n_samples * n_query].reshape(n_samples, n_query)
        else:
            # 1D or other - flatten and reshape
            total_size = posterior_samples_np.size
            if total_size >= n_samples * n_query:
                posterior_samples_np = posterior_samples_np.flatten()[:n_samples * n_query].reshape(n_samples, n_query)
            else:
                # Not enough data - use what we have
                n_available = total_size // n_query
                if n_available > 0:
                    posterior_samples_np = posterior_samples_np.flatten()[:n_available * n_query].reshape(n_available, n_query)
                else:
                    raise ValueError(f"Cannot reshape posterior_samples: shape={original_shape}, need {n_samples}x{n_query}")
        
        # Ensure we have the right shape: (n_samples, n_query_points)
        if posterior_samples_np.ndim != 2:
            raise ValueError(f"posterior_samples should be 2D after processing, got shape {posterior_samples_np.shape}")
        
        print(f"After processing: posterior_samples shape={posterior_samples_np.shape}, expected ({n_samples}, {n_query})")
        
        # Calculate std across samples (axis 0) - should give (n_query_points,)
        posterior_std = np.std(posterior_samples_np, axis=0)
        
        print(f"Posterior std shape: {posterior_std.shape}, range: {np.nanmin(posterior_std):.6f} to {np.nanmax(posterior_std):.6f}")
        print(f"  NaN count in std: {np.sum(np.isnan(posterior_std))}, Inf count: {np.sum(np.isinf(posterior_std))}")
        
        # Ensure shapes match
        if len(posterior_mean_np) != len(fa_range):
            min_len = min(len(posterior_mean_np), len(fa_range))
            posterior_mean_np = posterior_mean_np[:min_len]
            fa_range = fa_range[:min_len]
        
        if len(posterior_std) != len(fa_range):
            min_len = min(len(posterior_std), len(fa_range))
            posterior_std = posterior_std[:min_len]
            posterior_mean_np = posterior_mean_np[:min_len]
            fa_range = fa_range[:min_len]
        
        # Ensure we have valid data
        valid_mask = np.isfinite(posterior_mean_np) & np.isfinite(posterior_std)
        if not np.any(valid_mask):
            raise ValueError("No valid GP predictions")
        
        # Ensure we're using the full range - check if predictions cover full 0-1 range
        fa_range_valid = fa_range[valid_mask]
        posterior_mean_valid = posterior_mean_np[valid_mask]
        posterior_std_valid = posterior_std[valid_mask]
        
        # Debug: Check what range we actually got predictions for and quality values
        if len(fa_range_valid) > 0:
            print(f"GP predictions valid for FA range: {fa_range_valid.min():.3f} to {fa_range_valid.max():.3f}")
            print(f"GP predicted quality range: {posterior_mean_valid.min():.3f} to {posterior_mean_valid.max():.3f}")
            print(f"GP training quality range: {comp_gp.y_train.min():.3f} to {comp_gp.y_train.max():.3f}")
            if fa_range_valid.max() < 0.5:
                print(f"Warning: GP predictions only valid up to {fa_range_valid.max():.3f}, not full 0-1 range")
            
            # Check if GP is learning - quality should vary with FA ratio
            quality_range = posterior_mean_valid.max() - posterior_mean_valid.min()
            if quality_range < 0.1:
                print(f"WARNING: GP predictions show very little variation (range={quality_range:.4f})")
                print(f"  This suggests the GP is not learning a meaningful pattern from compositions.")
                print(f"  Possible causes:")
                print(f"    - Quality values in training data are too similar")
                print(f"    - Quality calculation might need adjustment")
                print(f"    - Not enough training data or data doesn't show composition dependence")
            
            # Check which predictions are invalid and why
            invalid_mask = ~valid_mask
            if np.any(invalid_mask):
                fa_invalid = fa_range[invalid_mask]
                mean_invalid = posterior_mean_np[invalid_mask]
                std_invalid = posterior_std[invalid_mask]
                print(f"Invalid predictions found for {len(fa_invalid)} points:")
                print(f"  FA range of invalid: {fa_invalid.min():.3f} to {fa_invalid.max():.3f}")
                print(f"  Mean values (first 5): {mean_invalid[:5]}")
                print(f"  Std values (first 5): {std_invalid[:5]}")
                print(f"  NaN in mean: {np.sum(np.isnan(mean_invalid))}, Inf in mean: {np.sum(np.isinf(mean_invalid))}")
                print(f"  NaN in std: {np.sum(np.isnan(std_invalid))}, Inf in std: {np.sum(np.isinf(std_invalid))}")
                
                # Check normalized query values for invalid predictions
                X_query_norm_invalid = X_query_norm[invalid_mask]
                print(f"  Normalized query values (first 5): {X_query_norm_invalid[:5]}")
                print(f"  Normalized range: {X_query_norm_invalid.min():.3f} to {X_query_norm_invalid.max():.3f}")
        
        # If the valid range doesn't cover 0-1, we might have an issue
        # But proceed with what we have
        if len(fa_range_valid) == 0:
            ax.text(0.5, 0.5, 'No Valid GP Predictions\n\nAll predictions are invalid', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=11)
            ax.set_title('Composition GP: FA Ratio → Quality', fontsize=12)
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.3)
            return
        
        # Clear axis first to ensure clean plot
        ax.clear()
        
        # Plot uncertainty bands first (so they're behind everything)
        puRd_color = plt.cm.PuRd(0.7)
        try:
            ax.fill_between(fa_range_valid, 
                           posterior_mean_valid - 2 * posterior_std_valid,
                           posterior_mean_valid + 2 * posterior_std_valid,
                           color=puRd_color, alpha=0.25, zorder=1)
        except Exception as fill_err:
            print(f"Warning: Could not plot uncertainty bands: {fill_err}")
        
        # Plot GP mean on top
        try:
            ax.plot(fa_range_valid, posterior_mean_valid, '-', color=puRd_color, 
                   linewidth=3, zorder=3)
        except Exception as plot_err:
            print(f"Warning: Could not plot GP mean: {plot_err}")
        
        # Plot training data on top (so it's visible)
        if comp_gp.X_train is not None and comp_gp.y_train is not None:
            try:
                # Denormalize X_train
                X_train_denorm = comp_gp.X_train * (comp_gp.X_max - comp_gp.X_min) + comp_gp.X_min
                X_train_denorm = X_train_denorm.ravel()
                
                # Only plot if we have valid data
                valid_train = np.isfinite(X_train_denorm) & np.isfinite(comp_gp.y_train)
                if np.any(valid_train):
                    ax.scatter(X_train_denorm[valid_train], comp_gp.y_train[valid_train], 
                              c='darkred', s=50, alpha=0.7, zorder=5,
                              edgecolors='black', linewidth=0.5)
            except Exception as train_err:
                print(f"Warning: Could not plot training data: {train_err}")
        
        # Set axis limits to show full range
        try:
            y_min = np.min(posterior_mean_valid - 2 * posterior_std_valid)
            y_max = np.max(posterior_mean_valid + 2 * posterior_std_valid)
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            elif len(posterior_mean_valid) > 0:
                # Fallback: use mean ± some range
                y_center = np.mean(posterior_mean_valid)
                ax.set_ylim(y_center - 100, y_center + 100)
        except Exception as lim_err:
            print(f"Warning: Could not set axis limits: {lim_err}")
        
        # Always set x-axis to full 0-1 range
        ax.set_xlabel('FA Ratio', fontsize=11)
        ax.set_ylabel('Predicted Quality', fontsize=11, color=puRd_color)
        ax.tick_params(axis='y', labelcolor=puRd_color)
        ax.set_title('Composition GP: FA Ratio → Quality', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)  # Always show full FA ratio range
        
    except Exception as e:
        # Don't clear if we're showing an error
        import traceback
        error_msg = f'GP Query Failed:\n{str(e)[:100]}'
        if len(str(e)) > 100:
            error_msg += '\n...'
        ax.text(0.5, 0.5, error_msg, 
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('Composition GP: FA Ratio → Quality', fontsize=12)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        print(f"Composition GP plotting error: {e}")
        traceback.print_exc()


def plot_decision_tree_importance(ax, tree_learner):
    """
    Plot decision tree feature importance.
    
    Shows which features (FA ratio, temp, spin, etc.) the tree considers most important.
    """
    if tree_learner is None or not tree_learner.is_trained or tree_learner.model is None:
        ax.text(0.5, 0.5, 'Decision Tree Not Trained', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Decision Tree: Feature Importance', fontsize=12)
        return
    
    feature_names = ['FA Ratio', 'Annealing Temp (°C)', 'Spin Speed (rpm)', 
                    'Is Chloroform', 'Is Chlorobenzene', 'Is None Antisolvent']
    importances = tree_learner.model.feature_importances_
    
    # Sort by importance
    sorted_idx = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]
    
    # Use PuRd colormap
    colors = [plt.cm.PuRd(0.3 + 0.5 * (imp / sorted_importances.max())) 
              for imp in sorted_importances]
    
    bars = ax.barh(range(len(sorted_names)), sorted_importances, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel('Feature Importance', fontsize=11)
    ax.set_title(f'Decision Tree: Feature Importance\n(n={tree_learner.model.n_estimators} trees)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, sorted_importances)):
        ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
    
    # Add summary stats
    n_features = len(importances)
    ax.text(0.98, 0.02, f'Total features: {n_features}\nMax importance: {sorted_importances[0]:.3f}',
            transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def plot_tree_predictions_vs_actual(ax, tree_learner, history_df: pd.DataFrame, quality_learner):
    """
    Plot decision tree predictions vs actual quality from history.
    
    Shows how well the tree is learning from completed experiments.
    """
    if tree_learner is None or not tree_learner.is_trained or tree_learner.model is None:
        ax.text(0.5, 0.5, 'Decision Tree Not Trained', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Decision Tree: Predictions vs Actual', fontsize=12)
        return
    
    if history_df is None or history_df.empty:
        ax.text(0.5, 0.5, 'No History Data', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Decision Tree: Predictions vs Actual', fontsize=12)
        return
    
    # Get predictions for history
    X_rows = []
    y_actual = []
    
    for _, row in history_df.iterrows():
        pl_val = row.get("PL_Intensity", row.get("PL_Intensity ", None))
        if pd.isna(pl_val) and pd.isna(row.get("Stability_Hours")):
            continue
        
        x = tree_learner._featurize_log_row(row)
        if x is None:
            continue
        
        y = quality_learner.calculate_quality_target(row)
        X_rows.append(x)
        y_actual.append(y)
    
    if len(X_rows) < 5:
        ax.text(0.5, 0.5, 'Not Enough Data Points', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Decision Tree: Predictions vs Actual', fontsize=12)
        return
    
    X = np.vstack(X_rows)
    y_actual = np.array(y_actual)
    y_pred = tree_learner.model.predict(X)
    
    # Calculate R²
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Detect and remove outliers using IQR method
    residuals = y_actual - y_pred
    q1 = np.percentile(residuals, 25)
    q3 = np.percentile(residuals, 75)
    iqr = q3 - q1
    lower_bound = q1 - 2.5 * iqr  # Use 2.5x IQR for more aggressive outlier removal
    upper_bound = q3 + 2.5 * iqr
    
    outlier_mask = (residuals >= lower_bound) & (residuals <= upper_bound)
    n_outliers = np.sum(~outlier_mask)
    
    y_actual_clean = y_actual[outlier_mask]
    y_pred_clean = y_pred[outlier_mask]
    
    # Recalculate metrics on cleaned data
    if len(y_actual_clean) > 5:
        ss_res_clean = np.sum((y_actual_clean - y_pred_clean) ** 2)
        ss_tot_clean = np.sum((y_actual_clean - np.mean(y_actual_clean)) ** 2)
        r2_clean = 1 - (ss_res_clean / ss_tot_clean) if ss_tot_clean > 0 else 0.0
        mae_clean = np.mean(np.abs(y_actual_clean - y_pred_clean))
        rmse_clean = np.sqrt(np.mean((y_actual_clean - y_pred_clean) ** 2))
    else:
        r2_clean = r2
        mae_clean = np.mean(np.abs(y_actual - y_pred))
        rmse_clean = np.sqrt(np.mean((y_actual - y_pred) ** 2))
        y_actual_clean = y_actual
        y_pred_clean = y_pred
        outlier_mask = np.ones(len(y_actual), dtype=bool)
        n_outliers = 0
    
    # Plot predictions vs actual (clean data)
    puRd_color = plt.cm.PuRd(0.7)
    ax.scatter(y_actual_clean, y_pred_clean, c=puRd_color, s=50, alpha=0.6, 
              edgecolors='black', linewidth=0.5, zorder=3)
    
    # Plot outliers in different color if any
    if n_outliers > 0:
        y_actual_outliers = y_actual[~outlier_mask]
        y_pred_outliers = y_pred[~outlier_mask]
        ax.scatter(y_actual_outliers, y_pred_outliers, c='orange', s=80, alpha=0.8,
                  marker='x', linewidths=2, zorder=4)
    
    # Perfect prediction line
    all_vals_clean = np.concatenate([y_actual_clean, y_pred_clean])
    if len(all_vals_clean) > 0:
        min_val_plot = np.percentile(all_vals_clean, 5)  # 5th percentile for lower bound
        max_val_plot = np.percentile(all_vals_clean, 95) # 95th percentile for upper bound
        
        # Add a hard floor for extremely negative values if present (e.g., penalties)
        min_val_plot = max(-600, min_val_plot)

        ax.plot([min_val_plot, max_val_plot], [min_val_plot, max_val_plot], 
               'r--', linewidth=2, alpha=0.7, zorder=2)
        
        # Set explicit limits for the axes to zoom in on the main data cluster
        padding = (max_val_plot - min_val_plot) * 0.1
        ax.set_xlim(min_val_plot - padding, max_val_plot + padding)
        ax.set_ylim(min_val_plot - padding, max_val_plot + padding)
    
    ax.set_xlabel('Actual Quality', fontsize=11)
    ax.set_ylabel('Predicted Quality', fontsize=11)
    title_text = f'Decision Tree: Predictions vs Actual\n(R² = {r2_clean:.3f}, n={len(y_actual_clean)})'
    if n_outliers > 0:
        title_text += f', {n_outliers} outliers removed'
    ax.set_title(title_text, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add text box with stats
    ax.text(0.05, 0.95, f'MAE: {mae_clean:.2f}\nRMSE: {rmse_clean:.2f}',
            transform=ax.transAxes, fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Print outlier info to console for debugging
    if n_outliers > 0:
        print(f"\nDecision Tree Outlier Detection:")
        print(f"  Removed {n_outliers} outliers ({(n_outliers/len(y_actual)*100):.1f}% of data)")
        print(f"  Residual range: [{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"  Outlier actual values: {y_actual[~outlier_mask]}")
        print(f"  Outlier predicted values: {y_pred[~outlier_mask]}")
        print(f"  Outlier residuals: {residuals[~outlier_mask]}")
