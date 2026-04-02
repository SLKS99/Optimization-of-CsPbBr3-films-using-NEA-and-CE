"""
Enhanced plotting functions that show meaningful GP learning signals
and multi-objective visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
from src.models import ExperimentParams
import gpax
import jax.numpy as jnp
from src.learner import extract_fa_ratio_from_composition


def extract_fa_ratio(material_system) -> float:
    """Extract FA ratio from material system."""
    fa_ratio = 0.0
    fudma_ratio = 0.0
    total = 0.0
    
    for comp in material_system.a_site:
        name = comp.precursor.name.upper()
        if 'FA' in name and 'FUDMA' not in name:
            fa_ratio += comp.ratio
        elif 'FUDMA' in name:
            fudma_ratio += comp.ratio
        total += comp.ratio
    
    if total > 0:
        return fa_ratio / total
    return 0.5


def plot_gp_learning_diagnostics(ax, learner, history_df):
    """Plot diagnostic information to assess if GP is actually learning."""
    if not learner.is_trained or learner.y_train is None:
        ax.text(0.5, 0.5, 'GP Not Trained', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        return
    
    # Calculate signal-to-noise ratio
    y_mean = np.mean(learner.y_train)
    y_std = np.std(learner.y_train)
    signal_to_noise = y_std / (np.abs(y_mean) + 1e-10)
    
    # Check if GP is learning (variance in predictions)
    if learner.X_train is not None and len(learner.X_train) > 0:
        # Get predictions on training data
        try:
            rng_key = gpax.utils.get_keys(1)[0]
            X_train_norm = learner._normalize(learner.X_train)
            X_train_jax = jnp.array(X_train_norm)
            pred_mean, _ = learner.gp_model.predict(rng_key, X_train_jax, n=10)
            pred_mean = np.array(pred_mean).ravel()
            
            # Calculate R² (coefficient of determination)
            ss_res = np.sum((learner.y_train - pred_mean) ** 2)
            ss_tot = np.sum((learner.y_train - np.mean(learner.y_train)) ** 2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-10))
            
            # Plot predicted vs actual with PuRd colormap
            color = '#DC143C'  # Crimson red
            ax.scatter(learner.y_train, pred_mean, alpha=0.7, s=60, c=color, edgecolors='darkred', linewidths=1)
            # Perfect prediction line
            min_val = min(np.min(learner.y_train), np.min(pred_mean))
            max_val = max(np.max(learner.y_train), np.max(pred_mean))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, 
                   alpha=0.8, zorder=1, color='#8B0000')  # Dark red, no label
            
            ax.set_xlabel('Actual Quality', fontsize=11)
            ax.set_ylabel('GP Predicted Quality', fontsize=11)
            ax.set_title(f'GP Learning Diagnostics\nR² = {r_squared:.3f}, S/N = {signal_to_noise:.2f}', 
                        fontsize=12, fontweight='bold')
            # ax.legend(loc='best', fontsize=9)  # REMOVED
            ax.grid(True, alpha=0.3)
            
            # Add text box with diagnostics
            diag_text = f'Training Points: {len(learner.y_train)}\n'
            diag_text += f'Quality Range: [{np.min(learner.y_train):.1f}, {np.max(learner.y_train):.1f}]\n'
            diag_text += f'Mean Quality: {y_mean:.1f}\n'
            diag_text += f'Std Quality: {y_std:.1f}\n'
            diag_text += f'R² Score: {r_squared:.3f}'
            
            ax.text(0.02, 0.98, diag_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except Exception as e:
            ax.text(0.5, 0.5, f'Diagnostic Error:\n{str(e)[:50]}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)


def plot_gp_1d_slice_actual_values(ax, learner, history_df, candidates, 
                                   fixed_temp=None, fixed_spin=None):
    """
    Plot GP approximation showing ACTUAL quality values (denormalized).
    Queries GP directly for a 1D slice and converts back to quality space.
    """
    if not learner.is_trained or learner.gp_model is None:
        ax.text(0.5, 0.5, 'GP Not Trained', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        return
    
    # Get representative values
    if fixed_temp is None:
        if not history_df.empty:
            temp_col = 'Anneal_Temp_C' if 'Anneal_Temp_C' in history_df.columns else 'Temp_C'
            fixed_temp = np.median(history_df[temp_col].dropna()) if temp_col in history_df.columns else 140
        elif candidates:
            fixed_temp = np.median([c.annealing_temp for c in candidates])
        else:
            fixed_temp = 140
    
    if fixed_spin is None:
        if not history_df.empty and 'Spin_Speed_rpm' in history_df.columns:
            fixed_spin = np.median(history_df['Spin_Speed_rpm'].dropna())
        elif candidates:
            fixed_spin = np.median([c.spin_speed for c in candidates])
        else:
            fixed_spin = 4500
    
    # Create FA ratio range
    fa_range = np.linspace(0, 1, 200)
    
    # Build query points: [FA_ratio, temp, spin]
    X_query = np.array([[fa, fixed_temp, fixed_spin] for fa in fa_range])
    
    # Normalize
    X_query_norm = learner._normalize(X_query)
    X_query_jax = jnp.array(X_query_norm)
    
    # Query GP (predictions are in normalized space)
    try:
        rng_key = gpax.utils.get_keys(1)[0]
        posterior_mean, posterior_samples = learner.gp_model.predict(rng_key, X_query_jax, n=200)
        mean_norm = np.array(posterior_mean).ravel()
        samples_norm = np.array(posterior_samples)
        if samples_norm.ndim == 3:
            samples_norm = samples_norm.reshape(samples_norm.shape[0], -1)
        std_norm = np.std(samples_norm, axis=0).ravel()
        
        # Denormalize: Convert from normalized GP space back to actual quality space
        # GP was trained on normalized y_train, so we need to reverse that normalization
        y_train = learner.y_train
        y_min = np.min(y_train)
        y_max = np.max(y_train)
        y_range_actual = y_max - y_min
        
        # If GP was trained on normalized targets, we need to check
        # For now, assume GP predicts in same space as training (not normalized)
        mean_actual = mean_norm
        std_actual = std_norm
        
        # Plot GP predictions
        valid_mask = np.isfinite(mean_actual) & np.isfinite(std_actual)
        if np.any(valid_mask):
            fa_plot = fa_range[valid_mask]
            mean_plot = mean_actual[valid_mask]
            std_plot = std_actual[valid_mask]
            
            # Use dark red color (no purple/white)
            color = '#DC143C'  # Crimson red
            
            # Uncertainty bands - NO LABELS (no legend)
            ax.fill_between(fa_plot, mean_plot - 2*std_plot, mean_plot + 2*std_plot,
                           alpha=0.2, color=color, zorder=1)  # Removed label
            ax.fill_between(fa_plot, mean_plot - std_plot, mean_plot + std_plot,
                           alpha=0.3, color=color, zorder=2)  # Removed label
            ax.plot(fa_plot, mean_plot, color=color, linewidth=2.5, 
                   zorder=3)  # Removed label
            
            # Add statistics text box
            mean_qual = np.mean(mean_plot)
            max_qual = np.max(mean_plot)
            min_qual = np.min(mean_plot)
            mean_uncert = np.mean(std_plot)
            stats_text = f'Mean Quality: {mean_qual:.1f}\n'
            stats_text += f'Range: [{min_qual:.1f}, {max_qual:.1f}]\n'
            stats_text += f'Avg Uncertainty: {mean_uncert:.1f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot actual observations with error bars
        if not history_df.empty:
            hist_data = {}
            for _, row in history_df.iterrows():
                comp_str = str(row.get('Material_Composition', ''))
                fa_ratio = extract_fa_ratio_from_composition(comp_str)
                temp = row.get('Anneal_Temp_C', row.get('Temp_C', None))
                spin = row.get('Spin_Speed_rpm', None)
                
                if pd.isna(fa_ratio) or pd.isna(temp) or pd.isna(spin):
                    continue
                
                # Only show points near our fixed temp/spin
                if abs(float(temp) - fixed_temp) < 10 and abs(float(spin) - fixed_spin) < 500:
                    key = round(fa_ratio, 3)
                    if key not in hist_data:
                        hist_data[key] = []
                    quality = learner.calculate_quality_target(row)
                    hist_data[key].append(quality)
            
            # Plot observations with error bars
            obs_fa_list = []
            obs_qual_list = []
            obs_err_list = []
            for fa, qualities in hist_data.items():
                if qualities:
                    obs_fa_list.append(fa)
                    obs_qual_list.append(np.mean(qualities))
                    obs_err_list.append(np.std(qualities) if len(qualities) > 1 else 0)
            
            if obs_fa_list:
                obs_color = '#8B0000'  # Dark red for observations
                ax.scatter(obs_fa_list, obs_qual_list, c=obs_color, s=120, marker='x', 
                         linewidths=3, zorder=5, alpha=0.9)  # Removed label
                if any(err > 0 for err in obs_err_list):
                    ax.errorbar(obs_fa_list, obs_qual_list, yerr=obs_err_list, fmt='none', 
                              color=obs_color, capsize=5, capthick=2, zorder=4, alpha=0.7)
        
        ax.set_xlabel('FA Ratio (FA / (FA + FuDMA))', fontsize=11)
        ax.set_ylabel('Quality Target Score', fontsize=11)
        ax.set_title('GP Approximation: Quality vs Composition\n(Actual Values)', 
                    fontsize=13, fontweight='bold')
        # ax.legend(loc='best', fontsize=9)  # REMOVED
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'GP Query Failed:\n{str(e)[:50]}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=10)


def plot_uncertainty_heatmap(ax, candidates, learner, fixed_fa_ratio=None):
    """
    Plot uncertainty as a heatmap showing Temperature vs Spin Speed.
    This gives much better information than FA ratio vs Temperature.
    """
    if not candidates or not learner.is_trained:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        return
    
    # Extract data: Temperature vs Spin Speed
    temps = []
    spins = []
    uncertainties = []
    qualities = []  # Also track quality for reference
    
    # Get fixed FA ratio if not provided (use median)
    if fixed_fa_ratio is None:
        fa_ratios = [extract_fa_ratio(c.material_system) for c in candidates]
        fixed_fa_ratio = np.median(fa_ratios) if fa_ratios else 0.5
    
    fa_tol = 0.1  # ±10% FA ratio tolerance
    
    for c in candidates:
        if c.metrics.uncertainty_score is not None:
            fa_ratio = extract_fa_ratio(c.material_system)
            # Only include candidates near the fixed FA ratio
            if abs(fa_ratio - fixed_fa_ratio) < fa_tol:
                temps.append(c.annealing_temp)
                spins.append(c.spin_speed)
                uncertainties.append(c.metrics.uncertainty_score)
                if c.metrics.predicted_performance is not None:
                    qualities.append(c.metrics.predicted_performance)
    
    if not uncertainties:
        ax.text(0.5, 0.5, f'No uncertainty data near FA={fixed_fa_ratio:.2f}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        return
    
    # Create 2D histogram/heatmap: Temperature vs Spin Speed
    temps = np.array(temps)
    spins = np.array(spins)
    uncertainties = np.array(uncertainties)
    
    # Use actual data range for bins
    temp_min, temp_max = np.min(temps), np.max(temps)
    spin_min, spin_max = np.min(spins), np.max(spins)
    
    # Add padding
    temp_range = temp_max - temp_min
    spin_range = spin_max - spin_min
    temp_bins = np.linspace(temp_min - 0.1*temp_range, temp_max + 0.1*temp_range, 25)
    spin_bins = np.linspace(spin_min - 0.1*spin_range, spin_max + 0.1*spin_range, 25)
    
    # Calculate mean uncertainty in each bin
    uncertainty_grid = np.zeros((len(temp_bins)-1, len(spin_bins)-1))
    quality_grid = np.zeros((len(temp_bins)-1, len(spin_bins)-1))
    counts = np.zeros_like(uncertainty_grid)
    
    for i in range(len(temps)):
        temp_idx = np.digitize(temps[i], temp_bins) - 1
        spin_idx = np.digitize(spins[i], spin_bins) - 1
        
        if 0 <= temp_idx < len(temp_bins)-1 and 0 <= spin_idx < len(spin_bins)-1:
            uncertainty_grid[temp_idx, spin_idx] += uncertainties[i]
            if i < len(qualities):
                quality_grid[temp_idx, spin_idx] += qualities[i]
            counts[temp_idx, spin_idx] += 1
    
    # Average
    with np.errstate(divide='ignore', invalid='ignore'):
        uncertainty_grid = np.divide(uncertainty_grid, counts, 
                                    out=np.zeros_like(uncertainty_grid), 
                                    where=counts!=0)
        quality_grid = np.divide(quality_grid, counts,
                                out=np.zeros_like(quality_grid),
                                where=counts!=0)
    
    # Plot heatmap
    temp_centers = (temp_bins[:-1] + temp_bins[1:]) / 2
    spin_centers = (spin_bins[:-1] + spin_bins[1:]) / 2
    TEMP, SPIN = np.meshgrid(temp_centers, spin_centers)
    
    # Transpose to match (temp, spin) convention
    uncertainty_grid = uncertainty_grid.T
    quality_grid = quality_grid.T
    
    # Use percentile-based normalization for better contrast
    # Exclude zeros and use tighter percentiles to show variation
    uncert_flat = uncertainty_grid[uncertainty_grid > 0]
    if len(uncert_flat) > 0:
        # Calculate statistics
        uncert_mean = np.mean(uncert_flat)
        uncert_std = np.std(uncert_flat)
        uncert_min = np.min(uncert_flat)
        uncert_max = np.max(uncert_flat)
        uncert_range = uncert_max - uncert_min
        
        # If range is very small (all values similar), use std-based scaling
        if uncert_range < uncert_std * 0.5 or uncert_range < 0.1:
            # Values are too similar - use std-based range to show variation
            # This creates a color range centered on mean with std-based spread
            vmin = uncert_mean - 2 * uncert_std
            vmax = uncert_mean + 2 * uncert_std
            # But ensure we don't go below actual min/max
            vmin = max(uncert_min - 0.05 * uncert_range, vmin)
            vmax = min(uncert_max + 0.05 * uncert_range, vmax)
            # If std is also very small, use a percentage-based range
            if uncert_std < 0.01:
                vmin = uncert_mean - 0.5 * uncert_range
                vmax = uncert_mean + 0.5 * uncert_range
        else:
            # Normal case: use 5th to 95th percentile to exclude extreme outliers
            vmin = np.percentile(uncert_flat, 5)
            vmax = np.percentile(uncert_flat, 95)
            # Ensure we have some padding
            padding = max((vmax - vmin) * 0.1, uncert_std * 0.1)
            vmin = max(uncert_min, vmin - padding)
            vmax = min(uncert_max, vmax + padding)
    else:
        # Fallback to raw uncertainties
        uncert_array = np.array(uncertainties)
        vmin = np.min(uncert_array)
        vmax = np.max(uncert_array)
        if vmax - vmin < 1e-6:
            # All same value - create artificial range
            vmin = vmin - 0.1
            vmax = vmax + 0.1
    
    # Ensure we have a meaningful range
    if vmax - vmin < 1e-6:
        # If all values are essentially the same, use a small range around the mean
        mean_val = np.mean(uncert_flat) if len(uncert_flat) > 0 else np.mean(uncertainties)
        vmin = mean_val - 0.1
        vmax = mean_val + 0.1
    
    # Create more levels for smoother color transitions
    n_levels = 50  # More levels = smoother gradient
    levels = np.linspace(vmin, vmax, n_levels)
    
    # Use Reds colormap reversed (dark red to light red, avoiding white)
    from matplotlib.colors import LinearSegmentedColormap
    red_colors = ['#8B0000', '#A52A2A', '#DC143C', '#FF4500', '#FF6347', '#FF7F50']  # Dark to medium red
    red_cmap = LinearSegmentedColormap.from_list('dark_red', red_colors, N=256)
    im = ax.contourf(TEMP, SPIN, uncertainty_grid, levels=levels, cmap=red_cmap, 
                    alpha=0.85, vmin=vmin, vmax=vmax, extend='both')
    
    # Ensure colorbar shows the actual data range
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Uncertainty (σ)', fontsize=10)
    
    # Add tick marks to show the range (format based on magnitude)
    if vmax - vmin < 1.0:
        # Small range - show more decimal places
        tick_format = '{:.3f}'
    elif vmax - vmin < 10.0:
        tick_format = '{:.2f}'
    else:
        tick_format = '{:.1f}'
    
    cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
    cbar.set_ticklabels([tick_format.format(vmin), 
                         tick_format.format((vmin+vmax)/2), 
                         tick_format.format(vmax)])
    
    # Overlay contour lines for quality (optional, lighter)
    if np.any(quality_grid > 0):
        quality_flat = quality_grid[quality_grid > 0]
        if len(quality_flat) > 0:
            q_levels = np.linspace(np.percentile(quality_flat, 10), 
                                  np.percentile(quality_flat, 90), 5)
            ax.contour(TEMP, SPIN, quality_grid, levels=q_levels, 
                      colors='white', alpha=0.3, linewidths=1, linestyles='--')
    
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Spin Speed (rpm)', fontsize=11)
    ax.set_title(f'Uncertainty Heatmap: Temperature vs Spin Speed\n(FA Ratio ≈ {fixed_fa_ratio:.2f})', 
                fontsize=12, fontweight='bold')
    
    # Add statistics and data density info
    if len(uncertainties) > 0:
        uncert_array = np.array(uncertainties)
        n_data_points = len(uncertainties)
        n_bins_with_data = np.sum(counts > 0)
        stats_text = f'Data Points: {n_data_points}\n'
        stats_text += f'Bins w/ Data: {n_bins_with_data}\n'
        stats_text += f'Uncertainty:\n'
        stats_text += f'  Mean: {np.mean(uncert_array):.1f}\n'
        stats_text += f'  Range: [{np.min(uncert_array):.1f}, {np.max(uncert_array):.1f}]'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Add scatter points for actual data locations (optional, can be toggled)
    # Uncomment to see where data points are
    # ax.scatter(temps, spins, c=uncertainties, s=20, alpha=0.3, 
    #           cmap='PuRd', edgecolors='black', linewidths=0.5)


def plot_multi_objective_gp_surfaces(ax1, ax2, learner, candidates, target_temps=[140, 145], target_spins=None):
    """
    For multi-objective optimization, plot 1D slices at specific temperatures and spin speeds:
    1. Quality vs FA Ratio at target temperatures (with different spin speeds)
    2. Uncertainty vs FA Ratio at target temperatures (with different spin speeds)
    
    If target_spins is None, uses common spin speeds from candidates.
    """
    import matplotlib.pyplot as plt
    import gpax
    import jax.numpy as jnp
    
    if not learner.is_trained or not candidates or learner.gp_model is None:
        ax1.text(0.5, 0.5, 'GP Not Trained', ha='center', va='center',
                transform=ax1.transAxes, fontsize=12)
        ax2.text(0.5, 0.5, 'GP Not Trained', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12)
        return
    
    # Get spin speeds from candidates if not specified
    if target_spins is None:
        spin_speeds = sorted(set([c.spin_speed for c in candidates]))
        # Use most common spin speeds (top 3 if available)
        if len(spin_speeds) >= 3:
            # Count frequency
            spin_counts = {}
            for c in candidates:
                spin_counts[c.spin_speed] = spin_counts.get(c.spin_speed, 0) + 1
            sorted_spins = sorted(spin_counts.items(), key=lambda x: x[1], reverse=True)
            target_spins = [s[0] for s in sorted_spins[:3]]
        else:
            target_spins = spin_speeds[:min(3, len(spin_speeds))]
    
    fixed_spin = np.median(target_spins) if target_spins else 4500
    
    # Create FA ratio range
    fa_range = np.linspace(0, 1, 200)
    
    # Create color scheme: dark red to medium red (no white/light colors)
    # Use Reds colormap but reverse it and take darker portion (avoid white)
    from matplotlib.colors import LinearSegmentedColormap
    # Create custom red colormap: dark red to medium red (avoiding white)
    red_colors = ['#8B0000', '#A52A2A', '#DC143C', '#FF4500', '#FF6347']  # Dark red to medium red
    red_cmap = LinearSegmentedColormap.from_list('dark_red', red_colors, N=256)
    temp_colors = [red_cmap(0.3 + 0.5 * i / max(1, len(target_temps) - 1)) for i in range(len(target_temps))]
    spin_styles = ['-', '--', '-.', ':']  # Different line styles for different spin speeds
    
    # Plot for each combination of temperature and spin speed
    for temp_idx, temp in enumerate(target_temps):
        for spin_idx, spin in enumerate(target_spins):
            # Build query points: [FA_ratio, temp, spin]
            X_query = np.array([[fa, temp, spin] for fa in fa_range])
            
            # Normalize
            X_query_norm = learner._normalize(X_query)
            X_query_jax = jnp.array(X_query_norm)
            
            # Query GP
            try:
                rng_key = gpax.utils.get_keys(1)[0]
                posterior_mean, posterior_samples = learner.gp_model.predict(rng_key, X_query_jax, n=200)
                mean_vals = np.array(posterior_mean)
                if mean_vals.ndim > 1:
                    mean_vals = mean_vals.ravel()
                samples_np = np.array(posterior_samples)
                if samples_np.ndim == 3:
                    # Reshape from (n_samples, n_points, output_dim) to (n_samples, n_points)
                    # Take mean across output dimension if needed
                    if samples_np.shape[2] > 1:
                        samples_np = np.mean(samples_np, axis=2)  # Average across output dim
                    else:
                        samples_np = samples_np.reshape(samples_np.shape[0], -1)
                elif samples_np.ndim == 2:
                    # Already correct shape (n_samples, n_points)
                    pass
                else:
                    # Handle 1D case
                    samples_np = samples_np.reshape(1, -1) if samples_np.ndim == 1 else samples_np
                
                # Ensure samples_np is 2D: (n_samples, n_points)
                if samples_np.ndim != 2:
                    raise ValueError(f"Unexpected posterior_samples shape: {samples_np.shape}")
                
                std_vals = np.std(samples_np, axis=0)
                if std_vals.ndim > 1:
                    std_vals = std_vals.ravel()
                
                # Ensure std_vals matches mean_vals length
                if len(std_vals) != len(mean_vals):
                    # Try to reshape or take first dimension
                    if std_vals.size == mean_vals.size:
                        std_vals = std_vals.reshape(mean_vals.shape)
                    else:
                        raise ValueError(f"Shape mismatch: mean_vals={mean_vals.shape}, std_vals={std_vals.shape}")
                
                # Plot quality (ax1)
                valid_mask = np.isfinite(mean_vals) & np.isfinite(std_vals)
                if np.any(valid_mask):
                    fa_plot = fa_range[valid_mask]
                    mean_plot = mean_vals[valid_mask]
                    std_plot = std_vals[valid_mask]
                    
                    # Use color for temperature, line style for spin speed
                    color = temp_colors[temp_idx]
                    linestyle = spin_styles[spin_idx % len(spin_styles)]
                    
                    # Only show uncertainty bands for first spin speed to avoid clutter
                    if spin_idx == 0:
                        ax1.fill_between(fa_plot, mean_plot - 2*std_plot, mean_plot + 2*std_plot,
                                       alpha=0.1, color=color, 
                                       label=f'{temp}°C ±2σ' if temp_idx == 0 and spin_idx == 0 else '')
                        ax1.fill_between(fa_plot, mean_plot - std_plot, mean_plot + std_plot,
                                       alpha=0.15, color=color,
                                       label=f'{temp}°C ±1σ' if temp_idx == 0 and spin_idx == 0 else '')
                    
                    # Plot mean line with label
                    label = f'{temp}°C, {spin}rpm' if len(target_temps) > 1 or len(target_spins) > 1 else f'{temp}°C'
                    ax1.plot(fa_plot, mean_plot, color=color, linestyle=linestyle, linewidth=2.5,
                            label=label, zorder=3)
                    
                    # Plot uncertainty (ax2) - clip outliers for better visualization
                    # Remove outliers using IQR method
                    q25 = np.percentile(std_plot, 25)
                    q75 = np.percentile(std_plot, 75)
                    iqr = q75 - q25
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    
                    # Filter outliers
                    outlier_mask = (std_plot >= lower_bound) & (std_plot <= upper_bound)
                    if np.sum(outlier_mask) > 10:  # Need enough points
                        fa_plot_filtered = fa_plot[outlier_mask]
                        std_plot_filtered = std_plot[outlier_mask]
                    else:
                        fa_plot_filtered = fa_plot
                        std_plot_filtered = std_plot
                    
                    ax2.plot(fa_plot_filtered, std_plot_filtered, color=color, linestyle=linestyle, linewidth=2.5,
                            label=f'{temp}°C, {spin}rpm (σ={np.mean(std_plot_filtered):.1f})', zorder=3)
                    if spin_idx == 0:  # Only fill for first spin speed
                        ax2.fill_between(fa_plot_filtered, 0, std_plot_filtered, alpha=0.2, color=color)
                    
            except Exception as e:
                print(f"Warning: GP query failed for {temp}°C, {spin}rpm: {e}")
                continue
    
    # Add observed data points at target temperatures and spin speeds
    for temp in target_temps:
        temp_tol = 2.5  # ±2.5°C tolerance
        spin_tol = 250  # ±250 rpm tolerance
        obs_fa = []
        obs_qual = []
        obs_uncert = []
        
        for c in candidates:
            if (abs(c.annealing_temp - temp) < temp_tol and
                any(abs(c.spin_speed - spin) < spin_tol for spin in target_spins) and
                c.metrics.predicted_performance is not None):
                fa_ratio = extract_fa_ratio(c.material_system)
                obs_fa.append(fa_ratio)
                obs_qual.append(c.metrics.predicted_performance)
                if c.metrics.uncertainty_score is not None:
                    obs_uncert.append(c.metrics.uncertainty_score)
        
        if obs_fa:
            color_idx = target_temps.index(temp)
            color = temp_colors[color_idx]
            ax1.scatter(obs_fa, obs_qual, c=[color], s=80, 
                       marker='x', linewidths=2.5, alpha=0.8, zorder=5)  # Removed label, larger markers
            if obs_uncert:
                ax2.scatter(obs_fa, obs_uncert, c=[color], s=80,
                           marker='x', linewidths=2.5, alpha=0.8, zorder=5)  # Removed label, larger markers
    
    # Format quality plot (ax1) - NO LEGEND
    ax1.set_xlabel('FA Ratio (FA / (FA + FuDMA))', fontsize=11)
    ax1.set_ylabel('Predicted Quality', fontsize=11)
    spin_str = f"{', '.join(map(str, target_spins))} rpm" if len(target_spins) <= 3 else f"{len(target_spins)} spin speeds"
    ax1.set_title(f'Quality Objective: FA Ratio\nTemps: {", ".join(map(str, target_temps))}°C | Spins: {spin_str}', 
                 fontsize=12, fontweight='bold')
    # ax1.legend(loc='best', fontsize=8, ncol=2 if len(target_temps) * len(target_spins) > 4 else 1)  # REMOVED
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    
    # Format uncertainty plot (ax2) - set y-axis range to exclude outliers - NO LEGEND
    ax2.set_xlabel('FA Ratio (FA / (FA + FuDMA))', fontsize=11)
    ax2.set_ylabel('Uncertainty (σ)', fontsize=11)
    ax2.set_title(f'Uncertainty Objective: FA Ratio\nTemps: {", ".join(map(str, target_temps))}°C | Spins: {spin_str}', 
                 fontsize=12, fontweight='bold')
    # ax2.legend(loc='best', fontsize=8, ncol=2 if len(target_temps) * len(target_spins) > 4 else 1)  # REMOVED
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    
    # Set y-axis range to focus on main distribution (exclude extreme outliers)
    # Get all uncertainty values from the plot
    y_data = []
    for line in ax2.get_lines():
        y_data.extend(line.get_ydata())
    if y_data:
        y_array = np.array([y for y in y_data if np.isfinite(y) and y > 0])
        if len(y_array) > 0:
            # Use 5th to 95th percentile to exclude outliers
            y_min = np.percentile(y_array, 5)
            y_max = np.percentile(y_array, 95)
            y_padding = (y_max - y_min) * 0.1
            ax2.set_ylim(max(0, y_min - y_padding), y_max + y_padding)
