import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from typing import List, Optional
from src.models import ExperimentParams
import gpax
import jax.numpy as jnp
from src.plotting_enhanced import (
    plot_gp_learning_diagnostics,
    plot_gp_1d_slice_actual_values,
    plot_uncertainty_heatmap,
    plot_multi_objective_gp_surfaces
)
from src.plotting_models import (
    plot_composition_gp,
    plot_decision_tree_importance,
    plot_tree_predictions_vs_actual
)
from src.plotting_shap import plot_shap_beeswarm
from src.plotting_recommendations import plot_tree_recommendations


def extract_fa_ratio(material_system) -> float:
    """
    Extract FA ratio from material system for binary FA/FuDMA compositions.
    Returns FA fraction (0-1) where 1 = pure FA, 0 = pure FuDMA.
    """
    fa_ratio = 0.0
    fudma_ratio = 0.0
    total = 0.0
    
    for comp in material_system.a_site:
        name = comp.precursor.name.upper()
        if 'FA' in name and 'FUDMA' not in name:
            fa_ratio += comp.ratio
        elif 'FUDMA' in name or 'FUDMA' in name.upper():
            fudma_ratio += comp.ratio
        total += comp.ratio
    
    if total > 0:
        return fa_ratio / total
    return 0.5  # Default to middle if unknown


def plot_gp_2d_surface(ax, learner, candidates, x_feature='fa_ratio', y_feature='temp'):
    """
    Plot 2D GP surface showing how quality varies across parameter space.
    For multi-objective, this shows the actual learned function.
    """
    if not learner.is_trained or learner.gp_model is None:
        ax.text(0.5, 0.5, 'GP Not Trained', ha='center', va='center', 
               transform=ax.transAxes, fontsize=12)
        return
    
    import gpax
    import jax.numpy as jnp
    from src.learner import extract_fa_ratio_from_composition
    
    # Create a grid for the 2D surface
    if x_feature == 'fa_ratio':
        x_range = np.linspace(0, 1, 50)
        x_label = 'FA Ratio'
    else:
        # Get range from learner if available
        if learner.X_min is not None:
            t_min = learner.X_min[1]
            t_max = learner.X_max[1]
            x_range = np.linspace(t_min, t_max, 50)
        else:
            x_range = np.linspace(60, 160, 50)
        x_label = 'Temperature (°C)'
    
    if y_feature == 'temp':
        if learner.X_min is not None:
            t_min = learner.X_min[1]
            t_max = learner.X_max[1]
            y_range = np.linspace(t_min, t_max, 50)
        else:
            y_range = np.linspace(60, 160, 50)
        y_label = 'Temperature (°C)'
    elif y_feature == 'spin':
        if learner.X_min is not None:
            s_min = learner.X_min[2]
            s_max = learner.X_max[2]
            y_range = np.linspace(s_min, s_max, 50)
        else:
            y_range = np.linspace(2000, 6000, 50)
        y_label = 'Spin Speed (rpm)'
    else:
        y_range = np.linspace(2000, 6000, 50)
        y_label = 'Spin Speed (rpm)'
    
    # Get median of third dimension for fixed slice
    if candidates:
        if x_feature == 'fa_ratio':
            fixed_spin = np.median([c.spin_speed for c in candidates])
            fixed_temp = np.median([c.annealing_temp for c in candidates])
        else:
            fixed_fa = np.median([extract_fa_ratio(c.material_system) for c in candidates])
            fixed_spin = np.median([c.spin_speed for c in candidates])
    else:
        fixed_spin = 4500
        fixed_temp = 140
        fixed_fa = 0.5
    
    # Create grid
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    
    # Build query points
    query_points = []
    for x_val in x_range:
        for y_val in y_range:
            if x_feature == 'fa_ratio':
                query_points.append([x_val, y_val, fixed_spin])
            else:
                query_points.append([fixed_fa, x_val, y_val])
    
    query_points = np.array(query_points)
    
    # Normalize using learner's normalization
    query_norm = learner._normalize(query_points)
    query_jax = jnp.array(query_norm)
    
    # Get GP predictions
    try:
        rng_key = gpax.utils.get_keys(1)[0]
        posterior_mean, posterior_samples = learner.gp_model.predict(rng_key, query_jax, n=100)
        mean_vals = np.array(posterior_mean).ravel()
        std_vals = np.std(np.array(posterior_samples), axis=0).ravel()
        
        # Reshape to grid
        Z_mean = mean_vals.reshape(len(y_range), len(x_range))
        Z_std = std_vals.reshape(len(y_range), len(x_range))
        
        # Plot mean surface
        im = ax.contourf(X_grid, Y_grid, Z_mean, levels=20, cmap='PuRd', alpha=0.8)
        ax.contour(X_grid, Y_grid, Z_mean, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Predicted Quality', fontsize=10)
        
        # Overlay uncertainty
        ax.contour(X_grid, Y_grid, Z_std, levels=5, colors='red', alpha=0.4, linestyles='--', linewidths=1)
        
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title('GP 2D Surface: Quality Prediction', fontsize=12, fontweight='bold')
        
    except Exception as e:
        ax.text(0.5, 0.5, f'GP Query Failed:\n{str(e)[:50]}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=10)


def plot_gp_approximation(
    history_df: pd.DataFrame,
    candidates: List[ExperimentParams],
    learner,
    ax_main,
    ax_acq
):
    """
    Plot GP approximation along FA/FuDMA composition axis with acquisition function.
    Shows ACTUAL quality values (not normalized) and real uncertainty.
    """
    from src.learner import extract_fa_ratio_from_composition
    
    if learner is None or not learner.is_trained:
        ax_main.text(0.5, 0.5, 'GP Model Not Trained\n(Need 5+ experiments)', 
                     ha='center', va='center', transform=ax_main.transAxes, fontsize=12)
        ax_main.set_title('GP Approximation of Quality vs FA Ratio')
        ax_acq.text(0.5, 0.5, 'No Acquisition Data', 
                    ha='center', va='center', transform=ax_acq.transAxes, fontsize=12)
        return
    
    # --- Plot Raw Data with Error Bars (from history) ---
    if not history_df.empty:
        # Group history by unique conditions (FA ratio, temp, spin)
        hist_data = {}
        for _, row in history_df.iterrows():
            comp_str = str(row.get('Material_Composition', ''))
            fa_ratio = extract_fa_ratio_from_composition(comp_str)
            temp = row.get('Anneal_Temp_C', row.get('Temp_C', None))
            spin = row.get('Spin_Speed_rpm', None)
            
            if pd.isna(fa_ratio) or pd.isna(temp) or pd.isna(spin):
                continue
            
            # Use FA ratio as key (or could use full condition)
            key = (round(fa_ratio, 3), round(float(temp), 1), round(float(spin), 0))
            
            if key not in hist_data:
                hist_data[key] = {'fa': fa_ratio, 'qualities': []}
            
            quality = learner.calculate_quality_target(row)
            hist_data[key]['qualities'].append(quality)
        
        # Plot raw data points with error bars
        if hist_data:
            hist_fa = []
            hist_quality_mean = []
            hist_quality_std = []
            
            for key, data in hist_data.items():
                if data['qualities']:
                    hist_fa.append(data['fa'])
                    qualities = np.array(data['qualities'])
                    hist_quality_mean.append(np.mean(qualities))
                    hist_quality_std.append(np.std(qualities) if len(qualities) > 1 else 0.0)
            
            if hist_fa:
                hist_fa = np.array(hist_fa)
                hist_quality_mean = np.array(hist_quality_mean)
                hist_quality_std = np.array(hist_quality_std)
                
                # Plot with error bars
                ax_main.errorbar(hist_fa, hist_quality_mean, yerr=hist_quality_std, 
                               fmt='rx', markersize=10, capsize=5, capthick=2,
                               linewidth=2, label='Observations (mean ± std)', zorder=5)
    
    # --- Plot GP Predictions: Use candidate predictions directly, binned by FA ratio ---
    # This approach shows what the GP actually predicts for real candidate conditions
    from src.learner import extract_fa_ratio_from_composition
    
    if candidates:
        # Extract FA ratios and predictions from candidates
        fa_ratios = []
        predicted_means = []
        predicted_stds = []
        acq_scores = []
        
        for c in candidates:
            if c.metrics.predicted_performance is not None and c.metrics.uncertainty_score is not None:
                fa_ratio = extract_fa_ratio(c.material_system)
                fa_ratios.append(fa_ratio)
                predicted_means.append(c.metrics.predicted_performance)
                predicted_stds.append(c.metrics.uncertainty_score)
                acq_scores.append(c.metrics.acquisition_score if c.metrics.acquisition_score else 0)
        
        if fa_ratios:
            fa_array = np.array(fa_ratios)
            mean_array = np.array(predicted_means)
            std_array = np.array(predicted_stds)
            acq_array = np.array(acq_scores)
            
            # Bin by FA ratio for smooth visualization
            n_bins = 30
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            binned_means = []
            binned_stds = []
            binned_acq = []
            valid_bins = []
            
            for i in range(n_bins):
                mask = (fa_array >= bin_edges[i]) & (fa_array < bin_edges[i+1])
                if np.sum(mask) > 0:
                    valid_bins.append(bin_centers[i])
                    binned_means.append(np.mean(mean_array[mask]))
                    # Combined uncertainty: model uncertainty + spread of predictions
                    binned_stds.append(np.sqrt(np.mean(std_array[mask]**2) + np.var(mean_array[mask])))
                    binned_acq.append(np.mean(acq_array[mask]))
            
            if len(valid_bins) > 0:
                valid_bins = np.array(valid_bins)
                binned_means = np.array(binned_means)
                binned_stds = np.array(binned_stds)
                binned_acq = np.array(binned_acq)
                
                # Interpolate for smooth curve
                from scipy.interpolate import interp1d
                try:
                    fa_range = np.linspace(0, 1, 200)
                    mean_interp = interp1d(valid_bins, binned_means, kind='cubic', 
                                          bounds_error=False, fill_value='extrapolate')
                    std_interp = interp1d(valid_bins, binned_stds, kind='linear',
                                         bounds_error=False, fill_value='extrapolate')
                    acq_interp = interp1d(valid_bins, binned_acq, kind='linear',
                                         bounds_error=False, fill_value='extrapolate')
                    
                    mean_smooth = mean_interp(fa_range)
                    std_smooth = std_interp(fa_range)
                    acq_smooth = acq_interp(fa_range)
                except:
                    # If interpolation fails, use binned values directly
                    mean_smooth = binned_means
                    std_smooth = binned_stds
                    acq_smooth = binned_acq
                    fa_range = valid_bins
            else:
                mean_smooth = []
                std_smooth = []
                acq_smooth = []
        else:
            mean_smooth = []
            std_smooth = []
            acq_smooth = []
    else:
        mean_smooth = []
        std_smooth = []
        acq_smooth = []
        fa_ratios = []
        predicted_means = []
        predicted_stds = []
        acq_scores = []
        
        # Filter candidates to those near median temp/spin for better 1D visualization
        temp_tol = 20  # ±20°C tolerance
        spin_tol = 1000  # ±1000 rpm tolerance
        
        for c in candidates:
            if (c.metrics.predicted_performance is not None and 
                c.metrics.uncertainty_score is not None and
                abs(c.annealing_temp - median_temp) < temp_tol and
                abs(c.spin_speed - median_spin) < spin_tol):
                fa_ratio = extract_fa_ratio(c.material_system)
                fa_ratios.append(fa_ratio)
                predicted_means.append(c.metrics.predicted_performance)
                predicted_stds.append(c.metrics.uncertainty_score)
                acq_scores.append(c.metrics.acquisition_score if c.metrics.acquisition_score else 0)
        
        if not fa_ratios:
            # Use all candidates if filtered set is empty
            for c in candidates:
                if c.metrics.predicted_performance is not None and c.metrics.uncertainty_score is not None:
                    fa_ratio = extract_fa_ratio(c.material_system)
                    fa_ratios.append(fa_ratio)
                    predicted_means.append(c.metrics.predicted_performance)
                    predicted_stds.append(c.metrics.uncertainty_score)
                    acq_scores.append(c.metrics.acquisition_score if c.metrics.acquisition_score else 0)
        
        if not fa_ratios:
            if not hist_data:
                ax_main.text(0.5, 0.5, 'No GP predictions or observations available', 
                             ha='center', va='center', transform=ax_main.transAxes, fontsize=12)
            return
        
        fa_ratios = np.array(fa_ratios)
        predicted_means = np.array(predicted_means)
        predicted_stds = np.array(predicted_stds)
        acq_scores = np.array(acq_scores)
        
        # Sort and bin for smoother visualization
        sort_idx = np.argsort(fa_ratios)
        fa_sorted = fa_ratios[sort_idx]
        mean_sorted = predicted_means[sort_idx]
        std_sorted = predicted_stds[sort_idx]
        acq_sorted = acq_scores[sort_idx]
        
        # Bin the data
        n_bins = 50
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mean_binned = []
        std_binned = []
        acq_binned = []
        fa_binned = []
        
        for i in range(n_bins):
            mask = (fa_sorted >= bin_edges[i]) & (fa_sorted < bin_edges[i+1])
            if np.sum(mask) > 0:
                fa_binned.append(bin_centers[i])
                mean_binned.append(np.mean(mean_sorted[mask]))
                std_binned.append(np.sqrt(np.mean(std_sorted[mask]**2) + np.var(mean_sorted[mask])))
                acq_binned.append(np.mean(acq_sorted[mask]))
        
        if len(fa_binned) > 0:
            from scipy.interpolate import interp1d
            fa_binned = np.array(fa_binned)
            mean_binned = np.array(mean_binned)
            std_binned = np.array(std_binned)
            acq_binned = np.array(acq_binned)
            
            if len(fa_binned) >= 2:
                mean_interp = interp1d(fa_binned, mean_binned, kind='linear', 
                                      bounds_error=False, fill_value='extrapolate')
                std_interp = interp1d(fa_binned, std_binned, kind='linear',
                                     bounds_error=False, fill_value='extrapolate')
                acq_interp = interp1d(fa_binned, acq_binned, kind='linear',
                                     bounds_error=False, fill_value='extrapolate')
                
                mean_smooth = mean_interp(fa_range)
                std_smooth = std_interp(fa_range)
                acq_smooth = acq_interp(fa_range)
            else:
                mean_smooth = np.full_like(fa_range, mean_binned[0])
                std_smooth = np.full_like(fa_range, std_binned[0])
                acq_smooth = np.full_like(fa_range, acq_binned[0])
        else:
            if not hist_data:
                ax_main.text(0.5, 0.5, 'No GP predictions available', 
                             ha='center', va='center', transform=ax_main.transAxes, fontsize=12)
            return
        
    # Plot uncertainty bands
    if len(mean_smooth) > 0 and len(std_smooth) > 0:
        # Ensure no NaN or Inf values
        valid_mask = np.isfinite(mean_smooth) & np.isfinite(std_smooth)
        if np.any(valid_mask):
            fa_plot = fa_range[valid_mask]
            mean_plot = mean_smooth[valid_mask]
            std_plot = std_smooth[valid_mask]
            
            # Use PuRd colormap
            puRd_color = plt.cm.PuRd(0.7)
            ax_main.fill_between(fa_plot, 
                                 mean_plot - 2*std_plot, 
                                 mean_plot + 2*std_plot,
                                 alpha=0.2, color=puRd_color, zorder=1)
            ax_main.fill_between(fa_plot, 
                                 mean_plot - std_plot, 
                                 mean_plot + std_plot,
                                 alpha=0.3, color=puRd_color, zorder=2)
            
            # Plot GP mean line (using candidate predictions, no fixed temp/spin)
            ax_main.plot(fa_plot, mean_plot, color=puRd_color, linewidth=2.5, 
                        zorder=3)
    
    ax_main.set_xlabel('FA Ratio (FA / (FA + FuDMA))', fontsize=11)
    ax_main.set_ylabel('Quality Target Score', fontsize=11)
    ax_main.set_title('GP Approximation: Quality vs Composition', fontsize=13, fontweight='bold')
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlim(0, 1)
    ax_main.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_main.set_xticklabels(['Pure FuDMA', '25% FA', '50% FA', '75% FA', 'Pure FA'])
    
    # --- Acquisition Plot ---
    if 'acq_smooth' in locals() and len(acq_smooth) > 0:
        valid_acq = np.isfinite(acq_smooth)
        if np.any(valid_acq):
            fa_acq = fa_range[valid_acq]
            acq_plot = acq_smooth[valid_acq]
            
            # Use PuRd colormap
            puRd_color = plt.cm.PuRd(0.6)
            ax_acq.fill_between(fa_acq, 0, acq_plot, alpha=0.4, color=puRd_color, zorder=1)
            ax_acq.plot(fa_acq, acq_plot, color=puRd_color, linewidth=2, zorder=2)
            
            max_idx = np.argmax(acq_plot)
            max_fa = fa_acq[max_idx]
            max_acq = acq_plot[max_idx]
            
            ax_acq.axvline(x=max_fa, color=plt.cm.PuRd(0.9), linestyle='--', linewidth=1.5, alpha=0.7, zorder=3)
            ax_acq.scatter([max_fa], [max_acq], c=plt.cm.PuRd(0.9), s=150, zorder=5, 
                           marker='*', edgecolors='darkred', linewidths=1)
            
            ax_acq.set_xlabel('FA Ratio (FA / (FA + FuDMA))', fontsize=11)
            ax_acq.set_ylabel('Acquisition Score (UCB)', fontsize=11)
            ax_acq.set_title('Acquisition Function: Where to Explore Next', fontsize=12)
            ax_acq.grid(True, alpha=0.3)
            ax_acq.set_xlim(0, 1)
            ax_acq.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax_acq.set_xticklabels(['Pure FuDMA', '25% FA', '50% FA', '75% FA', 'Pure FA'])
        else:
            ax_acq.text(0.5, 0.5, 'No valid acquisition data', 
                        ha='center', va='center', transform=ax_acq.transAxes, fontsize=12)
    else:
        ax_acq.text(0.5, 0.5, 'No Acquisition Data Available', 
                    ha='center', va='center', transform=ax_acq.transAxes, fontsize=12)
        ax_acq.set_title('Acquisition Function')


def plot_cycle_improvement(ax, history_file: str = 'data/optimization_history.csv'):
    """Plot optimization progress across cycles."""
    import os
    if os.path.exists(history_file):
        try:
            opt_history = pd.read_csv(history_file)
            if len(opt_history) > 0:
                cycles = opt_history['cycle'].values
                best_qualities = opt_history['best_quality'].values
                # Handle missing uncertainty column
                if 'avg_uncertainty_top10' in opt_history.columns:
                    uncertainties = opt_history['avg_uncertainty_top10'].values
                else:
                    uncertainties = np.zeros_like(cycles)  # Default to zeros if missing
                
                # Plot best quality over cycles
                # Use PuRd colormap
                puRd_color = plt.cm.PuRd(0.7)
                ax.plot(cycles, best_qualities, 'o-', color=puRd_color, linewidth=2.5, 
                       markersize=10, label='Best Quality', zorder=3)
                
                # Add improvement annotations
                if len(cycles) > 1:
                    for i in range(1, len(cycles)):
                        improvement = (best_qualities[i] - best_qualities[i-1]) / (best_qualities[i-1] + 1e-10) * 100
                        if abs(improvement) > 1:  # Only show significant changes
                            ax.annotate(f'{improvement:+.1f}%', 
                                       xy=(cycles[i], best_qualities[i]),
                                       xytext=(5, 5), textcoords='offset points',
                                       fontsize=8, color='green' if improvement > 0 else 'red',
                                       fontweight='bold')
                
                # Plot uncertainty on secondary y-axis (only if we have valid data)
                if len(uncertainties) > 0 and np.any(uncertainties > 0):
                    ax2 = ax.twinx()
                    ax2.plot(cycles, uncertainties, 's-', color=plt.cm.PuRd(0.5), linewidth=2.5,
                            markersize=10, alpha=0.8, label='Avg Uncertainty (Top 10)', zorder=2)
                    ax2.set_ylabel('Average Uncertainty (Top 10)', fontsize=10, color=plt.cm.PuRd(0.5))
                    ax2.tick_params(axis='y', labelcolor=plt.cm.PuRd(0.5), labelsize=9)
                    
                    # Add convergence threshold line
                    ax2.axhline(y=10.0, color='red', linestyle=':', alpha=0.7, 
                               linewidth=2, label='Convergence Threshold (10)', zorder=1)
                    
                    # Set y-axis range for uncertainty (exclude extreme outliers)
                    uncert_valid = uncertainties[uncertainties > 0]
                    if len(uncert_valid) > 0:
                        uncert_min = np.percentile(uncert_valid, 5)
                        uncert_max = np.percentile(uncert_valid, 95)
                        uncert_padding = (uncert_max - uncert_min) * 0.2
                        ax2.set_ylim(max(0, uncert_min - uncert_padding), uncert_max + uncert_padding)
                    
                else:
                    # If no uncertainty data, add note
                    ax.text(0.5, 0.98, 'No uncertainty data available', 
                           transform=ax.transAxes, fontsize=9, 
                           verticalalignment='top', ha='center',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
                
                ax.set_xlabel('Optimization Cycle', fontsize=11)
                ax.set_ylabel('Best Quality Score', fontsize=11, color=puRd_color)
                ax.tick_params(axis='y', labelcolor=puRd_color)
                ax.set_title('Cycle-by-Cycle Improvement', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                return True
        except Exception as e:
            pass
    
    ax.text(0.5, 0.5, 'No optimization history found\n(Run iterative optimization to track cycles)', 
           ha='center', va='center', transform=ax.transAxes, fontsize=11)
    ax.set_title('Cycle-by-Cycle Improvement', fontsize=12)
    return False


def plot_uncertainty_distribution(ax, candidates: List[ExperimentParams]):
    """Plot uncertainty distribution across candidates."""
    uncertainties = [c.metrics.uncertainty_score for c in candidates 
                    if c.metrics.uncertainty_score is not None]
    
    if uncertainties:
        uncertainties = np.array(uncertainties)
        
        # Histogram of uncertainties
        # Use PuRd colormap
        puRd_color = plt.cm.PuRd(0.7)
        ax.hist(uncertainties, bins=30, alpha=0.7, color=puRd_color, 
               edgecolor='darkred', linewidth=0.5)
        
        # Add statistics
        mean_unc = np.mean(uncertainties)
        median_unc = np.median(uncertainties)
        std_unc = np.std(uncertainties)
        
        ax.axvline(mean_unc, color=plt.cm.PuRd(0.9), linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_unc:.1f}')
        ax.axvline(median_unc, color='green', linestyle='--', linewidth=2, 
                  label=f'Median: {median_unc:.1f}')
        
        # Add text box with statistics
        stats_text = f'Mean: {mean_unc:.1f}\nMedian: {median_unc:.1f}\nStd: {std_unc:.1f}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Uncertainty Score (σ)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Uncertainty Distribution Across Candidates', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No uncertainty data available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_title('Uncertainty Distribution', fontsize=12)


def plot_pareto_front(ax, candidates: List[ExperimentParams], show_all: bool = False):
    """Plot Pareto front for multi-objective optimization."""
    from src.multi_objective import calculate_pareto_front
    
    # Get candidates with valid predictions
    valid_cands = [c for c in candidates 
                  if c.metrics.predicted_performance is not None 
                  and c.metrics.uncertainty_score is not None]
    
    if len(valid_cands) < 2:
        ax.text(0.5, 0.5, 'Need at least 2 candidates\nwith predictions for Pareto front', 
               ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_title('Pareto Front (Quality vs Uncertainty)', fontsize=12)
        return
    
    # Extract objectives
    qualities = np.array([c.metrics.predicted_performance for c in valid_cands])
    uncertainties = np.array([c.metrics.uncertainty_score for c in valid_cands])
    
    # Find Pareto front
    pareto_candidates = calculate_pareto_front(valid_cands, objectives=['quality', 'uncertainty'])
    
    if pareto_candidates:
        pareto_qualities = np.array([c.metrics.predicted_performance for c in pareto_candidates])
        pareto_uncertainties = np.array([c.metrics.uncertainty_score for c in pareto_candidates])
        
        # Sort by quality for plotting
        sort_idx = np.argsort(pareto_qualities)
        pareto_qualities = pareto_qualities[sort_idx]
        pareto_uncertainties = pareto_uncertainties[sort_idx]
        
        # Plot all candidates (light) - NO LABEL
        if show_all:
            ax.scatter(qualities, uncertainties, c='#FFE4E1', alpha=0.3, s=20, 
                      zorder=1)  # Very light red, removed label
        
        # Plot Pareto front with more visible markers - NO LABELS
        ax.plot(pareto_qualities, pareto_uncertainties, 'r-', linewidth=3, 
               color='#8B0000', zorder=3, alpha=0.8)  # Dark red, removed label
        # Make Pareto points much more visible - larger, brighter
        ax.scatter(pareto_qualities, pareto_uncertainties, c='#DC143C', s=400, 
                  marker='o', edgecolors='#8B0000', linewidths=3, zorder=5,
                  alpha=0.95)  # Medium red with dark red edge, removed label
        
        # Add numbers to Pareto points for clarity
        for i, (q, u) in enumerate(zip(pareto_qualities, pareto_uncertainties)):
            ax.annotate(f'{i+1}', (q, u), xytext=(0, 0), textcoords='offset points',
                       fontsize=12, fontweight='bold', color='white', ha='center', va='center',
                       zorder=6)
        
        # Highlight top candidates (make them less prominent so Pareto stands out) - NO LABEL
        top_cands = sorted(valid_cands, key=lambda x: x.rank_score, reverse=True)[:10]
        top_qual = [c.metrics.predicted_performance for c in top_cands]
        top_unc = [c.metrics.uncertainty_score for c in top_cands]
        ax.scatter(top_qual, top_unc, c='#FF6347', s=120, marker='*', 
                  edgecolors='#DC143C', linewidths=1.5, zorder=4,
                  alpha=0.7)  # Medium red, removed label
        
        # Add quadrant labels
        x_mid = np.median(qualities)
        y_mid = np.median(uncertainties)
        ax.axvline(x=x_mid, color='gray', linestyle=':', alpha=0.3, linewidth=1)
        ax.axhline(y=y_mid, color='gray', linestyle=':', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('Predicted Quality', fontsize=11)
        ax.set_ylabel('Uncertainty (σ)', fontsize=11)
        ax.set_title('Pareto Front: Quality vs Uncertainty Trade-off', fontsize=12, fontweight='bold')
        # ax.legend(loc='best', fontsize=9)  # REMOVED
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No Pareto-optimal candidates found', 
               ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_title('Pareto Front', fontsize=12)


def plot_optimization_progress(history_df: pd.DataFrame, candidates: List[ExperimentParams], 
                               learner: Optional[object] = None, output_file: str = "data/optimization_plot.png",
                               show_pareto: bool = False, comp_gp=None, tree_learner=None):
    """
    Generates comprehensive plots to visualize the optimization status.
    Includes: GP approximation, cycle improvement, uncertainty, Pareto front,
    composition GP learning, and decision tree learning.
    
    Args:
        history_df: Experiment history DataFrame
        candidates: List of candidate experiments
        learner: ActiveLearner (process GP)
        output_file: Path to save plot
        show_pareto: Whether to show Pareto front plots
        comp_gp: CompositionGP model (optional)
        tree_learner: ProcessTreeLearner model (optional)
    """
    if not candidates:
        return

    # Create figure - only showing plots 9-18 (removed plots 1-8)
    fig_height = 12
    fig = plt.figure(figsize=(24, fig_height))  # Wider to accommodate 4 columns
    
    # Use GridSpec for flexible layout - 3 rows, 4 columns for plots 9-18
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 1], 
                  hspace=0.5, wspace=0.4, left=0.05, right=0.98, top=0.96, bottom=0.05)
    
    # --- Row 1: Plots 9-12 (Quality by Parameter) ---
    # Plot 9: Quality by FA Ratio Range
    ax_fa_range = fig.add_subplot(gs[0, 0])
    from src.plotting_grouped import plot_quality_by_fa_range
    plot_quality_by_fa_range(ax_fa_range, history_df, candidates, learner=learner)
    
    # Plot 10: Quality by Antisolvent
    ax_antisolv = fig.add_subplot(gs[0, 1])
    from src.plotting_grouped import plot_quality_by_antisolvent
    plot_quality_by_antisolvent(ax_antisolv, history_df, candidates, learner=learner)
    
    # Plot 11: Quality by Temperature Range
    ax_temp_range = fig.add_subplot(gs[0, 2])
    from src.plotting_grouped import plot_quality_by_temperature_range
    plot_quality_by_temperature_range(ax_temp_range, history_df, candidates, learner=learner)
    
    # Plot 12: Quality by Spin Speed Range
    ax_spin_range = fig.add_subplot(gs[0, 3])
    from src.plotting_grouped import plot_quality_by_spin_speed_range
    plot_quality_by_spin_speed_range(ax_spin_range, history_df, candidates, learner=learner)
    
    # --- Row 2: Plots 13-16 ---
    # Plot 13: Exploration vs Exploitation
    ax_acq_qual = fig.add_subplot(gs[1, 0])
    
    # Filter candidates with valid predictions
    valid_cands = [c for c in candidates 
                   if c.metrics.acquisition_score is not None and c.metrics.predicted_performance is not None]
    
    if valid_cands:
        acq_scores = [c.metrics.acquisition_score for c in valid_cands]
        pred_quality = [c.metrics.predicted_performance for c in valid_cands]
        uncertainty = [c.metrics.uncertainty_score if c.metrics.uncertainty_score is not None else 0 
                      for c in valid_cands]
        
        # Define quadrants
        x_mid = np.median(pred_quality)
        y_mid = np.median(acq_scores)
        
        # Classify candidates into exploit vs explore
        exploit_cands = []  # High quality, high acquisition
        explore_cands = []  # Low quality, high acquisition
        other_cands = []
        
        for c, pred_q, acq in zip(valid_cands, pred_quality, acq_scores):
            if pred_q >= x_mid and acq >= y_mid:
                exploit_cands.append((pred_q, acq, c))
            elif pred_q < x_mid and acq >= y_mid:
                explore_cands.append((pred_q, acq, c))
            else:
                other_cands.append((pred_q, acq, c))
        
        # Plot all candidates (small, light) - NO LABEL
        if other_cands:
            other_pred = [x[0] for x in other_cands]
            other_acq = [x[1] for x in other_cands]
            ax_acq_qual.scatter(other_pred, other_acq, c='#FFE4E1', alpha=0.2, s=15, 
                              marker='o', zorder=1)  # Very light red, no label
        
        # Plot exploit candidates (dark red circles) - NO LABEL
        if exploit_cands:
            exploit_pred = [x[0] for x in exploit_cands]
            exploit_acq = [x[1] for x in exploit_cands]
            ax_acq_qual.scatter(exploit_pred, exploit_acq, c='#DC143C', alpha=0.7, s=80, 
                              marker='o', edgecolors='#8B0000', linewidths=1.5,
                              zorder=3)  # Medium red, no label
        
        # Plot explore candidates (medium red squares) - NO LABEL
        if explore_cands:
            explore_pred = [x[0] for x in explore_cands]
            explore_acq = [x[1] for x in explore_cands]
            ax_acq_qual.scatter(explore_pred, explore_acq, c='#FF6347', alpha=0.7, s=80, 
                              marker='s', edgecolors='#DC143C', linewidths=1.5,
                              zorder=3)  # Medium red, no label
        
        # Add quadrant lines
        ax_acq_qual.axvline(x=x_mid, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, zorder=2)
        ax_acq_qual.axhline(y=y_mid, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, zorder=2)
        
        # Highlight top 10 ranked candidates with numbered stars
        if len(candidates) > 0:
            sorted_cands = sorted(candidates, 
                                key=lambda x: x.rank_score if hasattr(x, 'rank_score') and x.rank_score else 0, 
                                reverse=True)[:10]
            
            top_pred = []
            top_acq = []
            top_labels = []
            
            for i, c in enumerate(sorted_cands, 1):
                if c.metrics.acquisition_score is not None and c.metrics.predicted_performance is not None:
                    top_pred.append(c.metrics.predicted_performance)
                    top_acq.append(c.metrics.acquisition_score)
                    top_labels.append(str(i))
            
            if top_pred:
                # Plot stars - NO LABEL, use red colors
                scatter = ax_acq_qual.scatter(top_pred, top_acq, c='#8B0000', s=250, 
                                            marker='*', edgecolors='#DC143C', linewidths=2,
                                            zorder=6, alpha=0.95)  # Dark red, no label
                
                # Add numbers next to stars (white for visibility on red)
                for i, (px, py, label) in enumerate(zip(top_pred, top_acq, top_labels)):
                    ax_acq_qual.annotate(label, (px, py), xytext=(0, 0), 
                                       textcoords='offset points', fontsize=10, 
                                       fontweight='bold', color='white', ha='center', va='center',
                                       zorder=7)
        
        # Add quadrant labels (simpler, no overlap)
        ax_acq_qual.text(0.02, 0.02, 'Low Q\nLow Acq', 
                        transform=ax_acq_qual.transAxes, fontsize=7,
                        verticalalignment='bottom', alpha=0.5)
        ax_acq_qual.text(0.98, 0.02, 'High Q\nLow Acq', 
                        transform=ax_acq_qual.transAxes, fontsize=7,
                        verticalalignment='bottom', horizontalalignment='right', alpha=0.5)
        ax_acq_qual.text(0.02, 0.98, 'Low Q\nHigh Acq', 
                        transform=ax_acq_qual.transAxes, fontsize=7,
                        verticalalignment='top', alpha=0.5)
        ax_acq_qual.text(0.98, 0.98, 'High Q\nHigh Acq', 
                        transform=ax_acq_qual.transAxes, fontsize=7,
                        verticalalignment='top', horizontalalignment='right', alpha=0.5)
        
        ax_acq_qual.set_xlabel('Predicted Quality', fontsize=10)
        ax_acq_qual.set_ylabel('Acquisition Score (UCB)', fontsize=10)
        ax_acq_qual.set_title('Exploration vs Exploitation', fontsize=11, fontweight='bold')
        ax_acq_qual.grid(True, alpha=0.3)
        
        # Add summary text instead of legend (compact)
        summary_text = f'Exploit: {len(exploit_cands)}\nExplore: {len(explore_cands)}\nTop 10: {len(top_pred) if top_pred else 0}'
        ax_acq_qual.text(0.02, 0.98, summary_text, transform=ax_acq_qual.transAxes,
                        fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    else:
        ax_acq_qual.text(0.5, 0.5, 'No Acquisition Data Available\n(GP model not trained)', 
                        ha='center', va='center', transform=ax_acq_qual.transAxes, fontsize=12)
        ax_acq_qual.set_title('Exploration vs Exploitation')
        ax_acq_qual.grid(True, alpha=0.3)
    
    # Plot 14: Composition GP plot - Always show, even if not trained
    ax_comp_gp = fig.add_subplot(gs[1, 1])
    # Check if comp_gp exists and is properly trained
    if comp_gp is not None:
        # Try to plot - the function will handle errors gracefully
        try:
            plot_composition_gp(ax_comp_gp, comp_gp, history_df)
        except Exception as e:
            # If plotting fails, show error message
            ax_comp_gp.text(0.5, 0.5, f'Composition GP Plot Error:\n{str(e)[:100]}\n\nCheck if GP is trained:\nis_trained={comp_gp.is_trained if hasattr(comp_gp, "is_trained") else "N/A"}', 
                           ha='center', va='center', transform=ax_comp_gp.transAxes, fontsize=9)
            ax_comp_gp.set_title('Composition GP: FA Ratio → Quality', fontsize=12)
            ax_comp_gp.grid(True, alpha=0.3)
    else:
        # Show placeholder with info about what this plot would show
        ax_comp_gp.text(0.5, 0.5, 'Composition GP Not Available\n\nThis plot shows:\n- FA ratio vs quality predictions\n- Training data points\n- Uncertainty bands\n- UCB acquisition function', 
                       ha='center', va='center', transform=ax_comp_gp.transAxes, fontsize=10)
        ax_comp_gp.set_title('Composition GP: FA Ratio → Quality', fontsize=12)
        ax_comp_gp.grid(True, alpha=0.3)
    
    # Plot 15: Decision Tree feature importance - Always show
    ax_tree_imp = fig.add_subplot(gs[1, 2])
    if tree_learner is not None and tree_learner.is_trained:
        plot_decision_tree_importance(ax_tree_imp, tree_learner)
    else:
        ax_tree_imp.text(0.5, 0.5, 'Decision Tree Not Available\n\nThis plot shows:\n- Feature importance rankings\n- Which parameters matter most', 
                        ha='center', va='center', transform=ax_tree_imp.transAxes, fontsize=10)
        ax_tree_imp.set_title('Decision Tree: Feature Importance', fontsize=12)
        ax_tree_imp.grid(True, alpha=0.3)
    
    # --- Row 3: Plots 17-18 ---
    # Plot 17: SHAP Feature Impact Analysis (shows feature impact)
    ax_shap = fig.add_subplot(gs[2, 0])
    if tree_learner is not None and tree_learner.is_trained and learner is not None:
        plot_shap_beeswarm(ax_shap, tree_learner, history_df, learner)
    else:
        ax_shap.text(0.5, 0.5, 'SHAP Not Available\n\nThis plot shows:\n- Which features impact quality most\n- FA ratio, spin, temp, etc.\n- Requires trained decision tree', 
                    ha='center', va='center', transform=ax_shap.transAxes, fontsize=10)
        ax_shap.set_title('SHAP Feature Impact Analysis', fontsize=12)
        ax_shap.grid(True, alpha=0.3)
    
    # Plot 18: Decision Tree predictions vs actual
    ax_tree_pred = fig.add_subplot(gs[2, 1])
    
    if tree_learner is not None and tree_learner.is_trained and learner is not None:
        plot_tree_predictions_vs_actual(ax_tree_pred, tree_learner, history_df, learner)
    else:
        ax_tree_pred.text(0.5, 0.5, 'Decision Tree Not Available\n\nThis plot shows:\n- Predicted vs actual quality\n- R² score and error metrics', 
                         ha='center', va='center', transform=ax_tree_pred.transAxes, fontsize=10)
        ax_tree_pred.set_title('Decision Tree: Predictions vs Actual', fontsize=12)
        ax_tree_pred.grid(True, alpha=0.3)
    
    # Plot 19: Decision Tree Recommendations
    ax_tree_rec = fig.add_subplot(gs[2, 2])
    
    if tree_learner is not None and tree_learner.is_trained:
        # Always show tree recommendations if trained, it's a useful alternative view
        plot_tree_recommendations(ax_tree_rec, candidates, tree_learner, history_df)
    else:
        ax_tree_rec.text(0.5, 0.5, 'Tree Recommendations\n\nNot Available (Need 5+ experiments)\n\nShows parameter regions favored\nby the Decision Tree model.', 
                        ha='center', va='center', transform=ax_tree_rec.transAxes, fontsize=10)
        ax_tree_rec.set_title('Tree Recommendations', fontsize=12)
        ax_tree_rec.grid(True, alpha=0.3)
    
    # Add main title
    # fig.suptitle('Gaussian Process Bayesian Optimization Dashboard\nFA/FuDMA Binary Perovskite System', 
    #              fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Optimization plots saved to {output_file}")
    plt.close()
