"""
Redesigned visualization module for Dual GP Analysis
Focus: Clarity, actionability, and intuitive presentation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import config


def create_enhanced_visualizations(results, output_dir):
    """
    Create enhanced, publication-quality visualizations.
    
    Two main outputs:
    1. Dashboard: Strategic overview with key insights
    2. Detailed Analysis: Comprehensive slice-based analysis
    """
    import os
    
    # Unnormalize data for visualization
    ranges = np.array([
        [config.CS_MIN, config.CS_MAX],
        [config.NEA_MIN, config.NEA_MAX],
        [config.CE_MIN, config.CE_MAX]
    ])
    
    # Extract results and unnormalize
    Xnew_norm = np.array(results['Xnew'])
    
    # Unnormalize manually (same as unnormalize_data_with_ranges)
    Xnew = np.zeros_like(Xnew_norm)
    for i in range(Xnew_norm.shape[1]):
        r_min, r_max = ranges[i]
        Xnew[:, i] = Xnew_norm[:, i] * (r_max - r_min) + r_min
        # Round to grid
        Xnew[:, i] = np.round(Xnew[:, i] / config.GRID_RESOLUTION) * config.GRID_RESOLUTION
    
    # Extract components
    Cs = Xnew[:, 0]
    NEA = Xnew[:, 1]
    CE = Xnew[:, 2]
    
    y_pred = results['y_pred']
    y_std = results.get('y_std', None)
    acq = results['acq']
    acq_tune = results['acq_tune']
    adjust_tune_score = results['adjust_tune_score']
    
    # Unnormalize next batch
    X_next_batch_int_norm = np.array(results['X_next_batch_int'])
    X_next_batch_int_acq_norm = np.array(results['X_next_batch_int_acq'])
    
    X_next_batch_int = np.zeros_like(X_next_batch_int_norm)
    X_next_batch_int_acq = np.zeros_like(X_next_batch_int_acq_norm)
    
    if len(X_next_batch_int_norm) > 0:
        for i in range(X_next_batch_int_norm.shape[1]):
            r_min, r_max = ranges[i]
            X_next_batch_int[:, i] = X_next_batch_int_norm[:, i] * (r_max - r_min) + r_min
            X_next_batch_int[:, i] = np.round(X_next_batch_int[:, i] / config.GRID_RESOLUTION) * config.GRID_RESOLUTION
    
    if len(X_next_batch_int_acq_norm) > 0:
        for i in range(X_next_batch_int_acq_norm.shape[1]):
            r_min, r_max = ranges[i]
            X_next_batch_int_acq[:, i] = X_next_batch_int_acq_norm[:, i] * (r_max - r_min) + r_min
            X_next_batch_int_acq[:, i] = np.round(X_next_batch_int_acq[:, i] / config.GRID_RESOLUTION) * config.GRID_RESOLUTION
    
    # Convert to numpy arrays
    adjust_tune_score_arr = np.array(adjust_tune_score)
    y_std_arr = np.array(y_std)
    y_pred_arr = np.array(y_pred)
    acq_tune_arr = np.array(acq_tune)
    acq_arr = np.array(acq)
    
    # Calculate derived metrics
    stability_score = 1.0 - adjust_tune_score_arr
    stability_score = (stability_score - stability_score.min()) / (stability_score.max() - stability_score.min() + 1e-10)
    
    uncertainty_norm = (y_std_arr - y_std_arr.min()) / (y_std_arr.max() - y_std_arr.min() + 1e-10)
    low_uncertainty_score = 1.0 - uncertainty_norm
    
    combined_score = 0.6 * stability_score + 0.4 * low_uncertainty_score
    
    # ==========================================
    # FIGURE 1: Executive Dashboard
    # ==========================================
    fig1 = plt.figure(figsize=(22, 14))
    gs1 = gridspec.GridSpec(3, 5, figure=fig1, hspace=0.4, wspace=0.35,
                           left=0.05, right=0.98, top=0.94, bottom=0.05)
    
    # ========== ROW 1: 3D Visualizations & Key Metrics ==========
    
    # Plot 1: 3D Surface - Predicted Intensity
    ax1 = fig1.add_subplot(gs1[0, 0:2], projection='3d')
    plot_3d_surface(ax1, Cs, NEA, CE, y_pred_arr, 'Cs', 'NEA', 'Intensity',
                    'Predicted Intensity Landscape', 'viridis')
    
    # Plot 2: 3D Surface - Acquisition Function  
    ax2 = fig1.add_subplot(gs1[0, 2:4], projection='3d')
    plot_3d_surface(ax2, Cs, NEA, CE, acq_tune_arr, 'Cs', 'NEA', 'Acquisition',
                    'Acquisition Function Landscape', 'plasma')
    
    # Plot 3: Pareto Front (Decision Support)
    ax3 = fig1.add_subplot(gs1[0, 4])
    plot_pareto_front(ax3, stability_score, low_uncertainty_score, y_pred_arr, combined_score)
    
    # ========== ROW 2: Critical 2D Projections ==========
    
    # Plot 4: NEA vs CE - Predictions (Most Important)
    ax4 = fig1.add_subplot(gs1[1, 0])
    plot_2d_heatmap(ax4, NEA, CE, y_pred_arr, 'NEA Conc.', 'CE Conc.',
                    'Predicted Intensity\n(NEA vs CE)', 'viridis',
                    X_next_batch_int_acq[:, [1, 2]] if len(X_next_batch_int_acq) > 0 else None)
    
    # Plot 5: NEA vs CE - Acquisition
    ax5 = fig1.add_subplot(gs1[1, 1])
    plot_2d_heatmap(ax5, NEA, CE, acq_tune_arr, 'NEA Conc.', 'CE Conc.',
                    'Acquisition Value\n(NEA vs CE)', 'plasma',
                    X_next_batch_int_acq[:, [1, 2]] if len(X_next_batch_int_acq) > 0 else None, 'yellow')
    
    # Plot 6: Cs vs NEA - Predictions
    ax6 = fig1.add_subplot(gs1[1, 2])
    plot_2d_heatmap(ax6, Cs, NEA, y_pred_arr, 'Cs Conc.', 'NEA Conc.',
                    'Predicted Intensity\n(Cs vs NEA)', 'viridis',
                    X_next_batch_int_acq[:, [0, 1]] if len(X_next_batch_int_acq) > 0 else None)
    
    # Plot 7: Cs vs CE - Predictions
    ax7 = fig1.add_subplot(gs1[1, 3])
    plot_2d_heatmap(ax7, Cs, CE, y_pred_arr, 'Cs Conc.', 'CE Conc.',
                    'Predicted Intensity\n(Cs vs CE)', 'viridis',
                    X_next_batch_int_acq[:, [0, 2]] if len(X_next_batch_int_acq) > 0 else None)
    
    # Plot 8: Uncertainty Map
    ax8 = fig1.add_subplot(gs1[1, 4])
    plot_2d_heatmap(ax8, NEA, CE, y_std_arr, 'NEA Conc.', 'CE Conc.',
                    'Model Uncertainty\n(NEA vs CE)', 'Reds')
    
    # ========== ROW 3: Analysis & Insights ==========
    
    # Plot 9: Component Effects
    ax9 = fig1.add_subplot(gs1[2, 0])
    plot_component_effects(ax9, Cs, NEA, CE, y_pred_arr)
    
    # Plot 10: Acquisition Distribution
    ax10 = fig1.add_subplot(gs1[2, 1])
    plot_acquisition_histogram(ax10, acq_tune_arr)
    
    # Plot 11: Prediction Quality Map
    ax11 = fig1.add_subplot(gs1[2, 2])
    plot_prediction_quality(ax11, y_pred_arr, y_std_arr, combined_score)
    
    # Plot 12: Top Recommendations Table
    ax12 = fig1.add_subplot(gs1[2, 3])
    plot_recommendations_table(ax12, X_next_batch_int_acq, Cs, NEA, CE, 
                              y_pred_arr, acq_tune_arr)
    
    # Plot 13: Summary Statistics
    ax13 = fig1.add_subplot(gs1[2, 4])
    plot_summary_stats(ax13, Cs, NEA, CE, y_pred_arr, y_std_arr, acq_tune_arr,
                      combined_score, X_next_batch_int_acq)
    
    plt.suptitle('Dual GP Optimization: Executive Dashboard', 
                fontsize=18, fontweight='bold', y=0.98)
    
    dashboard_file = os.path.join(output_dir, 'optimization_dashboard.png')
    plt.savefig(dashboard_file, dpi=config.PLOT_DPI, bbox_inches='tight')
    print(f"[OK] Executive dashboard saved: {dashboard_file}")
    plt.close()
    
    # ==========================================
    # FIGURE 2: Detailed CE Slice Analysis
    # ==========================================
    fig2 = plt.figure(figsize=(20, 12))
    gs2 = gridspec.GridSpec(3, 4, figure=fig2, hspace=0.4, wspace=0.35)
    
    ce_levels = np.linspace(CE.min(), CE.max(), 12)
    
    for i, ce_val in enumerate(ce_levels):
        if i >= 12:
            break
        
        row = i // 4
        col = i % 4
        ax = fig2.add_subplot(gs2[row, col])
        
        ce_mask = np.abs(CE - ce_val) < (CE.max() - CE.min()) / 24
        
        if np.sum(ce_mask) >= 10:
            plot_2d_heatmap(ax, Cs[ce_mask], NEA[ce_mask], y_pred_arr[ce_mask],
                          'Cs', 'NEA', f'CE = {ce_val:.3f}', 'viridis',
                          X_next_batch_int_acq[:, [0, 1]] if len(X_next_batch_int_acq) > 0 else None,
                          show_colorbar=True, fontsize_small=True)
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, f'CE = {ce_val:.3f}\nInsufficient Data',
                   ha='center', va='center', fontsize=10, transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.suptitle(f'Predicted Intensity: CE Concentration Slices ({config.PRECURSOR1} vs {config.PRECURSOR2})',
                fontsize=16, fontweight='bold', y=0.98)
    
    slices_file = os.path.join(output_dir, 'ce_slice_analysis.png')
    plt.savefig(slices_file, dpi=config.PLOT_DPI, bbox_inches='tight')
    print(f"[OK] CE slice analysis saved: {slices_file}")
    plt.close()
    
    # ==========================================
    # FIGURE 3: Comparative Analysis (NEW!)
    # ==========================================
    fig3 = plt.figure(figsize=(18, 10))
    gs3 = gridspec.GridSpec(2, 3, figure=fig3, hspace=0.35, wspace=0.3)
    
    # Comparison plots for different metrics
    ax_comp1 = fig3.add_subplot(gs3[0, 0])
    plot_2d_heatmap(ax_comp1, NEA, CE, y_pred_arr, 'NEA', 'CE',
                    'Predicted Intensity', 'viridis')
    
    ax_comp2 = fig3.add_subplot(gs3[0, 1])
    plot_2d_heatmap(ax_comp2, NEA, CE, acq_tune_arr, 'NEA', 'CE',
                    'Acquisition (Tuned)', 'plasma')
    
    ax_comp3 = fig3.add_subplot(gs3[0, 2])
    plot_2d_heatmap(ax_comp3, NEA, CE, stability_score.reshape(NEA.shape), 
                    'NEA', 'CE', 'Stability Score', 'RdYlGn')
    
    ax_comp4 = fig3.add_subplot(gs3[1, 0])
    plot_2d_heatmap(ax_comp4, Cs, NEA, y_pred_arr, 'Cs', 'NEA',
                    'Predicted Intensity', 'viridis')
    
    ax_comp5 = fig3.add_subplot(gs3[1, 1])
    plot_2d_heatmap(ax_comp5, Cs, NEA, acq_tune_arr, 'Cs', 'NEA',
                    'Acquisition (Tuned)', 'plasma')
    
    ax_comp6 = fig3.add_subplot(gs3[1, 2])
    plot_2d_heatmap(ax_comp6, Cs, CE, acq_tune_arr, 'Cs', 'CE',
                    'Acquisition (Tuned)', 'plasma')
    
    plt.suptitle('Comparative Analysis: All Metrics & Projections',
                fontsize=16, fontweight='bold', y=0.98)
    
    comparison_file = os.path.join(output_dir, 'comparative_analysis.png')
    plt.savefig(comparison_file, dpi=config.PLOT_DPI, bbox_inches='tight')
    print(f"[OK] Comparative analysis saved: {comparison_file}")
    plt.close()
    
    print(f"\n{'='*60}")
    print(f"All visualizations generated successfully!")
    print(f"  1. Executive Dashboard: {dashboard_file}")
    print(f"  2. CE Slice Analysis: {slices_file}")
    print(f"  3. Comparative Analysis: {comparison_file}")
    print(f"{'='*60}\n")


# ==========================================
# Helper Plotting Functions
# ==========================================

def plot_3d_surface(ax, x, y, z, z_vals, xlabel, ylabel, zlabel, title, cmap):
    """Create a clean 3D surface plot"""
    # Aggregate by averaging CE dimension
    ce_mid = z.mean()
    mask = np.abs(z - ce_mid) < (z.max() - z.min()) / 4
    
    if np.sum(mask) < 20:
        mask = np.ones(len(z), dtype=bool)
    
    xi = np.linspace(x.min(), x.max(), 40)
    yi = np.linspace(y.min(), y.max(), 40)
    Xi, Yi = np.meshgrid(xi, yi)
    
    points = np.column_stack([x[mask].ravel(), y[mask].ravel()])
    Zi = griddata(points, z_vals[mask].ravel(), (Xi, Yi), method='cubic')
    
    surf = ax.plot_surface(Xi, Yi, Zi, cmap=cmap, alpha=0.85, edgecolor='none',
                          linewidth=0, antialiased=True, shade=True, rcount=40, ccount=40)
    
    ax.set_xlabel(xlabel, fontsize=10, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=10)
    ax.set_zlabel(zlabel, fontsize=10, labelpad=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.view_init(elev=20, azim=135)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2)
    
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.08)


def plot_2d_heatmap(ax, x, y, z_vals, xlabel, ylabel, title, cmap, 
                   selected_points=None, marker_color='white', show_colorbar=True,
                   fontsize_small=False):
    """Create a clean 2D contour heatmap"""
    fs = 8 if fontsize_small else 10
    fs_title = 9 if fontsize_small else 11
    
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    Xi, Yi = np.meshgrid(xi, yi)
    
    points = np.column_stack([x.ravel(), y.ravel()])
    Zi = griddata(points, z_vals.ravel(), (Xi, Yi), method='cubic')
    
    contourf = ax.contourf(Xi, Yi, Zi, levels=25, cmap=cmap, alpha=0.95)
    ax.contour(Xi, Yi, Zi, levels=15, colors='black', alpha=0.15, linewidths=0.4)
    
    if show_colorbar:
        cbar = plt.colorbar(contourf, ax=ax, shrink=0.75, pad=0.02)
        cbar.ax.tick_params(labelsize=fs-2)
    
    if selected_points is not None and len(selected_points) > 0:
        # Filter to valid range
        valid_mask = ((selected_points[:, 0] >= x.min()) & 
                     (selected_points[:, 0] <= x.max()) &
                     (selected_points[:, 1] >= y.min()) & 
                     (selected_points[:, 1] <= y.max()))
        
        if np.sum(valid_mask) > 0:
            ax.scatter(selected_points[valid_mask, 0], selected_points[valid_mask, 1],
                      s=180, marker='*', c=marker_color, edgecolors='black',
                      linewidths=1.8, label='Next Batch', zorder=10)
            ax.legend(fontsize=fs-1, loc='best', framealpha=0.9)
    
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel(ylabel, fontsize=fs)
    ax.set_title(title, fontsize=fs_title, fontweight='bold', pad=6)
    ax.tick_params(labelsize=fs-1)
    ax.grid(True, alpha=0.2, linestyle='--')


def plot_pareto_front(ax, stability, uncertainty, intensity, combined):
    """Plot Pareto frontier for multi-objective optimization"""
    top_n = min(150, len(combined))
    top_indices = np.argsort(combined)[-top_n:]
    
    scatter = ax.scatter(stability[top_indices], uncertainty[top_indices],
                        c=intensity[top_indices], s=60, alpha=0.7, cmap='RdYlGn',
                        edgecolors='gray', linewidths=0.5)
    
    # Find Pareto optimal points
    pareto_indices = []
    for i in top_indices:
        is_dominated = False
        for j in top_indices:
            if (stability[j] >= stability[i] and uncertainty[j] >= uncertainty[i] and
                (stability[j] > stability[i] or uncertainty[j] > uncertainty[i])):
                is_dominated = True
                break
        if not is_dominated:
            pareto_indices.append(i)
    
    if len(pareto_indices) > 0:
        pareto_indices = np.array(pareto_indices)
        ax.scatter(stability[pareto_indices], uncertainty[pareto_indices],
                  s=280, marker='*', c='gold', edgecolors='darkred', linewidths=2.5,
                  label=f'Pareto Optimal ({len(pareto_indices)})', zorder=10)
    
    ax.set_xlabel('Stability', fontsize=10)
    ax.set_ylabel('Confidence\n(Low Uncertainty)', fontsize=10)
    ax.set_title('Pareto Front:\nStability vs Confidence', fontsize=11, fontweight='bold', pad=8)
    cbar = plt.colorbar(scatter, ax=ax, label='Intensity', shrink=0.7, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    if len(pareto_indices) > 0:
        ax.legend(fontsize=8, loc='lower left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([stability[top_indices].min() - 0.05, 1.05])
    ax.set_ylim([uncertainty[top_indices].min() - 0.05, 1.05])


def plot_component_effects(ax, cs, nea, ce, intensity):
    """Plot marginal effect of each component"""
    cs_bins = np.linspace(cs.min(), cs.max(), 8)
    nea_bins = np.linspace(nea.min(), nea.max(), 12)
    ce_bins = np.linspace(ce.min(), ce.max(), 8)
    
    tol_cs = (cs.max() - cs.min()) / 12
    tol_nea = (nea.max() - nea.min()) / 20
    tol_ce = (ce.max() - ce.min()) / 12
    
    cs_means = [intensity[np.abs(cs - c) < tol_cs].mean() 
                if np.sum(np.abs(cs - c) < tol_cs) > 0 else np.nan for c in cs_bins]
    nea_means = [intensity[np.abs(nea - n) < tol_nea].mean() 
                 if np.sum(np.abs(nea - n) < tol_nea) > 0 else np.nan for n in nea_bins]
    ce_means = [intensity[np.abs(ce - e) < tol_ce].mean() 
                if np.sum(np.abs(ce - e) < tol_ce) > 0 else np.nan for e in ce_bins]
    
    ax.plot(cs_bins, cs_means, 'o-', label='Cs', linewidth=2.5, markersize=8, color='#2E86DE')
    ax.plot(nea_bins, nea_means, 's-', label='NEA', linewidth=2.5, markersize=7, color='#EE5A6F')
    ax.plot(ce_bins, ce_means, '^-', label='CE', linewidth=2.5, markersize=7, color='#26DE81')
    
    ax.set_xlabel('Concentration', fontsize=10)
    ax.set_ylabel('Mean Intensity', fontsize=10)
    ax.set_title('Marginal Component Effects', fontsize=11, fontweight='bold', pad=8)
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)


def plot_acquisition_histogram(ax, acq_values):
    """Plot distribution of acquisition values"""
    ax.hist(acq_values, bins=60, color='coral', alpha=0.75, edgecolor='black', linewidth=0.8)
    
    percentiles = [50, 75, 90, 95]
    colors = ['blue', 'green', 'orange', 'red']
    for p, c in zip(percentiles, colors):
        val = np.percentile(acq_values, p)
        ax.axvline(val, color=c, linestyle='--', linewidth=2, alpha=0.8,
                  label=f'{p}th %ile: {val:.2f}')
    
    ax.set_xlabel('Acquisition Value', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Acquisition Distribution', fontsize=11, fontweight='bold', pad=8)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')


def plot_prediction_quality(ax, intensity, uncertainty, combined):
    """Scatter plot of prediction quality"""
    scatter = ax.scatter(intensity, uncertainty, c=combined, s=40, alpha=0.65,
                        cmap='viridis', edgecolors='none')
    
    ax.set_xlabel('Predicted Intensity', fontsize=10)
    ax.set_ylabel('Prediction Uncertainty', fontsize=10)
    ax.set_title('Prediction Quality Map', fontsize=11, fontweight='bold', pad=8)
    cbar = plt.colorbar(scatter, ax=ax, label='Combined Score', shrink=0.7, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)


def plot_recommendations_table(ax, next_batch, cs, nea, ce, intensity, acquisition):
    """Create a table of top recommended experiments"""
    ax.axis('off')
    
    if len(next_batch) == 0:
        ax.text(0.5, 0.5, 'No recommendations\navailable',
               ha='center', va='center', fontsize=12, transform=ax.transAxes)
        return
    
    # Find closest grid points
    table_data = []
    for idx, batch_comp in enumerate(next_batch[:min(10, len(next_batch))]):
        dist = np.sqrt(np.sum((np.column_stack([cs, nea, ce]) - batch_comp)**2, axis=1))
        grid_idx = np.argmin(dist)
        
        table_data.append([
            f"{idx+1}",
            f"{batch_comp[0]:.3f}",
            f"{batch_comp[1]:.3f}",
            f"{batch_comp[2]:.3f}",
            f"{intensity[grid_idx]:.0f}",
            f"{acquisition[grid_idx]:.3f}"
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['#', 'Cs', 'NEA', 'CE', 'Intensity', 'Acq.'],
                    cellLoc='center', loc='center',
                    colWidths=[0.08, 0.18, 0.18, 0.18, 0.18, 0.18])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)
    
    for i in range(6):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_edgecolor('white')
    
    for i in range(1, len(table_data) + 1):
        for j in range(6):
            table[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
            table[(i, j)].set_edgecolor('#bdc3c7')
    
    ax.set_title('Top 10 Recommended Experiments', fontsize=11, fontweight='bold', pad=25)


def plot_summary_stats(ax, cs, nea, ce, intensity, uncertainty, acquisition, 
                      combined, next_batch):
    """Create a summary statistics panel"""
    ax.axis('off')
    
    best_idx = np.argmax(combined)
    
    summary = f"""
╔═══════════════════════════════════╗
║   OPTIMIZATION SUMMARY            ║
╚═══════════════════════════════════╝

Search Space Configuration:
  {config.PRECURSOR1}:   {config.CS_MIN:.2f} → {config.CS_MAX:.2f}
  {config.PRECURSOR2}:  {config.NEA_MIN:.2f} → {config.NEA_MAX:.2f}
  {config.PRECURSOR3}:   {config.CE_MIN:.2f} → {config.CE_MAX:.2f}
  PbBr: {config.PBBR_FIXED:.2f} (fixed)

Grid & Recommendations:
  Total Grid Points:  {len(intensity):,}
  Next Batch Size:    {len(next_batch)}

Performance Metrics:
  Max Intensity:      {np.max(intensity):.1f}
  Min Uncertainty:    {np.min(uncertainty):.5f}
  Max Acquisition:    {np.max(acquisition):.4f}
  Mean Intensity:     {np.mean(intensity):.1f}

Best Composition Found:
  Cs:  {cs[best_idx]:.4f}
  NEA: {nea[best_idx]:.4f}
  CE:  {ce[best_idx]:.4f}
  Combined Score: {combined[best_idx]:.4f}
    """
    
    ax.text(0.05, 0.5, summary, fontsize=9, verticalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='#e8f8f5', 
                     alpha=0.4, edgecolor='#16a085', linewidth=2.5))
