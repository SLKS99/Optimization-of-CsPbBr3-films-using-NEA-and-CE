"""
Plot showing Decision Tree-based recommendations when GP isn't learning.
Shows which parameter combinations the Decision Tree predicts will perform best.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Optional
from src.models import ExperimentParams
from src.learner import extract_fa_ratio_from_composition


def plot_tree_recommendations(ax, candidates: List[ExperimentParams], 
                             tree_learner, history_df: pd.DataFrame):
    """
    Plot Decision Tree recommendations showing best parameter combinations.
    
    Shows:
    - Top candidates by Tree predictions
    - Parameter ranges that Tree recommends
    - Comparison with historical best performers
    """
    if tree_learner is None or not tree_learner.is_trained:
        ax.text(0.5, 0.5, 'Decision Tree Not Trained', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Tree Recommendations', fontsize=12)
        return
    
    # Get candidates with tree predictions
    valid_cands = [c for c in candidates 
                  if c.metrics.tree_predicted_quality is not None]
    
    if len(valid_cands) == 0:
        ax.text(0.5, 0.5, 'No Tree Predictions Available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Tree Recommendations', fontsize=12)
        return
    
    # Sort by tree predictions
    valid_cands.sort(key=lambda x: x.metrics.tree_predicted_quality if x.metrics.tree_predicted_quality is not None else -1e6, reverse=True)
    top_n = min(20, len(valid_cands))
    top_cands = valid_cands[:top_n]
    
    # Extract parameters from top candidates
    fa_ratios = []
    spin_speeds = []
    temps = []
    tree_preds = []
    
    for c in top_cands:
        try:
            fa_ratio = extract_fa_ratio_from_composition(str(c.material_system))
            fa_ratios.append(fa_ratio)
            spin_speeds.append(c.spin_speed)
            temps.append(c.annealing_temp)
            tree_preds.append(c.metrics.tree_predicted_quality)
        except Exception:
            continue
    
    if not fa_ratios:
        ax.text(0.5, 0.5, 'Error processing top candidates', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Tree Recommendations', fontsize=12)
        return
    
    # Create scatter plot: FA ratio vs Spin Speed, colored by Tree prediction
    fa_ratios = np.array(fa_ratios)
    spin_speeds = np.array(spin_speeds)
    temps = np.array(temps)
    tree_preds = np.array(tree_preds)
    
    # Normalize predictions for color mapping
    if len(tree_preds) > 0 and np.max(tree_preds) > np.min(tree_preds):
        pred_norm = (tree_preds - np.min(tree_preds)) / (np.max(tree_preds) - np.min(tree_preds))
    else:
        pred_norm = np.ones_like(tree_preds)
    
    # Use red colormap
    colors = plt.cm.PuRd(0.3 + 0.5 * pred_norm)
    
    # Plot scatter
    scatter = ax.scatter(fa_ratios, spin_speeds, c=tree_preds, s=100, 
                        alpha=0.7, cmap=plt.cm.PuRd, edgecolors='black', 
                        linewidths=1, zorder=3)
    
    # Add temperature as marker size (larger = higher temp)
    # Normalize temps for size
    if len(temps) > 0 and np.max(temps) > np.min(temps):
        temp_norm = 50 + 150 * (temps - np.min(temps)) / (np.max(temps) - np.min(temps))
    else:
        temp_norm = 100 * np.ones_like(temps)
    
    # Replot with size variation
    ax.scatter(fa_ratios, spin_speeds, c=tree_preds, s=temp_norm,
              alpha=0.6, cmap=plt.cm.PuRd, edgecolors='black', 
              linewidths=1.5, zorder=2)
    
    # Highlight top 5
    top5_cands = top_cands[:5]
    top5_fa = [extract_fa_ratio_from_composition(str(c.material_system)) for c in top5_cands]
    top5_spin = [c.spin_speed for c in top5_cands]
    ax.scatter(top5_fa, top5_spin, c='darkred', s=200, marker='*',
              edgecolors='black', linewidths=2, zorder=5, alpha=0.9)
    
    # Add labels for top 3
    for i, c in enumerate(top5_cands[:3]):
        fa = extract_fa_ratio_from_composition(str(c.material_system))
        ax.annotate(f'{i+1}', (fa, c.spin_speed), 
                   xytext=(0, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold', color='white',
                   ha='center', zorder=6)
    
    # Add recommended ranges (from your data analysis)
    # Best FA: 0.6-0.8, Best Spin: 3000-4000
    ax.axvspan(0.6, 0.8, alpha=0.1, color='green', zorder=0)
    ax.axhspan(3000, 4000, alpha=0.1, color='green', zorder=0)
    
    # Ensure axes show the data points
    if len(spin_speeds) > 0:
        s_min, s_max = np.min(spin_speeds), np.max(spin_speeds)
        if s_min == s_max:
            ax.set_ylim(s_min - 500, s_min + 500)
        else:
            padding = (s_max - s_min) * 0.1
            ax.set_ylim(s_min - padding, s_max + padding)
    else:
        ax.set_ylim(2000, 6000)

    ax.set_xlabel('FA Ratio', fontsize=11)
    ax.set_ylabel('Spin Speed (rpm)', fontsize=11)
    ax.set_title('Tree Recommendations: Top Candidates', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Tree Predicted Quality', fontsize=10)
    
    # Add summary text
    if len(top_cands) > 0:
        avg_fa = np.mean(fa_ratios)
        avg_spin = np.mean(spin_speeds)
        avg_temp = np.mean(temps)
        avg_pred = np.mean(tree_preds)
        
        summary_text = f"Top {top_n} Avg:\n"
        summary_text += f"FA: {avg_fa:.2f}\n"
        summary_text += f"Spin: {avg_spin:.0f} rpm\n"
        summary_text += f"Temp: {avg_temp:.0f}°C\n"
        summary_text += f"Pred: {avg_pred:.1f}"
        
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
