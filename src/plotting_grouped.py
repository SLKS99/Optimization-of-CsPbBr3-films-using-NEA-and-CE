"""
Grouped plotting functions to show data by antisolvent, solvent, and FA ratio ranges.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Dict, Tuple
from src.models import ExperimentParams
from src.plotting import extract_fa_ratio


def get_antisolvent_name(exp: ExperimentParams) -> str:
    """Get antisolvent name or 'None'."""
    if exp.antisolvent:
        name = exp.antisolvent.name
        # Shorten common names
        if 'Toluene' in name:
            return 'Toluene'
        elif 'Chlorobenzene' in name or 'CB' in name:
            return 'Chlorobenzene'
        elif 'Diethyl' in name:
            return 'Diethyl Ether'
        return name
    return 'None'


def get_fa_ratio_range(fa_ratio: float) -> str:
    """Categorize FA ratio into ranges."""
    if fa_ratio < 0.2:
        return 'Low FA (0-0.2)'
    elif fa_ratio < 0.4:
        return 'Low-Med FA (0.2-0.4)'
    elif fa_ratio < 0.6:
        return 'Mid FA (0.4-0.6)'
    elif fa_ratio < 0.8:
        return 'Mid-High FA (0.6-0.8)'
    else:
        return 'High FA (0.8-1.0)'


def plot_quality_by_antisolvent(ax, history_df: pd.DataFrame, candidates: List[ExperimentParams], learner=None):
    """Plot quality grouped by antisolvent type."""
    if history_df.empty:
        ax.text(0.5, 0.5, 'No historical data', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        return
    
    antisolvent_groups = {}
    
    # Initialize groups to ensure all antisolvents in history are present
    if 'Antisolvent_Name' in history_df.columns:
        for val in history_df['Antisolvent_Name'].unique():
            name = str(val).strip()
            if not name or name.lower() == 'nan' or pd.isna(val):
                name = 'None'
            antisolvent_groups[name] = []
    
    # From history
    if 'Antisolvent_Name' in history_df.columns:
        for _, row in history_df.iterrows():
            antisolv = row.get('Antisolvent_Name', 'None')
            # Handle NaN, None, empty string, and "nan" string - all mean no antisolvent
            if pd.isna(antisolv) or antisolv is None:
                antisolv = 'None'
            else:
                antisolv = str(antisolv).strip()
                if antisolv == '' or antisolv.lower() == 'nan':
                    antisolv = 'None'
            
            # Use same quality metric as learners (consistent scale)
            if learner is not None:
                quality = learner.calculate_quality_target(row)
            else:
                # Look for column with or without space
                quality = row.get('PL_Intensity', row.get('PL_Intensity ', None))
                if quality is not None:
                    quality = float(quality)
                
            if quality is not None and not pd.isna(quality):
                antisolvent_groups[antisolv].append(quality)
    
    # From candidates (if they have predictions)
    for c in candidates:
        antisolv = get_antisolvent_name(c)
        if c.metrics.predicted_performance is not None:
            if antisolv not in antisolvent_groups:
                antisolvent_groups[antisolv] = []
            antisolvent_groups[antisolv].append(c.metrics.predicted_performance)
    
    if not antisolvent_groups:
        ax.text(0.5, 0.5, 'No antisolvent data', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        return
    
    # Create box plot
    labels = []
    data = []
    colors = []
    
    # Use dark red colormap
    from matplotlib.colors import LinearSegmentedColormap
    red_colors = ['#8B0000', '#A52A2A', '#DC143C', '#FF4500', '#FF6347']
    red_cmap = LinearSegmentedColormap.from_list('dark_red', red_colors, N=256)
    
    all_values = []
    for i, (antisolv, qualities) in enumerate(sorted(antisolvent_groups.items())):
        labels.append(antisolv)
        data.append(qualities)
        colors.append(red_cmap(0.3 + 0.5 * i / max(1, len(antisolvent_groups) - 1)))
        all_values.extend(qualities)
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6, showfliers=False)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Adjust Y-axis to focus on the boxes, ignoring extreme outliers
    if all_values:
        q1, q3 = np.percentile(all_values, [25, 75])
        iqr = q3 - q1
        # Use a hard floor for visualization if values are extremely negative
        lower_bound = max(-500, q1 - 1.5 * iqr)
        upper_bound = min(max(all_values), q3 + 1.5 * iqr)
        
        # Ensure we show the bulk of the data
        p5, p95 = np.percentile(all_values, [5, 95])
        lower_bound = max(-600, min(lower_bound, p5))
        upper_bound = max(upper_bound, p95)
        
        padding = (upper_bound - lower_bound) * 0.1
        ax.set_ylim(lower_bound - padding, upper_bound + padding)

    ax.set_ylabel('Quality Score', fontsize=11)
    ax.set_xlabel('Antisolvent Type', fontsize=11)
    ax.set_title('Quality by Antisolvent Type', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')


def plot_quality_by_fa_range(ax, history_df: pd.DataFrame, candidates: List[ExperimentParams], learner=None):
    """Plot quality grouped by FA ratio ranges."""
    # Clear axis first to prevent overlapping plots
    ax.clear()
    
    # Group by FA ratio range
    fa_range_groups = {
        'Low FA (0-0.2)': [],
        'Low-Med FA (0.2-0.4)': [],
        'Mid FA (0.4-0.6)': [],
        'Mid-High FA (0.6-0.8)': [],
        'High FA (0.8-1.0)': []
    }
    
    # From history
    if 'Material_Composition' in history_df.columns:
        for _, row in history_df.iterrows():
            comp_str = str(row.get('Material_Composition', ''))
            fa_ratio = extract_fa_ratio_from_composition(comp_str)
            fa_range = get_fa_ratio_range(fa_ratio)
            
            # Use same quality metric as learners (consistent scale)
            if learner is not None:
                quality = learner.calculate_quality_target(row)
            else:
                # Look for column with or without space
                quality = row.get('PL_Intensity', row.get('PL_Intensity ', None))
                if quality is not None:
                    quality = float(quality)
                
            if quality is not None and not pd.isna(quality):
                fa_range_groups[fa_range].append(quality)
    
    # From candidates
    for c in candidates:
        fa_ratio = extract_fa_ratio(c.material_system)
        fa_range = get_fa_ratio_range(fa_ratio)
        if c.metrics.predicted_performance is not None:
            fa_range_groups[fa_range].append(c.metrics.predicted_performance)
    
    # Create box plot
    labels = list(fa_range_groups.keys())
    data = [fa_range_groups[label] for label in labels]
    
    # Filter out empty groups
    filtered_data = []
    filtered_labels = []
    for d, l in zip(data, labels):
        if len(d) > 0:
            filtered_data.append(d)
            filtered_labels.append(l)
    
    if not filtered_data:
        ax.text(0.5, 0.5, 'No FA ratio data', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        return
    
    # Use dark red colormap
    from matplotlib.colors import LinearSegmentedColormap
    red_colors = ['#8B0000', '#A52A2A', '#DC143C', '#FF4500', '#FF6347']
    red_cmap = LinearSegmentedColormap.from_list('dark_red', red_colors, N=256)
    
    bp = ax.boxplot(filtered_data, labels=filtered_labels, patch_artist=True, widths=0.6, showfliers=False)
    for i, patch in enumerate(bp['boxes']):
        color = red_cmap(0.3 + 0.5 * i / max(1, len(filtered_data) - 1))
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Adjust Y-axis to focus on the boxes, ignoring extreme outliers
    all_values = [v for group in filtered_data for v in group]
    if all_values:
        q1, q3 = np.percentile(all_values, [25, 75])
        iqr = q3 - q1
        # Use a hard floor for visualization if values are extremely negative
        lower_bound = max(-500, q1 - 1.5 * iqr)
        upper_bound = min(max(all_values), q3 + 1.5 * iqr)
        
        # Ensure we show the bulk of the data
        p5, p95 = np.percentile(all_values, [5, 95])
        lower_bound = max(-600, min(lower_bound, p5))
        upper_bound = max(upper_bound, p95)
        
        padding = (upper_bound - lower_bound) * 0.1
        ax.set_ylim(lower_bound - padding, upper_bound + padding)

    ax.set_ylabel('Quality Score', fontsize=11)
    ax.set_xlabel('FA Ratio Range', fontsize=11)
    ax.set_title('Quality by FA Ratio Range', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')


def plot_quality_by_temperature_range(ax, history_df: pd.DataFrame, candidates: List[ExperimentParams], learner=None):
    """Plot quality grouped by temperature ranges."""
    # Group by temperature
    temp_groups = {
        '< 100°C': [],
        '100-130°C': [],
        '130-135°C': [],
        '135-140°C': [],
        '140-145°C': [],
        '145-150°C': [],
        '150+°C': []
    }
    
    # From history
    temp_col = 'Anneal_Temp_C' if 'Anneal_Temp_C' in history_df.columns else 'Temp_C'
    if temp_col in history_df.columns:
        for _, row in history_df.iterrows():
            temp = float(row[temp_col]) if pd.notna(row[temp_col]) else None
            if temp is not None:
                if temp < 100:
                    group = '< 100°C'
                elif temp < 130:
                    group = '100-130°C'
                elif temp < 135:
                    group = '130-135°C'
                elif temp < 140:
                    group = '135-140°C'
                elif temp < 145:
                    group = '140-145°C'
                elif temp < 150:
                    group = '145-150°C'
                else:
                    group = '150+°C'
                
                # Use same quality metric as learners (consistent scale)
                if learner is not None:
                    quality = learner.calculate_quality_target(row)
                else:
                    # Look for column with or without space
                    quality = row.get('PL_Intensity', row.get('PL_Intensity ', None))
                    if quality is not None:
                        quality = float(quality)
                
                if quality is not None and not pd.isna(quality):
                    temp_groups[group].append(quality)
    
    # From candidates
    for c in candidates:
        temp = c.annealing_temp
        if temp < 100:
            group = '< 100°C'
        elif temp < 130:
            group = '100-130°C'
        elif temp < 135:
            group = '130-135°C'
        elif temp < 140:
            group = '135-140°C'
        elif temp < 145:
            group = '140-145°C'
        elif temp < 150:
            group = '145-150°C'
        else:
            group = '150+°C'
        
        if c.metrics.predicted_performance is not None:
            temp_groups[group].append(c.metrics.predicted_performance)
    
    # Create box plot
    labels = list(temp_groups.keys())
    data = [temp_groups[label] for label in labels]
    
    # Filter out empty groups
    filtered_data = []
    filtered_labels = []
    for d, l in zip(data, labels):
        if len(d) > 0:
            filtered_data.append(d)
            filtered_labels.append(l)
    
    if not filtered_data:
        ax.text(0.5, 0.5, 'No temperature data', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        return
    
    # Use dark red colormap
    from matplotlib.colors import LinearSegmentedColormap
    red_colors = ['#8B0000', '#A52A2A', '#DC143C', '#FF4500', '#FF6347']
    red_cmap = LinearSegmentedColormap.from_list('dark_red', red_colors, N=256)
    
    # HIDE OUTLIERS to focus on the boxes
    bp = ax.boxplot(filtered_data, labels=filtered_labels, patch_artist=True, widths=0.6, showfliers=False)
    for i, patch in enumerate(bp['boxes']):
        color = red_cmap(0.3 + 0.5 * i / max(1, len(filtered_data) - 1))
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Adjust Y-axis to focus on the boxes, ignoring extreme outliers
    all_values = [v for group in filtered_data for v in group]
    if all_values:
        q1, q3 = np.percentile(all_values, [25, 75])
        iqr = q3 - q1
        # Harder crop: ignore everything below -500 for visualization
        lower_bound = max(-500, q1 - 1.5 * iqr)
        upper_bound = min(max(all_values), q3 + 1.5 * iqr)
        
        # Ensure we show the bulk of the data
        p5, p95 = np.percentile(all_values, [5, 95])
        lower_bound = max(-600, min(lower_bound, p5))
        upper_bound = max(upper_bound, p95)
        
        padding = (upper_bound - lower_bound) * 0.1
        ax.set_ylim(lower_bound - padding, upper_bound + padding)

    ax.set_ylabel('Quality Score', fontsize=11)
    ax.set_xlabel('Temperature Range', fontsize=11)
    ax.set_title('Quality by Temperature Range', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')


def plot_quality_by_spin_speed_range(ax, history_df: pd.DataFrame, candidates: List[ExperimentParams], learner=None):
    """Plot quality grouped by spin speed ranges."""
    # Group by spin speed
    spin_groups = {
        '< 2000 rpm': [],
        '2000-3000 rpm': [],
        '3000-4000 rpm': [],
        '4000-4500 rpm': [],
        '4500-5000 rpm': [],
        '5000+ rpm': []
    }
    
    # From history
    spin_col = 'Spin_Speed_rpm'
    if spin_col in history_df.columns:
        for _, row in history_df.iterrows():
            spin = float(row[spin_col]) if pd.notna(row[spin_col]) else None
            if spin is not None:
                if spin < 2000:
                    group = '< 2000 rpm'
                elif spin < 3000:
                    group = '2000-3000 rpm'
                elif spin < 4000:
                    group = '3000-4000 rpm'
                elif spin < 4500:
                    group = '4000-4500 rpm'
                elif spin < 5000:
                    group = '4500-5000 rpm'
                else:
                    group = '5000+ rpm'
                
                # Use same quality metric as learners (consistent scale)
                if learner is not None:
                    quality = learner.calculate_quality_target(row)
                else:
                    # Look for column with or without space
                    quality = row.get('PL_Intensity', row.get('PL_Intensity ', None))
                    if quality is not None:
                        quality = float(quality)
                
                if quality is not None and not pd.isna(quality):
                    spin_groups[group].append(quality)
    
    # From candidates
    for c in candidates:
        spin = c.spin_speed
        if spin < 2000:
            group = '< 2000 rpm'
        elif spin < 3000:
            group = '2000-3000 rpm'
        elif spin < 4000:
            group = '3000-4000 rpm'
        elif spin < 4500:
            group = '4000-4500 rpm'
        elif spin < 5000:
            group = '4500-5000 rpm'
        else:
            group = '5000+ rpm'
        
        if c.metrics.predicted_performance is not None:
            spin_groups[group].append(c.metrics.predicted_performance)
    
    # Create box plot
    labels = list(spin_groups.keys())
    data = [spin_groups[label] for label in labels]
    
    # Filter out empty groups
    filtered_data = []
    filtered_labels = []
    for d, l in zip(data, labels):
        if len(d) > 0:
            filtered_data.append(d)
            filtered_labels.append(l)
    
    if not filtered_data:
        ax.text(0.5, 0.5, 'No spin speed data', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        return
    
    # Use dark red colormap
    from matplotlib.colors import LinearSegmentedColormap
    red_colors = ['#8B0000', '#A52A2A', '#DC143C', '#FF4500', '#FF6347']
    red_cmap = LinearSegmentedColormap.from_list('dark_red', red_colors, N=256)
    
    bp = ax.boxplot(filtered_data, labels=filtered_labels, patch_artist=True, widths=0.6, showfliers=False)
    for i, patch in enumerate(bp['boxes']):
        color = red_cmap(0.3 + 0.5 * i / max(1, len(filtered_data) - 1))
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Adjust Y-axis to focus on the boxes, ignoring extreme outliers
    all_values = [v for group in filtered_data for v in group]
    if all_values:
        q1, q3 = np.percentile(all_values, [25, 75])
        iqr = q3 - q1
        # Use a hard floor for visualization if values are extremely negative
        lower_bound = max(-500, q1 - 1.5 * iqr)
        upper_bound = min(max(all_values), q3 + 1.5 * iqr)
        
        # Ensure we show the bulk of the data
        p5, p95 = np.percentile(all_values, [5, 95])
        lower_bound = max(-600, min(lower_bound, p5))
        upper_bound = max(upper_bound, p95)
        
        padding = (upper_bound - lower_bound) * 0.1
        ax.set_ylim(lower_bound - padding, upper_bound + padding)

    ax.set_ylabel('Quality Score', fontsize=11)
    ax.set_xlabel('Spin Speed Range', fontsize=11)
    ax.set_title('Quality by Spin Speed Range', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')


def extract_fa_ratio_from_composition(comp_str: str) -> float:
    """Extract FA ratio from composition string."""
    import re
    fa_match = re.search(r'FA([\d.]+)', comp_str, re.IGNORECASE)
    fudma_match = re.search(r'FuDMA([\d.]+)', comp_str, re.IGNORECASE)
    
    fa_val = float(fa_match.group(1)) if fa_match else 0.0
    fudma_val = float(fudma_match.group(1)) if fudma_match else 0.0
    
    total = fa_val + fudma_val
    if total > 0:
        return fa_val / total
    return 0.5
