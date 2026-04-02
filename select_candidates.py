#!/usr/bin/env python3
"""
Candidate Selection Helper Script
=================================
Helps you select the best candidates when GP isn't learning well.
Uses Decision Tree predictions and physics-based metrics.
"""

import pandas as pd
import numpy as np
import sys
import os
import yaml
from typing import List, Optional
from src.learner import extract_fa_ratio_from_composition, ActiveLearner

def load_config(path: str = 'config.yaml') -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_historical_best(n: int = 5):
    """Load history and find top performers based on Quality Target."""
    cfg = load_config()
    history_path = cfg.get('data', {}).get('history', 'data/templates/experiments_log.csv')
    
    if not os.path.exists(history_path):
        return None
    
    df = pd.read_csv(history_path)
    if df.empty:
        return None
        
    # Initialize learner to calculate quality targets consistent with optimization
    quality_config = cfg.get('quality_target', {})
    learner = ActiveLearner(quality_config=quality_config)
    
    # Calculate quality for each row
    qualities = []
    fa_ratios = []
    for _, row in df.iterrows():
        # Handle 3-day data like in main.py
        pl_initial = row.get('PL_Intensity ', row.get('PL_Intensity', 0))
        pl_3d = row.get('PL_Intensity_3d_nm', None)
        stability = row.get('Stability_Hours', 0)
        
        if pd.notna(pl_3d) and pd.notna(pl_initial) and pl_initial > 0:
            if pd.isna(stability) or stability == 0:
                retention = pl_3d / pl_initial
                stability = retention * 72.0
        
        # Create a clean row for calculation
        row_clean = row.copy()
        row_clean['Stability_Hours'] = stability
        
        q = learner.calculate_quality_target(row_clean)
        qualities.append(q)
        fa_ratios.append(extract_fa_ratio_from_composition(str(row.get('Material_Composition', ''))))
        
    df['Quality_Target'] = qualities
    df['FA_Ratio'] = fa_ratios
    
    # Sort and return top N
    return df.nlargest(n, 'Quality_Target')

def select_candidates(
    candidates_file: str = 'data/candidates_analysis.csv',
    n_select: int = 10,
    strategy: str = 'tree',  # 'tree', 'rank', 'filtered', 'pareto'
    fa_range: tuple = (0.6, 0.8),
    spin_range: tuple = (3000, 5000),
    temp_range: tuple = (140, 145),
    output_file: str = None
):
    """
    Select best candidates using specified strategy.
    
    Args:
        candidates_file: Path to candidates_analysis.csv
        n_select: Number of candidates to select
        strategy: Selection strategy
            - 'tree': Sort by Decision Tree predictions
            - 'rank': Sort by Rank Score (physics + ML)
            - 'filtered': Filter by parameter ranges, then sort by tree
            - 'pareto': Use Pareto front selection
        fa_range: (min, max) FA ratio range to consider
        spin_range: (min, max) spin speed range
        temp_range: (min, max) temperature range
        output_file: Optional path to save selected candidates
    """
    
    if not os.path.exists(candidates_file):
        print(f"Error: {candidates_file} not found.")
        print("Run optimization first to generate candidates.")
        return None
    
    df = pd.read_csv(candidates_file)
    
    if df.empty:
        print("Error: No candidates found in file.")
        return None
    
    print(f"Loaded {len(df)} candidates from {candidates_file}")
    
    # Extract FA ratio if not present
    if 'FA_Ratio' not in df.columns and 'Material_Composition' in df.columns:
        df['FA_Ratio'] = df['Material_Composition'].apply(
            lambda x: extract_fa_ratio_from_composition(str(x))
        )
    
    # Strategy 1: Decision Tree predictions
    if strategy == 'tree':
        if 'Tree_Predicted_Quality' not in df.columns:
            print("Warning: Tree_Predicted_Quality not found. Falling back to Rank_Score.")
            strategy = 'rank'
        else:
            selected = df.nlargest(n_select, 'Tree_Predicted_Quality')
            print(f"\nSelected {len(selected)} candidates by Decision Tree predictions:")
            print(f"  Quality range: {selected['Tree_Predicted_Quality'].min():.1f} to {selected['Tree_Predicted_Quality'].max():.1f}")
    
    # Strategy 2: Rank Score (physics + ML)
    if strategy == 'rank':
        if 'Rank_Score' not in df.columns:
            print("Error: Rank_Score not found. Cannot use rank strategy.")
            return None
        selected = df.nlargest(n_select, 'Rank_Score')
        print(f"\nSelected {len(selected)} candidates by Rank Score:")
        print(f"  Rank Score range: {selected['Rank_Score'].min():.1f} to {selected['Rank_Score'].max():.1f}")
    
    # Strategy 3: Filtered by parameter ranges
    if strategy == 'filtered':
        # Filter by parameter ranges
        filtered = df.copy()
        
        if 'FA_Ratio' in filtered.columns:
            filtered = filtered[
                (filtered['FA_Ratio'] >= fa_range[0]) & 
                (filtered['FA_Ratio'] <= fa_range[1])
            ]
            print(f"  After FA ratio filter ({fa_range[0]}-{fa_range[1]}): {len(filtered)} candidates")
        
        if 'Spin_Speed_rpm' in filtered.columns:
            filtered = filtered[
                (filtered['Spin_Speed_rpm'] >= spin_range[0]) & 
                (filtered['Spin_Speed_rpm'] <= spin_range[1])
            ]
            print(f"  After spin speed filter ({spin_range[0]}-{spin_range[1]}): {len(filtered)} candidates")
        
        if 'Anneal_Temp_C' in filtered.columns:
            filtered = filtered[
                (filtered['Anneal_Temp_C'] >= temp_range[0]) & 
                (filtered['Anneal_Temp_C'] <= temp_range[1])
            ]
            print(f"  After temperature filter ({temp_range[0]}-{temp_range[1]}): {len(filtered)} candidates")
        
        if len(filtered) == 0:
            print("Error: No candidates match filter criteria. Try relaxing ranges.")
            return None
        
        # Sort by tree predictions or rank score
        if 'Tree_Predicted_Quality' in filtered.columns:
            selected = filtered.nlargest(n_select, 'Tree_Predicted_Quality')
            print(f"\nSelected {len(selected)} candidates (filtered + sorted by Tree predictions):")
        elif 'Rank_Score' in filtered.columns:
            selected = filtered.nlargest(n_select, 'Rank_Score')
            print(f"\nSelected {len(selected)} candidates (filtered + sorted by Rank Score):")
        else:
            selected = filtered.head(n_select)
            print(f"\nSelected {len(selected)} candidates (filtered only):")
    
    # Strategy 4: Pareto front
    if strategy == 'pareto':
        from src.multi_objective import select_diverse_pareto_front
        from src.models import ExperimentParams
        
        # Need to convert DataFrame back to ExperimentParams
        # For now, use a simpler approach: select diverse candidates
        print("Pareto selection requires ExperimentParams objects.")
        print("Falling back to diverse selection by quality and uncertainty...")
        
        if 'Tree_Predicted_Quality' in df.columns and 'Uncertainty_Score' in df.columns:
            # Simple diversity: maximize quality, ensure some uncertainty diversity
            selected = df.nlargest(n_select, 'Tree_Predicted_Quality')
        else:
            selected = df.nlargest(n_select, 'Rank_Score')
    
    # Display summary
    print("\n" + "="*80)
    print("SELECTED CANDIDATES:")
    print("="*80)
    
    # Show key columns
    display_cols = ['Experiment_ID']
    if 'FA_Ratio' in selected.columns:
        display_cols.append('FA_Ratio')
    if 'Spin_Speed_rpm' in selected.columns:
        display_cols.append('Spin_Speed_rpm')
    if 'Anneal_Temp_C' in selected.columns:
        display_cols.append('Anneal_Temp_C')
    if 'Tree_Predicted_Quality' in selected.columns:
        display_cols.append('Tree_Predicted_Quality')
    if 'Rank_Score' in selected.columns:
        display_cols.append('Rank_Score')
    if 'Antisolvent_Name' in selected.columns:
        display_cols.append('Antisolvent_Name')
    
    available_cols = [c for c in display_cols if c in selected.columns]
    print(selected[available_cols].to_string(index=False))
    
    # HISTORY COMPARISON
    hist_best = get_historical_best(3)
    if hist_best is not None:
        print("\n" + "="*80)
        print("HISTORY COMPARISON (TOP 3 PAST RESULTS):")
        print("="*80)
        hist_cols = ['Material_Composition', 'FA_Ratio', 'Spin_Speed_rpm', 'Anneal_Temp_C', 'Quality_Target']
        avail_hist = [c for c in hist_cols if c in hist_best.columns]
        print(hist_best[avail_hist].to_string(index=False))
        
        # Simple summary
        top_hist_q = hist_best['Quality_Target'].max()
        top_cand_q = selected['Tree_Predicted_Quality'].max() if 'Tree_Predicted_Quality' in selected.columns else None
        
        print("\nSummary Comparison:")
        print(f"  Historical Best Quality: {top_hist_q:.1f}")
        if top_cand_q is not None:
            improvement = ((top_cand_q - top_hist_q) / top_hist_q * 100) if top_hist_q > 0 else 0
            print(f"  Candidate Predicted Quality: {top_cand_q:.1f} ({improvement:+.1f}% vs best history)")
        else:
            print("  (Candidate predictions not available for direct comparison)")

    # Statistics
    print("\n" + "="*80)
    print("STATISTICS:")
    print("="*80)
    if 'FA_Ratio' in selected.columns:
        print(f"FA Ratio range: {selected['FA_Ratio'].min():.3f} to {selected['FA_Ratio'].max():.3f}")
        print(f"FA Ratio mean: {selected['FA_Ratio'].mean():.3f}")
    if 'Spin_Speed_rpm' in selected.columns:
        print(f"Spin Speed range: {selected['Spin_Speed_rpm'].min():.0f} to {selected['Spin_Speed_rpm'].max():.0f} rpm")
        print(f"Spin Speed mean: {selected['Spin_Speed_rpm'].mean():.0f} rpm")
    if 'Anneal_Temp_C' in selected.columns:
        print(f"Temperature range: {selected['Anneal_Temp_C'].min():.0f} to {selected['Anneal_Temp_C'].max():.0f} °C")
        print(f"Temperature mean: {selected['Anneal_Temp_C'].mean():.0f} °C")
    if 'Tree_Predicted_Quality' in selected.columns:
        print(f"Tree Predicted Quality range: {selected['Tree_Predicted_Quality'].min():.1f} to {selected['Tree_Predicted_Quality'].max():.1f}")
        print(f"Tree Predicted Quality mean: {selected['Tree_Predicted_Quality'].mean():.1f}")
    if 'Rank_Score' in selected.columns:
        print(f"Rank Score range: {selected['Rank_Score'].min():.1f} to {selected['Rank_Score'].max():.1f}")
        print(f"Rank Score mean: {selected['Rank_Score'].mean():.1f}")
    
    # Save if requested
    if output_file:
        selected.to_csv(output_file, index=False)
        print(f"\nSaved selected candidates to {output_file}")
    else:
        default_output = 'data/selected_candidates.csv'
        selected.to_csv(default_output, index=False)
        print(f"\nSaved selected candidates to {default_output}")
    
    return selected


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Select best candidates for experiments')
    parser.add_argument('--file', type=str, default='data/candidates_analysis.csv',
                       help='Path to candidates_analysis.csv')
    parser.add_argument('--n', type=int, default=10,
                       help='Number of candidates to select')
    parser.add_argument('--strategy', type=str, default='tree',
                       choices=['tree', 'rank', 'filtered', 'pareto'],
                       help='Selection strategy')
    parser.add_argument('--fa-min', type=float, default=0.6,
                       help='Minimum FA ratio for filtered strategy')
    parser.add_argument('--fa-max', type=float, default=0.8,
                       help='Maximum FA ratio for filtered strategy')
    parser.add_argument('--spin-min', type=int, default=3000,
                       help='Minimum spin speed for filtered strategy')
    parser.add_argument('--spin-max', type=int, default=4000,
                       help='Maximum spin speed for filtered strategy')
    parser.add_argument('--temp-min', type=int, default=140,
                       help='Minimum temperature for filtered strategy')
    parser.add_argument('--temp-max', type=int, default=145,
                       help='Maximum temperature for filtered strategy')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: data/selected_candidates.csv)')
    
    args = parser.parse_args()
    
    select_candidates(
        candidates_file=args.file,
        n_select=args.n,
        strategy=args.strategy,
        fa_range=(args.fa_min, args.fa_max),
        spin_range=(args.spin_min, args.spin_max),
        temp_range=(args.temp_min, args.temp_max),
        output_file=args.output
    )
