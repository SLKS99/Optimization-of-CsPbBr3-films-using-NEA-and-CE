#!/usr/bin/env python3
"""
Add Transition Experiments to Candidates CSV
==============================================
Adds all experiments from transition_experiments.pkl to candidates_analysis.csv
"""

import os
import sys
import pickle
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import ExperimentParams
from main import load_transition_experiments


def experiment_to_csv_row(exp: ExperimentParams, experiment_id: int) -> dict:
    """Convert ExperimentParams to CSV row format for candidates_analysis.csv"""
    # Get solvent name
    solvent_name = ""
    if exp.solvent:
        solvent_parts = []
        for s in exp.solvent:
            solvent_parts.append(f"{s.solvent.name}({s.ratio:.2g})")
        solvent_name = "+".join(solvent_parts)
    
    row = {
        'Experiment_ID': experiment_id,
        'Rank_Score': round(exp.rank_score, 1) if hasattr(exp, 'rank_score') and exp.rank_score else 0.0,
        'System': str(exp.material_system),
        'Solvent': solvent_name,
        'Concentration_M': exp.concentration,
        'Spin_Speed_rpm': exp.spin_speed,
        'Anneal_Temp_C': exp.annealing_temp,
        'ML_Predicted_Quality': exp.metrics.predicted_performance if exp.metrics.predicted_performance is not None else None,
        'ML_Acquisition_Score': exp.metrics.acquisition_score if exp.metrics.acquisition_score is not None else None
    }
    return row


def main():
    print("=" * 60)
    print("Add Transition Experiments to Candidates CSV")
    print("=" * 60)
    
    # Load transition experiments
    transition_file = os.path.join('data', 'transition_experiments.pkl')
    if not os.path.exists(transition_file):
        print(f"\nError: {transition_file} not found!")
        return
    
    print(f"\nLoading transition experiments from {transition_file}...")
    transition_experiments = load_transition_experiments()
    
    if not transition_experiments:
        print("\nNo transition experiments found!")
        return
    
    print(f"Found {len(transition_experiments)} transition experiments")
    
    # Load existing CSV
    csv_file = os.path.join('data', 'candidates_analysis.csv')
    existing_df = None
    max_id = 0
    
    if os.path.exists(csv_file):
        print(f"\nLoading existing CSV from {csv_file}...")
        existing_df = pd.read_csv(csv_file)
        if 'Experiment_ID' in existing_df.columns:
            max_id = int(existing_df['Experiment_ID'].max()) if len(existing_df) > 0 else 0
        print(f"Found {len(existing_df)} existing entries (max ID: {max_id})")
    else:
        print(f"\nCreating new CSV file: {csv_file}")
    
    # Convert transition experiments to CSV rows
    print(f"\nConverting {len(transition_experiments)} experiments to CSV format...")
    new_rows = []
    for i, exp in enumerate(transition_experiments, 1):
        experiment_id = max_id + i
        row = experiment_to_csv_row(exp, experiment_id)
        new_rows.append(row)
        if i <= 3:  # Print first few
            print(f"  Experiment {i}: {row['System']} @ {row['Anneal_Temp_C']}°C, {row['Spin_Speed_rpm']}rpm")
    
    new_df = pd.DataFrame(new_rows)
    
    # Combine with existing data
    if existing_df is not None:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
    
    # Sort by Rank_Score descending
    combined_df = combined_df.sort_values(by='Rank_Score', ascending=False, na_position='last')
    
    # Save to CSV
    print(f"\nSaving {len(combined_df)} total entries to {csv_file}...")
    combined_df.to_csv(csv_file, index=False)
    print(f"[OK] Successfully added {len(new_rows)} experiments to CSV")
    print(f"     Total entries in CSV: {len(combined_df)}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
