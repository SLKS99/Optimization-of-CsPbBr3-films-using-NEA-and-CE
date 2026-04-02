#!/usr/bin/env python3
"""
Add transition experiments to experiments_log.csv with all parameters filled.
User just needs to add PL/quality measurements.

Usage:
    python add_transition_to_log.py
"""

import pickle
import pandas as pd
import os
from pathlib import Path

TRANSITION_FILE = os.path.join('data', 'transition_experiments.pkl')
LOG_FILE = os.path.join('data', 'templates', 'experiments_log.csv')


def load_transition_experiments():
    """Load experiments from transition pickle file."""
    if not os.path.exists(TRANSITION_FILE):
        print(f"Error: {TRANSITION_FILE} not found!")
        return []
    
    with open(TRANSITION_FILE, 'rb') as f:
        experiments = pickle.load(f)
    
    return experiments


def experiment_to_log_row(exp, experiment_id):
    """Convert ExperimentParams to a dictionary matching experiments_log.csv format."""
    # Get antisolvent name (shortened)
    antisolvent_name = None
    if exp.antisolvent:
        name = exp.antisolvent.name
        # Shorten common names
        if 'Toluene' in name:
            antisolvent_name = 'Toluene'
        elif 'Chlorobenzene' in name or 'CB' in name:
            antisolvent_name = 'Chlorobenzene'
        elif 'Diethyl' in name:
            antisolvent_name = 'Diethyl Ether'
        else:
            antisolvent_name = name
    
    row = {
        'Experiment_ID': experiment_id,
        'Material_Composition': str(exp.material_system),
        'Solvent_Name': exp.solvent_name,
        'Concentration_M': 0.3,  # Always use 0.3 M
        'Spin_Speed_rpm': exp.spin_speed,
        'Spin_Time_s': exp.spin_time,
        'Spin_Acceleration_rpm_s': exp.spin_acceleration,
        'Anneal_Temp_C': exp.annealing_temp,
        'Anneal_Time_s': exp.annealing_time,
        'Dispense_Height_mm': exp.dispense_height,
        'Antisolvent_Name': antisolvent_name if antisolvent_name else '',
        'Antisolvent_Volume_uL': exp.antisolvent_volume if exp.antisolvent else '',
        'Antisolvent_Timing_s': exp.antisolvent_timing if exp.antisolvent else '',
        # Leave PL columns empty for user to fill
        'PL_Peak_nm': '',
        'PL_Intensity ': '',  # Note: space after Intensity
        'PL_FWHM_nm': '',
        'PL_Peak_3d_nm': '',
        'PL_Intensity_3d_nm': '',
        'PL_FWHM_3d_nm': ''
    }
    return row


def main():
    print("=" * 60)
    print("Add Transition Experiments to Log")
    print("=" * 60)
    
    # Load transition experiments
    experiments = load_transition_experiments()
    
    if not experiments:
        print("No experiments found in transition file.")
        return
    
    print(f"\nLoaded {len(experiments)} experiments from transition file.")
    
    # Load existing log
    if os.path.exists(LOG_FILE):
        existing_df = pd.read_csv(LOG_FILE)
        print(f"Loaded existing log with {len(existing_df)} entries.")
        max_id = existing_df['Experiment_ID'].max() if 'Experiment_ID' in existing_df.columns else 0
    else:
        print(f"Creating new log file: {LOG_FILE}")
        existing_df = pd.DataFrame()
        max_id = 0
    
    # Get the last 12 experiments
    n_to_add = min(12, len(experiments))
    last_12 = experiments[-n_to_add:] if n_to_add > 0 else []
    
    if not last_12:
        print("\nNo experiments to add.")
        return
    
    print(f"\nGetting last {n_to_add} experiments from transition file:")
    for i, exp in enumerate(last_12, 1):
        print(f"  [{i}] {exp.material_system} | Spin: {exp.spin_speed}rpm | Temp: {exp.annealing_temp}°C")
    
    # Prepare new rows
    new_rows = []
    
    # Get the next available Experiment_ID
    if not existing_df.empty and 'Experiment_ID' in existing_df.columns:
        next_id = int(existing_df['Experiment_ID'].max()) + 1
    else:
        next_id = 1
    
    print(f"\nAdding as Experiment_ID = {next_id}")
    for i, exp in enumerate(last_12, 1):
        row = experiment_to_log_row(exp, experiment_id=next_id)
        new_rows.append(row)
        print(f"  [{i}] {row['Material_Composition']} | Spin: {row['Spin_Speed_rpm']}rpm | Temp: {row['Anneal_Temp_C']}°C")
    
    if not new_rows:
        print("\nNo experiments to add. Need at least 1 experiment in transition file.")
        return
    
    # Create DataFrame from new rows
    new_df = pd.DataFrame(new_rows)
    
    # Check if these experiments already exist (to avoid duplicates)
    if not existing_df.empty:
        # Check for duplicates based on key parameters
        existing_keys = set()
        for _, row in existing_df.iterrows():
            key = (
                str(row.get('Material_Composition', '')).strip(),
                str(row.get('Solvent_Name', '')).strip(),
                float(row.get('Concentration_M', 0)) if pd.notna(row.get('Concentration_M')) else 0,
                float(row.get('Spin_Speed_rpm', 0)) if pd.notna(row.get('Spin_Speed_rpm')) else 0,
                float(row.get('Anneal_Temp_C', 0)) if pd.notna(row.get('Anneal_Temp_C')) else 0
            )
            existing_keys.add(key)
        
        # Filter out duplicates
        filtered_new_rows = []
        for row in new_rows:
            key = (
                str(row['Material_Composition']).strip(),
                str(row['Solvent_Name']).strip(),
                float(row['Concentration_M']),
                float(row['Spin_Speed_rpm']),
                float(row['Anneal_Temp_C'])
            )
            if key not in existing_keys:
                filtered_new_rows.append(row)
            else:
                print(f"  [SKIP] Duplicate found: {row['Material_Composition']}")
        
        if filtered_new_rows:
            new_df = pd.DataFrame(filtered_new_rows)
            print(f"\nFiltered: {len(new_rows) - len(filtered_new_rows)} duplicates removed.")
        else:
            print("\nAll experiments already exist in log. No new entries added.")
            return
    
    # Combine existing and new
    if not existing_df.empty:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
    
    # Ensure column order matches the template
    columns = [
        'Experiment_ID', 'Material_Composition', 'Solvent_Name', 'Concentration_M',
        'Spin_Speed_rpm', 'Spin_Time_s', 'Spin_Acceleration_rpm_s',
        'Anneal_Temp_C', 'Anneal_Time_s', 'Dispense_Height_mm',
        'Antisolvent_Name', 'Antisolvent_Volume_uL', 'Antisolvent_Timing_s',
        'PL_Peak_nm', 'PL_Intensity ', 'PL_FWHM_nm',
        'PL_Peak_3d_nm', 'PL_Intensity_3d_nm', 'PL_FWHM_3d_nm'
    ]
    combined_df = combined_df[columns]
    
    # Save to file
    combined_df.to_csv(LOG_FILE, index=False)
    
    print(f"\n[SUCCESS] Added {len(new_df)} new experiment(s) to {LOG_FILE}")
    print(f"Total experiments in log: {len(combined_df)}")
    print(f"\nNext steps:")
    print(f"  1. Open {LOG_FILE}")
    print(f"  2. Fill in PL_Peak_nm, PL_Intensity, PL_FWHM_nm columns")
    print(f"  3. Optionally fill in PL_Peak_3d_nm, PL_Intensity_3d_nm, PL_FWHM_3d_nm for 3-day stability")
    print(f"  4. Run 'python main.py' to use the new data for GP training")


if __name__ == "__main__":
    main()
