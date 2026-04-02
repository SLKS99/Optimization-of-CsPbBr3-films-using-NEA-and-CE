#!/usr/bin/env python3
"""
Generate prep instructions from transition experiments pickle file.
Useful if you need to regenerate prep instructions without running full GP optimization.
"""

import pickle
import os
import sys
from main import load_config, save_prep_instructions, get_solution_mapping, save_robot_recipe_yaml

TRANSITION_FILE = os.path.join('data', 'transition_experiments.pkl')


def load_transition_experiments():
    """Load experiments from transition pickle file."""
    if not os.path.exists(TRANSITION_FILE):
        print(f"Error: {TRANSITION_FILE} not found!")
        return []
    
    with open(TRANSITION_FILE, 'rb') as f:
        experiments = pickle.load(f)
    
    return experiments


def main():
    print("=" * 60)
    print("Generate Prep Instructions from Transition Experiments")
    print("=" * 60)
    
    # Load transition experiments
    experiments = load_transition_experiments()
    
    if not experiments:
        print("No experiments found in transition file.")
        return
    
    print(f"\nLoaded {len(experiments)} experiments from transition file.")
    
    # Get first 8 (or all if fewer)
    batch_size = 8
    experiments_to_use = experiments[:batch_size]
    
    print(f"Generating prep instructions for first {len(experiments_to_use)} experiments...\n")
    
    # Generate solution mapping
    solution_mapping = get_solution_mapping(experiments_to_use)
    print(f"Found {len(solution_mapping)} unique solution(s).\n")
    
    # Generate prep instructions
    prep_file = os.path.join('data', 'prep_instructions.txt')
    save_prep_instructions(solution_mapping, experiments_to_use, prep_file)
    
    # Generate robot recipe
    recipe_file = os.path.join('data', 'robot_recipe_generated.yaml')
    cfg = load_config()
    save_robot_recipe_yaml(
        experiments_to_use,
        solution_mapping,
        recipe_file,
        robot_recipe_cfg=cfg.get('robot_recipe'),
    )
    
    print(f"\n[OK] Prep instructions saved to: {prep_file}")
    print(f"[OK] Robot recipe saved to: {recipe_file}")
    
    # Show summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    for i, exp in enumerate(experiments_to_use, 1):
        print(f"\n[{i}] {exp.material_system}")
        print(f"    Solvent: {exp.solvent_name}")
        print(f"    Concentration: {exp.concentration} M")
        print(f"    Spin: {exp.spin_speed} rpm / {exp.spin_time}s")
        if exp.two_step_enabled:
            print(f"    (Two-step: {exp.first_spin_speed} rpm / {exp.first_spin_time}s first)")
        print(f"    Annealing: {exp.annealing_temp}°C for {exp.annealing_time}s")
        print(f"    Drop Volume: {exp.drop_volume} µL")
        if exp.antisolvent:
            print(f"    Antisolvent: {exp.antisolvent.name} ({exp.antisolvent_volume} µL)")


if __name__ == "__main__":
    main()
