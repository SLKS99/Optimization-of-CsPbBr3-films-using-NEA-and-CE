#!/usr/bin/env python3
"""
Update Optimization Plots
=========================
Regenerates enhanced optimization plots from existing data without running a new optimization cycle.

This script:
- Loads existing experiment history from experiments_log.csv
- Loads candidate data from candidates_analysis.csv
- Trains the GP model on existing data
- Generates comprehensive plots including:
  * GP Learning Diagnostics (R², signal-to-noise)
  * Uncertainty Heatmap (spatial variation)
  * Cycle Improvement tracking
  * Multi-objective Pareto front (if enabled)
  * Enhanced GP approximation with actual quality values

Usage:
    python update_plots.py
"""

import os
import sys
import pickle
import pandas as pd
import yaml

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.learner import ActiveLearner, CompositionGP # Also import CompositionGP
from src.plotting import plot_optimization_progress
from src.models import ExperimentParams, MaterialSystem, DecisionMetrics
from main import load_config, load_history, load_transition_experiments, calculate_suitability_score, get_variations_of_top_candidates, get_solution_mapping, save_prep_instructions, save_robot_recipe_yaml, save_selected_candidates_detailed, load_binary_fa_ratios # Import all necessary functions
from src.tree_model import ProcessTreeLearner # Ensure tree_learner is imported


def load_candidates_from_csv(csv_path: str, generator=None):
    """
    Load candidate data from CSV and create minimal ExperimentParams objects for plotting.
    We only need basic info for visualization.
    """
    if not os.path.exists(csv_path):
        return []
    
    import pandas as pd
    from src.models import ExperimentParams, MaterialSystem, MaterialComponent, Precursor, DecisionMetrics
    from src.plotting import extract_fa_ratio
    from main import load_data, load_config
    import re
    
    # Load precursors to get proper Precursor objects
    cfg = load_config()
    _, precursors_df = load_data(cfg)
    
    # Create a mapping of precursor names to Precursor objects
    precursor_map = {}
    if not precursors_df.empty:
        for _, row in precursors_df.iterrows():
            name = str(row.get('name', '')).strip()
            if name:
                precursor_map[name.upper()] = Precursor(
                    name=name,
                    type=str(row.get('type', 'A')),
                    ionic_radius=float(row.get('ionic_radius', 2.5)),
                    molecular_weight=float(row.get('molecular_weight', 100)),
                    solubility_parameter=float(row.get('solubility_parameter', 20))
                )
    
    # Defaults if not in CSV
    if 'FA' not in precursor_map:
        precursor_map['FA'] = Precursor(name='FA', type='A', ionic_radius=2.53, 
                                       molecular_weight=115.1, solubility_parameter=21.5)
    if 'FUDMA' not in precursor_map:
        precursor_map['FUDMA'] = Precursor(name='FuDMA', type='A', ionic_radius=2.85,
                                          molecular_weight=195.2, solubility_parameter=20.0)
    
    df = pd.read_csv(csv_path)
    candidates = []
    print(f"  Reading CSV with {len(df)} rows...")
    
    for idx, row in df.iterrows():
        try:
            # Parse composition string - CSV uses 'System' column
            comp_str = str(row.get('System', row.get('Material_Composition', '')))
            if pd.isna(comp_str) or comp_str == '' or comp_str == 'nan':
                continue
            
            # Extract FA and FuDMA from composition string
            fa_match = re.search(r'FA([\d.]+)', comp_str, re.IGNORECASE)
            fudma_match = re.search(r'FuDMA([\d.]+)', comp_str, re.IGNORECASE)
            
            a_site = []
            if fa_match:
                fa_val = float(fa_match.group(1))
                fa_prec = precursor_map.get('FA', precursor_map.get('FA', None))
                if fa_prec:
                    a_site.append(MaterialComponent(precursor=fa_prec, ratio=fa_val))
            if fudma_match:
                fudma_val = float(fudma_match.group(1))
                fudma_prec = precursor_map.get('FUDMA', precursor_map.get('FuDMA', None))
                if fudma_prec:
                    a_site.append(MaterialComponent(precursor=fudma_prec, ratio=fudma_val))
            
            if not a_site:
                continue
            
            # Normalize ratios
            total = sum(c.ratio for c in a_site)
            if total > 0:
                for c in a_site:
                    c.ratio = c.ratio / total
            
            material_system = MaterialSystem(a_site=a_site, b_site=[], x_site=[])
            
            # Create ExperimentParams - need all required fields
            exp = ExperimentParams(
                material_system=material_system,
                solvent=[],
                concentration=float(row.get('Concentration_M', 0.3)),
                spin_speed=int(float(row.get('Spin_Speed_rpm', 4000))),
                spin_time=int(float(row.get('Spin_Time_s', 30))),
                spin_acceleration=int(float(row.get('Spin_Acceleration_rpm_s', 2000))),
                annealing_temp=int(float(row.get('Anneal_Temp_C', 140))),
                annealing_time=int(float(row.get('Anneal_Time_s', 600))),
                dispense_height=float(row.get('Dispense_Height_mm', 15)),
                drop_volume=int(float(row.get('Drop_Volume_ul', 50))),
                antisolvent=None,
                antisolvent_timing=None,
                antisolvent_volume=None,
                two_step_enabled=False,
                first_spin_speed=None,
                first_spin_time=None,
                first_spin_acceleration=None,
                metrics=DecisionMetrics()
            )
            
            # Set GP predictions - CSV uses different column names
            if 'ML_Predicted_Quality' in row.index and pd.notna(row.get('ML_Predicted_Quality')):
                exp.metrics.predicted_performance = float(row['ML_Predicted_Quality'])
            if 'ML_Acquisition_Score' in row.index and pd.notna(row.get('ML_Acquisition_Score')):
                exp.metrics.acquisition_score = float(row['ML_Acquisition_Score'])
            # Uncertainty - check multiple possible column names
            if 'ML_Uncertainty' in row.index and pd.notna(row.get('ML_Uncertainty')):
                exp.metrics.uncertainty_score = float(row['ML_Uncertainty'])
            elif 'Uncertainty_Score' in row.index and pd.notna(row.get('Uncertainty_Score')):
                exp.metrics.uncertainty_score = float(row['Uncertainty_Score'])
            # If not in CSV, will be calculated when scoring
            
            # Get rank score
            if 'Rank_Score' in row.index and pd.notna(row.get('Rank_Score')):
                exp.rank_score = float(row['Rank_Score'])
            
            candidates.append(exp)
            if len(candidates) <= 5:  # Print first few successes
                print(f"    Successfully loaded candidate {len(candidates)}: {comp_str}")
        except Exception as e:
            # Skip rows that can't be parsed
            if idx < 5:  # Only print first few errors for debugging
                import traceback
                print(f"  Warning: Could not parse row {idx}: {str(e)[:100]}")
                if idx == 0:
                    print(f"    Row data: {list(dict(row).keys())[:5]}")
            continue
    
    print(f"  Successfully loaded {len(candidates)} candidates from CSV")
    return candidates


def candidates_from_transition():
    """Load candidates from transition experiments pickle file."""
    transition_file = os.path.join('data', 'transition_experiments.pkl')
    if os.path.exists(transition_file):
        try:
            with open(transition_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load transition file: {e}")
    return []


def main():
    print("=" * 60)
    print("Update Optimization Plots")
    print("=" * 60)
    
    # 1. Load config
    cfg = load_config()
    
    # 2. Load history
    history_df = load_history(cfg)
    print(f"\nLoaded {len(history_df)} experiments from history")
    
    # 3. Setup and train learner
    quality_config = cfg.get('quality_target', {})
    learner = ActiveLearner(quality_config=quality_config)
    learner.train(history_df)
    
    if not learner.is_trained:
        print("\nWarning: GP model could not be trained. Plots may be limited.")
    
    # 4. Load candidates for plotting
    candidates = []
    
    # Load from CSV first (has ALL candidates, not just transition)
    candidates_path = os.path.join('data', 'candidates_analysis.csv')
    if os.path.exists(candidates_path):
        print(f"\nLoading candidates from {candidates_path}...")
        candidates = load_candidates_from_csv(candidates_path)
        print(f"Loaded {len(candidates)} candidates from CSV")
        
        # Score them with GP if trained
        if learner.is_trained and candidates:
            print("Scoring candidates with GP...")
            learner.score_candidates_with_uncertainty(candidates)
            # Also score with tree if available
            try:
                from src.tree_model import ProcessTreeLearner
                temp_tree = ProcessTreeLearner()
                temp_tree.train_from_history(history_df, learner)
                if temp_tree.is_trained:
                    print("Scoring candidates with Decision Tree...")
                    temp_tree.score_candidates(candidates)
                    scored_count = sum(1 for c in candidates if c.metrics.tree_predicted_quality is not None)
                    print(f"  ✓ {scored_count} candidates scored with Decision Tree")
            except Exception as e:
                print(f"Warning: Could not score candidates with tree in update script: {e}")

            # Also calculate rank scores
            from main import calculate_suitability_score
            for exp in candidates:
                # Determine if tree should be used for rank
                use_tree = False
                if 'temp_tree' in locals() and temp_tree.is_trained:
                    use_tree = True
                exp.rank_score = calculate_suitability_score(exp, use_ml=learner.is_trained, use_tree=use_tree)
    
    # Also load transition experiments for reference
    transition_candidates = candidates_from_transition()
    if transition_candidates:
        print(f"Also loaded {len(transition_candidates)} transition experiments for reference")
    
    if not candidates:
        print("\nNo candidate data found. Plots will show only historical data.")
    
    # 5. Generate plots with enhanced visualization
    plot_path = os.path.join('data', 'optimization_plot.png')
    print(f"\nGenerating enhanced plots with new diagnostics...")
    
    # Check if Pareto mode is enabled
    use_pareto = cfg.get('generation', {}).get('use_pareto_front', False)
    if use_pareto:
        print("  Using multi-objective (Pareto) visualization mode")
    
    # Load composition GP and tree learner if available
    comp_gp = None
    tree_learner = None
    
    # Try to load and train composition GP (same way as main.py)
    comp_gp_cfg = cfg.get('composition_gp', {}) or {}
    if comp_gp_cfg.get('enabled', False):
        comp_data_path = comp_gp_cfg.get('data_path')
        if comp_data_path and os.path.exists(comp_data_path):
            try:
                from src.learner import CompositionGP
                from main import load_binary_fa_ratios
                
                print("  Loading composition GP data...")
                comp_gp = CompositionGP(ucb_beta=comp_gp_cfg.get('ucb_beta', 2.0))
                raw_df = pd.read_csv(comp_data_path)
                
                # Check if we need to process the binary data (same logic as main.py)
                if 'FA_Ratio' not in raw_df.columns or 'Quality_Target' not in raw_df.columns:
                    print("  Processing binary dataset (mapping compositions to FA ratios)...")
                    # Load FA ratio mapping
                    fa_map = load_binary_fa_ratios(cfg)
                    rows = []
                    for _, row in raw_df.iterrows():
                        # Filter for Read == 1 if that column exists
                        if 'Read' in raw_df.columns and row.get('Read') != 1:
                            continue
                        comp_label = str(row.get('Composition', '')).strip()
                        if comp_label not in fa_map:
                            continue
                        fa_ratio = fa_map[comp_label]
                        # Build a minimal Series compatible with calculate_quality_target
                        pseudo = pd.Series({
                            'PL_Intensity': row.get('Peak_1_Intensity', row.get('PL_Intensity', 0.0)),
                            'PL_FWHM_nm': row.get('Peak_1_FWHM', row.get('PL_FWHM_nm', 0.0)),
                            'PL_Peak_nm': row.get('Peak_1_Wavelength', row.get('PL_Peak_nm', None)),
                            'Stability_Hours': None,
                        })
                        quality = learner.calculate_quality_target(pseudo)
                        rows.append({'FA_Ratio': fa_ratio, 'Quality_Target': quality})
                    binary_df = pd.DataFrame(rows)
                    print(f"  Processed {len(binary_df)} binary data points")
                else:
                    # Generic preprocessed file with FA_Ratio / Quality_Target columns
                    binary_df = raw_df
                    print(f"  Using preprocessed data with {len(binary_df)} points")
                
                # Train on BOTH binary dataset AND experiment history (same as main.py)
                print("  Training Composition GP on binary dataset + experiment history...")
                comp_gp.train_from_binary_and_history(
                    binary_df=binary_df,
                    history_df=history_df,
                    quality_learner=learner,
                    fa_column=comp_gp_cfg.get('fa_column', 'FA_Ratio'),
                    quality_column=comp_gp_cfg.get('quality_column', 'Quality_Target'),
                )
                
                if comp_gp.is_trained:
                    print(f"  ✓ Composition GP trained successfully")
                else:
                    print(f"  ⚠ Composition GP training failed")
            except Exception as e:
                print(f"  Warning: Could not load/train composition GP: {e}")
                import traceback
                if __name__ == "__main__":
                    traceback.print_exc()
    
    # Try to load tree learner
    try:
        from src.tree_model import ProcessTreeLearner
        tree_learner = ProcessTreeLearner()
        tree_learner.train_from_history(history_df, learner)
    except Exception as e:
        print(f"  Warning: Could not train tree learner: {e}")
    
    plot_optimization_progress(history_df, candidates, learner=learner, 
                              output_file=plot_path, show_pareto=use_pareto,
                              comp_gp=comp_gp, tree_learner=tree_learner)
    print(f"[OK] Enhanced plots saved to {plot_path}")

    # New: Select a batch of candidates and save prep instructions and robot recipe
    print("\nSelecting optimal experiments for batch and generating prep instructions...")
    batch_size = cfg.get('generation', {}).get('batch_size', 8)

    # Check if GP is learning - if not, prioritize Decision Tree
    gp_learning = False
    if learner.is_trained:
        gp_preds = [c.metrics.predicted_performance for c in candidates 
                   if c.metrics.predicted_performance is not None]
        if len(gp_preds) > 0:
            gp_range = max(gp_preds) - min(gp_preds)
            gp_learning = gp_range >= 0.1  # GP is learning if predictions vary significantly
            if not gp_learning:
                print(f"  GP not learning (prediction range: {gp_range:.4f} < 0.1), using Decision Tree")

    # Re-calculate rank scores for all candidates first if not already done consistently
    for exp in candidates:
        use_tree_for_rank = False
        if tree_learner and tree_learner.is_trained:
            use_tree_for_rank = True
        exp.rank_score = calculate_suitability_score(exp, use_ml=learner.is_trained, use_tree=use_tree_for_rank)

    selected_experiments = []
    if use_pareto and learner.is_trained and gp_learning:
        from src.multi_objective import select_diverse_pareto_front
        selected_experiments = select_diverse_pareto_front(
            candidates, 
            n_select=batch_size,
            objectives=['quality', 'uncertainty']
        )
    elif tree_learner is not None and tree_learner.is_trained and not gp_learning:
        print("\nUsing Decision Tree-based selection (GP not learning, prioritizing Tree predictions)...")
        # Filter by best parameter ranges, then sort by tree predictions
        # This section should ideally be consistent with main.py's get_variations_of_top_candidates logic
        
        # For update_plots.py, we'll just use get_variations_of_top_candidates for simplicity
        # as it encapsulates the selection logic, including the 3-composition preference.
        selected_experiments = get_variations_of_top_candidates(
            candidates,
            target_count=batch_size,
            exploration_fraction=cfg.get('generation', {}).get('exploration_rate', 0.15),
            tree_learner=tree_learner
        )
    else:
        # Fallback to general get_variations_of_top_candidates
        selected_experiments = get_variations_of_top_candidates(
            candidates,
            target_count=batch_size,
            exploration_fraction=cfg.get('generation', {}).get('exploration_rate', 0.15),
            tree_learner=tree_learner
        )

    if not selected_experiments:
        print("No experiments selected for prep instructions.")
        return # Exit if no experiments selected

    # Generate outputs
    solution_mapping = get_solution_mapping(selected_experiments)
    prep_instructions_path = os.path.join('data', 'prep_instructions.txt')
    robot_recipe_path = os.path.join('data', 'robot_recipe_generated.yaml')
    selected_detailed_path = os.path.join('data', 'selected_candidates_detailed.csv')

    save_prep_instructions(solution_mapping, selected_experiments, prep_instructions_path)
    save_robot_recipe_yaml(
        selected_experiments,
        solution_mapping,
        robot_recipe_path,
        robot_recipe_cfg=cfg.get('robot_recipe'),
    )
    save_selected_candidates_detailed(selected_experiments, selected_detailed_path)

    print(f"[OK] Prep instructions saved to {prep_instructions_path}")
    print(f"[OK] Robot recipe saved to {robot_recipe_path}")
    print(f"[OK] Detailed selected candidates saved to {selected_detailed_path}")

    print("\nPlots included:")
    print("  - Quality by FA Ratio Range")
    print("  - Quality by Antisolvent Type")
    print("  - Quality by Temperature Range")
    print("  - Exploration vs Exploitation")
    if comp_gp and comp_gp.is_trained:
        print("  - Composition GP Learning Plot (FA Ratio → Quality)")
    else:
        print("  - Composition GP Plot (not available - check config/data)")
    if tree_learner and tree_learner.is_trained:
        print("  - Decision Tree Feature Importance")
        print("  - SHAP Feature Impact Analysis")
        print("  - Decision Tree Predictions vs Actual")
    else:
        print("  - Decision Tree plots (not available - need 10+ experiments)")
    
    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
