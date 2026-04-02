#!/usr/bin/env python3
"""
Monte Carlo Decision Tree for Perovskite Thin-Film Optimization
================================================================
All user-configurable settings live in config.yaml.
Edit that file to change compositions, solvents, file paths, etc.
"""
import pandas as pd
import os
import sys
import yaml
import math
import pickle
import argparse
from typing import List, Dict, Tuple, Optional
import numpy as np

from src.generator import MonteCarloGenerator
from src.composition_generator import CompositionAwareGenerator
from src.models import ExperimentParams, MaterialComponent, SolventComponent
from src.learner import ActiveLearner, CompositionGP
from src.dual_composition_gp import DualCompositionGP
from src.tree_model import ProcessTreeLearner
from src import plotting
from src.convergence import ConvergenceTracker
from src.multi_objective import select_diverse_pareto_front
from src.learner import extract_fa_ratio_from_composition

# ---------------------------------------------------------------------------
# Internal constants (not user-facing)
# -------------------------------------------------------------------
TRANSITION_FILE = os.path.join('data', 'transition_experiments.pkl')
DEFAULT_CONFIG_PATH = 'config.yaml'

def get_output_dir(cfg: dict) -> str:
    """
    Get output directory from config, or default to 'data/'.
    Creates directory if it doesn't exist.
    """
    output_dir = cfg.get('data', {}).get('output_directory')
    if output_dir and output_dir.strip():
        output_dir = output_dir.strip()
        # Normalize path separators
        output_dir = os.path.normpath(output_dir)
        # Ensure trailing slash
        if not output_dir.endswith(os.sep):
            output_dir += os.sep
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    return 'data' + os.sep  # Default to 'data/' directory


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    if not os.path.exists(path):
        print(f"Error: Config file '{path}' not found.")
        sys.exit(1)
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_data(cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    solvents_path = cfg['data']['solvents']
    precursors_path = cfg['data']['precursors']

    if not os.path.exists(solvents_path) or not os.path.exists(precursors_path):
        print("Error: Data files not found. Check paths in config.yaml.")
        sys.exit(1)

    solvents_df = pd.read_csv(solvents_path)
    precursors_df = pd.read_csv(precursors_path)
    return solvents_df, precursors_df


def load_history(cfg: dict) -> pd.DataFrame:
    log_path = cfg['data']['history']
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        
        # Clean up column names (strip whitespace)
        df.columns = [col.strip() for col in df.columns]
        
        # Clean up string values (strip whitespace)
        for col in df.select_dtypes(['object']).columns:
            df[col] = df[col].str.strip()
        
        # Calculate Stability_Hours from 3-day PL data if available
        # Stability = 72 hours if PL_Intensity_3d_nm exists and is > 0
        # Optionally, could calculate based on retention: (PL_3d / PL_initial) * 72
        if 'PL_Intensity_3d_nm' in df.columns and 'PL_Intensity ' in df.columns:
            # Check if Stability_Hours column exists, if not create it
            if 'Stability_Hours' not in df.columns:
                df['Stability_Hours'] = 0.0
            
            # Calculate stability for rows with 3-day PL data
            for idx, row in df.iterrows():
                pl_initial = row.get('PL_Intensity ', row.get('PL_Intensity', None))
                pl_3d = row.get('PL_Intensity_3d_nm', None)
                
                # If we have 3-day PL data and it's valid
                if pd.notna(pl_3d) and pd.notna(pl_initial) and pl_initial > 0:
                    # If Stability_Hours is not already set, calculate it
                    if pd.isna(row.get('Stability_Hours')) or row.get('Stability_Hours') == 0:
                        # Calculate retention ratio (0-1)
                        retention = float(pl_3d) / float(pl_initial)
                        # Stability = retention * 72 hours (3 days)
                        # Higher retention = better stability
                        df.at[idx, 'Stability_Hours'] = retention * 72.0
                # If no 3-day data but Stability_Hours is missing, leave as 0 (already set above)
        
        return df
    return pd.DataFrame()


def load_compositions(cfg: dict) -> List[List[Tuple[str, float]]]:
    """
    Parse wide-format ternary composition CSV into list of A-site recipes.
    Each recipe is a list of (precursor_name, ratio) tuples.
    """
    comp_path = cfg['data'].get('compositions')
    if not comp_path or not os.path.exists(comp_path):
        return []

    df = pd.read_csv(comp_path, index_col=0)
    mapping = cfg.get('precursor_mapping', {})
    normalize = cfg['generation'].get('normalize_ratios', 50)

    recipes = []
    num_cols = len(df.columns)
    for col in df.columns:
        recipe = []
        for row_label in df.index:
            raw_val = df.loc[row_label, col]
            try:
                val = float(raw_val)
            except (ValueError, TypeError):
                val = 0.0
            if val > 0:
                prec_name = mapping.get(row_label, row_label)
                recipe.append((prec_name, val / normalize))
        if recipe:
            recipes.append(recipe)
    return recipes


def load_binary_fa_ratios(cfg: dict) -> Dict[str, float]:
    """
    Build a mapping from composition label (e.g., A1, B3, ...) to FA fraction (0-1)
    using the same binary composition file used for A-site recipes.
    """
    comp_path = cfg['data'].get('compositions')
    if not comp_path or not os.path.exists(comp_path):
        return {}

    df = pd.read_csv(comp_path, index_col=0)
    mapping = cfg.get('precursor_mapping', {})

    # Find the row labels corresponding to FA and FuDMA
    fa_row_label = None
    fudma_row_label = None
    for row_label, prec_name in mapping.items():
        name_upper = str(prec_name).upper()
        if name_upper == 'FA':
            fa_row_label = row_label
        elif 'FUDMA' in name_upper.upper():
            fudma_row_label = row_label

    if fa_row_label is None or fudma_row_label is None:
        return {}

    if fa_row_label not in df.index or fudma_row_label not in df.index:
        return {}

    fa_series = df.loc[fa_row_label]
    fudma_series = df.loc[fudma_row_label]

    fa_ratios: Dict[str, float] = {}
    for col in df.columns:
        try:
            fa_val = float(fa_series[col])
            fudma_val = float(fudma_series[col])
        except (ValueError, TypeError):
            continue
        total = fa_val + fudma_val
        if total <= 0:
            continue
        fa_ratios[col] = fa_val / total

    return fa_ratios


# ---------------------------------------------------------------------------
# Transition experiment persistence
# ---------------------------------------------------------------------------
def load_transition_experiments() -> List[ExperimentParams]:
    if os.path.exists(TRANSITION_FILE):
        try:
            with open(TRANSITION_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load transition file: {e}")
    return []


def normalize_value(val):
    """Normalize values for comparison (handle NaN, whitespace, float precision)."""
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, (int, float)):
        return round(float(val), 2)  # Round to 2 decimals for comparison
    return str(val).strip()


def create_experiment_key(comp, solv, conc, spin, temp):
    """Create normalized key for experiment matching."""
    comp_norm = normalize_value(comp)
    solv_norm = normalize_value(solv)
    conc_norm = normalize_value(conc)
    spin_norm = normalize_value(spin)
    temp_norm = normalize_value(temp)
    return f"{comp_norm}|{solv_norm}|{conc_norm}|{spin_norm}|{temp_norm}"


def history_row_has_results(row: pd.Series) -> bool:
    """True once the log row has measurement data (planned rows are auto-appended with PL columns empty)."""
    for col in ('PL_Intensity', 'PL_Intensity ', 'PL_Peak_nm', 'PL_FWHM_nm'):
        v = row.get(col)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        if str(v).strip() == '':
            continue
        return True
    stab = row.get('Stability_Hours', None)
    if stab is not None and not (isinstance(stab, float) and pd.isna(stab)):
        if str(stab).strip() != '':
            try:
                float(stab)
                return True
            except (ValueError, TypeError):
                return True
    return False


def prune_transition_experiments(history_df: pd.DataFrame, transitions: List[ExperimentParams]) -> List[ExperimentParams]:
    """
    Remove transition experiments that have been completed (logged in history with results).

    Planned-only rows (auto-appended without PL data) do not count as completed.

    Matches experiments by: Material_Composition, Solvent_Name, Concentration_M,
    Spin_Speed_rpm, Anneal_Temp_C
    """
    if history_df.empty or not transitions:
        return transitions

    completed_keys = set()
    
    # Build set of completed experiment keys from history
    for _, row in history_df.iterrows():
        if not history_row_has_results(row):
            continue

        comp = row.get('Material_Composition', '')
        solv = row.get('Solvent_Name', '')
        conc = row.get('Concentration_M', None)
        spin = row.get('Spin_Speed_rpm', None)
        temp = row.get('Anneal_Temp_C', None)
        
        # Skip rows with missing critical data
        if pd.isna(comp) or pd.isna(solv) or pd.isna(conc) or pd.isna(spin) or pd.isna(temp):
            continue
            
        key = create_experiment_key(comp, solv, conc, spin, temp)
        completed_keys.add(key)

    # Filter out completed experiments from transition queue
    pruned = []
    for exp in transitions:
        exp_key = create_experiment_key(
            str(exp.material_system),
            exp.solvent_name,
            exp.concentration,
            exp.spin_speed,
            exp.annealing_temp
        )
        if exp_key not in completed_keys:
            pruned.append(exp)

    if len(pruned) != len(transitions):
        removed = len(transitions) - len(pruned)
        print(f"✓ Pruned {removed} transition experiment(s) already logged in history.")
        print(f"  {len(pruned)} experiment(s) still pending in transition queue.")
    
    return pruned


def save_transition_experiments(experiments: List[ExperimentParams]):
    existing = load_transition_experiments()
    combined = existing + experiments
    with open(TRANSITION_FILE, 'wb') as f:
        pickle.dump(combined, f)
    print(f"Saved {len(experiments)} new experiments (total in transition: {len(combined)}).")


def _experiments_log_template_columns(cfg: dict) -> List[str]:
    """Column order for experiments log (strips whitespace to match load_history)."""
    log_path = cfg['data']['history']
    for path in (log_path, os.path.join('data', 'templates', 'experiments_log.csv')):
        if path and os.path.exists(path):
            return [c.strip() for c in pd.read_csv(path, nrows=0).columns.tolist()]
    return [
        'Experiment_ID', 'Material_Composition', 'Solvent_Name', 'Concentration_M',
        'CsBr_M', 'NEABr_M', 'CE_M',
        'Spin_Speed_rpm', 'Spin_Time_s', 'Spin_Acceleration_rpm_s',
        'Anneal_Temp_C', 'Anneal_Time_s', 'Dispense_Height_mm',
        'Antisolvent_Name', 'Antisolvent_Volume_uL', 'Antisolvent_Timing_s',
        'PL_Peak_nm', 'PL_Intensity', 'PL_FWHM_nm',
        'PL_Peak_3d_nm', 'PL_Intensity_3d_nm', 'PL_FWHM_3d_nm',
    ]


def _antisolvent_log_name(exp: ExperimentParams) -> str:
    if not exp.antisolvent:
        return ''
    name = exp.antisolvent.name
    if 'Toluene' in name:
        return 'Toluene'
    if 'Chlorobenzene' in name or name == 'CB':
        return 'Chlorobenzene'
    if 'Diethyl' in name:
        return 'Diethyl Ether'
    return name


def _experiment_params_to_log_row(exp: ExperimentParams, experiment_id: int, columns: List[str]) -> dict:
    """Build one log row; PL columns left blank for user to fill after measurements."""
    def _m(x: Optional[float]):
        if x is None:
            return ''
        return round(float(x), 6)

    anti_vol = exp.antisolvent_volume if exp.antisolvent else ''
    anti_t = exp.antisolvent_timing if exp.antisolvent else ''

    values = {
        'Experiment_ID': experiment_id,
        'Material_Composition': str(exp.material_system),
        'Solvent_Name': exp.solvent_name,
        'Concentration_M': round(float(exp.concentration), 6),
        'CsBr_M': _m(exp.csbr_M),
        'NEABr_M': _m(exp.neabr_M),
        'CE_M': _m(exp.ce_M),
        'Spin_Speed_rpm': int(exp.spin_speed),
        'Spin_Time_s': int(exp.spin_time),
        'Spin_Acceleration_rpm_s': int(exp.spin_acceleration),
        'Anneal_Temp_C': int(exp.annealing_temp),
        'Anneal_Time_s': int(exp.annealing_time),
        'Dispense_Height_mm': float(exp.dispense_height),
        'Antisolvent_Name': _antisolvent_log_name(exp),
        'Antisolvent_Volume_uL': anti_vol,
        'Antisolvent_Timing_s': anti_t,
        'PL_Peak_nm': '',
        'PL_Intensity': '',
        'PL_FWHM_nm': '',
        'PL_Peak_3d_nm': '',
        'PL_Intensity_3d_nm': '',
        'PL_FWHM_3d_nm': '',
    }
    return {c: values.get(c, '') for c in columns}


def append_batch_to_experiments_log(experiments: List[ExperimentParams], cfg: dict) -> None:
    """Append this run's selected batch to the experiments log (process params + blanks for PL)."""
    if not experiments:
        return
    log_path = cfg['data']['history']
    parent = os.path.dirname(os.path.abspath(log_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    template_cols = _experiments_log_template_columns(cfg)

    if os.path.exists(log_path):
        existing_df = pd.read_csv(log_path)
        existing_df.columns = [c.strip() for c in existing_df.columns]
        if 'Experiment_ID' in existing_df.columns and len(existing_df) > 0:
            next_id = int(pd.to_numeric(existing_df['Experiment_ID'], errors='coerce').fillna(0).max()) + 1
        else:
            next_id = 1
    else:
        existing_df = pd.DataFrame(columns=template_cols)
        next_id = 1

    new_rows = [
        _experiment_params_to_log_row(exp, next_id + i, template_cols)
        for i, exp in enumerate(experiments)
    ]
    new_df = pd.DataFrame(new_rows)

    all_cols = list(dict.fromkeys(list(existing_df.columns) + template_cols))
    for c in all_cols:
        if c not in existing_df.columns:
            existing_df[c] = ''
        if c not in new_df.columns:
            new_df[c] = ''
    existing_df = existing_df.reindex(columns=all_cols)
    new_df = new_df.reindex(columns=all_cols)

    if existing_df.empty:
        combined = new_df
    else:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined.to_csv(log_path, index=False)
    print(f"Appended {len(new_df)} row(s) to experiments log: {log_path}")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def calculate_suitability_score(exp: ExperimentParams, use_ml: bool = False, use_tree: bool = False) -> float:
    score = 0.0

    # --- STRUCTURAL METRICS ---
    # Octahedral factor (B-X cage stability) - primary structural criterion
    mu = exp.metrics.octahedral_factor
    if 0.4 <= mu <= 0.9:
        # Optimal range for stable octahedra
        mu_center = 0.65  # ideal center
        mu_deviation = abs(mu - mu_center)
        score += 25 * max(0, 1 - (mu_deviation / 0.25))
    else:
        score -= 20  # outside stable range

    # Lattice spacing (for 2D/quasi-2D phases)
    if exp.metrics.a_site_lattice_spacing is not None:
        spacing = exp.metrics.a_site_lattice_spacing
        # Prefer 10-15 Å (good for DJ/RP phases)
        if 10 <= spacing <= 15:
            score += 15
        elif 8 <= spacing < 10 or 15 < spacing <= 18:
            score += 8  # acceptable
        else:
            score -= 10  # too small or too large

    # --- SOLVENT/PROCESSING METRICS ---
    # Precursor-solvent miscibility (Hansen delta)
    delta = exp.metrics.precursor_solvent_delta
    if delta is not None:
        if delta < 4:
            score += 20  # excellent miscibility
        elif delta < 6:
            score += 12  # good
        elif delta < 8:
            score += 5   # acceptable
        else:
            score -= 15  # poor

    # Antisolvent miscibility gap (5-10 optimal for controlled precipitation)
    if exp.metrics.antisolvent_miscibility_gap is not None:
        gap = exp.metrics.antisolvent_miscibility_gap
        if 5 < gap < 10:
            score += 12
        elif 3 < gap <= 5 or 10 <= gap < 12:
            score += 5
        else:
            score -= 5

    # Solvent donor number (important for Pb coordination)
    has_lead = any(comp.precursor.name == "Pb" for comp in exp.material_system.b_site)
    if has_lead:
        dn = exp.metrics.solvent_dn
        if 20 <= dn <= 30:
            score += 10  # optimal for Pb-halide dissolution
        elif 15 <= dn < 20 or 30 < dn <= 35:
            score += 5
        elif dn < 10:
            score -= 10  # poor coordination

    # Solvent acceptor number (Lewis acidity - affects precursor solvation)
    an = exp.metrics.solvent_an
    if 10 <= an <= 20:
        score += 5  # balanced acceptor strength
    elif an > 25:
        score -= 3  # too acidic, may degrade precursors

    # Solvent dielectric constant (ionic precursor compatibility)
    eps = exp.metrics.solvent_dielectric
    if 30 <= eps <= 50:
        score += 8  # good for ionic precursors (halides)
    elif 20 <= eps < 30:
        score += 4
    elif eps < 10:
        score -= 5  # poor for ionic species

    # Boiling point window (drying kinetics)
    bp = exp.metrics.solvent_boiling_point_avg
    if 80 <= bp <= 150:
        score += 8  # optimal evaporation rate
    elif 60 <= bp < 80 or 150 < bp <= 180:
        score += 4
    else:
        score -= 5

    # Drying time metric (relative, lower = faster)
    d = exp.metrics.estimated_drying_time
    if d > 0:
        # Prefer moderate drying (not too fast, not too slow)
        if 0.005 <= d <= 0.05:
            score += 8
        elif 0.001 <= d < 0.005 or 0.05 < d <= 0.1:
            score += 4

    # Film uniformity metric (U ∝ 1/(C·E·η))
    # Higher uniformity = better film quality
    if exp.metrics.film_uniformity_metric is not None:
        u = exp.metrics.film_uniformity_metric
        # Typical range ~1-100 depending on conditions
        if u > 5:
            score += 10  # excellent uniformity potential
        elif u > 2:
            score += 5   # good
        elif u < 0.5:
            score -= 5   # poor uniformity expected

    # A-site dipole moment (affects solubility and self-assembly)
    if exp.metrics.a_site_dipole_moment is not None:
        dipole = exp.metrics.a_site_dipole_moment
        # Moderate dipole (2-6 D) aids solubility without excessive H-bonding
        if 2 <= dipole <= 6:
            score += 8  # good for solubility and ordering
        elif 1 <= dipole < 2 or 6 < dipole <= 10:
            score += 4
        elif dipole > 12:
            score -= 5  # very high dipole may cause strong solvent interactions

    # A-site spacer length (geom proxy for spacer fit); prefer moderate lengths
    if exp.metrics.a_site_spacer_length is not None:
        sl = exp.metrics.a_site_spacer_length
        # Favor mid-length spacers; penalize very short or very long
        if 6 <= sl <= 12:
            score += 6
        elif 12 < sl <= 18 or 4 <= sl < 6:
            score += 3
        elif sl < 4 or sl > 22:
            score -= 5

    # --- ML BOOSTS ---
    # 1) GP / acquisition-based boost (quality + uncertainty)
    if use_ml and exp.metrics.acquisition_score is not None:
        # acquisition_score tracks learned quality objective
        # If acquisition_score is negative (bad predictions), we still want to nudged
        # but maybe floor it so it doesn't destroy the rank
        score += max(-500, exp.metrics.acquisition_score * 5)

    # 2) Process decision-tree boost (learned from full history)
    if use_tree and exp.metrics.tree_predicted_quality is not None:
        # Tree prediction is on the same quality scale as the GP target.
        # Floor the tree prediction at -500 to prevent extreme penalties from dominating
        tree_val = max(-500, float(exp.metrics.tree_predicted_quality))
        score += tree_val * 0.5

    return score


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def batch_well_labels(n: int) -> List[str]:
    """
    Plate-style well IDs A1..A12, B1..B12, ... (row-major). Matches next_batch_solution_mix.csv.
    """
    if n <= 0:
        return []
    wells = [f"{chr(65 + i)}{j + 1}" for i in range(8) for j in range(12)]
    if n > len(wells):
        raise ValueError(f"Batch size {n} exceeds supported layout ({len(wells)} wells).")
    return wells[:n]


def experiment_solution_key(exp: ExperimentParams) -> str:
    """Unique ink identity for robot vial slot mapping (includes additive molarities when set)."""
    parts = [str(exp.material_system), exp.solvent_name, f"{exp.concentration}M"]
    if exp.csbr_M is not None or exp.neabr_M is not None or exp.ce_M is not None:
        def _m(x: Optional[float]) -> float:
            return round(float(x), 6) if x is not None else 0.0

        parts.append(
            f"CsBr={_m(exp.csbr_M)} NEABr={_m(exp.neabr_M)} CE={_m(exp.ce_M)}"
        )
    return " | ".join(parts)


def save_analysis_file(experiments: List[ExperimentParams], output_file: str):
    rows = [{
        'Experiment_ID': i,
        'Rank_Score': round(exp.rank_score, 1),
        'System': str(exp.material_system),
        'Solvent': exp.solvent_name,
        'Concentration_M': exp.concentration,
        'Spin_Speed_rpm': exp.spin_speed,
        'Anneal_Temp_C': exp.annealing_temp,
        'Tree_Predicted_Quality': round(exp.metrics.tree_predicted_quality, 1) if exp.metrics.tree_predicted_quality is not None else None,
        'Uncertainty_Score': round(exp.metrics.uncertainty_score, 2) if exp.metrics.uncertainty_score is not None else None,
        'ML_Predicted_Quality': exp.metrics.predicted_performance,
        'ML_Acquisition_Score': exp.metrics.acquisition_score
    } for i, exp in enumerate(experiments, 1)]

    df = pd.DataFrame(rows).sort_values(by='Rank_Score', ascending=False)
    df.to_csv(output_file, index=False)
    print(f"Analysis saved to {output_file}")


def get_solution_mapping(experiments: List[ExperimentParams]) -> Dict[str, int]:
    mapping = {}
    next_sol_num = 1
    for exp in experiments:
        key = experiment_solution_key(exp)
        if key not in mapping:
            mapping[key] = next_sol_num
            next_sol_num += 1
    return mapping


def save_prep_instructions(mapping: Dict[str, int], experiments: List[ExperimentParams], output_file: str):
    wells = batch_well_labels(len(experiments))
    with open(output_file, 'w') as f:
        f.write("=== ROBOT RUN PLAN & PREP INSTRUCTIONS ===\n\n")
        f.write("STEP 1: SOLUTION PREPARATION\n")
        f.write("----------------------------\n")
        f.write("Well labels match data/next_batch_solution_mix.csv (one prep vial per row).\n\n")
        for well, exp in zip(wells, experiments):
            sol_key = experiment_solution_key(exp)
            sol_num = mapping[sol_key]
            f.write(f"[{well}]  Robot solution/vial slot {sol_num}\n")
            f.write(
                f"    Ink: {exp.material_system} in {exp.solvent_name}, "
                f"[PbBr2]={exp.concentration} M\n"
            )
            if exp.csbr_M is not None or exp.neabr_M is not None or exp.ce_M is not None:
                cs = exp.csbr_M if exp.csbr_M is not None else 0.0
                ne = exp.neabr_M if exp.neabr_M is not None else 0.0
                ce = exp.ce_M if exp.ce_M is not None else 0.0
                f.write(f"    Additives (M): CsBr={cs:.3f}, NEABr={ne:.3f}, crown ether={ce:.3f}\n")
            f.write("\n")

        f.write("\nSTEP 2: SUBSTRATE LOADING PLAN\n")
        f.write("------------------------------\n")
        for i, exp in enumerate(experiments, 1):
            sol_key = experiment_solution_key(exp)
            sol_num = mapping[sol_key]
            well = wells[i - 1]
            f.write(f"Substrate {i}: Load ink from well {well} (robot Solution Number {sol_num})\n")
            f.write(f"   -> Ink: {exp.material_system} | {exp.solvent_name} | [PbBr2]={exp.concentration} M\n")
            if exp.two_step_enabled:
                f.write(f"   -> Spin (Two-Step): Step1={exp.first_spin_speed}rpm/{exp.first_spin_time}s, Step2={exp.spin_speed}rpm/{exp.spin_time}s\n")
            else:
                f.write(f"   -> Spin (Single): {exp.spin_speed}rpm/{exp.spin_time}s\n")
            f.write(f"   -> Annealing: {exp.annealing_temp}°C for {exp.annealing_time}s ({exp.annealing_time//60}min)\n")
            f.write(f"   -> Recipe: Accel={exp.spin_acceleration}rpm/s, Height={exp.dispense_height}mm, Vol={exp.drop_volume}uL\n")
            if exp.antisolvent:
                f.write(f"   -> Antisolvent: {exp.antisolvent.name}, Vol={exp.antisolvent_volume}uL, Timing={exp.antisolvent_timing}s\n")
            else:
                f.write(f"   -> Antisolvent: None\n")
            f.write(f"   -> Reason: Rank Score {exp.rank_score:.1f}")
            if exp.metrics.composition_acquisition_score is not None:
                f.write(f" [Composition Dual-GP Score: {exp.metrics.composition_acquisition_score:.3f}]")
            f.write("\n\n")
        f.write("==========================================\n")
    print(f"Prep instructions saved to {output_file}")


def save_selected_candidates_detailed(experiments: List[ExperimentParams], output_file: str):
    """
    Saves a detailed CSV of selected candidate experiments, including all parameters.
    """
    rows = []
    for i, exp in enumerate(experiments, 1):
        row_data = {
            'Experiment_ID': i,
            'System': str(exp.material_system),
            'FA_Ratio': extract_fa_ratio_from_composition(str(exp.material_system)),
            'Solvent_Name': exp.solvent_name,
            'PbBr2_M': exp.concentration,
            'CsBr_M': exp.csbr_M,
            'NEABr_M': exp.neabr_M,
            'CE_M': exp.ce_M,
            'Drop_Volume_uL': exp.drop_volume,
            'Dispense_Height_mm': exp.dispense_height,
            'Two_Step_Enabled': exp.two_step_enabled,
            'First_Spin_Speed_rpm': exp.first_spin_speed,
            'First_Spin_Time_s': exp.first_spin_time,
            'First_Spin_Accel_rpm_s': exp.first_spin_acceleration,
            'Spin_Speed_rpm': exp.spin_speed,
            'Spin_Time_s': exp.spin_time,
            'Spin_Accel_rpm_s': exp.spin_acceleration,
            'Anneal_Temp_C': exp.annealing_temp,
            'Anneal_Time_s': exp.annealing_time,
            'Antisolvent_Name': exp.antisolvent.name if exp.antisolvent else 'None',
            'Antisolvent_Volume_uL': exp.antisolvent_volume,
            'Antisolvent_Timing_s': exp.antisolvent_timing,
            'Rank_Score': round(exp.rank_score, 1),
            'Tree_Predicted_Quality': round(exp.metrics.tree_predicted_quality, 1) if exp.metrics.tree_predicted_quality is not None else None,
            'Uncertainty_Score': round(exp.metrics.uncertainty_score, 2) if exp.metrics.uncertainty_score is not None else None,
            'ML_Predicted_Quality': round(exp.metrics.predicted_performance, 2) if exp.metrics.predicted_performance is not None else None,
            'ML_Acquisition_Score': round(exp.metrics.acquisition_score, 2) if exp.metrics.acquisition_score is not None else None,
            'Tolerance_Factor': round(exp.metrics.tolerance_factor, 3),
            'Octahedral_Factor': round(exp.metrics.octahedral_factor, 3),
            'Solvent_Boiling_Point_Avg': round(exp.metrics.solvent_boiling_point_avg, 1),
            'Solvent_DN': round(exp.metrics.solvent_dn, 2),
            'Solvent_AN': round(exp.metrics.solvent_an, 2),
            'Antisolvent_Miscibility_Gap': round(exp.metrics.antisolvent_miscibility_gap, 2) if exp.metrics.antisolvent_miscibility_gap is not None else None,
            'Estimated_Drying_Time': round(exp.metrics.estimated_drying_time, 5),
            'Film_Thickness_Metric': round(exp.metrics.film_thickness_metric, 1),
            'Film_Uniformity_Metric': round(exp.metrics.film_uniformity_metric, 4) if exp.metrics.film_uniformity_metric is not None else None,
            'Precursor_Solvent_Delta': round(exp.metrics.precursor_solvent_delta, 2) if exp.metrics.precursor_solvent_delta is not None else None,
            'A_Site_Lattice_Spacing': round(exp.metrics.a_site_lattice_spacing, 2) if exp.metrics.a_site_lattice_spacing is not None else None,
            'A_Site_Dipole_Moment': round(exp.metrics.a_site_dipole_moment, 2) if exp.metrics.a_site_dipole_moment is not None else None,
            'A_Site_Spacer_Length': round(exp.metrics.a_site_spacer_length, 2) if exp.metrics.a_site_spacer_length is not None else None,
        }
        rows.append(row_data)

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Detailed selected candidates saved to {output_file}")


def save_next_batch_solution_mix(experiments: List[ExperimentParams], output_file: str):
    """
    Save a simple composition/mixing table (all in M) similar to external Dual-GP's
    `next_batch_compositions.csv`.
    """
    wells = batch_well_labels(len(experiments))
    rows = []
    for idx, exp in enumerate(experiments):
        rows.append(
            {
                "Well": wells[idx],
                "Solvent": exp.solvent_name,
                "PbBr2_M": round(float(exp.concentration), 3),
                "CsBr_M": round(float(exp.csbr_M), 3) if exp.csbr_M is not None else None,
                "NEABr_M": round(float(exp.neabr_M), 3) if exp.neabr_M is not None else None,
                "CE_M": round(float(exp.ce_M), 3) if exp.ce_M is not None else None,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, float_format="%.3f")
    print(f"Next-batch solution mix saved to {output_file}")


def _spin_steps_for_robot_yaml(exp: ExperimentParams) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Return ((rpm, time_s, accel), (rpm, time_s, accel)) for first and second spin segments.
    Matches `microwell trial with syringe quench.yaml`: Spin_* then Second Spin_*.
    """
    if exp.two_step_enabled:
        first = (
            int(exp.first_spin_speed),
            int(exp.first_spin_time),
            int(exp.first_spin_acceleration),
        )
        second = (int(exp.spin_speed), int(exp.spin_time), int(exp.spin_acceleration))
        return first, second
    main = (int(exp.spin_speed), int(exp.spin_time), int(exp.spin_acceleration))
    return main, main


def save_robot_recipe_yaml(
    experiments: List[ExperimentParams],
    mapping: Dict[str, int],
    output_file: str,
    robot_recipe_cfg: Optional[dict] = None,
):
    """
    Emit robot YAML matching `microwell trial with syringe quench.yaml` (Process: Syringe Quenching).

    Key order and fields follow that template. Monte Carlo supplies annealing, both spin segments
    (two-step MC or duplicated single-step), pipette height, perovskite solution #/volume,
    quench delay (antisolvent timing) and Second Solution Volume in mL (antisolvent from MC).
    Static instrument fields come from config `robot_recipe` when provided.
    """
    rc = dict(robot_recipe_cfg or {})
    anneal_sta = int(rc.get("annealing_station_number", 1))
    dispense_delay = float(rc.get("dispense_spin_blade_delay_s", 0.0))
    dispense_rate = float(rc.get("dispensing_rate_ml_min", 50.0))
    hz_offset = int(rc.get("horizontal_quench_offset_mm", 1))
    lc = int(rc.get("liquid_class_index", 98))
    mix_cycles = int(rc.get("mixing_cycles_aspiration", 1))
    rack = int(rc.get("pipette_rack_number", 1))
    quench_h = rc.get("quench_height_mm", 10)
    if isinstance(quench_h, float) and quench_h.is_integer():
        quench_h = int(quench_h)
    sol_sta = int(rc.get("solvent_station_number", 1))
    spincoater = int(rc.get("spincoater_number", 1))
    tip = str(rc.get("tip_type", "1000 µL"))

    recipe_dict = {"default": {"step_1": {"Process": "Syringe Quenching"}}}

    experiments_to_run = experiments[:8]
    if len(experiments_to_run) < 8:
        print(f"Warning: Only {len(experiments_to_run)} variations available. Filling remaining slots with duplicates.")
        while len(experiments_to_run) < 8:
            experiments_to_run.append(experiments_to_run[len(experiments_to_run) % len(experiments)])

    step = recipe_dict["default"]["step_1"]

    for i, exp in enumerate(experiments_to_run, 1):
        solution_number = mapping[experiment_solution_key(exp)]
        (v1, t1, a1), (v2, t2, a2) = _spin_steps_for_robot_yaml(exp)

        quench_delay_s = float(exp.antisolvent_timing) if exp.antisolvent else 0.0
        second_ml = (float(exp.antisolvent_volume) / 1000.0) if exp.antisolvent else 0.0

        # Same key order as microwell trial with syringe quench.yaml
        substrate_config = {
            "Annealing Station Number": anneal_sta,
            "Annealing Temperature (°C)": int(exp.annealing_temp),
            "Annealing Time (s)": int(exp.annealing_time),
            "Discard Pipettes": True,
            "Dispense-Spin/Blade Delay (s)": dispense_delay,
            "Dispensing Rate (mL/min)": dispense_rate,
            "Horizontal Quench Offset (mm)": hz_offset,
            "Liquid Class Index Aspiration": lc,
            "Liquid Class Index Dispensing": lc,
            "Mixing Cycles Aspiration (0-9)": mix_cycles,
            "Pipette Dispensing Height Above Substrate (mm)": float(exp.dispense_height),
            "Pipette Rack Number": rack,
            "Put on Hotplate": True,
            "Quench Height Above Substrate (mm)": quench_h,
            "Quench-Spin Delay (s)": float(quench_delay_s),
            "Second Solution Volume (mL)": round(second_ml, 4),
            "Second Spin Acceleration (rpm/s)": a2,
            "Second Spin Time (s)": t2,
            "Second Spin Velocity (rpm)": v2,
            "Solution Number": int(solution_number),
            "Solution Volume (uL)": int(exp.drop_volume),
            "Solvent Station Number": sol_sta,
            "Spin Acceleration (rpm/s)": a1,
            "Spin Time (s)": t1,
            "Spin Velocity (rpm)": v1,
            "Spincoater Number": spincoater,
            "Tip Type": tip,
        }
        step[f"Substrate_{i}"] = substrate_config

    with open(output_file, 'w', encoding='utf-8', newline='\n') as f:
        yaml.dump(recipe_dict, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    print(f"YAML Recipe saved to {output_file}")


def get_variations_of_top_candidates(candidates: List[ExperimentParams], target_count: int = 8, exploration_fraction: float = 0.25, tree_learner=None) -> List[ExperimentParams]:
    """
    Select diverse candidates balancing exploitation (high score) and exploration (high uncertainty).
    
    When GP isn't learning well, prioritizes Decision Tree predictions.
    
    Args:
        candidates: List of scored candidates
        target_count: Total number to select
        exploration_fraction: Fraction of batch to use for exploration (max-uncertainty points)
        tree_learner: ProcessTreeLearner instance (optional, for tree-based selection)
    """
    # Check if Decision Tree is available and should be prioritized
    use_tree_priority = False
    if tree_learner is not None and tree_learner.is_trained:
        # Check if candidates have tree predictions
        tree_predictions = [c.metrics.tree_predicted_quality for c in candidates 
                           if c.metrics.tree_predicted_quality is not None]
        if len(tree_predictions) > 0:
            # Check if GP predictions are informative (not all the same)
            gp_predictions = [c.metrics.predicted_performance for c in candidates 
                             if c.metrics.predicted_performance is not None]
            if len(gp_predictions) > 0:
                gp_range = max(gp_predictions) - min(gp_predictions)
                # If GP predictions have very small range (< 0.1), GP isn't learning
                if gp_range < 0.1:
                    use_tree_priority = True
                    print("  Decision Tree priority: GP predictions show little variation, using Tree predictions")
            else:
                # No GP predictions, use tree
                use_tree_priority = True
                print("  Decision Tree priority: No GP predictions available, using Tree predictions")
    
    # Sort by appropriate metric
    if use_tree_priority:
        # Sort by Decision Tree predictions (primary) with rank score as tiebreaker
        sorted_by_score = sorted(candidates, 
                                key=lambda x: (x.metrics.tree_predicted_quality if x.metrics.tree_predicted_quality is not None else -1e6, 
                                             x.rank_score),
                                reverse=True)
    else:
        # Sort by rank score (exploitation)
        sorted_by_score = sorted(candidates, key=lambda x: x.rank_score, reverse=True)

    # If GP model is trained, also sort by uncertainty (exploration)
    has_uncertainty = any(c.metrics.uncertainty_score is not None and c.metrics.uncertainty_score > 0 
                          for c in candidates)
    
    n_explore = max(1, int(target_count * exploration_fraction))  # At least 1 exploration point
    n_exploit = target_count - n_explore
    
    selected = []
    
    # New logic: Prioritize top N unique compositions, then fill with spin variations
    N_COMPOSITIONS_TO_SELECT = 3 # Fixed to 3 unique compositions as requested

    # First, get all unique material systems from sorted candidates
    unique_systems_map = {} # Maps system_str to its best candidate (for scoring)
    for cand in sorted_by_score:
        sys_str = str(cand.material_system)
        if sys_str not in unique_systems_map:
            unique_systems_map[sys_str] = cand
    
    # Sort unique systems by the rank score of their best candidate
    sorted_unique_systems = sorted(unique_systems_map.items(), key=lambda item: item[1].rank_score, reverse=True)
    
    # Select the top N compositions
    top_compositions = [sys_str for sys_str, _ in sorted_unique_systems[:N_COMPOSITIONS_TO_SELECT]]
    
    print(f"  Focusing on top {len(top_compositions)} compositions: {', '.join(top_compositions)}")

    # Create buckets for candidates belonging to these top compositions
    composition_buckets = {sys: [] for sys in top_compositions}
    other_candidates = []

    for cand in candidates:
        sys_str = str(cand.material_system)
        if sys_str in composition_buckets:
            composition_buckets[sys_str].append(cand)
        else:
            other_candidates.append(cand)

    # Sort candidates within each bucket by rank score
    for sys in composition_buckets:
        composition_buckets[sys].sort(key=lambda x: x.rank_score, reverse=True)

    # Select candidates, prioritizing diversity in spin coating within the top compositions
    selected_from_top_compositions = []
    seen_spin_configs = set() # To ensure unique spin configs per composition

    # Distribute selections across the top compositions
    max_iterations = target_count * 2 # Safety break
    current_iteration = 0
    
    while len(selected_from_top_compositions) < target_count and current_iteration < max_iterations:
        for sys in top_compositions:
            if len(selected_from_top_compositions) >= target_count:
                break

            # Try to pick a candidate with a new spin config for this system
            found_candidate = False
            for cand in composition_buckets[sys]:
                spin_config_key = (cand.spin_speed, cand.spin_time, cand.first_spin_speed, cand.first_spin_time)
                if cand not in selected_from_top_compositions and (sys, spin_config_key) not in seen_spin_configs:
                    selected_from_top_compositions.append(cand)
                    seen_spin_configs.add((sys, spin_config_key))
                    found_candidate = True
                    break
            
            # If no new spin config found, pick the next best unique candidate for this system
            if not found_candidate:
                for cand in composition_buckets[sys]:
                    if cand not in selected_from_top_compositions:
                        selected_from_top_compositions.append(cand)
                        found_candidate = True
                        break
            
        current_iteration += 1

    selected = selected_from_top_compositions[:target_count]

    # If we still don't have enough (e.g., if there aren't enough variations within top 3 compositions)
    # Fill remaining slots with the best overall candidates that are not yet selected
    if len(selected) < target_count:
        remaining_needed = target_count - len(selected)
        for cand in sorted_by_score:
            if cand not in selected:
                selected.append(cand)
                remaining_needed -= 1
            if remaining_needed <= 0:
                break
    
    return selected


# ---------------------------------------------------------------------------
# LLM Agent Integration
# ---------------------------------------------------------------------------
def run_llm_agent_cycle(
    cfg: dict,
    history_df: pd.DataFrame,
    candidates_df: pd.DataFrame = None,
    film_images: List[str] = None,
    auto_apply: bool = False,
    min_confidence: float = None
) -> dict:
    """
    Run the LLM agent optimization cycle.
    
    Args:
        cfg: Config dictionary
        history_df: Experiment history
        candidates_df: GP candidates DataFrame
        film_images: List of film image paths
        auto_apply: Whether to auto-apply suggestions
        
    Returns:
        Agent results dictionary
    """
    api_key = cfg.get('llm_agent', {}).get('api_key') or os.environ.get('GEMINI_API_KEY')

    if not api_key:
        print("Warning: No Gemini API key found. Skipping LLM agent.")
        print("Set GEMINI_API_KEY env var or add llm_agent.api_key to config.yaml")
        return {}
    
    try:
        from src.llm_agent import create_agent
        
        print("\n" + "=" * 60)
        print("Running LLM Agent Optimization Cycle")
        print("=" * 60)
        
        agent = create_agent(api_key, 'config.yaml')
        
        # Convert candidates to DataFrame if needed
        if candidates_df is None and 'candidates' in cfg:
            candidates_df = pd.DataFrame()
        
        # Use provided min_confidence or fall back to config/default
        if min_confidence is None:
            min_confidence = cfg.get('llm_agent', {}).get('min_confidence', 0.7)
        
        results = agent.run_optimization_cycle(
            history_df=history_df,
            candidates_df=candidates_df,
            film_image_paths=film_images,
            auto_apply=auto_apply,
            min_confidence=min_confidence
        )
        
        # Print summary
        print("\n" + "-" * 40)
        print("LLM Agent Summary")
        print("-" * 40)
        summary = agent.get_recommendation_summary()
        print(summary)
        
        return results
        
    except ImportError as e:
        print(f"Warning: Could not import LLM agent: {e}")
        print("Install with: pip install google-generativeai Pillow")
        return {}
    except Exception as e:
        print(f"Warning: LLM agent error: {e}")
        return {}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_single_optimization_cycle(cfg, generator, learner, history_df, transition_experiments, args, tree_learner: ProcessTreeLearner = None, comp_gp=None):
    """Run a single optimization cycle (original behavior)."""
    # 5. Generate candidates
    n_attempts = cfg['generation'].get('n_attempts', 5000)
    print(f"Running Monte Carlo Simulation ({n_attempts} attempts)...")
    candidates = generator.generate_candidates(n_attempts=n_attempts)

    # 6. Optional single-composition filter
    target_cfg = cfg.get('target', {})
    if target_cfg.get('single_composition_mode', False):
        target_str = target_cfg.get('target_system', '')
        before = len(candidates)
        candidates = [c for c in candidates if str(c.material_system) == target_str]
        print(f"Single-composition mode ON: kept {len(candidates)} of {before} for {target_str}")
        if not candidates:
            print("No candidates found for target. Relax filters or disable single_composition_mode.")
            return

    # 7. No process-space GP scoring.
    # Monte Carlo + ProcessTreeLearner will handle process optimization; a separate Dual-GP guides composition (Cs/NEA/CE).

    # 7a. Separate Dual-GP in composition-additive space (Cs/NEA/CE)
    comp_dual_cfg = cfg.get('composition_dual_gp', {}) or {}
    comp_dual = None
    if comp_dual_cfg.get('enabled', True):
        try:
            comp_dual = DualCompositionGP(cfg, learner)
            comp_dual.train(history_df)
            if comp_dual.is_trained:
                comp_dual.score_candidates(candidates, transition_experiments=transition_experiments)
        except Exception as e:
            print(f"Warning: Composition Dual-GP unavailable: {e}")

    # 7b. Optional process decision-tree learner (Monte Carlo decision engine)
    if tree_learner is not None and tree_learner.is_trained:
        tree_learner.score_candidates(candidates)

    for exp in candidates:
        exp.rank_score = calculate_suitability_score(
            exp,
            use_ml=False,
            use_tree=bool(tree_learner and tree_learner.is_trained),
        )
        # Optional composition-space boost (kept separate from process GP)
        if exp.metrics.composition_acquisition_score is not None:
            boost_w = float(comp_dual_cfg.get('score_boost_weight', 0.0))
            exp.rank_score += boost_w * float(exp.metrics.composition_acquisition_score)

    print(f"Simulation complete. Found {len(candidates)} valid experimental conditions.")

    if not candidates:
        print("No valid candidates found.")
        return

    # 8. Outputs (simplified)
    output_dir = get_output_dir(cfg)

    # Initialize a ConvergenceTracker for this single run to record stats
    from src.convergence import ConvergenceTracker
    tracker = ConvergenceTracker()
    # Record this single optimization cycle
    best_quality = max(c.rank_score for c in candidates)
    sorted_candidates = sorted(candidates, key=lambda x: x.rank_score, reverse=True)
    top10_uncertainties = [c.metrics.uncertainty_score for c in sorted_candidates[:10] 
                          if c.metrics.uncertainty_score is not None]
    avg_uncertainty_top10 = np.mean(top10_uncertainties) if top10_uncertainties else 0.0
    
    # Count unique conditions
    unique_conditions = set()
    for c in candidates:
        from src.learner import extract_fa_ratio_from_composition
        fa_ratio = extract_fa_ratio_from_composition(str(c.material_system))
        key = (round(fa_ratio, 2), c.annealing_temp, c.spin_speed)
        unique_conditions.add(key)

    tracker.record_cycle(
        cycle=len(tracker.history) + 1,
        best_quality=best_quality,
        avg_uncertainty_top10=avg_uncertainty_top10,
        n_candidates=len(candidates),
        n_unique_conditions=len(unique_conditions)
    )

    # (Plotting disabled in the simplified workflow.)

    batch_size = cfg['generation'].get('batch_size', 8)
    
    # Check if GP is learning
    gp_learning = False
    if learner.is_trained:
        gp_preds = [c.metrics.predicted_performance for c in candidates 
                   if c.metrics.predicted_performance is not None]
        if len(gp_preds) > 0:
            gp_range = max(gp_preds) - min(gp_preds)
            gp_learning = gp_range >= 0.1
    
    # Select based on what's available
    if tree_learner is not None and tree_learner.is_trained and not gp_learning:
        print("\nUsing Decision Tree-based selection (GP not learning)...")
        from src.learner import extract_fa_ratio_from_composition
        
        # Filter by best parameter ranges
        filtered = []
        for c in candidates:
            fa_ratio = extract_fa_ratio_from_composition(str(c.material_system))
            if (0.6 <= fa_ratio <= 0.8 and 
                3000 <= c.spin_speed <= 4000 and 
                140 <= c.annealing_temp <= 145):
                filtered.append(c)
        
        if len(filtered) >= batch_size:
            candidates_to_select = filtered
        else:
            candidates_to_select = candidates
        
        candidates_to_select.sort(
            key=lambda x: (x.metrics.tree_predicted_quality if x.metrics.tree_predicted_quality is not None else -1e6),
            reverse=True
        )
        selected_experiments = candidates_to_select[:batch_size]
    else:
        selected_experiments = get_variations_of_top_candidates(
            candidates, 
            target_count=batch_size,
            tree_learner=tree_learner
        )
    
    print(f"\nSelected {len(selected_experiments)} optimal varied experiments for {batch_size} substrates.")

    save_transition_experiments(selected_experiments)
    append_batch_to_experiments_log(selected_experiments, cfg)

    solution_mapping = get_solution_mapping(selected_experiments)
    save_prep_instructions(solution_mapping, selected_experiments, os.path.join('data', 'prep_instructions.txt'))
    save_robot_recipe_yaml(
        selected_experiments,
        solution_mapping,
        os.path.join('data', 'robot_recipe_generated.yaml'),
        robot_recipe_cfg=cfg.get('robot_recipe'),
    )

    # Save a simple mixing table (all concentrations in M)
    mix_output_path = os.path.join(output_dir, 'next_batch_solution_mix.csv')
    save_next_batch_solution_mix(selected_experiments, mix_output_path)

    # Show improvement stats for this run
    stats = tracker.get_improvement_stats()
    print(f"\nOptimization Progress:")
    print(f"  Best Quality Achieved: {stats['best_quality']:.1f}")
    if stats['cycles'] > 1:
        print(f"  Total Improvement: {stats['total_improvement']:+.1f}%")
        print(f"  Avg Improvement/Cycle: {stats['avg_improvement_per_cycle']:+.1f}%")

    # 9. Optional LLM Agent Optimization Cycle
    if args.with_agent or cfg.get('llm_agent', {}).get('enabled', False):
        # Create candidates DataFrame for agent
        candidates_df = pd.read_csv(analysis_path)
        
        run_llm_agent_cycle(
            cfg=cfg,
            history_df=history_df,
            candidates_df=candidates_df,
            film_images=args.film_images,
            auto_apply=args.auto_apply,
            min_confidence=args.min_confidence
        )


def run_iterative_optimization(cfg, generator, learner, history_df, transition_experiments, args, tree_learner: ProcessTreeLearner = None, comp_gp=None):
    """
    Run iterative optimization with GP-guided sampling and convergence checking.
    
    Features:
    - Cycle 1: Random Monte Carlo (baseline exploration)
    - Cycle 2+: GP-guided Monte Carlo (focused sampling)
    - Convergence checking after each cycle
    - Adaptive exploration rate
    - Multi-objective optimization option
    """
    import numpy as np
    
    # Load settings
    gen_cfg = cfg['generation']
    n_attempts = gen_cfg.get('n_attempts', 5000)
    max_cycles = gen_cfg.get('max_cycles', 10)
    min_cycles = gen_cfg.get('min_cycles', 3)
    improvement_threshold = gen_cfg.get('improvement_threshold', 0.05)
    uncertainty_threshold = gen_cfg.get('uncertainty_threshold', 10.0)
    quality_threshold = gen_cfg.get('quality_threshold')
    base_exploration_rate = gen_cfg.get('exploration_rate', 0.15)
    use_gp_guided = gen_cfg.get('use_gp_guided_sampling', True)
    use_pareto = gen_cfg.get('use_pareto_front', False)
    batch_size = gen_cfg.get('batch_size', 8)
    
    # Initialize convergence tracker
    tracker = ConvergenceTracker()
    
    # Check if we have existing optimization history
    existing_cycles = len(tracker.history)
    start_cycle = existing_cycles  # Start from where we left off
    
    print("\n" + "="*60)
    print("ITERATIVE OPTIMIZATION MODE")
    print("="*60)
    if existing_cycles > 0:
        print(f"Resuming from Cycle {existing_cycles + 1} (found {existing_cycles} previous cycles)")
        stats = tracker.get_improvement_stats()
        print(f"Previous best quality: {stats['best_quality']:.1f}")
    else:
        print(f"Starting Cycle 1 (no previous optimization history found)")
        print(f"Note: Existing experiments in log will be used to train GP")
    print(f"Max cycles: {max_cycles}")
    print(f"Min cycles before convergence check: {min_cycles}")
    print(f"GP-guided sampling: {use_gp_guided}")
    print(f"Pareto optimization: {use_pareto}")
    print("="*60 + "\n")
    
    # Adjust max_cycles to account for existing cycles
    remaining_cycles = max_cycles - start_cycle
    if remaining_cycles <= 0:
        print(f"Already completed {existing_cycles} cycles (max: {max_cycles}).")
        print("Increase max_cycles in config.yaml to continue, or check convergence status.")
        return
    
    for cycle in range(remaining_cycles):
        current_cycle = start_cycle + cycle
        print(f"\n{'='*60}")
        print(f"CYCLE {current_cycle + 1} / {max_cycles}")
        print(f"{'='*60}\n")
        
        # Determine exploration rate (adaptive)
        if current_cycle > 0:
            exploration_rate = tracker.get_recommended_exploration_rate(base_exploration_rate)
            print(f"Adaptive exploration rate: {exploration_rate:.2%}")
        else:
            exploration_rate = base_exploration_rate
        
        # Generate candidates
        if current_cycle == 0 or not use_gp_guided or not learner.is_trained:
            # Cycle 1: Random Monte Carlo
            print(f"Running Random Monte Carlo Simulation ({n_attempts} attempts)...")
            candidates = generator.generate_candidates(n_attempts=n_attempts)
        else:
            # Cycle 2+: GP-guided Monte Carlo
            print(f"Running GP-Guided Monte Carlo Simulation ({n_attempts} attempts)...")
            candidates = generator.generate_gp_guided_candidates(
                learner=learner,
                n_attempts=n_attempts,
                exploration_rate=exploration_rate
            )
        
        # Optional single-composition filter
        target_cfg = cfg.get('target', {})
        if target_cfg.get('single_composition_mode', False):
            target_str = target_cfg.get('target_system', '')
            before = len(candidates)
            candidates = [c for c in candidates if str(c.material_system) == target_str]
            print(f"Single-composition mode: kept {len(candidates)} of {before}")
            if not candidates:
                print("No candidates found for target.")
                break
        
        # No process-space GPBO. Process optimization is handled by the tree learner + Monte Carlo.

        # Optional process decision-tree learner
        if tree_learner is not None and tree_learner.is_trained:
            tree_learner.score_candidates(candidates)
        
        for exp in candidates:
            exp.rank_score = calculate_suitability_score(
                exp,
                use_ml=False,
                use_tree=bool(tree_learner and tree_learner.is_trained),
            )

        # Separate Dual-GP in composition-additive space (Cs/NEA/CE)
        comp_dual_cfg = cfg.get('composition_dual_gp', {}) or {}
        if comp_dual_cfg.get('enabled', True):
            try:
                comp_dual = DualCompositionGP(cfg, learner)
                comp_dual.train(history_df)
                if comp_dual.is_trained:
                    comp_dual.score_candidates(candidates, transition_experiments=transition_experiments)
                    boost_w = float(comp_dual_cfg.get('score_boost_weight', 0.0))
                    if boost_w:
                        for exp in candidates:
                            if exp.metrics.composition_acquisition_score is not None:
                                exp.rank_score += boost_w * float(exp.metrics.composition_acquisition_score)
            except Exception as e:
                print(f"Warning: Composition Dual-GP unavailable: {e}")
        
        if not candidates:
            print("No valid candidates found.")
            break
        
        # Calculate statistics
        best_quality = max(c.rank_score for c in candidates)
        sorted_candidates = sorted(candidates, key=lambda x: x.rank_score, reverse=True)
        top10_uncertainties = [c.metrics.uncertainty_score for c in sorted_candidates[:10] 
                              if c.metrics.uncertainty_score is not None]
        avg_uncertainty_top10 = np.mean(top10_uncertainties) if top10_uncertainties else 0.0
        
        # Count unique conditions
        unique_conditions = set()
        for c in candidates:
            from src.learner import extract_fa_ratio_from_composition
            fa_ratio = extract_fa_ratio_from_composition(str(c.material_system))
            key = (round(fa_ratio, 2), c.annealing_temp, c.spin_speed)
            unique_conditions.add(key)
        
        # Record cycle
        record = tracker.record_cycle(
            cycle=current_cycle + 1,
            best_quality=best_quality,
            avg_uncertainty_top10=avg_uncertainty_top10,
            n_candidates=len(candidates),
            n_unique_conditions=len(unique_conditions)
        )
        
        # Print cycle summary
        print(f"\nCycle {current_cycle + 1} Summary:")
        print(f"  Best Quality: {best_quality:.1f}")
        print(f"  Top 10 Avg Uncertainty: {avg_uncertainty_top10:.1f}")
        print(f"  Unique Conditions: {len(unique_conditions)}")
        print(f"  Total Candidates: {len(candidates)}")
        
        # Check convergence
        conv_status = {'converged': False, 'reason': 'Not checked yet', 'recommendation': 'Continue'}
        if current_cycle + 1 >= min_cycles:
            conv_status = tracker.check_convergence(
                improvement_threshold=improvement_threshold,
                uncertainty_threshold=uncertainty_threshold,
                quality_threshold=quality_threshold,
                min_cycles=min_cycles
            )
            
            print(f"\nConvergence Status: {conv_status['reason']}")
            print(f"Recommendation: {conv_status['recommendation']}")
            
            if conv_status['converged']:
                print(f"\n{'='*60}")
                print("OPTIMIZATION CONVERGED")
                print(f"{'='*60}")
        
        # Select batch
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
        
        if use_pareto and learner.is_trained and gp_learning:
            print("\nUsing Pareto-optimal selection (GP is learning)...")
            selected_experiments = select_diverse_pareto_front(
                candidates, 
                n_select=batch_size,
                objectives=['quality', 'uncertainty']
            )
        elif tree_learner is not None and tree_learner.is_trained and not gp_learning:
            print("\nUsing Decision Tree-based selection (GP not learning, prioritizing Tree predictions)...")
            # Filter by best parameter ranges, then sort by tree predictions
            from src.learner import extract_fa_ratio_from_composition
            
            # Filter candidates by best parameter ranges (from your data analysis)
            filtered = []
            for c in candidates:
                fa_ratio = extract_fa_ratio_from_composition(str(c.material_system))
                # Best ranges: FA 0.6-0.8, Spin 3000-4000, Temp 140-145
                if (0.6 <= fa_ratio <= 0.8 and 
                    3000 <= c.spin_speed <= 4000 and 
                    140 <= c.annealing_temp <= 145):
                    filtered.append(c)
            
            # If we have enough filtered candidates, use them; otherwise use all
            if len(filtered) >= batch_size:
                candidates_to_select = filtered
                print(f"  Filtered to {len(filtered)} candidates in optimal parameter ranges")
                print(f"    (FA: 0.6-0.8, Spin: 3000-4000 rpm, Temp: 140-145°C)")
            else:
                candidates_to_select = candidates
                print(f"  Using all {len(candidates)} candidates (not enough in optimal ranges)")
            
            # Sort by tree predictions
            candidates_to_select.sort(
                key=lambda x: (x.metrics.tree_predicted_quality if x.metrics.tree_predicted_quality is not None else -1e6),
                reverse=True
            )
            selected_experiments = candidates_to_select[:batch_size]
            
            # Print selection summary
            if len(selected_experiments) > 0:
                avg_fa = np.mean([extract_fa_ratio_from_composition(str(c.material_system)) for c in selected_experiments])
                avg_spin = np.mean([c.spin_speed for c in selected_experiments])
                avg_temp = np.mean([c.annealing_temp for c in selected_experiments])
                avg_tree_pred = np.mean([c.metrics.tree_predicted_quality for c in selected_experiments 
                                       if c.metrics.tree_predicted_quality is not None])
                print(f"  Selected candidates:")
                print(f"    Avg FA Ratio: {avg_fa:.3f}")
                print(f"    Avg Spin Speed: {avg_spin:.0f} rpm")
                print(f"    Avg Temperature: {avg_temp:.0f}°C")
                print(f"    Avg Tree Predicted Quality: {avg_tree_pred:.1f}")
        else:
            selected_experiments = get_variations_of_top_candidates(
                candidates, 
                target_count=batch_size,
                exploration_fraction=exploration_rate,
                tree_learner=tree_learner
            )
        
        print(f"\nSelected {len(selected_experiments)} experiments for batch.")
        
        # Save outputs (simplified)
        
        # Get composition GP if available (for plotting) - use the one already trained
        # Use the comp_gp passed to the function (already trained)
        # (Plotting disabled in the simplified workflow.)
        
        save_transition_experiments(selected_experiments)
        append_batch_to_experiments_log(selected_experiments, cfg)

        solution_mapping = get_solution_mapping(selected_experiments)
        save_prep_instructions(solution_mapping, selected_experiments, 
                               os.path.join('data', 'prep_instructions.txt'))
        save_robot_recipe_yaml(
            selected_experiments,
            solution_mapping,
            os.path.join('data', 'robot_recipe_generated.yaml'),
            robot_recipe_cfg=cfg.get('robot_recipe'),
        )
        
        # Save a simple mixing table (all concentrations in M)
        mix_output_path = os.path.join('data', 'next_batch_solution_mix.csv')
        save_next_batch_solution_mix(selected_experiments, mix_output_path)
        
        # Show improvement stats
        if current_cycle > 0:
            stats = tracker.get_improvement_stats()
            print(f"\nOptimization Progress:")
            print(f"  Total Improvement: {stats['total_improvement']:+.1f}%")
            print(f"  Avg Improvement/Cycle: {stats['avg_improvement_per_cycle']:+.1f}%")
        
        # Optional LLM Agent
        if args.with_agent or cfg.get('llm_agent', {}).get('enabled', False):
            candidates_df = pd.read_csv(analysis_path)
            run_llm_agent_cycle(
                cfg=cfg,
                history_df=history_df,
                candidates_df=candidates_df,
                film_images=args.film_images,
                auto_apply=args.auto_apply
            )
        
        # Prompt for next cycle (unless converged or last cycle)
        if current_cycle < max_cycles - 1 and not conv_status.get('converged', False):
            print(f"\n{'='*60}")
            print("NEXT STEPS:")
            print("1. Run the selected experiments")
            print("2. Add results to experiments_log.csv")
            print("3. Run 'python main.py' again to continue optimization")
            print(f"{'='*60}\n")
            
            # Optionally wait for user input
            if args.auto_apply:
                print("Auto-apply mode: Continuing to next cycle...")
                # Reload history and retrain GP + tree
                history_df = load_history(cfg)
                learner.train(history_df)
                if tree_learner is not None:
                    tree_learner.train_from_history(history_df, learner)
            else:
                response = input("Press Enter to continue (or 'q' to quit): ")
                if response.lower() == 'q':
                    break
                # Reload history and retrain GP + tree
                history_df = load_history(cfg)
                learner.train(history_df)
                if tree_learner is not None:
                    tree_learner.train_from_history(history_df, learner)
    
    # Final summary
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    stats = tracker.get_improvement_stats()
    print(f"Total Cycles: {stats['cycles']}")
    print(f"Best Quality Achieved: {stats['best_quality']:.1f}")
    print(f"Total Improvement: {stats['total_improvement']:+.1f}%")
    print(f"Average Improvement/Cycle: {stats['avg_improvement_per_cycle']:+.1f}%")
    print(f"{'='*60}\n")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Monte Carlo Decision Tree for Perovskites")
    parser.add_argument('--with-agent', action='store_true',
                        help='Run LLM agent optimization cycle after GP')
    parser.add_argument('--auto-apply', action='store_true',
                        help='Auto-apply LLM agent suggestions')
    parser.add_argument('--min-confidence', type=float, default=0.7,
                        help='Minimum confidence for auto-apply (default: 0.7)')
    parser.add_argument('--film-images', type=str, nargs='*',
                        help='Film images to analyze with LLM agent')
    parser.add_argument('--agent-only', action='store_true',
                        help='Run only the LLM agent (skip GP generation)')
    parser.add_argument('--clear-transition', action='store_true',
                        help='Clear transition experiments and start fresh')
    parser.add_argument('--show-transition', action='store_true',
                        help='Show current transition experiments and exit')
    parser.add_argument('--reset-optimization', action='store_true',
                        help='Reset optimization history to start from cycle 1')
    args = parser.parse_args()
    
    # Handle transition management commands
    if args.clear_transition:
        if os.path.exists(TRANSITION_FILE):
            os.remove(TRANSITION_FILE)
            print("✓ Transition experiments cleared!")
        else:
            print("No transition experiments to clear.")
        return
    
    if args.show_transition:
        transitions = load_transition_experiments()
        if transitions:
            print(f"\n{'='*60}")
            print(f"Current Transition Experiments ({len(transitions)} pending)")
            print(f"{'='*60}")
            for i, exp in enumerate(transitions, 1):
                print(f"\n[{i}] {exp.material_system}")
                print(f"    Solvent: {exp.solvent_name}")
                print(f"    Spin: {exp.spin_speed} rpm, Anneal: {exp.annealing_temp}°C")
                print(f"    Concentration: {exp.concentration} M")
        else:
            print("No experiments in transition.")
        return
    
    if args.reset_optimization:
        history_file = 'data/optimization_history.csv'
        if os.path.exists(history_file):
            os.remove(history_file)
            print("✓ Optimization history cleared! Next run will start from Cycle 1.")
        else:
            print("No optimization history found (already at Cycle 1).")
        return
    
    print("=" * 60)
    print("Monte Carlo Decision Tree for Perovskites (Config-Driven)")
    print("=" * 60)

    # 1. Load config
    cfg = load_config()

    # 2. Load data
    solvents_df, precursors_df = load_data(cfg)
    history_df = load_history(cfg)
    transition_experiments = prune_transition_experiments(history_df, load_transition_experiments())

    if transition_experiments:
        print(f"Loaded {len(transition_experiments)} experiments currently in transition.")

    # Handle agent-only mode
    if args.agent_only:
        print("\nRunning LLM Agent only (skipping GP generation)...")
        # Load existing candidates if available
        candidates_df = None
        candidates_path = os.path.join('data', 'candidates_analysis.csv')
        if os.path.exists(candidates_path):
            candidates_df = pd.read_csv(candidates_path)
        
        run_llm_agent_cycle(
            cfg=cfg,
            history_df=history_df,
            candidates_df=candidates_df,
            film_images=args.film_images,
            auto_apply=args.auto_apply,
            min_confidence=args.min_confidence
        )
        return

    # 3. Parse compositions from file (optional). For fixed-material experiments, this can be omitted.
    a_site_recipes = load_compositions(cfg)

    # 4. Setup learner & generator
    quality_config = cfg.get('quality_target', {})
    learner = ActiveLearner(quality_config=quality_config)
    learner.train(history_df)

    # Train process decision-tree learner on completed experiments (if enough data)
    tree_learner = ProcessTreeLearner()
    tree_learner.train_from_history(history_df, learner)

    # Optional composition-only GP (Stage 0) for FA/FuDMA binaries
    comp_gp_cfg = cfg.get('composition_gp', {}) or {}
    composition_scorer = None
    if comp_gp_cfg.get('enabled', False):
        comp_data_path = comp_gp_cfg.get('data_path')
        if comp_data_path and os.path.exists(comp_data_path):
            try:
                raw_df = pd.read_csv(comp_data_path)
                comp_gp = CompositionGP(ucb_beta=comp_gp_cfg.get('ucb_beta', 2.0))

                # Special handling for binary fit export: use only Read 1, map Composition -> FA ratio,
                # and compute a quality target from peak 1 intensity/FWHM/wavelength.
                binary_df = None
                if 'Read' in raw_df.columns and 'Composition' in raw_df.columns:
                    fa_map = load_binary_fa_ratios(cfg)
                    rows = []
                    for _, row in raw_df.iterrows():
                        if row.get('Read') != 1:
                            continue
                        comp_label = str(row.get('Composition', '')).strip()
                        if comp_label not in fa_map:
                            continue
                        fa_ratio = fa_map[comp_label]
                        # Build a minimal Series compatible with calculate_quality_target
                        pseudo = pd.Series({
                            'PL_Intensity': row.get('Peak_1_Intensity', 0.0),
                            'PL_FWHM_nm': row.get('Peak_1_FWHM', 0.0),
                            'PL_Peak_nm': row.get('Peak_1_Wavelength', None),
                            'Stability_Hours': None,
                        })
                        quality = learner.calculate_quality_target(pseudo)
                        rows.append({'FA_Ratio': fa_ratio, 'Quality_Target': quality})
                    binary_df = pd.DataFrame(rows)
                else:
                    # Generic preprocessed file with FA_Ratio / Quality_Target columns
                    binary_df = raw_df
                
                # Train on BOTH binary dataset AND experiment history
                print("Training Composition GP on binary dataset + experiment history...")
                comp_gp.train_from_binary_and_history(
                    binary_df=binary_df,
                    history_df=history_df,
                    quality_learner=learner,
                    fa_column=comp_gp_cfg.get('fa_column', 'FA_Ratio'),
                    quality_column=comp_gp_cfg.get('quality_column', 'Quality_Target'),
                )

                if comp_gp.is_trained:
                    composition_scorer = comp_gp.score_fa_ratio
            except Exception as e:
                print(f"Warning: Failed to load/train composition GP from {comp_data_path}: {e}")

    # Use a composition-aware generator that can be biased by the composition GP
    generator_cls = CompositionAwareGenerator if composition_scorer is not None else MonteCarloGenerator
    generator = generator_cls(
        solvents_df,
        precursors_df,
        a_site_recipes=a_site_recipes if a_site_recipes else None,
        solvent_config=cfg.get('solvent'),
        stoichiometry_config=cfg.get('stoichiometry'),
        composition_additives_config=(cfg.get('composition_dual_gp', {}) or {}),
        fixed_material_system_config=(cfg.get('fixed_material_system', {}) or {}),
        spin_coating_config=cfg.get('spin_coating'),
        annealing_config=cfg.get('annealing'),
        concentration_config=cfg.get('concentration'),
        antisolvent_config=cfg.get('antisolvent'),
        **({'composition_scorer': composition_scorer} if generator_cls is CompositionAwareGenerator else {})
    )

    # 5. Check if iterative optimization is enabled
    iterative_mode = cfg['generation'].get('iterative_mode', False)
    
    # Get the trained comp_gp for plotting (if it was trained)
    comp_gp_for_plotting = None
    if composition_scorer is not None:
        # The comp_gp was created above, use it directly
        comp_gp_for_plotting = comp_gp

    if iterative_mode:
        # Iterative optimization with convergence checking
        run_iterative_optimization(
            cfg=cfg,
            generator=generator,
            learner=learner,
            history_df=history_df,
            transition_experiments=transition_experiments,
            args=args,
            tree_learner=tree_learner,
            comp_gp=comp_gp_for_plotting,
        )
    else:
        # Single-shot optimization (original behavior)
        run_single_optimization_cycle(
            cfg=cfg,
            generator=generator,
            learner=learner,
            history_df=history_df,
            transition_experiments=transition_experiments,
            args=args,
            tree_learner=tree_learner,
            comp_gp=comp_gp_for_plotting,
        )


if __name__ == "__main__":
    main()
