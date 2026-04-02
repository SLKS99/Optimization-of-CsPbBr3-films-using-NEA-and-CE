from typing import List, Optional
import math
from src.models import MaterialSystem, Solvent, ExperimentParams, SolventComponent
from src import physics

# Thresholds
DELTA_SOLUBILITY_MAX = 8.0  # MPa^0.5 distance tolerance for good miscibility
# Upper bound must include high-bp polar aprotic solvents (e.g. DMSO ~189 °C) used in perovskite inks.
BOILING_POINT_RANGE = (60, 210)  # Celsius

# Octahedral factor range (B-X cage stability)
OCTAHEDRAL_FACTOR_RANGE = (0.4, 0.9)

# Lattice spacing preferred window (Angstrom) for 2D/quasi-2D
LATTICE_SPACING_RANGE = (8.0, 18.0)


def check_structural_feasibility(system: MaterialSystem) -> bool:
    """
    Checks structural feasibility using:
    1. Octahedral factor (μ = rB/rX): must be in 0.4-0.9 range for stable B-X octahedra
    2. Lattice spacing (if available): prefer spacers with spacing in 8-18 Å range
    
    Tolerance factor is NOT used here since it's not meaningful for 2D/quasi-2D.
    """
    # 1. Octahedral factor check (always applies)
    mu = physics.calculate_octahedral_factor(system)
    if not (OCTAHEDRAL_FACTOR_RANGE[0] <= mu <= OCTAHEDRAL_FACTOR_RANGE[1]):
        return False
    
    # 2. Lattice spacing check (only if spacer data available)
    # Compute weighted average lattice spacing for A-site
    spacing_sum = 0.0
    spacing_weight = 0.0
    has_spacing_data = False
    
    for comp in system.a_site:
        ls = comp.precursor.lattice_spacing
        if ls is not None and not (isinstance(ls, float) and math.isnan(ls)):
            spacing_sum += ls * comp.ratio
            spacing_weight += comp.ratio
            has_spacing_data = True
    
    if has_spacing_data and spacing_weight > 0:
        avg_spacing = spacing_sum / spacing_weight
        # Only reject if spacing is way outside preferred range
        if not (LATTICE_SPACING_RANGE[0] <= avg_spacing <= LATTICE_SPACING_RANGE[1]):
            return False
    
    # If no spacing data, pass through (don't reject for missing data)
    return True

def check_solvent_compatibility(precursors: list, solvent_components: List[SolventComponent]) -> bool:
    """
    Checks if solvent is suitable for the precursors.
    Uses effective properties of mixed solvent.
    """
    eff_solvent = physics.get_effective_solvent_properties(solvent_components)

    # Boiling point window
    if eff_solvent.boiling_point < BOILING_POINT_RANGE[0] or eff_solvent.boiling_point > BOILING_POINT_RANGE[1]:
        return False

    # DN floor when lead present
    has_lead = any(p_comp.precursor.name == 'Pb' for p_comp in precursors)
    if has_lead and eff_solvent.dn < 10:
        return False

    # Solubility parameter proximity (skip if unknown)
    for p_comp in precursors:
        delta = physics.calculate_precursor_solvent_delta(
            p_comp.precursor.solubility_parameter, eff_solvent
        )
        if delta is not None and delta > DELTA_SOLUBILITY_MAX:
            return False

    return True

def check_process_feasibility(params: ExperimentParams) -> bool:
    """
    Checks processing parameters based on user rules.
    
    Rules:
    1. Drop volume * concentration limit
    2. Single-step spin: speed must be >= 3000 rpm
    3. Two-step spin: first step can be < 3000, second step >= 3000
    4. Annealing temp must be >= 95°C
    5. Annealing time must be > 5 minutes (300s)
    """
    # 1. Drop Volume check
    if params.drop_volume * params.concentration >= 50:
        return False
    
    # 2. Spin speed constraints
    if params.two_step_enabled:
        # Two-step: second step must be >= 3000 rpm (first step can be any)
        if params.spin_speed < 3000:
            return False
    else:
        # Single-step: must be >= 3000 rpm
        if params.spin_speed < 3000:
            return False
    
    # 3. Annealing temperature must be >= 95°C
    if params.annealing_temp < 95:
        return False
    
    # 4. Annealing time must be > 5 minutes (300 seconds)
    if params.annealing_time <= 300:
        return False
        
    return True

def check_antisolvent_selection(solvent_components: List[SolventComponent], antisolvent: Solvent) -> bool:
    """
    1. Miscibility Parameter (Delta delta): 5 < |d_sol - d_anti| < 10 (User optimal)
    """
    if antisolvent is None:
        return True
        
    delta_delta = physics.check_antisolvent_miscibility(solvent_components, antisolvent)
    
    if 5 < delta_delta < 10:
        return True
        
    return False
