import math
from typing import List
from src.models import MaterialSystem, Solvent, ExperimentParams, MaterialComponent, SolventComponent

# Constants
R = 8.314  # Gas constant J/(mol*K)

def calculate_weighted_radius(components: List[MaterialComponent]) -> float:
    """Calculates effective ionic radius based on molar ratios."""
    total_ratio = sum(c.ratio for c in components)
    if total_ratio == 0:
        return 0.0
    
    weighted_sum = sum(c.precursor.ionic_radius * c.ratio for c in components)
    return weighted_sum / total_ratio

def calculate_tolerance_factor(system: MaterialSystem) -> float:
    """
    Calculates Goldschmidt Tolerance Factor (t) for mixed systems.
    t = (rA_eff + rX_eff) / (sqrt(2) * (rB_eff + rX_eff))
    """
    rA = calculate_weighted_radius(system.a_site)
    rB = calculate_weighted_radius(system.b_site)
    rX = calculate_weighted_radius(system.x_site)
    
    if rB + rX == 0:
        return 0.0
        
    return (rA + rX) / (math.sqrt(2) * (rB + rX))

def calculate_octahedral_factor(system: MaterialSystem) -> float:
    """
    Calculates Octahedral Factor (mu) for mixed systems.
    mu = rB_eff / rX_eff
    """
    rB = calculate_weighted_radius(system.b_site)
    rX = calculate_weighted_radius(system.x_site)
    
    if rX == 0:
        return 0.0
        
    return rB / rX

def get_effective_solvent_properties(solvent_components: List[SolventComponent]) -> Solvent:
    """
    Calculates weighted average properties for a mixed solvent.
    Returns a virtual 'Solvent' object representing the mix.
    """
    total_ratio = sum(s.ratio for s in solvent_components)
    if total_ratio == 0:
        # Should not happen, but return dummy
        return solvent_components[0].solvent
        
    # Weighted averages
    hsp_d = sum(s.solvent.hsp_d * s.ratio for s in solvent_components) / total_ratio
    hsp_p = sum(s.solvent.hsp_p * s.ratio for s in solvent_components) / total_ratio
    hsp_h = sum(s.solvent.hsp_h * s.ratio for s in solvent_components) / total_ratio
    bp = sum(s.solvent.boiling_point * s.ratio for s in solvent_components) / total_ratio
    visc = sum(s.solvent.viscosity * s.ratio for s in solvent_components) / total_ratio
    vp = sum(s.solvent.vapor_pressure * s.ratio for s in solvent_components) / total_ratio
    dn = sum(s.solvent.dn * s.ratio for s in solvent_components) / total_ratio
    an = sum(s.solvent.an * s.ratio for s in solvent_components) / total_ratio
    eps = sum(s.solvent.dielectric_constant * s.ratio for s in solvent_components) / total_ratio
    
    return Solvent("Mixed", hsp_d, hsp_p, hsp_h, bp, visc, vp, dn, an, eps)

def calculate_hsp_distance(solvent1: Solvent, solvent2: Solvent) -> float:
    """
    Calculates distance between two solvents in Hansen space (Ra).
    Ra^2 = 4*(d1-d2)^2 + (p1-p2)^2 + (h1-h2)^2
    """
    delta_d = solvent1.hsp_d - solvent2.hsp_d
    delta_p = solvent1.hsp_p - solvent2.hsp_p
    delta_h = solvent1.hsp_h - solvent2.hsp_h
    
    return math.sqrt(4 * delta_d**2 + delta_p**2 + delta_h**2)

def calculate_precursor_solvent_compatibility(precursor_delta: float, solvent: Solvent) -> float:
    """
    Simple distance metric between precursor solubility parameter and solvent total HSP.
    """
    return abs(precursor_delta - solvent.hsp_total)

def calculate_precursor_solvent_delta(precursor_delta: float | None, solvent: Solvent | None) -> float | None:
    """
    Returns |δ_precursor - δ_solvent_total| when both are provided; otherwise None.
    """
    if precursor_delta is None or solvent is None:
        return None
    try:
        pd = float(precursor_delta)
        if math.isnan(pd):
            return None
        sd = float(solvent.hsp_total)
        if math.isnan(sd):
            return None
        return abs(pd - sd)
    except (TypeError, ValueError):
        return None

def estimate_film_thickness(params: ExperimentParams) -> float:
    """
    Empirical Relationship for Thickness (d): d approx C * V
    """
    # d is proportional to C * V
    return params.concentration * params.drop_volume

def estimate_drying_time(params: ExperimentParams) -> float:
    """
    Proportional to Viscosity / (Vapor Pressure * Temperature)
    t_evap approx eta / (P * T)
    """
    # Temperature in Kelvin for physics calc
    temp_k = params.annealing_temp + 273.15
    
    # Use effective solvent properties
    eff_solvent = get_effective_solvent_properties(params.solvent)
    
    # Avoid division by zero
    if eff_solvent.vapor_pressure <= 0:
        return float('inf')
        
    return eff_solvent.viscosity / (eff_solvent.vapor_pressure * temp_k)

def check_antisolvent_miscibility(solvent_components: List[SolventComponent], antisolvent: Solvent) -> float:
    """
    Miscibility Parameter (Delta delta) between Mixed Solvent and Antisolvent.
    """
    eff_solvent = get_effective_solvent_properties(solvent_components)
    return abs(eff_solvent.hsp_total - antisolvent.hsp_total)
