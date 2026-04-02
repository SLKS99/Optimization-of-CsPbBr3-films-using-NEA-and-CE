from dataclasses import dataclass, field
from typing import List, Optional, Dict

@dataclass
class Solvent:
    name: str
    hsp_d: float
    hsp_p: float
    hsp_h: float
    boiling_point: float  # Celsius
    viscosity: float  # mPa*s
    vapor_pressure: float  # kPa
    dn: float  # Donor Number
    an: float  # Acceptor Number
    dielectric_constant: float

    @property
    def hsp_total(self) -> float:
        return (self.hsp_d**2 + self.hsp_p**2 + self.hsp_h**2)**0.5

@dataclass
class SolventComponent:
    solvent: Solvent
    ratio: float # 0.0 to 1.0 (volume ratio)

@dataclass
class Precursor:
    name: str
    type: str  # 'A', 'B', or 'X'
    ionic_radius: float  # Angstrom
    molecular_weight: float  # g/mol
    solubility_parameter: float
    lattice_spacing: Optional[float] = None  # Angstrom (optional; quasi-2D spacer spacing)
    dipole_moment: Optional[float] = None  # Debye (optional; for A-site cations)
    spacer_length: Optional[float] = None  # Angstrom (optional; spacer length surrogate)

@dataclass
class MaterialComponent:
    precursor: Precursor
    ratio: float  # 0.0 to 1.0 (or percentage)

@dataclass
class MaterialSystem:
    # Now lists of components instead of single precursors
    a_site: List[MaterialComponent]
    b_site: List[MaterialComponent]
    x_site: List[MaterialComponent]
    
    def __str__(self):
        # Helper to print composition nicely
        # e.g. "Cs0.5FA0.5PbI3"
        a_str = "".join([f"{c.precursor.name}{c.ratio:.2g}" for c in self.a_site])
        b_str = "".join([f"{c.precursor.name}{c.ratio:.2g}" for c in self.b_site])
        x_str = "".join([f"{c.precursor.name}{c.ratio:.2g}" for c in self.x_site])
        return f"{a_str}{b_str}{x_str}"

@dataclass
class DecisionMetrics:
    tolerance_factor: float = 0.0
    octahedral_factor: float = 0.0
    solvent_boiling_point_avg: float = 0.0
    solvent_dn: float = 0.0
    solvent_an: float = 0.0
    solvent_dielectric: float = 0.0
    antisolvent_miscibility_gap: Optional[float] = None  # Delta delta
    estimated_drying_time: float = 0.0
    film_thickness_metric: float = 0.0  # C * V
    film_uniformity_metric: Optional[float] = None  # U ∝ 1/(C·E·η)
    precursor_solvent_delta: Optional[float] = None  # |δ_precursor - δ_solvent|
    a_site_lattice_spacing: Optional[float] = None  # Angstrom (weighted spacer spacing)
    a_site_dipole_moment: Optional[float] = None  # Debye (weighted A-site dipole)
    a_site_spacer_length: Optional[float] = None  # Angstrom (weighted spacer length)
    
    # ML Metrics (for Active Learning)
    predicted_performance: Optional[float] = None
    uncertainty_score: Optional[float] = None  # Standard Deviation
    acquisition_score: Optional[float] = None  # UCB or similar
    predicted_peak_nm: Optional[float] = None  # Predicted PL peak position (nm)
    peak_uncertainty_nm: Optional[float] = None  # Predicted peak uncertainty (std, nm)
    peak_acquisition_score: Optional[float] = None  # Peak-target acquisition score
    composition_predicted_quality: Optional[float] = None  # Composition-space GP predicted quality
    composition_uncertainty_score: Optional[float] = None  # Composition-space GP uncertainty
    composition_acquisition_score: Optional[float] = None  # Composition-space combined acquisition
    composition_predicted_peak_nm: Optional[float] = None  # Composition-space GP predicted peak (nm)
    composition_peak_uncertainty_nm: Optional[float] = None  # Composition-space GP peak uncertainty (nm)
    composition_peak_acquisition_score: Optional[float] = None  # Composition-space peak-target acquisition
    tree_predicted_quality: Optional[float] = None  # Decision-tree / RF predicted quality

@dataclass
class ExperimentParams:
    solvent: List[SolventComponent] 
    material_system: MaterialSystem
    concentration: float  # M (mol/L)
    
    # Recipe Parameters (NO Defaults here, must come before defaults)
    drop_volume: float  # uL (integer)
    spin_speed: int  # rpm (integer, main/second step speed)
    spin_time: int  # s (integer, main/second step time)
    annealing_temp: int  # Celsius (integer)
    annealing_time: int  # s (integer)
    
    # Recipe Parameters WITH Defaults (Must come last)
    spin_acceleration: int = 2000  # rpm/s (integer)
    dispense_height: int = 15  # mm (integer)
    
    # Two-step spin coating parameters (first step, optional)
    two_step_enabled: bool = False
    first_spin_speed: int = 0  # rpm (integer, first step - can be < 3000)
    first_spin_time: int = 0  # s (integer)
    first_spin_acceleration: int = 0  # rpm/s (integer)

    # Additive concentrations for composition-space optimization (optional)
    # Units: M (mol/L). These are independent “composition knobs” guided by Dual-GP.
    csbr_M: Optional[float] = None
    neabr_M: Optional[float] = None
    ce_M: Optional[float] = None
    
    antisolvent: Optional[Solvent] = None
    antisolvent_volume: float = 0.0
    antisolvent_timing: float = 0.0 # seconds after spin start
    
    # Metrics (with Default Factory)
    metrics: DecisionMetrics = field(default_factory=DecisionMetrics)
    
    # Rank Score (with Default)
    rank_score: float = 0.0
    
    @property
    def solvent_name(self) -> str:
        # Helper to display mixed solvent name
        return "+".join([f"{s.solvent.name}({s.ratio:.2g})" for s in self.solvent])
