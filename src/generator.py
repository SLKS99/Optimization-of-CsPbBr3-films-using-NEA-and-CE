import random
import math
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from src.models import Solvent, Precursor, MaterialSystem, ExperimentParams, MaterialComponent, SolventComponent, DecisionMetrics
from src import constraints
from src import physics

# ---------------------------------------------------------------------------
# Default values (used when config is not provided)
# ---------------------------------------------------------------------------
DEFAULT_DROP_VOLUME_UL = 50  # Suitable for ~10x10 mm substrates (integer)

# Default spin coating parameters (all integers)
DEFAULT_SPIN_COATING = {
    'two_step_enabled': True,
    'first_step': {
        'speed_rpm': [500, 1000, 1500, 2000],
        'time_s': [5, 10, 15, 20],
        'acceleration_rpm_s': [500, 1000, 1500]
    },
    'second_step': {
        'speed_rpm': [3000, 3500, 4000, 4500, 5000, 5500, 6000],
        'time_s': [30, 45, 60, 90],
        'acceleration_rpm_s': [1000, 1500, 2000, 2500, 3000]
    },
    'drop_volume_ul': [40, 50, 60, 70, 80],
    'dispense_height_mm': [10, 12, 15, 18, 20]
}

# Default annealing parameters (temp >= 95°C, time > 5 min)
DEFAULT_ANNEALING = {
    'temperature_c': [95, 100, 110, 120, 130, 140, 150],
    'time_s': [360, 480, 600, 720, 900, 1200]
}

# Default concentration options
DEFAULT_CONCENTRATION = {
    'values_M': [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
}

# Default antisolvent parameters
DEFAULT_ANTISOLVENT = {
    'use_probability': 0.7,
    'volume_ul': [50, 75, 100, 125, 150],
    'timing_before_end_s': [2, 4, 6, 8, 10, 12],
    # Alternative mode: specify the antisolvent drop time explicitly and derive total spin time.
    # If present, we will set spin_time = drop_time_s + 20 (and set antisolvent_timing accordingly).
    'drop_time_s': None
}


class MonteCarloGenerator:
    def __init__(
        self,
        solvents_df: pd.DataFrame,
        precursors_df: pd.DataFrame,
        a_site_recipes: Optional[List[List[Tuple[str, float]]]] = None,
        solvent_config: Optional[List[dict]] = None,
        b_site_config: Optional[List[dict]] = None,
        x_site_config: Optional[List[dict]] = None,
        stoichiometry_config: Optional[dict] = None,
        composition_additives_config: Optional[dict] = None,
        fixed_material_system_config: Optional[dict] = None,
        spin_coating_config: Optional[dict] = None,
        annealing_config: Optional[dict] = None,
        concentration_config: Optional[dict] = None,
        antisolvent_config: Optional[dict] = None
    ):
        self.solvents = self._parse_solvents(solvents_df)
        self.precursors_a = self._parse_precursors_dict(precursors_df, 'A')
        self.precursors_b = self._parse_precursors_dict(precursors_df, 'B')
        self.precursors_x = self._parse_precursors_dict(precursors_df, 'X')

        # Create a dictionary for easier solvent lookup by name
        self.solvent_dict = {s.name: s for s in self.solvents}

        # Store config-driven settings
        self.a_site_recipes = a_site_recipes
        self.solvent_config = solvent_config
        self.b_site_config = b_site_config
        self.x_site_config = x_site_config
        self.stoichiometry_config = stoichiometry_config or {}
        self.composition_additives_config = composition_additives_config or {}
        self.fixed_material_system_config = fixed_material_system_config or {}
        
        # Process parameter configs (use defaults if not provided)
        self.spin_coating_config = spin_coating_config or DEFAULT_SPIN_COATING
        self.annealing_config = annealing_config or DEFAULT_ANNEALING
        self.concentration_config = concentration_config or DEFAULT_CONCENTRATION
        self.antisolvent_config = antisolvent_config or DEFAULT_ANTISOLVENT

        # Build lookup: A-precursor name -> phase type (e.g., "3D", "2D")
        self.a_phase_map: Dict[str, str] = {}
        for phase_name, phase_cfg in self.stoichiometry_config.items():
            for prec_name in phase_cfg.get('a_precursors', []):
                self.a_phase_map[prec_name] = phase_name

    def _parse_solvents(self, df: pd.DataFrame) -> List[Solvent]:
        solvents = []
        for _, row in df.iterrows():
            s = Solvent(
                name=row['Name'],
                hsp_d=row['HSP_d'],
                hsp_p=row['HSP_p'],
                hsp_h=row['HSP_h'],
                boiling_point=row['Boiling_Point_C'],
                viscosity=row['Viscosity_mPas'],
                vapor_pressure=row['Vapor_Pressure_kPa'],
                dn=row['DN_kcal_mol'],
                an=row['AN_kcal_mol'],
                dielectric_constant=row['Dielectric_Constant']
            )
            solvents.append(s)
        return solvents

    def _parse_precursors_dict(self, df: pd.DataFrame, p_type: str) -> Dict[str, Precursor]:
        precursors = {}
        filtered_df = df[df['Type'] == p_type]
        for _, row in filtered_df.iterrows():
            lattice_spacing = row.get('Lattice_Spacing_A') if 'Lattice_Spacing_A' in row else None
            if pd.isna(lattice_spacing):
                lattice_spacing = None
            dipole_moment = row.get('Dipole_Moment_D') if 'Dipole_Moment_D' in row else None
            if pd.isna(dipole_moment):
                dipole_moment = None
            spacer_length = row.get('Spacer_Length_A') if 'Spacer_Length_A' in row else None
            if pd.isna(spacer_length):
                spacer_length = None
            p = Precursor(
                name=row['Name'],
                type=row['Type'],
                ionic_radius=row['Ionic_Radius_A'],
                molecular_weight=row['Molecular_Weight_g_mol'],
                solubility_parameter=row['Solubility_Parameter'],
                lattice_spacing=lattice_spacing,
                dipole_moment=dipole_moment,
                spacer_length=spacer_length
            )
            precursors[p.name] = p
        return precursors

    def _get_precursor(self, name: str, p_dict: Dict[str, Precursor]) -> Precursor:
        if name in p_dict:
            return p_dict[name]
        for key in p_dict:
            if key in name:
                return p_dict[key]
        raise ValueError(f"Precursor {name} not found in database.")

    def _get_solvent(self, name: str) -> Solvent:
        if name in self.solvent_dict:
            return self.solvent_dict[name]
        raise ValueError(f"Solvent {name} not found.")

    def _get_bx_for_a_recipe(
        self, a_components: List[MaterialComponent]
    ) -> Tuple[List[MaterialComponent], List[MaterialComponent]]:
        """
        Determine B-site and X-site components based on the A-site composition.
        Uses stoichiometry rules from config; computes weighted average X ratio
        for mixed 3D/2D compositions.
        """
        # Compute weighted X ratio based on A-site phases
        total_ratio = sum(c.ratio for c in a_components)
        if total_ratio == 0:
            total_ratio = 1.0

        x_ratio_weighted = 0.0
        for comp in a_components:
            phase = self.a_phase_map.get(comp.precursor.name, '3D')
            phase_cfg = self.stoichiometry_config.get(phase, {})
            # Default X ratio: 3 for 3D, 4 for 2D
            x_entries = phase_cfg.get('x_site', [{'name': 'I', 'ratio': 3.0}])
            x_ratio = sum(e.get('ratio', 1.0) for e in x_entries)
            x_ratio_weighted += x_ratio * (comp.ratio / total_ratio)

        # B-site: use first phase's config or default Pb=1
        first_phase = self.a_phase_map.get(a_components[0].precursor.name, '3D')
        phase_cfg = self.stoichiometry_config.get(first_phase, {})
        b_entries = phase_cfg.get('b_site', [{'name': 'Pb', 'ratio': 1.0}])
        x_entries = phase_cfg.get('x_site', [{'name': 'I', 'ratio': 3.0}])

        b_components = []
        for entry in b_entries:
            prec = self._get_precursor(entry['name'], self.precursors_b)
            b_components.append(MaterialComponent(prec, entry.get('ratio', 1.0)))

        # X-site: use weighted ratio
        x_components = []
        for entry in x_entries:
            prec = self._get_precursor(entry['name'], self.precursors_x)
            # Scale ratio proportionally if multiple X entries
            base_ratio = entry.get('ratio', 1.0)
            # For simplicity, use weighted total and distribute
            x_components.append(MaterialComponent(prec, x_ratio_weighted))
            break  # Only one X entry for now; extend if needed

        return b_components, x_components

    def _build_solvent_mix(self) -> List[SolventComponent]:
        """Build solvent mix from config or use default DMF:DMSO 9:1."""
        if self.solvent_config:
            mix = []
            for entry in self.solvent_config:
                solv = self._get_solvent(entry['name'])
                mix.append(SolventComponent(solv, entry.get('ratio', 1.0)))
            return mix
        else:
            # Default fallback
            try:
                dmf = self._get_solvent("DMF")
                dmso = self._get_solvent("DMSO")
                return [SolventComponent(dmf, 0.9), SolventComponent(dmso, 0.1)]
            except ValueError:
                return [SolventComponent(random.choice(self.solvents), 1.0)]

    def _default_a_site_recipes(self) -> List[List[Tuple[str, float]]]:
        """Fallback recipes if none provided via config."""
        return [
            [("Cs", 1.0)],
            [("Cs", 38/50), ("AVA", 4/50), ("BDA", 8/50)],
            [("Cs", 34/50), ("AVA", 16/50)],
            [("Cs", 30/50), ("AVA", 20/50)],
            [("Cs", 10/50), ("AVA", 40/50)],
            [("Cs", 14/50), ("AVA", 36/50)],
            [("Cs", 14/50), ("AVA", 32/50), ("BDA", 4/50)],
            [("Cs", 10/50), ("AVA", 4/50), ("BDA", 36/50)],
            [("Cs", 10/50), ("AVA", 8/50), ("BDA", 32/50)],
        ]

    def generate_random_experiment(self) -> ExperimentParams:
        # 1. Build material system
        # If a fixed material system is provided (e.g., CsPbBr3), use it.
        if self.fixed_material_system_config.get("enabled", False):
            a_name = self.fixed_material_system_config.get("a_name", "Cs")
            b_name = self.fixed_material_system_config.get("b_name", "Pb")
            x_name = self.fixed_material_system_config.get("x_name", "Br")
            x_ratio = float(self.fixed_material_system_config.get("x_ratio", 3.0))

            a_components = [MaterialComponent(self._get_precursor(a_name, self.precursors_a), 1.0)]
            b_components = [MaterialComponent(self._get_precursor(b_name, self.precursors_b), 1.0)]
            x_components = [MaterialComponent(self._get_precursor(x_name, self.precursors_x), x_ratio)]
            system = MaterialSystem(a_components, b_components, x_components)
        else:
            # Fallback: Select A-site recipe (legacy)
            recipes = self.a_site_recipes if self.a_site_recipes else self._default_a_site_recipes()
            selected_recipe = random.choice(recipes)

            a_components = []
            for name, ratio in selected_recipe:
                prec = self._get_precursor(name, self.precursors_a)
                a_components.append(MaterialComponent(prec, ratio))

            # Determine B/X stoichiometry based on A-site
            b_components, x_components = self._get_bx_for_a_recipe(a_components)
            system = MaterialSystem(a_components, b_components, x_components)

        # 3. Build solvent mix
        solvent_mix = self._build_solvent_mix()

        # 4. Select Processing Parameters from config (all integers)
        spin_cfg = self.spin_coating_config
        anneal_cfg = self.annealing_config
        conc_cfg = self.concentration_config
        anti_cfg = self.antisolvent_config
        
        # Concentration
        concentration = random.choice(conc_cfg.get('values_M', [0.3]))
        
        # Drop volume (integer)
        drop_volume = int(random.choice(spin_cfg.get('drop_volume_ul', [DEFAULT_DROP_VOLUME_UL])))
        
        # Dispense height (integer)
        dispense_height = int(random.choice(spin_cfg.get('dispense_height_mm', [15])))
        
        # Two-step spin coating logic
        two_step_enabled = spin_cfg.get('two_step_enabled', False)
        first_step_cfg = spin_cfg.get('first_step', DEFAULT_SPIN_COATING['first_step'])
        second_step_cfg = spin_cfg.get('second_step', DEFAULT_SPIN_COATING['second_step'])
        
        # Initialize first step values (all integers)
        first_spin_speed = 0
        first_spin_time = 0
        first_spin_acceleration = 0
        
        if two_step_enabled:
            # Two-step process: first step can be < 3000 rpm
            first_spin_speed = int(random.choice(first_step_cfg.get('speed_rpm', [1000])))
            first_spin_time = int(random.choice(first_step_cfg.get('time_s', [10])))
            first_spin_acceleration = int(random.choice(first_step_cfg.get('acceleration_rpm_s', [1000])))
        
        # Main/second step spin parameters (all integers)
        # For single-step: must be >= 3000 rpm (enforced by config)
        # For two-step: this is the second (high speed) step
        spin_speed = int(random.choice(second_step_cfg.get('speed_rpm', [4000])))
        spin_time = int(random.choice(second_step_cfg.get('time_s', [45])))
        spin_acceleration = int(random.choice(second_step_cfg.get('acceleration_rpm_s', [2000])))
        
        # Annealing parameters (integers, temp >= 95°C, time > 5 min enforced by config)
        anneal_temp = int(random.choice(anneal_cfg.get('temperature_c', [100])))
        anneal_time = int(random.choice(anneal_cfg.get('time_s', [600])))

        # 5. Select Antisolvent (Optional)
        use_probability = anti_cfg.get('use_probability', 0.7)
        use_antisolvent = random.random() < use_probability
        antisolvent = None
        antisolvent_volume = 0.0
        antisolvent_timing = 0.0
        if use_antisolvent:
            # Exclude primary solvents from antisolvent candidates
            primary_names = {sc.solvent.name for sc in solvent_mix}

            # If allowed_names is provided, restrict to that list (with simple alias handling).
            allowed_names = anti_cfg.get('allowed_names')
            if allowed_names:
                alias = {
                    "Chlorobenzene": "CB",
                    "chlorobenzene": "CB",
                }
                allowed_norm = {alias.get(str(n).strip(), str(n).strip()) for n in allowed_names}
                candidates = [s for s in self.solvents if s.name in allowed_norm and s.name not in primary_names]
            else:
                candidates = [s for s in self.solvents if s.name not in primary_names]

            if candidates:
                antisolvent = random.choice(candidates)
                antisolvent_volume = float(random.choice(anti_cfg.get('volume_ul', [100])))

                drop_time_list = anti_cfg.get('drop_time_s')
                if drop_time_list:
                    # New rule: total spin coating time = drop_time + 20s
                    drop_time = float(random.choice(drop_time_list))
                    spin_time = int(round(drop_time + 20.0))

                    # Antisolvent timing is the elapsed time at which we drop antisolvent.
                    # If two-step is enabled, assume the drop happens during the second step.
                    antisolvent_timing = float((first_spin_time if two_step_enabled else 0) + drop_time)
                else:
                    timing_before_end = random.choice(anti_cfg.get('timing_before_end_s', [6]))
                    # Calculate total spin time for antisolvent timing
                    total_spin_time = (first_spin_time if two_step_enabled else 0) + spin_time
                    antisolvent_timing = max(0.0, float(total_spin_time - timing_before_end))

        # Optional additive composition knobs (CsBr/NEABr/CE) - independent concentration space
        csbr_M = None
        neabr_M = None
        ce_M = None
        if self.composition_additives_config.get('enabled', False):
            cs_min = float(self.composition_additives_config.get('csbr_min', 0.10))
            cs_max = float(self.composition_additives_config.get('csbr_max', 0.18))
            nea_min = float(self.composition_additives_config.get('neabr_min', 0.0))
            nea_max = float(self.composition_additives_config.get('neabr_max', 0.4))
            ce_min = float(self.composition_additives_config.get('ce_min', 0.0))
            ce_max = float(self.composition_additives_config.get('ce_max', 0.02))
            step = float(self.composition_additives_config.get('grid_resolution', 0.01))

            def _grid_choice(lo: float, hi: float) -> float:
                if hi < lo:
                    lo, hi = hi, lo
                if step <= 0:
                    return random.uniform(lo, hi)
                grid = [round(x, 6) for x in np.arange(lo, hi + step / 2.0, step)]
                return float(random.choice(grid)) if grid else float(lo)

            csbr_M = _grid_choice(cs_min, cs_max)
            neabr_M = _grid_choice(nea_min, nea_max)
            ce_M = _grid_choice(ce_min, ce_max)

        return ExperimentParams(
            solvent=solvent_mix,
            material_system=system,
            concentration=round(concentration, 2),
            csbr_M=csbr_M,
            neabr_M=neabr_M,
            ce_M=ce_M,
            drop_volume=drop_volume,
            spin_speed=spin_speed,
            spin_time=spin_time,
            annealing_temp=anneal_temp,
            annealing_time=anneal_time,
            spin_acceleration=spin_acceleration,
            dispense_height=dispense_height,
            two_step_enabled=two_step_enabled,
            first_spin_speed=first_spin_speed,
            first_spin_time=first_spin_time,
            first_spin_acceleration=first_spin_acceleration,
            antisolvent=antisolvent,
            antisolvent_volume=antisolvent_volume,
            antisolvent_timing=antisolvent_timing
        )

    def calculate_metrics(self, exp: ExperimentParams) -> DecisionMetrics:
        """Calculates all key physics metrics for the experiment."""
        t_factor = physics.calculate_tolerance_factor(exp.material_system)
        mu_factor = physics.calculate_octahedral_factor(exp.material_system)

        eff_solvent = physics.get_effective_solvent_properties(exp.solvent)
        bp_avg = eff_solvent.boiling_point
        dn = eff_solvent.dn
        an = eff_solvent.an
        eps = eff_solvent.dielectric_constant
        visc = eff_solvent.viscosity
        vp = eff_solvent.vapor_pressure

        miscibility = None
        if exp.antisolvent:
            miscibility = physics.check_antisolvent_miscibility(exp.solvent, exp.antisolvent)

        drying_time = physics.estimate_drying_time(exp)
        thickness_metric = physics.estimate_film_thickness(exp)

        # Film uniformity metric: U ∝ 1/(C·E·η)
        # E = evaporation rate ∝ vapor_pressure
        # Higher U = more uniform film
        film_uniformity = None
        if exp.concentration > 0 and vp > 0 and visc > 0:
            film_uniformity = 1.0 / (exp.concentration * vp * visc)

        # Worst-case precursor-solvent δ distance among known values
        deltas = []
        a_spacing_weighted = None
        a_dipole_weighted = None
        a_length_weighted = None
        spacing_weight_sum = 0.0
        spacing_sum = 0.0
        dipole_weight_sum = 0.0
        dipole_sum = 0.0
        length_weight_sum = 0.0
        length_sum = 0.0

        for comp in (exp.material_system.a_site + exp.material_system.b_site + exp.material_system.x_site):
            delta = physics.calculate_precursor_solvent_delta(comp.precursor.solubility_parameter, eff_solvent)
            if delta is not None:
                deltas.append(delta)
            if comp.precursor.lattice_spacing is not None and not pd.isna(comp.precursor.lattice_spacing):
                spacing_sum += comp.precursor.lattice_spacing * comp.ratio
                spacing_weight_sum += comp.ratio

        # A-site dipole moment (weighted average)
        for comp in exp.material_system.a_site:
            if comp.precursor.dipole_moment is not None and not pd.isna(comp.precursor.dipole_moment):
                dipole_sum += comp.precursor.dipole_moment * comp.ratio
                dipole_weight_sum += comp.ratio
            if comp.precursor.spacer_length is not None and not pd.isna(comp.precursor.spacer_length):
                length_sum += comp.precursor.spacer_length * comp.ratio
                length_weight_sum += comp.ratio

        precursor_delta = max(deltas) if deltas else None
        if spacing_weight_sum > 0:
            a_spacing_weighted = spacing_sum / spacing_weight_sum
        if dipole_weight_sum > 0:
            a_dipole_weighted = dipole_sum / dipole_weight_sum
        if length_weight_sum > 0:
            a_length_weighted = length_sum / length_weight_sum

        return DecisionMetrics(
            tolerance_factor=round(t_factor, 3),
            octahedral_factor=round(mu_factor, 3),
            solvent_boiling_point_avg=round(bp_avg, 1),
            solvent_dn=round(dn, 2),
            solvent_an=round(an, 2),
            solvent_dielectric=round(eps, 2),
            antisolvent_miscibility_gap=round(miscibility, 2) if miscibility else None,
            estimated_drying_time=round(drying_time, 5),
            film_thickness_metric=round(thickness_metric, 1),
            film_uniformity_metric=round(film_uniformity, 4) if film_uniformity is not None else None,
            precursor_solvent_delta=round(precursor_delta, 2) if precursor_delta is not None else None,
            a_site_lattice_spacing=round(a_spacing_weighted, 2) if a_spacing_weighted is not None else None,
            a_site_dipole_moment=round(a_dipole_weighted, 2) if a_dipole_weighted is not None else None,
            a_site_spacer_length=round(a_length_weighted, 2) if a_length_weighted is not None else None
        )

    def generate_candidates(self, n_attempts: int = 1000) -> List[ExperimentParams]:
        valid_experiments = []

        for _ in range(n_attempts):
            try:
                exp = self.generate_random_experiment()
            except ValueError:
                continue

            # Check Structural Constraints
            if not constraints.check_structural_feasibility(exp.material_system):
                continue

            # Check Solvent Compatibility
            all_precursors = exp.material_system.a_site + exp.material_system.b_site + exp.material_system.x_site
            if not constraints.check_solvent_compatibility(all_precursors, exp.solvent):
                continue

            # Check Process Constraints
            if not constraints.check_process_feasibility(exp):
                continue

            # Check Antisolvent Constraints
            if exp.antisolvent:
                if not constraints.check_antisolvent_selection(exp.solvent, exp.antisolvent):
                    continue

            # Calculate and attach metrics
            exp.metrics = self.calculate_metrics(exp)

            # Guard drying time window (relative metric, not actual seconds)
            # Typical range for DMF/DMSO mixes: 0.001 - 0.1
            # Lower = faster drying, higher = slower
            if not (0.001 <= exp.metrics.estimated_drying_time <= 0.5):
                continue

            valid_experiments.append(exp)

        return valid_experiments

    def generate_gp_guided_candidates(
        self, 
        learner, 
        n_attempts: int = 1000, 
        exploration_rate: float = 0.15,
        n_initial_random: int = 500
    ) -> List[ExperimentParams]:
        """
        Generate candidates using GP-guided sampling.
        """
        import numpy as np
        
        if not learner.is_trained or learner.gp_model is None:
            return self.generate_candidates(n_attempts)
        
        valid_experiments = []
        n_explore = int(n_attempts * exploration_rate)
        n_exploit = n_attempts - n_explore
        
        print(f"  Generating {n_initial_random} initial random candidates for GP evaluation...")
        initial_candidates = []
        attempts = 0
        while len(initial_candidates) < n_initial_random and attempts < n_initial_random * 3:
            attempts += 1
            try:
                exp = self.generate_random_experiment()
            except ValueError:
                continue
            
            if not constraints.check_structural_feasibility(exp.material_system):
                continue
            all_precursors = exp.material_system.a_site + exp.material_system.b_site + exp.material_system.x_site
            if not constraints.check_solvent_compatibility(all_precursors, exp.solvent):
                continue
            if not constraints.check_process_feasibility(exp):
                continue
            if exp.antisolvent and not constraints.check_antisolvent_selection(exp.solvent, exp.antisolvent):
                continue
            
            exp.metrics = self.calculate_metrics(exp)
            if not (0.001 <= exp.metrics.estimated_drying_time <= 0.5):
                continue
            
            initial_candidates.append(exp)
        
        print(f"  Scoring {len(initial_candidates)} candidates with GP...")
        learner.score_candidates_with_uncertainty(initial_candidates)
        
        scored_candidates = [c for c in initial_candidates 
                           if c.metrics.predicted_performance is not None 
                           and c.metrics.uncertainty_score is not None]
        
        if not scored_candidates:
            return self.generate_candidates(n_attempts)
        
        pred_vals = np.array([c.metrics.predicted_performance for c in scored_candidates])
        uncert_vals = np.array([c.metrics.uncertainty_score for c in scored_candidates])
        
        pred_norm = (pred_vals - pred_vals.min()) / (pred_vals.max() - pred_vals.min() + 1e-10)
        uncert_norm = (uncert_vals - uncert_vals.min()) / (uncert_vals.max() - uncert_vals.min() + 1e-10)
        
        weights = 0.5 * pred_norm + 0.5 * uncert_norm
        weights = weights / (weights.sum() + 1e-10)
        
        print(f"  Sampling {n_exploit} candidates from promising regions...")
        promising_candidates = []
        attempts = 0
        max_attempts = n_exploit * 5
        
        indices = np.random.choice(len(scored_candidates), size=min(n_exploit, len(scored_candidates)), p=weights, replace=True)
        
        for idx in indices:
            if len(promising_candidates) >= n_exploit or attempts >= max_attempts:
                break
            attempts += 1
            try:
                exp = self.generate_random_experiment()
                if not constraints.check_structural_feasibility(exp.material_system):
                    continue
                all_precursors = exp.material_system.a_site + exp.material_system.b_site + exp.material_system.x_site
                if not constraints.check_solvent_compatibility(all_precursors, exp.solvent):
                    continue
                if not constraints.check_process_feasibility(exp):
                    continue
                if exp.antisolvent and not constraints.check_antisolvent_selection(exp.solvent, exp.antisolvent):
                    continue
                exp.metrics = self.calculate_metrics(exp)
                if not (0.001 <= exp.metrics.estimated_drying_time <= 0.5):
                    continue
                promising_candidates.append(exp)
            except ValueError:
                continue
        
        print(f"  Adding {n_explore} random exploration candidates...")
        explore_candidates = []
        attempts = 0
        while len(explore_candidates) < n_explore and attempts < n_explore * 3:
            attempts += 1
            try:
                exp = self.generate_random_experiment()
            except ValueError:
                continue
            
            if not constraints.check_structural_feasibility(exp.material_system):
                continue
            all_precursors = exp.material_system.a_site + exp.material_system.b_site + exp.material_system.x_site
            if not constraints.check_solvent_compatibility(all_precursors, exp.solvent):
                continue
            if not constraints.check_process_feasibility(exp):
                continue
            if exp.antisolvent and not constraints.check_antisolvent_selection(exp.solvent, exp.antisolvent):
                continue
            
            exp.metrics = self.calculate_metrics(exp)
            if not (0.001 <= exp.metrics.estimated_drying_time <= 0.5):
                continue
            
            explore_candidates.append(exp)
        
        valid_experiments = promising_candidates + explore_candidates
        print(f"  Generated {len(valid_experiments)} GP-guided candidates ({len(promising_candidates)} from promising regions, {len(explore_candidates)} random)")
        
        return valid_experiments

