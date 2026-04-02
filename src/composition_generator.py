from typing import List, Tuple, Optional, Callable
import random

from src.generator import MonteCarloGenerator
from src.models import ExperimentParams, MaterialComponent, MaterialSystem


class CompositionAwareGenerator(MonteCarloGenerator):
    """
    Thin wrapper around MonteCarloGenerator that biases A-site composition
    sampling using a composition-level scorer (e.g., a composition-only GP
    trained on FA/FuDMA binaries).

    The scorer should accept an FA fraction in [0, 1] and return a scalar score.
    Higher scores are sampled more frequently.
    """

    def __init__(
        self,
        *args,
        composition_scorer: Optional[Callable[[float], float]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.composition_scorer = composition_scorer
        self._recipe_weights_cache = None
        self._cached_recipes_id = None

    @staticmethod
    def _estimate_fa_ratio_from_recipe(recipe: List[Tuple[str, float]]) -> float:
        """
        Estimate FA fraction (0-1) from an A-site recipe consisting of (name, ratio) pairs.
        Assumes a binary FA/FuDMA system but falls back gracefully if not present.
        """
        if not recipe:
            return 0.5

        fa_total = 0.0
        fudma_total = 0.0

        for name, ratio in recipe:
            name_upper = str(name).upper()
            if "FA" in name_upper and "FUDMA" not in name_upper:
                fa_total += float(ratio)
            elif "FUDMA" in name_upper:
                fudma_total += float(ratio)

        total = fa_total + fudma_total
        if total <= 0:
            # If we can't infer FA vs FuDMA, default to mid-point
            return 0.5
        return fa_total / total

    def _choose_recipe(self, recipes: List[List[Tuple[str, float]]]) -> List[Tuple[str, float]]:
        """
        Choose an A-site recipe. If a composition_scorer is provided and recipes
        look like FA/FuDMA binaries, bias the sampling according to the scorer.
        Uses caching to avoid expensive GP calls for every candidate.
        """
        if not recipes:
            return []

        if self.composition_scorer is None:
            return random.choice(recipes)

        # Check if we can use cached weights
        recipes_id = id(recipes)
        if self._recipe_weights_cache is not None and self._cached_recipes_id == recipes_id:
            return random.choices(recipes, weights=self._recipe_weights_cache, k=1)[0]

        print(f"  Calculating composition weights for {len(recipes)} recipes...")
        # Compute FA ratio for each recipe
        fa_ratios = [self._estimate_fa_ratio_from_recipe(r) for r in recipes]
        
        # Batch score all ratios at once for performance
        if hasattr(self.composition_scorer, "__self__") and hasattr(self.composition_scorer.__self__, "score_fa_batch"):
            # If the scorer is the CompositionGP.score_fa_ratio method, use its batch counterpart
            raw_scores = self.composition_scorer.__self__.score_fa_batch(fa_ratios)
        else:
            # Fallback to individual calls if it's a simple lambda or other function
            raw_scores = []
            for fa in fa_ratios:
                try:
                    s = float(self.composition_scorer(fa))
                except Exception:
                    s = 0.0
                raw_scores.append(s)

        # Convert raw scores to positive weights; fall back to uniform if degenerate
        max_score = max(raw_scores)
        min_score = min(raw_scores)
        span = max_score - min_score
        
        if span <= 1e-8:
            # All scores identical or invalid
            self._recipe_weights_cache = [1.0] * len(recipes)
        else:
            self._recipe_weights_cache = [(s - min_score) + 1e-3 for s in raw_scores]
        
        self._cached_recipes_id = recipes_id
        
        # Use weighted random choice
        return random.choices(recipes, weights=self._recipe_weights_cache, k=1)[0]

    @property
    def _resolved_recipes(self) -> List[List[Tuple[str, float]]]:
        """Get the set of A-site recipes once, avoiding repeated default calls."""
        if not hasattr(self, "_cached_resolved_recipes"):
            self._cached_resolved_recipes = self.a_site_recipes if self.a_site_recipes else self._default_a_site_recipes()
        return self._cached_resolved_recipes

    def generate_random_experiment(self) -> ExperimentParams:
        """
        Override the base generator to bias the A-site recipe selection, then
        delegate the rest of the recipe construction logic.
        """
        # 1. Select A-site recipe (optionally biased by composition scorer)
        recipes = self._resolved_recipes
        selected_recipe = self._choose_recipe(recipes)

        if not selected_recipe:
            raise ValueError("No recipe selected")

        a_components = []
        for name, ratio in selected_recipe:
            prec = self._get_precursor(name, self.precursors_a)
            a_components.append(MaterialComponent(prec, ratio))

        # 2. Determine B/X stoichiometry based on A-site
        b_components, x_components = self._get_bx_for_a_recipe(a_components)

        system = MaterialSystem(a_components, b_components, x_components)

        # 3. Build solvent mix
        solvent_mix = self._build_solvent_mix()

        # 4. Reuse the remaining logic via a helper on the base class.
        #    We reconstruct an ExperimentParams using the same process-parameter
        #    sampling as the original implementation.
        spin_cfg = self.spin_coating_config
        anneal_cfg = self.annealing_config
        conc_cfg = self.concentration_config
        anti_cfg = self.antisolvent_config

        # Concentration
        concentration = random.choice(conc_cfg.get("values_M", [0.3]))

        # Drop volume (integer)
        drop_volume = int(random.choice(spin_cfg.get("drop_volume_ul", [50])))

        # Dispense height (integer)
        dispense_height = int(random.choice(spin_cfg.get("dispense_height_mm", [15])))

        # Two-step spin coating logic
        two_step_enabled = spin_cfg.get("two_step_enabled", False)
        first_step_cfg = spin_cfg.get("first_step")
        second_step_cfg = spin_cfg.get("second_step")

        # Initialize first step values (all integers)
        first_spin_speed = 0
        first_spin_time = 0
        first_spin_acceleration = 0

        if two_step_enabled and first_step_cfg:
            first_spin_speed = int(random.choice(first_step_cfg.get("speed_rpm", [1000])))
            first_spin_time = int(random.choice(first_step_cfg.get("time_s", [10])))
            first_spin_acceleration = int(random.choice(first_step_cfg.get("acceleration_rpm_s", [1000])))

        # Main/second step spin parameters (all integers)
        second_step_cfg = second_step_cfg or {}
        spin_speed = int(random.choice(second_step_cfg.get("speed_rpm", [4000])))
        spin_time = int(random.choice(second_step_cfg.get("time_s", [45])))
        spin_acceleration = int(random.choice(second_step_cfg.get("acceleration_rpm_s", [2000])))

        # Annealing parameters
        anneal_temp = int(random.choice(anneal_cfg.get("temperature_c", [100])))
        anneal_time = int(random.choice(anneal_cfg.get("time_s", [600])))

        # 5. Antisolvent (optional)
        use_probability = anti_cfg.get("use_probability", 0.7)
        use_antisolvent = random.random() < use_probability
        antisolvent = None
        antisolvent_volume = 0.0
        antisolvent_timing = 0.0
        if use_antisolvent:
            primary_names = {sc.solvent.name for sc in solvent_mix}
            candidates = [s for s in self.solvents if s.name not in primary_names]
            if candidates:
                antisolvent = random.choice(candidates)
                antisolvent_volume = float(random.choice(anti_cfg.get("volume_ul", [100])))
                timing_before_end = random.choice(anti_cfg.get("timing_before_end_s", [6]))
                total_spin_time = (first_spin_time if two_step_enabled else 0) + spin_time
                antisolvent_timing = max(0.0, float(total_spin_time - timing_before_end))

        return ExperimentParams(
            solvent=solvent_mix,
            material_system=system,
            concentration=round(concentration, 2),
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
            antisolvent_timing=antisolvent_timing,
        )

