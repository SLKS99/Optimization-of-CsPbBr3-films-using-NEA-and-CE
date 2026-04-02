from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.models import ExperimentParams
from src.learner import extract_fa_ratio_from_composition, ActiveLearner


class ProcessTreeLearner:
    """
    True Monte Carlo decision-tree style learner:
    - Trains on completed experiments
    - Uses simple, robust features (composition + processing)
    - Predicts quality for new candidates inside the Monte Carlo loop
    """

    def __init__(self, n_estimators: int = 200, random_state: int = 0):
        self.model: Optional[RandomForestRegressor] = None
        self.is_trained: bool = False
        self.n_estimators = n_estimators
        self.random_state = random_state

    @staticmethod
    def _featurize_log_row(row: pd.Series) -> Optional[np.ndarray]:
        """
        Features from history row:
        - FA ratio (from composition)
        - Anneal temperature
        - Spin speed
        - Concentration
        - Antisolvent is Chloroform (0/1)
        - Antisolvent is Chlorobenzene (0/1)
        - No Antisolvent (0/1)
        """
        comp = row.get("Material_Composition", "")
        temp = row.get("Anneal_Temp_C", row.get("Temp_C", None))
        spin = row.get("Spin_Speed_rpm", None)

        if pd.isna(comp) or pd.isna(temp) or pd.isna(spin):
            return None

        fa_ratio = extract_fa_ratio_from_composition(str(comp))

        # Antisolvent flags
        anti_name = row.get("Antisolvent_Name", None)
        is_chloroform = 0.0
        is_chlorobenzene = 0.0
        is_none_antisolvent = 0.0
        if isinstance(anti_name, str):
            anti_name_lower = anti_name.strip().lower()
            if "chloroform" in anti_name_lower:
                is_chloroform = 1.0
            elif "chlorobenzene" in anti_name_lower or "cb" in anti_name_lower:
                is_chlorobenzene = 1.0
            elif anti_name_lower == 'none' or anti_name_lower == '':
                is_none_antisolvent = 1.0
        else:
            # If no antisolvent name, assume None
            is_none_antisolvent = 1.0

        return np.array(
            [
                fa_ratio,
                float(temp),
                float(spin),
                is_chloroform,
                is_chlorobenzene,
                is_none_antisolvent,
            ]
        )

    @staticmethod
    def _featurize_experiment(exp: ExperimentParams) -> np.ndarray:
        """
        Features for candidate experiments (same structure as log rows).
        """
        fa_ratio = extract_fa_ratio_from_composition(str(exp.material_system))
        temp = float(exp.annealing_temp)
        spin = float(exp.spin_speed)

        # Antisolvent flags
        is_chloroform = 0.0
        is_chlorobenzene = 0.0
        is_toluene = 0.0 # New: Add flag for Toluene
        is_none_antisolvent = 0.0
        if exp.antisolvent is not None:
            anti_name_lower = exp.antisolvent.name.strip().lower()
            if "chloroform" in anti_name_lower:
                is_chloroform = 1.0
            elif "chlorobenzene" in anti_name_lower or "cb" in anti_name_lower:
                is_chlorobenzene = 1.0
            elif "toluene" in anti_name_lower:
                is_toluene = 1.0 # Set flag if Toluene is used
        else:
            is_none_antisolvent = 1.0

        return np.array(
            [
                fa_ratio,
                temp,
                spin,
                is_chloroform,
                is_chlorobenzene,
                is_toluene, # Include Toluene feature
                is_none_antisolvent,
            ]
        )

    def train_from_history(self, history_df: pd.DataFrame, quality_learner: ActiveLearner):
        """
        Train the tree model on completed experiments, using the same composite
        quality target as the GP learner. This way the tree learns from actual
        measured outcomes, not from rules.
        """
        if history_df is None or history_df.empty:
            print("ProcessTreeLearner: No history available.")
            self.is_trained = False
            return

        X_rows = []
        y_vals = []

        # Ensure column names are stripped for consistent access
        history_df.columns = [c.strip() for c in history_df.columns]

        for idx, row in history_df.iterrows():
            # Must have at least PL or Stability to be useful (same as GP)
            # Check multiple possible column names for robustness
            pl_val = row.get("PL_Intensity", row.get("PL_Intensity ", None))
            stability_val = row.get("Stability_Hours", None)
            
            if pd.isna(pl_val) and pd.isna(stability_val):
                continue

            x = self._featurize_log_row(row)
            if x is None:
                continue

            y = quality_learner.calculate_quality_target(row)
            if pd.isna(y):
                continue
                
            X_rows.append(x)
            y_vals.append(y)

        if len(X_rows) < 5:  # Lower threshold to 5 to match GP
            print(f"ProcessTreeLearner: Not enough valid points (found {len(X_rows)}, need 5+).")
            self.is_trained = False
            return

        X = np.vstack(X_rows)
        y = np.array(y_vals, dtype=float)

        # Basic cleaning
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]

        if len(X) < 5:
            print(f"ProcessTreeLearner: Insufficient clean data after filtering (found {len(X)} points).")
            self.is_trained = False
            return

        print(f"ProcessTreeLearner: Training on {len(X)} history points...")
        try:
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
            )
            rf.fit(X, y)
            self.model = rf
            self.is_trained = True
            print(f"ProcessTreeLearner: Successfully trained.")
        except Exception as e:
            print(f"ProcessTreeLearner: Error training Random Forest: {e}")
            self.is_trained = False

    def score_candidates(self, candidates: List[ExperimentParams]):
        """
        Predict quality for each candidate and store it on exp.metrics.tree_predicted_quality.
        """
        if not self.is_trained or self.model is None or not candidates:
            return

        X_list = [self._featurize_experiment(exp) for exp in candidates]
        X = np.vstack(X_list)
        preds = self.model.predict(X)

        for exp, val in zip(candidates, preds):
            exp.metrics.tree_predicted_quality = float(val)

