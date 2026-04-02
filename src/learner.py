import pandas as pd
import numpy as np
import gpax
import jax.numpy as jnp
import re
from typing import List, Tuple, Optional, Dict
from src.models import ExperimentParams
from src.delayed_gpbo import transition_constraint

def extract_fa_ratio_from_composition(comp_str: str) -> float:
    """
    Extract FA ratio from composition string like 'FA0.46FuDMA0.54Pb1I3.1'
    Returns FA fraction (0-1) where 1 = pure FA, 0 = pure FuDMA.
    """
    if pd.isna(comp_str) or not comp_str:
        return 0.5  # Default to middle
    
    comp_str = str(comp_str).strip()
    
    # Try to extract FA and FuDMA ratios
    fa_match = re.search(r'FA([\d.]+)', comp_str, re.IGNORECASE)
    fudma_match = re.search(r'FuDMA([\d.]+)', comp_str, re.IGNORECASE)
    
    if fa_match and fudma_match:
        fa_num = float(fa_match.group(1))
        fudma_num = float(fudma_match.group(1))
        total = fa_num + fudma_num
        if total > 0:
            return fa_num / total
    elif fa_match:
        # If only FA found, assume it's the ratio
        fa_num = float(fa_match.group(1))
        # Normalize if > 1 (likely absolute ratio)
        if fa_num > 1.0:
            # Try to find total from Pb ratio or use default
            return min(1.0, fa_num / 1.0)
        return fa_num
    
    # Fallback: try to estimate from composition string
    if 'FA' in comp_str.upper() and 'FUDMA' in comp_str.upper():
        # Rough estimate: count characters or use position
        return 0.5  # Unknown, default
    
    return 0.5  # Default if parsing fails


class ActiveLearner:
    def __init__(self, quality_config: Optional[Dict] = None):
        self.is_trained = False
        self.X_train = None
        self.y_train = None
        self.X_min = None
        self.X_max = None
        self.gp_model = None  # Store trained GP model
        self.gp_model_peak = None  # Optional: second GP for peak wavelength
        self.peak_is_trained = False
        
        # Quality target configuration (defaults if not provided)
        self.quality_config = quality_config or {}
        self.pl_weight = self.quality_config.get('pl_weight', 1.0)
        self.pl_scale = self.quality_config.get('pl_scale', 100.0)
        self.fwhm_weight = self.quality_config.get('fwhm_weight', 50.0)
        self.fwhm_ideal = self.quality_config.get('fwhm_ideal_nm', 20.0)
        self.stability_weight = self.quality_config.get('stability_weight', 2.0)
        
        # Peak wavelength targeting
        peak_cfg = self.quality_config.get('peak_wavelength', {})
        self.peak_enabled = peak_cfg.get('enabled', False)
        self.peak_target = peak_cfg.get('target_nm', 700.0)
        self.peak_tolerance = peak_cfg.get('tolerance_nm', 20.0)
        self.peak_penalty = peak_cfg.get('penalty_weight', 30.0)
        self.peak_penalty = peak_cfg.get('penalty_weight', 30.0)

        # Dual-GP (separate peak GP + quality GP) configuration.
        # This follows the external Dual-GP idea: one GP targets wavelength, one targets intensity/quality,
        # then combine normalized acquisitions with weights.
        dual_cfg = self.quality_config.get('dual_gp', {}) or {}
        self.dual_gp_enabled = bool(dual_cfg.get('enabled', False))
        self.dual_gp_beta = float(dual_cfg.get('ucb_beta', 2.0))
        self.dual_weight_quality = float(dual_cfg.get('quality_weight', 0.5))
        self.dual_weight_peak = float(dual_cfg.get('peak_weight', 0.5))
        # If true, and if a row has both PL_Peak_nm and PL_Peak_3d_nm, use the 3d value for peak training.
        self.dual_use_peak_3d_if_available = bool(dual_cfg.get('use_peak_3d_if_available', True))

        # Process-GP toggle: you said you do NOT want any GP learning spin-coating processing.
        # Keep this off by default; we still use this class for `calculate_quality_target()`.
        process_gp_cfg = self.quality_config.get('process_gp', {}) or {}
        self.process_gp_enabled = bool(process_gp_cfg.get('enabled', False))
        
    def _featurize(self, exp_params: ExperimentParams) -> np.ndarray:
        """
        Simplified feature vector using only the most varied parameters.
        Features: 
        1. FA Ratio (from composition) - 0 to 1
        2. Annealing Temperature (°C)
        3. Spin Speed (rpm) - second/main step
        """
        # Extract FA ratio from material system
        fa_ratio = extract_fa_ratio_from_composition(str(exp_params.material_system))
        
        return np.array([
            fa_ratio,
            float(exp_params.annealing_temp),
            float(exp_params.spin_speed)
        ])
    
    def _featurize_from_log_row(self, row: pd.Series) -> Optional[np.ndarray]:
        """
        Extract simplified features from a log row.
        Returns None if required fields are missing.
        """
        comp = row.get('Material_Composition', '')
        temp = row.get('Anneal_Temp_C', row.get('Temp_C', None))
        spin = row.get('Spin_Speed_rpm', None)
        
        if pd.isna(comp) or pd.isna(temp) or pd.isna(spin):
            return None
        
        fa_ratio = extract_fa_ratio_from_composition(str(comp))
        
        return np.array([fa_ratio, float(temp), float(spin)])

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        if self.X_min is None:
            return X
        range_vals = self.X_max - self.X_min
        range_vals[range_vals == 0] = 1.0 
        return (X - self.X_min) / range_vals

    def calculate_quality_target(self, row: pd.Series) -> float:
        """
        Combines PL, FWHM, Peak wavelength, Stability into a single target score for GPBO.
        Higher is better. Weights are configurable via config.yaml.
        
        Note: Stability is only included if available. Missing stability data doesn't
        penalize the score, ensuring fair comparison between experiments with/without
        stability measurements.
        """
        # Extract metrics (handle column name variations with/without trailing spaces)
        pl = row.get('PL_Intensity', row.get('PL_Intensity ', 0))
        fwhm = row.get('PL_FWHM_nm', 0)
        stability = row.get('Stability_Hours', None)  # None if missing
        peak_wavelength = row.get('PL_Peak_nm', row.get('Peak_Wavelength_nm', None))
        
        # Clean data
        if pd.isna(pl): pl = 0
        if pd.isna(fwhm): fwhm = 50  # Default wide FWHM (bad)
        if pd.isna(stability): stability = None  # Keep as None if missing
        if pd.isna(peak_wavelength): peak_wavelength = None
        
        score = 0.0
        
        # 1. PL Intensity (configurable weight and scale)
        score += self.pl_weight * (pl / self.pl_scale)
        
        # 2. FWHM: Reward narrower peaks (configurable ideal)
        if fwhm > 0:
            score += self.fwhm_weight * (self.fwhm_ideal / fwhm)
            
        # 3. Stability: Only include if available (don't penalize missing data)
        # This ensures fair comparison: experiments without stability aren't
        # automatically worse than those with it
        if stability is not None and not pd.isna(stability) and stability > 0:
            score += self.stability_weight * stability
        
        # 4. Peak wavelength targeting (for phase control)
        if self.peak_enabled and peak_wavelength is not None:
            deviation = abs(peak_wavelength - self.peak_target)
            if deviation > self.peak_tolerance:
                # Penalty increases linearly outside tolerance window
                penalty = self.peak_penalty * (deviation - self.peak_tolerance) / 10.0
                score -= penalty
            else:
                # Bonus for being on-target
                score += self.peak_penalty * 0.5  # Half the penalty weight as bonus
        
        return score

    def train(self, history_df: pd.DataFrame):
        if not self.process_gp_enabled:
            # Intentionally skip training any process-space GP.
            self.is_trained = False
            self.gp_model = None
            self.gp_model_peak = None
            self.peak_is_trained = False
            return

        if history_df.empty or len(history_df) < 5:
            print("Not enough history to train GP model yet. (Need 5+ points)")
            self.is_trained = False
            return

        # Step 1: Extract features and targets, filtering invalid rows
        data_rows = []
        peak_rows = []
        for _, row in history_df.iterrows():
            # Must have at least PL or Stability to be useful
            pl_val = row.get('PL_Intensity', row.get('PL_Intensity ', None))
            if pd.isna(pl_val) and pd.isna(row.get('Stability_Hours')):
                continue
            
            # Extract simplified features
            x_vec = self._featurize_from_log_row(row)
            if x_vec is None:
                continue
            
            # Calculate quality target
            quality = self.calculate_quality_target(row)
            
            data_rows.append({
                'x': x_vec,
                'y': quality,
                'row': row  # Keep reference for debugging
            })

            # Optional: peak wavelength target for separate GP
            # Prefer 3d peak if configured and present; else initial peak.
            peak_val = None
            if self.dual_gp_enabled or self.peak_enabled:
                if self.dual_use_peak_3d_if_available:
                    peak_val = row.get('PL_Peak_3d_nm', None)
                if peak_val is None or pd.isna(peak_val) or float(peak_val) <= 0:
                    peak_val = row.get('PL_Peak_nm', row.get('Peak_Wavelength_nm', None))
                if peak_val is not None and not pd.isna(peak_val):
                    try:
                        peak_float = float(peak_val)
                    except (ValueError, TypeError):
                        peak_float = None
                    if peak_float is not None and peak_float > 0:
                        peak_rows.append({'x': x_vec, 'y_peak': peak_float})
        
        if len(data_rows) < 5:
            print("Not enough valid data points after filtering.")
            self.is_trained = False
            return
        
        # Step 2: Aggregate replicates by unique (FA_ratio, temp, spin) combinations
        # Group by feature vector (rounded to avoid floating point issues)
        groups = {}
        for item in data_rows:
            x_key = tuple(np.round(item['x'], 4))  # Round to 4 decimals for grouping
            if x_key not in groups:
                groups[x_key] = {'x': item['x'], 'y_values': [], 'count': 0}
            groups[x_key]['y_values'].append(item['y'])
            groups[x_key]['count'] += 1
        
        # Step 3: Compute mean and std for each unique condition
        X_unique = []
        y_unique = []
        y_std = []
        replicate_counts = []
        
        for key, group_data in groups.items():
            X_unique.append(group_data['x'])
            y_values = np.array(group_data['y_values'])
            y_unique.append(np.mean(y_values))
            y_std.append(np.std(y_values) if len(y_values) > 1 else 0.0)
            replicate_counts.append(group_data['count'])
        
        X = np.array(X_unique)
        y = np.array(y_unique)
        
        # Report aggregation results
        n_replicates = sum(1 for c in replicate_counts if c > 1)
        print(f"Training GP on {len(X)} unique conditions (aggregated from {len(data_rows)} total measurements)")
        if n_replicates > 0:
            print(f"  - {n_replicates} conditions had replicates (using mean)")
            max_reps = max(replicate_counts)
            print(f"  - Max replicates per condition: {max_reps}")
        
        if len(X) < 5:
            print("Not enough unique conditions after aggregation (need 5+).")
            self.is_trained = False
            return
        
        # Check for NaN/Inf values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("Warning: NaN/Inf values in features. Cleaning...")
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
            X = X[valid_mask]
            y = y[valid_mask]
            if len(X) < 5:
                print("Not enough valid data after cleaning.")
                self.is_trained = False
                return
        
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            print("Warning: NaN/Inf values in targets. Cleaning...")
            valid_mask = ~(np.isnan(y) | np.isinf(y))
            X = X[valid_mask]
            y = y[valid_mask]
            if len(X) < 5:
                print("Not enough valid data after cleaning.")
                self.is_trained = False
                return
        
        # Remove duplicate rows (add small random noise to break exact duplicates)
        X_unique = []
        y_unique = []
        seen = set()
        for i, (x_row, y_val) in enumerate(zip(X, y)):
            x_tuple = tuple(np.round(x_row, 6))  # Round to avoid floating point issues
            if x_tuple not in seen:
                seen.add(x_tuple)
                X_unique.append(x_row)
                y_unique.append(y_val)
            else:
                # Add tiny random noise to break exact duplicate (prevents singular covariance)
                noise = np.random.normal(0, 1e-5, size=x_row.shape)
                X_unique.append(x_row + noise)
                y_unique.append(y_val)
        
        X = np.array(X_unique)
        y = np.array(y_unique)
        
        if len(X) < 5:
            print("Not enough unique data points after deduplication.")
            self.is_trained = False
            return
        
        self.X_min = np.min(X, axis=0)
        self.X_max = np.max(X, axis=0)
        
        # Check for zero-variance features and add small range
        range_vals = self.X_max - self.X_min
        zero_var_mask = range_vals < 1e-10
        if np.any(zero_var_mask):
            print(f"Warning: {np.sum(zero_var_mask)} feature(s) have zero variance. Adding small range.")
            self.X_max[zero_var_mask] = self.X_max[zero_var_mask] + 1.0
            self.X_min[zero_var_mask] = self.X_min[zero_var_mask] - 1.0
        
        self.X_train = self._normalize(X)
        self.y_train = y
        
        # Store replicate statistics for plotting
        self.y_train_std = np.array(y_std)
        self.replicate_counts = replicate_counts
        
        # Store feature names for better visualization
        self.feature_names = ['FA_Ratio', 'Annealing_Temp_C', 'Spin_Speed_rpm']
        
        # Check normalized data
        if np.any(np.isnan(self.X_train)) or np.any(np.isinf(self.X_train)):
            print("Error: Normalization produced NaN/Inf values.")
            self.is_trained = False
            return
        
        # Train GP model once and store it
        X_train_jax = jnp.array(self.X_train)
        y_train_jax = jnp.array(self.y_train)
        rng_key = gpax.utils.get_keys(1)[0]
        
        try:
            # Try RBF kernel first (more numerically stable)
            self.gp_model = gpax.ExactGP(
                input_dim=X_train_jax.shape[1], 
                kernel='RBF'
            )
            self.gp_model.fit(
                rng_key, 
                X_train_jax, 
                y_train_jax, 
                num_warmup=2000, 
                num_samples=2000
            )
            self.is_trained = True
            print(f"Active Learner trained on {len(X)} past experiments (Quality Target).")
        except (ValueError, RuntimeError) as e:
            # If RBF fails, try Matern with fewer samples
            print(f"RBF kernel failed ({str(e)[:100]}...), trying Matern kernel...")
            try:
                self.gp_model = gpax.ExactGP(
                    input_dim=X_train_jax.shape[1], 
                    kernel='Matern'
                )
                self.gp_model.fit(
                    rng_key, 
                    X_train_jax, 
                    y_train_jax, 
                    num_warmup=1000,  # Reduced warmup
                    num_samples=1000  # Reduced samples
                )
                self.is_trained = True
                print(f"Active Learner trained on {len(X)} past experiments (Quality Target) with Matern kernel.")
            except Exception as e2:
                print(f"Error training GP model: {e2}")
                print("This may be due to:")
                print("  - Duplicate or very similar data points")
                print("  - Numerical instability in covariance matrix")
                print("  - Too few unique data points")
                print("  - Features with zero or very small variance")
                print("Continuing without GP model (will use physics-based scoring only)...")
                self.is_trained = False
                self.gp_model = None

        # Optional: train separate GP for peak wavelength (Dual-GP mode)
        self.peak_is_trained = False
        self.gp_model_peak = None
        if self.dual_gp_enabled and len(peak_rows) >= 5:
            try:
                Xp = np.array([r['x'] for r in peak_rows])
                yp = np.array([r['y_peak'] for r in peak_rows])
                Xp_norm = self._normalize(Xp)
                Xp_jax = jnp.array(Xp_norm)
                yp_jax = jnp.array(yp)
                rng_key_peak = gpax.utils.get_keys(1)[0]

                # Peak GP typically smoother; RBF is fine here.
                self.gp_model_peak = gpax.ExactGP(input_dim=Xp_jax.shape[1], kernel='RBF')
                self.gp_model_peak.fit(
                    rng_key_peak,
                    Xp_jax,
                    yp_jax,
                    num_warmup=1000,
                    num_samples=1000
                )
                self.peak_is_trained = True
                print(f"Active Learner: trained Peak GP on {len(peak_rows)} points.")
            except Exception as e:
                print(f"Active Learner: could not train Peak GP (continuing with quality-only GP): {e}")
                self.peak_is_trained = False
                self.gp_model_peak = None

    def score_candidates_with_uncertainty(
        self,
        candidates: List[ExperimentParams],
        transition_experiments: List[ExperimentParams] = None
    ):
        """
        Score candidates with the process GP:
        - predicted_performance (posterior mean)
        - uncertainty_score (posterior std)
        - acquisition_score (UCB, optionally penalized by transition similarity)
        """
        if not self.is_trained or self.gp_model is None:
            return

        # Batch processing to avoid memory issues with large candidate sets
        batch_size = 100
        X_trans_norm = None
        if transition_experiments:
            X_trans = np.array([self._featurize(c) for c in transition_experiments])
            X_trans_norm = self._normalize(X_trans)

        rng_key_predict = gpax.utils.get_keys(1)[0]

        for batch_start in range(0, len(candidates), batch_size):
            batch_end = min(batch_start + batch_size, len(candidates))
            batch_candidates = candidates[batch_start:batch_end]

            X_cand = np.array([self._featurize(c) for c in batch_candidates])
            X_cand_norm = self._normalize(X_cand)
            X_cand_jax = jnp.array(X_cand_norm)

            # Use stored GP model for prediction (no retraining!)
            acq_values = gpax.acquisition.UCB(rng_key_predict, self.gp_model, X_cand_jax, beta=self.dual_gp_beta if self.dual_gp_enabled else 2.0)
            posterior_mean, posterior_samples = self.gp_model.predict(rng_key_predict, X_cand_jax, n=200)

            # Convert JAX arrays to numpy for safe scalar extraction
            acq_values_np = np.array(acq_values).ravel()
            posterior_mean_np = np.array(posterior_mean).ravel()
            posterior_samples_np = np.array(posterior_samples)

            # Calculate uncertainty (std dev) from posterior samples
            if posterior_samples_np.ndim == 3:
                posterior_samples_np = posterior_samples_np.reshape(posterior_samples_np.shape[0], -1)
            posterior_std = np.std(posterior_samples_np, axis=0).ravel()

            # Ensure arrays match batch size
            n_batch = len(batch_candidates)
            if len(posterior_std) != n_batch:
                posterior_std = np.full(n_batch, np.mean(np.std(posterior_samples_np, axis=0)))

            # Optional: peak GP acquisition (Dual-GP)
            peak_acq_np = None
            peak_mean_np = None
            peak_std_np = None
            if self.dual_gp_enabled and self.peak_is_trained and self.gp_model_peak is not None:
                try:
                    peak_mean, peak_samples = self.gp_model_peak.predict(rng_key_predict, X_cand_jax, n=200)
                    peak_mean_np = np.array(peak_mean).ravel()
                    peak_samples_np = np.array(peak_samples)
                    if peak_samples_np.ndim == 3:
                        peak_samples_np = peak_samples_np.reshape(peak_samples_np.shape[0], -1)
                    peak_std_np = np.std(peak_samples_np, axis=0).ravel()

                    # Targeted acquisition: closer to target is better + exploration
                    deviation = np.abs(peak_mean_np - float(self.peak_target))
                    peak_acq_np = (-deviation + self.dual_gp_beta * peak_std_np).ravel()
                except Exception:
                    peak_acq_np = None
                    peak_mean_np = None
                    peak_std_np = None

            # Combine acquisitions if Dual-GP is enabled and peak acquisition is available.
            combined_acq_np = acq_values_np
            if self.dual_gp_enabled and peak_acq_np is not None and len(peak_acq_np) == len(acq_values_np):
                # Normalize each acquisition into [0, 1] for stable weighting
                def _norm01(v: np.ndarray) -> np.ndarray:
                    v = np.asarray(v).astype(float)
                    vmin = float(np.min(v))
                    vmax = float(np.max(v))
                    denom = (vmax - vmin) if (vmax - vmin) > 1e-12 else 1.0
                    return (v - vmin) / denom

                q_norm = _norm01(acq_values_np)
                p_norm = _norm01(peak_acq_np)
                wq = float(self.dual_weight_quality)
                wp = float(self.dual_weight_peak)
                wsum = (wq + wp) if (wq + wp) > 1e-12 else 1.0
                combined_acq_np = (wq * q_norm + wp * p_norm) / wsum

            # Delayed-workflow penalty: discourage selecting candidates near transition experiments
            if X_trans_norm is not None:
                penalty_weights = transition_constraint(X_cand_norm, X_trans_norm, constraint_factor=5.0)
                combined_acq_np = combined_acq_np * np.array(penalty_weights).ravel()

            for i, exp in enumerate(batch_candidates):
                acq_val = combined_acq_np[i] if i < len(combined_acq_np) else 0.0
                mean_val = posterior_mean_np[i] if i < len(posterior_mean_np) else 0.0
                std_val = posterior_std[i] if i < len(posterior_std) else 0.0

                exp.metrics.acquisition_score = float(acq_val)
                exp.metrics.predicted_performance = float(mean_val)
                exp.metrics.uncertainty_score = float(std_val)

                if peak_mean_np is not None and i < len(peak_mean_np):
                    exp.metrics.predicted_peak_nm = float(peak_mean_np[i])
                if peak_std_np is not None and i < len(peak_std_np):
                    exp.metrics.peak_uncertainty_nm = float(peak_std_np[i])
                if peak_acq_np is not None and i < len(peak_acq_np):
                    exp.metrics.peak_acquisition_score = float(peak_acq_np[i])


class CompositionGP:
    """
    Lightweight 1D composition-only GP.

    This is intended to be trained on a binary FA/FuDMA library where each row
    has an FA fraction (0-1) and an associated quality/score. It can then be
    queried with FA fractions to provide a scalar composition score that we
    feed into the Monte Carlo generator to bias composition sampling.
    """

    def __init__(self, ucb_beta: float = 2.0):
        self.is_trained = False
        self.X_train = None
        self.y_train = None
        self.X_min = None
        self.X_max = None
        self.gp_model = None
        self.ucb_beta = float(ucb_beta)

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        if self.X_min is None:
            return X
        range_vals = self.X_max - self.X_min
        range_vals[range_vals == 0] = 1.0
        return (X - self.X_min) / range_vals

    def train_from_dataframe(
        self,
        df: pd.DataFrame,
        fa_column: str = "FA_Ratio",
        quality_column: str = "Quality_Target",
    ):
        if df is None or df.empty:
            print("CompositionGP: no data provided.")
            self.is_trained = False
            return

        if fa_column not in df.columns or quality_column not in df.columns:
            print(f"CompositionGP: columns '{fa_column}' and/or '{quality_column}' not found.")
            self.is_trained = False
            return

        fa_vals = df[fa_column].astype(float).to_numpy()
        q_vals = df[quality_column].astype(float).to_numpy()

        mask = ~np.isnan(fa_vals) & ~np.isnan(q_vals)
        fa_vals = fa_vals[mask]
        q_vals = q_vals[mask]

        if len(fa_vals) < 5:
            print("CompositionGP: not enough valid points (need 5+).")
            self.is_trained = False
            return

        X = fa_vals.reshape(-1, 1)
        y = q_vals

        self.X_min = np.min(X, axis=0)
        self.X_max = np.max(X, axis=0)
        self.X_train = self._normalize(X)
        self.y_train = y

        X_train_jax = jnp.array(self.X_train)
        y_train_jax = jnp.array(self.y_train)
        rng_key = gpax.utils.get_keys(1)[0]

        try:
            self.gp_model = gpax.ExactGP(input_dim=1, kernel="RBF")
            self.gp_model.fit(
                rng_key,
                X_train_jax,
                y_train_jax,
                num_warmup=1000,
                num_samples=1000,
            )
            self.is_trained = True
            print(f"CompositionGP: trained on {len(X)} composition points.")
        except Exception as e:
            print(f"CompositionGP: error training GP: {e}")
            self.is_trained = False
            self.gp_model = None

    def train_from_binary_and_history(
        self,
        binary_df: pd.DataFrame,
        history_df: pd.DataFrame,
        quality_learner,
        fa_column: str = "FA_Ratio",
        quality_column: str = "Quality_Target",
    ):
        """
        Train Composition GP on both binary dataset AND experiment history.
        
        Args:
            binary_df: DataFrame from binary fit export (with FA_Ratio and Quality_Target)
            history_df: Experiment history DataFrame (will extract FA ratio and quality)
            quality_learner: ActiveLearner instance to compute quality targets from history
            fa_column: Column name for FA ratio in binary_df
            quality_column: Column name for quality in binary_df
        """
        all_fa_vals = []
        all_q_vals = []
        
        # 1. Add binary dataset points
        if binary_df is not None and not binary_df.empty:
            if fa_column in binary_df.columns and quality_column in binary_df.columns:
                fa_binary = binary_df[fa_column].astype(float).to_numpy()
                q_binary = binary_df[quality_column].astype(float).to_numpy()
                mask = ~np.isnan(fa_binary) & ~np.isnan(q_binary)
                all_fa_vals.extend(fa_binary[mask].tolist())
                all_q_vals.extend(q_binary[mask].tolist())
                print(f"  Added {np.sum(mask)} points from binary dataset")
        
        # 2. Add experiment history points (extract FA ratio and quality)
        if history_df is not None and not history_df.empty and quality_learner is not None:
            history_points = 0
            for _, row in history_df.iterrows():
                # Must have at least PL or Stability to be useful
                pl_val = row.get('PL_Intensity', row.get('PL_Intensity ', None))
                if pd.isna(pl_val) and pd.isna(row.get('Stability_Hours')):
                    continue
                
                # Extract FA ratio from composition
                comp = row.get('Material_Composition', '')
                if pd.isna(comp):
                    continue
                
                fa_ratio = extract_fa_ratio_from_composition(str(comp))
                if pd.isna(fa_ratio) or fa_ratio < 0 or fa_ratio > 1:
                    continue
                
                # Calculate quality target (same as Process GP uses)
                quality = quality_learner.calculate_quality_target(row)
                if pd.isna(quality):
                    continue
                
                all_fa_vals.append(float(fa_ratio))
                all_q_vals.append(float(quality))
                history_points += 1
            
            if history_points > 0:
                print(f"  Added {history_points} points from experiment history")
        
        # 3. Train on combined dataset
        if len(all_fa_vals) < 5:
            print(f"CompositionGP: not enough total points ({len(all_fa_vals)}), need 5+.")
            self.is_trained = False
            return
        
        fa_array = np.array(all_fa_vals)
        q_array = np.array(all_q_vals)
        
        # Remove duplicates (same FA ratio, keep average quality)
        unique_fa = {}
        for fa, q in zip(fa_array, q_array):
            fa_key = round(fa, 4)  # Round to avoid floating point issues
            if fa_key not in unique_fa:
                unique_fa[fa_key] = []
            unique_fa[fa_key].append(q)
        
        # Average quality for duplicate FA ratios
        fa_unique = []
        q_unique = []
        for fa_key, q_vals_list in unique_fa.items():
            fa_unique.append(fa_key)
            q_unique.append(np.mean(q_vals_list))
        
        X = np.array(fa_unique).reshape(-1, 1)
        y = np.array(q_unique)
        
        self.X_min = np.min(X, axis=0)
        self.X_max = np.max(X, axis=0)
        self.X_train = self._normalize(X)
        self.y_train = y

        X_train_jax = jnp.array(self.X_train)
        y_train_jax = jnp.array(self.y_train)
        rng_key = gpax.utils.get_keys(1)[0]

        try:
            self.gp_model = gpax.ExactGP(input_dim=1, kernel="RBF")
            self.gp_model.fit(
                rng_key,
                X_train_jax,
                y_train_jax,
                num_warmup=1000,
                num_samples=1000,
            )
            self.is_trained = True
            print(f"CompositionGP: trained on {len(X)} unique composition points ({len(all_fa_vals)} total including duplicates).")
        except Exception as e:
            print(f"CompositionGP: error training GP: {e}")
            self.is_trained = False
            self.gp_model = None

    def score_fa_ratio(self, fa_ratio: float) -> float:
        """
        Return a scalar composition score for a given FA fraction in [0, 1].
        Uses UCB acquisition on the 1D GP. If the model is not trained, returns 0.
        """
        return self.score_fa_batch([fa_ratio])[0]

    def score_fa_batch(self, fa_ratios: List[float]) -> List[float]:
        """
        Batch version of score_fa_ratio for better performance.
        """
        if not self.is_trained or self.gp_model is None or not fa_ratios:
            return [0.0] * len(fa_ratios)

        x = np.array([[float(f)] for f in fa_ratios])
        x_norm = self._normalize(x)
        x_jax = jnp.array(x_norm)
        rng_key = gpax.utils.get_keys(1)[0]

        try:
            acq_vals = gpax.acquisition.UCB(rng_key, self.gp_model, x_jax, beta=self.ucb_beta)
            acq_np = np.array(acq_vals).ravel()
            return [float(v) for v in acq_np]
        except Exception as e:
            print(f"CompositionGP: Error in batch scoring: {e}")
            return [0.0] * len(fa_ratios)
