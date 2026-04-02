import numpy as np
import pandas as pd
import jax.numpy as jnp
import gpax
import numpyro.distributions as dist

from typing import Optional, Dict, List, Tuple

from src.models import ExperimentParams
from src.learner import ActiveLearner


def _norm01(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v).astype(float)
    if v.size == 0:
        return v
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    denom = (vmax - vmin) if (vmax - vmin) > 1e-12 else 1.0
    return (v - vmin) / denom


class DualCompositionGP:
    """
    Separate Dual-GP model living in composition space (Cs, NEA, CE).

    - GP #1 predicts a quality target (same scalar target used by ActiveLearner)
      and uses UCB acquisition.
    - GP #2 predicts peak wavelength and uses target-seeking acquisition:
        -|mu - target| + beta * sigma

    The final composition acquisition is a weighted combination of the two,
    after normalizing each acquisition to [0, 1].
    """

    def __init__(self, cfg: Dict, quality_learner: ActiveLearner):
        self.cfg = cfg or {}
        self.quality_learner = quality_learner

        space = (self.cfg.get("composition_dual_gp", {}) or {})
        self.enabled = bool(space.get("enabled", True))

        self.cs_min = float(space.get("csbr_min", 0.10))
        self.cs_max = float(space.get("csbr_max", 0.18))
        self.nea_min = float(space.get("nea_min", 0.0))
        self.nea_max = float(space.get("nea_max", 0.4))
        self.ce_min = float(space.get("ce_min", 0.0))
        self.ce_max = float(space.get("ce_max", 0.02))

        self.ucb_beta = float(space.get("ucb_beta", 5.0))
        self.quality_weight = float(space.get("quality_weight", 0.5))
        self.peak_weight = float(space.get("peak_weight", 0.5))

        peak_cfg = (cfg.get("quality_target", {}).get("peak_wavelength", {}) or {})
        self.peak_target_nm = float(peak_cfg.get("target_nm", 700.0))

        self.use_peak_3d_if_available = bool(space.get("use_peak_3d_if_available", True))

        # Models
        self.is_trained = False
        self.peak_is_trained = False
        self.gp_quality = None
        self.gp_peak = None

        # Normalization ranges for independent concentration bounds
        self.ranges = np.array(
            [[self.cs_min, self.cs_max], [self.nea_min, self.nea_max], [self.ce_min, self.ce_max]],
            dtype=float,
        )

        # Transition penalty strength (discourage sampling near pending experiments)
        self.transition_constraint_factor = float(space.get("transition_constraint_factor", 5.0))

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.zeros_like(X)
        for d in range(X.shape[1]):
            x_min, x_max = self.ranges[d, 0], self.ranges[d, 1]
            denom = (x_max - x_min) if (x_max - x_min) > 1e-12 else 1.0
            out[:, d] = (X[:, d] - x_min) / denom
        return np.clip(out, 0.0, 1.0)

    def _extract_xyz_from_history(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        needed = ["CsBr_M", "NEABr_M", "CE_M"]
        for c in needed:
            if c not in df.columns:
                return np.empty((0, 3)), np.empty((0,))
        X = df[needed].astype(float).to_numpy()
        return X, np.ones(len(X))

    def train(self, history_df: pd.DataFrame):
        if not self.enabled:
            self.is_trained = False
            self.peak_is_trained = False
            return

        if history_df is None or history_df.empty or len(history_df) < 5:
            self.is_trained = False
            self.peak_is_trained = False
            return

        if self.quality_learner is None:
            self.is_trained = False
            self.peak_is_trained = False
            return

        # Filter rows that have composition values
        needed = ["CsBr_M", "NEABr_M", "CE_M"]
        df = history_df.copy()
        for c in needed:
            if c not in df.columns:
                self.is_trained = False
                self.peak_is_trained = False
                return

        df = df.dropna(subset=needed)
        if len(df) < 5:
            self.is_trained = False
            self.peak_is_trained = False
            return

        X = df[needed].astype(float).to_numpy()
        Xn = self._normalize(X)

        # Quality target from the same scoring function you already use
        yq = np.array([self.quality_learner.calculate_quality_target(row) for _, row in df.iterrows()], dtype=float)

        # Train quality GP (variational GP for speed/robustness)
        noise_prior = dist.HalfNormal(scale=float(self.cfg.get("composition_dual_gp", {}).get("noise_prior_scale", 0.1)))

        rng_key1, _ = gpax.utils.get_keys()
        self.gp_quality = gpax.viGP(input_dim=3, kernel="RBF", noise_prior_dist=noise_prior)
        self.gp_quality.fit(rng_key1, Xn, yq, jitter=1e-5)
        self.is_trained = True

        # Peak GP (optional)
        peak_vals: List[float] = []
        peak_X: List[np.ndarray] = []
        for _, row in df.iterrows():
            peak_val = None
            if self.use_peak_3d_if_available:
                peak_val = row.get("PL_Peak_3d_nm", None)
            if peak_val is None or pd.isna(peak_val) or float(peak_val) <= 0:
                peak_val = row.get("PL_Peak_nm", None)
            if peak_val is None or pd.isna(peak_val):
                continue
            try:
                p = float(peak_val)
            except (ValueError, TypeError):
                continue
            if p <= 0:
                continue
            peak_vals.append(p)
            peak_X.append(np.array([float(row["CsBr_M"]), float(row["NEABr_M"]), float(row["CE_M"])], dtype=float))

        if len(peak_vals) >= 5:
            Xp = np.vstack(peak_X)
            Xpn = self._normalize(Xp)
            yp = np.array(peak_vals, dtype=float)
            rng_key2, _ = gpax.utils.get_keys()
            self.gp_peak = gpax.viGP(input_dim=3, kernel="RBF", noise_prior_dist=noise_prior)
            self.gp_peak.fit(rng_key2, Xpn, yp, jitter=1e-5)
            self.peak_is_trained = True
        else:
            self.gp_peak = None
            self.peak_is_trained = False

    def score_candidates(self, candidates: List[ExperimentParams], transition_experiments: Optional[List[ExperimentParams]] = None):
        if not self.is_trained or self.gp_quality is None:
            return

        # Collect composition vectors
        X = []
        idx_map = []
        for i, c in enumerate(candidates):
            if c.csbr_M is None or c.neabr_M is None or c.ce_M is None:
                continue
            X.append([float(c.csbr_M), float(c.neabr_M), float(c.ce_M)])
            idx_map.append(i)

        if not X:
            return

        X = np.array(X, dtype=float)
        Xn = self._normalize(X)
        Xn_jax = jnp.array(Xn)
        rng_key, _ = gpax.utils.get_keys()

        # Quality acquisition (UCB)
        acq_q = gpax.acquisition.UCB(rng_key, self.gp_quality, Xn_jax, beta=self.ucb_beta, maximize=True)
        mu_q, samp_q = self.gp_quality.predict(rng_key, Xn_jax, noiseless=False, jitter=1e-4)
        mu_q_np = np.array(mu_q).ravel()
        samp_q_np = np.array(samp_q)
        if samp_q_np.ndim > 1:
            std_q = np.std(samp_q_np, axis=0).ravel()
        else:
            std_q = np.ones_like(mu_q_np) * 0.0
        acq_q_np = np.array(acq_q).ravel()

        # Peak acquisition (target-seeking)
        peak_mu = None
        peak_std = None
        peak_acq = None
        if self.peak_is_trained and self.gp_peak is not None:
            mu_p, samp_p = self.gp_peak.predict(rng_key, Xn_jax, noiseless=False, jitter=1e-4)
            peak_mu = np.array(mu_p).ravel()
            samp_p_np = np.array(samp_p)
            if samp_p_np.ndim > 1:
                peak_std = np.std(samp_p_np, axis=0).ravel()
            else:
                peak_std = np.ones_like(peak_mu) * 0.0
            deviation = np.abs(peak_mu - float(self.peak_target_nm))
            peak_acq = (-deviation + self.ucb_beta * peak_std).ravel()

        combined = acq_q_np
        if peak_acq is not None and len(peak_acq) == len(acq_q_np):
            wq = float(self.quality_weight)
            wp = float(self.peak_weight)
            denom = (wq + wp) if (wq + wp) > 1e-12 else 1.0
            combined = (wq * _norm01(acq_q_np) + wp * _norm01(peak_acq)) / denom

        # Transition penalty in composition space (normalize + Euclidean distance)
        if transition_experiments:
            Xt = []
            for t in transition_experiments:
                if t.csbr_M is None or t.neabr_M is None or t.ce_M is None:
                    continue
                Xt.append([float(t.csbr_M), float(t.neabr_M), float(t.ce_M)])
            if Xt:
                Xt = np.array(Xt, dtype=float)
                Xtn = self._normalize(Xt)

                d2 = ((Xn[:, None, :] - Xtn[None, :, :]) ** 2).sum(axis=2)
                min_d2 = np.min(d2, axis=1)
                penalty = np.exp(-self.transition_constraint_factor * 10.0 * min_d2)
                weights = 1.0 - penalty
                combined = combined * weights

        # Attach to candidates (do NOT overwrite process-GP fields)
        for local_i, cand_i in enumerate(idx_map):
            exp = candidates[cand_i]
            exp.metrics.composition_predicted_quality = float(mu_q_np[local_i])
            exp.metrics.composition_uncertainty_score = float(std_q[local_i])
            exp.metrics.composition_acquisition_score = float(combined[local_i])

            if peak_mu is not None:
                exp.metrics.composition_predicted_peak_nm = float(peak_mu[local_i])
            if peak_std is not None:
                exp.metrics.composition_peak_uncertainty_nm = float(peak_std[local_i])
            if peak_acq is not None:
                exp.metrics.composition_peak_acquisition_score = float(peak_acq[local_i])

