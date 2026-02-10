"""
Instability scoring functions for stability evaluation
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


def instability_score(
    df: pd.DataFrame,
    compositions_with_multiple_peaks: np.ndarray,
    target_wavelength: float,
    multiple_peak_penalty: float,
    wavelength_tolerance: float,
    degradation_weight: float,
    position_weight: float
) -> np.ndarray:
    """
    Calculate instability score considering multiple peaks, intensity degradation, and position deviation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: composition_number, iteration, initial_peak_positions,
        final_peak_positions, initial_peak_intensities, final_peak_intensities
    compositions_with_multiple_peaks : np.ndarray
        Array of compositions with multiple peaks (structured array with composition_number and iteration)
    target_wavelength : float
        Target wavelength for peak analysis
    multiple_peak_penalty : float
        Penalty score for multiple peaks
    wavelength_tolerance : float
        Tolerance range for target wavelength
    degradation_weight : float
        Weight for intensity degradation component
    position_weight : float
        Weight for position deviation component
        
    Returns:
    --------
    np.ndarray
        Array of instability scores
    """
    max_score = 3
    stb_scores = []
    
    # Create set for fast lookup
    multiple_peaks_set = set()
    if len(compositions_with_multiple_peaks) > 0:
        if compositions_with_multiple_peaks.dtype.names:
            # Structured array
            multiple_peaks_set = set(
                (int(row['composition_number']), int(row['iteration']))
                for row in compositions_with_multiple_peaks
            )
        else:
            # Simple array - assume all are from iteration 0
            multiple_peaks_set = set((int(comp), 0) for comp in compositions_with_multiple_peaks)
    
    for index, row in df.iterrows():
        current_comp = (row['composition_number'], row['iteration'])
        
        peak_positions_int = row['initial_peak_positions']
        peak_positions_fin = row['final_peak_positions']
        peak_intensities_int = row['initial_peak_intensities']
        peak_intensities_fin = row['final_peak_intensities']
        
        if peak_intensities_int == 0 and peak_intensities_fin == 0:
            stb_scores.append(max_score)
            continue
        
        # Intensity change component
        intensity_change = np.abs(peak_intensities_int - peak_intensities_fin) / max(
            peak_intensities_int + peak_intensities_fin, 1e-10
        )
        intensity_score = min(intensity_change, 1) * degradation_weight
        
        # Position deviation component
        initial_position_deviation = max(
            abs(peak_positions_int - target_wavelength) - wavelength_tolerance, 0
        )
        final_position_deviation = max(
            abs(peak_positions_fin - target_wavelength) - wavelength_tolerance, 0
        )
        position_score = (
            min(initial_position_deviation, final_position_deviation) / target_wavelength
        ) * position_weight
        
        # Multiple peaks penalty
        multiple_peaks_score = multiple_peak_penalty if current_comp in multiple_peaks_set else 0
        
        # Total score
        total_score = intensity_score + position_score + multiple_peaks_score
        total_score = min(total_score, max_score)
        stb_scores.append(total_score)
    
    return np.array(stb_scores)


def instability_score_simple(
    df: pd.DataFrame,
    compositions_with_multiple_peaks: np.ndarray,
    target_wavelength: float,
    wavelength_tolerance: float,
    degradation_weight: float,
    position_weight: float,
    multiple_peak_penalty: float,
) -> np.ndarray:
    """
    Simplified score that combines:
    - how close the peak wavelength is to the target (e.g. pure blue ~460 nm)
    - how bright the peak is at the initial read

    This version is intended for workflows where we are NOT tracking
    time‑dependent degradation, but instead want to:
    - **maximize initial intensity**
    - **strongly prefer wavelengths near the target**
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns:
        initial_peak_positions, final_peak_positions,
        initial_peak_intensities, final_peak_intensities,
        composition_number, iteration, and (optionally)
        initial_peak_fwhm, initial_left_area, initial_right_area.
    compositions_with_multiple_peaks : np.ndarray
        Structured or simple array identifying compositions that
        show multiple peaks. If structured, it must contain fields
        ('composition_number', 'iteration').
    target_wavelength : float
        Target wavelength (e.g. 460 for pure blue)
    wavelength_tolerance : float
        Range around target where no wavelength penalty is applied
    degradation_weight : float
        Weight for the *intensity* term (higher = care more about brightness)
    position_weight : float
        Weight for the wavelength term (higher = care more about color / λ)
    multiple_peak_penalty : float
        Additional penalty added when a composition has multiple peaks.
        
    Returns
    -------
    np.ndarray
        Array of penalty‑like scores in [0, 1],
        where 0 = ideal (bright + on‑target wavelength),
        1 = worst (dim + far from target).
    """
    max_score = 1.0
    stb_scores: List[float] = []

    # Use global statistics for normalization
    max_initial_intensity = max(float(df["initial_peak_intensities"].max()), 1e-10)

    has_fwhm = "initial_peak_fwhm" in df.columns
    has_left = "initial_left_area" in df.columns
    has_right = "initial_right_area" in df.columns

    if has_fwhm:
        fwhm_vals = df["initial_peak_fwhm"].replace(0, np.nan)
        valid_fwhm = fwhm_vals.dropna()
        fwhm_min = float(valid_fwhm.min()) if not valid_fwhm.empty else 0.0
        fwhm_max = float(valid_fwhm.max()) if not valid_fwhm.empty else 1.0
    else:
        fwhm_min, fwhm_max = 0.0, 1.0
    
    # Create a lookup set for multiple‑peak compositions
    multiple_peaks_set = set()
    if compositions_with_multiple_peaks is not None and len(compositions_with_multiple_peaks) > 0:
        if getattr(compositions_with_multiple_peaks, "dtype", None) is not None and compositions_with_multiple_peaks.dtype.names:
            # Structured array: expect fields 'composition_number' and 'iteration'
            multiple_peaks_set = set(
                (int(row["composition_number"]), int(row["iteration"]))
                for row in compositions_with_multiple_peaks
            )
        else:
            # Simple list/array of indices – assume iteration 0
            multiple_peaks_set = set((int(comp), 0) for comp in compositions_with_multiple_peaks)

    for _, row in df.iterrows():
        peak_positions_int = row["initial_peak_positions"]
        peak_positions_fin = row["final_peak_positions"]
        peak_intensities_int = row["initial_peak_intensities"]
        peak_fwhm_int = row["initial_peak_fwhm"] if has_fwhm else np.nan
        left_area = row["initial_left_area"] if has_left else np.nan
        right_area = row["initial_right_area"] if has_right else np.nan
        
        # ----- Wavelength / color penalty -----
        # Use the closer of initial / final wavelength to the target
        position_deviation = min(
            abs(peak_positions_fin - target_wavelength),
            abs(peak_positions_int - target_wavelength),
        )
        if np.isnan(position_deviation):
            position_deviation = 0.0

        if position_deviation <= wavelength_tolerance:
            position_penalty = 0.0
        else:
            position_penalty = min(position_deviation / target_wavelength, 1.0)
        
        # ----- Intensity penalty (based only on initial intensity) -----
        if np.isnan(peak_intensities_int) or peak_intensities_int <= 0:
            # No signal → worst case
            intensity_penalty = 1.0
        else:
            norm_intensity = min(peak_intensities_int / max_initial_intensity, 1.0)
            # Brightest peaks (norm_intensity≈1) → low penalty
            intensity_penalty = 1.0 - norm_intensity

        # ----- FWHM / width penalty (narrower = better) -----
        if has_fwhm and not np.isnan(peak_fwhm_int) and fwhm_max > fwhm_min:
            fwhm_norm = (peak_fwhm_int - fwhm_min) / (fwhm_max - fwhm_min)
            width_penalty = float(np.clip(fwhm_norm, 0.0, 1.0))
        else:
            width_penalty = 0.0

        # ----- Asymmetry penalty (left/right area similarity, lower = better) -----
        if has_left and has_right and (not np.isnan(left_area)) and (not np.isnan(right_area)):
            area_sum = max(left_area + right_area, 1e-10)
            asymmetry = abs(left_area - right_area) / area_sum
            asym_penalty = float(np.clip(asymmetry, 0.0, 1.0))
        else:
            asym_penalty = 0.0
        
        # ----- Combine with configurable weights -----
        # We interpret:
        # - degradation_weight: intensity weight
        # - position_weight: wavelength / color weight
        # Distribute remaining weight across width & asymmetry if available.
        w_int = degradation_weight
        w_pos = position_weight
        remaining = max(1.0 - (w_int + w_pos), 0.0)
        w_width = remaining * 0.6  # give more of remaining to FWHM
        w_asym = remaining * 0.4   # and some to symmetry

        total_score = (
            w_int * intensity_penalty
            + w_pos * position_penalty
            + w_width * width_penalty
            + w_asym * asym_penalty
        )
        # Add multiple‑peak penalty if this composition is in the set
        comp_key = (int(row.get("composition_number", -1)), int(row.get("iteration", 0)))
        if comp_key in multiple_peaks_set:
            total_score += multiple_peak_penalty

        total_score = min(total_score, max_score)
        stb_scores.append(total_score)
    
    return np.array(stb_scores)
