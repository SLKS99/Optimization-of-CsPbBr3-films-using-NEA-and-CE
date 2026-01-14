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
    target_wavelength: float,
    wavelength_tolerance: float,
    degradation_weight: float,
    position_weight: float
) -> np.ndarray:
    """
    Simplified instability score focusing on intensity change and peak position.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: initial_peak_positions, final_peak_positions,
        initial_peak_intensities, final_peak_intensities
    target_wavelength : float
        Target wavelength for peak analysis
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
    max_score = 1
    stb_scores = []
    
    for _, row in df.iterrows():
        peak_positions_int = row['initial_peak_positions']
        peak_positions_fin = row['final_peak_positions']
        peak_intensities_int = row['initial_peak_intensities']
        peak_intensities_fin = row['final_peak_intensities']
        
        # Intensity change component
        intensity_change = np.abs(peak_intensities_fin - peak_intensities_int) / max(
            peak_intensities_int, 1e-10
        )
        intensity_score = min(intensity_change, max_score) * degradation_weight
        
        # Position deviation component
        position_deviation = min(
            abs(peak_positions_fin - target_wavelength),
            abs(peak_positions_int - target_wavelength)
        )
        position_score = (
            0 if position_deviation <= wavelength_tolerance
            else (position_deviation / target_wavelength) * position_weight
        )
        position_score = min(position_score, max_score)
        
        # Total score
        total_score = intensity_score + position_score
        total_score = min(total_score, 1)
        stb_scores.append(total_score)
    
    return np.array(stb_scores)
