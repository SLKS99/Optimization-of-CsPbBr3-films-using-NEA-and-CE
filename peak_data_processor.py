"""
Peak data processing module for pre-fitted peak data
Reads peak data directly from CSV files (peak fitting already done)
Extracts peak information and creates intensity and peak datasets
"""

import os
import re
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict


def extract_iteration_number(filename: str) -> Optional[int]:
    """
    Extract iteration number from filename.
    Supports patterns like:
    - "iteration 1 peak data.csv"
    - "iteration1_peak_data.csv"
    - "iteration_1_peak.csv"
    - "iter1.csv"
    
    Parameters:
    -----------
    filename : str
        Filename to extract iteration number from
        
    Returns:
    --------
    Optional[int]
        Iteration number if found, None otherwise
    """
    # Remove file extension
    name = os.path.splitext(filename)[0].lower()
    
    # Try various patterns
    patterns = [
        r'iteration\s*(\d+)',  # "iteration 1" or "iteration1"
        r'iter\s*(\d+)',       # "iter 1" or "iter1"
        r'_(\d+)_',            # "_1_" in middle
        r'^(\d+)',             # Starts with number
    ]
    
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    
    # If no pattern matches, check if filename is just a number
    try:
        return int(name)
    except ValueError:
        return None


def find_peak_data_files(peak_data_dir: str) -> List[Tuple[str, int]]:
    """
    Find all peak data CSV files in the peak_data directory and extract iteration numbers.
    
    Parameters:
    -----------
    peak_data_dir : str
        Directory containing peak data files
        
    Returns:
    --------
    List[Tuple[str, int]]
        List of (filepath, iteration_number) tuples, sorted by iteration number
    """
    if not os.path.exists(peak_data_dir):
        return []
    
    peak_files = []
    for filename in os.listdir(peak_data_dir):
        if filename.lower().endswith('.csv'):
            filepath = os.path.join(peak_data_dir, filename)
            iteration = extract_iteration_number(filename)
            
            if iteration is not None:
                peak_files.append((filepath, iteration))
            else:
                # If no iteration number found, assume it's iteration 0 (initial)
                peak_files.append((filepath, 0))
    
    # Sort by iteration number
    peak_files.sort(key=lambda x: x[1])
    return peak_files


def load_peak_data(file_path: str, wells_to_ignore: List[str] = None) -> pd.DataFrame:
    """
    Load peak data from CSV file (peak fitting already done).
    Reads peak information directly from CSV columns.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing peak data (already fitted)
    wells_to_ignore : List[str], optional
        List of well names to exclude from analysis
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with peak data (no fitting performed, just reading CSV)
    """
    if wells_to_ignore is None:
        wells_to_ignore = []
    
    df = pd.read_csv(file_path)
    
    # Filter out ignored wells
    if wells_to_ignore:
        df = df[~df['Well'].isin(wells_to_ignore)]
    
    return df


def extract_initial_final_peaks(
    peak_df: pd.DataFrame,
    initial_read: int = 1,
    final_read: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract initial and final read data from peak dataframe.
    
    Parameters:
    -----------
    peak_df : pd.DataFrame
        DataFrame with peak fitting data
    initial_read : int
        Read number for initial measurement (default: 1)
    final_read : int
        Read number for final measurement (default: 2)
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Initial and final read dataframes
    """
    initial_df = peak_df[peak_df['Read'] == initial_read].copy()
    final_df = peak_df[peak_df['Read'] == final_read].copy()
    
    return initial_df, final_df


def get_primary_peak_data(row: pd.Series) -> Tuple[float, float]:
    """
    Extract primary (highest intensity) peak position and intensity from a row.
    
    Parameters:
    -----------
    row : pd.Series
        Row from peak dataframe
        
    Returns:
    --------
    Tuple[float, float]
        Peak position (wavelength) and intensity
    """
    # Get Peak 1 data (primary peak)
    peak_pos = row.get('Peak_1_Wavelength', np.nan)
    peak_int = row.get('Peak_1_Intensity', np.nan)
    
    # If Peak 1 is missing, try to find the highest intensity peak
    if pd.isna(peak_pos) or peak_int == 0:
        max_intensity = 0
        best_peak_pos = np.nan
        
        for i in range(1, 6):  # Check peaks 1-5
            peak_int_col = f'Peak_{i}_Intensity'
            peak_pos_col = f'Peak_{i}_Wavelength'
            
            if peak_int_col in row and peak_pos_col in row:
                intensity = row[peak_int_col]
                position = row[peak_pos_col]
                
                if not pd.isna(intensity) and not pd.isna(position) and intensity > max_intensity:
                    max_intensity = intensity
                    best_peak_pos = position
        
        peak_pos = best_peak_pos if not pd.isna(best_peak_pos) else np.nan
        peak_int = max_intensity if max_intensity > 0 else np.nan
    
    return peak_pos, peak_int


def detect_composition_columns(composition_df: pd.DataFrame) -> List[str]:
    """
    Automatically detect composition column names from composition dataframe index.
    Returns Cs, NEA, CE (excluding PbBr which is fixed).
    
    Parameters:
    -----------
    composition_df : pd.DataFrame
        Composition dataframe with component names as index
        
    Returns:
    --------
    List[str]
        List of component names (Cs, NEA, CE) - excludes PbBr
    """
    # Get component names from index - check all rows including potentially empty ones
    component_names = []
    for idx in composition_df.index:
        if pd.notna(idx):
            name = str(idx).strip()
            if name and name.lower() != 'nan' and name.lower() != '':
                component_names.append(name)
    
    # Also check if there are more rows in the raw file that might have been skipped
    # Re-read the file to check for empty rows that might contain CE
    try:
        df_check = pd.read_excel(composition_file_path, header=None)
        print(f"  Checking raw file: found {len(df_check)} total rows")
        # Check all rows in first column for component names
        for i in range(len(df_check)):
            val = df_check.iloc[i, 0]
            if pd.notna(val):
                name = str(val).strip()
                if name and name.lower() not in ['nan', '']:
                    # Check if this name is already in component_names
                    if name not in component_names:
                        # Check if it looks like a component name
                        if any(keyword in name.lower() for keyword in ['con', 'cs', 'nea', 'ce', 'crown', 'pb']):
                            component_names.append(name)
                            print(f"  Found additional component: {name}")
    except:
        pass  # If we can't re-read, just use what we have
    
    # Filter out PbBr (fixed component, not optimized)
    component_names = [name for name in component_names if name.lower() not in ['pbbr', 'pb_br', 'pb-br', 'pb br']]
    
    print(f"  Found components in file: {component_names}")
    
    # Try to find Cs, NEA, CE by name (flexible matching)
    cs_candidates = [name for name in component_names if 'cs' in name.lower() and 'con' in name.lower()]
    if not cs_candidates:
        cs_candidates = [name for name in component_names if 'cs' in name.lower()]
    
    nea_candidates = [name for name in component_names if 'nea' in name.lower() and 'con' in name.lower()]
    if not nea_candidates:
        nea_candidates = [name for name in component_names if 'nea' in name.lower()]
    
    ce_candidates = [name for name in component_names if ('ce' in name.lower() or 'crown' in name.lower()) and 'con' in name.lower()]
    if not ce_candidates:
        # Also check for just "CE" (without "Con")
        ce_candidates = [name for name in component_names if name.lower() == 'ce' or 'crown' in name.lower()]
    if not ce_candidates:
        ce_candidates = [name for name in component_names if 'ce' in name.lower()]
    
    # If we found all three, return them
    if len(cs_candidates) > 0 and len(nea_candidates) > 0 and len(ce_candidates) > 0:
        result = [cs_candidates[0], nea_candidates[0], ce_candidates[0]]
        print(f"  Detected: Cs={result[0]}, NEA={result[1]}, CE={result[2]}")
        return result
    
    # If we only found 2, check what's missing
    if len(cs_candidates) > 0 and len(nea_candidates) > 0 and len(ce_candidates) == 0:
        raise ValueError(
            f"CE (Crown Ether) component not found in composition file!\n"
            f"Found components: {component_names}\n"
            f"Please add a row named 'CE Con' or 'Crown Ether Con' to your Excel file."
        )
    
    # If we have at least 3 components, return first 3
    if len(component_names) >= 3:
        print(f"  Using first 3 components: {component_names[:3]}")
        return component_names[:3]
    
    # Otherwise, raise error with helpful message
    raise ValueError(
        f"Composition file must have at least 3 components (Cs, NEA, CE) in index.\n"
        f"Found: {component_names}\n"
        f"Expected row names: 'Cs Con', 'NEA Con', 'CE Con' (or similar variations)"
    )


def create_intensity_dataset(
    initial_df: pd.DataFrame,
    final_df: pd.DataFrame,
    composition_df: pd.DataFrame,
    composition_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create intensity dataset from peak data.
    
    Parameters:
    -----------
    initial_df : pd.DataFrame
        Initial read peak data
    final_df : pd.DataFrame
        Final read peak data
    composition_df : pd.DataFrame
        Composition data with component names as index, well names as columns
    composition_cols : List[str], optional
        Names of composition columns to use. If None, auto-detects from composition_df index.
        
    Returns:
    --------
    pd.DataFrame
        Intensity dataset with columns: composition components, 'Intensity initial'
    """
    # Auto-detect composition columns if not provided
    if composition_cols is None:
        composition_cols = detect_composition_columns(composition_df)
        print(f"  Auto-detected composition columns: {composition_cols}")
    else:
        print(f"  Using provided composition columns: {composition_cols}")
    
    intensity_data = []
    
    # Sort wells to ensure consistent ordering
    wells = sorted(initial_df['Well'].unique())
    
    # Match wells between peak data and composition data
    for well in wells:
        initial_row = initial_df[initial_df['Well'] == well].iloc[0]
        final_row = final_df[final_df['Well'] == well].iloc[0] if len(final_df[final_df['Well'] == well]) > 0 else None
        
        # Get initial and final intensities (matching notebook workflow)
        # Extract all peaks and get max intensity
        pos_init, int_init = extract_all_peaks(initial_row)
        pos_init[np.isnan(pos_init)] = 0
        int_init[np.isnan(int_init)] = 0
        
        if final_row is not None:
            pos_fin, int_fin = extract_all_peaks(final_row)
            pos_fin[np.isnan(pos_fin)] = 0
            int_fin[np.isnan(int_fin)] = 0
        else:
            int_fin = np.zeros(5)
        
        # Reduce to max peak and get max intensity
        reduced_int_init = reduce_to_max_peak(int_init.reshape(1, -1))[0]
        reduced_int_fin = reduce_to_max_peak(int_fin.reshape(1, -1))[0]
        
        initial_intensity = np.max(reduced_int_init) if np.any(reduced_int_init > 0) else 0.0
        final_intensity = np.max(reduced_int_fin) if np.any(reduced_int_fin > 0) else 0.0
        
        # Get composition values
        comp_values = []
        if well in composition_df.columns:
            comp = composition_df[well]
            # Extract values for each component name from composition_cols
            for col_name in composition_cols:
                # Find matching index in composition dataframe
                matching_idx = None
                for idx in composition_df.index:
                    if str(idx).strip() == col_name.strip():
                        matching_idx = idx
                        break
                
                if matching_idx is not None:
                    # Use exact match - values are already in 0.1-1.0 range, no division needed
                    comp_values.append(composition_df.loc[matching_idx, well])
                else:
                    # Fallback: use position in index (if composition_cols order matches index order)
                    try:
                        idx_pos = list(composition_df.index).index(col_name)
                        comp_values.append(comp.iloc[idx_pos] if len(comp) > idx_pos else 0)
                    except (ValueError, IndexError):
                        # Last resort: use position based on composition_cols order
                        idx_pos = composition_cols.index(col_name) if col_name in composition_cols else 0
                        comp_values.append(comp.iloc[idx_pos] if len(comp) > idx_pos else 0)
        else:
            # Well not found in composition data
            comp_values = [0] * len(composition_cols)
        
        # Create row data - use all composition columns dynamically
        row_data = {
            'Intensity initial': initial_intensity,
            'Intensity final': final_intensity  # Include final intensity (matching notebook)
        }
        # Add composition values for each detected column
        for i, col_name in enumerate(composition_cols):
            row_data[col_name] = comp_values[i] if i < len(comp_values) else 0
        
        intensity_data.append(row_data)
    
    df_intensity = pd.DataFrame(intensity_data)
    print(f"  Created DataFrame with columns: {list(df_intensity.columns)}")
    print(f"  Expected composition columns: {composition_cols}")
    return df_intensity


def extract_all_peaks(row: pd.Series, max_peaks: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract all peak positions and intensities from a row.
    Reads directly from CSV columns - no peak fitting performed.
    
    Parameters:
    -----------
    row : pd.Series
        Row from peak dataframe (already contains fitted peak data)
    max_peaks : int
        Maximum number of peaks to extract (default: 5)
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Arrays of peak positions and intensities (read from CSV columns)
    """
    positions = []
    intensities = []
    
    for i in range(1, max_peaks + 1):
        pos_col = f'Peak_{i}_Wavelength'
        int_col = f'Peak_{i}_Intensity'
        
        if pos_col in row and int_col in row:
            pos = row[pos_col]
            intensity = row[int_col]
            
            # Only add if both are valid (not NaN and not empty)
            if pd.notna(pos) and pd.notna(intensity) and pos != 0 and intensity != 0:
                positions.append(float(pos))
                intensities.append(float(intensity))
            else:
                positions.append(0.0)
                intensities.append(0.0)
        else:
            positions.append(0.0)
            intensities.append(0.0)
    
    return np.array(positions), np.array(intensities)


def extract_peak_shape_features(
    row: pd.Series,
    max_peaks: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract peak shape features (FWHM and left/right areas) from a row.
    Reads directly from CSV columns.

    Parameters
    ----------
    row : pd.Series
        Row from peak dataframe (already contains fitted peak data)
    max_peaks : int
        Maximum number of peaks to extract (default: 5)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Arrays of FWHM, left areas, right areas for each peak.
        Missing values are filled with 0.0.
    """
    fwhm_vals = []
    left_areas = []
    right_areas = []

    for i in range(1, max_peaks + 1):
        fwhm_col = f"Peak_{i}_FWHM"
        left_col = f"Peak_{i}_LeftArea"
        right_col = f"Peak_{i}_RightArea"

        fwhm = row.get(fwhm_col, 0.0)
        left = row.get(left_col, 0.0)
        right = row.get(right_col, 0.0)

        if pd.isna(fwhm):
            fwhm = 0.0
        if pd.isna(left):
            left = 0.0
        if pd.isna(right):
            right = 0.0

        fwhm_vals.append(float(fwhm))
        left_areas.append(float(left))
        right_areas.append(float(right))

    return np.array(fwhm_vals), np.array(left_areas), np.array(right_areas)

def reduce_to_max_peak(peaks_array: np.ndarray) -> np.ndarray:
    """
    Reduce multiple peaks to only the maximum peak per composition.
    All other peaks are set to zero.
    
    Parameters:
    -----------
    peaks_array : np.ndarray
        Array of peaks (N_compositions x N_peaks)
        
    Returns:
    --------
    np.ndarray
        Reduced array with only max peak per composition
    """
    reduced = peaks_array.copy()
    
    for i in range(reduced.shape[0]):
        row = reduced[i, :]
        non_zero_indices = np.nonzero(row)[0]
        
        if len(non_zero_indices) > 1:
            # Find index of maximum peak
            max_peak_index = np.argmax(row[non_zero_indices])
            max_idx = non_zero_indices[max_peak_index]
            
            # Set all peaks to zero except the maximum
            for j in non_zero_indices:
                if j != max_idx:
                    reduced[i, j] = 0
    
    return reduced


def create_peak_dataset(
    initial_df: pd.DataFrame,
    final_df: pd.DataFrame,
    composition_number_start: int = 1,
    iteration: int = 0
) -> pd.DataFrame:
    """
    Create combined peak dataset with initial and final peak positions/intensities.
    Reads peak data directly from CSV (no fitting - data already fitted).
    Matches notebook workflow: extracts all peaks from CSV, reduces to max peak.
    
    Parameters:
    -----------
    initial_df : pd.DataFrame
        Initial read peak data (from CSV, already fitted)
    final_df : pd.DataFrame
        Final read peak data (from CSV, already fitted)
    composition_number_start : int
        Starting composition number (default: 1)
    iteration : int
        Iteration number (default: 0)
        
    Returns:
    --------
    pd.DataFrame
        Peak dataset with columns: initial_peak_positions, final_peak_positions,
        initial_peak_intensities, final_peak_intensities, composition_number, iteration
    """
    # Sort wells to ensure consistent ordering
    wells = sorted(initial_df['Well'].unique())
    
    # Extract all peaks from CSV columns (Peak_1_Wavelength, Peak_1_Intensity, etc.)
    # No peak fitting - just reading the data
    peaks_pos_initial_list = []
    peaks_int_initial_list = []
    peaks_fwhm_initial_list = []
    peaks_leftarea_initial_list = []
    peaks_rightarea_initial_list = []

    peaks_pos_final_list = []
    peaks_int_final_list = []
    peaks_fwhm_final_list = []
    peaks_leftarea_final_list = []
    peaks_rightarea_final_list = []
    
    for well in wells:
        initial_row = initial_df[initial_df['Well'] == well].iloc[0]
        final_row = final_df[final_df['Well'] == well].iloc[0] if len(final_df[final_df['Well'] == well]) > 0 else None
        
        # Read peak data directly from CSV columns
        pos_init, int_init = extract_all_peaks(initial_row)
        fwhm_init, left_init, right_init = extract_peak_shape_features(initial_row)

        peaks_pos_initial_list.append(pos_init)
        peaks_int_initial_list.append(int_init)
        peaks_fwhm_initial_list.append(fwhm_init)
        peaks_leftarea_initial_list.append(left_init)
        peaks_rightarea_initial_list.append(right_init)
        
        if final_row is not None:
            pos_fin, int_fin = extract_all_peaks(final_row)
            fwhm_fin, left_fin, right_fin = extract_peak_shape_features(final_row)
        else:
            pos_fin = np.zeros(5)
            int_fin = np.zeros(5)
            fwhm_fin = np.zeros(5)
            left_fin = np.zeros(5)
            right_fin = np.zeros(5)

        peaks_pos_final_list.append(pos_fin)
        peaks_int_final_list.append(int_fin)
        peaks_fwhm_final_list.append(fwhm_fin)
        peaks_leftarea_final_list.append(left_fin)
        peaks_rightarea_final_list.append(right_fin)
    
    # Convert to arrays
    peaks_pos_initial = np.array(peaks_pos_initial_list)
    peaks_int_initial = np.array(peaks_int_initial_list)
    peaks_fwhm_initial = np.array(peaks_fwhm_initial_list)
    peaks_leftarea_initial = np.array(peaks_leftarea_initial_list)
    peaks_rightarea_initial = np.array(peaks_rightarea_initial_list)

    peaks_pos_final = np.array(peaks_pos_final_list)
    peaks_int_final = np.array(peaks_int_final_list)
    peaks_fwhm_final = np.array(peaks_fwhm_final_list)
    peaks_leftarea_final = np.array(peaks_leftarea_final_list)
    peaks_rightarea_final = np.array(peaks_rightarea_final_list)
    
    # Replace NaN with 0
    for arr in [
        peaks_pos_initial,
        peaks_int_initial,
        peaks_fwhm_initial,
        peaks_leftarea_initial,
        peaks_rightarea_initial,
        peaks_pos_final,
        peaks_int_final,
        peaks_fwhm_final,
        peaks_leftarea_final,
        peaks_rightarea_final,
    ]:
        arr[np.isnan(arr)] = 0
    
    # Reduce to a single dominant peak per composition (matching notebook workflow)
    reduced_peaks_pos_initial = reduce_to_max_peak(peaks_pos_initial)
    reduced_peaks_pos_final = reduce_to_max_peak(peaks_pos_final)
    reduced_peaks_int_initial = reduce_to_max_peak(peaks_int_initial)
    reduced_peaks_int_final = reduce_to_max_peak(peaks_int_final)

    reduced_fwhm_initial = reduce_to_max_peak(peaks_fwhm_initial)
    reduced_leftarea_initial = reduce_to_max_peak(peaks_leftarea_initial)
    reduced_rightarea_initial = reduce_to_max_peak(peaks_rightarea_initial)
    
    # Extract max peak values (matching notebook: np.max(..., axis=1))
    peak_positions_int = np.max(reduced_peaks_pos_initial, axis=1)
    peak_positions_fin = np.max(reduced_peaks_pos_final, axis=1)
    peak_intensities_int = np.max(reduced_peaks_int_initial, axis=1)
    peak_intensities_fin = np.max(reduced_peaks_int_final, axis=1)

    peak_fwhm_int = np.max(reduced_fwhm_initial, axis=1)
    left_area_int = np.max(reduced_leftarea_initial, axis=1)
    right_area_int = np.max(reduced_rightarea_initial, axis=1)
    
    # Create dataframe
    peak_data = []
    for idx in range(len(wells)):
        peak_data.append({
            'initial_peak_positions': peak_positions_int[idx] if peak_positions_int[idx] > 0 else 0.0,
            'final_peak_positions': peak_positions_fin[idx] if peak_positions_fin[idx] > 0 else 0.0,
            'initial_peak_intensities': peak_intensities_int[idx] if peak_intensities_int[idx] > 0 else 0.0,
            'final_peak_intensities': peak_intensities_fin[idx] if peak_intensities_fin[idx] > 0 else 0.0,
            # Shape features for instability / quality scoring
            'initial_peak_fwhm': peak_fwhm_int[idx] if peak_fwhm_int[idx] > 0 else 0.0,
            'initial_left_area': left_area_int[idx] if left_area_int[idx] > 0 else 0.0,
            'initial_right_area': right_area_int[idx] if right_area_int[idx] > 0 else 0.0,
            'composition_number': composition_number_start + idx,
            'iteration': iteration
        })
    
    df_peaks = pd.DataFrame(peak_data)
    return df_peaks


def identify_multiple_peaks(
    initial_df: pd.DataFrame,
    final_df: pd.DataFrame,
    composition_number_start: int = 1,
    iteration: int = 0,
    min_peaks: int = 2
) -> np.ndarray:
    """
    Identify compositions with multiple peaks.
    Uses Total_Quality_Peaks column from CSV or counts non-zero peaks.
    Matches notebook workflow: identifies multiple peaks BEFORE reducing to max peak.
    
    Parameters:
    -----------
    initial_df : pd.DataFrame
        Initial read peak data (from CSV, already fitted)
    final_df : pd.DataFrame
        Final read peak data (from CSV, already fitted)
    composition_number_start : int
        Starting composition number
    iteration : int
        Iteration number
    min_peaks : int
        Minimum number of peaks to consider as "multiple" (default: 2)
        
    Returns:
    --------
    np.ndarray
        Structured array with dtype [('composition_number', int), ('iteration', int)]
        containing compositions with multiple peaks
    """
    # Sort wells to ensure consistent ordering
    wells = sorted(initial_df['Well'].unique())
    
    # Method 1: Use Total_Quality_Peaks column if available (fastest)
    multiple_peaks_list = []
    
    for idx, well in enumerate(wells):
        initial_row = initial_df[initial_df['Well'] == well].iloc[0]
        final_row = final_df[final_df['Well'] == well].iloc[0] if len(final_df[final_df['Well'] == well]) > 0 else None
        
        # Check Total_Quality_Peaks column (if available)
        initial_peaks = initial_row.get('Total_Quality_Peaks', None)
        if pd.notna(initial_peaks) and initial_peaks >= min_peaks:
            multiple_peaks_list.append((composition_number_start + idx, iteration))
            continue
        
        if final_row is not None:
            final_peaks = final_row.get('Total_Quality_Peaks', None)
            if pd.notna(final_peaks) and final_peaks >= min_peaks:
                multiple_peaks_list.append((composition_number_start + idx, iteration))
                continue
        
        # Method 2: Count non-zero peaks (if Total_Quality_Peaks not available)
        pos_init, int_init = extract_all_peaks(initial_row)
        pos_init[np.isnan(pos_init)] = 0
        int_init[np.isnan(int_init)] = 0
        
        if final_row is not None:
            pos_fin, int_fin = extract_all_peaks(final_row)
            pos_fin[np.isnan(pos_fin)] = 0
            int_fin[np.isnan(int_fin)] = 0
        else:
            pos_fin = np.zeros(5)
            int_fin = np.zeros(5)
        
        # Count non-zero peaks
        num_peaks_init = np.sum((pos_init != 0) | (int_init != 0))
        num_peaks_fin = np.sum((pos_fin != 0) | (int_fin != 0))
        
        if num_peaks_init >= min_peaks or num_peaks_fin >= min_peaks:
            multiple_peaks_list.append((composition_number_start + idx, iteration))
    
    # Create structured array
    if multiple_peaks_list:
        dtype = [('composition_number', int), ('iteration', int)]
        multiple_peaks_array = np.array(multiple_peaks_list, dtype=dtype)
    else:
        dtype = [('composition_number', int), ('iteration', int)]
        multiple_peaks_array = np.array([], dtype=dtype)
    
    return multiple_peaks_array


def process_peak_data_file(
    peak_file_path: str,
    composition_file_path: str,
    output_dir: str,
    wells_to_ignore: List[str] = None,
    initial_read: int = 1,
    final_read: int = 2,
    composition_cols: Optional[List[str]] = None,
    iteration: int = 0,
    auto_detect_composition_cols: bool = True
) -> Tuple[str, str, str]:
    """
    Process a peak data file and create all required datasets.
    
    Parameters:
    -----------
    peak_file_path : str
        Path to peak fitting CSV file
    composition_file_path : str
        Path to composition CSV file
    output_dir : str
        Directory to save output files
    wells_to_ignore : List[str], optional
        Wells to exclude
    initial_read : int
        Read number for initial measurement
    final_read : int
        Read number for final measurement
    composition_cols : List[str], optional
        Composition column names (default: ['CsPbI4', 'PEA2PbI4'])
    iteration : int
        Iteration number (default: 0)
        
    Returns:
    --------
    Tuple[str, str, str]
        Paths to: intensity dataset, peak dataset, multiple peaks file
    """
    # Don't set default - let auto-detection handle it if auto_detect_composition_cols is True
    # if composition_cols is None:
    #     composition_cols = ['CsPbI4', 'PEA2PbI4']
    
    if wells_to_ignore is None:
        wells_to_ignore = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"  Loading peak data from: {os.path.basename(peak_file_path)}")
    peak_df = load_peak_data(peak_file_path, wells_to_ignore)
    
    print(f"  Loading composition data from: {os.path.basename(composition_file_path)}")
    # Load based on file extension
    if composition_file_path.lower().endswith(('.xlsx', '.xls')):
        try:
            # First read without index_col to see all rows
            df_raw = pd.read_excel(composition_file_path, header=None)
            print(f"  Raw Excel file has {len(df_raw)} rows")
            
            # Find the header row (usually row 0 with column names)
            # Then read with index_col=0, skipping the header
            composition_df = pd.read_excel(composition_file_path, index_col=0)
            
            # Debug: print what we found
            print(f"  After reading with index_col=0, found {len(composition_df.index)} rows")
            print(f"  Row names: {list(composition_df.index)}")
            
        except ImportError:
            raise ImportError(
                "Excel file support requires openpyxl. Install with: pip install openpyxl"
            )
    else:
        composition_df = pd.read_csv(composition_file_path, index_col=0)
    
    # Auto-detect composition columns if enabled and not provided
    if auto_detect_composition_cols and composition_cols is None:
        composition_cols = detect_composition_columns(composition_df)
        print(f"  Auto-detected composition columns from file: {composition_cols}")
    elif composition_cols is None:
        raise ValueError(
            "composition_cols must be provided if auto_detect_composition_cols is False"
        )
    
    # Extract initial and final reads
    initial_df, final_df = extract_initial_final_peaks(peak_df, initial_read, final_read)
    
    # Create datasets
    df_intensity = create_intensity_dataset(initial_df, final_df, composition_df, composition_cols)
    df_peaks = create_peak_dataset(initial_df, final_df, iteration=iteration)
    multiple_peaks = identify_multiple_peaks(initial_df, final_df, iteration=iteration)
    
    # Save files based on iteration
    if iteration == 0:
        intensity_file = os.path.join(output_dir, 'df_intensity_initial.csv')
        peak_file = os.path.join(output_dir, 'df_combined_initial.csv')
        multiple_peaks_file = os.path.join(output_dir, 'multiple_peaks_initial.npy')
    else:
        intensity_file = os.path.join(output_dir, f'df_intensity_{iteration}.csv')
        peak_file = os.path.join(output_dir, f'df_combined_{iteration}.csv')
        multiple_peaks_file = os.path.join(output_dir, f'multiple_peaks_{iteration}.npy')
    
    df_intensity.to_csv(intensity_file, index=False)
    df_peaks.to_csv(peak_file, index=False)
    np.save(multiple_peaks_file, multiple_peaks)
    
    print(f"  Created: {os.path.basename(intensity_file)}")
    print(f"  Created: {os.path.basename(peak_file)}")
    print(f"  Created: {os.path.basename(multiple_peaks_file)} ({len(multiple_peaks)} multiple peaks)")
    
    return intensity_file, peak_file, multiple_peaks_file


def find_composition_file(compositions_dir: str, iteration: int = None) -> Optional[str]:
    """
    Find composition file, optionally matching iteration number.
    
    Parameters:
    -----------
    compositions_dir : str
        Directory containing composition files
    iteration : int, optional
        Iteration number to match (e.g., find "Iteration 1 compositions.xlsx")
        
    Returns:
    --------
    Optional[str]
        Path to composition file, or None if not found
    """
    if not os.path.exists(compositions_dir):
        return None
    
    # If iteration specified, try to find matching file
    if iteration is not None:
        patterns = [
            f'*iteration {iteration}*',
            f'*iteration{iteration}*',
            f'*iteration_{iteration}*',
            f'*iter {iteration}*',
            f'*iter{iteration}*',
        ]
        
        import fnmatch
        for pattern in patterns:
            for filename in os.listdir(compositions_dir):
                if fnmatch.fnmatch(filename.lower(), pattern.lower()):
                    filepath = os.path.join(compositions_dir, filename)
                    if filename.lower().endswith(('.csv', '.xlsx', '.xls')):
                        return filepath
    
    # Otherwise, find any CSV or Excel file
    for filename in os.listdir(compositions_dir):
        if filename.lower().endswith(('.csv', '.xlsx', '.xls')):
            return os.path.join(compositions_dir, filename)
    
    return None


def process_all_peak_data_files(
    peak_data_dir: str,
    compositions_dir: str,
    output_base_dir: str,
    wells_to_ignore: List[str] = None,
    initial_read: int = 1,
    final_read: int = 2,
    composition_cols: Optional[List[str]] = None,
    composition_file_name: str = None,
    match_composition_to_iteration: bool = True,
    auto_detect_composition_cols: bool = True
) -> Dict[int, Tuple[str, str, str]]:
    """
    Process all peak data files in the peak_data directory.
    
    Parameters:
    -----------
    peak_data_dir : str
        Directory containing peak data CSV files
    compositions_dir : str
        Directory containing composition CSV files
    output_base_dir : str
        Base directory for output files
    wells_to_ignore : List[str], optional
        Wells to exclude
    initial_read : int
        Read number for initial measurement
    final_read : int
        Read number for final measurement
    composition_cols : List[str], optional
        Composition column names
    composition_file_name : str
        Name of composition file (default: "compositions.csv")
        
    Returns:
    --------
    Dict[int, Tuple[str, str, str]]
        Dictionary mapping iteration numbers to (intensity_file, peak_file, multiple_peaks_file)
    """
    # Don't set default - let auto-detection handle it if auto_detect_composition_cols is True
    # if composition_cols is None:
    #     composition_cols = ['CsPbI4', 'PEA2PbI4']
    
    if wells_to_ignore is None:
        wells_to_ignore = []
    
    # Find all peak data files
    peak_files = find_peak_data_files(peak_data_dir)
    
    if not peak_files:
        print(f"No peak data files found in {peak_data_dir}")
        return {}
    
    print(f"\nFound {len(peak_files)} peak data file(s):")
    for filepath, iteration in peak_files:
        print(f"  Iteration {iteration}: {os.path.basename(filepath)}")
    
    # Determine output directories
    initial_datasets_dir = os.path.join(output_base_dir, 'initial_datasets')
    next_iterations_dir = os.path.join(output_base_dir, 'next_iterations')
    multiple_peaks_dir = os.path.join(output_base_dir, 'multiple_peaks')
    combined_set_dir = os.path.join(output_base_dir, 'combined_set')
    
    os.makedirs(initial_datasets_dir, exist_ok=True)
    os.makedirs(next_iterations_dir, exist_ok=True)
    os.makedirs(multiple_peaks_dir, exist_ok=True)
    os.makedirs(combined_set_dir, exist_ok=True)
    
    # Find composition file(s)
    # If match_composition_to_iteration is True, try to match composition files to iterations
    # Otherwise, use a single composition file for all iterations
    
    # Process each peak data file
    processed_files = {}
    
    print("\nProcessing peak data files...")
    for filepath, iteration in peak_files:
        # Treat iteration 1 as initial dataset (iteration 0)
        if iteration == 1:
            actual_iteration = 0
            print(f"\nProcessing iteration {iteration} (treating as initial dataset)...")
        else:
            actual_iteration = iteration
            print(f"\nProcessing iteration {iteration}...")
        
        # Find composition file for this iteration
        # Use original iteration number for file matching
        if match_composition_to_iteration:
            composition_file_path = find_composition_file(compositions_dir, iteration)
        else:
            # Use default composition file or find any composition file
            if composition_file_name:
                composition_file_path = os.path.join(compositions_dir, composition_file_name)
                if not os.path.exists(composition_file_path):
                    composition_file_path = find_composition_file(compositions_dir)
            else:
                composition_file_path = find_composition_file(compositions_dir)
        
        if not composition_file_path or not os.path.exists(composition_file_path):
            raise FileNotFoundError(
                f"No composition file found for iteration {iteration} in {compositions_dir}"
            )
        
        print(f"  Using composition file: {os.path.basename(composition_file_path)}")
        
        if actual_iteration == 0:
            output_dir = initial_datasets_dir
        else:
            output_dir = next_iterations_dir
        
        try:
            intensity_file, peak_file, multiple_peaks_file = process_peak_data_file(
                filepath,
                composition_file_path,
                output_dir,
                wells_to_ignore,
                initial_read,
                final_read,
                composition_cols,
                actual_iteration,  # Use actual_iteration (0 for iteration 1)
                auto_detect_composition_cols
            )
            
            # Also save to combined_set and multiple_peaks directories for iterations > 0
            if actual_iteration > 0:
                # Copy peak file to combined_set
                import shutil
                combined_peak_file = os.path.join(combined_set_dir, f'df_combined_{actual_iteration}.csv')
                shutil.copy2(peak_file, combined_peak_file)
                
                # Copy multiple peaks file
                combined_multiple_peaks_file = os.path.join(multiple_peaks_dir, f'multiple_peaks_{actual_iteration}.npy')
                shutil.copy2(multiple_peaks_file, combined_multiple_peaks_file)
            
            # Store using original iteration number for reference
            processed_files[iteration] = (intensity_file, peak_file, multiple_peaks_file)
            
        except Exception as e:
            print(f"  Error processing iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nSuccessfully processed {len(processed_files)} file(s)")
    return processed_files
