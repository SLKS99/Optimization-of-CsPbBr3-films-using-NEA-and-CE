"""
Data loading and preprocessing module
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional


def load_luminescence_data(file_path: str, wells_to_ignore: List[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Load and parse luminescence data from CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing luminescence data
    wells_to_ignore : List[str], optional
        List of well names to exclude from analysis
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary with keys 'Read 1', 'Read 2', etc. containing dataframes
    """
    if wells_to_ignore is None:
        wells_to_ignore = []
    
    raw_data = pd.read_csv(file_path, header=None)
    raw_data = raw_data.replace('OVRFLW', np.NaN)
    
    # Generate list of cells (wells)
    cells = []
    for i in range(1, 9):
        for j in range(1, 13):
            cells.append(chr(64 + i) + str(j))
    
    for well in wells_to_ignore:
        if well in cells:
            cells.remove(well)
    
    # Find rows with measurement data
    rows = []
    
    # Find all read rows by searching for "Read X:EM Spectrum" pattern
    for i in range(1, 100):  # Search up to read 100
        read_row = raw_data[raw_data[raw_data.columns[0]] == f'Read {i}:EM Spectrum'].index.tolist()
        if read_row:
            rows += read_row
    
    # Add Results row as final marker
    results_rows = raw_data[raw_data[raw_data.columns[0]] == 'Results'].index.tolist()
    if results_rows:
        rows += results_rows
    else:
        # If no Results row, use the last row as marker
        rows.append(len(raw_data))
    
    if not rows:
        raise ValueError("Could not find any measurement data in the file")
    
    # Create dictionary of dataframes
    data_dict = {}
    for i in range(1, len(rows)):
        read_name = f'Read {i}'
        start_idx = rows[i-1] + 2
        end_idx = rows[i] - 1
        
        if end_idx > start_idx:
            df = raw_data.iloc[start_idx:end_idx].copy()
            df = df.drop([0], axis=1)
            new_header = df.iloc[0]
            df = df[1:]
            df.columns = new_header
            
            # Remove ignored wells
            for well in wells_to_ignore:
                if well in df.columns:
                    df = df.drop(well, axis=1)
            
            # Convert to float
            df = df.astype(float)
            data_dict[read_name] = df
    
    return data_dict


def load_composition_data(file_path: str, wells_to_ignore: List[str] = None) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Load composition data and create composition arrays.
    Supports both CSV and Excel files (.xlsx, .xls).
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV or Excel file containing composition data
    wells_to_ignore : List[str], optional
        List of well names to exclude
        
    Returns:
    --------
    Tuple[pd.DataFrame, np.ndarray, List[str]]
        Composition dataframe, composition array, and list of cells with data
    """
    if wells_to_ignore is None:
        wells_to_ignore = []
    
    # Load based on file extension
    if file_path.lower().endswith(('.xlsx', '.xls')):
        try:
            composition = pd.read_excel(file_path, index_col=0)
        except ImportError:
            raise ImportError(
                "Excel file support requires openpyxl. Install with: pip install openpyxl"
            )
    else:
        composition = pd.read_csv(file_path, index_col=0)
    
    # Generate list of cells
    cells = []
    for i in range(1, 9):
        for j in range(1, 13):
            cells.append(chr(64 + i) + str(j))
    
    for well in wells_to_ignore:
        if well in cells:
            cells.remove(well)
    
    # Create composition arrays
    A, B, C = [], [], []
    cells_no_data = []
    no_data = np.zeros(len(cells), dtype=bool)
    
    for j, cell_name in enumerate(cells):
        if cell_name in composition.columns:
            comp = composition[cell_name]
            if (comp == 0.0).all():
                cells_no_data.append(cell_name)
                no_data[j] = True
            else:
                # Values are already in 0.1-1.0 range, no division needed
                A.append(comp.iloc[0])
                B.append(comp.iloc[1])
                C.append(comp.iloc[2])
        else:
            cells_no_data.append(cell_name)
            no_data[j] = True
    
    comp_array = np.column_stack((A, B, C))
    cells_with_data = [cell for cell in cells if cell not in cells_no_data]
    
    return composition, comp_array, cells_with_data


def create_luminescence_dataframe(
    data_dict: Dict[str, pd.DataFrame],
    luminescence_read_numbers: List[int],
    start_wavelength: int,
    end_wavelength: int,
    wavelength_step_size: int
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a combined luminescence dataframe from multiple reads.
    
    Parameters:
    -----------
    data_dict : Dict[str, pd.DataFrame]
        Dictionary of dataframes for each read
    luminescence_read_numbers : List[int]
        List of read numbers to include
    start_wavelength : int
        Starting wavelength
    end_wavelength : int
        Ending wavelength
    wavelength_step_size : int
        Step size for wavelength
        
    Returns:
    --------
    Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]
        Combined dataframe, luminescence vector, wavelength array, time array
    """
    luminescence_time = np.array([int(i) * 15 for i in luminescence_read_numbers])  # Assuming 15 min steps
    luminescence_wavelength = np.arange(start_wavelength, end_wavelength + wavelength_step_size, wavelength_step_size)
    
    # Combine dataframes
    luminescence_df = pd.DataFrame()
    for i in luminescence_read_numbers:
        read_name = f'Read {i}'
        if read_name in data_dict:
            temp_df = data_dict[read_name]
            luminescence_df = pd.concat([luminescence_df, temp_df], ignore_index=True)
    
    luminescence_df = luminescence_df.fillna(0.0)
    luminescence_vec = np.array(luminescence_df)
    
    return luminescence_df, luminescence_vec, luminescence_wavelength, luminescence_time
