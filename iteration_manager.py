"""
Iteration management and dataset combination
"""

import os
import re
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple


def combine_intensity_datasets(
    base_directory: str,
    initial_datasets_dir: str,
    next_iterations_dir: str,
    updated_datasets_dir: str,
    base_file: str
) -> str:
    """
    Combine initial intensity dataset with subsequent iteration datasets.
    
    Parameters:
    -----------
    base_directory : str
        Base directory path
    initial_datasets_dir : str
        Directory containing initial datasets
    next_iterations_dir : str
        Directory containing iteration files
    updated_datasets_dir : str
        Directory to save combined dataset
    base_file : str
        Name of the base file
        
    Returns:
    --------
    str
        Path to the combined dataset file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(base_directory, updated_datasets_dir), exist_ok=True)
    
    # Load initial dataset
    initial_path = os.path.join(base_directory, initial_datasets_dir, base_file)
    if not os.path.exists(initial_path):
        raise FileNotFoundError(f"Initial dataset not found: {initial_path}")
    
    combined_df = pd.read_csv(initial_path)
    
    # List iteration files
    iterations_path = os.path.join(base_directory, next_iterations_dir)
    if os.path.exists(iterations_path):
        iteration_files = [
            f for f in os.listdir(iterations_path)
            if f.startswith('df_intensity_') and f.endswith('.csv')
        ]
        
        # Sort by iteration number
        def extract_iteration(filename):
            match = re.search(r'df_intensity_(\d+)', filename)
            return int(match.group(1)) if match else 0
        
        iteration_files.sort(key=extract_iteration)
        
        # Append each iteration
        for file_name in iteration_files:
            file_path = os.path.join(iterations_path, file_name)
            df_new = pd.read_csv(file_path)
            combined_df = pd.concat([combined_df, df_new], ignore_index=True)
    
    # Save combined dataset
    combined_file_path = os.path.join(
        base_directory, updated_datasets_dir, 'intensity_combined_full.csv'
    )
    combined_df.to_csv(combined_file_path, index=False)
    print(f"Combined intensity dataset saved to {combined_file_path}")
    
    return combined_file_path


def combine_combined_peak_datasets(
    base_directory: str,
    initial_datasets_dir: str,
    combined_set_dir: str,
    updated_datasets_dir: str,
    base_file: str
) -> str:
    """
    Combine initial combined peak dataset with subsequent iteration datasets.
    
    Parameters:
    -----------
    base_directory : str
        Base directory path
    initial_datasets_dir : str
        Directory containing initial datasets
    combined_set_dir : str
        Directory containing combined set iteration files
    updated_datasets_dir : str
        Directory to save combined dataset
    base_file : str
        Name of the base file
        
    Returns:
    --------
    str
        Path to the combined dataset file
    """
    os.makedirs(os.path.join(base_directory, updated_datasets_dir), exist_ok=True)
    
    initial_path = os.path.join(base_directory, initial_datasets_dir, base_file)
    if not os.path.exists(initial_path):
        raise FileNotFoundError(f"Initial peak dataset not found: {initial_path}")
    
    combined_df = pd.read_csv(initial_path)
    
    # List iteration files
    combined_path = os.path.join(base_directory, combined_set_dir)
    if os.path.exists(combined_path):
        iteration_files = [
            f for f in os.listdir(combined_path)
            if f.startswith('df_combined_') and f.endswith('.csv')
        ]
        
        def extract_iteration(filename):
            match = re.search(r'df_combined_(\d+)', filename)
            return int(match.group(1)) if match else 0
        
        iteration_files.sort(key=extract_iteration)
        
        for file_name in iteration_files:
            file_path = os.path.join(combined_path, file_name)
            df_new = pd.read_csv(file_path)
            combined_df = pd.concat([combined_df, df_new], ignore_index=True)
    
    combined_file_path = os.path.join(
        base_directory, updated_datasets_dir, 'combined_peak_data_full.csv'
    )
    combined_df.to_csv(combined_file_path, index=False)
    print(f"Combined peak dataset saved to {combined_file_path}")
    
    return combined_file_path


def update_npy_file(
    base_directory: str,
    initial_folder: str,
    updated_folder: str,
    new_files_directory: str,
    base_file: str
) -> str:
    """
    Update .npy file with multiple peaks data from iterations.
    
    Parameters:
    -----------
    base_directory : str
        Base directory path
    initial_folder : str
        Folder containing initial .npy file
    updated_folder : str
        Folder to save updated .npy file
    new_files_directory : str
        Directory containing iteration .npy files
    base_file : str
        Name of the base .npy file
        
    Returns:
    --------
    str
        Path to the updated .npy file
    """
    os.makedirs(os.path.join(base_directory, updated_folder), exist_ok=True)
    
    dtype = [('composition_number', int), ('iteration', int)]
    combined_data = np.array([], dtype=dtype)
    
    # Load initial dataset (iteration 0)
    initial_path = os.path.join(base_directory, initial_folder, base_file)
    if os.path.exists(initial_path):
        initial_data = np.load(initial_path)
        if initial_data.dtype.names:
            # Already structured
            combined_data = np.concatenate((combined_data, initial_data), axis=0)
        else:
            # Convert to structured
            initial_tagged_data = np.array(
                [(int(comp_num), 0) for comp_num in initial_data], dtype=dtype
            )
            combined_data = np.concatenate((combined_data, initial_tagged_data), axis=0)
    
    # Process iteration files
    new_files_path = os.path.join(base_directory, new_files_directory)
    if os.path.exists(new_files_path):
        iteration_files = [f for f in os.listdir(new_files_path) if f.endswith('.npy')]
        
        def extract_iteration(filename):
            match = re.search(r'(\d+)', filename)
            return int(match.group(1)) if match else 0
        
        iteration_files.sort(key=extract_iteration)
        
        for file_name in iteration_files:
            new_data_path = os.path.join(new_files_path, file_name)
            iteration_number = extract_iteration(file_name)
            new_data = np.load(new_data_path)
            
            if new_data.dtype.names:
                combined_data = np.concatenate((combined_data, new_data), axis=0)
            else:
                new_tagged_data = np.array(
                    [(int(comp_num), iteration_number) for comp_num in new_data], dtype=dtype
                )
                combined_data = np.concatenate((combined_data, new_tagged_data), axis=0)
    
    # Save updated file
    updated_file_name = f"multiple_peaks_updated_{datetime.now().strftime('%Y%m%d%H%M%S')}.npy"
    updated_path = os.path.join(base_directory, updated_folder, updated_file_name)
    np.save(updated_path, combined_data)
    print(f"Updated .npy file saved to {updated_path}")
    
    return updated_path


def delete_old_files(directory: str, age_days: int, exclude_files: tuple = None):
    """
    Delete files older than specified age.
    
    Parameters:
    -----------
    directory : str
        Directory to clean
    age_days : int
        Age threshold in days
    exclude_files : tuple, optional
        Filenames to never delete (e.g. current combined outputs needed by visualization)
    """
    if not os.path.exists(directory):
        return
    
    exclude = exclude_files or ()
    current_time = time.time()
    for filename in os.listdir(directory):
        if filename in exclude:
            continue
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_stat = os.stat(file_path)
            creation_time = file_stat.st_ctime
            if (current_time - creation_time) // (24 * 3600) >= age_days:
                os.remove(file_path)
                print(f"Deleted old file: {filename}")
