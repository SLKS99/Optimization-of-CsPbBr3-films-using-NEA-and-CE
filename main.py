"""
Main script for Dual GP Analysis
Automates the entire workflow from data loading to next batch prediction
"""

import os
import sys
import numpy as np
import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# Import custom modules
import config
from data_loader import (
    load_luminescence_data,
    load_composition_data,
    create_luminescence_dataframe
)
from peak_data_processor import (
    process_peak_data_file,
    process_all_peak_data_files,
    load_peak_data,
    extract_initial_final_peaks,
    create_intensity_dataset,
    create_peak_dataset,
    identify_multiple_peaks
)
from gp_models import (
    setup_kernel_prior,
    normalize_data,
    train_intensity_gp,
    create_search_grid,
    calculate_acquisition,
    tune_gp
)
from instability_scoring import instability_score_simple
from iteration_manager import (
    combine_intensity_datasets,
    combine_combined_peak_datasets,
    update_npy_file,
    delete_old_files
)


def setup_directories(base_dir: str):
    """Create necessary directories if they don't exist."""
    dirs = [
        os.path.join(base_dir, config.PEAK_DATA_DIR),
        os.path.join(base_dir, config.COMPOSITIONS_DIR),
        os.path.join(base_dir, config.INITIAL_DATASETS_DIR),
        os.path.join(base_dir, config.NEXT_ITERATIONS_DIR),
        os.path.join(base_dir, config.MULTIPLE_PEAKS_DIR),
        os.path.join(base_dir, config.COMBINED_SET_DIR),
        os.path.join(base_dir, config.UPDATED_DATASETS_DIR),
        os.path.join(base_dir, config.RESULTS_DIR),
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("Directory structure verified.")


def load_and_prepare_data():
    """Load and prepare all data files."""
    print("\n=== Loading Data ===")
    
    # Construct file paths (support both relative and absolute paths)
    if os.path.isabs(config.BASE_DIRECTORY):
        base_path = config.BASE_DIRECTORY
    else:
        base_path = os.path.join(os.getcwd(), config.BASE_DIRECTORY)
    
    # Check if using peak data or raw luminescence data
    if config.USE_PEAK_DATA:
        # Use peak data folder
        peak_data_dir = os.path.join(base_path, config.PEAK_DATA_DIR)
        compositions_dir = os.path.join(base_path, config.COMPOSITIONS_DIR)
        
        # Check if directories exist
        if not os.path.exists(peak_data_dir):
            raise FileNotFoundError(
                f"Peak data directory not found: {peak_data_dir}\n"
                f"Please create the directory and add your peak data files there."
            )
        if not os.path.exists(compositions_dir):
            raise FileNotFoundError(
                f"Compositions directory not found: {compositions_dir}\n"
                f"Please create the directory and add your composition file there."
            )
        
        # Automatically process all peak data files if enabled
        if config.AUTO_PROCESS_PEAK_DATA:
            print("Auto-processing peak data files...")
            # Always pass None for composition_cols to force auto-detection
            processed_files = process_all_peak_data_files(
                peak_data_dir,
                compositions_dir,
                base_path,
                config.WELLS_TO_IGNORE,
                config.INITIAL_READ,
                config.FINAL_READ,
                None,  # Always auto-detect - don't use config.COMPOSITION_COLUMNS which may have old values
                config.COMPOSITION_FILE_NAME,
                match_composition_to_iteration=True,  # Match composition files to iterations
                auto_detect_composition_cols=True  # Auto-detect from composition file
            )
            print("Peak data processing complete.")
        
        # Peak data processing is complete - composition data not needed for return
        # The datasets have already been created and saved
        print("Peak data processing complete. Datasets created successfully.")
        return None, None, None, None, None
        
    else:
        # Use raw luminescence data (original workflow)
        data_file_path = os.path.join(base_path, config.DATA_FILE_NAME)
        composition_file_path = os.path.join(base_path, config.COMPOSITION_FILE_NAME)
        
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(
                f"Data file not found: {data_file_path}\n"
                f"Please place your luminescence data file at: {data_file_path}"
            )
        if not os.path.exists(composition_file_path):
            raise FileNotFoundError(
                f"Composition file not found: {composition_file_path}\n"
                f"Please place your composition file at: {composition_file_path}"
            )
        
        # Load data
        print(f"Loading luminescence data from: {data_file_path}")
        data_dict = load_luminescence_data(data_file_path, config.WELLS_TO_IGNORE)
        
        print(f"Loading composition data from: {composition_file_path}")
        composition_df, comp_array, cells_with_data = load_composition_data(
            composition_file_path, config.WELLS_TO_IGNORE
        )
        
        # Create luminescence dataframe
        print("Creating luminescence dataframe...")
        luminescence_df, luminescence_vec, wavelength_array, time_array = create_luminescence_dataframe(
            data_dict,
            config.LUMINESCENCE_READ_NUMBERS,
            config.START_WAVELENGTH,
            config.END_WAVELENGTH,
            config.WAVELENGTH_STEP_SIZE
        )
        
        print("Data loading complete.")
        return data_dict, composition_df, comp_array, cells_with_data, luminescence_df


def combine_iteration_datasets():
    """Combine datasets from all iterations."""
    print("\n=== Combining Iteration Datasets ===")
    
    # Support both relative and absolute paths
    if os.path.isabs(config.BASE_DIRECTORY):
        base_dir = config.BASE_DIRECTORY
    else:
        base_dir = os.path.join(os.getcwd(), config.BASE_DIRECTORY)
    
    # Combine intensity datasets
    intensity_file = "df_intensity_initial.csv"  # Initial iteration dataset
    try:
        intensity_combined = combine_intensity_datasets(
            base_dir,
            config.INITIAL_DATASETS_DIR,
            config.NEXT_ITERATIONS_DIR,
            config.UPDATED_DATASETS_DIR,
            intensity_file
        )
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Skipping intensity dataset combination. Using initial dataset only.")
        intensity_combined = os.path.join(
            base_dir, config.INITIAL_DATASETS_DIR, intensity_file
        )
    
    # Combine peak datasets
    peak_file = "df_combined_initial.csv"
    try:
        combined_peak_data = combine_combined_peak_datasets(
            base_dir,
            config.INITIAL_DATASETS_DIR,
            config.COMBINED_SET_DIR,
            config.UPDATED_DATASETS_DIR,
            peak_file
        )
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Skipping peak dataset combination.")
        combined_peak_data = None
    
    # Update multiple peaks file
    npy_file = "multiple_peaks_initial.npy"
    try:
        multiple_peaks_combined = update_npy_file(
            base_dir,
            config.INITIAL_DATASETS_DIR,
            config.UPDATED_DATASETS_DIR,
            config.MULTIPLE_PEAKS_DIR,
            npy_file
        )
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Skipping multiple peaks update.")
        multiple_peaks_combined = None
    
    # Load combined datasets
    df_intensity = pd.read_csv(intensity_combined) if os.path.exists(intensity_combined) else None
    
    # Check if the loaded file has the correct number of composition columns
    if df_intensity is not None:
        intensity_cols = [col for col in df_intensity.columns if col not in ['Intensity initial', 'Intensity final']]
        if len(intensity_cols) < 3:
            # Try reloading from initial file instead
            initial_file = os.path.join(base_dir, config.INITIAL_DATASETS_DIR, intensity_file)
            if os.path.exists(initial_file):
                print(f"Reloading from initial file (combined file has old format)...")
                df_intensity = pd.read_csv(initial_file)
                intensity_cols = [col for col in df_intensity.columns if col not in ['Intensity initial', 'Intensity final']]
                print(f"Found {len(intensity_cols)} composition columns: {intensity_cols}")
    
    multiple_peaks = np.load(multiple_peaks_combined) if multiple_peaks_combined and os.path.exists(multiple_peaks_combined) else np.array([])
    peak_data = pd.read_csv(combined_peak_data) if combined_peak_data and os.path.exists(combined_peak_data) else None
    
    # Add iteration and composition number columns if needed
    if peak_data is not None:
        peak_data['iteration'] = peak_data.index // config.COMPOSITIONS_PER_ITERATION
        peak_data['composition_number'] = peak_data.index % config.COMPOSITIONS_PER_ITERATION + 1
    
    # Auto-detect composition columns from intensity dataset if not in config
    if df_intensity is not None and (not hasattr(config, 'COMPOSITION_COLUMNS') or config.COMPOSITION_COLUMNS is None):
        # Get column names excluding intensity columns
        intensity_cols = [
            col for col in df_intensity.columns 
            if col not in ['Intensity initial', 'Intensity final']
        ]
        print(f"Found composition columns in dataset: {intensity_cols}")
        if len(intensity_cols) >= 3:
            # Store detected columns for use in GP analysis (Cs, NEA, CE)
            config.COMPOSITION_COLUMNS = intensity_cols[:3]
            print(f"Auto-detected composition columns from dataset: {config.COMPOSITION_COLUMNS}")
        elif len(intensity_cols) == 2:
            # Check if this is an old file - try to reload from initial_datasets
            initial_file = os.path.join(base_dir, config.INITIAL_DATASETS_DIR, 'df_intensity_initial.csv')
            if os.path.exists(initial_file):
                df_check = pd.read_csv(initial_file)
                check_cols = [col for col in df_check.columns if col not in ['Intensity initial', 'Intensity final']]
                if len(check_cols) >= 3:
                    print(f"Found updated file with {len(check_cols)} columns. Reloading...")
                    df_intensity = df_check
                    config.COMPOSITION_COLUMNS = check_cols[:3]
                    print(f"Auto-detected composition columns: {config.COMPOSITION_COLUMNS}")
                else:
                    raise ValueError(
                        f"Intensity dataset must have 3 composition columns (Cs, NEA, CE). "
                        f"Found only {len(intensity_cols)}: {intensity_cols}. "
                        f"Please delete old CSV files and re-run to regenerate with correct columns."
                    )
            else:
                raise ValueError(
                    f"Intensity dataset must have 3 composition columns (Cs, NEA, CE). "
                    f"Found only {len(intensity_cols)}: {intensity_cols}"
                )
    
    # Clean old files
    updated_dir = os.path.join(base_dir, config.UPDATED_DATASETS_DIR)
    delete_old_files(updated_dir, config.DELETE_OLD_FILES_AFTER_DAYS)
    
    return df_intensity, multiple_peaks, peak_data


def run_dual_gp_analysis(df_intensity, peak_data, multiple_peaks):
    """Run the Dual GP analysis and generate next batch predictions."""
    print("\n=== Running Dual GP Analysis ===")
    
    # Setup GP priors
    kernel_prior_func = setup_kernel_prior()
    import numpyro.distributions as dist
    noise_prior_dist = dist.HalfNormal(scale=config.NOISE_PRIOR_SCALE)
    
    # Prepare training data
    # Get composition columns (either from config or auto-detected from dataset)
    if hasattr(config, 'COMPOSITION_COLUMNS') and config.COMPOSITION_COLUMNS is not None:
        composition_cols = config.COMPOSITION_COLUMNS
    else:
        # Auto-detect from intensity dataset columns (exclude intensity columns)
        intensity_cols = [
            col for col in df_intensity.columns 
            if col not in ['Intensity initial', 'Intensity final']
        ]
        if len(intensity_cols) < 3:
            raise ValueError(
                f"Intensity dataset must have at least 3 composition columns (Cs, NEA, CE). Found: {intensity_cols}"
            )
        composition_cols = intensity_cols[:3]
        print(f"Using auto-detected composition columns: {composition_cols}")
    
    if len(composition_cols) != 3:
        raise ValueError(
            f"COMPOSITION_COLUMNS must have exactly 3 elements (Cs, NEA, CE), got {len(composition_cols)}: {composition_cols}"
        )
    X_int = df_intensity[composition_cols].values
    y_int = df_intensity['Intensity initial'].values
    
    # Define concentration ranges for normalization
    # Order: [Cs, NEA, CE]
    ranges = np.array([
        [config.CS_MIN, config.CS_MAX],
        [config.NEA_MIN, config.NEA_MAX],
        [config.CE_MIN, config.CE_MAX]
    ])
    
    # Normalize using component-specific ranges
    # Create a closure to capture ranges
    def norm_func(x):
        return normalize_data(x, ranges=ranges)
    X_int_norm = norm_func(X_int)
    y_train_normalized = y_int
    
    # Create search grid with independent ranges
    print("Creating search grid...")
    Xnew = create_search_grid(
        cs_min=config.CS_MIN,
        cs_max=config.CS_MAX,
        nea_min=config.NEA_MIN,
        nea_max=config.NEA_MAX,
        ce_min=config.CE_MIN,
        ce_max=config.CE_MAX,
        grid_resolution=config.GRID_RESOLUTION,
        normalize_func=norm_func,
        ranges=ranges
    )
    
    # Train intensity GP
    print("Training intensity GP model...")
    gp_model = train_intensity_gp(
        X_int_norm, y_train_normalized,
        kernel_prior_func, noise_prior_dist
    )
    
    # Calculate acquisition
    print("Calculating acquisition function...")
    acq, y_pred, y_sampled, y_std = calculate_acquisition(
        gp_model, Xnew, beta=config.UCB_BETA
    )
    
    # Calculate instability scores
    print("Calculating instability scores...")
    if peak_data is not None:
        y_tune_score = instability_score_simple(
            peak_data,
            target_wavelength=config.TARGET_WAVELENGTH,
            wavelength_tolerance=config.WAVELENGTH_TOLERANCE,
            degradation_weight=config.DEGRADATION_WEIGHT,
            position_weight=config.POSITION_WEIGHT
        )
        y_tune_score = (y_tune_score - np.min(y_tune_score)) / (
            np.max(y_tune_score) - np.min(y_tune_score) + 1e-10
        )
        
        # Tune GP
        print("Training stability GP model...")
        init_tune_score, adjust_tune_score = tune_gp(
            X_int_norm, y_tune_score, Xnew,
            kernel_prior_func, noise_prior_dist,
            thre_percent=config.TUNE_THRESHOLD_PERCENT
        )
        
        # Adjust acquisition
        acq_tune = acq.at[np.where(
            init_tune_score > np.quantile(init_tune_score, config.TUNE_THRESHOLD_PERCENT)
        )].set(0)
        acq_tune = acq * adjust_tune_score
    else:
        print("Warning: No peak data available. Using standard acquisition only.")
        adjust_tune_score = jnp.ones_like(acq)
        acq_tune = acq
    
    # Select next batch
    print("Selecting next batch of experiments...")
    batch_indices_int = np.argsort(acq)[-config.BATCH_SIZE:]
    batch_indices_int_acq = np.argsort(acq_tune)[-config.BATCH_SIZE:]
    
    X_next_batch_int = Xnew[batch_indices_int]
    X_next_batch_int_acq = Xnew[batch_indices_int_acq]
    
    print(f"Selected {len(X_next_batch_int_acq)} compositions for next iteration.")
    
    return {
        'gp_model': gp_model,
        'Xnew': Xnew,
        'y_pred': y_pred,
        'y_std': y_std,
        'acq': acq,
        'acq_tune': acq_tune,
        'adjust_tune_score': adjust_tune_score,
        'X_next_batch_int': X_next_batch_int,
        'X_next_batch_int_acq': X_next_batch_int_acq,
        'batch_indices_int': batch_indices_int,
        'batch_indices_int_acq': batch_indices_int_acq
    }


def unnormalize_data_with_ranges(values, ranges, step_size=0.01):
    """
    Unnormalize values using component-specific ranges.
    
    Parameters:
    -----------
    values : np.ndarray
        Normalized values (N x D) in [0, 1] range
    ranges : np.ndarray
        Array of shape (D, 2) with [min, max] for each dimension
    step_size : float
        Step size for rounding
        
    Returns:
    --------
    np.ndarray
        Unnormalized values in original ranges
    """
    values = np.array(values)
    unnormalized = np.zeros_like(values)
    
    for d in range(values.shape[1]):
        x_min, x_max = ranges[d, 0], ranges[d, 1]
        # Unnormalize: x_norm * (max - min) + min
        unnormalized[:, d] = values[:, d] * (x_max - x_min) + x_min
        # Round to step size
        unnormalized[:, d] = np.round(unnormalized[:, d] / step_size) * step_size
        # Clip to bounds
        unnormalized[:, d] = np.clip(unnormalized[:, d], x_min, x_max)
    
    return unnormalized


def save_next_batch_results(X_next_batch_int_acq, output_dir):
    """Save next batch predictions to CSV file."""
    print("\n=== Saving Results ===")
    
    # Define concentration ranges for unnormalization
    # Order: [Cs, NEA, CE]
    ranges = np.array([
        [config.CS_MIN, config.CS_MAX],
        [config.NEA_MIN, config.NEA_MAX],
        [config.CE_MIN, config.CE_MAX]
    ])
    
    # Unnormalize using component-specific ranges
    X_unnormalized = unnormalize_data_with_ranges(
        X_next_batch_int_acq,
        ranges=ranges,
        step_size=config.GRID_RESOLUTION
    )
    
    # Extract components
    Cs = X_unnormalized[:, 0]
    NEA = X_unnormalized[:, 1]
    CE = X_unnormalized[:, 2]
    
    # PbBr is fixed at 0.2 (not optimized)
    PbBr = np.full(len(Cs), config.PBBR_FIXED)
    
    # Round to 2 decimal places (keep as decimal concentrations, not percentages)
    Cs_rounded = np.round(Cs, 2)
    NEA_rounded = np.round(NEA, 2)
    CE_rounded = np.round(CE, 2)
    PbBr_rounded = np.round(PbBr, 2)
    
    # Create dataframe
    wells = [f"{chr(65+i)}{j+1}" for i in range(8) for j in range(12)]
    df = pd.DataFrame({
        'Well': wells[:len(Cs_rounded)],
        'PbBr': PbBr_rounded,  # Fixed component
        config.PRECURSOR1: Cs_rounded,  # Cs
        config.PRECURSOR2: NEA_rounded,  # NEA
        config.PRECURSOR3: CE_rounded   # CE
    })
    
    # Save to CSV with 2 decimal places formatting
    output_file = os.path.join(output_dir, 'next_batch_compositions.csv')
    df.to_csv(output_file, index=False, float_format='%.2f')
    print(f"Next batch compositions saved to: {output_file}")
    
    return df


def create_visualizations(results, output_dir):
    """Create visualizations for 3D independent concentration space."""
    if not config.SAVE_PLOTS:
        return
    
    print("\n=== Creating Visualizations ===")
    
    # Unnormalize data for visualization
    ranges = np.array([
        [config.CS_MIN, config.CS_MAX],
        [config.NEA_MIN, config.NEA_MAX],
        [config.CE_MIN, config.CE_MAX]
    ])
    
    Xnew_norm = np.array(results['Xnew'])
    Xnew = unnormalize_data_with_ranges(Xnew_norm, ranges=ranges, step_size=config.GRID_RESOLUTION)
    
    y_pred = results['y_pred']
    y_std = results['y_std']
    acq = results['acq']
    acq_tune = results['acq_tune']
    adjust_tune_score = results['adjust_tune_score']
    
    X_next_batch_int_norm = np.array(results['X_next_batch_int'])
    X_next_batch_int_acq_norm = np.array(results['X_next_batch_int_acq'])
    
    X_next_batch_int = unnormalize_data_with_ranges(X_next_batch_int_norm, ranges=ranges, step_size=config.GRID_RESOLUTION)
    X_next_batch_int_acq = unnormalize_data_with_ranges(X_next_batch_int_acq_norm, ranges=ranges, step_size=config.GRID_RESOLUTION)
    
    # Extract components
    Cs = Xnew[:, 0]
    NEA = Xnew[:, 1]
    CE = Xnew[:, 2]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(24, 16), dpi=config.PLOT_DPI)
    gs = fig.add_gridspec(4, 4, hspace=0.5, wspace=0.5)
    
    # Helper function to create heatmap plots (define first)
    def create_heatmap(ax, x, y, values, xlabel, ylabel, title, cmap, selected_points=None, selected_color='red'):
        """Create a smooth heatmap/contour plot."""
        # Create fine grid for interpolation
        x_grid = np.linspace(x.min(), x.max(), 100)
        y_grid = np.linspace(y.min(), y.max(), 100)
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
        
        # Interpolate values onto grid
        try:
            values_grid = griddata((x, y), values, (X_mesh, Y_mesh), method='cubic', fill_value=np.nan)
            im = ax.contourf(X_mesh, Y_mesh, values_grid, levels=20, cmap=cmap, alpha=0.85, extend='both')
            ax.contour(X_mesh, Y_mesh, values_grid, levels=20, colors='black', alpha=0.2, linewidths=0.5)
        except:
            # Fallback to scatter if interpolation fails
            im = ax.scatter(x, y, c=values, s=10, alpha=0.6, cmap=cmap)
        
        if selected_points is not None and len(selected_points) > 0:
            ax.scatter(selected_points[:, 0], selected_points[:, 1], marker='*',
                      color=selected_color, s=200, label='Selected', edgecolors='black', linewidths=2, zorder=10)
        
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=5)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=8)
        if selected_points is not None and len(selected_points) > 0:
            ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Helper function to create 2D slice visualizations at different CE levels
    def create_slice_grid(gs_start_row, gs_start_col, x, y, z, values, cmap, title_base, label, 
                         selected_points=None, n_slices=4):
        """Create a grid of 2D slices at different CE levels."""
        ce_levels = np.linspace(z.min(), z.max(), n_slices)
        
        # Create subplot for each slice
        axes = []
        for i, ce_val in enumerate(ce_levels):
            row = gs_start_row + i // 2
            col = gs_start_col + i % 2
            ax = fig.add_subplot(gs[row, col])
            
            # Find points near this CE level (within 5% of range)
            tolerance = (z.max() - z.min()) / (n_slices * 2)
            mask = np.abs(z - ce_val) < tolerance
            
            if np.sum(mask) > 10:
                x_slice = x[mask]
                y_slice = y[mask]
                v_slice = values[mask]
                
                # Create fine grid for interpolation
                x_grid = np.linspace(x.min(), x.max(), 50)
                y_grid = np.linspace(y.min(), y.max(), 50)
                X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
                
                # Interpolate values onto grid
                try:
                    V_grid = griddata((x_slice, y_slice), v_slice, (X_mesh, Y_mesh), 
                                     method='cubic', fill_value=np.nan)
                    im = ax.contourf(X_mesh, Y_mesh, V_grid, levels=15, cmap=cmap, alpha=0.85, extend='both')
                    ax.contour(X_mesh, Y_mesh, V_grid, levels=15, colors='black', alpha=0.2, linewidths=0.3)
                except:
                    # Fallback to scatter
                    im = ax.scatter(x_slice, y_slice, c=v_slice, s=20, alpha=0.6, cmap=cmap)
                
                # Add selected points if they're near this CE level
                if selected_points is not None and len(selected_points) > 0:
                    selected_mask = np.abs(selected_points[:, 2] - ce_val) < tolerance
                    if np.sum(selected_mask) > 0:
                        ax.scatter(selected_points[selected_mask, 0], 
                                 selected_points[selected_mask, 1],
                                 marker='*', color='yellow', s=150, 
                                 edgecolors='black', linewidths=1.5, zorder=10, label='Selected')
                
                ax.set_xlabel(f'{config.PRECURSOR1} Concentration', fontsize=8)
                ax.set_ylabel(f'{config.PRECURSOR2} Concentration', fontsize=8)
                ax.set_title(f'CE = {ce_val:.2f}', fontsize=9, fontweight='bold', pad=5)
                cbar = plt.colorbar(im, ax=ax, label=label, shrink=0.8)
                cbar.ax.tick_params(labelsize=7)
                if selected_points is not None and np.sum(selected_mask) > 0:
                    ax.legend(fontsize=7, loc='upper right')
                ax.grid(True, alpha=0.3, linestyle='--')
            else:
                ax.axis('off')
                ax.text(0.5, 0.5, f'CE = {ce_val:.2f}\n(Insufficient data)', 
                       ha='center', va='center', transform=ax.transAxes)
            
            axes.append(ax)
        
        return axes
    
    # ========== Row 1-2: GP Predictions - Slices at different CE levels ==========
    fig.text(0.25, 0.97, 'GP Predictions', 
            ha='center', fontsize=12, fontweight='bold', transform=fig.transFigure)
    create_slice_grid(0, 0, Cs, NEA, CE, y_pred, 'viridis', '', 'Intensity',
                     X_next_batch_int if len(X_next_batch_int) > 0 else None, n_slices=4)
    
    # ========== Row 1-2: Acquisition Function - Slices at different CE levels ==========
    fig.text(0.25, 0.73, 'Acquisition Function', 
            ha='center', fontsize=12, fontweight='bold', transform=fig.transFigure)
    create_slice_grid(2, 0, Cs, NEA, CE, np.array(acq), 'plasma', '', 'Acquisition',
                     None, n_slices=4)
    
    # ========== Row 3: Adjusted Acquisition and Uncertainty ==========
    # Adjusted Acquisition: NEA vs CE (single 2D plot)
    ax5 = fig.add_subplot(gs[0, 2])
    create_heatmap(ax5, NEA, CE, np.array(acq_tune),
                  f'{config.PRECURSOR2} Concentration', f'{config.PRECURSOR3} Concentration',
                  'Adjusted Acquisition: NEA vs CE', 'inferno',
                  X_next_batch_int_acq[:, [1, 2]] if len(X_next_batch_int_acq) > 0 else None, 'yellow')
    
    # Uncertainty: NEA vs CE (single 2D plot)
    ax6 = fig.add_subplot(gs[0, 3])
    create_heatmap(ax6, NEA, CE, np.array(y_std),
                  f'{config.PRECURSOR2} Concentration', f'{config.PRECURSOR3} Concentration',
                  'Prediction Uncertainty: NEA vs CE', 'Reds',
                  None, 'red')
    
    # ========== Row 3: 2D Heatmaps (Predictions) ==========
    # Cs vs NEA - Predictions
    ax7 = fig.add_subplot(gs[2, 0])
    create_heatmap(ax5, Cs, NEA, y_pred, 
                  f'{config.PRECURSOR1} Concentration', f'{config.PRECURSOR2} Concentration',
                  'Predictions: Cs vs NEA', 'viridis',
                  X_next_batch_int[:, [0, 1]] if len(X_next_batch_int) > 0 else None)
    
    # Cs vs CE - Predictions
    ax6 = fig.add_subplot(gs[1, 1])
    create_heatmap(ax6, Cs, CE, y_pred,
                  f'{config.PRECURSOR1} Concentration', f'{config.PRECURSOR3} Concentration',
                  'Predictions: Cs vs CE', 'viridis',
                  X_next_batch_int[:, [0, 2]] if len(X_next_batch_int) > 0 else None)
    
    # NEA vs CE - Predictions
    ax7 = fig.add_subplot(gs[1, 2])
    create_heatmap(ax7, NEA, CE, y_pred,
                  f'{config.PRECURSOR2} Concentration', f'{config.PRECURSOR3} Concentration',
                  'Predictions: NEA vs CE', 'viridis',
                  X_next_batch_int[:, [1, 2]] if len(X_next_batch_int) > 0 else None)
    
    # NEA vs CE - Adjusted Acquisition
    ax8 = fig.add_subplot(gs[1, 3])
    create_heatmap(ax8, NEA, CE, acq_tune,
                  f'{config.PRECURSOR2} Concentration', f'{config.PRECURSOR3} Concentration',
                  'Adjusted Acquisition: NEA vs CE', 'plasma',
                  X_next_batch_int_acq[:, [1, 2]] if len(X_next_batch_int_acq) > 0 else None, 'yellow')
    
    # ========== Row 3: Additional Heatmaps ==========
    # Cs vs NEA - Adjusted Acquisition
    ax9 = fig.add_subplot(gs[2, 0])
    create_heatmap(ax9, Cs, NEA, acq_tune,
                  f'{config.PRECURSOR1} Concentration', f'{config.PRECURSOR2} Concentration',
                  'Adjusted Acquisition: Cs vs NEA', 'plasma',
                  X_next_batch_int_acq[:, [0, 1]] if len(X_next_batch_int_acq) > 0 else None, 'yellow')
    
    # Cs vs CE - Adjusted Acquisition
    ax10 = fig.add_subplot(gs[2, 1])
    create_heatmap(ax10, Cs, CE, acq_tune,
                   f'{config.PRECURSOR1} Concentration', f'{config.PRECURSOR3} Concentration',
                   'Adjusted Acquisition: Cs vs CE', 'plasma',
                   X_next_batch_int_acq[:, [0, 2]] if len(X_next_batch_int_acq) > 0 else None, 'yellow')
    
    # Best Stability Compositions Plot (Low Uncertainty + High Stability)
    ax11 = fig.add_subplot(gs[2, 2])
    
    # Convert to numpy arrays
    adjust_tune_score_arr = np.array(adjust_tune_score)
    y_std_arr = np.array(y_std)
    
    # Calculate stability score (inverse of instability, normalized)
    stability_score = 1.0 - adjust_tune_score_arr  # Higher stability = lower instability
    stability_score = (stability_score - stability_score.min()) / (stability_score.max() - stability_score.min() + 1e-10)
    
    # Calculate uncertainty (normalized, inverted so low uncertainty = high score)
    uncertainty_norm = (y_std_arr - y_std_arr.min()) / (y_std_arr.max() - y_std_arr.min() + 1e-10)
    low_uncertainty_score = 1.0 - uncertainty_norm  # Low uncertainty = high score
    
    # Combined score: high stability + low uncertainty
    # Weight can be adjusted: 0.5 means equal weight, higher = more weight on stability
    stability_weight = 0.6
    uncertainty_weight = 0.4
    combined_score = stability_weight * stability_score + uncertainty_weight * low_uncertainty_score
    
    # Find top compositions
    top_n = min(50, len(combined_score))
    top_indices = np.argsort(combined_score)[-top_n:]
    
    # Create scatter plot colored by combined score
    scatter_stab = ax11.scatter(
        stability_score[top_indices], 
        low_uncertainty_score[top_indices],
        c=combined_score[top_indices], 
        s=100, alpha=0.7, cmap='RdYlGn', edgecolors='black', linewidths=1
    )
    
    # Highlight top 10
    top_10_indices = top_indices[-10:]
    ax11.scatter(
        stability_score[top_10_indices],
        low_uncertainty_score[top_10_indices],
        s=200, marker='*', color='gold', edgecolors='black', linewidths=2,
        label='Top 10 Best', zorder=10
    )
    
    ax11.set_xlabel('Stability Score', fontsize=9)
    ax11.set_ylabel('Low Uncertainty Score', fontsize=9)
    ax11.set_title('Best Stability Compositions', fontsize=10, fontweight='bold', pad=5)
    cbar = plt.colorbar(scatter_stab, ax=ax11, label='Combined Score', shrink=0.8)
    cbar.ax.tick_params(labelsize=8)
    ax11.legend(fontsize=8)
    ax11.grid(True, alpha=0.3, linestyle='--')
    
    # Summary Statistics
    ax12 = fig.add_subplot(gs[3, 3])
    ax12.axis('off')
    top_idx = top_indices[-1] if len(top_indices) > 0 else 0
    summary_text = f"""
    Search Space Summary:
    • {config.PRECURSOR1}: {config.CS_MIN:.2f} - {config.CS_MAX:.2f}
    • {config.PRECURSOR2}: {config.NEA_MIN:.2f} - {config.NEA_MAX:.2f}
    • {config.PRECURSOR3}: {config.CE_MIN:.2f} - {config.CE_MAX:.2f}
    • PbBr (fixed): {config.PBBR_FIXED:.2f}
    
    Selected Compositions: {len(X_next_batch_int_acq)}
    Max Acquisition Value: {np.max(acq_tune):.4f}
    Min Uncertainty: {np.min(y_std_arr):.4f}
    Mean Predicted Intensity: {np.mean(y_pred):.2f}
    
    Top Stability Composition:
    Cs: {Cs[top_idx]:.3f}
    NEA: {NEA[top_idx]:.3f}
    CE: {CE[top_idx]:.3f}
    """
    ax12.text(0.1, 0.5, summary_text, fontsize=9, verticalalignment='center',
              family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Dual GP Analysis: 3D Concentration Space', fontsize=16, fontweight='bold', y=0.995)
    
    plot_file = os.path.join(output_dir, 'dual_gp_analysis.png')
    plt.savefig(plot_file, dpi=config.PLOT_DPI, bbox_inches='tight')
    print(f"Visualization saved to: {plot_file}")
    plt.close()


def main():
    """Main execution function."""
    print("=" * 60)
    print("Dual GP Analysis - Automated Workflow")
    print("=" * 60)
    
    # Determine base directory path
    if os.path.isabs(config.BASE_DIRECTORY):
        base_dir = config.BASE_DIRECTORY
    else:
        base_dir = os.path.join(os.getcwd(), config.BASE_DIRECTORY)
    
    # Setup directories
    setup_directories(base_dir)
    
    # First, load and prepare data (this processes peak data files if AUTO_PROCESS_PEAK_DATA is True)
    print("\n=== Step 1: Processing Peak Data ===")
    data_dict, composition_df, comp_array, cells_with_data, luminescence_df = load_and_prepare_data()
    
    # Then combine datasets from all iterations
    print("\n=== Step 2: Combining Iteration Datasets ===")
    df_intensity, multiple_peaks, peak_data = combine_iteration_datasets()
    
    if df_intensity is None:
        print("Error: Could not load intensity data. Please check your data files.")
        print("Make sure peak data processing completed successfully.")
        return
    
    # Run Dual GP analysis
    results = run_dual_gp_analysis(df_intensity, peak_data, multiple_peaks)
    
    # Save results
    results_dir = os.path.join(base_dir, config.RESULTS_DIR)
    os.makedirs(results_dir, exist_ok=True)
    
    next_batch_df = save_next_batch_results(results['X_next_batch_int_acq'], results_dir)
    
    # Create visualizations
    create_visualizations(results, results_dir)
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print(f"Results saved to: {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
