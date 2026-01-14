# Configuration Parameters Guide

This document explains all parameters in `config.py` and how they're used throughout the codebase.

## Parameter Categories

### 1. Experimental Parameters
- **EXPERIMENT_NAME**: Name of your experiment (for reference)
- **PRECURSOR1, PRECURSOR2, PRECURSOR3**: Names of your three precursors (used in plots and output CSV)

### 2. Measurement Parameters (Raw Data Workflow Only)
These are **only used** if `USE_PEAK_DATA = False`:
- **START_WAVELENGTH**: Starting wavelength for raw luminescence data
- **END_WAVELENGTH**: Ending wavelength for raw luminescence data
- **WAVELENGTH_STEP_SIZE**: Step size for wavelength array
- **TIME_STEP**: Minutes between measurements
- **NUMBER_OF_READS**: Total number of reads
- **LUMINESCENCE_READ_NUMBERS**: List of read numbers to include in analysis
- **INTERPOLATE**: Whether to interpolate raw data

**Note**: If using peak data (`USE_PEAK_DATA = True`), these parameters are ignored.

### 3. Peak Data Parameters
- **USE_PEAK_DATA**: Set to `True` to use pre-fitted peak data instead of raw luminescence data
- **AUTO_PROCESS_PEAK_DATA**: Automatically process all peak data files in `peak_data/` folder
- **INITIAL_READ**: Read number for initial measurement (typically 1)
- **FINAL_READ**: Read number for final measurement (typically 2)
- **COMPOSITION_COLUMNS**: Column names in composition file for the first two components
  - Example: `['CsPbI4', 'PEA2PbI4']`
  - Used to extract composition values from your composition file
  - Must match the row names in your composition Excel/CSV file

### 4. Composition Parameters
- **COMPOSITIONS_PER_ITERATION**: Number of compositions per iteration
  - Default: 96 (for 8x12 well plate)
  - Used to calculate iteration and composition numbers from dataset indices

### 5. GP Model Parameters
- **BATCH_SIZE**: Number of compositions to select for next iteration
  - Default: 96 (full plate)
- **GP_KERNEL**: Kernel type for Gaussian Process (default: "RBF")
- **UCB_BETA**: Beta parameter for Upper Confidence Bound acquisition function
  - Higher values = more exploration
  - Lower values = more exploitation
- **GRID_RESOLUTION**: Step size for search grid (default: 1)
- **GRID_MAX**: Maximum value for ternary composition grid (default: 50)
- **NOISE_PRIOR_SCALE**: Scale parameter for GP noise prior
  - Controls uncertainty in predictions
  - Default: 0.1

### 6. Instability Score Parameters
All these parameters are used in `instability_scoring.py`:

- **TARGET_WAVELENGTH**: Target wavelength for peak analysis
  - Used to calculate position deviation scores
  - Default: 705 nm
- **WAVELENGTH_TOLERANCE**: Tolerance range around target wavelength
  - Peaks within this range get zero position penalty
  - Default: 10 nm
- **DEGRADATION_WEIGHT**: Weight for intensity degradation component
  - Higher = more penalty for intensity changes
  - Default: 0.3
- **POSITION_WEIGHT**: Weight for peak position deviation component
  - Higher = more penalty for position deviations
  - Default: 0.7
- **MULTIPLE_PEAK_PENALTY**: Penalty score for compositions with multiple peaks
  - Applied when `Total_Quality_Peaks >= 2`
  - Default: 0.5
- **TUNE_THRESHOLD_PERCENT**: Percentile threshold for adjusting acquisition function
  - Compositions above this threshold get acquisition set to 0
  - Default: 0.8 (80th percentile)

### 7. Output Settings
- **SAVE_PLOTS**: Whether to save visualization plots (default: True)
- **PLOT_DPI**: Resolution for saved plots (default: 200)
- **DELETE_OLD_FILES_AFTER_DAYS**: Auto-delete old files in `updated_datasets/` (default: 1)

## Parameter Usage Map

### Used in `instability_scoring.py`:
- ✅ TARGET_WAVELENGTH
- ✅ WAVELENGTH_TOLERANCE
- ✅ DEGRADATION_WEIGHT
- ✅ POSITION_WEIGHT
- ✅ MULTIPLE_PEAK_PENALTY

### Used in `main.py`:
- ✅ All GP model parameters
- ✅ All instability score parameters
- ✅ COMPOSITIONS_PER_ITERATION
- ✅ COMPOSITION_COLUMNS
- ✅ PRECURSOR1, PRECURSOR2, PRECURSOR3
- ✅ NOISE_PRIOR_SCALE

### Used in `peak_data_processor.py`:
- ✅ INITIAL_READ
- ✅ FINAL_READ
- ✅ COMPOSITION_COLUMNS
- ✅ WELLS_TO_IGNORE

### Used in `data_loader.py`:
- ✅ START_WAVELENGTH, END_WAVELENGTH, WAVELENGTH_STEP_SIZE (raw data only)
- ✅ LUMINESCENCE_READ_NUMBERS (raw data only)
- ✅ WELLS_TO_IGNORE

## No Hardcoded Values

All parameters are now configurable through `config.py`. The following were previously hardcoded but are now configurable:

1. ✅ Instability score parameters (were hardcoded defaults in function signatures)
2. ✅ Compositions per iteration (was hardcoded as 96)
3. ✅ Noise prior scale (was hardcoded as 0.1)
4. ✅ Composition column names (now uses COMPOSITION_COLUMNS)
5. ✅ Output CSV column names (now uses PRECURSOR1, PRECURSOR2, PRECURSOR3)

## Recommendations

### For Peak Data Workflow:
- Set `USE_PEAK_DATA = True`
- Configure `INITIAL_READ` and `FINAL_READ` to match your data
- Set `COMPOSITION_COLUMNS` to match your composition file structure
- Adjust instability score parameters based on your target wavelength and tolerance needs

### For Raw Data Workflow:
- Set `USE_PEAK_DATA = False`
- Configure wavelength range and read numbers
- Set interpolation settings

### For Different Plate Sizes:
- Adjust `COMPOSITIONS_PER_ITERATION` if not using 96-well plates
- Adjust `BATCH_SIZE` accordingly

### For Different Target Wavelengths:
- Update `TARGET_WAVELENGTH` to your desired peak wavelength
- Adjust `WAVELENGTH_TOLERANCE` based on acceptable range
