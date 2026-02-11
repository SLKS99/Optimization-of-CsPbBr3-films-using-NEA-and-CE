"""
Configuration file for Dual GP Analysis
Edit these parameters according to your experimental setup
"""

# Precursor Names (for plotting)
PRECURSOR1 = "Cs"
PRECURSOR2 = "NEA"
PRECURSOR3 = "CE"

# Measurement Parameters (for raw luminescence data workflow only)
# These are only used if USE_PEAK_DATA = False
START_WAVELENGTH = 401
END_WAVELENGTH = 700
WAVELENGTH_STEP_SIZE = 1
LUMINESCENCE_READ_NUMBERS = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41]

# Wells to ignore (empty list if none)
WELLS_TO_IGNORE = []  # Example: ['A1', 'B2', 'C3']

# Composition Parameters
COMPOSITIONS_PER_ITERATION = 96  # Number of compositions per iteration (typically 96 for 8x12 plate)

# Folder Structure (relative to script location or absolute paths)
BASE_DIRECTORY = "data"  # Main data directory
PEAK_DATA_DIR = "peak_data"  # Folder for peak data CSV files (e.g., "iteration 1 peak data.csv")
COMPOSITIONS_DIR = "compositions"  # Folder for composition CSV files
INITIAL_DATASETS_DIR = "initial_datasets"  # Folder with initial datasets (auto-generated)
NEXT_ITERATIONS_DIR = "next_iterations"  # Folder with iteration data files (auto-generated)
MULTIPLE_PEAKS_DIR = "multiple_peaks"  # Folder with multiple peaks data (auto-generated)
COMBINED_SET_DIR = "combined_set"  # Folder with combined peak data (auto-generated)
UPDATED_DATASETS_DIR = "updated_datasets"  # Output folder for combined datasets
RESULTS_DIR = "results"  # Output folder for results

# File Names
DATA_FILE_NAME = "luminescence_data.csv"  # Main luminescence data file (if using raw data)
COMPOSITION_FILE_NAME = "compositions.csv"  # Default composition file name
PEAK_DATA_FILE_NAME = ""  # Used only by process_peak_file.py (standalone); main.py uses peak_data folder

# Peak Data Parameters (if using pre-fitted peak data)
USE_PEAK_DATA = True  # Set to True to use pre-fitted peak data instead of raw luminescence data
AUTO_PROCESS_PEAK_DATA = True  # Automatically process all peak data files in peak_data folder
INITIAL_READ = 1  # Read number for initial measurement
FINAL_READ = 2  # Read number for final measurement
COMPOSITION_COLUMNS = None  # Column names in composition file (None = auto-detect from file)
# If auto-detection fails, set this to a list like ['Cs', 'NEA'] to manually specify

# Peak Data File Naming Convention
# Files should be named like: "iteration 1 peak data.csv", "iteration 2 peak data.csv", etc.
# Or: "iteration1_peak_data.csv", "iteration_1_peak.csv", etc.
# The code will extract the iteration number from the filename

# GP Model Parameters
BATCH_SIZE = 30
UCB_BETA = 5
GRID_RESOLUTION = 0.01  # Grid step size for search space

# Concentration Ranges (Independent - NOT ternary constraint)
# PbBr is fixed at 0.2 and not optimized
PBBR_FIXED = 0.2  # Fixed concentration for PbBr

# Independent concentration ranges for optimization:
CS_MIN = 0.1   # Cs minimum concentration (10%)
CS_MAX = 0.18   # Cs maximum concentration (20%)
NEA_MIN = 0.0  # NEA minimum concentration (0%)
NEA_MAX = 0.4  # NEA maximum concentration (40%)
CE_MIN = 0.0    # CE (Crown Ether) minimum concentration (0 M)
CE_MAX = 0.02   # CE (Crown Ether) maximum concentration (0.02 M)
# Compositions and peak data with CE > CE_MAX are excluded from analysis

NOISE_PRIOR_SCALE = 0.1  # Scale parameter for GP noise prior (HalfNormal distribution)

# Instability Score Parameters
TARGET_WAVELENGTH = 460   # Target peak wavelength (nm). Model recommends compositions predicted closest to this.
WAVELENGTH_TOLERANCE = 3  # Used in stability scoring

# Optimization Objective Weights (must sum to 1.0)
# Set WAVELENGTH_WEIGHT=1.0 to select compositions predicted closest to TARGET_WAVELENGTH only
# Lower values add intensity as secondary criterion (may recommend high-intensity but off-target compositions)
WAVELENGTH_WEIGHT = 1.0   # Weight for wavelength-position objective (1.0 = only optimize for target wavelength)
INTENSITY_WEIGHT = 0.0   # Weight for intensity objective
DEGRADATION_WEIGHT = 0.1
POSITION_WEIGHT = 0.8
MULTIPLE_PEAK_PENALTY = 0.5

# Output Settings
SAVE_PLOTS = True
PLOT_DPI = 200
DELETE_OLD_FILES_AFTER_DAYS = 1
