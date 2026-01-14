# Folder Structure Guide

## Overview

The Dual GP analysis uses an organized folder structure to manage peak data, compositions, and generated datasets.

## Directory Structure

```
data/
├── peak_data/              # INPUT: Place your peak data CSV files here
│   ├── iteration 1 peak data.csv
│   ├── iteration 2 peak data.csv
│   └── iteration 3 peak data.csv
│
├── compositions/           # INPUT: Place your composition CSV files here
│   └── compositions.csv
│
├── initial_datasets/       # AUTO-GENERATED: Initial iteration datasets
│   ├── df_intensity_initial_PEA.csv
│   ├── df_combined_initial_PEA.csv
│   └── multiple_peaks_initial.npy
│
├── next_iterations/       # AUTO-GENERATED: Iteration intensity datasets
│   ├── df_intensity_1_PEA.csv
│   ├── df_intensity_2_PEA.csv
│   └── ...
│
├── multiple_peaks/        # AUTO-GENERATED: Multiple peaks data
│   ├── multiple_peaks_1.npy
│   ├── multiple_peaks_2.npy
│   └── ...
│
├── combined_set/          # AUTO-GENERATED: Combined peak datasets
│   ├── df_combined_1.csv
│   ├── df_combined_2.csv
│   └── ...
│
├── updated_datasets/      # AUTO-GENERATED: Combined datasets from all iterations
│   ├── intensity_combined_full.csv
│   ├── combined_peak_data_full.csv
│   └── multiple_peaks_updated_*.npy
│
└── results/               # OUTPUT: Analysis results
    ├── next_batch_compositions.csv
    └── dual_gp_analysis.png
```

## Folder Descriptions

### Input Folders (You Manage)

#### `peak_data/`
- **Purpose**: Store your peak-fitted CSV files
- **Naming**: Use iteration numbers in filenames
  - Examples: `iteration 1 peak data.csv`, `iteration 2 peak data.csv`
- **Format**: CSV files with columns: Read, Well, Peak_1_Wavelength, Peak_1_Intensity, Total_Quality_Peaks, etc.
- **Processing**: Files are automatically processed when you run `main.py`

#### `compositions/`
- **Purpose**: Store your composition CSV files
- **Naming**: Any name ending in `.csv` (default: `compositions.csv`)
- **Format**: CSV with component names as rows, well names as columns
- **Usage**: One composition file is used for all iterations

### Auto-Generated Folders (Code Manages)

#### `initial_datasets/`
- **Purpose**: Stores datasets from iteration 0 (initial/starting data)
- **Created**: When processing peak data files with iteration 0 or no iteration number
- **Files**:
  - `df_intensity_initial_PEA.csv` - Initial intensity measurements
  - `df_combined_initial_PEA.csv` - Initial peak positions/intensities
  - `multiple_peaks_initial.npy` - Initial multiple peaks data

#### `next_iterations/`
- **Purpose**: Stores intensity datasets for each iteration (> 0)
- **Created**: When processing peak data files with iteration numbers > 0
- **Files**: `df_intensity_N_PEA.csv` where N is the iteration number

#### `multiple_peaks/`
- **Purpose**: Stores multiple peaks data for each iteration
- **Created**: When processing peak data files
- **Files**: `multiple_peaks_N.npy` where N is the iteration number

#### `combined_set/`
- **Purpose**: Stores combined peak datasets for each iteration
- **Created**: When processing peak data files
- **Files**: `df_combined_N.csv` where N is the iteration number

#### `updated_datasets/`
- **Purpose**: Stores combined datasets from all iterations
- **Created**: When running the main analysis
- **Files**:
  - `intensity_combined_full.csv` - All intensity data combined
  - `combined_peak_data_full.csv` - All peak data combined
  - `multiple_peaks_updated_*.npy` - All multiple peaks data combined

#### `results/`
- **Purpose**: Stores analysis output
- **Created**: When running the main analysis
- **Files**:
  - `next_batch_compositions.csv` - Recommended compositions for next iteration
  - `dual_gp_analysis.png` - Visualization plots

## Workflow

1. **Setup**: Run `python setup_folders.py` to create all folders

2. **Add Data**:
   - Place peak data files in `data/peak_data/`
   - Place composition file in `data/compositions/`

3. **Run Analysis**: Run `python main.py`
   - Automatically processes all peak data files
   - Creates datasets in auto-generated folders
   - Combines all iterations
   - Generates results

4. **Add More Iterations**:
   - Add new peak data files to `data/peak_data/`
   - Run `python main.py` again
   - New iterations are automatically detected and processed

## File Naming Conventions

### Peak Data Files
- Include iteration number in filename
- Supported patterns:
  - `iteration 1 peak data.csv`
  - `iteration1_peak_data.csv`
  - `iteration_1_peak.csv`
  - `iter1.csv`
- Files without iteration numbers → treated as iteration 0

### Composition Files
- Any name ending in `.csv`
- Default: `compositions.csv`
- If multiple files exist, first one found is used

## Best Practices

1. **Keep Original Files**: Don't delete peak data files after processing
2. **Consistent Naming**: Use consistent naming pattern for peak data files
3. **Backup**: Keep backups of your input files
4. **Version Control**: Consider versioning your composition files if they change
5. **Clean Up**: Old files in `updated_datasets/` are automatically cleaned up (configurable)

## Troubleshooting

### Files Not Found
- Check folder names match exactly (case-sensitive on some systems)
- Verify files are in correct folders
- Check file extensions (.csv)

### Iteration Numbers Not Detected
- Ensure iteration number is in filename
- Check naming pattern matches supported formats
- Files without iteration numbers default to iteration 0

### Wrong Composition File Used
- Check `COMPOSITION_FILE_NAME` in `config.py`
- Ensure only one composition file is in `compositions/` folder
- Or specify exact filename in config
