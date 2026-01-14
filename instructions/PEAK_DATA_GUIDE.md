# Peak Data Processing Guide

This guide explains how to use pre-fitted peak data instead of raw luminescence data.

## Overview

If you already have peak-fitted data (e.g., from a peak fitting software), you can use it directly instead of processing raw luminescence data. The peak data processor will:

1. Extract initial and final peak positions/intensities
2. Identify compositions with multiple peaks
3. Create the required datasets automatically

## Peak Data File Format

Your peak data CSV file should have the following columns:

- **Read**: Read number (1 for initial, 2 for final, etc.)
- **Well**: Well identifier (A1, A2, B1, etc.)
- **Composition**: Composition identifier (optional, can be same as Well)
- **Total_Quality_Peaks**: Number of peaks detected
- **Peak_1_Wavelength**: Wavelength of primary peak
- **Peak_1_Intensity**: Intensity of primary peak
- **Peak_1_FWHM**: Full width at half maximum (optional)
- **Peak_2_Wavelength**: Wavelength of second peak (if multiple peaks)
- **Peak_2_Intensity**: Intensity of second peak (if multiple peaks)
- **Peak_3_Wavelength**, **Peak_3_Intensity**, etc. (for additional peaks)

### Example Format

```csv
Read,Composition,Well,R_squared,Total_Quality_Peaks,Peak_1_Wavelength,Peak_1_Intensity,Peak_2_Wavelength,Peak_2_Intensity,...
1,A1,A1,0.996,1,519.23,4500.88,,,,...
2,A1,A1,0.996,1,519.23,3638.21,,,,...
1,D2,D2,0.999,2,479.9,4136.59,506.87,5688.39,...
```

## Setup Instructions

### Step 1: Set Up Folders

Run the setup script to create the folder structure:

```bash
python setup_folders.py
```

This creates:
- `data/peak_data/` - for your peak data CSV files
- `data/compositions/` - for your composition CSV files

### Step 2: Configure Settings

Edit `config.py`:

```python
# Set to True to use pre-fitted peak data
USE_PEAK_DATA = True

# Automatically process all peak data files in peak_data folder
AUTO_PROCESS_PEAK_DATA = True

# Specify which reads to use (typically 1=initial, 2=final)
INITIAL_READ = 1
FINAL_READ = 2

# Composition column names (adjust based on your composition file)
COMPOSITION_COLUMNS = ['CsPbI4', 'PEA2PbI4']
```

### Step 3: Place Your Files

1. **Peak Data Files**: Place your peak data CSV files in `data/peak_data/`
   - Name files with iteration numbers: `iteration 1 peak data.csv`, `iteration 2 peak data.csv`, etc.
   - Or use patterns like: `iteration1_peak_data.csv`, `iteration_2_peak.csv`
   - The code will automatically extract iteration numbers from filenames

2. **Composition File**: Place your composition CSV file in `data/compositions/`
   - Name it `compositions.csv` (or any name ending in `.csv`)

### Step 4: Run Analysis

Simply run:

```bash
python main.py
```

The script will:
- Automatically find all peak data files in `data/peak_data/`
- Extract iteration numbers from filenames
- Process each file to create datasets
- Combine all iterations automatically
- Run the Dual GP analysis

### File Naming Examples

The code recognizes iteration numbers from various naming patterns:

- ✅ `iteration 1 peak data.csv` → Iteration 1
- ✅ `iteration 2 peak data.csv` → Iteration 2
- ✅ `iteration1_peak_data.csv` → Iteration 1
- ✅ `iteration_2_peak.csv` → Iteration 2
- ✅ `iter1.csv` → Iteration 1
- ✅ `1_peak_data.csv` → Iteration 1

Files without iteration numbers are treated as iteration 0 (initial dataset).

## How It Works

### 1. Peak Extraction

The processor extracts the primary (highest intensity) peak from each measurement:
- **Initial peak**: From Read 1 (or specified `INITIAL_READ`)
- **Final peak**: From Read 2 (or specified `FINAL_READ`)

### 2. Multiple Peak Detection

Compositions with `Total_Quality_Peaks >= 2` in either initial or final read are flagged as having multiple peaks.

### 3. Dataset Creation

**Intensity Dataset** (`df_intensity_initial_PEA.csv`):
- Columns: Composition components (e.g., CsPbI4, PEA2PbI4), Intensity initial
- One row per well
- Uses initial peak intensity

**Peak Dataset** (`df_combined_initial_PEA.csv`):
- Columns: initial_peak_positions, final_peak_positions, initial_peak_intensities, final_peak_intensities, composition_number, iteration
- One row per well
- Contains both initial and final peak data

**Multiple Peaks File** (`multiple_peaks_initial.npy`):
- Structured numpy array
- Contains (composition_number, iteration) pairs for compositions with multiple peaks

## Troubleshooting

### "Well not found in composition data"

- Check that well names in peak data match those in composition file
- Verify composition file has well names as columns
- Check `WELLS_TO_IGNORE` in config.py

### "Composition values are zero"

- Verify composition file structure matches expected format
- Check `COMPOSITION_COLUMNS` in config.py matches your composition file
- Ensure composition file has correct index (component names)

### "Peak data not found"

- Verify `PEAK_DATA_FILE_NAME` in config.py matches your file name
- Check file is in the `data/` folder
- Ensure file extension is `.csv`

### Multiple peaks not detected

- Check `Total_Quality_Peaks` column exists in your CSV
- Verify values are numeric (not strings)
- Check that reads with multiple peaks have `Total_Quality_Peaks >= 2`

## Integration with Iterations

After processing your initial peak data, you can add iteration files:

1. **For each iteration**, create peak data files:
   - `data/next_iterations/df_intensity_N_PEA.csv` (N = iteration number)
   - `data/combined_set/df_combined_N.csv`
   - `data/multiple_peaks/multiple_peaks_N.npy`

2. The main script will automatically combine all iterations when you run `python main.py`

## Example Workflow

```bash
# 1. Configure config.py
USE_PEAK_DATA = True
PEAK_DATA_FILE_NAME = "my_peak_data.csv"

# 2. Process peak data
python process_peak_file.py

# 3. Run analysis
python main.py

# 4. After experiments, add iteration files and run again
python main.py  # Automatically combines iterations
```

## Notes

- The processor automatically handles missing peaks (uses 0.0)
- Primary peak is selected as the highest intensity peak
- Well ordering is consistent (sorted alphabetically)
- Composition values are divided by 4 (adjust in code if needed)
