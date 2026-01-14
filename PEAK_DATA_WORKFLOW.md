# Peak Data Processing Workflow

This document explains how peak data is processed to create the required datasets, matching the workflow from your notebook.

## Overview

Since peak fitting is already done in your peak data CSV files, the code extracts peak information directly without performing peak fitting.

## Processing Steps

### 1. Load Peak Data
- Reads CSV file with columns: `Read`, `Well`, `Peak_1_Wavelength`, `Peak_1_Intensity`, `Total_Quality_Peaks`, etc.
- Separates into initial read (Read 1) and final read (Read 2)

### 2. Extract All Peaks
For each well, extracts up to 5 peaks:
- `Peak_1_Wavelength`, `Peak_1_Intensity`
- `Peak_2_Wavelength`, `Peak_2_Intensity`
- `Peak_3_Wavelength`, `Peak_3_Intensity`
- `Peak_4_Wavelength`, `Peak_4_Intensity`
- `Peak_5_Wavelength`, `Peak_5_Intensity`

### 3. Identify Multiple Peaks
**Before reducing to max peak**, identifies compositions with multiple peaks:
- Counts non-zero peaks in each composition
- Marks compositions with 2+ peaks as having multiple peaks
- Checks both initial and final reads

### 4. Reduce to Max Peak
For each composition:
- If multiple peaks exist, keeps only the **maximum peak** (highest intensity)
- Sets all other peaks to zero
- Applied to both positions and intensities

### 5. Extract Max Values
- Uses `np.max(..., axis=1)` to get single peak position/intensity per composition
- Results in:
  - `peak_positions_int`: Initial peak positions
  - `peak_positions_fin`: Final peak positions
  - `peak_intensities_int`: Initial peak intensities
  - `peak_intensities_fin`: Final peak intensities

### 6. Create Datasets

#### Intensity Dataset (`df_intensity_initial_PEA.csv`)
Columns:
- Composition columns (e.g., `Cs`, `NEA`) - auto-detected from composition file
- `Intensity initial` - Max peak intensity from Read 1
- `Intensity final` - Max peak intensity from Read 2

#### Peak Dataset (`df_combined_initial_PEA.csv`)
Columns:
- `initial_peak_positions` - Max peak position from Read 1
- `final_peak_positions` - Max peak position from Read 2
- `initial_peak_intensities` - Max peak intensity from Read 1
- `final_peak_intensities` - Max peak intensity from Read 2
- `composition_number` - Composition identifier (1-96)
- `iteration` - Iteration number (0 for initial)

#### Multiple Peaks (`multiple_peaks_initial.npy`)
- Structured numpy array
- Contains `(composition_number, iteration)` pairs for compositions with multiple peaks
- Identified BEFORE reducing to max peak

## Key Differences from Previous Code

1. **Extracts ALL peaks first** (up to 5 peaks per composition)
2. **Identifies multiple peaks BEFORE reducing** (matches notebook)
3. **Reduces to max peak** (keeps only highest intensity peak)
4. **Includes final intensity** in intensity dataset
5. **Uses np.max()** to extract single values from reduced arrays

## Example

For a composition with:
- Read 1: Peak 1 at 519nm (intensity 4500), Peak 2 at 520nm (intensity 100)
- Read 2: Peak 1 at 519nm (intensity 3600)

Processing:
1. Extract: `[519, 520, 0, 0, 0]` positions, `[4500, 100, 0, 0, 0]` intensities
2. Identify: Multiple peaks detected (2 peaks)
3. Reduce: `[519, 0, 0, 0, 0]` positions, `[4500, 0, 0, 0, 0]` intensities
4. Extract max: `519` position, `4500` intensity
5. Save: Composition marked as having multiple peaks, but uses max peak (519nm, 4500)

## File Output

All files are automatically created in:
- `data/initial_datasets/` for iteration 0
- `data/next_iterations/` for iteration > 0

The code matches your notebook workflow exactly!
