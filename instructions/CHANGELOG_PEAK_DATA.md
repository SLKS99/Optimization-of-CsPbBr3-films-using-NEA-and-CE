# Peak Data Processing Feature - Changelog

## Overview

Added support for using pre-fitted peak data instead of raw luminescence data. This allows users to skip the peak fitting step if they already have peak-fitted data from another source.

## New Files Created

1. **`peak_data_processor.py`**
   - Main module for processing peak-fitted CSV files
   - Functions to extract initial/final peaks, create datasets, identify multiple peaks

2. **`process_peak_file.py`**
   - Standalone script to process a single peak data file
   - Creates initial datasets automatically

3. **`PEAK_DATA_GUIDE.md`**
   - Comprehensive guide for using peak data
   - Format requirements, setup instructions, troubleshooting

## Modified Files

1. **`config.py`**
   - Added `USE_PEAK_DATA` flag
   - Added `PEAK_DATA_FILE_NAME` parameter
   - Added `INITIAL_READ` and `FINAL_READ` parameters
   - Added `COMPOSITION_COLUMNS` parameter

2. **`main.py`**
   - Updated `load_and_prepare_data()` to handle peak data mode
   - Automatically processes peak data if `USE_PEAK_DATA = True`
   - Creates initial datasets on-the-fly

3. **`QUICKSTART.md`**
   - Added Option B for using peak data
   - Instructions for peak data processing

## Key Features

### Automatic Peak Extraction
- Extracts primary (highest intensity) peak from each measurement
- Handles missing peaks gracefully
- Supports up to 5 peaks per measurement

### Multiple Peak Detection
- Automatically identifies compositions with multiple peaks
- Uses `Total_Quality_Peaks` column from CSV
- Creates structured numpy array for multiple peaks data

### Dataset Creation
- **Intensity Dataset**: Composition values + initial peak intensity
- **Peak Dataset**: Initial and final peak positions/intensities
- **Multiple Peaks**: List of compositions with multiple peaks

### Integration
- Seamlessly integrates with existing iteration workflow
- Automatically combines with iteration datasets
- Works with all existing GP analysis functions

## Usage

### Quick Start

1. Set in `config.py`:
   ```python
   USE_PEAK_DATA = True
   PEAK_DATA_FILE_NAME = "your_peak_data.csv"
   ```

2. Run processor:
   ```bash
   python process_peak_file.py
   ```

3. Run analysis:
   ```bash
   python main.py
   ```

## File Format Requirements

Peak data CSV should have:
- `Read` column (1, 2, etc.)
- `Well` column (A1, A2, etc.)
- `Total_Quality_Peaks` column
- `Peak_1_Wavelength`, `Peak_1_Intensity` columns
- `Peak_2_Wavelength`, `Peak_2_Intensity` columns (if multiple peaks)
- Additional peak columns as needed

## Backward Compatibility

- Original raw luminescence data workflow still works
- Set `USE_PEAK_DATA = False` to use original workflow
- No changes required to existing code if not using peak data

## Benefits

1. **Faster Processing**: Skip peak fitting step
2. **Flexibility**: Use peak data from any source
3. **Consistency**: Same output format regardless of input type
4. **Automation**: Automatic dataset creation
5. **Integration**: Works seamlessly with iteration workflow
