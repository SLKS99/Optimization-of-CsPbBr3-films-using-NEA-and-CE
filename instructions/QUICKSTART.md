# Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Set Up Folders

Run the setup script to create the folder structure:

```bash
python setup_folders.py
```

This will create:
```
data/
├── peak_data/              # Place your peak data CSV files here
├── compositions/           # Place your composition CSV files here
├── initial_datasets/       # Auto-generated from peak data
├── next_iterations/        # Auto-generated from peak data
├── multiple_peaks/         # Auto-generated from peak data
├── combined_set/           # Auto-generated from peak data
├── updated_datasets/       # Auto-generated combined datasets
└── results/                # Output folder
```

## Step 3: Add Your Data Files

### Option A: Using Raw Luminescence Data

Place your data files in the `data/` folder:

1. **Luminescence Data**: `data/luminescence_data.csv`
   - Your measurement data with reads labeled as "Read 1:EM Spectrum", etc.
   - Columns should be well names (A1, A2, B1, etc.)

2. **Composition Data**: `data/compositions.csv`
   - Composition values for each well
   - Rows: Component names (CsPbI3, PEA2PbI4, BDAPbI4)
   - Columns: Well names (A1, A2, etc.)

### Option B: Using Pre-Fitted Peak Data (Recommended)

If you already have peak-fitted data:

1. **Peak Data**: Place your peak fitting CSV files in `data/peak_data/` folder
   - Name files like: `iteration 1 peak data.csv`, `iteration 2 peak data.csv`, etc.
   - Or: `iteration1_peak_data.csv`, `iteration_2_peak.csv`, etc.
   - Should have columns: Read, Well, Peak_1_Wavelength, Peak_1_Intensity, Total_Quality_Peaks, etc.
   - See `PEAK_DATA_GUIDE.md` for detailed format requirements

2. **Composition Data**: Place your composition CSV file in `data/compositions/` folder
   - Name it: `compositions.csv` (or any name, the code will find it)
   - Same format as Option A

3. **Configure**: Edit `config.py`:
   ```python
   USE_PEAK_DATA = True
   AUTO_PROCESS_PEAK_DATA = True  # Automatically process all files
   ```

4. **Run**: The main script will automatically process all peak data files:
   ```bash
   python main.py
   ```

3. **Initial Datasets** (in `data/initial_datasets/`):
   - `df_intensity_initial_PEA.csv` - Initial intensity measurements
   - `df_combined_initial_PEA.csv` - Initial peak data
   - `multiple_peaks_initial.npy` - Initial multiple peaks data

## Step 4: Configure Parameters

Edit `config.py` to match your setup:

- Update `DATA_FILE_NAME` and `COMPOSITION_FILE_NAME` if your files have different names
- Adjust `WELLS_TO_IGNORE` if you have wells to exclude
- Set `START_WAVELENGTH`, `END_WAVELENGTH`, `TIME_STEP`, etc. to match your measurements
- Modify `PRECURSOR1`, `PRECURSOR2`, `PRECURSOR3` for your precursors

## Step 5: Run Analysis

Simply run:

```bash
python main.py
```

The script will:
- Load and combine all iteration datasets automatically
- Train GP models
- Calculate optimal next batch compositions
- Save results to `data/results/`

## Step 6: Check Results

Results are saved in `data/results/`:

- **`next_batch_compositions.csv`**: Recommended compositions for your next iteration
  - Columns: Well, CsPbI3, PEA2PbI4, BDAPbI4
  - Values are in percentages

- **`dual_gp_analysis.png`**: Visualization showing:
  - Test predictions
  - Acquisition function
  - Adjusted acquisition
  - Stability scores

## Adding New Iterations

After running experiments:

1. Add new intensity data: `data/next_iterations/df_intensity_N_PEA.csv` (N = iteration number)
2. Add new peak data: `data/combined_set/df_combined_N.csv`
3. Add multiple peaks data: `data/multiple_peaks/multiple_peaks_N.npy`
4. Run `python main.py` again - it will automatically combine all iterations!

## Troubleshooting

### "File not found" errors
- Check that files are in the correct folders
- Verify file names match `config.py`
- Ensure folder structure was created (run `setup_folders.py`)

### Import errors
- Install all packages: `pip install -r requirements.txt`
- Check Python version (3.8+)

### Data format errors
- Verify CSV headers match expected format
- Check well names are correct (A1, A2, B1, etc.)
- Ensure numeric values are actually numbers

## Example Workflow

```bash
# 1. First time setup
python setup_folders.py

# 2. Add your initial data files to data/ folder

# 3. Configure config.py

# 4. Run first analysis
python main.py

# 5. After running experiments, add iteration files:
#    - data/next_iterations/df_intensity_1_PEA.csv
#    - data/combined_set/df_combined_1.csv
#    - data/multiple_peaks/multiple_peaks_1.npy

# 6. Run analysis again (automatically combines iterations)
python main.py

# 7. Check results in data/results/
```

## Need Help?

- Check `README.md` for detailed documentation
- Review `config.py` for all configurable parameters
- Examine the code modules for customization options
