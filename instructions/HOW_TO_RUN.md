# How to Run the Dual GP Analysis

## Quick Start

### Step 1: Install Dependencies

Open a terminal/command prompt in this directory and run:

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- numpy, pandas, scipy
- matplotlib, scikit-learn
- python-ternary
- atomai, GPax, jax, numpyro
- openpyxl (for Excel file support)

### Step 2: Verify Your Files Are in Place

Make sure you have:
- ✅ Peak data file: `data/peak_data/Iteration 1 Peak Data.csv`
- ✅ Composition file: `data/compositions/Iteration 1 compositions.xlsx`

### Step 3: Run the Analysis

Simply run:

```bash
python main.py
```

That's it! The script will:
1. ✅ Automatically find and process your peak data file
2. ✅ Auto-detect composition column names from your Excel file
3. ✅ Create all required datasets
4. ✅ Train GP models
5. ✅ Generate next batch recommendations
6. ✅ Save results and visualizations

## What Happens When You Run

The script will:

1. **Load Data**:
   - Finds `Iteration 1 Peak Data.csv` in `data/peak_data/`
   - Extracts iteration number: **1**
   - Finds matching composition file: `Iteration 1 compositions.xlsx`
   - Auto-detects component names (Cs, NEA, CE)

2. **Process Peak Data**:
   - Extracts initial (Read 1) and final (Read 2) peaks
   - Creates intensity dataset
   - Creates peak dataset
   - Identifies multiple peaks

3. **Run Dual GP Analysis**:
   - Trains intensity GP model
   - Calculates acquisition function
   - Calculates instability scores
   - Trains stability GP model
   - Generates next batch recommendations

4. **Save Results**:
   - `data/results/next_batch_compositions.csv` - Recommended compositions
   - `data/results/dual_gp_analysis.png` - Visualization plots

## Output Files

After running, check `data/results/`:

### `next_batch_compositions.csv`
Contains recommended compositions for your next iteration:
```csv
Well,Cs,NEA,CE
A1,30.0,30.0,40.0
A2,35.0,25.0,40.0
...
```

### `dual_gp_analysis.png`
Visualization showing:
- Test predictions
- Acquisition function
- Adjusted acquisition
- Stability scores

## Troubleshooting

### "Module not found" error
```bash
pip install -r requirements.txt
```

### "File not found" error
- Check that files are in correct folders:
  - `data/peak_data/Iteration 1 Peak Data.csv`
  - `data/compositions/Iteration 1 compositions.xlsx`

### "No composition file found" error
- Make sure composition file is in `data/compositions/` folder
- Check filename matches iteration number

### Excel file errors
```bash
pip install openpyxl
```

## Adding More Iterations

After running experiments:

1. Add new peak data file: `data/peak_data/Iteration 2 Peak Data.csv`
2. Add new composition file: `data/compositions/Iteration 2 compositions.xlsx`
3. Run `python main.py` again
4. The script automatically combines all iterations!

## Command Line Options

Currently, all settings are in `config.py`. To change parameters:

1. Edit `config.py`
2. Run `python main.py`

## Example Output

```
============================================================
Dual GP Analysis - Automated Workflow
============================================================
Directory structure verified.

=== Loading Data ===
Auto-processing peak data files...

Found 1 peak data file(s):
  Iteration 1: Iteration 1 Peak Data.csv

Processing peak data files...

Processing iteration 1...
  Loading peak data from: Iteration 1 Peak Data.csv
  Loading composition data from: Iteration 1 compositions.xlsx
  Auto-detected composition columns from file: ['Cs', 'NEA']
  Created: df_intensity_1_PEA.csv
  Created: df_combined_1.csv
  Created: multiple_peaks_1.npy (5 multiple peaks)

✓ Successfully processed 1 file(s)

=== Combining Iteration Datasets ===
...

=== Running Dual GP Analysis ===
Creating search grid...
Training intensity GP model...
Calculating acquisition function...
Calculating instability scores...
Training stability GP model...
Selecting next batch of experiments...
Selected 96 compositions for next iteration.

=== Saving Results ===
Next batch compositions saved to: data/results/next_batch_compositions.csv

=== Creating Visualizations ===
Visualization saved to: data/results/dual_gp_analysis.png

============================================================
Analysis Complete!
Results saved to: data/results
============================================================
```

## Need Help?

- Check `instructions/QUICKSTART.md` for detailed setup
- Check `instructions/CONFIG_PARAMETERS.md` for all config options
- Check `instructions/PEAK_DATA_GUIDE.md` for peak data format details
