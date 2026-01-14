# Dual GP Analysis - User-Friendly Version

This is an automated, user-friendly version of the Dual Gaussian Process (GP) method for analyzing luminescence datasets and predicting optimal compositions for next iterations.

## Features

- **Automated Workflow**: Runs the entire analysis pipeline automatically
- **Local File Support**: Uses local folders instead of Google Drive
- **Iteration Management**: Automatically detects and combines data from multiple iterations
- **Easy Configuration**: Simple configuration file for all parameters
- **Clear Output**: Generates CSV files with next batch predictions and visualization plots

## Installation

### Requirements

Install the required Python packages:

```bash
pip install numpy pandas scipy matplotlib scikit-learn python-ternary atomai GPax jax numpyro
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

## Setup

### 1. Folder Structure

Create the following folder structure in your project directory:

```
your_project/
├── data/                          # Main data directory
│   ├── luminescence_data.csv      # Your luminescence measurement data
│   ├── compositions.csv            # Your composition data
│   ├── initial_datasets/          # Initial datasets
│   │   ├── df_intensity_initial_PEA.csv
│   │   ├── df_combined_initial_PEA.csv
│   │   └── multiple_peaks_initial.npy
│   ├── next_iterations/           # Iteration data files
│   │   ├── df_intensity_1_PEA.csv
│   │   ├── df_intensity_2_PEA.csv
│   │   └── ...
│   ├── multiple_peaks/            # Multiple peaks data
│   │   ├── multiple_peaks_1.npy
│   │   └── ...
│   ├── combined_set/              # Combined peak data
│   │   ├── df_combined_1.csv
│   │   └── ...
│   ├── updated_datasets/          # Auto-generated combined datasets
│   └── results/                   # Output folder (auto-created)
├── config.py                      # Configuration file
├── main.py                        # Main script
├── data_loader.py                 # Data loading module
├── gp_models.py                   # GP model functions
├── instability_scoring.py         # Instability scoring functions
└── iteration_manager.py           # Iteration management
```

### 2. Configuration

Edit `config.py` to match your experimental setup:

- **Experimental Parameters**: Precursor names, measurement ranges, time steps
- **File Names**: Names of your data files
- **Folder Paths**: Adjust if using different folder structure
- **GP Parameters**: Model parameters, batch size, acquisition function settings
- **Instability Score Parameters**: Target wavelength, weights, etc.

### 3. Data Files

#### Luminescence Data (`luminescence_data.csv`)
Your CSV file should contain luminescence measurements with format:
- Rows labeled as "Read 1:EM Spectrum", "Read 2:EM Spectrum", etc.
- Columns labeled with well names (A1, A2, B1, etc.)
- Wavelength data in rows

#### Composition Data (`compositions.csv`)
CSV file with:
- Index: Component names (e.g., CsPbI3, PEA2PbI4, BDAPbI4)
- Columns: Well names (A1, A2, B1, etc.)
- Values: Composition values (will be divided by 4 in the code)

#### Initial Datasets
Place your initial datasets in `initial_datasets/`:
- `df_intensity_initial_PEA.csv`: Initial intensity data
- `df_combined_initial_PEA.csv`: Initial combined peak data
- `multiple_peaks_initial.npy`: Initial multiple peaks data

#### Iteration Files
For each iteration, add files to respective folders:
- `next_iterations/df_intensity_N_PEA.csv` (N = iteration number)
- `combined_set/df_combined_N.csv`
- `multiple_peaks/multiple_peaks_N.npy`

## Usage

### Basic Usage

Simply run the main script:

```bash
python main.py
```

The script will:
1. Load and combine all iteration datasets
2. Train GP models
3. Calculate acquisition functions
4. Generate next batch predictions
5. Save results and visualizations

### Output Files

Results are saved in `data/results/`:
- `next_batch_compositions.csv`: Recommended compositions for next iteration
- `dual_gp_analysis.png`: Visualization of predictions and acquisition functions

## File Format Examples

### Next Batch Compositions CSV

```csv
Well,CsPbI3,PEA2PbI4,BDAPbI4
A1,24.0,0.0,76.0
A2,80.0,0.0,20.0
...
```

### Intensity Dataset CSV

Should contain columns:
- `CsPbI4`: First component value
- `PEA2PbI4`: Second component value
- `Intensity initial`: Initial intensity value
- (Other columns as needed)

### Combined Peak Dataset CSV

Should contain columns:
- `initial_peak_positions`: Initial peak wavelength
- `final_peak_positions`: Final peak wavelength
- `initial_peak_intensities`: Initial peak intensity
- `final_peak_intensities`: Final peak intensity
- `composition_number`: Composition identifier
- `iteration`: Iteration number

## Customization

### Changing Parameters

Edit `config.py` to adjust:
- Measurement parameters (wavelength range, time steps)
- GP model parameters (batch size, UCB beta, grid resolution)
- Instability scoring weights
- File paths and names

### Adding New Features

The code is modular:
- `data_loader.py`: Modify data loading/preprocessing
- `gp_models.py`: Adjust GP model training
- `instability_scoring.py`: Change scoring functions
- `iteration_manager.py`: Modify iteration handling

## Troubleshooting

### File Not Found Errors

- Check that all required files are in the correct folders
- Verify file names match those in `config.py`
- Ensure folder structure matches the setup instructions

### Data Format Errors

- Verify CSV files have correct headers
- Check that well names match expected format (A1, A2, etc.)
- Ensure composition values are numeric

### Import Errors

- Install all required packages: `pip install -r requirements.txt`
- Check Python version (3.8+ recommended)

## Citation

If you use this code, please cite:
- Dual GP method: [Yongtao Liu et al.](https://atomai.readthedocs.io/)
- Original analysis: [Sheryl L. Sanchez and Jonghee Yang](https://scholar.google.com/citations?hl=en&user=zFRKT-MAAAAJ)

## License

[Add your license information here]

## Contact

For questions or issues, please [add contact information]
