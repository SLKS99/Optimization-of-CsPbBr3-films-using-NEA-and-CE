# Quick Start Guide (simplified): start a new experiment

## Overview (current architecture)

This repo now uses **two** components:

1. **Monte Carlo + Random Forest**: explores **processing** (spin coating + anneal).  
   This is the “process optimizer”. There is **no GP trained on processing**.
2. **Dual Composition GP (Cs/NEA/CE)**: explores **composition additives** with uncertainty + acquisition.  
   This guides which Cs/NEA/CE concentrations are preferred while Monte Carlo varies the process.

---

## How to run

### Step 1: Configure the experiment

Edit `config.yaml`. For this experiment family, the important parts are:

- Fixed settings: solvent (DMSO), concentration (0.15 M), antisolvent (CB), drop timing rule.
- Composition Dual‑GP ranges: `composition_dual_gp` (Cs/NEA/CE ranges + grid resolution).

### Step 2: Run one optimization cycle

```bash
python main.py
```

This will:
1. Load experiment history (`data/templates/experiments_log.csv`)
2. Train the **Random Forest process model** (once you have enough history)
3. Train the **composition Dual‑GP** (once you have ~5 rows with `Cs_M/NEA_M/CE_M`)
4. Generate candidates using Monte Carlo
5. Rank candidates using physics + Random Forest + composition Dual‑GP boost
6. Output recipes/instructions for the next batch

### Step 3: Add results and repeat

After you run experiments, append rows to `data/templates/experiments_log.csv`.  
Make sure you fill:

- `Cs_M`, `NEA_M`, `CE_M`
- `PL_Peak_nm`, `PL_Intensity `, `PL_FWHM_nm`
- optional delayed feedback: `PL_Peak_3d_nm`, `PL_Intensity_3d_nm`, `PL_FWHM_3d_nm`

Then run `python main.py` again.

### Step 4: Update plots without running optimization

```bash
python update_plots.py
```

This regenerates all plots from existing data without running a new optimization cycle. Useful for:
- Viewing updated plots after adding new experiment results
- Checking model learning progress
- Sharing visualization with collaborators

---

## Understanding the Plots

The dashboard (`data/optimization_plot.png`) now includes:

### Row 1: Process GP Learning
- **Left**: GP approximation showing quality vs FA ratio (with uncertainty bands)
- **Middle**: Acquisition function (UCB) showing exploration vs exploitation
- **Right**: Cycle-by-cycle improvement tracking

### Row 2: Search Space & Uncertainty
- **Left**: Process parameter exploration map (temp vs spin speed)
- **Middle**: Uncertainty heatmap (where GP is most uncertain)
- **Right**: Quality progress over experiments

### Row 3: Diagnostics & Grouped Analysis
- **Left**: GP learning diagnostics (R², signal-to-noise)
- **Middle**: Quality by temperature range
- **Right**: Quality by FA ratio range

### Row 4: Grouped Analysis & Exploration
- **Left**: Quality by antisolvent usage
- **Middle**: Quality by temperature range
- **Right**: Exploration vs Exploitation quadrant plot

### Row 5: Model Learning (NEW!)
- **Left**: **Composition GP Learning**
  - Shows FA ratio → quality predictions
  - Training data points (from binary dataset)
  - GP mean and uncertainty bands
  - UCB acquisition function
  - Best FA ratio marker
  
- **Middle**: **Decision Tree Feature Importance**
  - Shows which features the tree considers most important
  - Helps understand what the tree is learning
  
- **Right**: **Decision Tree Predictions vs Actual**
  - Scatter plot: predicted quality vs actual quality
  - R² score showing how well the tree fits the data
  - MAE and RMSE metrics

---

## What Each Model Does

### Composition GP (Stage 0)
- **Input**: FA ratio (0-1)
- **Output**: Composition score (higher = better FA range)
- **Training**: Binary dataset (`Read == 1` only)
- **Usage**: Biases Monte Carlo to sample more from high-scoring FA ranges

### Process Decision Tree (Stage 1)
- **Input**: FA ratio + annealing temp + spin speed + concentration + antisolvent flag
- **Output**: Predicted quality score
- **Training**: Completed experiments from `experiments_log.csv`
- **Usage**: Adds to `rank_score` to favor candidates with learned good process patterns

### Process GP (existing)
- **Input**: FA ratio + annealing temp + spin speed (3D)
- **Output**: Predicted quality + uncertainty
- **Training**: Completed experiments from `experiments_log.csv`
- **Usage**: Provides acquisition scores (UCB) and uncertainty estimates

---

## Workflow Summary

1. **Initial Setup**: Binary dataset → Composition GP learns good FA ranges
2. **Cycle 1**: Random Monte Carlo (with Composition GP bias) → Generate candidates
3. **Scoring**: All three models score candidates → Select top batch
4. **Experiment**: Run selected experiments → Add results to `experiments_log.csv`
5. **Cycle 2+**: Retrain all models → GP-guided Monte Carlo → Repeat

---

## Troubleshooting

### Composition Dual‑GP not training?
- Need ~5 completed rows with **`Cs_M/NEA_M/CE_M`** present
- Need at least one PL metric so a quality target can be computed (e.g. `PL_Intensity `)

### Random Forest not training?
- Need at least ~5 completed experiments in `experiments_log.csv`
- Each experiment needs: Material_Composition, Anneal_Temp_C, Spin_Speed_rpm, and at least PL_Intensity or Stability_Hours

### Plots not showing model learning?
- Run `python update_plots.py` after adding new experiment results
- Check that models are trained (look for "trained on X points" messages)
- Check that `data/candidates_analysis.csv` exists (for candidate plots)

---

## Next Steps

1. **Run your first cycle**: `python main.py`
2. **Check the plots**: Open `data/optimization_plot.png` to see how models are learning
3. **Run experiments**: Use `data/prep_instructions.txt` to prepare films
4. **Add results**: Update `experiments_log.csv` with measured PL, FWHM, stability
5. **Continue optimization**: Run `python main.py` again to start next cycle

The system will automatically:
- Retrain all models on updated data
- Use Composition GP to focus on good FA ranges
- Use Decision Tree to favor learned good process conditions
- Use Process GP to balance exploration vs exploitation
- Avoid duplicate experiments (delayed workflow)
