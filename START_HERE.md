## Start here (new experiment workflow)

This project now follows a simple split:

- **Monte Carlo + Random Forest**: explores **processing** (spin/anneal/etc.). No GP is trained on processing.
- **Dual Composition GP (Cs/NEA/CE)**: guides **composition additives** using uncertainty + acquisition (like the external Dual‑GP).

### What you edit first

- `config.yaml`
  - **Fixed settings** already set for this experiment family:
    - solvent = `DMSO`
    - base salt concentration (PbBr2) = `0.15 M`
    - antisolvent = `CB`
    - spin time rule = `drop_time_s + 20s`
  - Dual‑GP composition ranges are in `composition_dual_gp`.

### Your “source of truth” data file

- `data/templates/experiments_log.csv`
  - You must fill **`CsBr_M`, `NEABr_M`, `CE_M`** for each experiment row.
  - Add measured outputs: `PL_Peak_nm`, `PL_Intensity `, `PL_FWHM_nm`
  - If you have delayed feedback (3‑day), also fill `PL_Peak_3d_nm`, `PL_Intensity_3d_nm`, `PL_FWHM_3d_nm`.

### Run one iteration

From the repo root:

```bash
python main.py
```

Outputs you’ll use:
- `data/prep_instructions.txt`
- `data/robot_recipe_generated.yaml`
- `data/next_batch_solution_mix.csv` (mix table in M, like external Dual‑GP)

### When Dual‑GP starts helping

After you have about **5 completed rows** with `Cs_M/NEA_M/CE_M` + PL metrics, the composition Dual‑GP will train and start boosting candidates via `composition_acquisition_score`.

