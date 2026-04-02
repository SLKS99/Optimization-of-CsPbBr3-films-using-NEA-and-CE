# Monte Carlo Decision Tree for Perovskite Processing

This tool uses a Monte Carlo simulation combined with a physics-based decision tree to generate and filter experimental recipes for perovskite spin coating.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install pandas
    ```

2.  **Data Input**:
    *   **Solvents**: Edit `data/templates/solvents.csv` to add available solvents and their Hansen Solubility Parameters (HSP), Viscosity, Boiling Point, etc.
    *   **Precursors**: Edit `data/templates/precursors.csv` to define your A, B, and X site materials (Ionic Radius, etc.).

## How to start (recommended)

Read `START_HERE.md`, then run:

```bash
python main.py
```

## How It Works

1.  **Generator**: Randomly samples combinations of precursors (A, B, X), solvents, and processing parameters (Spin Speed, Concentration, Temperature, etc.).
2.  **Physics Constraints**:
    *   **Structural Stability**: Checks Goldschmidt Tolerance Factor ($t$) and Octahedral Factor ($\mu$) to ensure the material can form a perovskite structure.
    *   **Solvent Compatibility**: Checks Boiling Point and Donor Number (DN). (HSP distance logic included but optional based on data).
    *   **Process Feasibility**: Checks drop volume vs concentration limits and antisolvent miscibility.
3.  **Output**:
    *   Generates a list of valid experimental conditions.
    *   Saves a robot-readable CSV recipe at `data/robot_recipe_generated.csv`.

## Extending the Logic

*   **Formulas**: Edit `src/physics.py` to refine the scientific calculations.
*   **Constraints**: Edit `src/constraints.py` to change the acceptance criteria (e.g., tighten tolerance factor range).
*   **Sampling**: Edit `src/generator.py` to change the range of random values (e.g., if you have a faster spin coater).

