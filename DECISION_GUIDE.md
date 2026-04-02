# Decision-Making Guide: Selecting Candidates When GP Isn't Learning

## Overview

When the Composition GP isn't learning effectively, you can still make informed decisions using:
1. **Decision Tree Predictions** (primary)
2. **Physics-Based Metrics** (always available)
3. **Quality Score Calculation** (from your experimental results)
4. **Alternative Selection Strategies**

---

## 1. Understanding Quality Scores

### Quality Target Calculation

The system calculates quality from your experimental results using `calculate_quality_target()`:

```python
Quality = (PL_weight × PL_intensity / PL_scale) 
        + (FWHM_weight × ideal_FWHM / actual_FWHM)
        + (Stability_weight × Stability_Hours)
        + (Peak_penalty if wavelength off-target)
```

**Default weights** (configurable in `config.yaml`):
- `pl_weight`: 1.0
- `pl_scale`: 100.0 (normalizes PL intensity)
- `fwhm_weight`: 50.0
- `fwhm_ideal_nm`: 20.0
- `stability_weight`: 2.0

**Interpretation**:
- Higher quality = better material
- Typical range: -100 to +500 (depends on your data)
- Negative values = poor quality (low PL, wide FWHM, etc.)

---

## 2. How Candidates Are Ranked

### Rank Score Components

```python
Rank_Score = Physics_Score 
           + (GP_Acquisition_Score × 20)  # If GP trained
           + (Tree_Predicted_Quality × 0.5)  # If Tree trained
```

**Physics Score** includes:
- Structural metrics (octahedral factor, lattice spacing)
- Solvent compatibility (Hansen parameters, miscibility)
- Processing metrics (drying time, film uniformity)
- Typical range: -50 to +150

**ML Boosts**:
- **GP Boost**: `acquisition_score × 20` (only if GP is trained and learning)
- **Tree Boost**: `tree_predicted_quality × 0.5` (always available if tree is trained)

---

## 3. Decision Strategies When GP Isn't Learning

### Strategy 1: Use Decision Tree Predictions Directly

**When to use**: GP shows flat predictions, but Decision Tree is trained (R² > 0.8)

**How to select**:
1. Check `candidates_analysis.csv` for `Tree_Predicted_Quality` column
2. Sort by `Tree_Predicted_Quality` (descending)
3. Select top N candidates

**Python code**:
```python
import pandas as pd
df = pd.read_csv('data/candidates_analysis.csv')
# Sort by tree predictions
top_candidates = df.nlargest(10, 'Tree_Predicted_Quality')
```

**Advantages**:
- Tree learns from ALL your experimental history
- Includes composition effects (FA ratio is most important feature)
- More reliable when GP isn't learning

---

### Strategy 2: Use Physics-Based Ranking

**When to use**: Early cycles, or when both GP and Tree are unreliable

**How to select**:
1. Check `candidates_analysis.csv` for physics metrics
2. Look for candidates with:
   - High `Film_Uniformity_Metric` (> 5)
   - Good `Precursor_Solvent_Delta` (< 4)
   - Optimal `Octahedral_Factor` (0.4-0.9, ideally ~0.65)
   - Moderate `Estimated_Drying_Time` (0.005-0.05)
3. Sort by `Rank_Score` (which emphasizes physics when ML isn't working)

**Advantages**:
- Always available (no training needed)
- Based on material science principles
- Good for initial exploration

---

### Strategy 3: Pareto Front Selection

**When to use**: Want to balance exploitation (high quality) and exploration (high uncertainty)

**How it works**:
- Finds candidates that are "non-dominated" in quality vs uncertainty space
- Automatically selected by `select_diverse_pareto_front()`
- Provides diverse set covering different trade-offs

**How to use**:
```python
from src.multi_objective import select_diverse_pareto_front
selected = select_diverse_pareto_front(candidates, n_select=10)
```

**Advantages**:
- Balances exploration and exploitation
- Provides diverse candidate set
- Works even when GP predictions are flat (uses uncertainty from tree or physics)

---

### Strategy 4: Manual Selection Based on Feature Importance

**When to use**: You want to target specific parameter ranges

**Based on your Decision Tree feature importance**:
1. **FA Ratio** (importance: 0.717) - Most important!
   - Focus on FA ratios that showed high quality in history
   - Check "Quality by FA Ratio Range" plot
   - Your data shows: Mid-High FA (0.6-0.8) performs best

2. **Spin Speed** (importance: 0.140)
   - Your data shows: 3000-4000 rpm performs best
   - Avoid: 4000-4500 rpm (lowest quality)

3. **Annealing Temp** (importance: 0.132)
   - Your data shows: 140-145°C performs best
   - Avoid: 135-140°C (lowest quality)

4. **Antisolvent** (importance: 0.012) - Less important
   - Your data shows: "None" performs best
   - But effect is small

**Selection criteria**:
```python
# Filter candidates
good_candidates = [
    c for c in candidates
    if 0.6 <= extract_fa_ratio(c) <= 0.8  # Best FA range
    and 3000 <= c.spin_speed <= 4000  # Best spin range
    and 140 <= c.annealing_temp <= 145  # Best temp range
]
# Then sort by tree_predicted_quality
```

---

## 4. Composition Selection (When GP Isn't Learning)

### Option A: Use Historical Best Performers

1. Check your `experiments_log.csv` for best quality experiments
2. Extract their FA ratios
3. Focus Monte Carlo generation on similar compositions

**Example**:
```python
import pandas as pd
history = pd.read_csv('data/experiments_log.csv')
# Find best experiments
best = history.nlargest(5, 'Quality_Target')
# Extract FA ratios
fa_ratios = [extract_fa_ratio_from_composition(comp) for comp in best['Material_Composition']]
# Average or use range
target_fa = np.mean(fa_ratios)  # e.g., 0.65
```

### Option B: Use Decision Tree Feature Importance

- FA Ratio is most important (0.717)
- Your plots show: Mid-High FA (0.6-0.8) performs best
- **Recommendation**: Focus on FA = 0.65-0.75 range

### Option C: Uniform Sampling (Current Fallback)

- If GP scores are all identical, system falls back to uniform random
- This is fine for exploration, but not optimal

---

## 5. Parameter Selection (Process Conditions)

### Based on Decision Tree + Grouped Plots

**Spin Speed**:
- Best: 3000-4000 rpm (highest median quality)
- Avoid: 4000-4500 rpm (lowest quality)
- **Recommendation**: 3500-3800 rpm

**Annealing Temperature**:
- Best: 140-145°C (highest median quality)
- Avoid: 135-140°C (lowest quality)
- **Recommendation**: 142-144°C

**Antisolvent**:
- Best: None (no antisolvent)
- But effect is small (importance: 0.012)
- **Recommendation**: Try both with/without, but prioritize "None"

**Concentration**:
- Very low importance (0.000)
- Use standard values from config

---

## 6. Practical Workflow

### Step-by-Step Decision Process

1. **Check Model Status**:
   ```python
   # After running optimization
   # Check console output:
   # - "GP predictions valid for FA range: X to Y"
   # - "GP predicted quality range: A to B"
   # - "Decision Tree: R² = X"
   ```

2. **If GP isn't learning** (flat predictions, small quality range):
   - **Primary**: Use Decision Tree predictions
   - **Secondary**: Use physics-based ranking
   - **Tertiary**: Manual selection based on feature importance

3. **Select Candidates**:
   ```python
   # Load candidates
   import pandas as pd
   df = pd.read_csv('data/candidates_analysis.csv')
   
   # Option 1: Sort by Tree predictions
   top_by_tree = df.nlargest(10, 'Tree_Predicted_Quality')
   
   # Option 2: Sort by Rank Score (includes physics)
   top_by_rank = df.nlargest(10, 'Rank_Score')
   
   # Option 3: Filter + sort
   filtered = df[
       (df['FA_Ratio'] >= 0.6) & (df['FA_Ratio'] <= 0.8) &
       (df['Spin_Speed_rpm'] >= 3000) & (df['Spin_Speed_rpm'] <= 4000) &
       (df['Anneal_Temp_C'] >= 140) & (df['Anneal_Temp_C'] <= 145)
   ]
   top_filtered = filtered.nlargest(10, 'Tree_Predicted_Quality')
   ```

4. **Verify Selection**:
   - Check physics metrics (should be positive)
   - Check tree predictions (should be reasonable)
   - Check parameter ranges (should match best performers)

---

## 7. Quality Score Interpretation

### What is a "Good" Quality Score?

**From your experimental results**:
- Quality is calculated from: PL intensity, FWHM, stability, peak wavelength
- Higher = better material
- Typical range depends on your data

**To understand your quality scale**:
```python
import pandas as pd
history = pd.read_csv('data/experiments_log.csv')
# Calculate quality for all experiments
from src.learner import ActiveLearner
learner = ActiveLearner()
qualities = []
for _, row in history.iterrows():
    q = learner.calculate_quality_target(row)
    qualities.append(q)

print(f"Quality range: {min(qualities):.1f} to {max(qualities):.1f}")
print(f"Mean quality: {np.mean(qualities):.1f}")
print(f"Top 10% threshold: {np.percentile(qualities, 90):.1f}")
```

**Interpretation**:
- If max quality is 500, then candidates with tree_predicted_quality > 400 are promising
- If mean quality is 100, then candidates > 150 are above average
- Use percentiles to set thresholds

---

## 8. Recommended Selection Criteria

### When GP Isn't Learning, Use This Priority:

1. **Decision Tree Prediction** (if R² > 0.8):
   - Sort by `Tree_Predicted_Quality`
   - Select top 5-10 candidates

2. **Parameter Filtering** (based on feature importance):
   - FA Ratio: 0.6-0.8 (best range from your data)
   - Spin Speed: 3000-4000 rpm
   - Temperature: 140-145°C
   - Antisolvent: Prefer "None"

3. **Physics Validation**:
   - Ensure `Rank_Score` > 0 (positive physics score)
   - Check `Film_Uniformity_Metric` > 2
   - Verify `Octahedral_Factor` in 0.4-0.9 range

4. **Diversity** (if selecting multiple):
   - Don't pick all candidates with identical parameters
   - Spread across FA ratio range (0.6-0.8)
   - Vary spin speed slightly (3200-3800 rpm)
   - Vary temperature slightly (141-144°C)

---

## 9. Example Selection Script

```python
import pandas as pd
import numpy as np
from src.learner import extract_fa_ratio_from_composition

# Load candidates
df = pd.read_csv('data/candidates_analysis.csv')

# Filter by best parameter ranges (from your data analysis)
df['FA_Ratio'] = df['Material_Composition'].apply(
    lambda x: extract_fa_ratio_from_composition(str(x))
)

filtered = df[
    (df['FA_Ratio'] >= 0.6) & (df['FA_Ratio'] <= 0.8) &
    (df['Spin_Speed_rpm'] >= 3000) & (df['Spin_Speed_rpm'] <= 4000) &
    (df['Anneal_Temp_C'] >= 140) & (df['Anneal_Temp_C'] <= 145)
]

# Sort by Decision Tree predictions (primary) or Rank Score (fallback)
if 'Tree_Predicted_Quality' in filtered.columns:
    selected = filtered.nlargest(10, 'Tree_Predicted_Quality')
else:
    selected = filtered.nlargest(10, 'Rank_Score')

# Display selected candidates
print("Selected Candidates:")
print(selected[['Experiment_ID', 'FA_Ratio', 'Spin_Speed_rpm', 
                'Anneal_Temp_C', 'Tree_Predicted_Quality', 'Rank_Score']])

# Save selection
selected.to_csv('data/selected_candidates.csv', index=False)
```

---

## 10. Summary: Decision Tree vs GP

| Aspect | Decision Tree | Composition GP |
|--------|---------------|----------------|
| **What it learns** | Composition + Process → Quality | Composition → Quality |
| **Training data** | Your experimental history | Binary dataset + history |
| **When useful** | Always (if trained) | Only if binary data shows composition effects |
| **Feature importance** | Shows which parameters matter | N/A (1D, composition only) |
| **Reliability** | High (R² = 0.937 in your case) | Low if not learning |
| **Recommendation** | **Primary decision tool** | Secondary (if learning) |

**Bottom Line**: When GP isn't learning, **rely on the Decision Tree**. It's learning from your actual experiments and shows clear feature importance (FA ratio matters most!).
