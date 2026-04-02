# Detailed Model Explanation: Monte Carlo + Gaussian Process Bayesian Optimization

## Overview

This system combines **Monte Carlo Simulation** with **Gaussian Process Bayesian Optimization (GPBO)** to intelligently search for optimal perovskite film conditions. It's an **active learning** system that gets smarter with each experiment.

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION LOOP                        │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
   ┌────▼────┐                            ┌────▼────┐
   │  Cycle │                            │  Cycle │
   │   1    │                            │  2+    │
   └────┬────┘                            └────┬────┘
        │                                       │
        │ Random Monte Carlo                    │ GP-Guided Monte Carlo
        │                                       │
        └───────────────┬───────────────────────┘
                        │
                ┌───────▼────────┐
                │  Generate      │
                │  Candidates    │
                └───────┬────────┘
                        │
                ┌───────▼────────┐
                │  Score with    │
                │  GP Model      │
                │  (Delayed      │
                │   Reward)      │
                └───────┬────────┘
                        │
                ┌───────▼────────┐
                │  Select Best   │
                │  Candidates    │
                └───────┬────────┘
                        │
                ┌───────▼────────┐
                │  Save to      │
                │  Transition   │
                │  (Pending)    │
                └───────┬────────┘
                        │
                ┌───────▼────────┐
                │  Run           │
                │  Experiments   │
                │  (Delayed)     │
                └───────┬────────┘
                        │
                ┌───────▼────────┐
                │  Log Results   │
                │  to CSV       │
                └───────┬────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
   ┌────▼────┐                    ┌────▼────┐
   │  Retrain│                    │  LLM    │
   │  GP     │                    │  Agent │
   │  Model  │                    │  Cycle │
   └────┬────┘                    └────┬────┘
        │                               │
        │  ┌───────────────────────────┘
        │  │
        │  │  Analyze patterns
        │  │  Suggest restrictions
        │  │  Update config.yaml
        │  │
        └──┴──► Repeat until convergence
```

---

## 2. Monte Carlo Simulation Component

### Purpose
Generate random experimental candidates that satisfy physical and chemical constraints.

### How It Works

#### Step 1: Random Parameter Selection
The `MonteCarloGenerator` randomly selects:
- **Material Composition**: FA/FuDMA ratio (from config recipes)
- **Process Parameters**: 
  - Spin speed, time, acceleration (two-step enabled)
  - Annealing temperature (≥95°C), time (>5 min)
  - Concentration (0.3 M in your case)
  - Drop volume, dispense height
  - Antisolvent (optional, 70% probability)

#### Step 2: Constraint Checking
Each candidate is validated against:
1. **Structural Feasibility**: Tolerance factor, octahedral factor
2. **Solvent Compatibility**: Hansen solubility parameters
3. **Process Constraints**: Temperature ≥95°C, time >5 min, spin speed rules
4. **Antisolvent Selection**: Miscibility gap requirements
5. **Drying Time**: Must be in reasonable range (0.001-0.5)

#### Step 3: Physics-Based Metrics
For each valid candidate, calculate:
- Tolerance factor (Goldschmidt)
- Octahedral factor
- Precursor-solvent miscibility
- Estimated drying time
- Film thickness estimate
- Film uniformity metric

**Output**: List of ~1000-5000 valid experimental candidates

---

## 3. Gaussian Process (GP) Model

### What is a Gaussian Process?

A GP is a **probabilistic machine learning model** that:
- Learns a smooth function mapping **inputs** (experiment parameters) → **outputs** (quality score)
- Provides **uncertainty estimates** (not just predictions)
- Works well with **small datasets** (5-50 experiments)

### Feature Engineering

The GP uses a **3D feature space**:
1. **FA Ratio** (0-1): Fraction of FA in FA/FuDMA mixture
2. **Annealing Temperature** (°C): 130-150°C in your case
3. **Spin Speed** (rpm): 4000-5000 rpm (main step)

**Why only 3 features?**
- These are the most varied parameters
- Reduces dimensionality (curse of dimensionality)
- Captures main sources of variation

### Training Process

#### Step 1: Data Preparation
```python
# From experiments_log.csv:
- Extract features: (FA_ratio, temp, spin_speed)
- Calculate quality target from PL, FWHM, stability, peak wavelength
- Group replicates (same conditions) → use mean ± std
```

#### Step 2: Normalization
```python
# Normalize features to [0, 1] range
X_normalized = (X - X_min) / (X_max - X_min)
```

#### Step 3: GP Training
```python
# Using GPax library (JAX-based)
gp_model = gpax.ExactGP(input_dim=3, kernel='Matern')
gp_model.fit(rng_key, X_train, y_train, 
             num_warmup=2000, num_samples=2000)
```

**What happens during training:**
- GP learns the **covariance structure** (how similar experiments are)
- Uses **Matern kernel**: Assumes smooth, differentiable function
- **Bayesian inference**: Samples from posterior distribution
- Learns **hyperparameters**: Length scales, noise variance

### GP Predictions

For any new candidate, GP provides:

1. **Mean Prediction** (μ): Expected quality score
   ```
   μ = E[quality | features]
   ```

2. **Uncertainty** (σ): Standard deviation of prediction
   ```
   σ = std[quality | features]
   ```
   - **High uncertainty** = GP is unsure (exploration opportunity)
   - **Low uncertainty** = GP is confident (exploitation opportunity)

3. **Posterior Distribution**: Full probability distribution
   ```
   quality ~ Normal(μ, σ²)
   ```

**Key Insight**: GP uncertainty is **spatial** - it's higher in regions with few/no experiments, lower near observed data points.

---

## 4. Quality Target Calculation

The GP optimizes a **composite quality score**:

```python
Quality = w₁·PL + w₂·(FWHM_ideal/FWHM) + w₃·Stability + w₄·Peak_Penalty
```

Where:
- **PL Intensity** (w₁=1.0): Higher is better, normalized by scale (100)
- **FWHM** (w₂=50.0): Narrower peaks are better (ideal = 20 nm)
- **Stability** (w₃=2.0): Hours of stability (only if measured)
- **Peak Wavelength** (w₄=30.0): Penalty if outside 790±20 nm

**Example**:
- PL = 5000 → contribution = 1.0 × (5000/100) = 50
- FWHM = 25 nm → contribution = 50 × (20/25) = 40
- Stability = 72 hours → contribution = 2.0 × 72 = 144
- Peak = 800 nm (within tolerance) → bonus = 30 × 0.5 = 15
- **Total Quality = 249**

---

## 5. Bayesian Optimization (Acquisition Function)

### Upper Confidence Bound (UCB)

The acquisition function balances **exploration** vs **exploitation**:

```
UCB = μ + β·σ
```

Where:
- **μ** = Predicted quality (exploitation)
- **σ** = Uncertainty (exploration)
- **β** = Trade-off parameter (β=2.0 in your system)

**Interpretation**:
- **High μ, low σ** → High quality, confident → **Exploit** (refine known good region)
- **Low μ, high σ** → Unknown region → **Explore** (learn about new areas)

### How Candidates Are Ranked

1. **Rank Score** = Suitability Score + ML Boost
   ```python
   Rank_Score = Physics_Score + 20 × Acquisition_Score
   ```

2. **Physics Score**: Based on structural metrics, solvent compatibility, etc.

3. **ML Boost**: Multiplies acquisition score by 20 to give GP predictions significant weight

**Top candidates** = High rank score = Good physics + High GP acquisition value

---

## 6. GP-Guided Monte Carlo Sampling

### Cycle 1: Pure Random
- Generate 5000 random candidates
- No GP guidance (GP not trained yet)

### Cycle 2+: GP-Guided

#### Step 1: Initial Random Sampling (500 candidates)
```python
# Generate 500 random candidates
initial_candidates = generate_random(500)
```

#### Step 2: GP Scoring
```python
# Score with GP
learner.score_candidates_with_uncertainty(initial_candidates)
# Each candidate gets: predicted_performance, uncertainty_score, acquisition_score
```

#### Step 3: Calculate Sampling Weights
```python
# Normalize predictions and uncertainties to [0, 1]
pred_norm = (pred - pred.min()) / (pred.max() - pred.min())
uncert_norm = (uncert - uncert.min()) / (uncert.max() - uncert.min())

# Combined weight: 50% exploitation, 50% exploration
weights = 0.5 × pred_norm + 0.5 × uncert_norm
weights = weights / sum(weights)  # Normalize to probabilities
```

#### Step 4: Weighted Sampling
```python
# Sample 4250 candidates (85% of 5000) from promising regions
# Sample according to weights (higher weight = more likely to be sampled)
indices = np.random.choice(len(candidates), size=4250, p=weights)

# For each sampled index, generate similar candidate
for idx in indices:
    template = candidates[idx]
    # Generate new candidate (similar to template)
    new_candidate = generate_random()  # But biased toward template region
```

#### Step 5: Random Exploration (750 candidates)
```python
# 15% pure random exploration
explore_candidates = generate_random(750)
```

**Result**: 5000 candidates, but **85% biased toward promising regions** identified by GP

---

## 7. Multi-Objective Optimization (Pareto Front)

When `use_pareto_front: true`:

### Two Objectives
1. **Maximize Quality** (predicted_performance)
2. **Minimize Uncertainty** (uncertainty_score)

### Pareto-Optimal Solutions

A candidate is **Pareto-optimal** if:
- No other candidate has **both** higher quality **and** lower uncertainty
- It's on the "frontier" of the quality-uncertainty trade-off

**Example**:
```
Candidate A: Quality=100, Uncertainty=50  → Pareto-optimal
Candidate B: Quality=90,  Uncertainty=40  → Pareto-optimal (lower uncertainty!)
Candidate C: Quality=80,  Uncertainty=60  → NOT Pareto-optimal (dominated by A)
```

### Selection Strategy

Instead of just picking highest quality, select **diverse Pareto-optimal candidates**:
- Some high-quality (exploitation)
- Some low-uncertainty (exploration)
- Balanced trade-off

---

## 8. Iterative Optimization Loop

### Cycle Structure

```python
for cycle in range(max_cycles):  # Default: 10 cycles
    # 1. Generate candidates
    if cycle == 0:
        candidates = random_monte_carlo(5000)
    else:
        candidates = gp_guided_monte_carlo(5000, exploration_rate=0.15)
    
    # 2. Score with GP
    learner.score_candidates_with_uncertainty(candidates)
    
    # 3. Calculate rank scores
    for candidate in candidates:
        candidate.rank_score = calculate_suitability_score(candidate, use_ml=True)
    
    # 4. Select top candidates (batch_size = 8)
    top_candidates = sorted(candidates, key=lambda x: x.rank_score, reverse=True)[:8]
    
    # 5. Save to transition experiments
    save_transition_experiments(top_candidates)
    
    # 6. Track progress
    tracker.record_cycle(
        best_quality=max(rank_scores),
        avg_uncertainty_top10=mean(uncertainties[:10]),
        ...
    )
    
    # 7. Check convergence
    if tracker.check_convergence():
        break  # Stop if converged
    
    # 8. Retrain GP (after experiments are run and logged)
    # (Happens in next cycle when new data is available)
```

### Convergence Criteria

Optimization stops when **any** of these are met:

1. **Quality Threshold**: Best quality ≥ 1500
2. **Uncertainty Threshold**: Top 10 avg uncertainty ≤ 10
3. **Improvement Plateau**: <5% improvement over last 3 cycles
4. **Max Cycles**: Reached 10 cycles

### Adaptive Exploration Rate

```python
# Early cycles: More exploration (20-25%)
# Later cycles: More exploitation (10-15%)
exploration_rate = tracker.get_recommended_exploration_rate(base=0.15)
```

**Logic**: As GP learns more, reduce exploration, focus on refining good regions.

---

## 9. Delayed Reward GPBO (Transition Experiments)

### The Problem: Delayed Feedback

In real-world optimization:
- **GP generates candidates** → **User runs experiments** → **Results come back days/weeks later**
- GP needs to **avoid suggesting similar experiments** while waiting for results
- Need to track which experiments are **"in transition"** (generated but not yet completed)

### Solution: Transition Experiments

#### Step 1: Save Candidates to Transition
```python
# After GP scores candidates and selects top batch
selected_experiments = get_variations_of_top_candidates(candidates, batch_size=12)
save_transition_experiments(selected_experiments)  # Save to .pkl file
```

#### Step 2: GP Accounts for Transition Experiments
When scoring new candidates, GP **penalizes** candidates that are too similar to experiments currently in transition:

```python
def score_candidates_with_uncertainty(candidates, transition_experiments):
    # 1. Get GP predictions (mean, uncertainty)
    posterior_mean, posterior_samples = gp_model.predict(X_candidates)
    
    # 2. Calculate acquisition score (UCB)
    acquisition_score = UCB(posterior_mean, uncertainty, beta=2.0)
    
    # 3. Apply transition constraint (if transition experiments exist)
    if transition_experiments:
        # Penalize candidates too close to pending experiments
        transition_penalty = transition_constraint(
            X_candidates, 
            X_transition, 
            constraint_factor=5.0
        )
        # Reduce acquisition score for similar candidates
        acquisition_score = acquisition_score * transition_penalty
    
    # 4. Store scores
    candidate.metrics.acquisition_score = acquisition_score
    candidate.metrics.predicted_performance = posterior_mean
    candidate.metrics.uncertainty_score = uncertainty
```

#### Step 3: Transition Constraint Function
```python
def transition_constraint(X_candidates, X_transition, constraint_factor=5.0):
    """
    Returns penalty weights (0-1) based on distance to transition experiments.
    
    - Distance = 0 (identical) → penalty = 1.0 (strongly penalized)
    - Distance = large → penalty = 0.0 (no penalty)
    """
    for candidate in X_candidates:
        # Calculate minimum distance to any transition experiment
        distances = euclidean_distance(candidate, X_transition)
        min_distance = min(distances)
        
        # Gaussian penalty: exp(-α × d²)
        penalty = exp(-constraint_factor × 10 × min_distance²)
        constraint_weight = 1.0 - penalty  # Invert: 1 = no penalty, 0 = full penalty
    
    return constraint_weights
```

**Effect**: GP avoids suggesting experiments similar to ones already running, preventing wasted effort.

#### Step 4: Prune Completed Experiments
When experiments are logged to `experiments_log.csv`, they're automatically removed from transition:

```python
def prune_transition_experiments(history_df, transition_experiments):
    """
    Remove experiments from transition that have been completed.
    Matches by: (FA_ratio, temp, spin_speed, concentration)
    """
    completed = []
    for exp in transition_experiments:
        # Check if this experiment exists in history
        if experiment_in_history(exp, history_df):
            completed.append(exp)
    
    # Remove completed from transition
    remaining = [e for e in transition_experiments if e not in completed]
    return remaining
```

### Benefits

1. **Prevents Duplicate Experiments**: Won't suggest same experiment twice
2. **Efficient Resource Use**: Focuses on unexplored regions
3. **Handles Delays**: Works even when experiments take days/weeks
4. **Automatic Cleanup**: Removes completed experiments from transition

### Workflow Timeline

```
Day 1:  GP generates 12 candidates → Save to transition
Day 2:  User starts running experiments (still in transition)
Day 3:  GP generates new candidates → Avoids similar to Day 1 candidates
Day 5:  First experiment completes → Logged to CSV → Removed from transition
Day 7:  More experiments complete → GP retrains with new data
Day 10: All experiments complete → GP fully updated → Next cycle
```

---

## 10. LLM Agent (AI-Powered Pattern Recognition)

### Overview

The **LLM Agent** uses Google Gemini AI to:
1. **Learn patterns** from GP results and experiment history
2. **Analyze film images** for uniformity and quality
3. **Suggest parameter restrictions** to accelerate optimization
4. **Automatically update config.yaml** with high-confidence suggestions

### Integration in Workflow

The LLM agent runs **after** GP generates candidates and can be enabled with:
```bash
python main.py --with-agent --auto-apply --min-confidence 0.7
```

### Agent Capabilities

#### 1. Pattern Analysis

The agent analyzes:
- **Experiment history** (`experiments_log.csv`)
- **GP candidate predictions** (`candidates_analysis.csv`)
- **Optimization trends** (cycle-by-cycle improvement)

**What it learns:**
```python
PatternInsight(
    parameter="annealing_temperature",
    observation="Experiments at 140-145°C consistently show higher PL intensity",
    confidence=0.85,
    suggested_action="Restrict temperature range to [140, 145]°C",
    evidence="15/20 experiments in this range had PL > 4000"
)
```

#### 2. Film Image Analysis

Uses **Gemini Vision** to analyze film photos:

```python
FilmAnalysis(
    uniformity_score=75,  # 0-100
    coverage_estimate=90,  # Percentage
    color_consistency="Good",
    defects_detected=["Minor pinholes in corners"],
    recommendations=["Increase spin speed to improve edge coverage"]
)
```

**Features:**
- Detects pinholes, cracks, non-uniformity
- Estimates coverage percentage
- Assesses color consistency
- Provides actionable recommendations

#### 3. Config Optimization Suggestions

The agent suggests changes to `config.yaml`:

```python
ConfigSuggestion(
    section="annealing",
    parameter="temperature_c",
    current_value=[130, 135, 140, 145, 150],
    suggested_value=[140, 145],  # Narrowed based on patterns
    reasoning="GP predictions and experiment history show optimal range is 140-145°C",
    confidence=0.82
)
```

**Auto-Apply Logic:**
- If `confidence ≥ min_confidence` (default 0.7) and `auto_apply=True`
- Agent automatically updates `config.yaml`
- Creates backup before changes
- Logs all changes for review

### Agent Workflow

```
┌─────────────────────────────────────────┐
│         LLM Agent Cycle                 │
└─────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
   ┌────▼────┐            ┌─────▼─────┐
   │ Analyze │            │  Analyze  │
   │Patterns │            │   Film    │
   │         │            │  Images   │
   └────┬────┘            └─────┬─────┘
        │                       │
        └───────────┬───────────┘
                    │
            ┌───────▼────────┐
            │  Generate      │
            │  Suggestions   │
            └───────┬────────┘
                    │
            ┌───────▼────────┐
            │  Filter by     │
            │  Confidence    │
            └───────┬────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
   ┌────▼────┐            ┌─────▼─────┐
   │ Auto-   │            │  Manual  │
   │ Apply   │            │  Review  │
   │ (High   │            │  (Low    │
   │  Conf)  │            │   Conf)  │
   └────┬────┘            └─────┬─────┘
        │                       │
        └───────────┬───────────┘
                    │
            ┌───────▼────────┐
            │  Update        │
            │  config.yaml   │
            └─────────────────┘
```

### Example Agent Output

```
[Pattern 1] annealing_temperature
  Observation: Experiments at 140-145°C show 30% higher average PL intensity
  Confidence: 0.85
  Suggested Action: Restrict temperature to [140, 145]°C
  Evidence: 12/15 experiments in this range exceeded quality threshold

[Pattern 2] spin_speed
  Observation: Two-step spin coating with first step 1500 rpm improves uniformity
  Confidence: 0.78
  Suggested Action: Set first_step.speed_rpm to [1500, 2000]
  Evidence: Film images show better coverage with slower initial spin

[Film Analysis] experiment_5.jpg
  Uniformity Score: 82/100
  Coverage: 95%
  Defects: Minor edge thinning
  Recommendation: Increase drop volume to 60 µL for better edge coverage
```

### Benefits

1. **Accelerates Optimization**: Identifies patterns faster than manual analysis
2. **Reduces Search Space**: Suggests restrictions based on evidence
3. **Visual Quality Assessment**: Analyzes film images automatically
4. **Configurable Confidence**: Only applies high-confidence suggestions
5. **Audit Trail**: Logs all suggestions and changes

### Configuration

```yaml
llm_agent:
  enabled: true
  api_key: "your-gemini-api-key"
  min_confidence: 0.7  # Minimum confidence to auto-apply
  auto_apply: true     # Automatically apply high-confidence suggestions
  film_images_dir: "data/film_images"
```

---

## 11. Replicate Aggregation

### Problem
Same experimental conditions may be run multiple times → different quality scores

### Solution
Group replicates by unique (FA_ratio, temp, spin_speed):

```python
# Group replicates
groups = {}
for experiment in history:
    key = (round(fa_ratio, 4), temp, spin_speed)
    if key not in groups:
        groups[key] = {'qualities': []}
    groups[key]['qualities'].append(quality)

# Train GP on aggregated data
for key, group in groups.items():
    mean_quality = np.mean(group['qualities'])
    std_quality = np.std(group['qualities'])
    # Use mean for training, std for uncertainty
```

**Benefits**:
- GP learns from **mean quality** (more stable)
- Accounts for **experimental noise** (std)
- Prevents duplicate data points

---

## 12. Complete Workflow Example

### Initial State
- **0 experiments** in history
- **GP not trained**

### Cycle 1
1. Generate **5000 random candidates** (Monte Carlo)
2. Score with **physics metrics only** (no GP)
3. Select **top 8 candidates**
4. User runs experiments, logs results
5. **GP trains** on 8 experiments

### Cycle 2
1. Generate **5000 GP-guided candidates**:
   - 500 initial random → score with GP
   - 4250 weighted sampling (biased toward promising regions)
   - 750 pure random exploration
2. Score all with GP (predicted quality + uncertainty)
3. Select **top 8 candidates** (high rank score)
4. User runs experiments, logs results
5. **GP retrains** on 16 total experiments

### Cycle 3+
- Same as Cycle 2, but GP is **more confident**
- **Exploration rate decreases** (15% → 12% → 10%)
- Focus shifts to **refining good regions**

### Convergence
After 5 cycles:
- Best quality = 1800 (above threshold of 1500)
- Top 10 uncertainty = 8.5 (below threshold of 10)
- **→ Optimization converged!**

---

## 13. Key Advantages

### 1. Sample Efficiency
- **Traditional**: Try random combinations → need 100+ experiments
- **GPBO**: Learn patterns → find optimum in 20-40 experiments

### 2. Uncertainty Quantification
- Know **where** you're uncertain
- Balance exploration vs exploitation intelligently

### 3. Multi-Objective
- Optimize quality **and** uncertainty simultaneously
- Find Pareto-optimal solutions

### 4. Adaptive
- Exploration rate adapts to learning progress
- Focus shifts from exploration → exploitation

### 5. Physics-Informed
- Monte Carlo ensures **physically feasible** candidates
- GP learns from **actual experimental data**

---

## 14. Diagnostics: How to Know if It's Working

### GP Learning Diagnostics
- **R² Score**:
  - R² > 0.3: GP is learning well
  - R² < 0.1: GP not learning (need more/better data)
- **Predicted vs Actual**: Points should cluster on diagonal line

### Uncertainty Heatmap
- **Varied colors**: GP is learning spatial patterns ✓
- **Uniform color**: GP not learning (all regions equally uncertain) ✗

### Cycle Improvement
- **Best quality increasing**: Optimization working ✓
- **Uncertainty decreasing**: GP becoming more confident ✓
- **Plateau**: May have converged

---

## 15. Mathematical Foundation

### Gaussian Process

A GP defines a distribution over functions:

```
f(x) ~ GP(μ(x), k(x, x'))
```

Where:
- **μ(x)**: Mean function (often 0)
- **k(x, x')**: Covariance kernel (Matern in your case)

### Matern Kernel

```
k(x, x') = σ² × (1 + √3·d/ℓ) × exp(-√3·d/ℓ)
```

Where:
- **d** = ||x - x'|| (Euclidean distance)
- **ℓ** = Length scale (learned hyperparameter)
- **σ²** = Signal variance (learned hyperparameter)

**Interpretation**: Similar inputs (small d) → high covariance → similar outputs

### Posterior Distribution

After observing data D = {(x₁, y₁), ..., (xₙ, yₙ)}:

```
p(f* | x*, D) = Normal(μ*, σ*²)
```

Where:
- **μ*** = k(x*, X) × [K + σₙ²I]⁻¹ × y
- **σ*²** = k(x*, x*) - k(x*, X) × [K + σₙ²I]⁻¹ × k(X, x*)

**K** = Covariance matrix of training data
**σₙ²** = Noise variance (experimental error)

---

## 16. Limitations and Considerations

### 1. Small Data Regime
- GP works best with 10-50 experiments
- Very few experiments (<5) → unreliable predictions
- Many experiments (>100) → computational cost increases

### 2. Feature Space
- Only 3 features (FA ratio, temp, spin)
- Other parameters (concentration, antisolvent) not in GP
- Assumes these 3 are most important

### 3. Quality Target
- Composite score may not capture all objectives
- Weights are configurable but require domain knowledge

### 4. Convergence
- May converge to **local optimum** (not global)
- Depends on initial random sampling

### 5. Experimental Noise
- Real experiments have noise
- GP accounts for this via noise variance (σₙ²)

---

## 17. Summary

This system combines:
1. **Monte Carlo**: Generates physically feasible candidates
2. **Gaussian Process**: Learns quality function from data
3. **Bayesian Optimization**: Balances exploration/exploitation
4. **Delayed Reward GPBO**: Handles experiments in transition (pending results)
5. **LLM Agent**: AI-powered pattern recognition and config optimization
6. **Active Learning**: Gets smarter with each experiment
7. **Multi-Objective**: Optimizes quality and uncertainty

**Result**: Efficiently finds optimal perovskite film conditions with minimal experiments, automatically learns patterns, and adapts to delayed experimental feedback!
