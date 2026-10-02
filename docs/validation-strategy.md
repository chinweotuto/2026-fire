# Multi-Sport Prediction Platform: Validation & Backtesting Specification

**Focus**: Chronological Validation, Walk-Forward Evaluation, Metric Reporting, and Drift Monitoring  
**Core Constraint**: Zero Lookahead Bias; Never Randomly Shuffle Temporal Records  

---

## 1. Why Standard K-Fold Cross-Validation is Prohibited

In sports modeling, random $K$-fold cross-validation is fundamentally invalid because:
1. **Information Leakage**: Training on a match from May 2024 to predict a match from September 2023 exposes the model to post-event team strengths, tactical shifts, and future Elo shifts.
2. **Serial Correlation**: Player and team form exhibits auto-regressive properties over time.
3. **Over-Optimistic Metrics**: Models evaluated on randomly shuffled splits appear artificially accurate in backtests and fail catastrophically in live production.

---

## 2. Chronological Split Architecture

All model development follows a three-stage temporal partition:

```
[================= TRAINING SPLIT =================] [== VALIDATION ==] [==== TEST ====]
  2018-01-01  ----------------------->  2022-12-31    2023-01-01 ->    2024-01-01 ->
  (Feature engineering & base model training)         2023-12-31       2025-12-31
                                                      - Hyperparams    - Final un-
                                                      - Calibration      touched out-
                                                      - Ensemble         of-sample
                                                        weights          evaluation
```

### 2.1 The Three Chronological Partitions
1. **Training Set ($D_{train}$)**:
   - Contains all historical matches prior to the validation cutoff.
   - Used exclusively to learn model parameters (e.g. tree splits, logistic coefficients).
2. **Validation Set ($D_{val}$)**:
   - Contains subsequent chronological period (e.g., full succeeding season).
   - Used for:
     a) Selecting hyperparameters.
     b) Fitting probability calibrators (Platt / Isotonic).
     c) Optimizing ensemble weights $\mathbf{w}$.
3. **Test Set ($D_{test}$)**:
   - Pure, untouched forward-looking out-of-sample holdout.
   - Evaluated exactly once to report true expected production performance.

---

## 3. Rolling Walk-Forward Backtesting

To measure stability across changing competitive eras, the system implements **Expanding Window Walk-Forward Backtesting**:

```
Fold 1: Train [2018 - 2020]  ---> Test on [2021]
Fold 2: Train [2018 - 2021]  ---> Test on [2022]
Fold 3: Train [2018 - 2022]  ---> Test on [2023]
Fold 4: Train [2018 - 2023]  ---> Test on [2024]
Fold 5: Train [2018 - 2024]  ---> Test on [2025]
```

### 3.1 Point-in-Time Prediction Simulation Algorithm
For every test match $m$ on date $T_m$:
1. Freeze feature state at timestamp $T_m - \epsilon$.
2. Generate pre-match feature vector $x_m$.
3. Compute calibrated probabilities $\hat{p}_m$.
4. Log prediction $(\text{match\_id}, \hat{p}_m, T_m)$ to persistent database.
5. Reveal actual match outcome $y_m$.
6. Calculate instantaneous loss:
   $$\ell_m = -\sum_{c} y_{mc} \ln(\hat{p}_{mc})$$
7. Update historical Elo ratings and form caches with match $m$ results.
8. Advance to next chronological match.

---

## 4. Comprehensive Evaluation Metrics

For every model and market, the system automatically computes:

| Metric Category | Metrics Computed | Formula / Definition | Objective |
| :--- | :--- | :--- | :--- |
| **Probability Quality (Primary)** | Multi-Class Log Loss | $-\frac{1}{N} \sum_{i} \sum_{c} y_{ic} \ln(p_{ic})$ | Heavily penalizes overconfident wrong predictions. |
| **Probability Quality (Primary)** | Brier Score | $\frac{1}{N} \sum_{i} \sum_{c} (p_{ic} - y_{ic})^2$ | Mean squared error in probability space ($0.0 = \text{perfect}$). |
| **Calibration** | Expected Calibration Error (ECE) | $\sum_{m=1}^M \frac{\|B_m\|}{N} \|\text{acc}(B_m) - \text{conf}(B_m)\|$ | Measures alignment between predicted % and empirical hit rate. |
| **Classification** | Accuracy | $\frac{\text{Correct Predictions}}{N}$ | Overall hit rate. |
| **Classification** | Balanced Accuracy | $\frac{1}{C} \sum_{c} \text{Recall}_c$ | Prevents illusion of success on skewed classes (e.g. Football draws). |
| **Classification** | Macro Precision, Recall, F1 | Harmonic mean of class-level precision and recall. | Balances minority class performance. |
| **Categorical Diagnosis** | Confusion Matrix | $C \times C$ contingency table | Highlights systematic bias (e.g. under-predicting draws). |

---

## 5. Granular Performance Stratification

Global accuracy hides critical blind spots. The platform slices and displays performance by:

1. **Probability Confidence Deciles**:
   - Bin 1: $[0.50, 0.55)$
   - Bin 2: $[0.55, 0.60)$
   - Bin 3: $[0.60, 0.65)$
   - Bin 4: $[0.65, 0.70)$
   - Bin 5: $[0.70, 0.80)$
   - Bin 6: $[0.80, 1.00]$
   *A model is considered calibrated if matches predicted at 70% win approximately 70% of the time.*

2. **Competition Slice**:
   - English Premier League vs. La Liga vs. Bundesliga.
3. **Surface Slice (Tennis)**:
   - Hard Court vs. Clay Court vs. Grass Court.
4. **Home vs. Away Slice (Football & Basketball)**:
   - Isolates home bias and away upset detection.
5. **Seasonal Slice**:
   - Compares performance across 2021, 2022, 2023, 2024 to verify temporal stability.

---

## 6. Continuous Learning & Model Drift Detection

### 6.1 Champion vs. Candidate Promotion Gatekeeper
When new matches are imported:
1. Retraining a candidate model produces a new candidate version (e.g., `v1.2.0-candidate`).
2. The candidate is backtested on the rolling validation window against the active **Champion** (`v1.1.0-prod`).
3. **Promotion Rule**:
   $$\text{Promote Candidate } \iff \begin{cases} 
   \text{LogLoss}_{\text{candidate}} < \text{LogLoss}_{\text{champion}} - 0.005 \\ 
   \text{Brier}_{\text{candidate}} \le \text{Brier}_{\text{champion}} \\ 
   \text{ECE}_{\text{candidate}} \le 0.05 
   \end{cases}$$
4. If criteria are met, candidate is flagged `is_active = TRUE` and champion is archived.

### 6.2 Concept & Performance Drift Alerts
If a 30-day rolling window experiences:
- Log Loss increase $> +15\%$ above baseline validation
- ECE $> 0.08$
- The system automatically emits a status: `STATUS: MODEL DRIFT DETECTED - AUDIT REQUIRED`.
