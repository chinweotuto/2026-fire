# Multi-Sport Prediction Platform: Model Strategy & Calibration Specification

**Focus**: Algorithmic Hierarchy, Probabilistic Modeling, Calibration, Ensembling, and Explainability  

---

## 1. Algorithmic Hierarchy & Progression

To prevent over-engineering and confirm that machine learning provides genuine predictive value, every market implements a strict multi-tier hierarchy:

```
Historical matches
       ↓
Feature engineering
       ↓
Elo ratings
Recent form
Home/away or surface form
Attack/defence
Player/team statistics
Rest/injuries
       ↓
┌───────────────────────────┐
│ Statistical models        │
│                           │
│ Poisson (Football)        │
│ Elo (All sports)          │
│ Logistic Regression       │
└─────────────┬─────────────┘
              ↓
┌───────────────────────────┐
│ XGBoost                   │
│                           │
│ Learns nonlinear patterns │
└─────────────┬─────────────┘
              ↓
       Ensemble
              ↓
     Probability calibration
              ↓
        Final output
```


### 1.1 Baseline Models
1. **Majority Class Baseline**:
   - Assigns empirical marginal prior probabilities: $P(y = c) = \frac{N_c}{N}$.
   - Serves as the zero-information reference point for Log Loss and Brier Score.
2. **Elo Baseline**:
   - Computes logistic probability directly from the rating difference:
     $$P(\text{Home}) = \frac{1}{1 + 10^{-(\text{Elo}_H - \text{Elo}_A + \text{Adv}) / 400}}$$
   - In 3-way markets (Football), draws are assigned based on rating proximity:
     $$P(\text{Draw}) = \gamma \cdot \exp\left(-\frac{(\text{Elo}_H - \text{Elo}_A)^2}{2\sigma_d^2}\right)$$
3. **Simple Logistic Model**:
   - Single-feature or two-feature linear baseline (e.g. Elo diff + Rest diff).

### 1.2 Machine Learning Models
1. **Regularized Logistic Regression (ElasticNet / L2)**:
   - High interpretability, stable gradients, well-behaved log-odds.
   - Provides a linear benchmark for feature weights.
2. **Random Forest Classifier**:
   - Captures non-linear feature interactions and threshold effects without extreme sensitivity to scaling.
3. **Gradient Boosted Trees (HistGradientBoosting / XGBoost / LightGBM)**:
   - State-of-the-art tabular learning.
   - Tuned with conservative learning rates ($\eta \in [0.02, 0.08]$), maximum depth $\le 5$, and minimum samples per leaf $\ge 20$ to prevent overfitting on noisy sports outcomes.

---

## 2. Mandatory Probability Calibration

Raw model outputs from tree-based algorithms (Random Forest, Gradient Boosting) are notoriously uncalibrated:
- Random Forest tends to push probabilities toward the center ($0.2 < P < 0.8$) due to averaging across trees.
- Gradient Boosting tends to push probabilities toward extreme edges ($0$ or $1$) due to loss minimization.

### 2.1 Calibration Methods
For every model, calibration is trained strictly on a chronologically partitioned **Validation Set** ($D_{val}$), completely isolated from the **Training Set** ($D_{train}$) and **Test Set** ($D_{test}$):

1. **Platt Scaling (Sigmoid)**:
   $$P_{\text{calibrated}}(y=1 \mid \hat{f}) = \frac{1}{1 + \exp(A \hat{f} + B)}$$
   - Parameterized by scalar coefficients $(A, B)$ optimized via maximum likelihood on $D_{val}$.
   - Best suited for smaller datasets ($N < 1,000$) or when distortion is monotonic and sigmoidal.
2. **Isotonic Regression**:
   - Non-parametric, piecewise constant non-decreasing step function.
   - Free from functional form assumptions; highly effective for large datasets ($N \ge 1,000$).
3. **Multi-Class Normalization**:
   - In 3-way markets (Football 1X2), binary calibrations are performed One-vs-Rest, followed by simplex projection:
     $$\tilde{P}_c = \frac{P_{\text{calibrated}}(c)}{\sum_{k \in \mathcal{C}} P_{\text{calibrated}}(k)}$$
   - Guarantees $\sum_{c} \tilde{P}_c = 1.0$ and $\tilde{P}_c \ge 0$.

### 2.2 Calibration Metrics
- **Brier Score**:
  $$\text{BS} = \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C (P_{ic} - y_{ic})^2$$
- **Logarithmic Loss (Cross-Entropy)**:
  $$\text{LogLoss} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{ic} \ln(P_{ic})$$
- **Expected Calibration Error (ECE)**:
  Binned grouping into $M = 10$ probability intervals:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$

---

## 3. Ensemble Architecture

An ensemble is evaluated against individual models:

$$\hat{P}_{\text{ensemble}}(c) = \sum_{k=1}^K w_k \hat{P}_k(c), \quad \text{subject to } \sum_{k=1}^K w_k = 1, \quad w_k \ge 0$$

1. **Equal Weighted Ensemble**:
   $w_k = \frac{1}{K}$.
2. **Optimization-Learned Weights**:
   Weights $\mathbf{w}$ are learned by minimizing Log Loss or Brier Score exclusively on the out-of-sample **Validation Set** using sequential least-squares programming (SLSQP):
   $$\min_{\mathbf{w}} \mathcal{L}(\mathbf{w} \mid D_{val}), \quad \text{s.t. } \mathbf{w} \in \Delta^{K-1}$$
3. **Anti-Leakage Prohibition**:
   **Under no circumstances may ensemble weights be tuned or selected using the final Test Set ($D_{test}$).**

---

## 4. Model Explainability & Feature Importance

Every prediction returned by the API is accompanied by top local contributing factors:

1. **Tree-Based Feature Importances / Permutation Importance**:
   Measures mean decrease in out-of-sample Log Loss when feature $j$ is randomly shuffled.
2. **SHAP (SHapley Additive exPlanations)**:
   Computes exact marginal contributions for tree and linear models:
   $$\hat{f}(x) = \phi_0 + \sum_{j=1}^M \phi_j(x)$$
3. **Causality Disclaimer**:
   All UI outputs and API responses explicitly format feature attributions as statistical associations (e.g. *"Positive factor: Home team 5-match goal differential (+4)"*) and never claim mechanistic causation.
