# Multi-Sport Prediction Platform: Model Strategy & Calibration Specification

**Focus**: Algorithmic Hierarchy, Dixon-Coles Probabilistic Modeling, Dynamic MOV Elo, Temperature/Vector Calibration, Ensembling, and Explainability  

---

## 1. Algorithmic Hierarchy & Progression

To prevent over-engineering and confirm that machine learning provides genuine predictive value, every market implements a strict multi-tier hierarchy:

```
Historical matches (14,000+ records)
       ↓
Anti-Leakage Feature Engineering
├── Margin-of-Victory (MOV) Dynamic Elo ratings (surface & home/away)
├── Exponential Weighted Moving Average (EWMA) Momentum (alpha = 0.85)
├── Non-Penalty xG Approximations & Field Tilt (Football)
├── Surface Dominance & Set Load Fatigue (Tennis)
├── Pace (Possessions/48m), ORtg/DRtg & Four Factors (Basketball)
       ↓
┌────────────────────────────────────────────────────────┐
│ Statistical Models Layer                               │
│ • Dixon-Coles Bivariate Poisson Model (Football)       │
│ • Margin-of-Victory Dynamic Elo Probabilities          │
│ • L2-Regularized Logistic Regression                   │
└───────────────────────────┬────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────┐
│ XGBoost Gradient Boosting Layer                        │
│ • Learns complex non-linear feature interactions       │
│ • Multi-class regularized log-loss objective           │
└───────────────────────────┬────────────────────────────┘
                            ↓
         Ensemble Blending (SLSQP Log-Loss Optimization)
                            ↓
Probability Calibration Layer (Isotonic / Temperature / Vector Scaling)
                            ↓
              Final Calibrated Prediction API
```

### 1.1 Baseline & Statistical Models
1. **Majority Class Baseline**:
   - Assigns empirical marginal prior probabilities: $P(y = c) = \frac{N_c}{N}$.
   - Serves as the zero-information reference point for Log Loss and Brier Score.
2. **Margin-of-Victory (MOV) Dynamic Elo Baseline**:
   - Continuous score margin multiplier prevents rating compression on blowout victories:
     $$\text{Multiplier}_{\text{MOV}} = ((|\Delta \text{Score}| + 3.0)^{0.8}) / (7.5 + 0.006 \cdot \Delta \text{Elo})$$
3. **Dixon-Coles Bivariate Goal Model (Football)**:
   - Overcomes independence shortcomings of basic Poisson models by applying low-scoring score-matrix adjustments for $(0,0), (1,0), (0,1), (1,1)$ outcomes:
     $$P_{\text{DC}}(i, j) = \tau(i, j; \lambda_H, \mu_A, \rho) \cdot \frac{\lambda_H^i e^{-\lambda_H}}{i!} \cdot \frac{\mu_A^j e^{-\mu_A}}{j!}$$

### 1.2 Machine Learning & Ensembling
1. **Regularized Logistic Regression (L2)**:
   - High interpretability, stable gradients, well-behaved baseline log-odds.
2. **XGBoost Classifier**:
   - Non-linear boosted decision trees with column subsampling ($0.85$), L1/L2 regularization ($\alpha=0.1, \lambda=1.0$), and maximum tree depth $\le 4$.
3. **SLSQP Blending**:
   - Optimal convex weights $\mathbf{w}^* = \arg\min_{\mathbf{w}} \mathcal{L}_{\text{log}}(\mathbf{w} \mid D_{\text{val}})$ subject to $\sum w_k = 1, w_k \ge 0$.

---

## 2. Mandatory Probability Calibration

Raw model outputs from gradient boosting are calibrated to match empirical observation frequencies.

### 2.1 Calibration Methods Evaluated Out-of-Sample
1. **Temperature Scaling**:
   - Multi-class log-odds scaling: $\tilde{\mathbf{p}} = \text{Softmax}(\mathbf{z} / T)$ where $T > 0$ optimizes validation cross-entropy.
2. **Vector / Diagonal Dirichlet Scaling**:
   - Parameterizes multi-class simplex mapping: $\tilde{\mathbf{p}} = \text{Softmax}(\mathbf{W} \ln \mathbf{p} + \mathbf{b})$.
3. **Isotonic Regression**:
   - Non-parametric piecewise constant step function with out-of-bounds clipping.
4. **Platt Scaling (Sigmoid)**:
   - Logistic transformation $P(y=1) = \frac{1}{1 + \exp(Az + B)}$.

### 2.2 Calibration Metrics
- **Brier Score**: $\text{BS} = \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C (P_{ic} - y_{ic})^2$
- **Logarithmic Loss**: $\text{LogLoss} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{ic} \ln(P_{ic})$
- **Expected Calibration Error (ECE)**: Binned confidence calibration gap across deciles.

