# Multi-Sport Prediction Platform: Prediction Markets Specification

**Scope**: Formal definitions, mathematical translations, target encodings, and data sufficiency gates for all supported betting and statistical markets.  

---

## 1. Unified Market Engine Framework

A **Prediction Market** is defined by the following contract:

```python
class MarketDefinition:
    market_id: str                      # Unique slug (e.g. 'football_match_result_1x2')
    sport: str                          # 'football', 'basketball', 'tennis'
    market_name: str                    # Human-readable title
    target_type: str                    # 'MULTICLASS', 'BINARY', 'REGRESSION'
    outcomes: list[str]                 # Mutually exclusive outcomes (sum(probabilities) == 1.0)
    required_features: list[str]        # Essential feature keys
    minimum_training_samples: int       # Data sufficiency threshold
    enabled: bool                       # Active status in the engine
    version: str                        # Market algorithm version
```

---

## 2. Football (Soccer) Markets

### 2.1 Market 1: Match Result (1X2)
- **Target**: Multinomial outcome $Y \in \{\text{'HOME'}, \text{'DRAW'}, \text{'AWAY'}\}$.
- **Probabilities**: $P(\text{Home}), P(\text{Draw}), P(\text{Away})$ such that $\sum P = 1.0$.
- **Min Historical Matches**: 300 competitive league fixtures.
- **Model Type**: Calibrated Multinomial Classifier (Logistic Regression, Gradient Boosting, Softmax).

### 2.2 Market 2: Over / Under Goals (Thresholds: 0.5, 1.5, 2.5, 3.5, 4.5)
- **Target**: Binary outcome for each threshold $K \in \{0.5, 1.5, 2.5, 3.5, 4.5\}$:
  $$Y_{over\_K} = \mathbb{I}(\text{home\_goals} + \text{away\_goals} > K)$$
- **Mathematical Relationship**:
  Can be predicted either:
  1. Via direct binary classification per threshold (e.g. calibrated LightGBM/Ridge for Over 2.5).
  2. Via bivariate Poisson / Dixon-Coles expected goal intensities $(\lambda_{\text{home}}, \mu_{\text{away}})$:
     $$P(\text{Total Goals} = g) = \sum_{h+a=g} P(H=h, A=a \mid \lambda, \mu, \rho)$$
- **Min Historical Matches**: 500 matches.

### 2.3 Market 3: Both Teams to Score (BTTS)
- **Target**: Binary outcome $Y \in \{\text{'YES'}, \text{'NO'}\}$:
  $$Y_{\text{BTTS}} = \mathbb{I}(\text{home\_goals} \ge 1 \land \text{away\_goals} \ge 1)$$
- **Min Historical Matches**: 500 matches.

### 2.4 Market 4: Double Chance
- **Target**: 
  - $P(\text{Home/Draw}) = P(\text{Home}) + P(\text{Draw})$
  - $P(\text{Home/Away}) = P(\text{Home}) + P(\text{Away})$
  - $P(\text{Draw/Away}) = P(\text{Draw}) + P(\text{Away})$
- **Derivation**: Mathematically derived directly from calibrated 1X2 probabilities (zero modeling overhead).

### 2.5 Market 5: Correct Score Matrix
- **Target**: Grid of exact scores $(h, a)$ for $h, a \in \{0, 1, 2, 3, 4, 5+\}$.
- **Formulation**: Dixon-Coles bivariate Poisson with low-score correlation adjustment parameter $\tau(x, y)$:
  $$\tau(x, y) = \begin{cases} 
  1 - \lambda \mu \rho & x=0, y=0 \\ 
  1 + \mu \rho & x=0, y=1 \\ 
  1 + \lambda \rho & x=1, y=0 \\ 
  1 - \rho & x=1, y=1 \\ 
  1 & \text{otherwise} 
  \end{cases}$$
- **Min Historical Matches**: 1,000 matches.

### 2.6 Football Markets Requiring Additional Data (Deferred / Gated)
- **Corners Over/Under & Cards Over/Under**: Only enabled when match reports contain `HC, AC, HY, AY, HR, AR`. Supported in Football-Data.co.uk E0/SP1 datasets.
- **Player Anytime Goalscorer**: Disabled in baseline due to absence of individual player tracking in open standard CSVs.

---

## 3. Basketball (NBA) Markets

### 3.1 Market 1: Match Winner (Moneyline)
- **Target**: Binary outcome $Y \in \{\text{'HOME'}, \text{'AWAY'}\}$.
- **Probabilities**: $P(\text{Home}) + P(\text{Away}) = 1.0$.
- **Min Historical Matches**: 300 games.

### 3.2 Market 2: Total Game Points (Over / Under)
- **Target**: Binary outcome for dynamic or fixed lines $L$ (e.g. 215.5, 220.5, 225.5):
  $$Y = \mathbb{I}(\text{home\_pts} + \text{away\_pts} > L)$$
- **Formulation**:
  Gaussian distribution around predicted match total $\hat{T} = \hat{Y}_{\text{home\_pts}} + \hat{Y}_{\text{away\_pts}}$ with residual standard error $\sigma \approx 11.5$ points:
  $$P(\text{Total} > L) = 1 - \Phi\left(\frac{L - \hat{T}}{\sigma}\right)$$
- **Min Historical Matches**: 500 games.

### 3.3 Market 3: Point Spread / Margin of Victory
- **Target**: Expected point differential $\Delta_{\text{pts}} = \text{home\_pts} - \text{away\_pts}$.
- **Probabilities for Spread Line $S$**:
  $$P(\text{Home Covers } S) = 1 - \Phi\left(\frac{S - \hat{\Delta}}{\sigma_{\Delta}}\right)$$
- **Min Historical Matches**: 500 games.

---

## 4. Tennis (ATP/WTA) Markets

### 4.1 Market 1: Match Winner
- **Target**: Binary outcome $Y \in \{\text{'PLAYER\_1'}, \text{'PLAYER\_2'}\}$.
- **Probabilities**: $P(\text{P1}) + P(\text{P2}) = 1.0$.
- **Min Historical Matches**: 300 tour-level matches.

### 4.2 Market 2: Set Betting (Exact Score in Sets)
- **Best-of-Three Outcomes**: `2-0`, `2-1`, `0-2`, `1-2`.
- **Best-of-Five Outcomes**: `3-0`, `3-1`, `3-2`, `0-3`, `1-3`, `2-3`.
- **Formulation**:
  Based on the probability of Player 1 winning an individual game on serve ($p_A$) and Player 2 winning on serve ($p_B$), set probabilities are derived analytically using Markov chains (Barnett & Clarke / O'Malley models) or trained via calibrated multinomial classifier.
- **Min Historical Matches**: 500 matches with set-by-set recorded scores.

### 4.3 Market 3: Total Games (Over / Under Line $G$, e.g. 21.5, 22.5)
- **Target**: Total games played across all sets $\sum G_{\text{set}}$.
- **Min Historical Matches**: 500 matches.

---

## 5. Market Enablement Matrix

| Sport | Market Identifier | Target Type | Supported in MVP? | Required Data Source |
| :--- | :--- | :--- | :--- | :--- |
| **Football** | `football_1x2` | Multiclass | **YES (Primary)** | Football-Data.co.uk / Sackmann / Manual |
| **Football** | `football_over_under_2_5` | Binary | **YES (Primary)** | Football-Data.co.uk / Manual |
| **Football** | `football_btts` | Binary | **YES (Primary)** | Football-Data.co.uk / Manual |
| **Football** | `football_double_chance` | Derived | **YES** | Derived from 1X2 |
| **Football** | `football_corners` | Binary / Count | **YES (Secondary)** | Football-Data.co.uk |
| **Football** | `football_cards` | Binary / Count | **YES (Secondary)** | Football-Data.co.uk |
| **Basketball**| `basketball_moneyline` | Binary | **YES (Primary)** | NBA Boxscores (NocturneBear / Kaggle) |
| **Basketball**| `basketball_total_points`| Binary / Reg | **YES (Primary)** | NBA Boxscores |
| **Basketball**| `basketball_spread` | Binary / Reg | **YES (Primary)** | NBA Boxscores |
| **Tennis** | `tennis_match_winner` | Binary | **YES (Primary)** | Kadantte/tennis_atp & WTA |
| **Tennis** | `tennis_set_betting` | Multiclass | **YES (Primary)** | Kadantte/tennis_atp & WTA |
| **Tennis** | `tennis_total_games` | Binary / Reg | **YES (Secondary)** | Kadantte/tennis_atp & WTA |
| **Tennis** | `tennis_tiebreak_occurs`| Binary | **Deferred** | Requires detailed game log breakdown |
