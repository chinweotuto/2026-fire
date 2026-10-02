# Multi-Sport Prediction Platform: Feature Engineering Specification

**Focus**: Mathematical Formulations, Time-Series Transformations, and Sport-Specific Feature Sets  
**Principle**: 100% Pre-Match Causality (No Lookahead Bias)  

---

## 1. Unified Feature Engineering Philosophy

Features represent the state of knowledge *immediately prior* to kickoff or match commencement.
For any participant $i$ entering match $m$ at time $T$:
- Only completed matches involving participant $i$ with $T_{prev} < T$ are ingested.
- Exponential decay weighting or fixed rolling windows ($N \in \{3, 5, 10\}$) capture recent form.
- Raw counting statistics are normalized on a per-game or per-minute basis.
- Relative differential features ($\Delta = \text{Feature}_{\text{Participant A}} - \text{Feature}_{\text{Participant B}}$) are computed for comparative modeling.

---

## 2. Sport 1: Football (Soccer) Feature Engine

### 2.1 Team Strength & Rating
- `home_elo`: Pre-match Elo rating of the home team.
- `away_elo`: Pre-match Elo rating of the away team.
- `elo_difference`: $\Delta_{elo} = \text{home\_elo} - \text{away\_elo} + \text{home\_advantage\_bonus}$ (standard home advantage bonus = +65 Elo points).
- `elo_expected_home_win`: $\frac{1}{1 + 10^{-(\Delta_{elo}) / 400}}$.

### 2.2 Form & Momentum (Rolling Windows $N \in \{3, 5, 10\}$)
- `home_points_last_N`, `away_points_last_N`: Points earned in the last $N$ matches (Win=3, Draw=1, Loss=0).
- `home_win_rate_last_N`, `away_win_rate_last_N`: Proportion of wins in last $N$.
- `home_goals_scored_last_N`, `away_goals_scored_last_N`: Sum of goals scored.
- `home_goals_conceded_last_N`, `away_goals_conceded_last_N`: Sum of goals allowed.
- `home_goal_diff_last_N`, `away_goal_diff_last_N`: Goal differential over last $N$.

### 2.3 Venue-Specific Form (Home Team at Home / Away Team on Road)
- `home_team_home_points_last_5`: Home team performance exclusively in previous home fixtures.
- `away_team_away_points_last_5`: Away team performance exclusively in previous away fixtures.
- `home_team_home_goal_diff_last_5`, `away_team_away_goal_diff_last_5`.

### 2.4 Offensive & Defensive Efficiency (Per-Game Rates over last 10)
- `home_goals_per_game`, `away_goals_per_game`.
- `home_goals_conceded_per_game`, `away_goals_conceded_per_game`.
- `home_shots_per_game`, `away_shots_per_game`.
- `home_shots_on_target_per_game`, `away_shots_on_target_per_game`.
- `home_shots_allowed_per_game`, `away_shots_allowed_per_game`.
- `home_shot_conversion_rate`: $\frac{\text{Goals}}{\text{Shots on Target}}$.
- `home_save_rate`: $1 - \frac{\text{Goals Conceded}}{\text{Shots on Target Faced}}$.
- `home_corners_per_game`, `away_corners_per_game`.
- `home_cards_per_game`, `away_cards_per_game`.

### 2.5 Match Context & Schedule Congestion
- `home_rest_days`: Days elapsed since previous competitive fixture:
  $$\text{rest\_days} = \min(14, \frac{T - T_{prev}}{86400})$$
- `away_rest_days`.
- `rest_advantage`: $\text{home\_rest\_days} - \text{away\_rest\_days}$.
- `home_matches_last_7_days`, `away_matches_last_7_days`.
- `home_matches_last_14_days`, `away_matches_last_14_days`.

### 2.6 Head-to-Head (H2H)
- `h2h_matches_played`: Number of meetings between the two teams in the preceding 3 years.
- `h2h_home_win_rate`, `h2h_draw_rate`, `h2h_away_win_rate`.
- *Cautionary Regularization*: If `h2h_matches_played` < 3, H2H features are smoothed toward the unconditioned prior to prevent overfitting on tiny samples.

---

## 3. Sport 2: Basketball (NBA) Feature Engine

### 3.1 Team Strength & Rating
- `home_elo`, `away_elo`, `elo_difference` (with standard NBA home court Elo adjustment $\approx +80$).

### 3.2 Four Factors & Advanced Offensive/Defensive Metrics (Rolling 10 Games)
1. **Shooting (eFG%)**:
   $$eFG\% = \frac{FGM + 0.5 \times FG3M}{FGA}$$
2. **Turnovers (TOV Rate)**:
   $$TOV\% = \frac{TOV}{FGA + 0.44 \times FTA + TOV}$$
3. **Rebounding (Rebound Rates)**:
   $$ORB\% = \frac{OREB}{OREB + \text{Opp\_DREB}}, \quad DRB\% = \frac{DREB}{DREB + \text{Opp\_OREB}}$$
4. **Free Throws (FT Rate)**:
   $$FT\_Rate = \frac{FTM}{FGA}$$

### 3.3 Pace & Possessions (Rolling 10 Games)
- **Estimated Possessions per Game**:
  $$Possessions \approx FGA + 0.44 \times FTA - OREB + TOV$$
- **Pace**: Possessions per 48 minutes ($48 \times \frac{Possessions}{Minutes}$).
- **Offensive Rating (ORtg)**: Points scored per 100 possessions ($100 \times \frac{PTS}{Possessions}$).
- **Defensive Rating (DRtg)**: Points conceded per 100 possessions.
- **Net Rating**: $ORtg - DRtg$.
- **Expected Match Pace**: $\frac{\text{Home Pace} \times \text{Away Pace}}{\text{League Average Pace}}$.

### 3.4 Schedule & Fatigue
- `home_rest_days`, `away_rest_days`.
- `home_is_back_to_back` (1 if game played yesterday, 0 otherwise).
- `away_is_back_to_back`.
- `home_games_in_last_7_days`, `away_games_in_last_7_days`.
- `net_rest_advantage` = `home_rest_days - away_rest_days`.

---

## 4. Sport 3: Tennis (ATP/WTA) Feature Engine

### 4.1 Player Strength & Rankings
- `p1_elo`, `p2_elo`, `elo_difference`.
- `p1_surface_elo`, `p2_surface_elo`, `surface_elo_difference`.
- `p1_rank`, `p2_rank` (Current ATP/WTA rankings).
- `log_rank_difference`: $\ln(\text{p2\_rank}) - \ln(\text{p1\_rank})$ (Rank 1 vs 10 is a wider gulf than Rank 101 vs 110).

### 4.2 Form & Surface Mastery (Rolling 5 & 10 Matches)
- `p1_win_rate_last_10`, `p2_win_rate_last_10`.
- `p1_surface_win_rate_career`, `p2_surface_win_rate_career` (Over all previous matches on that specific surface).
- `p1_surface_win_rate_last_10`, `p2_surface_win_rate_last_10`.
- `surface_win_rate_diff`.

### 4.3 Serve & Return Metrics (Rolling 10 Matches on Same Surface)
- `p1_first_serve_in_pct`: $\frac{\text{1stIn}}{\text{svpt}}$.
- `p1_first_serve_won_pct`: $\frac{\text{1stWon}}{\text{1stIn}}$.
- `p1_second_serve_won_pct`: $\frac{\text{2ndWon}}{\text{svpt} - \text{1stIn}}$.
- `p1_ace_rate`: $\frac{\text{Aces}}{\text{svpt}}$.
- `p1_df_rate`: $\frac{\text{DoubleFaults}}{\text{svpt}}$.
- `p1_break_points_saved_pct`: $\frac{\text{bpSaved}}{\text{bpFaced}}$.
- `p1_return_points_won_pct`: $\frac{\text{Opp\_svpt} - (\text{Opp\_1stWon} + \text{Opp\_2ndWon})}{\text{Opp\_svpt}}$.
- `p1_break_points_conversion_pct`: $\frac{\text{Opp\_bpFaced} - \text{Opp\_bpSaved}}{\text{Opp\_bpFaced}}$.

### 4.4 Match Physical Context
- `p1_days_since_last_match`, `p2_days_since_last_match`.
- `p1_minutes_played_in_tournament`: Cumulative physical court time in preceding rounds of the same tournament.
- `tournament_level_code`: Quantitative weight (Grand Slam=2000, Masters=1000, 500, 250).
- `is_best_of_five`: 1 for Men's Grand Slam matches, 0 for best-of-three.
- `h2h_win_rate`: Head-to-head win rate (regularized toward 0.5 when prior encounters < 3).

---

## 5. Missing Data Treatment & Safeguards

| Feature Category | Statistical Imputation / Handling Strategy | Fallback Quality Flag |
| :--- | :--- | :--- |
| **New Participant / Low History** (< 5 prior matches) | Prior mean imputation with Bayesian shrinkage toward league/tour median. | `data_quality = 'LOW'` |
| **Missing Surface Stats** | Blend overall career stats with universal surface baseline. | `data_quality = 'MEDIUM'` |
| **Missing Shots or Corners (Football)** | Nullable; features set to league average with a binary indicator `is_in_game_stat_missing = 1`. Never coerced to 0. | `data_quality = 'MEDIUM'` |
| **Complete Match History Available** | Full point-in-time calculation with valid sample size $\ge 10$ matches. | `data_quality = 'HIGH'` |
