# Multi-Sport Prediction Platform: Data Requirements & Sourcing Specification

**Platform Scope**: Football (Soccer), Basketball (NBA), Tennis (ATP/WTA)  
**System Objective**: Legitimate, leakage-free, out-of-sample probabilistic forecasting  
**Date**: October 2026  
**Status**: Approved Specification  

---

## 1. Executive Summary & Ground Truth Policy

A sports prediction system succeeds or fails based entirely on data fidelity, temporal integrity, and feature validity.
This platform enforces three non-negotiable principles:
1. **Zero Fabrication**: No synthetic match statistics or assumed outcomes are ever generated or treated as ground truth.
2. **Strict Temporal Integrity (No Lookahead/Leakage)**: Every feature for event at time $T$ is computed exclusively using data with timestamp $t < T$.
3. **Data Sufficiency Gating**: Prediction models are never trained or executed when sample size or feature completeness falls below predefined statistical thresholds.

---

## 2. Research & Comparison of Available Open Datasets

### 2.1 Football (Soccer) Datasets

| Dataset Source | Coverage | Granularity | Key Columns Available | Limitations / Missing Data | License / Terms |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Football-Data.co.uk** *(Recommended for Baseline & Primary MVP)* | Top 25 European leagues, 1993–present (30+ seasons) | Match level (final + halftime + in-game box statistics) | `Div`, `Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR`, `HTHG`, `HTAG`, `HTR`, `HS`, `AS`, `HST`, `AST`, `HC`, `AC`, `HF`, `AF`, `HY`, `AY`, `HR`, `AR`, Closing & Opening bookmaker odds (`B365H`, `B365D`, `B365A`, `Avg>2.5`, etc.) | **Missing**: Official event xG/xGA, player-level lineups/minutes, individual injury tracking. | Free for research, personal, and academic use. Raw CSV format without API keys. |
| **StatsBomb Open Data** | Selected historical tournaments (World Cups, select seasons of Big 5 leagues, Messi collection, FA WSL) | Ultra-detailed event level (coordinates, passes, pressures, 360 frames) | Full event stream, shot locations, freeze frames, official StatsBomb xG, line-ups, tactical formations | **Missing**: Continuous full-season coverage across all modern domestic leagues week-over-week. Extremely large JSON files. | Free for non-commercial use with StatsBomb attribution. |
| **Kaggle European Soccer (Hugo Mathien)** | 11 European leagues, 2008–2016 (25,000+ matches) | Match level + FIFA EA player ratings + betting odds | Match scores, lineup player IDs, FIFA player overall/potential attributes, betting odds | Outdated (stops at 2016); FIFA attributes are game ratings, not purely empirical on-pitch metrics. | Open database license (ODbL). |
| **FBref / Opta / Understat** | Modern major leagues (2017–present) | Match + team + player season/game level | xG, xGA, npxG, shot-creating actions, progressive passes, aerial duels | Scraping constraints and strict commercial terms; paid API (StatsPerform/Opta) for production automated live feeds. | Proprietary / Commercial restrictions. |

**Recommendation for Football**:
- **Primary Source**: `Football-Data.co.uk` (English Premier League `E0`, Championship `E1`, La Liga `SP1`, Bundesliga `D1`, Serie A `I1`, Ligue 1 `F1`).
- Provides 100% verified match results, half-time results, shots, shots on target, corners, fouls, and cards.
- Matches can be ingested directly into normalized tables.

---

### 2.2 Basketball (NBA) Datasets

| Dataset Source | Coverage | Granularity | Key Columns Available | Limitations / Missing Data | License / Terms |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NBA Official Historical Box Scores (Kaggle / NocturneBear 2010–2024)** *(Recommended for Primary MVP)* | NBA Regular Season & Playoffs (2010–2024, 30,000+ team games) | Game level & Team boxscore | `SEASON_YEAR`, `TEAM_ID`, `TEAM_NAME`, `GAME_ID`, `GAME_DATE`, `MATCHUP`, `WL`, `MIN`, `FGM`, `FGA`, `FG_PCT`, `FG3M`, `FG3A`, `FG3_PCT`, `FTM`, `FTA`, `FT_PCT`, `OREB`, `DREB`, `REB`, `AST`, `TOV`, `STL`, `BLK`, `PF`, `PTS`, `PLUS_MINUS` | Game-level team box scores require joining home and away game rows via `GAME_ID`. Player injury reports require external ingestion. | Open community mirror under MIT/CC-BY. |
| **nba_api (Python Package / stats.nba.com)** | Complete NBA history (1946–present) | Play-by-play, shot chart, player tracking, advanced box scores | Official NBA tracking metrics, Pace, ORtg, DRtg, Net Rating, eFG%, TS%, USG%, Rebound % | Live API endpoint has strict IP rate-limiting, unpredictable header changes, and frequent blocking if queried in bulk. | Non-commercial personal use. |
| **Basketball-Reference (Sports Reference)** | BAA/NBA/ABA from 1947 to present | Full box scores, Four Factors, advanced metrics, SRS, Pace, ORtg | Four factors, team game logs, player game logs, playoff series | Strict bot limits (max 20 requests per minute); automated scrapers are aggressively rate-limited and banned. | Terms prohibit aggressive scraping. |

**Recommendation for Basketball**:
- **Primary Source**: Open NBA Historical Box Score repository (2010–2024 team game logs), augmented with an automated pipeline for ongoing games.
- Provides all inputs needed to compute:
  - Estimated Possessions: $Possessions \approx FGA + 0.44 \times FTA - OREB + TOV$
  - True Shooting Percentage: $TS\% = \frac{PTS}{2 \times (FGA + 0.44 \times FTA)}$
  - Effective Field Goal Percentage: $eFG\% = \frac{FGM + 0.5 \times FG3M}{FGA}$
  - Offensive Rating ($ORtg = 100 \times \frac{PTS}{Poss}$), Defensive Rating ($DRtg$)
  - Pace, Rest Days, Back-to-Back scheduling flags, Rolling form.

---

### 2.3 Tennis (ATP & WTA) Datasets

| Dataset Source | Coverage | Granularity | Key Columns Available | Limitations / Missing Data | License / Terms |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Jeff Sackmann ATP/WTA Datasets (Kadantte/tennis_atp & tennis_wta open mirrors)** *(Recommended for Primary MVP)* | 1968–2024+ (Tour-level, Grand Slams, Masters 1000, 500, 250) | Match level with comprehensive in-match server/returner box scores | `tourney_id`, `tourney_name`, `surface`, `tourney_level`, `tourney_date`, `match_num`, `winner_id`, `winner_name`, `winner_hand`, `winner_ht`, `winner_age`, `winner_rank`, `winner_rank_points`, `loser_id`, `loser_name`, `loser_rank`, `score`, `best_of`, `round`, `minutes`, `w_ace`, `w_df`, `w_svpt`, `w_1stIn`, `w_1stWon`, `w_2ndWon`, `w_SvGms`, `w_bpSaved`, `w_bpFaced`, `l_ace`, `l_df`, `l_svpt`, `l_1stIn`, `l_1stWon`, `l_2ndWon`, `l_bpSaved`, `l_bpFaced` | Point-by-point shot charts are in a separate dataset (Match Charting Project); walkovers/retirements must be parsed from `score` strings. | Creative Commons CC-BY-NC-SA 4.0. |
| **Tennis-Data.co.uk** | 2001–present (ATP & WTA) | Match level + betting odds | Tournament, Date, Surface, Round, Winner, Loser, Set scores, B365W, B365L, Pinnacle odds | Lacks detailed service box scores (no break points or ace counts in early years). | Free non-commercial use. |
| **Tennis Abstract (Jeff Sackmann)** | Active professional tour | Advanced statistics, Elo ratings, surface splits, head-to-head | Surface Elo, Return %, Hold %, Dominance Ratio | Web interface; bulk consumption relies on the underlying GitHub repositories. | Non-commercial research. |

**Recommendation for Tennis**:
- **Primary Source**: `Kadantte/tennis_atp` and `tennis_wta` annual match archives (2015–2024 for training, validation, and backtesting).
- Provides complete match information, surface data, player physical profiles, ATP/WTA rankings, and in-depth serve/return statistics.

---

## 3. Data Schema & Column Mapping Reference

### 3.1 Normalized Match Schema (Core Fields)

Every sport translates into a normalized event entity:

```sql
matches (
    match_id            VARCHAR(64) PRIMARY KEY,
    sport               VARCHAR(16) NOT NULL, -- 'football', 'basketball', 'tennis'
    competition_id      VARCHAR(64) NOT NULL,
    season              VARCHAR(16) NOT NULL,
    event_date          TIMESTAMP WITH TIME ZONE NOT NULL,
    home_participant_id VARCHAR(64) NOT NULL, -- Team ID or Player 1 ID
    away_participant_id VARCHAR(64) NOT NULL, -- Team ID or Player 2 ID
    winner_id           VARCHAR(64),          -- NULL if draw or undecided
    home_score          INTEGER,
    away_score          INTEGER,
    status              VARCHAR(24) NOT NULL, -- 'SCHEDULED', 'COMPLETED', 'POSTPONED', 'CANCELLED'
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    source              VARCHAR(32) NOT NULL,
    source_id           VARCHAR(64) NOT NULL
);
```

### 3.2 Sport-Specific Match Statistics Schema

#### Football (`football_match_stats`)
- `match_id` (FK)
- `half_time_home_goals`, `half_time_away_goals`
- `home_shots`, `away_shots`
- `home_shots_on_target`, `away_shots_on_target`
- `home_corners`, `away_corners`
- `home_fouls`, `away_fouls`
- `home_yellow_cards`, `away_yellow_cards`
- `home_red_cards`, `away_red_cards`
- `home_xg`, `away_xg` (Nullable: recorded as NULL when absent, NEVER coerced to 0.0)

#### Basketball (`basketball_match_stats`)
- `match_id` (FK)
- `home_fgm`, `home_fga`, `home_fg3m`, `home_fg3a`, `home_ftm`, `home_fta`
- `away_fgm`, `away_fga`, `away_fg3m`, `away_fg3a`, `away_ftm`, `away_fta`
- `home_oreb`, `home_dreb`, `home_reb`, `away_oreb`, `away_dreb`, `away_reb`
- `home_ast`, `away_ast`
- `home_tov`, `away_tov`
- `home_stl`, `away_stl`
- `home_blk`, `away_blk`
- `home_pace`, `away_pace`
- `home_offensive_rating`, `away_offensive_rating`

#### Tennis (`tennis_match_stats`)
- `match_id` (FK)
- `surface` ('Hard', 'Clay', 'Grass', 'Carpet')
- `tourney_level` ('G' = Grand Slam, 'M' = Masters 1000, 'A' = ATP 250/500, 'F' = Finals)
- `round` ('R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F')
- `best_of` (3 or 5)
- `minutes`
- `p1_rank`, `p2_rank`
- `p1_aces`, `p1_double_faults`, `p1_serve_points`, `p1_first_in`, `p1_first_won`, `p1_second_won`
- `p2_aces`, `p2_double_faults`, `p2_serve_points`, `p2_first_in`, `p2_first_won`, `p2_second_won`
- `p1_break_points_saved`, `p1_break_points_faced`
- `p2_break_points_saved`, `p2_break_points_faced`

---

## 4. Data Quality Gating & Integrity Rules

1. **Zero Imputation of Absent Metrics**:
   - If `home_xg` is unavailable, the column is stored as `NULL`.
   - The feature engine generates `has_xg_metric = 0` and employs fallback statistical predictors (e.g. shot-based conversion proxy) rather than injecting `0.0`.
2. **Canonical Entity Resolution**:
   - Entities have a canonical mapping table (`entity_aliases`):
     - `"Man Utd"`, `"Manchester United"`, `"Manchester United FC"` $\rightarrow$ `team_manchester_united`.
     - `"Djokovic N."`, `"Novak Djokovic"`, `"N. Djokovic"` $\rightarrow$ `player_novak_djokovic`.
3. **Plausibility & Outlier Checks**:
   - Football: Score $\in [0, 20]$, Shots $\ge$ Shots on Target, Corners $\ge 0$.
   - Basketball: Points $\in [40, 220]$, FGM $\le$ FGA, FG3M $\le$ FG3A, FTM $\le$ FTA.
   - Tennis: Aces $\le$ Serve Points, 1st In $\le$ Serve Points, 1st Won $\le$ 1st In.
