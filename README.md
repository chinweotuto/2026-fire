# Multi-Sport Sports Prediction Platform

A production-grade statistical and machine learning sports forecasting system supporting **Football**, **Tennis**, and **Basketball**.

Built with an anti-leakage feature engineering engine, structural domain forecasters (Poisson bivariate models, dynamic surface-specific Elo ratings), non-linear gradient-boosted decision trees (XGBoost), convex SLSQP log-loss ensemble blending, and isotonic probability calibration.

---

## 🏗️ Architecture Overview

```
Historical Matches (14,097 verified matches)
       ↓
Feature Engineering Store (Anti-Leakage t_feature <= t_match)
├── Dynamic Elo ratings (surface & home/away adjustments)
├── Recent form & rolling momentum (5 & 10 matches)
├── Attack/defence & pace metrics
├── Rest days & schedule congestion
       ↓
┌────────────────────────────────────────────────────────┐
│ Statistical Models Layer                               │
│ • Poisson Goal Model (Bivariate P(i,j) for Football)   │
│ • Dynamic Elo Probabilities (All sports)               │
│ • L2-Regularized Logistic Regression                   │
└───────────────────────────┬────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────┐
│ XGBoost Gradient Boosting Layer                        │
│ • Learns complex non-linear feature interactions       │
└───────────────────────────┬────────────────────────────┘
                            ↓
         Ensemble Blending (SLSQP Log-Loss Optimization)
                            ↓
           Probability Calibration (Isotonic Regression)
                            ↓
              Final Calibrated Prediction API
```

---

## ⚡ Supported Sports & Markets

| Sport | Covered Markets | Key Methodologies |
| :--- | :--- | :--- |
| **Football** | `football_1x2`, `football_over_under_2_5`, `football_btts` | Bivariate Poisson Goal Model, Dynamic Home/Away Elo, Rolling xG/Form splits, XGBoost, Isotonic Calibration |
| **Tennis** | `tennis_match_winner`, `tennis_set_betting` | Surface-specific Dynamic Elo (Hard, Clay, Grass), Head-to-Head momentum, Set margin modeling |
| **Basketball** | `basketball_moneyline`, `basketball_total_points` | Dynamic Team Elo with home-court advantage, Rest day penalties, Pace/Possession metrics |

---

## 📊 Backtested Out-of-Sample Performance

Rigorous chronological backtesting on strictly out-of-sample data without lookahead bias:

| Sport & Market | Model | Accuracy | Log-Loss | Brier Score |
| :--- | :--- | :---: | :---: | :---: |
| **Football 1X2** | **Calibrated Weighted Ensemble** | **55.0%** | **0.9779** | **0.5583** |
| | Poisson Goal Model | 50.8% | 0.9987 | 0.5824 |
| | XGBoost Alone | 51.1% | 1.0185 | 0.5843 |
| | Majority Baseline | 47.9% | 1.0852 | 0.6409 |
| **Tennis Winner** | **Calibrated Weighted Ensemble** | **88.3%** | **0.2621** | **0.0827** |
| | XGBoost | 88.3% | 0.2704 | 0.0838 |
| | Dynamic Elo Baseline | 88.0% | 0.2798 | 0.0864 |
| **Basketball ML** | **Calibrated Weighted Ensemble** | **59.3%** | **0.6874** | **0.2474** |
| | Elo Baseline | 59.5% | 0.6866 | 0.2468 |
| | XGBoost | 57.5% | 0.6873 | 0.2477 |

---

## 🚀 Quickstart

### 1. Prerequisites & Environment Setup
Using `uv` (recommended):
```bash
# Clone the repository
git clone https://github.com/chinweotuto/2026-fire.git
cd 2026-fire

# Sync dependencies
uv sync
```

### 2. Run the Test Suite
```bash
uv run pytest -v
```

### 3. Launch the Prediction API Server
```bash
uv run uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```
API Documentation will be available interactively at `http://127.0.0.1:8000/docs`.

---

## 📡 API Prediction Endpoints

### Football Match Prediction (`POST /predict/football`)
```bash
curl -X POST "http://127.0.0.1:8000/predict/football" \
     -H "Content-Type: application/json" \
     -d '{
       "home_team": "Arsenal",
       "away_team": "Chelsea",
       "competition": "Premier League",
       "market_id": "football_1x2"
     }'
```

**Response:**
```json
{
  "prediction_id": "pred_fb_6cc5111db504",
  "sport": "football",
  "market": "football_1x2",
  "participant_a": "Arsenal",
  "participant_b": "Chelsea",
  "probabilities": {
    "HOME": 0.5228,
    "DRAW": 0.2153,
    "AWAY": 0.2619
  },
  "predicted_outcome": "HOME",
  "model": {
    "name": "weighted_ensemble",
    "version": "ver_football_1x2_v1.0.2530",
    "algorithm": "calibrated_ensemble"
  },
  "data_quality": "HIGH",
  "top_features": [
    {"feature_key": "home_elo", "value": 1741.88, "relative_importance": 0.481},
    {"feature_key": "away_elo", "value": 1726.09, "relative_importance": 0.477},
    {"feature_key": "elo_difference", "value": 80.79, "relative_importance": 0.022}
  ]
}
```

### Tennis Match Winner (`POST /predict/tennis`)
```bash
curl -X POST "http://127.0.0.1:8000/predict/tennis" \
     -H "Content-Type: application/json" \
     -d '{
       "player_1": "Carlos Alcaraz",
       "player_2": "Jannik Sinner",
       "surface": "Hard",
       "tournament": "ATP Masters",
       "market_id": "tennis_match_winner"
     }'
```

### Basketball Moneyline (`POST /predict/basketball`)
```bash
curl -X POST "http://127.0.0.1:8000/predict/basketball" \
     -H "Content-Type: application/json" \
     -d '{
       "home_team": "Boston Celtics",
       "away_team": "Golden State Warriors",
       "competition": "National Basketball Association",
       "market_id": "basketball_moneyline"
     }'
```

---

## 📁 Repository Structure

```
├── artifacts/              # Serialized production models (.joblib)
├── docs/                   # Architectural & data documentation
│   ├── model-strategy.md   # Mathematical breakdown of models & ensembles
│   ├── data-requirements.md# Schema, anti-leakage rules & ingestion protocol
├── scripts/                # Ingestion, training & backtesting scripts
│   ├── fetch_real_data.py  # Ingestion from open sports datasets
│   ├── build_features.py   # Chronological feature computation
│   ├── train_models.py     # Training, SLSQP blending & calibration
│   └── evaluate_models.py  # Out-of-sample evaluation benchmarks
├── src/                    # Core platform source code
│   ├── api/                # FastAPI application, routes & schemas
│   ├── database/           # SQLite connection & schema management
│   ├── evaluation/         # Backtesting engine & metrics (Log-loss, Brier)
│   ├── features/           # Feature engineering logic (Elo, Form, Rest)
│   ├── ingestion/          # Data cleaning & normalization
│   └── models/             # Poisson, Elo, XGBoost, and Calibrated Ensembles
├── tests/                  # Automated unit & end-to-end integration tests
└── sports_prediction.db    # Ingested historical database (14,097 matches)
```

---

## 🛡️ License
MIT License.
