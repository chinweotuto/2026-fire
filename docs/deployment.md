# Multi-Sport Prediction Platform: Deployment & Operations Specification

**Target Environment**: Production Linux / Windows Server / Containerized Cloud  
**Frameworks**: FastAPI, Uvicorn, SQLAlchemy 2.0 / Alembic, Pydantic v2, Scikit-Learn, LightGBM / XGBoost, React / TypeScript  

---

## 1. System Topology & Services

```
[ Client Browser / Frontend Dashboard ]
               │
               │ HTTPS / JSON REST
               ▼
[ Nginx Reverse Proxy / Load Balancer ]
               │
               ▼
[ FastAPI Application Server (Uvicorn Workers) ]
       │                         │
       │ SQLAlchemy 2.0          │ Joblib Model Loader
       ▼                         ▼
[ PostgreSQL 16+ Database ]   [ Local / Object Model Registry ]
  - Relational matches         - Trained model artifacts
  - Historical statistics      - Feature encoders
  - Auditable predictions      - Calibration mapping files
```

---

## 2. Environment Configuration & Security

### 2.1 Configuration File (`.env.example`)

```bash
# Database Configuration
DATABASE_URL=postgresql+psycopg2://postgres:password@localhost:5432/sports_prediction
# Fallback local testing DB:
# DATABASE_URL=sqlite:///./sports_prediction.db

# Application Configuration
APP_ENV=production
DEBUG=false
API_HOST=0.0.0.0
API_PORT=8000
API_SECRET_KEY=change_this_to_a_secure_random_64_char_string
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173

# Model & Artifact Storage
MODEL_ARTIFACTS_DIR=./artifacts/models
DATA_RAW_DIR=./data/raw
DATA_PROCESSED_DIR=./data/processed

# Security & Limits
RATE_LIMIT_PER_MINUTE=120
MAX_UPLOAD_SIZE_MB=50
```

### 2.2 Security Standards
1. **Zero Secret Leakage**: No API keys, database credentials, or tokens are checked into version control.
2. **Strict Input Schema Validation**: All incoming requests (including manual match entry) are parsed and sanitized through Pydantic models. Out-of-range statistics (e.g. negative corners, impossible scores) are rejected with HTTP 422 Unprocessable Entity.
3. **Audit Trail**: Every inference records `prediction_id`, `model_version`, `features_version`, and exact timestamp.

---

## 3. Automation Scripts Suite (`scripts/`)

The platform provides a unified CLI automation suite located in `scripts/`:

| Script Name | Purpose & Functionality |
| :--- | :--- |
| `scripts/download_data.py` | Connects to verified legitimate open sources (Kadantte Tennis, Football-Data.co.uk, NBA historical boxscores) and stores raw CSVs with SHA-256 integrity hashes. |
| `scripts/import_data.py` | Ingests raw CSVs or JSON files into the normalized database schema; resolves aliases; eliminates duplicates. |
| `scripts/validate_data.py` | Executes data quality assertions, checks for future-dated records, inspects outlier scores, and validates temporal consistency. |
| `scripts/build_features.py` | Computes strictly pre-match rolling form, Elo ratings, head-to-head records, and physical fatigue features. |
| `scripts/train_models.py` | Fits baseline, linear, tree, and ensemble models across chronological splits. |
| `scripts/evaluate_models.py` | Generates out-of-sample comparison reports (Log Loss, Brier, Accuracy, Calibration). |
| `scripts/backtest.py` | Runs expanding walk-forward evaluation and records step-by-step simulations. |
| `scripts/generate_predictions.py` | Ingests upcoming scheduled fixtures, extracts pre-match features, and outputs calibrated probabilities. |
| `scripts/promote_model.py` | Compares candidate model performance to current production champion; promotes or rejects. |

---

## 4. API Endpoints Specification

### 4.1 System & Meta Endpoints
- `GET /health` $\rightarrow$ `{ "status": "healthy", "database": "connected", "active_models": 3 }`
- `GET /sports` $\rightarrow$ Lists supported sports (`football`, `basketball`, `tennis`) and supported markets.
- `GET /markets/{sport}` $\rightarrow$ Returns active markets, target definitions, and minimum sample thresholds.

### 4.2 Prediction Endpoints
- `POST /predict/football`
- `POST /predict/basketball`
- `POST /predict/tennis`
- `GET /prediction/{prediction_id}` $\rightarrow$ Retrieves logged prediction with historical verification status.

### 4.3 Match & Data Ingestion Endpoints
- `POST /matches` $\rightarrow$ Manual entry of completed or scheduled match with dynamic sport-specific schema.
- `POST /matches/bulk` $\rightarrow$ Bulk CSV or JSON match upload.
- `GET /matches/upcoming` $\rightarrow$ Lists upcoming unpredicted fixtures.

### 4.4 Model Governance Endpoints
- `GET /models/{sport}` $\rightarrow$ Lists registered model versions.
- `GET /models/{sport}/performance` $\rightarrow$ Returns out-of-sample Log Loss, Brier Score, and calibration curves.
- `POST /training/start` $\rightarrow$ Triggers background model training run.
- `GET /training/{run_id}` $\rightarrow$ Checks progress of active training job.
