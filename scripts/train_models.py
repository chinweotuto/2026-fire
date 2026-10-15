"""Automation script: Train, calibrate, and register production models."""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from src.core.config import settings
from src.core.logging import logger
from src.db.session import SessionLocal
from src.db.models import Match, MatchFeatures, ModelRecord, ModelVersion
from src.markets.registry import market_registry
from src.models.ml_models import RandomForestProbModel, GradientBoostingProbModel, LogisticRegressionModel, XGBoostProbModel
from src.models.statistical import PoissonGoalModel, EloProbModel
from src.models.calibration import CalibratedModelWrapper
from src.models.ensemble import WeightedEnsembleModel, TieredStatisticalXGBoostEnsemble
from src.evaluation.metrics import evaluate_predictions



def train_market_model(db: Session, sport_id: str, market_id: str):
    market = market_registry.get_market(market_id)
    if not market:
        logger.error(f"Market {market_id} not found in registry")
        return None

    logger.info(f"Training production model for {sport_id} / {market_id}...")

    # Load matches with features
    query = (
        db.query(Match, MatchFeatures)
        .join(MatchFeatures, Match.match_id == MatchFeatures.match_id)
        .filter(Match.sport_id == sport_id)
        .filter(Match.status == "COMPLETED")
        .order_by(Match.event_date.asc())
    )
    rows = query.all()
    if not rows:
        logger.warning(f"No records found for {sport_id}")
        return None

    feature_records, targets, timestamps = [], [], []
    for match, feat in rows:
        t_val = market.extract_target(match)
        if t_val is not None:
            feature_records.append(feat.features_json)
            targets.append(t_val)
            timestamps.append(match.event_date)

    n_samples = len(feature_records)
    if n_samples < market.minimum_samples:
        logger.warning(f"Insufficient samples for {market_id}: {n_samples} < {market.minimum_samples}")
        return None

    df_X = pd.DataFrame(feature_records).fillna(0.0)
    feature_names = list(df_X.columns)
    X = df_X.values
    y = np.array(targets)

    # 70% Train, 15% Validation, 15% Test
    n_train = int(n_samples * 0.70)
    n_val = int(n_samples * 0.85)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_val], y[n_train:n_val]
    X_test, y_test = X[n_val:], y[n_val:]

    # 1. Tiered Statistical + XGBoost Ensemble
    ensemble = TieredStatisticalXGBoostEnsemble(sport_id=sport_id, market_id=market_id)
    ensemble.fit(X_train, y_train, feature_names=feature_names)
    ensemble.fit_weights(X_val, y_val)

    # 2. Probability Calibration Layer: Compare candidate calibration methods on validation data
    candidate_methods = ["isotonic", "temperature", "vector", "sigmoid"]
    best_method = "isotonic"
    best_val_loss = float("inf")
    best_calibrated_model = None

    for method in candidate_methods:
        try:
            cand = CalibratedModelWrapper(ensemble, method=method).fit_calibration(X_val, y_val)
            val_probs = cand.predict_proba(X_val)
            cand_metrics = evaluate_predictions(y_val, val_probs, cand.classes_)
            if cand_metrics["log_loss"] < best_val_loss:
                best_val_loss = cand_metrics["log_loss"]
                best_method = method
                best_calibrated_model = cand
        except Exception:
            continue

    if best_calibrated_model is None:
        best_calibrated_model = CalibratedModelWrapper(ensemble, method="isotonic").fit_calibration(X_val, y_val)
        best_method = "isotonic"

    calibrated_model = best_calibrated_model

    # 3. Final out-of-sample evaluation on Test
    test_probs = calibrated_model.predict_proba(X_test)
    test_metrics = evaluate_predictions(y_test, test_probs, calibrated_model.classes_)

    logger.info(
        f"Tiered Statistical + XGBoost Ensemble trained for {market_id}: "
        f"Calibrator = {best_method}, "
        f"Test Log Loss = {test_metrics['log_loss']}, "
        f"Brier = {test_metrics['brier_score']}, "
        f"Accuracy = {test_metrics['accuracy'] * 100:.1f}%, "
        f"ECE = {test_metrics['expected_calibration_error']}"
    )

    # Persist artifact
    artifacts_dir = settings.MODELS_DIR / sport_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_file = artifacts_dir / f"{market_id}_ensemble.joblib"
    joblib.dump({
        "model": calibrated_model,
        "feature_names": feature_names,
        "classes": calibrated_model.classes_,
        "metrics": test_metrics,
        "component_weights": [round(float(w), 4) for w in ensemble.weights_]
    }, artifact_file)

    # Register in DB
    model_record = db.query(ModelRecord).filter(ModelRecord.model_id == f"model_{market_id}").first()
    if not model_record:
        model_record = ModelRecord(
            model_id=f"model_{market_id}",
            sport_id=sport_id,
            market_id=market_id,
            model_name=f"{market.display_name} Statistical+XGBoost Ensemble",
            algorithm="statistical_xgboost_calibrated_ensemble"
        )

        db.add(model_record)
        db.flush()

    version_tag = f"v1.1.{int(datetime.now().timestamp()) % 10000}"
    model_ver = ModelVersion(
        version_id=f"ver_{market_id}_{version_tag}",
        model_id=model_record.model_id,
        version_tag=version_tag,
        training_start_date=timestamps[0],
        training_end_date=timestamps[n_val - 1],
        training_samples=n_train + len(X_val),
        validation_log_loss=test_metrics["log_loss"],
        validation_brier_score=test_metrics["brier_score"],
        validation_accuracy=test_metrics["accuracy"],
        calibration_method=best_method,
        hyperparameters={"ensemble_weights": ensemble.weights_.tolist()},
        artifact_path=str(artifact_file),
        is_active=True,
        promoted_at=datetime.now(timezone.utc)
    )
    db.add(model_ver)
    db.commit()
    logger.info(f"Model version {version_tag} registered and promoted to active.")
    return model_ver


def main():
    db = SessionLocal()
    try:
        markets_to_train = [
            ("football", "football_1x2"),
            ("football", "football_over_under_2_5"),
            ("football", "football_btts"),
            ("tennis", "tennis_match_winner"),
            ("tennis", "tennis_set_betting"),
            ("basketball", "basketball_moneyline"),
            ("basketball", "basketball_total_points")
        ]

        trained_count = 0
        for sport, market in markets_to_train:
            ver = train_market_model(db, sport, market)
            if ver:
                trained_count += 1

        logger.info(f"Training run completed. Successfully trained and promoted {trained_count} market models.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
