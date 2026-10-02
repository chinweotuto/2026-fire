"""Chronological Walk-Forward Backtesting Engine."""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Type
from sqlalchemy.orm import Session
from src.core.logging import logger
from src.db.models import Match, MatchFeatures
from src.markets.base import BasePredictionMarket
from src.models.base import BaseSportModel, MajorityBaselineModel, EloBaselineModel
from src.models.ml_models import (
    LogisticRegressionModel,
    RandomForestProbModel,
    GradientBoostingProbModel,
    XGBoostProbModel
)
from src.models.calibration import CalibratedModelWrapper
from src.models.ensemble import WeightedEnsembleModel, AverageEnsembleModel
from src.evaluation.metrics import evaluate_predictions


class ChronologicalBacktester:
    """Executes expanding-window out-of-sample evaluation on historical match stream."""

    def __init__(
        self,
        sport_id: str,
        market: BasePredictionMarket,
        db: Session
    ):
        self.sport_id = sport_id
        self.market = market
        self.db = db

    def load_dataset(self) -> tuple[pd.DataFrame, np.ndarray, List[str], List[datetime]]:
        """Load sorted chronological dataset with pre-computed features and target outcomes."""
        query = (
            self.db.query(Match, MatchFeatures)
            .join(MatchFeatures, Match.match_id == MatchFeatures.match_id)
            .filter(Match.sport_id == self.sport_id)
            .filter(Match.status == "COMPLETED")
            .order_by(Match.event_date.asc())
        )

        rows = query.all()
        if not rows:
            return pd.DataFrame(), np.array([]), [], []

        feature_records = []
        targets = []
        timestamps = []

        for match, feat in rows:
            target_val = self.market.extract_target(match)
            if target_val is None:
                continue

            f_json = feat.features_json
            feature_records.append(f_json)
            targets.append(target_val)
            timestamps.append(match.event_date)

        df_X = pd.DataFrame(feature_records).fillna(0.0)
        feature_names = list(df_X.columns)
        y = np.array(targets)
        return df_X, y, feature_names, timestamps

    def run_backtest(
        self,
        train_fraction: float = 0.70,
        val_fraction: float = 0.15
    ) -> Dict[str, Any]:
        """Split data chronologically: Train (70%), Validation (15%), Test (15%)."""
        df_X, y, feature_names, timestamps = self.load_dataset()
        n = len(df_X)

        if n < self.market.minimum_samples:
            return {
                "status": "INSUFFICIENT_DATA",
                "message": f"Sample size {n} is below minimum requirement {self.market.minimum_samples}",
                "sample_count": n
            }

        X = df_X.values
        n_train = int(n * train_fraction)
        n_val = int(n * (train_fraction + val_fraction))

        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:n_val], y[n_train:n_val]
        X_test, y_test = X[n_val:], y[n_val:]

        train_start = timestamps[0].strftime("%Y-%m-%d")
        train_end = timestamps[n_train - 1].strftime("%Y-%m-%d")
        test_start = timestamps[n_val].strftime("%Y-%m-%d")
        test_end = timestamps[-1].strftime("%Y-%m-%d")

        logger.info(
            f"Backtesting {self.sport_id} / {self.market.market_id}: "
            f"Train {len(X_train)} ({train_start} to {train_end}), "
            f"Val {len(X_val)}, Test {len(X_test)} ({test_start} to {test_end})"
        )

        # Build algorithmic hierarchy: Statistical models + XGBoost
        from src.models.statistical import PoissonGoalModel

        candidate_models: List[BaseSportModel] = [
            MajorityBaselineModel(),
            EloBaselineModel(),
            LogisticRegressionModel(C=1.0)
        ]

        if self.sport_id == "football":
            candidate_models.append(PoissonGoalModel(market_type=self.market.market_id))

        candidate_models.append(XGBoostProbModel(learning_rate=0.05, n_estimators=120, max_depth=4))
        candidate_models.append(RandomForestProbModel(n_estimators=100, max_depth=5))


        # 1. Fit models on training data
        fitted_models: List[BaseSportModel] = []
        for m in candidate_models:
            m.fit(X_train, y_train, feature_names=feature_names)
            fitted_models.append(m)

        # 2. Fit probability calibrators on out-of-sample validation data
        calibrated_models: List[BaseSportModel] = []
        for m in fitted_models:
            if m.name == "majority_baseline":
                calibrated_models.append(m)
                continue
            cal_m = CalibratedModelWrapper(m, method="isotonic")
            cal_m.fit_calibration(X_val, y_val)
            calibrated_models.append(cal_m)

        # 3. Build Ensembles
        ml_calibrated = [m for m in calibrated_models if "majority" not in m.name]
        avg_ens = AverageEnsembleModel(ml_calibrated)
        avg_ens.classes_ = fitted_models[0].classes_

        weighted_ens = WeightedEnsembleModel(ml_calibrated)
        weighted_ens.classes_ = fitted_models[0].classes_
        weighted_ens.fit_weights(X_val, y_val)

        from src.models.ensemble import TieredStatisticalXGBoostEnsemble
        tiered_ens = TieredStatisticalXGBoostEnsemble(self.sport_id, self.market.market_id)
        tiered_ens.fit(X_train, y_train, feature_names=feature_names)
        tiered_ens.fit_weights(X_val, y_val)
        cal_tiered_ens = CalibratedModelWrapper(tiered_ens, method="isotonic").fit_calibration(X_val, y_val)
        cal_tiered_ens.name = "statistical_xgboost_calibrated_ensemble"

        all_eval_models = calibrated_models + [avg_ens, weighted_ens, cal_tiered_ens]


        # 4. Out-of-sample Test Evaluation
        results = {}
        for m in all_eval_models:
            probs = m.predict_proba(X_test)
            metrics = evaluate_predictions(y_test, probs, m.classes_)
            results[m.name] = metrics

        # Model comparison summary table
        comparison_table = []
        for name, m_metrics in results.items():
            comparison_table.append({
                "model_name": name,
                "log_loss": m_metrics["log_loss"],
                "brier_score": m_metrics["brier_score"],
                "accuracy": m_metrics["accuracy"],
                "balanced_accuracy": m_metrics["balanced_accuracy"],
                "ece": m_metrics["expected_calibration_error"]
            })

        # Sort by Log Loss ascending (best first)
        comparison_table.sort(key=lambda x: x["log_loss"])

        return {
            "status": "SUCCESS",
            "sport": self.sport_id,
            "market": self.market.market_id,
            "training_period": f"{train_start} to {train_end}",
            "test_period": f"{test_start} to {test_end}",
            "sample_counts": {
                "total": n,
                "train": len(X_train),
                "val": len(X_val),
                "test": len(X_test)
            },
            "comparison": comparison_table,
            "detailed_metrics": results
        }
