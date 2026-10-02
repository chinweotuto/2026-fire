"""Ensemble System (Equal-Weighted & Validation-Learned Weights)."""

import numpy as np
from typing import List, Optional, Dict, Any
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from src.models.base import BaseSportModel


class AverageEnsembleModel(BaseSportModel):
    """Ensemble combining multiple models via equal-weighted probability averaging."""

    def __init__(self, models: List[BaseSportModel]):
        super().__init__(name="average_ensemble")
        self.models = models

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "AverageEnsembleModel":
        for m in self.models:
            m.fit(X, y, feature_names=feature_names)
        self.classes_ = self.models[0].classes_
        self.feature_names_ = feature_names or []
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        all_probs = [m.predict_proba(X) for m in self.models]
        avg = np.mean(all_probs, axis=0)
        # Normalize
        row_sums = avg.sum(axis=1, keepdims=True)
        return avg / np.where(row_sums == 0, 1.0, row_sums)


class WeightedEnsembleModel(BaseSportModel):
    """Ensemble with weights learned strictly on out-of-sample validation data."""

    def __init__(self, models: List[BaseSportModel]):
        super().__init__(name="weighted_ensemble")
        self.models = models
        self.weights_: np.ndarray = np.ones(len(models)) / len(models)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "WeightedEnsembleModel":
        for m in self.models:
            m.fit(X, y, feature_names=feature_names)
        self.classes_ = self.models[0].classes_
        self.feature_names_ = feature_names or []
        return self

    def fit_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> "WeightedEnsembleModel":
        """Learn ensemble weights strictly on validation data to minimize validation Log Loss."""
        val_probs_list = [m.predict_proba(X_val) for m in self.models]
        K = len(self.models)

        def objective(weights: np.ndarray) -> float:
            weights = weights / np.sum(weights)
            blended = np.zeros_like(val_probs_list[0])
            for i, p in enumerate(val_probs_list):
                blended += weights[i] * p
            # Clip for numerical stability
            blended = np.clip(blended, 1e-6, 1.0 - 1e-6)
            row_sums = blended.sum(axis=1, keepdims=True)
            blended = blended / row_sums
            return float(log_loss(y_val, blended, labels=self.classes_))

        # Initial equal weights
        init_weights = np.ones(K) / K
        bounds = [(0.0, 1.0) for _ in range(K)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        res = minimize(objective, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)
        if res.success:
            self.weights_ = res.x / np.sum(res.x)
        else:
            self.weights_ = init_weights

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        all_probs = [m.predict_proba(X) for m in self.models]
        blended = np.zeros_like(all_probs[0])
        for i, p in enumerate(all_probs):
            blended += self.weights_[i] * p
        row_sums = blended.sum(axis=1, keepdims=True)
        return blended / np.where(row_sums == 0, 1.0, row_sums)


class TieredStatisticalXGBoostEnsemble(BaseSportModel):
    """Tiered Statistical + XGBoost Ensemble.

    Architecture:
    Historical Matches
           ↓
    Feature Engineering (Elo, Form, Attack/Defense, Pace, Rest, H2H)
           ↓
    Statistical Models:
      - Poisson Goal Model (Football)
      - Dynamic Elo Model (All sports)
      - Logistic Regression (Linear baseline)
           ↓
    XGBoost:
      - Non-linear boosted trees learning feature interactions
           ↓
    Validation-Learned Ensemble Blending (SLSQP Log Loss Minimization)
           ↓
    Isotonic Probability Calibration
           ↓
    Final Calibrated Output
    """

    def __init__(self, sport_id: str, market_id: str):
        super().__init__(name=f"{sport_id}_{market_id}_tiered_ensemble")
        self.sport_id = sport_id.lower()
        self.market_id = market_id

        # 1. Statistical Models
        from src.models.statistical import PoissonGoalModel, EloProbModel
        from src.models.ml_models import LogisticRegressionModel, XGBoostProbModel

        self.statistical_models: List[BaseSportModel] = []
        if self.sport_id == "football":
            self.statistical_models.append(PoissonGoalModel(market_type=market_id))
        self.statistical_models.append(EloProbModel())
        self.statistical_models.append(LogisticRegressionModel(C=1.0))

        # 2. XGBoost Non-Linear Learner
        self.xgb_model = XGBoostProbModel(learning_rate=0.05, n_estimators=120, max_depth=4)

        # 3. All component models
        self.models: List[BaseSportModel] = self.statistical_models + [self.xgb_model]
        self.weights_: np.ndarray = np.ones(len(self.models)) / len(self.models)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "TieredStatisticalXGBoostEnsemble":
        self.feature_names_ = feature_names or []
        for m in self.models:
            m.fit(X, y, feature_names=feature_names)
        self.classes_ = self.models[0].classes_
        return self

    def fit_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> "TieredStatisticalXGBoostEnsemble":
        """Learn blending weights strictly on validation data to minimize Log Loss."""
        val_probs_list = [m.predict_proba(X_val) for m in self.models]
        K = len(self.models)

        def objective(weights: np.ndarray) -> float:
            weights = weights / np.sum(weights)
            blended = np.zeros_like(val_probs_list[0])
            for i, p in enumerate(val_probs_list):
                blended += weights[i] * p
            blended = np.clip(blended, 1e-6, 1.0 - 1e-6)
            row_sums = blended.sum(axis=1, keepdims=True)
            blended = blended / row_sums
            return float(log_loss(y_val, blended, labels=self.classes_))

        init_weights = np.ones(K) / K
        bounds = [(0.0, 1.0) for _ in range(K)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        res = minimize(objective, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)
        if res.success:
            self.weights_ = res.x / np.sum(res.x)
        else:
            self.weights_ = init_weights

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        all_probs = [m.predict_proba(X) for m in self.models]
        blended = np.zeros_like(all_probs[0])
        for i, p in enumerate(all_probs):
            blended += self.weights_[i] * p
        row_sums = blended.sum(axis=1, keepdims=True)
        return blended / np.where(row_sums == 0, 1.0, row_sums)


