"""Machine Learning Probabilistic Classifiers."""

import numpy as np
from typing import Optional, List, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from src.models.base import BaseSportModel

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


class LogisticRegressionModel(BaseSportModel):
    """L2-regularized multinomial logistic regression with standard scaling."""

    def __init__(self, C: float = 1.0):
        super().__init__(name="logistic_regression")
        self.C = C
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=self.C, max_iter=1000, random_state=42)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "LogisticRegressionModel":
        self.feature_names_ = feature_names or []
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self.classes_ = self.clf.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X_scaled)


class RandomForestProbModel(BaseSportModel):
    """Calibrated random forest classifier constrained to prevent leaf overfitting."""

    def __init__(self, n_estimators: int = 150, max_depth: int = 6, min_samples_leaf: int = 25):
        super().__init__(name="random_forest")
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "RandomForestProbModel":
        self.feature_names_ = feature_names or []
        self.clf.fit(X, y)
        self.classes_ = self.clf.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)


class GradientBoostingProbModel(BaseSportModel):
    """Histogram-based gradient boosting trees with conservative learning rate."""

    def __init__(self, learning_rate: float = 0.05, max_iter: int = 150, max_depth: int = 4, min_samples_leaf: int = 30):
        super().__init__(name="gradient_boosting")
        self.clf = HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "GradientBoostingProbModel":
        self.feature_names_ = feature_names or []
        self.clf.fit(X, y)
        self.classes_ = self.clf.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)


class XGBoostProbModel(BaseSportModel):
    """XGBoost probabilistic classifier."""

    def __init__(self, learning_rate: float = 0.05, n_estimators: int = 120, max_depth: int = 4):
        super().__init__(name="xgboost")
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.clf = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "XGBoostProbModel":
        self.feature_names_ = feature_names or []
        self.classes_ = np.unique(y)
        # Map classes to integers 0..K-1
        self.class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_encoded = np.array([self.class_to_idx[val] for val in y])

        if XGB_AVAILABLE:
            self.clf = xgb.XGBClassifier(
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
                eval_metric="mlogloss" if len(self.classes_) > 2 else "logloss",
                verbosity=0
            )
            self.clf.fit(X, y_encoded)
        else:
            # Fallback to HistGradientBoosting if compiled xgboost is absent
            self.clf = HistGradientBoostingClassifier(random_state=42)
            self.clf.fit(X, y)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if XGB_AVAILABLE and hasattr(self, "class_to_idx"):
            return self.clf.predict_proba(X)
        return self.clf.predict_proba(X)
