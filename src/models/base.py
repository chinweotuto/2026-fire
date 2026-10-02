"""Base model interface and probabilistic baseline models."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from sklearn.metrics import log_loss, brier_score_loss


class BaseSportModel(ABC):
    """Abstract interface for all probabilistic prediction models."""

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.classes_: np.ndarray = np.array([])
        self.feature_names_: List[str] = []

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "BaseSportModel":
        """Fit model parameters on training dataset."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities summing to 1.0."""
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict most probable class."""
        probs = self.predict_proba(X)
        max_idx = np.argmax(probs, axis=1)
        return self.classes_[max_idx]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Compute Log Loss, Brier Score, and Accuracy on evaluation data."""
        probs = self.predict_proba(X)
        y = np.asarray(y)

        # Multi-class or binary log loss
        ll = float(log_loss(y, probs, labels=self.classes_))
        
        # Brier score (multi-class Brier: mean squared error across all one-hot classes)
        # Convert y to one-hot
        y_one_hot = np.zeros_like(probs)
        for i, val in enumerate(y):
            idx = np.where(self.classes_ == val)[0]
            if len(idx) > 0:
                y_one_hot[i, idx[0]] = 1.0

        bs = float(np.mean(np.sum((probs - y_one_hot) ** 2, axis=1)))
        preds = self.predict(X)
        acc = float(np.mean(preds == y))

        return {
            "log_loss": round(ll, 4),
            "brier_score": round(bs, 4),
            "accuracy": round(acc, 4)
        }


class MajorityBaselineModel(BaseSportModel):
    """Assigns empirical training set class prior frequencies."""

    def __init__(self):
        super().__init__(name="majority_baseline")
        self.priors_: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "MajorityBaselineModel":
        self.classes_, counts = np.unique(y, return_counts=True)
        self.priors_ = counts / len(y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = len(X)
        return np.tile(self.priors_, (n_samples, 1))


class EloBaselineModel(BaseSportModel):
    """Probabilistic prediction directly from pre-computed Elo difference."""

    def __init__(self, elo_diff_feature: str = "elo_difference"):
        super().__init__(name="elo_baseline")
        self.elo_diff_feature = elo_diff_feature

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "EloBaselineModel":
        self.classes_ = np.unique(y)
        self.feature_names_ = feature_names or []
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = len(X)
        # Find elo diff column
        idx = 0
        if self.feature_names_ and self.elo_diff_feature in self.feature_names_:
            idx = self.feature_names_.index(self.elo_diff_feature)

        elo_diff = X[:, idx]
        probs = np.zeros((n_samples, len(self.classes_)))

        if len(self.classes_) == 2:
            # Binary: P(Home/P1) = 1 / (1 + 10^(-diff/400))
            p1 = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
            p0 = 1.0 - p1
            probs[:, 0] = p0
            probs[:, 1] = p1
        else:
            # 3-way (Football: HOME, DRAW, AWAY)
            # P(Draw) approx 0.28 gaussian around diff=0
            p_draw = 0.28 * np.exp(-((elo_diff / 400.0) ** 2))
            rem = 1.0 - p_draw
            p_home_rel = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
            p_home = rem * p_home_rel
            p_away = rem * (1.0 - p_home_rel)

            for i, c in enumerate(self.classes_):
                c_str = str(c).upper()
                if "HOME" in c_str or "1" in c_str:
                    probs[:, i] = p_home
                elif "DRAW" in c_str or "X" in c_str:
                    probs[:, i] = p_draw
                else:
                    probs[:, i] = p_away

        # Normalize
        row_sums = probs.sum(axis=1, keepdims=True)
        return probs / np.where(row_sums == 0, 1.0, row_sums)
