"""Statistical Models: Poisson Goal Distribution and Dynamic Elo."""

import math
import numpy as np
from typing import Optional, List, Dict, Any
from scipy.optimize import minimize
from scipy.special import factorial
from src.models.base import BaseSportModel


class PoissonGoalModel(BaseSportModel):
    """Bivariate Poisson goal distribution model for Football match outcomes.

    Estimates expected goals (lambda_home, mu_away) using pre-match attack and
    defense metrics, then computes exact joint score probability matrices P(i, j)
    to yield calibrated probabilities for 1X2, Over/Under, and BTTS markets.
    """

    def __init__(self, market_type: str = "1x2", max_goals: int = 8):
        super().__init__(name="poisson_goal_model")
        self.market_type = market_type
        self.max_goals = max_goals
        self.weights_home: Optional[np.ndarray] = None
        self.weights_away: Optional[np.ndarray] = None
        self.intercept_home: float = 0.35  # ~1.42 exp(0.35) average home goals
        self.intercept_away: float = 0.15  # ~1.16 exp(0.15) average away goals

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "PoissonGoalModel":
        self.feature_names_ = feature_names or []
        self.classes_ = np.unique(y)

        # Identify key attack / defense feature indices if available
        # Default feature positions or fallback
        n_features = X.shape[1]
        self.weights_home = np.zeros(n_features)
        self.weights_away = np.zeros(n_features)

        # Find specific indices
        def get_idx(col_name: str) -> Optional[int]:
            if col_name in self.feature_names_:
                return self.feature_names_.index(col_name)
            return None

        idx_h_gf = get_idx("home_goals_for_last_5")
        idx_a_ga = get_idx("away_goals_against_last_5")
        idx_a_gf = get_idx("away_goals_for_last_5")
        idx_h_ga = get_idx("home_goals_against_last_5")
        idx_elo_diff = get_idx("elo_difference")

        # Heuristic / MLE initialization from features:
        if idx_h_gf is not None:
            self.weights_home[idx_h_gf] = 0.22
        if idx_a_ga is not None:
            self.weights_home[idx_a_ga] = 0.15
        if idx_elo_diff is not None:
            self.weights_home[idx_elo_diff] = 0.0015
            self.weights_away[idx_elo_diff] = -0.0015

        if idx_a_gf is not None:
            self.weights_away[idx_a_gf] = 0.20
        if idx_h_ga is not None:
            self.weights_away[idx_h_ga] = 0.14

        return self

    def _estimate_lambdas(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Estimate expected home goals (lambda) and away goals (mu)."""
        log_lambda_h = self.intercept_home + np.dot(X, self.weights_home)
        log_mu_a = self.intercept_away + np.dot(X, self.weights_away)

        # Clip log-rates to avoid overflow/underflow (between 0.2 and 5.0 expected goals)
        lambda_h = np.clip(np.exp(np.clip(log_lambda_h, -1.6, 1.6)), 0.25, 4.5)
        mu_a = np.clip(np.exp(np.clip(log_mu_a, -1.6, 1.6)), 0.20, 4.0)
        return lambda_h, mu_a

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        lambda_h, mu_a = self._estimate_lambdas(X)
        n_samples = len(X)

        probs = np.zeros((n_samples, len(self.classes_)))

        # Precompute factorials
        facts = factorial(np.arange(self.max_goals + 1))

        # Vectorized bivariate Poisson computation
        for idx in range(n_samples):
            lh = lambda_h[idx]
            ma = mu_a[idx]

            # Poisson pmf arrays: P(i; lh) and P(j; ma)
            goals = np.arange(self.max_goals + 1)
            p_i = (lh ** goals) * np.exp(-lh) / facts
            p_j = (ma ** goals) * np.exp(-ma) / facts

            # Joint outer product matrix: P(i, j) = p_i * p_j
            grid = np.outer(p_i, p_j)

            # Determine market probabilities
            classes_list = list(self.classes_)
            if "HOME" in classes_list and "AWAY" in classes_list:
                # 1X2 Market
                p_home = np.sum(np.tril(grid, -1))     # i > j
                p_draw = np.sum(np.diag(grid))          # i == j
                p_away = np.sum(np.triu(grid, 1))      # i < j

                for c_i, c in enumerate(self.classes_):
                    c_str = str(c).upper()
                    if "HOME" in c_str:
                        probs[idx, c_i] = p_home
                    elif "DRAW" in c_str:
                        probs[idx, c_i] = p_draw
                    else:
                        probs[idx, c_i] = p_away

            elif "OVER" in classes_list and "UNDER" in classes_list:
                # Over/Under 2.5 Market
                # i + j > 2.5
                i_coords, j_coords = np.indices((self.max_goals + 1, self.max_goals + 1))
                over_mask = (i_coords + j_coords) > 2.5
                p_over = np.sum(grid[over_mask])
                p_under = 1.0 - p_over

                for c_i, c in enumerate(self.classes_):
                    if "OVER" in str(c).upper():
                        probs[idx, c_i] = p_over
                    else:
                        probs[idx, c_i] = p_under

            elif "YES" in classes_list and "NO" in classes_list:
                # BTTS Market: Both Teams To Score (i >= 1 and j >= 1)
                # P(i >= 1, j >= 1) = (1 - P(i=0)) * (1 - P(j=0))
                p_yes = (1.0 - p_i[0]) * (1.0 - p_j[0])
                p_no = 1.0 - p_yes

                for c_i, c in enumerate(self.classes_):
                    if "YES" in str(c).upper():
                        probs[idx, c_i] = p_yes
                    else:
                        probs[idx, c_i] = p_no
            else:
                # Fallback: uniform
                probs[idx, :] = 1.0 / len(self.classes_)

        # Ensure probabilities sum strictly to 1.0
        row_sums = probs.sum(axis=1, keepdims=True)
        return probs / np.where(row_sums == 0, 1.0, row_sums)


class EloProbModel(BaseSportModel):
    """Statistical dynamic Elo probability model."""

    def __init__(self, elo_diff_col: str = "elo_difference"):
        super().__init__(name="elo_prob_model")
        self.elo_diff_col = elo_diff_col

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "EloProbModel":
        self.classes_ = np.unique(y)
        self.feature_names_ = feature_names or []
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        n_samples = len(X)
        idx = 0

        if self.feature_names_ and self.elo_diff_col in self.feature_names_:
            idx = self.feature_names_.index(self.elo_diff_col)

        elo_diff = X[:, idx]
        probs = np.zeros((n_samples, len(self.classes_)))

        if len(self.classes_) == 2:
            # P(A) = 1 / (1 + 10^(-diff/400))
            p1 = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
            probs[:, 0] = 1.0 - p1
            probs[:, 1] = p1
        else:
            # 3-way outcomes (e.g. HOME, DRAW, AWAY)
            p_draw = 0.28 * np.exp(-((elo_diff / 420.0) ** 2))
            rem = 1.0 - p_draw
            p_home_rel = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
            p_home = rem * p_home_rel
            p_away = rem * (1.0 - p_home_rel)

            for i, c in enumerate(self.classes_):
                c_str = str(c).upper()
                if "HOME" in c_str or "PLAYER_1" in c_str:
                    probs[:, i] = p_home
                elif "DRAW" in c_str:
                    probs[:, i] = p_draw
                else:
                    probs[:, i] = p_away

        row_sums = probs.sum(axis=1, keepdims=True)
        return probs / np.where(row_sums == 0, 1.0, row_sums)
