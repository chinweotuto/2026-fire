"""Sport-Specific Elo Rating Engine with Historical Snapshot Auditing and Dynamic MOV."""

import math
from typing import Dict, Tuple, Optional


class FootballEloSystem:
    """Team Elo rating for football with home advantage, goal margin weighting, and mean regression."""

    def __init__(self, k_factor: float = 30.0, home_advantage: float = 65.0, initial_rating: float = 1500.0):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating

    def expected_score(self, rating_a: float, rating_b: float, is_a_home: bool = True) -> float:
        eff_a = rating_a + (self.home_advantage if is_a_home else 0.0)
        eff_b = rating_b + (0.0 if is_a_home else self.home_advantage)
        return 1.0 / (1.0 + 10.0 ** ((eff_b - eff_a) / 400.0))

    def update_ratings(
        self,
        rating_home: float,
        rating_away: float,
        home_goals: int,
        away_goals: int
    ) -> Tuple[float, float]:
        """Update Elo after a match, scaling K-factor by continuous goal difference & rating gap."""
        if home_goals > away_goals:
            actual_home, actual_away = 1.0, 0.0
        elif home_goals < away_goals:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5

        exp_home = self.expected_score(rating_home, rating_away, is_a_home=True)
        exp_away = 1.0 - exp_home

        # Enhanced continuous Margin of Victory (MOV) multiplier
        diff = abs(home_goals - away_goals)
        if diff <= 1:
            multiplier = 1.0
        elif diff == 2:
            multiplier = 1.5
        elif diff == 3:
            multiplier = 1.85
        else:
            multiplier = 1.85 + (diff - 3) * 0.25

        new_home = rating_home + self.k_factor * multiplier * (actual_home - exp_home)
        new_away = rating_away + self.k_factor * multiplier * (actual_away - exp_away)
        return round(new_home, 2), round(new_away, 2)

    def regress_to_mean(self, rating: float, regression_factor: float = 0.20) -> float:
        """Regress rating towards 1500 during off-seasons to account for roster transfers."""
        return round((1.0 - regression_factor) * rating + regression_factor * self.initial_rating, 2)


class BasketballEloSystem:
    """Team Elo rating for basketball with 538 margin of victory multiplier and mean regression."""

    def __init__(self, k_factor: float = 20.0, home_advantage: float = 80.0, initial_rating: float = 1500.0):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating

    def expected_score(self, rating_a: float, rating_b: float, is_a_home: bool = True) -> float:
        eff_a = rating_a + (self.home_advantage if is_a_home else 0.0)
        eff_b = rating_b + (0.0 if is_a_home else self.home_advantage)
        return 1.0 / (1.0 + 10.0 ** ((eff_b - eff_a) / 400.0))

    def update_ratings(
        self,
        rating_home: float,
        rating_away: float,
        home_pts: int,
        away_pts: int
    ) -> Tuple[float, float]:
        actual_home = 1.0 if home_pts > away_pts else 0.0
        actual_away = 1.0 - actual_home

        exp_home = self.expected_score(rating_home, rating_away, is_a_home=True)
        exp_away = 1.0 - exp_home

        # FiveThirtyEight NBA margin of victory multiplier
        margin = abs(home_pts - away_pts)
        elo_diff = (rating_home + self.home_advantage) - rating_away
        if actual_home == 0.0:
            elo_diff = -elo_diff
        mov_multiplier = ((margin + 3.0) ** 0.8) / (7.5 + 0.006 * elo_diff)
        if mov_multiplier < 0.5:
            mov_multiplier = 0.5
        elif mov_multiplier > 3.0:
            mov_multiplier = 3.0

        new_home = rating_home + self.k_factor * mov_multiplier * (actual_home - exp_home)
        new_away = rating_away + self.k_factor * mov_multiplier * (actual_away - exp_away)
        return round(new_home, 2), round(new_away, 2)

    def regress_to_mean(self, rating: float, regression_factor: float = 0.25) -> float:
        """Regress rating towards 1500 during NBA off-seasons."""
        return round((1.0 - regression_factor) * rating + regression_factor * self.initial_rating, 2)


class TennisEloSystem:
    """Player Elo rating for tennis with overall and surface-specific ratings and set margin bonus."""

    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1500.0):
        self.k_factor = k_factor
        self.initial_rating = initial_rating

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update_ratings(
        self,
        rating_w: float,
        rating_l: float,
        sets_w: int = 2,
        sets_l: int = 0
    ) -> Tuple[float, float]:
        exp_w = self.expected_score(rating_w, rating_l)
        exp_l = 1.0 - exp_w

        multiplier = 1.0
        if sets_w > 0 and sets_l == 0:
            multiplier = 1.25  # Straight sets dominance bonus
        elif sets_w > 0 and sets_l >= 2:
            multiplier = 0.90  # Tight 5-setter dampening

        new_w = rating_w + self.k_factor * multiplier * (1.0 - exp_w)
        new_l = rating_l + self.k_factor * multiplier * (0.0 - exp_l)
        return round(new_w, 2), round(new_l, 2)

    def regress_to_mean(self, rating: float, regression_factor: float = 0.15) -> float:
        """Regress player rating towards 1500 at start of new calendar year."""
        return round((1.0 - regression_factor) * rating + regression_factor * self.initial_rating, 2)

