"""Basketball (NBA) prediction markets."""

from typing import Optional, Any
from src.markets.base import BasePredictionMarket


class BasketballMoneylineMarket(BasePredictionMarket):
    def __init__(self):
        super().__init__(
            market_id="basketball_moneyline",
            sport_id="basketball",
            display_name="Match Winner (Moneyline)",
            target_type="BINARY",
            outcomes=["HOME", "AWAY"],
            minimum_samples=100
        )

    def extract_target(self, match: Any) -> Optional[str]:
        if match.score_a is None or match.score_b is None:
            return None
        return "HOME" if match.score_a > match.score_b else "AWAY"


class BasketballTotalPointsMarket(BasePredictionMarket):
    def __init__(self, line: float = 215.5):
        self.line = line
        super().__init__(
            market_id="basketball_total_points",
            sport_id="basketball",
            display_name=f"Total Points Over/Under {line}",
            target_type="BINARY",
            outcomes=["UNDER", "OVER"],
            minimum_samples=100
        )

    def extract_target(self, match: Any) -> Optional[str]:
        if match.score_a is None or match.score_b is None:
            return None
        total = match.score_a + match.score_b
        return "OVER" if total > self.line else "UNDER"


class BasketballSpreadMarket(BasePredictionMarket):
    def __init__(self, line: float = -4.5):
        self.line = line
        super().__init__(
            market_id="basketball_spread",
            sport_id="basketball",
            display_name=f"Point Spread (Home {line})",
            target_type="BINARY",
            outcomes=["AWAY_COVERS", "HOME_COVERS"],
            minimum_samples=100
        )

    def extract_target(self, match: Any) -> Optional[str]:
        if match.score_a is None or match.score_b is None:
            return None
        diff = match.score_a - match.score_b
        # Home covers if diff + line > 0
        return "HOME_COVERS" if (diff + self.line) > 0 else "AWAY_COVERS"
