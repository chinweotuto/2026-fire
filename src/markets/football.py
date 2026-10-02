"""Football prediction markets."""

from typing import Optional, Any
from src.markets.base import BasePredictionMarket


class Football1X2Market(BasePredictionMarket):
    def __init__(self):
        super().__init__(
            market_id="football_1x2",
            sport_id="football",
            display_name="Match Result (1X2)",
            target_type="MULTICLASS",
            outcomes=["HOME", "DRAW", "AWAY"],
            minimum_samples=100
        )

    def extract_target(self, match: Any) -> Optional[str]:
        if match.score_a is None or match.score_b is None:
            return None
        if match.score_a > match.score_b:
            return "HOME"
        elif match.score_b > match.score_a:
            return "AWAY"
        return "DRAW"


class FootballOverUnder25Market(BasePredictionMarket):
    def __init__(self):
        super().__init__(
            market_id="football_over_under_2_5",
            sport_id="football",
            display_name="Total Goals Over/Under 2.5",
            target_type="BINARY",
            outcomes=["UNDER", "OVER"],
            minimum_samples=100
        )

    def extract_target(self, match: Any) -> Optional[str]:
        if match.score_a is None or match.score_b is None:
            return None
        total = match.score_a + match.score_b
        return "OVER" if total > 2.5 else "UNDER"


class FootballBTTSMarket(BasePredictionMarket):
    def __init__(self):
        super().__init__(
            market_id="football_btts",
            sport_id="football",
            display_name="Both Teams To Score (BTTS)",
            target_type="BINARY",
            outcomes=["NO", "YES"],
            minimum_samples=100
        )

    def extract_target(self, match: Any) -> Optional[str]:
        if match.score_a is None or match.score_b is None:
            return None
        return "YES" if match.score_a > 0 and match.score_b > 0 else "NO"
