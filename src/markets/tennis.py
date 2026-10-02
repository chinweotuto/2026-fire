"""Tennis prediction markets."""

from typing import Optional, Any
from src.markets.base import BasePredictionMarket


class TennisMatchWinnerMarket(BasePredictionMarket):
    def __init__(self):
        super().__init__(
            market_id="tennis_match_winner",
            sport_id="tennis",
            display_name="Match Winner",
            target_type="BINARY",
            outcomes=["PLAYER_1", "PLAYER_2"],
            minimum_samples=100
        )

    def extract_target(self, match: Any) -> Optional[str]:
        if not match.winner_id:
            return None
        return "PLAYER_1" if match.winner_id == match.participant_a_id else "PLAYER_2"


class TennisSetBettingMarket(BasePredictionMarket):
    def __init__(self):
        super().__init__(
            market_id="tennis_set_betting",
            sport_id="tennis",
            display_name="Set Betting (Exact Sets)",
            target_type="MULTICLASS",
            outcomes=["2-0", "2-1", "0-2", "1-2"],
            minimum_samples=100
        )

    def extract_target(self, match: Any) -> Optional[str]:
        if match.score_a is None or match.score_b is None:
            return None
        s1 = match.score_a
        s2 = match.score_b
        score_str = f"{s1}-{s2}"
        if score_str in self.outcomes:
            return score_str
        # Grand slam best of 5 fallback mapping to winner
        if s1 > s2:
            return "2-0" if s2 == 0 else "2-1"
        else:
            return "0-2" if s1 == 0 else "1-2"
