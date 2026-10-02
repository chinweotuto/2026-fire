"""Prediction Markets Registry."""

from typing import Dict, List, Optional
from src.markets.base import BasePredictionMarket
from src.markets.football import Football1X2Market, FootballOverUnder25Market, FootballBTTSMarket
from src.markets.tennis import TennisMatchWinnerMarket, TennisSetBettingMarket
from src.markets.basketball import BasketballMoneylineMarket, BasketballTotalPointsMarket, BasketballSpreadMarket


class MarketRegistry:
    """Central registry and lookup service for all prediction markets."""

    def __init__(self):
        self._markets: Dict[str, BasePredictionMarket] = {}
        self._register_defaults()

    def _register_defaults(self):
        defaults = [
            Football1X2Market(),
            FootballOverUnder25Market(),
            FootballBTTSMarket(),
            TennisMatchWinnerMarket(),
            TennisSetBettingMarket(),
            BasketballMoneylineMarket(),
            BasketballTotalPointsMarket(),
            BasketballSpreadMarket()
        ]
        for m in defaults:
            self._markets[m.market_id] = m

    def get_market(self, market_id: str) -> Optional[BasePredictionMarket]:
        return self._markets.get(market_id)

    def get_markets_for_sport(self, sport_id: str) -> List[BasePredictionMarket]:
        return [m for m in self._markets.values() if m.sport_id == sport_id]

    def list_all_markets(self) -> List[BasePredictionMarket]:
        return list(self._markets.values())


market_registry = MarketRegistry()
