"""Prediction Market Definitions and Contracts."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np


class BasePredictionMarket(ABC):
    """Abstract contract for sports prediction markets."""

    def __init__(
        self,
        market_id: str,
        sport_id: str,
        display_name: str,
        target_type: str,
        outcomes: List[str],
        minimum_samples: int = 100
    ):
        self.market_id = market_id
        self.sport_id = sport_id
        self.display_name = display_name
        self.target_type = target_type
        self.outcomes = outcomes
        self.minimum_samples = minimum_samples

    @abstractmethod
    def extract_target(self, match: Any) -> Optional[str]:
        """Extract the ground truth outcome string from completed match entity."""
        pass

    def check_sufficiency(self, sample_count: int) -> bool:
        return sample_count >= self.minimum_samples
