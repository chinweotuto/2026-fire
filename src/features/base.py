"""Base interfaces for time-aware feature extraction."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, Any, List
from src.data.validation import validate_anti_leakage


class BaseFeatureEngine(ABC):
    """Abstract base class ensuring point-in-time causality."""

    def __init__(self, sport_id: str, version: str = "v1.0.0"):
        self.sport_id = sport_id
        self.version = version

    @abstractmethod
    def extract_features(
        self,
        match: Any,
        historical_matches: List[Any],
        as_of_timestamp: datetime
    ) -> Dict[str, float]:
        """Extract pre-match features using only matches before as_of_timestamp."""
        pass

    def check_temporal_integrity(
        self,
        historical_matches: List[Any],
        as_of_timestamp: datetime
    ) -> None:
        """Verify that every match in history occurred strictly before as_of_timestamp."""
        for m in historical_matches:
            m_date = m.event_date
            if m_date.tzinfo is None:
                m_date = m_date.replace(tzinfo=timezone.utc)
            if as_of_timestamp.tzinfo is None:
                as_of_timestamp = as_of_timestamp.replace(tzinfo=timezone.utc)

            validate_anti_leakage(m_date, as_of_timestamp, context_name=f"{self.sport_id}_engine")
