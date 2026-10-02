"""Automated test suite enforcing zero data leakage."""

import pytest
from datetime import datetime, timezone, timedelta
from src.data.validation import validate_anti_leakage, AntiLeakageError
from src.sports.football.features import FootballFeatureEngine
from src.sports.tennis.features import TennisFeatureEngine
from src.sports.basketball.features import BasketballFeatureEngine


class MockMatch:
    def __init__(self, match_id: str, date: datetime, part_a: str, part_b: str, score_a: int = 2, score_b: int = 1):
        self.match_id = match_id
        self.event_date = date
        self.participant_a_id = part_a
        self.participant_b_id = part_b
        self.score_a = score_a
        self.score_b = score_b
        self.football_stats = None
        self.basketball_stats = None
        self.tennis_stats = None
        self.winner_id = part_a if score_a > score_b else part_b


def test_anti_leakage_assertion_passes_for_past_data():
    t_past = datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc)
    t_pred = datetime(2023, 1, 10, 15, 0, tzinfo=timezone.utc)
    # Should not raise
    validate_anti_leakage(t_past, t_pred)


def test_anti_leakage_assertion_fails_for_future_data():
    t_future = datetime(2023, 1, 15, 12, 0, tzinfo=timezone.utc)
    t_pred = datetime(2023, 1, 10, 15, 0, tzinfo=timezone.utc)
    with pytest.raises(AntiLeakageError):
        validate_anti_leakage(t_future, t_pred)


def test_football_engine_rejects_future_match():
    engine = FootballFeatureEngine()
    t_now = datetime(2024, 5, 1, 15, 0, tzinfo=timezone.utc)
    target_match = MockMatch("target", t_now, "team_a", "team_b")

    # Contaminate history with a future match
    t_future = t_now + timedelta(days=5)
    bad_history = [
        MockMatch("past_1", t_now - timedelta(days=10), "team_a", "team_c"),
        MockMatch("future_1", t_future, "team_a", "team_d")  # LEAKAGE
    ]

    with pytest.raises(AntiLeakageError):
        engine.extract_features(target_match, bad_history, as_of_timestamp=t_now)
