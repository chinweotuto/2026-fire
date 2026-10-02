"""Unit tests for data quality and validation layer."""

import pytest
from datetime import datetime, timezone
from src.data.validation import validate_match_record, DataValidationError


def test_valid_match():
    record = {
        "sport_id": "football",
        "competition_id": "comp_e0",
        "season": "2023",
        "event_date": "2023-05-01T15:00:00Z",
        "participant_a": "Arsenal",
        "participant_b": "Chelsea",
        "score_a": 2,
        "score_b": 1,
        "status": "COMPLETED"
    }
    # Should pass without error
    validate_match_record(record)


def test_identical_participants_rejected():
    record = {
        "sport_id": "football",
        "competition_id": "comp_e0",
        "season": "2023",
        "event_date": "2023-05-01T15:00:00Z",
        "participant_a": "Arsenal",
        "participant_b": "Arsenal",
        "score_a": 1,
        "score_b": 1
    }
    with pytest.raises(DataValidationError, match="identical"):
        validate_match_record(record)


def test_negative_score_rejected():
    record = {
        "sport_id": "football",
        "competition_id": "comp_e0",
        "season": "2023",
        "event_date": "2023-05-01T15:00:00Z",
        "participant_a": "Arsenal",
        "participant_b": "Chelsea",
        "score_a": -1,
        "score_b": 2
    }
    with pytest.raises(DataValidationError, match="Negative score"):
        validate_match_record(record)


def test_basketball_draw_rejected():
    record = {
        "sport_id": "basketball",
        "competition_id": "comp_nba",
        "season": "2023",
        "event_date": "2023-05-01T15:00:00Z",
        "participant_a": "Boston Celtics",
        "participant_b": "Miami Heat",
        "score_a": 105,
        "score_b": 105
    }
    with pytest.raises(DataValidationError, match="cannot end in a draw"):
        validate_match_record(record)
