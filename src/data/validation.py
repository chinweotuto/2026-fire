"""Data Quality Validation and Anti-Leakage Auditing Layer."""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


class DataValidationError(Exception):
    """Raised when incoming match or statistic data violates domain integrity."""
    pass


class AntiLeakageError(Exception):
    """CRITICAL: Raised when future information is detected in feature or training pipelines."""
    pass


def validate_match_record(data: Dict[str, Any]) -> None:
    """Validate common match record fields before insertion."""
    required_fields = ["sport_id", "competition_id", "season", "event_date", "participant_a", "participant_b"]
    for field in required_fields:
        if field not in data or data[field] is None or data[field] == "":
            raise DataValidationError(f"Missing required field: '{field}'")

    # Participant A and B cannot be the same entity
    if str(data["participant_a"]).strip().lower() == str(data["participant_b"]).strip().lower():
        raise DataValidationError(f"Invalid match: Participant A and B are identical ('{data['participant_a']}')")

    event_date = data["event_date"]
    if isinstance(event_date, str):
        try:
            event_date = datetime.fromisoformat(event_date.replace("Z", "+00:00"))
        except Exception as e:
            raise DataValidationError(f"Invalid event_date format '{data['event_date']}': {e}")

    # Future date check for completed matches
    status = data.get("status", "COMPLETED").upper()
    now_utc = datetime.now(timezone.utc)
    if event_date.tzinfo is None:
        event_date = event_date.replace(tzinfo=timezone.utc)

    if status == "COMPLETED" and event_date > now_utc:
        raise DataValidationError(
            f"Future-dated completed match detected: event_date {event_date} > current time {now_utc}"
        )

    # Score plausibility checks
    score_a = data.get("score_a")
    score_b = data.get("score_b")
    sport_id = data["sport_id"].lower()

    if status == "COMPLETED":
        if score_a is None or score_b is None:
            raise DataValidationError("Completed match must have non-null score_a and score_b")
        if score_a < 0 or score_b < 0:
            raise DataValidationError(f"Negative score detected: ({score_a}, {score_b})")

        if sport_id == "football":
            if score_a > 25 or score_b > 25:
                raise DataValidationError(f"Implausible football score: ({score_a}, {score_b})")
        elif sport_id == "basketball":
            if score_a < 30 or score_b < 30 or score_a > 220 or score_b > 220:
                raise DataValidationError(f"Implausible basketball score: ({score_a}, {score_b})")
            if score_a == score_b:
                raise DataValidationError(f"Basketball matches cannot end in a draw: ({score_a}, {score_b})")
        elif sport_id == "tennis":
            if score_a not in (0, 1, 2, 3) or score_b not in (0, 1, 2, 3):
                raise DataValidationError(f"Implausible tennis set score: ({score_a}, {score_b})")
            if score_a == score_b:
                raise DataValidationError(f"Tennis matches cannot end in a tie: ({score_a}, {score_b})")


def validate_anti_leakage(
    feature_timestamp: datetime,
    prediction_timestamp: datetime,
    context_name: str = "FeatureEngine"
) -> None:
    """Enforce strict temporal cut rule: feature_timestamp <= prediction_timestamp.
    
    Raises AntiLeakageError immediately if any lookahead/future data is accessed.
    """
    if feature_timestamp.tzinfo is None:
        feature_timestamp = feature_timestamp.replace(tzinfo=timezone.utc)
    if prediction_timestamp.tzinfo is None:
        prediction_timestamp = prediction_timestamp.replace(tzinfo=timezone.utc)

    if feature_timestamp > prediction_timestamp:
        raise AntiLeakageError(
            f"CRITICAL DATA LEAKAGE in [{context_name}]: Feature timestamp {feature_timestamp} "
            f"is AFTER prediction cutoff timestamp {prediction_timestamp}!"
        )
