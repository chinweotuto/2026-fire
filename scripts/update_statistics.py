"""Incremental feature and statistics updater.

Scans the database for any matches that do not yet have pre-match causal features
computed, and updates them incrementally using the streaming feature builders.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from sqlalchemy.orm import Session
from src.db.session import SessionLocal
from src.db.models import Match, MatchFeatures
from src.sports.football.features import FootballFeatureEngine
from src.sports.tennis.features import TennisFeatureEngine
from src.sports.basketball.features import BasketballFeatureEngine
from scripts.build_features import build_football_features, build_tennis_features, build_basketball_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def update_unprocessed_statistics():
    db: Session = SessionLocal()
    try:
        processed_ids = set(r[0] for r in db.query(MatchFeatures.match_id).all())
        all_matches = db.query(Match).order_by(Match.event_date.asc()).all()
        unprocessed = [m for m in all_matches if m.match_id not in processed_ids]

        logger.info(f"Total matches in DB: {len(all_matches)}. Unprocessed: {len(unprocessed)}.")

        if not unprocessed:
            logger.info("All matches have up-to-date features and statistics.")
            return

        sports_found = set(m.sport_id for m in unprocessed)
        for s in sports_found:
            logger.info(f"Updating statistics for sport: {s}...")
            if s == "football":
                build_football_features(db, FootballFeatureEngine())
            elif s == "tennis":
                build_tennis_features(db, TennisFeatureEngine())
            elif s == "basketball":
                build_basketball_features(db, BasketballFeatureEngine())

        logger.info("Incremental statistics update complete.")
    finally:
        db.close()


if __name__ == "__main__":
    update_unprocessed_statistics()
