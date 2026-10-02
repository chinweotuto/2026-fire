"""Automation script: Ingest raw data files into normalized database schema."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy.orm import Session
from src.core.config import settings
from src.core.logging import logger
from src.db.session import SessionLocal, init_db
from src.db.models import (
    Sport,
    Competition,
    Match,
    FootballMatchStats,
    BasketballMatchStats,
    TennisMatchStats
)
from src.data.canonical import resolve_canonical_participant
from src.data.validation import validate_match_record, DataValidationError
from src.data.sources.football import FootballDataSource
from src.data.sources.tennis import TennisDataSource
from src.data.sources.basketball import BasketballDataSource


def ensure_core_sports(db: Session) -> None:
    sports_data = [
        ("football", "Football / Soccer"),
        ("basketball", "Basketball"),
        ("tennis", "Tennis")
    ]
    for sid, dname in sports_data:
        if not db.query(Sport).filter(Sport.sport_id == sid).first():
            db.add(Sport(sport_id=sid, display_name=dname))
    db.commit()


def import_matches(db: Session, matches_data: list) -> int:
    imported_count = 0
    for data in matches_data:
        try:
            validate_match_record(data)
        except DataValidationError as e:
            logger.debug(f"Skipping invalid record: {e}")
            continue

        sport_id = data["sport_id"]
        comp_id = data["competition_id"]
        comp_name = data.get("competition_name", comp_id)

        # Ensure competition exists
        comp = db.query(Competition).filter(Competition.competition_id == comp_id).first()
        if not comp:
            comp = Competition(
                competition_id=comp_id,
                sport_id=sport_id,
                name=comp_name,
                tier="TIER_1"
            )
            db.add(comp)
            db.flush()

        # Resolve canonical participants
        entity_type = "PLAYER" if sport_id == "tennis" else "TEAM"
        part_a = resolve_canonical_participant(
            db, data["participant_a"], sport_id, entity_type=entity_type, source=data["source"]
        )
        part_b = resolve_canonical_participant(
            db, data["participant_b"], sport_id, entity_type=entity_type, source=data["source"]
        )

        winner_id = None
        if "winner_name" in data and data["winner_name"]:
            winner_part = resolve_canonical_participant(
                db, data["winner_name"], sport_id, entity_type=entity_type, source=data["source"]
            )
            winner_id = winner_part.participant_id
        elif data.get("score_a") is not None and data.get("score_b") is not None:
            if data["score_a"] > data["score_b"]:
                winner_id = part_a.participant_id
            elif data["score_b"] > data["score_a"]:
                winner_id = part_b.participant_id

        # Check if match exists
        existing_match = db.query(Match).filter(Match.match_id == data["match_id"]).first()
        if existing_match:
            continue

        match = Match(
            match_id=data["match_id"],
            sport_id=sport_id,
            competition_id=comp_id,
            season=data["season"],
            event_date=data["event_date"],
            participant_a_id=part_a.participant_id,
            participant_b_id=part_b.participant_id,
            winner_id=winner_id,
            score_a=data.get("score_a"),
            score_b=data.get("score_b"),
            status=data.get("status", "COMPLETED"),
            surface=data.get("surface"),
            round=data.get("round"),
            source=data["source"],
            source_id=data["source_id"]
        )
        db.add(match)
        db.flush()

        # Add sport specific statistics
        if sport_id == "football" and "football_stats" in data:
            st = data["football_stats"]
            f_stat = FootballMatchStats(match_id=match.match_id, **st)
            db.add(f_stat)
        elif sport_id == "basketball" and "basketball_stats" in data:
            st = data["basketball_stats"]
            b_stat = BasketballMatchStats(match_id=match.match_id, **st)
            db.add(b_stat)
        elif sport_id == "tennis" and "tennis_stats" in data:
            st = data["tennis_stats"]
            t_stat = TennisMatchStats(match_id=match.match_id, **st)
            db.add(t_stat)

        imported_count += 1
        if imported_count % 200 == 0:
            db.commit()

    db.commit()
    return imported_count


def main():
    init_db()
    db = SessionLocal()
    try:
        ensure_core_sports(db)
        raw_dir = settings.DATA_RAW_DIR
        total_imported = 0

        # Football
        fb_files = list((raw_dir / "football").glob("*.csv"))
        fb_source = FootballDataSource()
        for f in fb_files:
            logger.info(f"Parsing football file: {f.name}...")
            matches = fb_source.parse_matches(f)
            count = import_matches(db, matches)
            total_imported += count
            logger.info(f"Imported {count} football matches from {f.name}")

        # Tennis
        tennis_files = list((raw_dir / "tennis").glob("*.csv"))
        tennis_source = TennisDataSource()
        for f in tennis_files:
            logger.info(f"Parsing tennis file: {f.name}...")
            matches = tennis_source.parse_matches(f)
            count = import_matches(db, matches)
            total_imported += count
            logger.info(f"Imported {count} tennis matches from {f.name}")

        # Basketball
        bball_files = list((raw_dir / "basketball").glob("*.csv"))
        bball_source = BasketballDataSource()
        for f in bball_files:
            logger.info(f"Parsing basketball file: {f.name}...")
            matches = bball_source.parse_matches(f)
            count = import_matches(db, matches)
            total_imported += count
            logger.info(f"Imported {count} basketball matches from {f.name}")

        logger.info(f"Import process complete. Total matches successfully imported: {total_imported}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
