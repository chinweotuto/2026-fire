"""Automation script: High-performance streaming pre-match feature calculation."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import timezone
from sqlalchemy.orm import Session
from src.core.logging import logger
from src.db.session import SessionLocal
from src.db.models import Match, MatchFeatures
from src.sports.football.features import FootballFeatureEngine
from src.sports.tennis.features import TennisFeatureEngine
from src.sports.basketball.features import BasketballFeatureEngine


def build_football_features(db: Session, engine: FootballFeatureEngine):
    logger.info("Computing streaming features for Football...")
    matches = (
        db.query(Match)
        .filter(Match.sport_id == "football")
        .order_by(Match.event_date.asc())
        .all()
    )

    ratings = {}
    matches_by_team = {}
    h2h_matches = {}
    saved_count = 0

    for m in matches:
        m_date = m.event_date
        if m_date.tzinfo is None:
            m_date = m_date.replace(tzinfo=timezone.utc)

        h_id, a_id = m.participant_a_id, m.participant_b_id
        r_h = ratings.get(h_id, engine.elo_system.initial_rating)
        r_a = ratings.get(a_id, engine.elo_system.initial_rating)
        h_m = matches_by_team.get(h_id, [])
        a_m = matches_by_team.get(a_id, [])
        h2h = h2h_matches.get((h_id, a_id), [])

        features = engine.extract_features_fast(m, r_h, r_a, h_m, a_m, h2h, m_date)

        existing = db.query(MatchFeatures).filter(MatchFeatures.match_id == m.match_id).first()
        if not existing:
            db.add(MatchFeatures(
                match_id=m.match_id,
                features_version=engine.version,
                feature_timestamp=m_date,
                features_json=features
            ))
            saved_count += 1
        else:
            existing.features_version = engine.version
            existing.feature_timestamp = m_date
            existing.features_json = features
            saved_count += 1

        # Advance state
        s_a = m.score_a if m.score_a is not None else 0
        s_b = m.score_b if m.score_b is not None else 0
        new_h, new_a = engine.elo_system.update_ratings(r_h, r_a, s_a, s_b)
        ratings[h_id] = new_h
        ratings[a_id] = new_a
        matches_by_team.setdefault(h_id, []).append(m)
        matches_by_team.setdefault(a_id, []).append(m)
        h2h_matches.setdefault((h_id, a_id), []).append(m)
        h2h_matches.setdefault((a_id, h_id), []).append(m)

        if saved_count > 0 and saved_count % 500 == 0:
            db.commit()

    db.commit()
    logger.info(f"Finished Football features. Total records processed: {saved_count}")


def build_tennis_features(db: Session, engine: TennisFeatureEngine):
    logger.info("Computing streaming features for Tennis...")
    matches = (
        db.query(Match)
        .filter(Match.sport_id == "tennis")
        .order_by(Match.event_date.asc())
        .all()
    )

    overall_elo = {}
    surface_elo = {}
    matches_by_player = {}
    saved_count = 0

    for m in matches:
        m_date = m.event_date
        if m_date.tzinfo is None:
            m_date = m_date.replace(tzinfo=timezone.utc)

        p1_id, p2_id = m.participant_a_id, m.participant_b_id
        surf = (m.surface or "Hard").strip().capitalize()

        p1_ov = overall_elo.get(p1_id, engine.elo_system.initial_rating)
        p2_ov = overall_elo.get(p2_id, engine.elo_system.initial_rating)
        p1_sf = surface_elo.get(f"{p1_id}_{surf}", p1_ov)
        p2_sf = surface_elo.get(f"{p2_id}_{surf}", p2_ov)
        p1_m = matches_by_player.get(p1_id, [])
        p2_m = matches_by_player.get(p2_id, [])

        features = engine.extract_features_fast(m, p1_ov, p2_ov, p1_sf, p2_sf, p1_m, p2_m, m_date)

        existing = db.query(MatchFeatures).filter(MatchFeatures.match_id == m.match_id).first()
        if not existing:
            db.add(MatchFeatures(
                match_id=m.match_id,
                features_version=engine.version,
                feature_timestamp=m_date,
                features_json=features
            ))
            saved_count += 1
        else:
            existing.features_version = engine.version
            existing.feature_timestamp = m_date
            existing.features_json = features
            saved_count += 1

        # Advance state
        w_id = m.winner_id
        if w_id:
            l_id = p2_id if w_id == p1_id else p1_id
            rw = overall_elo.get(w_id, engine.elo_system.initial_rating)
            rl = overall_elo.get(l_id, engine.elo_system.initial_rating)
            w_sets = max(m.score_a or 0, m.score_b or 0)
            l_sets = min(m.score_a or 0, m.score_b or 0)
            new_rw, new_rl = engine.elo_system.update_ratings(rw, rl, sets_w=w_sets, sets_l=l_sets)
            overall_elo[w_id] = new_rw
            overall_elo[l_id] = new_rl

            # Surface elo
            rsw = surface_elo.get(f"{w_id}_{surf}", rw)
            rsl = surface_elo.get(f"{l_id}_{surf}", rl)
            new_rsw, new_rsl = engine.elo_system.update_ratings(rsw, rsl, sets_w=w_sets, sets_l=l_sets)
            surface_elo[f"{w_id}_{surf}"] = new_rsw
            surface_elo[f"{l_id}_{surf}"] = new_rsl

        matches_by_player.setdefault(p1_id, []).append(m)
        matches_by_player.setdefault(p2_id, []).append(m)

        if saved_count > 0 and saved_count % 1000 == 0:
            db.commit()

    db.commit()
    logger.info(f"Finished Tennis features. Total records processed: {saved_count}")


def build_basketball_features(db: Session, engine: BasketballFeatureEngine):
    logger.info("Computing streaming features for Basketball...")
    matches = (
        db.query(Match)
        .filter(Match.sport_id == "basketball")
        .order_by(Match.event_date.asc())
        .all()
    )

    ratings = {}
    matches_by_team = {}
    saved_count = 0

    for m in matches:
        m_date = m.event_date
        if m_date.tzinfo is None:
            m_date = m_date.replace(tzinfo=timezone.utc)

        h_id, a_id = m.participant_a_id, m.participant_b_id
        r_h = ratings.get(h_id, engine.elo_system.initial_rating)
        r_a = ratings.get(a_id, engine.elo_system.initial_rating)
        h_m = matches_by_team.get(h_id, [])
        a_m = matches_by_team.get(a_id, [])

        features = engine.extract_features_fast(m, r_h, r_a, h_m, a_m, m_date)

        existing = db.query(MatchFeatures).filter(MatchFeatures.match_id == m.match_id).first()
        if not existing:
            db.add(MatchFeatures(
                match_id=m.match_id,
                features_version=engine.version,
                feature_timestamp=m_date,
                features_json=features
            ))
            saved_count += 1
        else:
            existing.features_version = engine.version
            existing.feature_timestamp = m_date
            existing.features_json = features
            saved_count += 1

        s_a = m.score_a if m.score_a is not None else 100
        s_b = m.score_b if m.score_b is not None else 100
        new_h, new_a = engine.elo_system.update_ratings(r_h, r_a, s_a, s_b)
        ratings[h_id] = new_h
        ratings[a_id] = new_a
        matches_by_team.setdefault(h_id, []).append(m)
        matches_by_team.setdefault(a_id, []).append(m)

        if saved_count > 0 and saved_count % 500 == 0:
            db.commit()

    db.commit()
    logger.info(f"Finished Basketball features. Total records processed: {saved_count}")


def main():
    db = SessionLocal()
    try:
        build_football_features(db, FootballFeatureEngine())
        build_tennis_features(db, TennisFeatureEngine())
        build_basketball_features(db, BasketballFeatureEngine())
        logger.info("All streaming feature pipelines finished successfully!")
    finally:
        db.close()


if __name__ == "__main__":
    main()

