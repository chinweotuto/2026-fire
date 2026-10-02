"""CLI Prediction Generator.

Generates calibrated out-of-sample or upcoming predictions with top contributing features.
Outputs JSON compliant with Phase 22 specification.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uuid
import json
import argparse
from datetime import datetime, timezone
import numpy as np
from sqlalchemy.orm import Session
from src.db.session import SessionLocal
from src.db.models import Match, Participant, ModelRecord, ModelVersion, Prediction
from src.data.canonical import resolve_canonical_participant
from src.api.app import load_active_market_model
from src.sports.football.features import FootballFeatureEngine
from src.sports.tennis.features import TennisFeatureEngine
from src.sports.basketball.features import BasketballFeatureEngine
from src.models.explainability import extract_top_contributing_features


def generate_prediction(sport: str, entity_a: str, entity_b: str, market_id: str,
                        surface: str = "Hard", tournament: str = "Grand Slam", round_name: str = "Final",
                        competition: str = "General", match_time: str = None):
    db: Session = SessionLocal()
    try:
        match_dt = datetime.fromisoformat(match_time.replace("Z", "+00:00")) if match_time else datetime.now(timezone.utc)
        if match_dt.tzinfo is None:
            match_dt = match_dt.replace(tzinfo=timezone.utc)

        bundle = load_active_market_model(db, sport, market_id)
        if not bundle:
            print(f"Error: No active model found for {sport} market {market_id}")
            return None

        model = bundle["model"]
        feature_names = bundle["feature_names"]
        classes = model.classes_

        entity_type = "PLAYER" if sport == "tennis" else "TEAM"
        part_a = resolve_canonical_participant(db, entity_a, sport, entity_type=entity_type)
        part_b = resolve_canonical_participant(db, entity_b, sport, entity_type=entity_type)

        history = (
            db.query(Match)
            .filter(Match.sport_id == sport)
            .filter(Match.event_date < match_dt)
            .order_by(Match.event_date.asc())
            .all()
        )

        dummy_match = Match(
            match_id=f"cli_pred_{uuid.uuid4().hex[:8]}",
            sport_id=sport,
            participant_a_id=part_a.participant_id,
            participant_b_id=part_b.participant_id,
            surface=surface if sport == "tennis" else None,
            round=round_name if sport == "tennis" else None,
            event_date=match_dt
        )

        if sport == "football":
            engine = FootballFeatureEngine()
            features = engine.extract_features(dummy_match, history, as_of_timestamp=match_dt)
        elif sport == "tennis":
            engine = TennisFeatureEngine()
            features = engine.extract_features(dummy_match, history, as_of_timestamp=match_dt)
        elif sport == "basketball":
            engine = BasketballFeatureEngine()
            features = engine.extract_features(dummy_match, history, as_of_timestamp=match_dt)
        else:
            raise ValueError(f"Unsupported sport: {sport}")

        vector = [features.get(k, 0.0) for k in feature_names]
        X_sample = [vector]

        probs = model.predict_proba(X_sample)[0]
        prob_dict = {str(c): round(float(p), 4) for c, p in zip(classes, probs)}
        pred_outcome = str(classes[probs.argmax()])

        top_feats = extract_top_contributing_features(model, feature_names, X_sample, top_k=5)
        data_quality = "HIGH" if len(history) >= 200 else ("MEDIUM" if len(history) >= 50 else "LOW")

        response = {
            "sport": sport,
            "market": market_id,
            "participant_a": part_a.canonical_name,
            "participant_b": part_b.canonical_name,
            "probabilities": prob_dict,
            "predicted_outcome": pred_outcome,
            "model": {
                "name": f"{sport}_{market_id}_ensemble",
                "version": bundle.get("version_id", "v1.0.0")
            },
            "data_quality": data_quality,
            "top_features": [f["feature_key"] for f in top_feats]
        }

        print(json.dumps(response, indent=2))
        return response
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Generate Sports Probabilistic Prediction")
    parser.add_argument("--sport", choices=["football", "tennis", "basketball"], required=True)
    parser.add_argument("--a", required=True, help="Participant A (Home or Player 1)")
    parser.add_argument("--b", required=True, help="Participant B (Away or Player 2)")
    parser.add_argument("--market", required=True, help="Market ID (e.g. football_1x2, tennis_match_winner, basketball_moneyline)")
    parser.add_argument("--surface", default="Hard", help="Tennis surface (Hard, Clay, Grass)")
    parser.add_argument("--comp", default="General", help="Competition name")
    parser.add_argument("--time", default=None, help="Match time in ISO format")

    args = parser.parse_args()
    generate_prediction(
        sport=args.sport,
        entity_a=args.a,
        entity_b=args.b,
        market_id=args.market,
        surface=args.surface,
        competition=args.comp,
        match_time=args.time
    )


if __name__ == "__main__":
    main()
