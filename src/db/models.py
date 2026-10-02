"""SQLAlchemy relational schema models for normalized sports data."""

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Text,
    JSON,
    Index
)
from sqlalchemy.orm import relationship
from src.db.session import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Sport(Base):
    __tablename__ = "sports"

    sport_id = Column(String(32), primary_key=True)  # 'football', 'basketball', 'tennis'
    display_name = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    competitions = relationship("Competition", back_populates="sport")
    participants = relationship("Participant", back_populates="sport")


class Competition(Base):
    __tablename__ = "competitions"

    competition_id = Column(String(64), primary_key=True)
    sport_id = Column(String(32), ForeignKey("sports.sport_id"), nullable=False)
    name = Column(String(128), nullable=False)
    country = Column(String(64), nullable=True)
    tier = Column(String(32), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    sport = relationship("Sport", back_populates="competitions")
    matches = relationship("Match", back_populates="competition")


class Participant(Base):
    __tablename__ = "participants"

    participant_id = Column(String(64), primary_key=True)
    sport_id = Column(String(32), ForeignKey("sports.sport_id"), nullable=False)
    entity_type = Column(String(16), nullable=False)  # 'TEAM' or 'PLAYER'
    canonical_name = Column(String(128), nullable=False)
    country_code = Column(String(8), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    sport = relationship("Sport", back_populates="participants")
    aliases = relationship("ParticipantAlias", back_populates="participant")


class ParticipantAlias(Base):
    __tablename__ = "participant_aliases"

    alias_id = Column(Integer, primary_key=True, autoincrement=True)
    participant_id = Column(String(64), ForeignKey("participants.participant_id"), nullable=False)
    alias_name = Column(String(128), nullable=False, unique=True, index=True)
    source = Column(String(32), nullable=False)

    participant = relationship("Participant", back_populates="aliases")


class Match(Base):
    __tablename__ = "matches"

    match_id = Column(String(64), primary_key=True)
    sport_id = Column(String(32), ForeignKey("sports.sport_id"), nullable=False, index=True)
    competition_id = Column(String(64), ForeignKey("competitions.competition_id"), nullable=False)
    season = Column(String(16), nullable=False)
    event_date = Column(DateTime(timezone=True), nullable=False, index=True)

    participant_a_id = Column(String(64), ForeignKey("participants.participant_id"), nullable=False)
    participant_b_id = Column(String(64), ForeignKey("participants.participant_id"), nullable=False)
    winner_id = Column(String(64), ForeignKey("participants.participant_id"), nullable=True)

    score_a = Column(Integer, nullable=True)
    score_b = Column(Integer, nullable=True)
    status = Column(String(24), nullable=False, default="COMPLETED")  # 'SCHEDULED', 'COMPLETED', etc.

    surface = Column(String(24), nullable=True)  # For Tennis: 'HARD', 'CLAY', 'GRASS'
    round = Column(String(24), nullable=True)
    source = Column(String(32), nullable=False)
    source_id = Column(String(64), nullable=False)

    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    competition = relationship("Competition", back_populates="matches")
    participant_a = relationship("Participant", foreign_keys=[participant_a_id])
    participant_b = relationship("Participant", foreign_keys=[participant_b_id])
    winner = relationship("Participant", foreign_keys=[winner_id])

    football_stats = relationship("FootballMatchStats", back_populates="match", uselist=False)
    basketball_stats = relationship("BasketballMatchStats", back_populates="match", uselist=False)
    tennis_stats = relationship("TennisMatchStats", back_populates="match", uselist=False)
    elo_ratings = relationship("PreMatchEloSnapshot", back_populates="match")
    features = relationship("MatchFeatures", back_populates="match", uselist=False)
    predictions = relationship("Prediction", back_populates="match")

    __table_args__ = (
        Index("idx_matches_sport_date", "sport_id", "event_date"),
        Index("idx_matches_participants", "participant_a_id", "participant_b_id"),
    )


class FootballMatchStats(Base):
    __tablename__ = "football_match_stats"

    stat_id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(64), ForeignKey("matches.match_id"), nullable=False, unique=True)

    ht_home_goals = Column(Integer, nullable=True)
    ht_away_goals = Column(Integer, nullable=True)
    home_shots = Column(Integer, nullable=True)
    away_shots = Column(Integer, nullable=True)
    home_shots_on_target = Column(Integer, nullable=True)
    away_shots_on_target = Column(Integer, nullable=True)
    home_corners = Column(Integer, nullable=True)
    away_corners = Column(Integer, nullable=True)
    home_fouls = Column(Integer, nullable=True)
    away_fouls = Column(Integer, nullable=True)
    home_yellow_cards = Column(Integer, nullable=True)
    away_yellow_cards = Column(Integer, nullable=True)
    home_red_cards = Column(Integer, nullable=True)
    away_red_cards = Column(Integer, nullable=True)
    home_xg = Column(Float, nullable=True)  # Strictly nullable, never coerced to 0.0
    away_xg = Column(Float, nullable=True)

    match = relationship("Match", back_populates="football_stats")


class BasketballMatchStats(Base):
    __tablename__ = "basketball_match_stats"

    stat_id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(64), ForeignKey("matches.match_id"), nullable=False, unique=True)

    home_fgm = Column(Integer, nullable=True)
    home_fga = Column(Integer, nullable=True)
    home_fg3m = Column(Integer, nullable=True)
    home_fg3a = Column(Integer, nullable=True)
    home_ftm = Column(Integer, nullable=True)
    home_fta = Column(Integer, nullable=True)
    home_oreb = Column(Integer, nullable=True)
    home_dreb = Column(Integer, nullable=True)
    home_reb = Column(Integer, nullable=True)
    home_ast = Column(Integer, nullable=True)
    home_tov = Column(Integer, nullable=True)
    home_stl = Column(Integer, nullable=True)
    home_blk = Column(Integer, nullable=True)
    home_pace = Column(Float, nullable=True)
    home_offensive_rating = Column(Float, nullable=True)

    away_fgm = Column(Integer, nullable=True)
    away_fga = Column(Integer, nullable=True)
    away_fg3m = Column(Integer, nullable=True)
    away_fg3a = Column(Integer, nullable=True)
    away_ftm = Column(Integer, nullable=True)
    away_fta = Column(Integer, nullable=True)
    away_oreb = Column(Integer, nullable=True)
    away_dreb = Column(Integer, nullable=True)
    away_reb = Column(Integer, nullable=True)
    away_ast = Column(Integer, nullable=True)
    away_tov = Column(Integer, nullable=True)
    away_stl = Column(Integer, nullable=True)
    away_blk = Column(Integer, nullable=True)
    away_pace = Column(Float, nullable=True)
    away_offensive_rating = Column(Float, nullable=True)

    match = relationship("Match", back_populates="basketball_stats")


class TennisMatchStats(Base):
    __tablename__ = "tennis_match_stats"

    stat_id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(64), ForeignKey("matches.match_id"), nullable=False, unique=True)

    surface = Column(String(24), nullable=True)
    tourney_level = Column(String(16), nullable=True)
    best_of = Column(Integer, default=3)
    minutes = Column(Integer, nullable=True)

    p1_rank = Column(Integer, nullable=True)
    p2_rank = Column(Integer, nullable=True)

    p1_aces = Column(Integer, nullable=True)
    p1_double_faults = Column(Integer, nullable=True)
    p1_serve_points = Column(Integer, nullable=True)
    p1_first_in = Column(Integer, nullable=True)
    p1_first_won = Column(Integer, nullable=True)
    p1_second_won = Column(Integer, nullable=True)
    p1_break_points_saved = Column(Integer, nullable=True)
    p1_break_points_faced = Column(Integer, nullable=True)

    p2_aces = Column(Integer, nullable=True)
    p2_double_faults = Column(Integer, nullable=True)
    p2_serve_points = Column(Integer, nullable=True)
    p2_first_in = Column(Integer, nullable=True)
    p2_first_won = Column(Integer, nullable=True)
    p2_second_won = Column(Integer, nullable=True)
    p2_break_points_saved = Column(Integer, nullable=True)
    p2_break_points_faced = Column(Integer, nullable=True)

    match = relationship("Match", back_populates="tennis_stats")


class PreMatchEloSnapshot(Base):
    __tablename__ = "elo_ratings"

    elo_id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(64), ForeignKey("matches.match_id"), nullable=False)
    participant_id = Column(String(64), ForeignKey("participants.participant_id"), nullable=False)
    rating_type = Column(String(32), nullable=False)  # 'OVERALL', 'HARD', 'CLAY', 'GRASS'
    pre_match_elo = Column(Float, nullable=False)
    post_match_elo = Column(Float, nullable=True)
    rating_timestamp = Column(DateTime(timezone=True), nullable=False)

    match = relationship("Match", back_populates="elo_ratings")
    participant = relationship("Participant")


class MatchFeatures(Base):
    __tablename__ = "match_features"

    feature_id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(64), ForeignKey("matches.match_id"), nullable=False, unique=True)
    features_version = Column(String(32), nullable=False)
    feature_timestamp = Column(DateTime(timezone=True), nullable=False)
    features_json = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    match = relationship("Match", back_populates="features")


class PredictionMarket(Base):
    __tablename__ = "prediction_markets"

    market_id = Column(String(64), primary_key=True)
    sport_id = Column(String(32), ForeignKey("sports.sport_id"), nullable=False)
    market_name = Column(String(128), nullable=False)
    target_type = Column(String(32), nullable=False)  # 'MULTICLASS', 'BINARY', 'REGRESSION'
    outcomes = Column(JSON, nullable=False)
    required_features = Column(JSON, nullable=False)
    minimum_training_samples = Column(Integer, default=100)
    enabled = Column(Boolean, default=True)
    version = Column(String(16), default="1.0.0")


class ModelRecord(Base):
    __tablename__ = "models"

    model_id = Column(String(64), primary_key=True)
    sport_id = Column(String(32), ForeignKey("sports.sport_id"), nullable=False)
    market_id = Column(String(64), nullable=False)
    model_name = Column(String(128), nullable=False)
    algorithm = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    versions = relationship("ModelVersion", back_populates="model")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    version_id = Column(String(64), primary_key=True)
    model_id = Column(String(64), ForeignKey("models.model_id"), nullable=False)
    version_tag = Column(String(32), nullable=False)
    training_start_date = Column(DateTime(timezone=True), nullable=False)
    training_end_date = Column(DateTime(timezone=True), nullable=False)
    training_samples = Column(Integer, nullable=False)

    validation_log_loss = Column(Float, nullable=True)
    validation_brier_score = Column(Float, nullable=True)
    validation_accuracy = Column(Float, nullable=True)
    calibration_method = Column(String(32), default="isotonic")
    hyperparameters = Column(JSON, nullable=True)
    artifact_path = Column(String(256), nullable=True)

    is_active = Column(Boolean, default=False)
    promoted_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    model = relationship("ModelRecord", back_populates="versions")
    predictions = relationship("Prediction", back_populates="model_version")


class Prediction(Base):
    __tablename__ = "predictions"

    prediction_id = Column(String(64), primary_key=True)
    match_id = Column(String(64), ForeignKey("matches.match_id"), nullable=False)
    market_id = Column(String(64), nullable=False)
    version_id = Column(String(64), ForeignKey("model_versions.version_id"), nullable=False)

    prediction_timestamp = Column(DateTime(timezone=True), nullable=False)
    probabilities = Column(JSON, nullable=False)
    predicted_outcome = Column(String(64), nullable=False)
    data_quality = Column(String(16), nullable=False)  # 'HIGH', 'MEDIUM', 'LOW'
    top_features = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    match = relationship("Match", back_populates="predictions")
    model_version = relationship("ModelVersion", back_populates="predictions")
    result = relationship("PredictionResult", back_populates="prediction", uselist=False)

    __table_args__ = (
        Index("idx_predictions_match_market", "match_id", "market_id"),
    )


class PredictionResult(Base):
    __tablename__ = "prediction_results"

    result_id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(String(64), ForeignKey("predictions.prediction_id"), nullable=False, unique=True)
    actual_outcome = Column(String(64), nullable=False)
    is_correct = Column(Boolean, nullable=False)
    log_loss = Column(Float, nullable=False)
    brier_score = Column(Float, nullable=False)
    evaluated_at = Column(DateTime(timezone=True), default=utc_now)

    prediction = relationship("Prediction", back_populates="result")
