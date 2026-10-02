"""Pydantic schemas for request and response validation."""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


# -------------------------------------------------------------
# PREDICTION REQUESTS & RESPONSES
# -------------------------------------------------------------

class FootballPredictionRequest(BaseModel):
    home_team: str = Field(..., example="Arsenal")
    away_team: str = Field(..., example="Chelsea")
    competition: str = Field(default="English Premier League", example="English Premier League")
    match_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    market_id: str = Field(default="football_1x2", example="football_1x2")


class TennisPredictionRequest(BaseModel):
    player_1: str = Field(..., example="Carlos Alcaraz")
    player_2: str = Field(..., example="Jannik Sinner")
    surface: str = Field(default="Hard", example="Hard")
    tournament: str = Field(default="ATP Masters", example="ATP Masters")
    round: str = Field(default="Final", example="Final")
    match_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    market_id: str = Field(default="tennis_match_winner", example="tennis_match_winner")


class BasketballPredictionRequest(BaseModel):
    home_team: str = Field(..., example="Boston Celtics")
    away_team: str = Field(..., example="Golden State Warriors")
    competition: str = Field(default="National Basketball Association", example="National Basketball Association")
    match_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    market_id: str = Field(default="basketball_moneyline", example="basketball_moneyline")


class TopFeatureItem(BaseModel):
    feature_key: str
    display_name: str
    value: float
    relative_importance: float


class ModelInfo(BaseModel):
    name: str
    version: str
    algorithm: Optional[str] = None


class PredictionResponse(BaseModel):
    prediction_id: str
    sport: str
    market: str
    participant_a: str
    participant_b: str
    probabilities: Dict[str, float]
    predicted_outcome: str
    model: ModelInfo
    data_quality: str  # 'HIGH', 'MEDIUM', 'LOW'
    top_features: List[TopFeatureItem]
    created_at: datetime


# -------------------------------------------------------------
# MANUAL MATCH ENTRY SCHEMAS
# -------------------------------------------------------------

class FootballStatsInput(BaseModel):
    ht_home_goals: Optional[int] = None
    ht_away_goals: Optional[int] = None
    home_shots: Optional[int] = None
    away_shots: Optional[int] = None
    home_shots_on_target: Optional[int] = None
    away_shots_on_target: Optional[int] = None
    home_corners: Optional[int] = None
    away_corners: Optional[int] = None
    home_fouls: Optional[int] = None
    away_fouls: Optional[int] = None
    home_yellow_cards: Optional[int] = None
    away_yellow_cards: Optional[int] = None
    home_red_cards: Optional[int] = None
    away_red_cards: Optional[int] = None
    home_xg: Optional[float] = None  # None if unknown, NEVER coerced to 0.0
    away_xg: Optional[float] = None


class BasketballStatsInput(BaseModel):
    home_fgm: Optional[int] = None
    home_fga: Optional[int] = None
    home_fg3m: Optional[int] = None
    home_fg3a: Optional[int] = None
    home_ftm: Optional[int] = None
    home_fta: Optional[int] = None
    home_oreb: Optional[int] = None
    home_dreb: Optional[int] = None
    home_reb: Optional[int] = None
    home_ast: Optional[int] = None
    home_tov: Optional[int] = None
    home_stl: Optional[int] = None
    home_blk: Optional[int] = None
    away_fgm: Optional[int] = None
    away_fga: Optional[int] = None
    away_fg3m: Optional[int] = None
    away_fg3a: Optional[int] = None
    away_ftm: Optional[int] = None
    away_fta: Optional[int] = None
    away_oreb: Optional[int] = None
    away_dreb: Optional[int] = None
    away_reb: Optional[int] = None
    away_ast: Optional[int] = None
    away_tov: Optional[int] = None
    away_stl: Optional[int] = None
    away_blk: Optional[int] = None


class TennisStatsInput(BaseModel):
    surface: Optional[str] = "Hard"
    tournament_level: Optional[str] = "A"
    round: Optional[str] = "Final"
    best_of: Optional[int] = 3
    minutes: Optional[int] = None
    p1_aces: Optional[int] = None
    p1_double_faults: Optional[int] = None
    p1_serve_points: Optional[int] = None
    p1_first_in: Optional[int] = None
    p1_first_won: Optional[int] = None
    p1_second_won: Optional[int] = None
    p1_break_points_saved: Optional[int] = None
    p1_break_points_faced: Optional[int] = None
    p2_aces: Optional[int] = None
    p2_double_faults: Optional[int] = None
    p2_serve_points: Optional[int] = None
    p2_first_in: Optional[int] = None
    p2_first_won: Optional[int] = None
    p2_second_won: Optional[int] = None
    p2_break_points_saved: Optional[int] = None
    p2_break_points_faced: Optional[int] = None


class MatchEntryRequest(BaseModel):
    sport_id: str = Field(..., example="football")
    competition_name: str = Field(..., example="Premier League")
    season: str = Field(default="2024", example="2024")
    event_date: datetime = Field(..., example="2024-05-19T16:00:00Z")
    participant_a: str = Field(..., example="Manchester City")
    participant_b: str = Field(..., example="West Ham United")
    score_a: int = Field(..., example=3)
    score_b: int = Field(..., example=1)
    status: str = Field(default="COMPLETED")
    surface: Optional[str] = None
    round: Optional[str] = None
    football_stats: Optional[FootballStatsInput] = None
    basketball_stats: Optional[BasketballStatsInput] = None
    tennis_stats: Optional[TennisStatsInput] = None


class MatchEntryResponse(BaseModel):
    match_id: str
    status: str
    message: str
