"""API integration tests."""

import pytest
from starlette.testclient import TestClient
from src.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["total_historical_matches"] > 0


def test_sports_endpoint(client):
    res = client.get("/sports")
    assert res.status_code == 200
    sports = res.json()
    sport_ids = [s["sport_id"] for s in sports]
    assert "football" in sport_ids
    assert "tennis" in sport_ids
    assert "basketball" in sport_ids


def test_markets_endpoint(client):
    res = client.get("/markets/football")
    assert res.status_code == 200
    markets = res.json()
    m_ids = [m["market_id"] for m in markets]
    assert "football_1x2" in m_ids


def test_football_predict_endpoint(client):
    payload = {
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "competition": "Premier League",
        "market_id": "football_1x2"
    }
    res = client.post("/predict/football", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["sport"] == "football"
    assert data["market"] == "football_1x2"
    assert "probabilities" in data
    probs = data["probabilities"]
    # Probabilities must sum to ~1.0
    total = sum(probs.values())
    assert abs(total - 1.0) < 0.05
    assert len(data["top_features"]) > 0


def test_tennis_predict_endpoint(client):
    payload = {
        "player_1": "Carlos Alcaraz",
        "player_2": "Jannik Sinner",
        "surface": "Hard",
        "tournament": "ATP Masters",
        "round": "Final",
        "market_id": "tennis_match_winner"
    }
    res = client.post("/predict/tennis", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["sport"] == "tennis"
    probs = data["probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 0.05


def test_basketball_predict_endpoint(client):
    payload = {
        "home_team": "Boston Celtics",
        "away_team": "Miami Heat",
        "competition": "NBA",
        "market_id": "basketball_moneyline"
    }
    res = client.post("/predict/basketball", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["sport"] == "basketball"
    probs = data["probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 0.05


def test_manual_match_entry(client):
    payload = {
        "sport_id": "football",
        "competition_name": "Premier League",
        "season": "2024",
        "event_date": "2024-05-19T16:00:00Z",
        "participant_a": "Arsenal",
        "participant_b": "Everton",
        "score_a": 2,
        "score_b": 1,
        "football_stats": {
            "home_shots": 18,
            "away_shots": 5,
            "home_shots_on_target": 7,
            "away_shots_on_target": 2,
            "home_corners": 8,
            "away_corners": 1
        }
    }
    res = client.post("/matches", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert "Arsenal" in data["message"]
