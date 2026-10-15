"""End-to-end integration and smoke verification script."""

import time
import httpx
from starlette.testclient import TestClient
from src.api.app import app


def test_system_e2e():
    try:
        live_client = httpx.Client(base_url="http://127.0.0.1:8000", timeout=5.0)
        res = live_client.get("/health")
        if res.status_code == 200:
            client = live_client
        else:
            client = TestClient(app)
    except Exception:
        client = TestClient(app)

    # 1. Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["total_historical_matches"] >= 14000
    print("Health check passed.")


    # 2. Sports and markets
    res_sports = client.get("/sports")
    assert res_sports.status_code == 200
    sports = [s["sport_id"] for s in res_sports.json()]
    assert set(["football", "tennis", "basketball"]).issubset(set(sports))
    print("Sports verification passed.")

    # 3. Model performance for each sport
    for sport in ["football", "tennis", "basketball"]:
        res_perf = client.get(f"/models/{sport}/performance")
        assert res_perf.status_code == 200
        models = res_perf.json()
        assert len(models) > 0
        for m in models:
            assert m["validation_log_loss"] is not None
            assert m["validation_brier_score"] is not None
            assert m["validation_accuracy"] is not None
        print(f"Model performance for {sport} passed ({len(models)} models).")

    # 4. Probabilistic Prediction: Football
    fb_payload = {
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "competition": "Premier League",
        "market_id": "football_1x2"
    }
    res_fb = client.post("/predict/football", json=fb_payload)
    assert res_fb.status_code == 200
    fb_data = res_fb.json()
    assert abs(sum(fb_data["probabilities"].values()) - 1.0) < 0.05
    assert len(fb_data["top_features"]) > 0
    print("Football prediction passed:", fb_data["probabilities"])

    # 5. Probabilistic Prediction: Tennis
    tn_payload = {
        "player_1": "Carlos Alcaraz",
        "player_2": "Novak Djokovic",
        "surface": "Hard",
        "tournament": "Australian Open",
        "round": "Final",
        "market_id": "tennis_match_winner"
    }
    res_tn = client.post("/predict/tennis", json=tn_payload)
    assert res_tn.status_code == 200
    tn_data = res_tn.json()
    assert abs(sum(tn_data["probabilities"].values()) - 1.0) < 0.05
    print("Tennis prediction passed:", tn_data["probabilities"])

    # 6. Probabilistic Prediction: Basketball
    bb_payload = {
        "home_team": "Boston Celtics",
        "away_team": "Miami Heat",
        "competition": "NBA",
        "market_id": "basketball_moneyline"
    }
    res_bb = client.post("/predict/basketball", json=bb_payload)
    assert res_bb.status_code == 200
    bb_data = res_bb.json()
    assert abs(sum(bb_data["probabilities"].values()) - 1.0) < 0.05
    print("Basketball prediction passed:", bb_data["probabilities"])

    # 7. Upcoming Matches endpoint
    res_upcoming = client.get("/matches/upcoming?sport_id=football")
    assert res_upcoming.status_code == 200
    assert len(res_upcoming.json()) > 0
    print("Upcoming matches endpoint passed.")

    # 8. Recent predictions endpoint
    res_recent = client.get("/predictions/recent")
    assert res_recent.status_code == 200
    assert len(res_recent.json()) > 0
    print("Recent predictions endpoint passed.")

    # 9. Data Quality & Anti-Leakage Audit
    res_audit = client.get("/data/audit")
    assert res_audit.status_code == 200
    audit_data = res_audit.json()
    assert audit_data["leakage_violations"] == 0
    assert audit_data["status"] == "HIGH"
    print("Data audit endpoint passed:", audit_data["sport_breakdown"])

    # 10. Training Trigger and Walk-Forward Backtesting Evaluation
    res_train = client.post("/training/start?sport_id=football&market_id=football_1x2")
    assert res_train.status_code == 200
    run_id = res_train.json()["run_id"]
    time.sleep(1.5)
    res_status = client.get(f"/training/{run_id}")
    assert res_status.status_code == 200
    status_data = res_status.json()
    assert status_data["status"] == "COMPLETED"
    assert "candidate_metrics" in status_data
    assert "production_metrics" in status_data
    print("Continuous learning training run passed:", status_data["candidate_metrics"])

    # 11. Static Frontend Assets
    assert client.get("/").status_code == 200
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    print("Frontend static assets served correctly.")

    print("\nALL END-TO-END VERIFICATION CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_system_e2e()
