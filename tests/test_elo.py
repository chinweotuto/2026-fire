"""Unit tests for sport-specific Elo rating calculations."""

import pytest
from src.features.elo import FootballEloSystem, BasketballEloSystem, TennisEloSystem


def test_football_elo():
    elo = FootballEloSystem(k_factor=30.0, home_advantage=65.0, initial_rating=1500.0)
    exp_h = elo.expected_score(1500.0, 1500.0, is_a_home=True)
    assert exp_h > 0.50  # Home advantage should make expected score > 0.50

    new_h, new_a = elo.update_ratings(1500.0, 1500.0, home_goals=2, away_goals=0)
    assert new_h > 1500.0
    assert new_a < 1500.0


def test_basketball_elo():
    elo = BasketballEloSystem(k_factor=20.0, home_advantage=80.0, initial_rating=1500.0)
    exp_h = elo.expected_score(1500.0, 1500.0, is_a_home=True)
    assert exp_h > 0.50

    # Large blowout margin should produce higher adjustment
    h_win_small, a_loss_small = elo.update_ratings(1500.0, 1500.0, home_pts=102, away_pts=100)
    h_win_blowout, a_loss_blowout = elo.update_ratings(1500.0, 1500.0, home_pts=130, away_pts=90)
    assert (h_win_blowout - 1500.0) > (h_win_small - 1500.0)


def test_tennis_elo():
    elo = TennisEloSystem(k_factor=32.0, initial_rating=1500.0)
    exp_a = elo.expected_score(1600.0, 1400.0)
    assert exp_a > 0.70

    new_w, new_l = elo.update_ratings(1500.0, 1500.0, sets_w=2, sets_l=0)
    assert new_w > 1500.0
    assert new_l < 1500.0
