"""Football (Soccer) Pre-Match Feature Engine with EWMA and xG approximations."""

import math
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from src.features.base import BaseFeatureEngine
from src.features.elo import FootballEloSystem


class FootballFeatureEngine(BaseFeatureEngine):
    """Generates 100% pre-match causal features for football fixtures with EWMA and xG."""

    def __init__(self, version: str = "v1.1.0"):
        super().__init__(sport_id="football", version=version)
        self.elo_system = FootballEloSystem()

    def extract_features_fast(
        self,
        match: Any,
        home_elo: float,
        away_elo: float,
        home_matches: List[Any],
        away_matches: List[Any],
        h2h_matches: List[Any],
        as_of_timestamp: datetime
    ) -> Dict[str, float]:
        home_id = match.participant_a_id
        away_id = match.participant_b_id
        elo_diff = (home_elo + self.elo_system.home_advantage) - away_elo

        def get_team_stats(team_id: str, matches: List[Any], window: int) -> Dict[str, float]:
            recent = matches[-window:] if len(matches) >= window else matches
            if not recent:
                return {
                    "points": 1.2, "win_rate": 0.33, "goals_for": 1.2, "goals_against": 1.2,
                    "shots": 12.0, "shots_on_target": 4.0, "shots_allowed": 12.0,
                    "clean_sheets": 0.25, "xg_proxy": 1.30, "ewma_gf": 1.20, "ewma_ga": 1.20
                }

            pts, wins, gf, ga, shots, sot, s_allowed, cs = 0, 0, 0, 0, 0, 0, 0, 0
            ewma_gf, ewma_ga, ewma_weight_total = 0.0, 0.0, 0.0
            n = len(recent)

            for i, m in enumerate(recent):
                is_home = (m.participant_a_id == team_id)
                t_score = m.score_a if is_home else m.score_b
                opp_score = m.score_b if is_home else m.score_a
                t_score = t_score if t_score is not None else 0
                opp_score = opp_score if opp_score is not None else 0

                gf += t_score
                ga += opp_score
                if opp_score == 0:
                    cs += 1

                if t_score > opp_score:
                    pts += 3
                    wins += 1
                elif t_score == opp_score:
                    pts += 1

                # EWMA weight with decay alpha = 0.85
                w = 0.85 ** (n - 1 - i)
                ewma_gf += w * t_score
                ewma_ga += w * opp_score
                ewma_weight_total += w

                if m.football_stats:
                    st = m.football_stats
                    shots += (st.home_shots if is_home else st.away_shots) or 12
                    sot += (st.home_shots_on_target if is_home else st.away_shots_on_target) or 4
                    s_allowed += (st.away_shots if is_home else st.home_shots) or 12
                else:
                    shots += 12
                    sot += 4
                    s_allowed += 12

            ewma_denom = max(1e-5, ewma_weight_total)
            avg_sot = sot / n
            avg_shots = shots / n
            # Non-penalty xG proxy formula: 0.31 * sot + 0.045 * off_target
            off_target = max(0.0, avg_shots - avg_sot)
            xg_proxy = 0.31 * avg_sot + 0.045 * off_target

            return {
                "points": pts / n,
                "win_rate": wins / n,
                "goals_for": gf / n,
                "goals_against": ga / n,
                "shots": avg_shots,
                "shots_on_target": avg_sot,
                "shots_allowed": s_allowed / n,
                "clean_sheets": cs / n,
                "xg_proxy": xg_proxy,
                "ewma_gf": ewma_gf / ewma_denom,
                "ewma_ga": ewma_ga / ewma_denom
            }

        h_stat_5 = get_team_stats(home_id, home_matches, 5)
        h_stat_10 = get_team_stats(home_id, home_matches, 10)
        a_stat_5 = get_team_stats(away_id, away_matches, 5)
        a_stat_10 = get_team_stats(away_id, away_matches, 10)

        # Isolated venue form
        home_isolated = [m for m in home_matches if m.participant_a_id == home_id]
        away_isolated = [m for m in away_matches if m.participant_b_id == away_id]
        h_venue_stats = get_team_stats(home_id, home_isolated, 5)
        a_venue_stats = get_team_stats(away_id, away_isolated, 5)

        # Rest days
        def calc_rest(matches: List[Any]) -> float:
            if not matches:
                return 7.0
            last_date = matches[-1].event_date
            if last_date.tzinfo is None:
                last_date = last_date.replace(tzinfo=timezone.utc)
            delta = (as_of_timestamp - last_date).total_seconds() / 86400.0
            return max(1.0, min(21.0, delta))

        home_rest = calc_rest(home_matches)
        away_rest = calc_rest(away_matches)

        def matches_in_days(matches: List[Any], days: int) -> int:
            cnt = 0
            for m in reversed(matches):
                m_date = m.event_date
                if m_date.tzinfo is None:
                    m_date = m_date.replace(tzinfo=timezone.utc)
                if (as_of_timestamp - m_date).total_seconds() / 86400.0 <= days:
                    cnt += 1
                else:
                    break
            return cnt

        # Head to Head
        h2h_wins = 0
        h2h_draws = 0
        for m in h2h_matches[-5:]:
            is_home_first = (m.participant_a_id == home_id)
            hs = m.score_a if is_home_first else m.score_b
            as_ = m.score_b if is_home_first else m.score_a
            hs = hs or 0
            as_ = as_ or 0
            if hs > as_:
                h2h_wins += 1
            elif hs == as_:
                h2h_draws += 1
        h2h_n = len(h2h_matches[-5:]) if h2h_matches else 0
        h2h_win_rate = (h2h_wins + 1.0) / (h2h_n + 2.0) if h2h_n > 0 else 0.45
        h2h_draw_rate = (h2h_draws + 0.5) / (h2h_n + 2.0) if h2h_n > 0 else 0.28

        return {
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_difference": elo_diff,
            "home_points_last_5": h_stat_5["points"],
            "home_points_last_10": h_stat_10["points"],
            "away_points_last_5": a_stat_5["points"],
            "away_points_last_10": a_stat_10["points"],
            "home_goals_for_last_5": h_stat_5["goals_for"],
            "home_goals_against_last_5": h_stat_5["goals_against"],
            "away_goals_for_last_5": a_stat_5["goals_for"],
            "away_goals_against_last_5": a_stat_5["goals_against"],
            "home_shots_per_game": h_stat_10["shots"],
            "home_shots_on_target_per_game": h_stat_10["shots_on_target"],
            "away_shots_per_game": a_stat_10["shots"],
            "away_shots_on_target_per_game": a_stat_10["shots_on_target"],
            "home_clean_sheet_rate": h_stat_10["clean_sheets"],
            "away_clean_sheet_rate": a_stat_10["clean_sheets"],
            "home_xg_proxy": h_stat_5["xg_proxy"],
            "away_xg_proxy": a_stat_5["xg_proxy"],
            "home_ewma_goals_for": h_stat_5["ewma_gf"],
            "away_ewma_goals_for": a_stat_5["ewma_gf"],
            "home_venue_points_last_5": h_venue_stats["points"],
            "away_venue_points_last_5": a_venue_stats["points"],
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "rest_advantage": home_rest - away_rest,
            "home_matches_last_7_days": float(matches_in_days(home_matches, 7)),
            "away_matches_last_7_days": float(matches_in_days(away_matches, 7)),
            "h2h_win_rate": h2h_win_rate,
            "h2h_draw_rate": h2h_draw_rate
        }

    def extract_features(
        self,
        match: Any,
        historical_matches: List[Any],
        as_of_timestamp: datetime
    ) -> Dict[str, float]:
        self.check_temporal_integrity(historical_matches, as_of_timestamp)
        home_id = match.participant_a_id
        away_id = match.participant_b_id

        ratings: Dict[str, float] = {}
        for h in historical_matches:
            r_home = ratings.get(h.participant_a_id, self.elo_system.initial_rating)
            r_away = ratings.get(h.participant_b_id, self.elo_system.initial_rating)
            h_score = h.score_a if h.score_a is not None else 0
            a_score = h.score_b if h.score_b is not None else 0
            new_h, new_a = self.elo_system.update_ratings(r_home, r_away, h_score, a_score)
            ratings[h.participant_a_id] = new_h
            ratings[h.participant_b_id] = new_a

        h_matches = [m for m in historical_matches if m.participant_a_id == home_id or m.participant_b_id == home_id]
        a_matches = [m for m in historical_matches if m.participant_a_id == away_id or m.participant_b_id == away_id]
        h2h = [
            m for m in historical_matches
            if (m.participant_a_id == home_id and m.participant_b_id == away_id) or
               (m.participant_a_id == away_id and m.participant_b_id == home_id)
        ]

        return self.extract_features_fast(
            match,
            ratings.get(home_id, self.elo_system.initial_rating),
            ratings.get(away_id, self.elo_system.initial_rating),
            h_matches,
            a_matches,
            h2h,
            as_of_timestamp
        )

