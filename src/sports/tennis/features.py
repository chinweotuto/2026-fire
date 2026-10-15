"""Tennis Pre-Match Feature Engine with EWMA and Surface Dominance."""

import math
from datetime import datetime, timezone
from typing import Dict, List, Any
from src.features.base import BaseFeatureEngine
from src.features.elo import TennisEloSystem


class TennisFeatureEngine(BaseFeatureEngine):
    """Generates 100% pre-match causal features for tennis matches with EWMA and surface dominance."""

    def __init__(self, version: str = "v1.1.0"):
        super().__init__(sport_id="tennis", version=version)
        self.elo_system = TennisEloSystem()

    def extract_features_fast(
        self,
        match: Any,
        p1_overall_elo: float,
        p2_overall_elo: float,
        p1_surf_elo: float,
        p2_surf_elo: float,
        p1_matches: List[Any],
        p2_matches: List[Any],
        as_of_timestamp: datetime
    ) -> Dict[str, float]:
        p1_id = match.participant_a_id
        p2_id = match.participant_b_id
        surface = (match.surface or "Hard").strip().capitalize()

        def get_player_form(player_id: str, matches: List[Any], window: int) -> Dict[str, float]:
            recent = matches[-window:] if len(matches) >= window else matches
            if not recent:
                return {
                    "win_rate": 0.5, "first_serve_in": 0.62, "first_serve_won": 0.70,
                    "second_serve_won": 0.50, "ace_rate": 0.08, "df_rate": 0.04,
                    "ewma_win": 0.50, "serve_dominance": 1.15, "recent_sets_played": 6.0
                }

            wins, svpt, first_in, first_won, second_won, aces, dfs = 0, 0, 0, 0, 0, 0, 0
            ewma_wins, ewma_weight_total, total_sets = 0.0, 0.0, 0
            n = len(recent)

            for i, m in enumerate(recent):
                is_win = (m.winner_id == player_id)
                if is_win:
                    wins += 1

                w = 0.85 ** (n - 1 - i)
                ewma_wins += w * (1.0 if is_win else 0.0)
                ewma_weight_total += w

                s_a = m.score_a or 0
                s_b = m.score_b or 0
                total_sets += (s_a + s_b)

                if m.tennis_stats:
                    st = m.tennis_stats
                    is_p1 = (m.participant_a_id == player_id)
                    s_svpt = (st.p1_serve_points if is_p1 else st.p2_serve_points) or 60
                    s_1in = (st.p1_first_in if is_p1 else st.p2_first_in) or 38
                    s_1won = (st.p1_first_won if is_p1 else st.p2_first_won) or 27
                    s_2won = (st.p1_second_won if is_p1 else st.p2_second_won) or 11
                    s_ace = (st.p1_aces if is_p1 else st.p2_aces) or 4
                    s_df = (st.p1_double_faults if is_p1 else st.p2_double_faults) or 2

                    svpt += s_svpt
                    first_in += s_1in
                    first_won += s_1won
                    second_won += s_2won
                    aces += s_ace
                    dfs += s_df

            svpt_denom = max(1, svpt)
            first_in_denom = max(1, first_in)
            second_denom = max(1, svpt - first_in)
            ewma_denom = max(1e-5, ewma_weight_total)

            p_1in = first_in / svpt_denom
            p_1won = first_won / first_in_denom
            p_2won = second_won / second_denom
            serve_dominance = (p_1in * p_1won) + ((1.0 - p_1in) * p_2won)

            return {
                "win_rate": wins / n,
                "first_serve_in": p_1in,
                "first_serve_won": p_1won,
                "second_serve_won": p_2won,
                "ace_rate": aces / svpt_denom,
                "df_rate": dfs / svpt_denom,
                "ewma_win": ewma_wins / ewma_denom,
                "serve_dominance": serve_dominance * 2.0,
                "recent_sets_played": float(total_sets)
            }

        p1_form_5 = get_player_form(p1_id, p1_matches, 5)
        p1_form_10 = get_player_form(p1_id, p1_matches, 10)
        p2_form_5 = get_player_form(p2_id, p2_matches, 5)
        p2_form_10 = get_player_form(p2_id, p2_matches, 10)

        def surface_win_rate(player_id: str, matches: List[Any], surf: str) -> float:
            surf_matches = [m for m in matches if (m.surface or "Hard").strip().capitalize() == surf]
            if not surf_matches:
                return 0.50
            wins = sum(1 for m in surf_matches if m.winner_id == player_id)
            return (wins + 1.0) / (len(surf_matches) + 2.0)

        p1_surf_wr = surface_win_rate(p1_id, p1_matches, surface)
        p2_surf_wr = surface_win_rate(p2_id, p2_matches, surface)

        def calc_rest(matches: List[Any]) -> float:
            if not matches:
                return 7.0
            last_date = matches[-1].event_date
            if last_date.tzinfo is None:
                last_date = last_date.replace(tzinfo=timezone.utc)
            delta = (as_of_timestamp - last_date).total_seconds() / 86400.0
            return max(1.0, min(30.0, delta))

        p1_rest = calc_rest(p1_matches)
        p2_rest = calc_rest(p2_matches)

        rank1 = 100.0
        rank2 = 100.0
        if match.tennis_stats:
            rank1 = float(match.tennis_stats.p1_rank or 100)
            rank2 = float(match.tennis_stats.p2_rank or 100)
        log_rank_diff = math.log(max(1.0, rank2)) - math.log(max(1.0, rank1))

        return {
            "p1_elo": p1_overall_elo,
            "p2_elo": p2_overall_elo,
            "elo_difference": p1_overall_elo - p2_overall_elo,
            "p1_surface_elo": p1_surf_elo,
            "p2_surface_elo": p2_surf_elo,
            "surface_elo_difference": p1_surf_elo - p2_surf_elo,
            "log_rank_difference": log_rank_diff,
            "p1_win_rate_last_5": p1_form_5["win_rate"],
            "p1_win_rate_last_10": p1_form_10["win_rate"],
            "p2_win_rate_last_5": p2_form_5["win_rate"],
            "p2_win_rate_last_10": p2_form_10["win_rate"],
            "p1_ewma_win_rate": p1_form_5["ewma_win"],
            "p2_ewma_win_rate": p2_form_5["ewma_win"],
            "p1_serve_dominance": p1_form_10["serve_dominance"],
            "p2_serve_dominance": p2_form_10["serve_dominance"],
            "p1_recent_sets": p1_form_5["recent_sets_played"],
            "p2_recent_sets": p2_form_5["recent_sets_played"],
            "p1_surface_win_rate": p1_surf_wr,
            "p2_surface_win_rate": p2_surf_wr,
            "surface_win_rate_diff": p1_surf_wr - p2_surf_wr,
            "p1_1st_serve_in": p1_form_10["first_serve_in"],
            "p1_1st_serve_won": p1_form_10["first_serve_won"],
            "p1_2nd_serve_won": p1_form_10["second_serve_won"],
            "p2_1st_serve_in": p2_form_10["first_serve_in"],
            "p2_1st_serve_won": p2_form_10["first_serve_won"],
            "p2_2nd_serve_won": p2_form_10["second_serve_won"],
            "p1_rest_days": p1_rest,
            "p2_rest_days": p2_rest,
            "rest_advantage": p1_rest - p2_rest
        }

    def extract_features(
        self,
        match: Any,
        historical_matches: List[Any],
        as_of_timestamp: datetime
    ) -> Dict[str, float]:
        self.check_temporal_integrity(historical_matches, as_of_timestamp)
        p1_id = match.participant_a_id
        p2_id = match.participant_b_id
        surface = (match.surface or "Hard").strip().capitalize()

        overall_elo: Dict[str, float] = {}
        surface_elo: Dict[str, float] = {}

        for h in historical_matches:
            w_id = h.winner_id
            if not w_id:
                continue
            l_id = h.participant_b_id if w_id == h.participant_a_id else h.participant_a_id

            rw = overall_elo.get(w_id, self.elo_system.initial_rating)
            rl = overall_elo.get(l_id, self.elo_system.initial_rating)
            w_sets = max(h.score_a or 0, h.score_b or 0)
            l_sets = min(h.score_a or 0, h.score_b or 0)
            new_rw, new_rl = self.elo_system.update_ratings(rw, rl, sets_w=w_sets, sets_l=l_sets)
            overall_elo[w_id] = new_rw
            overall_elo[l_id] = new_rl

            h_surface = (h.surface or "Hard").strip().capitalize()
            if h_surface == surface:
                rsw = surface_elo.get(w_id, self.elo_system.initial_rating)
                rsl = surface_elo.get(l_id, self.elo_system.initial_rating)
                new_rsw, new_rsl = self.elo_system.update_ratings(rsw, rsl, sets_w=w_sets, sets_l=l_sets)
                surface_elo[w_id] = new_rsw
                surface_elo[l_id] = new_rsl

        p1_m = [m for m in historical_matches if m.participant_a_id == p1_id or m.participant_b_id == p1_id]
        p2_m = [m for m in historical_matches if m.participant_a_id == p2_id or m.participant_b_id == p2_id]

        p1_overall = overall_elo.get(p1_id, self.elo_system.initial_rating)
        p2_overall = overall_elo.get(p2_id, self.elo_system.initial_rating)

        return self.extract_features_fast(
            match,
            p1_overall,
            p2_overall,
            surface_elo.get(p1_id, p1_overall),
            surface_elo.get(p2_id, p2_overall),
            p1_m,
            p2_m,
            as_of_timestamp
        )

