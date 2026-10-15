"""Basketball (NBA) Pre-Match Feature Engine with EWMA and Four Factors."""

from datetime import datetime, timezone
from typing import Dict, List, Any
from src.features.base import BaseFeatureEngine
from src.features.elo import BasketballEloSystem


class BasketballFeatureEngine(BaseFeatureEngine):
    """Generates 100% pre-match causal features for basketball games with EWMA and Four Factors."""

    def __init__(self, version: str = "v1.1.0"):
        super().__init__(sport_id="basketball", version=version)
        self.elo_system = BasketballEloSystem()

    def extract_features_fast(
        self,
        match: Any,
        home_elo: float,
        away_elo: float,
        home_matches: List[Any],
        away_matches: List[Any],
        as_of_timestamp: datetime
    ) -> Dict[str, float]:
        home_id = match.participant_a_id
        away_id = match.participant_b_id
        elo_diff = (home_elo + self.elo_system.home_advantage) - away_elo

        def get_team_advanced_stats(team_id: str, matches: List[Any], window: int) -> Dict[str, float]:
            recent = matches[-window:] if len(matches) >= window else matches
            if not recent:
                return {
                    "win_rate": 0.5, "pts_per_game": 110.0, "opp_pts": 110.0,
                    "efg_pct": 0.53, "tov_rate": 0.13, "pace": 98.0,
                    "ortg": 112.0, "drtg": 112.0, "net_rtg": 0.0,
                    "ewma_pts": 110.0, "ewma_opp_pts": 110.0, "orb_pct": 0.25, "ft_rate": 0.20
                }

            wins, total_pts, opp_pts = 0, 0, 0
            fgm, fga, fg3m, ftm, fta, tov, oreb, dreb = 0, 0, 0, 0, 0, 0, 0, 0
            ewma_pts, ewma_opp_pts, ewma_weight_total = 0.0, 0.0, 0.0
            n = len(recent)

            for i, m in enumerate(recent):
                is_home = (m.participant_a_id == team_id)
                t_score = m.score_a if is_home else m.score_b
                o_score = m.score_b if is_home else m.score_a
                t_score = t_score if t_score is not None else 110
                o_score = o_score if o_score is not None else 110

                total_pts += t_score
                opp_pts += o_score
                if t_score > o_score:
                    wins += 1

                w = 0.85 ** (n - 1 - i)
                ewma_pts += w * t_score
                ewma_opp_pts += w * o_score
                ewma_weight_total += w

                if m.basketball_stats:
                    st = m.basketball_stats
                    fgm += (st.home_fgm if is_home else st.away_fgm) or 40
                    fga += (st.home_fga if is_home else st.away_fga) or 88
                    fg3m += (st.home_fg3m if is_home else st.away_fg3m) or 12
                    ftm += (st.home_ftm if is_home else st.away_ftm) or 18
                    fta += (st.home_fta if is_home else st.away_fta) or 23
                    tov += (st.home_tov if is_home else st.away_tov) or 14
                    oreb += (st.home_oreb if is_home else st.away_oreb) or 10
                    dreb += (st.home_dreb if is_home else st.away_dreb) or 34
                else:
                    fgm += 40
                    fga += 88
                    fg3m += 12
                    ftm += 18
                    fta += 23
                    tov += 14
                    oreb += 10
                    dreb += 34

            fga_safe = max(1, fga)
            possessions = fga + 0.44 * fta - oreb + tov
            poss_safe = max(1.0, possessions)
            ewma_denom = max(1e-5, ewma_weight_total)

            efg = (fgm + 0.5 * fg3m) / fga_safe
            tov_rate = tov / max(1.0, fga + 0.44 * fta + tov)
            pace = 48.0 * (possessions / (n * 48.0))
            ortg = 100.0 * (total_pts / poss_safe)
            drtg = 100.0 * (opp_pts / poss_safe)
            orb_pct = oreb / max(1.0, oreb + dreb)
            ft_rate = ftm / fga_safe

            return {
                "win_rate": wins / n,
                "pts_per_game": total_pts / n,
                "opp_pts": opp_pts / n,
                "efg_pct": efg,
                "tov_rate": tov_rate,
                "pace": pace,
                "ortg": ortg,
                "drtg": drtg,
                "net_rtg": ortg - drtg,
                "ewma_pts": ewma_pts / ewma_denom,
                "ewma_opp_pts": ewma_opp_pts / ewma_denom,
                "orb_pct": orb_pct,
                "ft_rate": ft_rate
            }

        h_stat = get_team_advanced_stats(home_id, home_matches, 10)
        a_stat = get_team_advanced_stats(away_id, away_matches, 10)

        def get_rest_info(matches: List[Any]) -> tuple[float, float]:
            if not matches:
                return 3.0, 0.0
            last_date = matches[-1].event_date
            if last_date.tzinfo is None:
                last_date = last_date.replace(tzinfo=timezone.utc)
            delta = (as_of_timestamp - last_date).total_seconds() / 86400.0
            rest = max(1.0, min(14.0, delta))
            is_b2b = 1.0 if delta <= 1.5 else 0.0
            return rest, is_b2b

        h_rest, h_b2b = get_rest_info(home_matches)
        a_rest, a_b2b = get_rest_info(away_matches)

        return {
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_difference": elo_diff,
            "home_win_rate_last_10": h_stat["win_rate"],
            "away_win_rate_last_10": a_stat["win_rate"],
            "home_pts_per_game": h_stat["pts_per_game"],
            "away_pts_per_game": a_stat["pts_per_game"],
            "home_opp_pts": h_stat["opp_pts"],
            "away_opp_pts": a_stat["opp_pts"],
            "home_ewma_pts": h_stat["ewma_pts"],
            "away_ewma_pts": a_stat["ewma_pts"],
            "home_efg_pct": h_stat["efg_pct"],
            "away_efg_pct": a_stat["efg_pct"],
            "home_tov_rate": h_stat["tov_rate"],
            "away_tov_rate": a_stat["tov_rate"],
            "home_orb_pct": h_stat["orb_pct"],
            "away_orb_pct": a_stat["orb_pct"],
            "home_ft_rate": h_stat["ft_rate"],
            "away_ft_rate": a_stat["ft_rate"],
            "home_ortg": h_stat["ortg"],
            "away_ortg": a_stat["ortg"],
            "home_drtg": h_stat["drtg"],
            "away_drtg": a_stat["drtg"],
            "home_net_rtg": h_stat["net_rtg"],
            "away_net_rtg": a_stat["net_rtg"],
            "expected_pace": (h_stat["pace"] + a_stat["pace"]) / 2.0,
            "home_rest_days": h_rest,
            "away_rest_days": a_rest,
            "home_is_back_to_back": h_b2b,
            "away_is_back_to_back": a_b2b,
            "rest_advantage": h_rest - a_rest
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
            h_score = h.score_a if h.score_a is not None else 100
            a_score = h.score_b if h.score_b is not None else 100
            new_h, new_a = self.elo_system.update_ratings(r_home, r_away, h_score, a_score)
            ratings[h.participant_a_id] = new_h
            ratings[h.participant_b_id] = new_a

        h_matches = [m for m in historical_matches if m.participant_a_id == home_id or m.participant_b_id == home_id]
        a_matches = [m for m in historical_matches if m.participant_a_id == away_id or m.participant_b_id == away_id]

        return self.extract_features_fast(
            match,
            ratings.get(home_id, self.elo_system.initial_rating),
            ratings.get(away_id, self.elo_system.initial_rating),
            h_matches,
            a_matches,
            as_of_timestamp
        )

