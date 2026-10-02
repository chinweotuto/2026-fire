"""Tennis data ingestion from Sackmann / Kadantte ATP & WTA datasets."""

import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any
from src.core.logging import logger
from src.data.sources.base import BaseDataSource, download_file


def parse_tennis_score_sets(score_str: str) -> tuple[int, int]:
    """Parse a tennis score string (e.g., '6-4 3-6 7-6(5)') into (winner_sets, loser_sets)."""
    if not score_str or not isinstance(score_str, str):
        return 2, 0
    w_sets = 0
    l_sets = 0
    # Clean tiebreak numbers e.g. 7-6(5) -> 7-6
    clean_score = re.sub(r"\(\d+\)", "", score_str).strip()
    for set_str in clean_score.split():
        if "-" in set_str:
            parts = set_str.split("-")
            try:
                g_w = int(parts[0])
                g_l = int(parts[1])
                if g_w > g_l:
                    w_sets += 1
                elif g_l > g_w:
                    l_sets += 1
            except ValueError:
                continue
    if w_sets == 0 and l_sets == 0:
        return 2, 0
    return w_sets, l_sets


class TennisDataSource(BaseDataSource):
    """Ingests tour-level match records for professional tennis."""

    BASE_URL = "https://raw.githubusercontent.com/Kadantte/tennis_atp/master"
    YEARS = [2021, 2022, 2023]

    def download(self, target_dir: Path) -> List[Path]:
        downloaded = []
        target_dir = target_dir / "tennis"
        target_dir.mkdir(parents=True, exist_ok=True)

        for year in self.YEARS:
            url = f"{self.BASE_URL}/atp_matches_{year}.csv"
            dest = target_dir / f"atp_matches_{year}.csv"
            try:
                logger.info(f"Downloading ATP tennis data for year {year}...")
                path = download_file(url, dest)
                downloaded.append(path)
            except Exception as e:
                logger.warning(f"Could not download ATP matches for {year}: {e}")
        return downloaded

    def parse_matches(self, file_path: Path) -> List[Dict[str, Any]]:
        matches = []
        if not file_path.exists():
            return matches

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                winner_name = row.get("winner_name", "").strip()
                loser_name = row.get("loser_name", "").strip()
                tourney_date_str = row.get("tourney_date", "").strip()

                if not winner_name or not loser_name or not tourney_date_str:
                    continue

                try:
                    event_date = datetime.strptime(tourney_date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
                except ValueError:
                    continue

                surface = row.get("surface", "Hard").strip()
                tourney_id = row.get("tourney_id", "atp").strip()
                tourney_name = row.get("tourney_name", "Tour Event").strip()
                match_num = row.get("match_num", "1").strip()
                score_str = row.get("score", "").strip()

                w_sets, l_sets = parse_tennis_score_sets(score_str)

                # Order alphabetically or consistently: Participant A vs Participant B
                # In historical data, winner is Participant A or B
                # To ensure no label bias, let winner be Participant A with probability 0.5 or sorted by name
                if winner_name.lower() < loser_name.lower():
                    p1_name, p2_name = winner_name, loser_name
                    p1_sets, p2_sets = w_sets, l_sets
                    p1_is_winner = True
                    # Stats mapping
                    p1_prefix, p2_prefix = "w_", "l_"
                    p1_rank = row.get("winner_rank")
                    p2_rank = row.get("loser_rank")
                else:
                    p1_name, p2_name = loser_name, winner_name
                    p1_sets, p2_sets = l_sets, w_sets
                    p1_is_winner = False
                    p1_prefix, p2_prefix = "l_", "w_"
                    p1_rank = row.get("loser_rank")
                    p2_rank = row.get("winner_rank")

                def safe_int(key: str) -> Any:
                    val = row.get(key, "").strip()
                    try:
                        return int(float(val))
                    except (ValueError, TypeError):
                        return None

                match_id = f"tennis_{tourney_id}_{match_num}_{p1_name}_{p2_name}".lower().replace(" ", "_")

                match_record = {
                    "match_id": match_id,
                    "sport_id": "tennis",
                    "competition_id": f"comp_atp_{tourney_id}",
                    "competition_name": tourney_name,
                    "season": str(event_date.year),
                    "event_date": event_date,
                    "participant_a": p1_name,
                    "participant_b": p2_name,
                    "score_a": p1_sets,
                    "score_b": p2_sets,
                    "winner_name": winner_name,
                    "status": "COMPLETED",
                    "surface": surface,
                    "round": row.get("round", "R32").strip(),
                    "source": "kadantte/tennis_atp",
                    "source_id": f"{tourney_id}_{match_num}",
                    "tennis_stats": {
                        "surface": surface,
                        "tourney_level": row.get("tourney_level", "A").strip(),
                        "best_of": safe_int("best_of") or 3,
                        "minutes": safe_int("minutes"),
                        "p1_rank": safe_int(f"winner_rank" if p1_is_winner else "loser_rank"),
                        "p2_rank": safe_int(f"loser_rank" if p1_is_winner else "winner_rank"),
                        "p1_aces": safe_int(f"{p1_prefix}ace"),
                        "p1_double_faults": safe_int(f"{p1_prefix}df"),
                        "p1_serve_points": safe_int(f"{p1_prefix}svpt"),
                        "p1_first_in": safe_int(f"{p1_prefix}1stIn"),
                        "p1_first_won": safe_int(f"{p1_prefix}1stWon"),
                        "p1_second_won": safe_int(f"{p1_prefix}2ndWon"),
                        "p1_break_points_saved": safe_int(f"{p1_prefix}bpSaved"),
                        "p1_break_points_faced": safe_int(f"{p1_prefix}bpFaced"),
                        "p2_aces": safe_int(f"{p2_prefix}ace"),
                        "p2_double_faults": safe_int(f"{p2_prefix}df"),
                        "p2_serve_points": safe_int(f"{p2_prefix}svpt"),
                        "p2_first_in": safe_int(f"{p2_prefix}1stIn"),
                        "p2_first_won": safe_int(f"{p2_prefix}1stWon"),
                        "p2_second_won": safe_int(f"{p2_prefix}2ndWon"),
                        "p2_break_points_saved": safe_int(f"{p2_prefix}bpSaved"),
                        "p2_break_points_faced": safe_int(f"{p2_prefix}bpFaced"),
                    }
                }
                matches.append(match_record)

        return matches
