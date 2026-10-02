"""Basketball (NBA) box score data ingestion."""

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any
from src.core.logging import logger
from src.data.sources.base import BaseDataSource, download_file


class BasketballDataSource(BaseDataSource):
    """Ingests historical NBA team game box scores."""

    RAW_URL = "https://raw.githubusercontent.com/NocturneBear/NBA-Data-2010-2024/main/play_off_totals_2010_2024.csv"

    def download(self, target_dir: Path) -> List[Path]:
        target_dir = target_dir / "basketball"
        target_dir.mkdir(parents=True, exist_ok=True)
        dest = target_dir / "nba_playoff_totals.csv"
        try:
            logger.info("Downloading NBA box scores...")
            path = download_file(self.RAW_URL, dest)
            return [path]
        except Exception as e:
            logger.warning(f"Could not download NBA data: {e}")
            return []

    def parse_matches(self, file_path: Path) -> List[Dict[str, Any]]:
        if not file_path.exists():
            return []

        # Group box score rows by GAME_ID
        games_by_id: Dict[str, List[Dict[str, str]]] = {}
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                game_id = row.get("GAME_ID", "").strip()
                if game_id:
                    games_by_id.setdefault(game_id, []).append(row)

        matches = []
        for game_id, rows in games_by_id.items():
            if len(rows) != 2:
                continue

            # Identify home vs away
            # In NBA data, MATCHUP: "ATL @ CHI" means CHI is Home, ATL is Away.
            # "BOS vs. NYK" means BOS is Home, NYK is Away.
            row0, row1 = rows[0], rows[1]
            matchup0 = row0.get("MATCHUP", "")
            if "@" in matchup0:
                away_row, home_row = row0, row1
            else:
                home_row, away_row = row0, row1

            home_team = home_row.get("TEAM_NAME", "").strip()
            away_team = away_row.get("TEAM_NAME", "").strip()
            raw_date = home_row.get("GAME_DATE", "").strip()

            if not home_team or not away_team or not raw_date:
                continue

            try:
                # Format: 2011-05-10T00:00:00 or 2011-05-10
                date_clean = raw_date.split("T")[0]
                event_date = datetime.strptime(date_clean, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            def safe_int(r: Dict[str, str], k: str) -> Any:
                v = r.get(k, "").strip()
                try:
                    return int(float(v))
                except (ValueError, TypeError):
                    return None

            home_pts = safe_int(home_row, "PTS")
            away_pts = safe_int(away_row, "PTS")

            if home_pts is None or away_pts is None:
                continue

            season = home_row.get("SEASON_YEAR", "2023-24").strip()
            match_id = f"nba_{game_id}_{home_team}_{away_team}".lower().replace(" ", "_")

            match_record = {
                "match_id": match_id,
                "sport_id": "basketball",
                "competition_id": "comp_nba",
                "competition_name": "National Basketball Association",
                "season": season,
                "event_date": event_date,
                "participant_a": home_team,
                "participant_b": away_team,
                "score_a": home_pts,
                "score_b": away_pts,
                "winner_name": home_team if home_pts > away_pts else away_team,
                "status": "COMPLETED",
                "source": "nba_boxscores",
                "source_id": game_id,
                "basketball_stats": {
                    "home_fgm": safe_int(home_row, "FGM"),
                    "home_fga": safe_int(home_row, "FGA"),
                    "home_fg3m": safe_int(home_row, "FG3M"),
                    "home_fg3a": safe_int(home_row, "FG3A"),
                    "home_ftm": safe_int(home_row, "FTM"),
                    "home_fta": safe_int(home_row, "FTA"),
                    "home_oreb": safe_int(home_row, "OREB"),
                    "home_dreb": safe_int(home_row, "DREB"),
                    "home_reb": safe_int(home_row, "REB"),
                    "home_ast": safe_int(home_row, "AST"),
                    "home_tov": safe_int(home_row, "TOV"),
                    "home_stl": safe_int(home_row, "STL"),
                    "home_blk": safe_int(home_row, "BLK"),
                    "away_fgm": safe_int(away_row, "FGM"),
                    "away_fga": safe_int(away_row, "FGA"),
                    "away_fg3m": safe_int(away_row, "FG3M"),
                    "away_fg3a": safe_int(away_row, "FG3A"),
                    "away_ftm": safe_int(away_row, "FTM"),
                    "away_fta": safe_int(away_row, "FTA"),
                    "away_oreb": safe_int(away_row, "OREB"),
                    "away_dreb": safe_int(away_row, "DREB"),
                    "away_reb": safe_int(away_row, "REB"),
                    "away_ast": safe_int(away_row, "AST"),
                    "away_tov": safe_int(away_row, "TOV"),
                    "away_stl": safe_int(away_row, "STL"),
                    "away_blk": safe_int(away_row, "BLK"),
                }
            }
            matches.append(match_record)

        return matches
