"""Football data ingestion from Football-Data.co.uk."""

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any
from src.core.logging import logger
from src.data.sources.base import BaseDataSource, download_file


class FootballDataSource(BaseDataSource):
    """Ingests historical football league data from Football-Data.co.uk."""

    LEAGUES = {
        "E0": "English Premier League",
        "SP1": "Spanish La Liga",
        "D1": "German Bundesliga",
        "I1": "Italian Serie A"
    }

    # Season codes corresponding to 2021-2024
    SEASONS = ["2122", "2223", "2324"]

    def download(self, target_dir: Path) -> List[Path]:
        downloaded = []
        target_dir = target_dir / "football"
        target_dir.mkdir(parents=True, exist_ok=True)

        for season in self.SEASONS:
            for div in self.LEAGUES:
                url = f"https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
                dest = target_dir / f"{div}_{season}.csv"
                try:
                    logger.info(f"Downloading football data: {div} season {season}...")
                    path = download_file(url, dest)
                    downloaded.append(path)
                except Exception as e:
                    logger.warning(f"Could not download {url}: {e}")
        return downloaded

    def parse_matches(self, file_path: Path) -> List[Dict[str, Any]]:
        matches = []
        if not file_path.exists():
            return matches

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                home_team = row.get("HomeTeam", "").strip()
                away_team = row.get("AwayTeam", "").strip()
                raw_date = row.get("Date", "").strip()
                raw_fthg = row.get("FTHG", "").strip()
                raw_ftag = row.get("FTAG", "").strip()

                if not home_team or not away_team or not raw_date or not raw_fthg or not raw_ftag:
                    continue

                try:
                    fthg = int(raw_fthg)
                    ftag = int(raw_ftag)
                except ValueError:
                    continue

                # Parse date (format could be dd/mm/yy or dd/mm/yyyy)
                event_date = None
                for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
                    try:
                        event_date = datetime.strptime(raw_date, fmt).replace(tzinfo=timezone.utc)
                        break
                    except ValueError:
                        continue

                if not event_date:
                    continue

                div = row.get("Div", "FOOTBALL").strip()
                comp_name = self.LEAGUES.get(div, div)
                match_id = f"fb_{div}_{event_date.strftime('%Y%m%d')}_{home_team}_{away_team}".lower().replace(" ", "_")

                def safe_int(key: str) -> Any:
                    val = row.get(key, "").strip()
                    try:
                        return int(val)
                    except ValueError:
                        return None

                match_record = {
                    "match_id": match_id,
                    "sport_id": "football",
                    "competition_id": f"comp_{div.lower()}",
                    "competition_name": comp_name,
                    "season": file_path.stem.split("_")[-1] if "_" in file_path.stem else "unknown",
                    "event_date": event_date,
                    "participant_a": home_team,
                    "participant_b": away_team,
                    "score_a": fthg,
                    "score_b": ftag,
                    "status": "COMPLETED",
                    "source": "football-data.co.uk",
                    "source_id": row.get("Div", "E0"),
                    "football_stats": {
                        "ht_home_goals": safe_int("HTHG"),
                        "ht_away_goals": safe_int("HTAG"),
                        "home_shots": safe_int("HS"),
                        "away_shots": safe_int("AS"),
                        "home_shots_on_target": safe_int("HST"),
                        "away_shots_on_target": safe_int("AST"),
                        "home_corners": safe_int("HC"),
                        "away_corners": safe_int("AC"),
                        "home_fouls": safe_int("HF"),
                        "away_fouls": safe_int("AF"),
                        "home_yellow_cards": safe_int("HY"),
                        "away_yellow_cards": safe_int("AY"),
                        "home_red_cards": safe_int("HR"),
                        "away_red_cards": safe_int("AR"),
                        "home_xg": None,  # Strictly None when absent
                        "away_xg": None
                    }
                }
                matches.append(match_record)

        return matches
