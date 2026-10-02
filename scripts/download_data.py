"""Automation script: Download verified open historical sports datasets."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import settings
from src.core.logging import logger
from src.data.sources.football import FootballDataSource
from src.data.sources.tennis import TennisDataSource
from src.data.sources.basketball import BasketballDataSource


def main():
    logger.info("Starting historical data download for Football, Basketball, and Tennis...")
    raw_dir = settings.DATA_RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        ("Football", FootballDataSource()),
        ("Tennis", TennisDataSource()),
        ("Basketball", BasketballDataSource())
    ]

    total_files = 0
    for sport_name, source in sources:
        logger.info(f"--- Fetching {sport_name} datasets ---")
        try:
            files = source.download(raw_dir)
            total_files += len(files)
            logger.info(f"Successfully downloaded {len(files)} files for {sport_name}")
        except Exception as e:
            logger.error(f"Error fetching {sport_name}: {e}")

    logger.info(f"Data download finished. Total files: {total_files}")
    return 0 if total_files > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
