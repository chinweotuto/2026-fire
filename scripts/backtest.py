"""Automation script: Rolling walk-forward backtesting CLI."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.logging import logger
from src.db.session import SessionLocal
from src.markets.registry import market_registry
from src.evaluation.backtester import ChronologicalBacktester


def main():
    logger.info("Starting chronological walk-forward backtesting...")
    db = SessionLocal()
    try:
        targets = [
            ("football", "football_1x2"),
            ("football", "football_over_under_2_5"),
            ("tennis", "tennis_match_winner"),
            ("basketball", "basketball_moneyline")
        ]

        for sport, m_id in targets:
            market = market_registry.get_market(m_id)
            if not market:
                continue

            bt = ChronologicalBacktester(sport, market, db)
            res = bt.run_backtest(train_fraction=0.70, val_fraction=0.15)
            if res["status"] == "SUCCESS":
                best_model = res["comparison"][0]
                logger.info(
                    f"[{sport.upper()}] {market.display_name} -> "
                    f"Best Model: {best_model['model_name']} "
                    f"(Log Loss: {best_model['log_loss']}, Brier: {best_model['brier_score']}, "
                    f"Acc: {best_model['accuracy'] * 100:.1f}%)"
                )
            else:
                logger.warning(f"[{sport}] {m_id}: {res.get('message')}")

        logger.info("Backtesting run complete.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
