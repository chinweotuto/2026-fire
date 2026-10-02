"""Automation script: Out-of-sample model comparison and evaluation reporting."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.logging import logger
from src.db.session import SessionLocal
from src.markets.registry import market_registry
from src.evaluation.backtester import ChronologicalBacktester


def run_evaluation():
    db = SessionLocal()
    try:
        targets = [
            ("football", "football_1x2"),
            ("tennis", "tennis_match_winner"),
            ("basketball", "basketball_moneyline")
        ]

        print("\n" + "=" * 75)
        print("  CHRONOLOGICAL OUT-OF-SAMPLE MODEL COMPARISON REPORT")
        print("=" * 75 + "\n")

        for sport, market_id in targets:
            market = market_registry.get_market(market_id)
            if not market:
                continue

            backtester = ChronologicalBacktester(sport, market, db)
            res = backtester.run_backtest(train_fraction=0.70, val_fraction=0.15)

            if res["status"] != "SUCCESS":
                print(f"[{sport.upper()}] {market.display_name}: {res.get('message')}\n")
                continue

            print(f"--- {sport.upper()}: {market.display_name} ---")
            print(f"Training Window: {res['training_period']} (N={res['sample_counts']['train']})")
            print(f"Out-of-sample Test: {res['test_period']} (N={res['sample_counts']['test']})\n")

            header = f"{'Model':<32} {'Log Loss':<10} {'Brier':<10} {'Accuracy':<10} {'ECE':<8}"
            print(header)
            print("-" * len(header))

            for row in res["comparison"]:
                name = row["model_name"]
                ll = f"{row['log_loss']:.4f}"
                bs = f"{row['brier_score']:.4f}"
                acc = f"{row['accuracy'] * 100:.1f}%"
                ece = f"{row['ece']:.4f}"
                print(f"{name:<32} {ll:<10} {bs:<10} {acc:<10} {ece:<8}")
            print("\n")

    finally:
        db.close()


if __name__ == "__main__":
    run_evaluation()
