"""Automation script: Model Governance and Promotion Gatekeeper."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime, timezone
from sqlalchemy.orm import Session
from src.core.logging import logger
from src.db.session import SessionLocal
from src.db.models import ModelRecord, ModelVersion


def evaluate_and_promote_version(db: Session, model_id: str, candidate_version_id: str) -> bool:
    """Evaluate candidate model against current champion."""
    candidate = db.query(ModelVersion).filter(ModelVersion.version_id == candidate_version_id).first()
    if not candidate:
        logger.error(f"Candidate {candidate_version_id} not found")
        return False

    champion = (
        db.query(ModelVersion)
        .filter(ModelVersion.model_id == model_id, ModelVersion.is_active == True)
        .first()
    )

    if not champion:
        logger.info(f"No existing champion found for {model_id}. Automatically promoting candidate.")
        candidate.is_active = True
        candidate.promoted_at = datetime.now(timezone.utc)
        db.commit()
        return True

    # Promotion Rule
    cand_ll = candidate.validation_log_loss or 999.0
    champ_ll = champion.validation_log_loss or 999.0
    cand_brier = candidate.validation_brier_score or 999.0
    champ_brier = champion.validation_brier_score or 999.0

    logger.info(f"Comparing Candidate (LL={cand_ll:.4f}, BS={cand_brier:.4f}) vs Champion (LL={champ_ll:.4f}, BS={champ_brier:.4f})")

    if cand_ll < (champ_ll - 0.005) and cand_brier <= champ_brier:
        logger.info(f"PROMOTION APPROVED: Candidate {candidate.version_tag} is superior. Promoting to Active.")
        champion.is_active = False
        candidate.is_active = True
        candidate.promoted_at = datetime.now(timezone.utc)
        db.commit()
        return True
    else:
        logger.info(f"PROMOTION REJECTED: Candidate {candidate.version_tag} did not meet required statistical lift threshold.")
        return False


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/promote_model.py <model_id> <candidate_version_id>")
        sys.exit(1)

    m_id = sys.argv[1]
    c_ver_id = sys.argv[2]
    db = SessionLocal()
    try:
        evaluate_and_promote_version(db, m_id, c_ver_id)
    finally:
        db.close()


if __name__ == "__main__":
    main()
