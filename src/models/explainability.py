"""Model Explainability and Local Feature Attribution."""

import numpy as np
from typing import Dict, Any, List, Optional
from src.models.base import BaseSportModel


def extract_top_contributing_features(
    model: BaseSportModel,
    feature_names: List[str],
    sample_vector: np.ndarray,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Identify the top statistical features contributing to a prediction.
    
    NOTE: Attributions represent statistical associations, NOT physical causation.
    """
    if not feature_names:
        return []

    # If the model has internal feature importances (e.g. Random Forest, Gradient Boosting)
    importances = None
    target_obj = model
    if hasattr(model, "base_model"):
        target_obj = model.base_model

    if hasattr(target_obj, "clf") and hasattr(target_obj.clf, "feature_importances_"):
        importances = target_obj.clf.feature_importances_
    elif hasattr(target_obj, "clf") and hasattr(target_obj.clf, "coef_"):
        coef = target_obj.clf.coef_
        importances = np.mean(np.abs(coef), axis=0)
    else:
        # Fallback to normalized feature magnitudes
        importances = np.abs(sample_vector[0])
        importances = importances / max(1e-6, np.sum(importances))

    if importances is None or len(importances) != len(feature_names):
        sorted_indices = np.argsort(np.abs(sample_vector[0]))[::-1]
    else:
        sorted_indices = np.argsort(importances)[::-1]

    top_features = []
    for idx in sorted_indices[:top_k]:
        name = feature_names[idx]
        val = float(sample_vector[0][idx])
        weight = float(importances[idx]) if importances is not None else 0.0

        readable_name = name.replace("_", " ").title()
        top_features.append({
            "feature_key": name,
            "display_name": readable_name,
            "value": round(val, 3),
            "relative_importance": round(weight, 3)
        })

    return top_features
