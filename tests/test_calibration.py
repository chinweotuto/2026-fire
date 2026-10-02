"""Unit tests for probability calibration and ECE calculation."""

import numpy as np
import pytest
from src.models.calibration import ProbabilityCalibrator, calculate_expected_calibration_error


def test_probability_simplex_sum():
    cal = ProbabilityCalibrator(method="isotonic")
    # 3-class synthetic setup
    classes = np.array(["HOME", "DRAW", "AWAY"])
    probs_val = np.array([
        [0.6, 0.2, 0.2],
        [0.3, 0.4, 0.3],
        [0.2, 0.2, 0.6],
        [0.5, 0.3, 0.2],
        [0.1, 0.2, 0.7]
    ])
    y_val = np.array(["HOME", "DRAW", "AWAY", "HOME", "AWAY"])

    cal.fit(probs_val, y_val, classes)

    # Test transformation
    test_probs = np.array([
        [0.55, 0.25, 0.20],
        [0.15, 0.20, 0.65]
    ])
    calibrated = cal.transform(test_probs)

    assert calibrated.shape == test_probs.shape
    # All rows must sum to 1.0 (simplex projection)
    assert np.allclose(calibrated.sum(axis=1), [1.0, 1.0], atol=1e-5)
    assert np.all(calibrated >= 0.0)


def test_expected_calibration_error():
    classes = np.array(["0", "1"])
    probs = np.array([
        [0.2, 0.8],
        [0.1, 0.9],
        [0.3, 0.7],
        [0.8, 0.2]
    ])
    y = np.array(["1", "1", "1", "0"])
    ece = calculate_expected_calibration_error(probs, y, classes, n_bins=5)
    assert 0.0 <= ece <= 1.0
