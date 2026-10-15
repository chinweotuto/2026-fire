"""Probability Calibration Engine (Platt Sigmoid, Isotonic Regression, Temperature Scaling, and Dirichlet Vector Scaling)."""

import numpy as np
from typing import Dict, Any, List, Optional
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from src.models.base import BaseSportModel


class TemperatureScalingCalibrator:
    """Optimizes a single temperature scalar T > 0 on validation log-odds: Softmax(z / T)."""

    def __init__(self):
        self.temperature: float = 1.0

    def fit(self, probs: np.ndarray, y: np.ndarray, classes: np.ndarray) -> "TemperatureScalingCalibrator":
        eps = 1e-7
        probs = np.clip(probs, eps, 1.0 - eps)
        logits = np.log(probs)

        # Convert y to class indices
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_indices = np.array([class_to_idx[val] for val in y])

        def loss_fn(T: np.ndarray) -> float:
            t_val = max(0.05, float(T[0]))
            scaled_logits = logits / t_val
            # Softmax
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            return float(log_loss(y_indices, softmax_probs, labels=list(range(len(classes)))))

        res = minimize(loss_fn, np.array([1.0]), method="Nelder-Mead", bounds=[(0.1, 10.0)])
        if res.success:
            self.temperature = max(0.05, float(res.x[0]))
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        eps = 1e-7
        probs = np.clip(probs, eps, 1.0 - eps)
        logits = np.log(probs)
        scaled_logits = logits / self.temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return softmax_probs


class VectorScalingCalibrator:
    """Multi-class Vector / Diagonal Dirichlet Scaling: Softmax(W * log(p) + b)."""

    def __init__(self):
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None

    def fit(self, probs: np.ndarray, y: np.ndarray, classes: np.ndarray) -> "VectorScalingCalibrator":
        eps = 1e-7
        probs = np.clip(probs, eps, 1.0 - eps)
        log_p = np.log(probs)
        n_classes = len(classes)

        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_indices = np.array([class_to_idx[val] for val in y])

        # Optimize diagonal weights and biases with L2 regularization
        def loss_fn(params: np.ndarray) -> float:
            w = params[:n_classes]
            b = params[n_classes:]
            scaled = log_p * w + b
            exp_s = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
            p_cal = exp_s / np.sum(exp_s, axis=1, keepdims=True)
            l2_reg = 0.001 * np.sum((w - 1.0) ** 2) + 0.001 * np.sum(b ** 2)
            return float(log_loss(y_indices, p_cal, labels=list(range(n_classes)))) + l2_reg

        init_params = np.concatenate([np.ones(n_classes), np.zeros(n_classes)])
        res = minimize(loss_fn, init_params, method="L-BFGS-B")
        if res.success:
            self.weights = res.x[:n_classes]
            self.bias = res.x[n_classes:]
        else:
            self.weights = np.ones(n_classes)
            self.bias = np.zeros(n_classes)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if self.weights is None:
            return probs
        eps = 1e-7
        probs = np.clip(probs, eps, 1.0 - eps)
        log_p = np.log(probs)
        scaled = log_p * self.weights + self.bias
        exp_s = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_s / np.sum(exp_s, axis=1, keepdims=True)


class ProbabilityCalibrator:
    """Calibrates predicted probabilities to match true empirical frequencies."""

    def __init__(self, method: str = "isotonic"):
        assert method in ("isotonic", "sigmoid", "temperature", "vector"), f"Unsupported method: {method}"
        self.method = method
        self.calibrators: List[Any] = []
        self.vector_calibrator: Optional[Any] = None
        self.classes_: np.ndarray = np.array([])

    def fit(self, probs: np.ndarray, y: np.ndarray, classes: np.ndarray) -> "ProbabilityCalibrator":
        """Fit calibrators on out-of-sample validation probabilities."""
        self.classes_ = classes
        n_classes = len(classes)
        self.calibrators = []

        if self.method == "temperature":
            self.vector_calibrator = TemperatureScalingCalibrator().fit(probs, y, classes)
        elif self.method == "vector":
            self.vector_calibrator = VectorScalingCalibrator().fit(probs, y, classes)
        elif n_classes == 2:
            # Binary calibration
            y_binary = (y == classes[1]).astype(float)
            p1 = probs[:, 1]
            if self.method == "isotonic":
                cal = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
                cal.fit(p1, y_binary)
            else:
                cal = LogisticRegression()
                cal.fit(p1.reshape(-1, 1), y_binary)
            self.calibrators.append(cal)
        else:
            # Multi-class One-vs-Rest calibration
            for i, c in enumerate(classes):
                y_c = (y == c).astype(float)
                p_c = probs[:, i]
                if self.method == "isotonic":
                    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
                    cal.fit(p_c, y_c)
                else:
                    cal = LogisticRegression()
                    cal.fit(p_c.reshape(-1, 1), y_c)
                self.calibrators.append(cal)

        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Apply calibration and project back onto the probability simplex."""
        if self.method in ("temperature", "vector") and self.vector_calibrator is not None:
            return self.vector_calibrator.transform(probs)

        n_samples, n_classes = probs.shape
        calibrated = np.zeros_like(probs)

        if n_classes == 2 and self.calibrators:
            cal = self.calibrators[0]
            p1 = probs[:, 1]
            if self.method == "isotonic":
                p1_cal = cal.predict(p1)
            else:
                p1_cal = cal.predict_proba(p1.reshape(-1, 1))[:, 1]
            p1_cal = np.clip(p1_cal, 0.001, 0.999)
            calibrated[:, 1] = p1_cal
            calibrated[:, 0] = 1.0 - p1_cal
        elif self.calibrators:
            for i, cal in enumerate(self.calibrators):
                p_c = probs[:, i]
                if self.method == "isotonic":
                    c_cal = cal.predict(p_c)
                else:
                    c_cal = cal.predict_proba(p_c.reshape(-1, 1))[:, 1]
                calibrated[:, i] = np.clip(c_cal, 0.001, 0.999)

            # Simplex projection (sum to 1.0)
            row_sums = calibrated.sum(axis=1, keepdims=True)
            calibrated = calibrated / np.where(row_sums == 0, 1.0, row_sums)
        else:
            return probs

        return calibrated


def calculate_expected_calibration_error(
    probs: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE) across probability confidence deciles."""
    confidences = np.max(probs, axis=1)
    predictions = classes[np.argmax(probs, axis=1)]
    accuracies = (predictions == y).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)

    return round(float(ece), 4)


class CalibratedModelWrapper(BaseSportModel):
    """Wraps any base model with an out-of-sample fitted probability calibrator."""

    def __init__(self, base_model: BaseSportModel, method: str = "isotonic"):
        super().__init__(name=f"{base_model.name}_calibrated_{method}")
        self.base_model = base_model
        self.method = method
        self.calibrator = ProbabilityCalibrator(method=method)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "CalibratedModelWrapper":
        self.base_model.fit(X, y, feature_names=feature_names)
        self.classes_ = self.base_model.classes_
        self.feature_names_ = self.base_model.feature_names_
        return self

    def fit_calibration(self, X_val: np.ndarray, y_val: np.ndarray) -> "CalibratedModelWrapper":
        """Fit calibration strictly on out-of-sample validation data."""
        self.classes_ = self.base_model.classes_
        self.feature_names_ = self.base_model.feature_names_
        raw_val_probs = self.base_model.predict_proba(X_val)
        self.calibrator.fit(raw_val_probs, y_val, self.classes_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw_probs = self.base_model.predict_proba(X)
        if self.calibrator.calibrators or self.calibrator.vector_calibrator is not None:
            return self.calibrator.transform(raw_probs)
        return raw_probs

