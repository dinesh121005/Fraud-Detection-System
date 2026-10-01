"""Probability Calibration for Gradient Boosted Classifiers."""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
import numpy as np
from typing import Literal

class ProbabilityCalibrator:
    """
    Calibrates raw tree model probability scores against validation ground truth.
    Supports 'platt' (Sigmoid) scaling and 'isotonic' regression.
    """

    def __init__(self, method: Literal["platt", "isotonic"] = "isotonic"):
        self.method = method
        self.calibrator = None

    def fit(self, raw_probs: np.ndarray, y_val: np.ndarray):
        """Fit calibration mapping on validation set."""
        raw_probs = np.clip(raw_probs, 1e-6, 1.0 - 1e-6)
        
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator.fit(raw_probs, y_val)
        else:
            # Platt scaling: Logistic regression on logit
            logits = np.log(raw_probs / (1.0 - raw_probs)).reshape(-1, 1)
            self.calibrator = LogisticRegression(solver="lbfgs", C=1.0)
            self.calibrator.fit(logits, y_val)

        return self

    def predict(self, raw_probs: np.ndarray) -> np.ndarray:
        """Transform raw probabilities into well-calibrated posterior probabilities."""
        raw_probs = np.clip(raw_probs, 1e-6, 1.0 - 1e-6)
        if self.calibrator is None:
            return raw_probs

        if self.method == "isotonic":
            calibrated = self.calibrator.predict(raw_probs)
        else:
            logits = np.log(raw_probs / (1.0 - raw_probs)).reshape(-1, 1)
            calibrated = self.calibrator.predict_proba(logits)[:, 1]

        return np.clip(calibrated, 0.0, 1.0)

    @staticmethod
    def evaluate_calibration(raw_probs: np.ndarray, calibrated_probs: np.ndarray, y_val: np.ndarray) -> dict:
        """Calculate Brier score reduction."""
        raw_brier = brier_score_loss(y_val, raw_probs)
        cal_brier = brier_score_loss(y_val, calibrated_probs)
        return {
            "raw_brier_score": round(float(raw_brier), 5),
            "calibrated_brier_score": round(float(cal_brier), 5),
            "improvement": round(float(raw_brier - cal_brier), 5)
        }
