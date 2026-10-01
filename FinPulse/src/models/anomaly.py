"""Unsupervised Anomaly Detection using Isolation Forest."""
from sklearn.ensemble import IsolationForest
import numpy as np

class IsolationForestAnomalyDetector:
    """
    Detects zero-day / outlier financial transaction behavior.
    Fitted exclusively on verified legitimate (non-fraud) transactions.
    Outputs normalized anomaly risk in range [0.0, 1.0].
    """

    def __init__(self, random_state: int = 42, n_estimators: int = 150, contamination: float = 0.01):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=256,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        self.offset_ = -0.5

    def fit(self, X_legit: np.ndarray):
        """Fit exclusively on legitimate (y=0) training transactions."""
        self.model.fit(X_legit)
        self.offset_ = self.model.offset_
        return self

    def score_anomaly(self, X: np.ndarray) -> np.ndarray:
        """
        Compute continuous anomaly risk score in [0.0, 1.0].
        Scores near 1.0 indicate severe anomalous behavioral patterns.
        """
        raw_scores = self.model.decision_function(X) # Positive = normal, Negative = anomaly
        # Sigmoid inversion: negative decision_function maps to values near 1.0
        normalized_anomaly = 1.0 / (1.0 + np.exp(10.0 * (raw_scores - self.offset_)))
        return np.clip(normalized_anomaly, 0.0, 1.0)
