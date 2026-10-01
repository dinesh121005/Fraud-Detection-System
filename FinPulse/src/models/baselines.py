"""Baseline model implementations: Logistic Regression and Random Forest."""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Dict, Any, Optional
import numpy as np

class LogisticRegressionBaseline:
    """Standardized cost-sensitive linear baseline model."""

    def __init__(self, random_state: int = 42, C: float = 1.0):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=C,
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state
            ))
        ])

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

class RandomForestBaseline:
    """Non-linear bagging tree baseline model."""

    def __init__(self, random_state: int = 42, n_estimators: int = 200, max_depth: int = 12):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
