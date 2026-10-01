"""Gradient Boosted Tree Candidates: XGBoost, LightGBM, and CatBoost."""
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from typing import Dict, Any, Optional
import numpy as np

class XGBoostModel:
    """Primary high-performance gradient boosted candidate (retains repo baseline continuity)."""

    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        scale_pos_weight: float = 1.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8
    ):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            tree_method="hist",
            eval_metric="logloss",
            random_state=random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

class LightGBMModel:
    """High-speed leaf-wise gradient boosting candidate."""

    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 300,
        num_leaves: int = 63,
        learning_rate: float = 0.05,
        scale_pos_weight: float = 1.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8
    ):
        self.model = LGBMClassifier(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            verbose=-1
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

class CatBoostModel:
    """Categorical-native ordered gradient boosting candidate."""

    def __init__(
        self,
        random_state: int = 42,
        iterations: int = 500,
        depth: int = 6,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,
        **kwargs
    ):
        self.model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            auto_class_weights="Balanced",
            verbose=0,
            random_seed=random_state,
            **kwargs
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
