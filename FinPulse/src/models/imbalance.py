"""Imbalance handling: Cost-sensitive loss weights and resampling strategies."""
import numpy as np
from typing import Tuple, Optional

class ImbalanceHandler:
    """Manages class imbalance handling strategies for fraud detection."""

    @staticmethod
    def calculate_scale_pos_weight(y: np.ndarray) -> float:
        """
        Calculate standard scale_pos_weight for XGBoost and LightGBM:
        scale_pos_weight = N_negatives / N_positives
        """
        positives = np.sum(y == 1)
        negatives = np.sum(y == 0)
        return float(negatives / max(positives, 1))

    @staticmethod
    def random_undersample(
        X: np.ndarray,
        y: np.ndarray,
        ratio: float = 10.0,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Undersample the majority class to a controlled ratio (default 10:1 neg:pos).
        Applied ONLY on training sets.
        """
        rng = np.random.default_rng(random_state)
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]

        target_neg_count = min(len(neg_indices), int(len(pos_indices) * ratio))
        selected_neg_indices = rng.choice(neg_indices, size=target_neg_count, replace=False)

        combined_indices = np.concatenate([pos_indices, selected_neg_indices])
        rng.shuffle(combined_indices)

        return X[combined_indices], y[combined_indices]

    @staticmethod
    def apply_smote(
        X: np.ndarray,
        y: np.ndarray,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE synthetic minority over-sampling on training fold.
        Falls back to random sampling if imbalanced-learn is not installed.
        """
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=random_state)
            return smote.fit_resample(X, y)
        except ImportError:
            # Fallback interpolation if imblearn is absent
            pos_indices = np.where(y == 1)[0]
            if len(pos_indices) < 2:
                return X, y
            rng = np.random.default_rng(random_state)
            synthetic_count = min(len(pos_indices) * 2, 5000)
            idx1 = rng.choice(pos_indices, size=synthetic_count)
            idx2 = rng.choice(pos_indices, size=synthetic_count)
            lambdas = rng.uniform(0.1, 0.9, size=(synthetic_count, 1))
            X_synthetic = X[idx1] + lambdas * (X[idx2] - X[idx1])
            y_synthetic = np.ones(synthetic_count, dtype=int)
            return np.vstack([X, X_synthetic]), np.concatenate([y, y_synthetic])
