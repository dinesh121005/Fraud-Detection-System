"""Dual-threshold optimizer for business decision tiers: APPROVE / REVIEW / BLOCK."""
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
from typing import Dict, Any, Tuple

class ThresholdOptimizer:
    """
    Optimizes operational decision cutoffs on validation probability distributions.
    Enforces dual thresholds:
    - tau_review: Bound FPR <= max_review_fpr (e.g. 1.0%) to control analyst workload
    - tau_block: Bound Precision >= min_block_precision (e.g. 85%) to avoid declining valid cardholders
    """

    def __init__(self, target_fpr: float = 0.01, min_block_precision: float = 0.85):
        self.target_fpr = target_fpr
        self.min_block_precision = min_block_precision
        self.tau_review = 0.30
        self.tau_block = 0.70

    def optimize_thresholds(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """Derive tau_review and tau_block from validation probabilities."""
        # 1. Determine tau_review using ROC curve to keep FPR <= target_fpr
        fpr_pts, tpr_pts, roc_thresh = roc_curve(y_true, y_proba)
        valid_fpr_indices = np.where(fpr_pts <= self.target_fpr)[0]
        if len(valid_fpr_indices) > 0:
            idx = valid_fpr_indices[-1]
            self.tau_review = float(np.clip(roc_thresh[idx], 0.05, 0.45))
        else:
            self.tau_review = 0.20

        # 2. Determine tau_block using PR curve targeting min_block_precision
        prec_pts, recall_pts, pr_thresh = precision_recall_curve(y_true, y_proba)
        valid_prec_indices = np.where(prec_pts[:-1] >= self.min_block_precision)[0]
        if len(valid_prec_indices) > 0:
            idx = valid_prec_indices[0] # lowest threshold meeting precision
            self.tau_block = float(np.clip(pr_thresh[idx], self.tau_review + 0.1, 0.90))
        else:
            self.tau_block = 0.70

        # Operational metrics at these thresholds
        n = len(y_true)
        decisions = self.apply_decision_policy(y_proba)
        
        approved = int(np.sum(decisions == "APPROVE"))
        reviewed = int(np.sum(decisions == "REVIEW"))
        blocked = int(np.sum(decisions == "BLOCK"))

        return {
            "tau_review": round(self.tau_review, 4),
            "tau_block": round(self.tau_block, 4),
            "distribution": {
                "approved_count": approved,
                "approved_pct": round((approved / n) * 100, 2),
                "reviewed_count": reviewed,
                "reviewed_pct": round((reviewed / n) * 100, 2),
                "blocked_count": blocked,
                "blocked_pct": round((blocked / n) * 100, 2)
            }
        }

    def apply_decision_policy(self, y_proba: np.ndarray) -> np.ndarray:
        """Vectorized decision assignment: APPROVE, REVIEW, or BLOCK."""
        conditions = [
            y_proba < self.tau_review,
            (y_proba >= self.tau_review) & (y_proba < self.tau_block),
            y_proba >= self.tau_block
        ]
        choices = ["APPROVE", "REVIEW", "BLOCK"]
        return np.select(conditions, choices, default="REVIEW")
