"""
FinPulse R7-H — Shadow / Challenger Model Scoring Engine.

Executes candidate/challenger models concurrently alongside the frozen production model:
- Production model remains 100% authoritative for the transaction decision
- Challenger model evaluates identical 32-feature vectors in non-blocking shadow mode
- Logs and records challenger risk score and prediction delta
- Challenger failures are fully isolated: never impact production decisions or latency
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from src.monitoring.logger import get_logger
from src.monitoring.metrics import record_error

logger = get_logger("FinPulse.Challenger.Shadow")


@dataclass(frozen=True)
class ShadowScoringComparison:
    """Side-by-side comparison of production vs challenger inference."""
    transaction_id: str
    production_model_version: str
    production_decision: str
    production_prob: float
    production_risk_score: float
    challenger_model_version: str
    challenger_prob: float
    challenger_risk_score: float
    decision_divergence: bool
    score_delta: float
    challenger_latency_ms: float
    timestamp: float = field(default_factory=time.time)


class ShadowModelRunner:
    """
    Manages non-blocking challenger model inference with fault isolation.
    """

    def __init__(
        self,
        challenger_model: Optional[Any] = None,
        challenger_calibrator: Optional[Any] = None,
        challenger_version: str = "finpulse-v4-challenger"
    ):
        self.challenger_model = challenger_model
        self.challenger_calibrator = challenger_calibrator
        self.challenger_version = challenger_version
        self.comparison_history: List[ShadowScoringComparison] = []

    def evaluate_shadow(
        self,
        transaction_id: str,
        feature_vector_32: np.ndarray,
        production_result: Dict[str, Any]
    ) -> Optional[ShadowScoringComparison]:
        """
        Evaluate challenger model on identical 32-feature vector.
        Guarantees that any exception in challenger is caught and isolated.
        """
        if self.challenger_model is None:
            return None

        t0 = time.perf_counter()
        try:
            # 1. Challenger inference
            if len(feature_vector_32.shape) == 1:
                vec = feature_vector_32.reshape(1, -1)
            else:
                vec = feature_vector_32

            raw_prob = float(self.challenger_model.predict_proba(vec)[0][1]) if hasattr(self.challenger_model, "predict_proba") else 0.0
            
            if self.challenger_calibrator:
                cal_prob = float(self.challenger_calibrator.predict(np.array([raw_prob]))[0])
            else:
                cal_prob = raw_prob

            # Approximate challenger risk score on same scale [0, 100]
            challenger_score = round(cal_prob * 100.0, 2)
            c_decision = "BLOCK" if challenger_score >= 70.0 else ("REVIEW" if challenger_score >= 30.0 else "APPROVE")
            
            c_latency = round((time.perf_counter() - t0) * 1000.0, 3)

            prod_dec = production_result.get("decision", "APPROVE")
            prod_score = float(production_result.get("risk_score", 0.0))
            prod_prob = float(production_result.get("calibrated_probability", 0.0))
            prod_version = str(production_result.get("model_version", "finpulse-v3"))

            comparison = ShadowScoringComparison(
                transaction_id=transaction_id,
                production_model_version=prod_version,
                production_decision=prod_dec,
                production_prob=prod_prob,
                production_risk_score=prod_score,
                challenger_model_version=self.challenger_version,
                challenger_prob=round(cal_prob, 4),
                challenger_risk_score=challenger_score,
                decision_divergence=(prod_dec != c_decision),
                score_delta=round(challenger_score - prod_score, 2),
                challenger_latency_ms=c_latency
            )

            self.comparison_history.append(comparison)
            return comparison

        except Exception as e:
            record_error("challenger", "inference_error")
            logger.warning("Challenger shadow scoring failed (production unaffected)", error=str(e), tx_id=transaction_id)
            return None

    def get_divergence_metrics(self) -> Dict[str, Any]:
        """Compute divergence statistics between production and challenger."""
        if not self.comparison_history:
            return {"total": 0, "divergence_rate": 0.0, "mean_abs_delta": 0.0}

        total = len(self.comparison_history)
        divergences = sum(1 for c in self.comparison_history if c.decision_divergence)
        mean_delta = float(np.mean([abs(c.score_delta) for c in self.comparison_history]))

        return {
            "total_evaluated": total,
            "divergent_decisions": divergences,
            "divergence_rate": round(divergences / total, 4),
            "mean_absolute_score_delta": round(mean_delta, 2),
            "challenger_version": self.challenger_version
        }
