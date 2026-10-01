"""FinPulse Hybrid Risk Engine combining ML, Anomaly, Velocity, Behavioral, and Rule signals."""
from typing import Dict, Any, List
import numpy as np
from .rules import DeterministicRuleEngine

class FinPulseRiskEngine:
    """
    Combines probabilistic ML outputs with deterministic business rules,
    sliding-window velocities, and unsupervised anomaly detection scores.
    """

    def __init__(self, config: Dict[str, Any] = None):
        cfg = config or {}
        weights = cfg.get("weights", {})
        self.w_ml = weights.get("w_ml", 0.45)
        self.w_anomaly = weights.get("w_anomaly", 0.15)
        self.w_velocity = weights.get("w_velocity", 0.15)
        self.w_behavioral = weights.get("w_behavioral", 0.15)
        self.w_rules = weights.get("w_rules", 0.10)

        thresholds = cfg.get("thresholds", {})
        self.tau_review = thresholds.get("tau_review", 30.0)
        self.tau_block = thresholds.get("tau_block", 70.0)

        self.rule_engine = DeterministicRuleEngine()

    def evaluate_risk(
        self,
        tx: Dict[str, Any],
        features_dict: Dict[str, Any],
        ml_prob: float,
        anomaly_score: float,
        reasons: List[str]
    ) -> Dict[str, Any]:
        """
        Produce unified risk decision and explanation payload.
        """
        # 1. Evaluate deterministic rules
        rule_risk, rule_violations, hard_block = self.rule_engine.evaluate_rules(tx, features_dict)

        # 2. Derive velocity risk component
        tx_count_5m = float(features_dict.get("tx_count_5m", 1))
        tx_count_1h = float(features_dict.get("tx_count_1h", 1))
        velocity_risk = min(1.0, (tx_count_5m / 5.0) * 0.6 + (tx_count_1h / 15.0) * 0.4)

        # 3. Derive behavioral risk component
        zscore = abs(float(features_dict.get("amount_zscore", 0.0)))
        is_new_device = float(features_dict.get("is_new_device", 0))
        is_new_loc = float(features_dict.get("is_new_location", 0))
        behavioral_risk = min(1.0, (zscore / 4.0) * 0.5 + is_new_device * 0.3 + is_new_loc * 0.2)

        # 4. Weighted Linear Aggregation with Multi-Signal Compounding
        combined_score = (
            self.w_ml * ml_prob +
            self.w_anomaly * anomaly_score +
            self.w_velocity * velocity_risk +
            self.w_behavioral * behavioral_risk +
            self.w_rules * rule_risk
        )

        # Multi-signal synergy escalation: compound risk if multiple dimensions fire
        compounding_triggers = int(velocity_risk > 0.3) + int(behavioral_risk > 0.3) + int(rule_risk > 0.4) + int(ml_prob > 0.4)
        if compounding_triggers >= 2:
            combined_score = min(1.0, combined_score + (compounding_triggers * 0.12))

        final_risk = float(round(combined_score * 100.0, 1))

        # Check hard overrides
        if hard_block:
            final_risk = max(final_risk, 85.0)
            decision = "BLOCK"
            risk_level = "HIGH"
        elif final_risk >= self.tau_block:
            decision = "BLOCK"
            risk_level = "HIGH"
        elif final_risk >= self.tau_review:
            decision = "REVIEW"
            risk_level = "MEDIUM"
        else:
            decision = "APPROVE"
            risk_level = "LOW"

        # Combine model SHAP reasons with rule violations
        all_reasons = rule_violations + reasons
        all_reasons = list(dict.fromkeys(all_reasons))[:3]

        return {
            "transaction_id": tx.get("transaction_id", "unknown_tx"),
            "fraud_probability": round(float(ml_prob), 4),
            "risk_score": final_risk,
            "risk_level": risk_level,
            "decision": decision,
            "model_version": "finpulse-v2.0",
            "signals": {
                "ml_probability": round(float(ml_prob), 4),
                "anomaly_score": round(float(anomaly_score), 4),
                "velocity_risk": round(float(velocity_risk), 4),
                "behavioral_risk": round(float(behavioral_risk), 4),
                "rule_risk": round(float(rule_risk), 4)
            },
            "top_reasons": all_reasons
        }
