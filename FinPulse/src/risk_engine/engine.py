"""FinPulse Hybrid Risk Engine combining ML, Anomaly, Velocity, Behavioral, and Rule signals."""
from typing import Dict, Any, List, Optional
import numpy as np

from .rules import DeterministicRuleEngine
from .contract import (
    HybridRiskWeights,
    HybridRiskThresholds,
    R4MLThresholds,
    CompoundingPolicy,
    NormalizedRiskSignals,
    validate_calibrated_probability,
    normalize_velocity_signal,
    normalize_behavioral_signal,
    normalize_rules_signal,
    normalize_anomaly_signal,
    ContractViolationError,
)
from .hybrid import HybridRiskEngine, HybridRiskResult

class FinPulseRiskEngine:
    """
    R5 Hybrid Risk Engine:
    Combines calibrated ML probability with deterministic business rules,
    sliding-window velocities, customer behavioral deviation, and unsupervised anomaly scores.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        
        # 1. Weights Configuration (Default: 45% ML, 15% Vel, 15% Beh, 15% Rules, 10% Anomaly)
        weights_cfg = cfg.get("weights", {})
        self.weights = HybridRiskWeights(
            w_ml=float(weights_cfg.get("w_ml", 0.45)),
            w_velocity=float(weights_cfg.get("w_velocity", 0.15)),
            w_behavioral=float(weights_cfg.get("w_behavioral", 0.15)),
            w_rules=float(weights_cfg.get("w_rules", 0.15)),
            w_anomaly=float(weights_cfg.get("w_anomaly", 0.10))
        )
        # Retain direct attribute access for backward compatibility
        self.w_ml = self.weights.w_ml
        self.w_velocity = self.weights.w_velocity
        self.w_behavioral = self.weights.w_behavioral
        self.w_rules = self.weights.w_rules
        self.w_anomaly = self.weights.w_anomaly

        # 2. Risk Decision Thresholds (Domain: [0.0, 100.0])
        thresh_cfg = cfg.get("thresholds", {})
        self.thresholds = HybridRiskThresholds(
            tau_review=float(thresh_cfg.get("tau_review", 30.0)),
            tau_block=float(thresh_cfg.get("tau_block", 70.0))
        )
        self.tau_review = self.thresholds.tau_review
        self.tau_block = self.thresholds.tau_block

        # 3. Independent R4 ML Threshold Policy for Cross-Layer Verification
        self.ml_thresholds = R4MLThresholds()

        # 4. Explicit Multi-Signal Compounding Policy
        comp_cfg = cfg.get("compounding", {})
        self.compounding_policy = CompoundingPolicy(
            velocity_threshold=float(comp_cfg.get("velocity_threshold", 0.30)),
            behavioral_threshold=float(comp_cfg.get("behavioral_threshold", 0.30)),
            rules_threshold=float(comp_cfg.get("rules_threshold", 0.40)),
            ml_threshold=float(comp_cfg.get("ml_threshold", 0.40)),
            min_triggers=int(comp_cfg.get("min_triggers", 2)),
            boost_per_trigger=float(comp_cfg.get("boost_per_trigger", 0.12))
        )

        # 5. Production Model Version
        self.model_version = str(cfg.get("model_version", "finpulse-v3"))

        # 6. Deterministic Rule Engine
        self.rule_engine = DeterministicRuleEngine()

    def evaluate_risk(
        self,
        tx: Dict[str, Any],
        features_dict: Dict[str, Any],
        calibrated_probability: Optional[float] = None,
        anomaly_score: Optional[float] = None,
        reasons: Optional[List[str]] = None,
        # Backward-compatibility parameters
        ml_prob: Optional[float] = None,
        anomaly_signal: Optional[float] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Produce unified hybrid risk decision and explanation payload.

        Parameters:
        - tx: Raw transaction attributes dictionary.
        - features_dict: Pre-computed 32-feature vector dictionary.
        - calibrated_probability: Canonical R4 Platt-calibrated ML probability in [0.0, 1.0].
        - anomaly_score: Anomaly detection score in [0.0, 1.0].
        - reasons: SHAP feature attribution top reasons.
        - ml_prob: Deprecated backward-compatible alias for calibrated_probability.
        - anomaly_signal: Canonical alias for anomaly_score.
        """
        # Resolve canonical ML probability vs backward-compatible alias
        if calibrated_probability is None:
            if ml_prob is not None:
                calibrated_probability = ml_prob
            else:
                raise ContractViolationError(
                    "Missing canonical 'calibrated_probability' (or legacy 'ml_prob') in evaluate_risk"
                )

        calibrated_p = validate_calibrated_probability(calibrated_probability)

        # Resolve anomaly signal
        raw_anomaly = anomaly_score if anomaly_score is not None else anomaly_signal
        norm_anomaly = normalize_anomaly_signal(raw_anomaly, fallback=0.0)

        # 1. Evaluate deterministic business rules
        rule_res = self.rule_engine.evaluate(tx, features_dict)
        rule_risk = rule_res.rules_signal
        rule_violations = rule_res.reasons
        hard_block = bool(kwargs.get("hard_block", False)) or rule_res.hard_block
        norm_rules = normalize_rules_signal(rule_risk)

        # 2. Derive normalized velocity risk component in [0.0, 1.0]
        norm_velocity = normalize_velocity_signal(
            features_dict.get("tx_count_5m", 1),
            features_dict.get("tx_count_1h", 1)
        )

        # 3. Derive normalized behavioral risk component in [0.0, 1.0]
        norm_behavioral = normalize_behavioral_signal(
            features_dict.get("amount_zscore", 0.0),
            features_dict.get("is_new_device", 0),
            features_dict.get("is_new_location", 0)
        )

        # 4. Construct validated normalized signals contract
        signals = NormalizedRiskSignals(
            calibrated_probability=calibrated_p,
            velocity_signal=norm_velocity,
            behavioral_signal=norm_behavioral,
            rules_signal=norm_rules,
            anomaly_signal=norm_anomaly
        )

        # 5. Weighted Linear Aggregation
        weighted_score = self.weights.compute_weighted_fusion(signals)

        # 6. Explicit Multi-Signal Compounding Escalation (applied post-weighted-fusion)
        comp_eval = self.compounding_policy.evaluate(signals)
        combined_score = min(1.0, weighted_score + comp_eval.escalation_boost)

        final_risk = float(round(combined_score * 100.0, 1))

        # 7. Evaluate R4 ML Decision Policy independently for auditability
        ml_decision = self.ml_thresholds.apply_policy(calibrated_p)

        # 8. Check hard overrides & evaluate R5 Hybrid Decision Policy
        if hard_block:
            final_risk = max(final_risk, 85.0)
            decision = "BLOCK"
            risk_level = "HIGH"
        else:
            decision, risk_level = self.thresholds.apply_policy(final_risk)

        # Combine model SHAP reasons with rule violations
        reason_list = list(reasons or [])
        all_reasons = rule_violations + reason_list
        all_reasons = list(dict.fromkeys(all_reasons))[:3]

        # Diagnostics attribution
        pre_esc_score = round(weighted_score * 100.0, 2)
        esc_boost_score = round(comp_eval.escalation_boost * 100.0, 2)
        diag_contribs = {
            "ml": round(calibrated_p * self.weights.w_ml * 100.0, 2),
            "velocity": round(norm_velocity * self.weights.w_velocity * 100.0, 2),
            "behavioral": round(norm_behavioral * self.weights.w_behavioral * 100.0, 2),
            "rules": round(norm_rules * self.weights.w_rules * 100.0, 2),
            "anomaly": round(norm_anomaly * self.weights.w_anomaly * 100.0, 2)
        }
        if pre_esc_score > 0.0:
            diag_percentages = {
                k: round((v / pre_esc_score) * 100.0, 2)
                for k, v in diag_contribs.items()
            }
        else:
            diag_percentages = {k: 0.0 for k in diag_contribs}

        diagnostics_data = {
            "weighted_score": pre_esc_score,
            "escalation_boost": esc_boost_score,
            "trigger_count": comp_eval.trigger_count,
            "active_triggers": [k for k, v in comp_eval.trigger_flags.items() if v],
            "contributions": diag_contribs,
            "contribution_percentages": diag_percentages,
            "hard_block": bool(hard_block),
            "pre_escalation_score": pre_esc_score,
            "pre_clamp_score": round(pre_esc_score + esc_boost_score, 2),
            "final_score": final_risk
        }

        return {
            "transaction_id": tx.get("transaction_id", "unknown_tx"),
            "fraud_probability": round(float(calibrated_p), 4),
            "calibrated_probability": round(float(calibrated_p), 4),
            "risk_score": final_risk,
            "risk_level": risk_level,
            "decision": decision,
            "ml_decision": ml_decision,
            "model_version": self.model_version,
            "signals": {
                "calibrated_probability": round(float(calibrated_p), 4),
                "ml_probability": round(float(calibrated_p), 4),
                "velocity_signal": round(float(norm_velocity), 4),
                "velocity_risk": round(float(norm_velocity), 4),
                "behavioral_signal": round(float(norm_behavioral), 4),
                "behavioral_risk": round(float(norm_behavioral), 4),
                "rules_signal": round(float(norm_rules), 4),
                "rule_risk": round(float(norm_rules), 4),
                "anomaly_signal": round(float(norm_anomaly), 4),
                "anomaly_score": round(float(norm_anomaly), 4)
            },
            "compounding": {
                "trigger_count": comp_eval.trigger_count,
                "escalation_boost": comp_eval.escalation_boost,
                "is_active": comp_eval.is_active
            },
            "diagnostics": diagnostics_data,
            "rule_result": rule_res.to_dict(),
            "top_reasons": all_reasons
        }
