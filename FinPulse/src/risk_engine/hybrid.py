"""R5.2 — Hybrid Risk Engine Core with Structured Deterministic Diagnostics.

Extends the deterministic R5.1 HybridRiskEngine with structured diagnostic attribution:
- Pre-escalation weighted score
- Multi-signal trigger attribution and count
- Multi-signal compounding escalation boost
- Individual signal contributions and contribution percentages
- Hard-block override state and score reconciliation
- Separation of R4 ML decision from R5 hybrid decision
- Propagation and preservation of R4 production model version

Invariants:
1. Canonical ML Signal:
   - Consumes `calibrated_probability` in [0.0, 1.0] from R4.
2. Input Validation:
   - Strictly validates all 5 signals in [0.0, 1.0].
   - Deterministically rejects missing, null, NaN, infinite, negative, >1.0, or non-numeric values.
   - Zero silent fallback or fabrication of signal values.
3. Weighted Fusion:
   - Deterministic linear aggregation using baseline weights:
     ML=0.45, Velocity=0.15, Behavioral=0.15, Rules=0.15, Anomaly=0.10 (sum = 1.0).
   - Normalized score multiplied by 100.0 to produce weighted_score in [0.0, 100.0].
4. Compounding Escalation Policy:
   - Active trigger conditions:
     * ml:         calibrated_probability > 0.40
     * velocity:   velocity_signal > 0.30
     * behavioral: behavioral_signal > 0.30
     * rules:      rules_signal > 0.40
   - Multi-signal synergy escalation:
     * K >= 2 -> escalation_boost = (K * 0.12) * 100.0 = K * 12.0
     * K < 2  -> escalation_boost = 0.0
   - Final score: min(100.0, weighted_score + escalation_boost)
5. Hard Overrides:
   - If hard_block is True, decision is strictly BLOCK, risk_level is HIGH,
     and score is bounded to at least hard_block_min_score (default 85.0).
6. Diagnostics:
   - Full mathematical reconciliation:
     weighted_score + escalation_boost == final_score (subject to clamping at 100.0 and hard-block override).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Literal, Union, Tuple

from .contract import (
    HybridRiskWeights,
    HybridRiskThresholds,
    R4MLThresholds,
    CompoundingPolicy,
    R4InferenceOutput,
    NormalizedRiskSignals,
    ContractViolationError,
)


@dataclass(frozen=True)
class HybridRiskDiagnostics:
    """Structured deterministic diagnostic attribution for R5 hybrid risk evaluation."""
    weighted_score: float
    escalation_boost: float
    trigger_count: int
    active_triggers: List[str]
    contributions: Dict[str, float]
    contribution_percentages: Dict[str, float]
    hard_block: bool
    pre_escalation_score: float
    final_score: float
    pre_clamp_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert diagnostics to serializable dictionary."""
        return {
            "weighted_score": self.weighted_score,
            "escalation_boost": self.escalation_boost,
            "trigger_count": self.trigger_count,
            "active_triggers": list(self.active_triggers),
            "contributions": dict(self.contributions),
            "contribution_percentages": dict(self.contribution_percentages),
            "hard_block": self.hard_block,
            "pre_escalation_score": self.pre_escalation_score,
            "pre_clamp_score": self.pre_clamp_score,
            "final_score": self.final_score
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict()


@dataclass(frozen=True)
class HybridRiskResult:
    """Canonical R5 hybrid risk scoring and decisioning result contract."""
    transaction_id: str
    model_version: str
    calibrated_probability: float
    signals: Dict[str, float]
    risk_score: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    decision: Literal["APPROVE", "REVIEW", "BLOCK"]
    ml_decision: Literal["APPROVE", "REVIEW", "BLOCK"]
    contributions: Dict[str, float]
    diagnostics: Dict[str, Any]
    reasons: List[str] = field(default_factory=list)
    hard_block: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Diagnostic properties for direct attribute access
    @property
    def weighted_score(self) -> float:
        return self.diagnostics.get("weighted_score", 0.0)

    @property
    def pre_escalation_score(self) -> float:
        return self.diagnostics.get("pre_escalation_score", 0.0)

    @property
    def escalation_boost(self) -> float:
        return self.diagnostics.get("escalation_boost", 0.0)

    @property
    def trigger_count(self) -> int:
        return self.diagnostics.get("trigger_count", 0)

    @property
    def active_triggers(self) -> List[str]:
        return self.diagnostics.get("active_triggers", [])

    @property
    def contribution_percentages(self) -> Dict[str, float]:
        return self.diagnostics.get("contribution_percentages", {})

    @property
    def final_score(self) -> float:
        return self.diagnostics.get("final_score", self.risk_score)

    @property
    def ml_contribution(self) -> float:
        return self.contributions.get("ml_contribution", self.contributions.get("ml", 0.0))

    @property
    def velocity_contribution(self) -> float:
        return self.contributions.get("velocity_contribution", self.contributions.get("velocity", 0.0))

    @property
    def behavioral_contribution(self) -> float:
        return self.contributions.get("behavioral_contribution", self.contributions.get("behavioral", 0.0))

    @property
    def rules_contribution(self) -> float:
        return self.contributions.get("rules_contribution", self.contributions.get("rules", 0.0))

    @property
    def anomaly_contribution(self) -> float:
        return self.contributions.get("anomaly_contribution", self.contributions.get("anomaly", 0.0))

    @property
    def rule_result(self) -> Optional[Dict[str, Any]]:
        return self.metadata.get("rule_result")

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to standard dictionary contract."""
        return {
            "transaction_id": self.transaction_id,
            "model_version": self.model_version,
            "calibrated_probability": self.calibrated_probability,
            "signals": self.signals,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "decision": self.decision,
            "ml_decision": self.ml_decision,
            "contributions": self.contributions,
            "diagnostics": self.diagnostics,
            "reasons": list(self.reasons),
            "hard_block": self.hard_block,
            "metadata": dict(self.metadata)
        }

    def __getitem__(self, key: str) -> Any:
        """Allow subscript indexing for backward compatibility with dictionary consumers."""
        return self.to_dict()[key]

    def __contains__(self, key: str) -> bool:
        """Allow 'key in result' membership testing."""
        return key in self.to_dict()


class HybridRiskEngine:
    """
    R5.2 Hybrid Risk Engine:
    Executes deterministic weighted fusion, compounding escalation, and structured
    diagnostic attribution across 5 validated risk dimensions.
    """

    def __init__(
        self,
        weights: Optional[Union[HybridRiskWeights, Dict[str, float]]] = None,
        thresholds: Optional[Union[HybridRiskThresholds, Dict[str, float]]] = None,
        ml_thresholds: Optional[R4MLThresholds] = None,
        compounding_policy: Optional[CompoundingPolicy] = None,
        model_version: str = "finpulse-v3",
        hard_block_min_score: float = 85.0,
        enable_compounding: bool = False
    ):
        # 1. Weights Configuration (Baseline: 45/15/15/15/10)
        if isinstance(weights, HybridRiskWeights):
            self.weights = weights
        elif isinstance(weights, dict):
            self.weights = HybridRiskWeights(
                w_ml=float(weights.get("w_ml", 0.45)),
                w_velocity=float(weights.get("w_velocity", 0.15)),
                w_behavioral=float(weights.get("w_behavioral", 0.15)),
                w_rules=float(weights.get("w_rules", 0.15)),
                w_anomaly=float(weights.get("w_anomaly", 0.10))
            )
        else:
            self.weights = HybridRiskWeights()

        # 2. Hybrid Risk Decision Thresholds (Domain: [0.0, 100.0])
        if isinstance(thresholds, HybridRiskThresholds):
            self.thresholds = thresholds
        elif isinstance(thresholds, dict):
            self.thresholds = HybridRiskThresholds(
                tau_review=float(thresholds.get("tau_review", 30.0)),
                tau_block=float(thresholds.get("tau_block", 70.0))
            )
        else:
            self.thresholds = HybridRiskThresholds()

        # 3. Independent R4 ML Threshold Policy for Cross-Layer Auditing
        self.ml_thresholds = ml_thresholds or R4MLThresholds()

        # 4. Compounding Policy
        self.compounding_policy = compounding_policy or CompoundingPolicy(
            velocity_threshold=0.30,
            behavioral_threshold=0.30,
            rules_threshold=0.40,
            ml_threshold=0.40,
            min_triggers=2,
            boost_per_trigger=0.12
        )

        # 5. Production Model Version
        self.model_version = str(model_version)

        # 6. Minimum Score for Hard Blocks
        self.hard_block_min_score = float(hard_block_min_score)

        # 7. Compounding Enable Flag
        self.enable_compounding = bool(enable_compounding)

    @staticmethod
    def validate_signal(signal_name: str, value: Any) -> float:
        """
        Strictly validate a normalized signal value.

        Must be numeric, finite, non-null, and within [0.0, 1.0].
        Raises ContractViolationError on any violation.
        """
        if value is None:
            raise ContractViolationError(f"Signal '{signal_name}' cannot be None or missing.")

        # Reject boolean values (since bool is a subclass of int in Python)
        if isinstance(value, bool):
            raise ContractViolationError(
                f"Signal '{signal_name}' must be a numeric float, got boolean: {value}"
            )

        try:
            val_float = float(value)
        except (ValueError, TypeError) as e:
            raise ContractViolationError(
                f"Signal '{signal_name}' must be numeric float, got {type(value).__name__}: {e}"
            )

        if math.isnan(val_float) or math.isinf(val_float):
            raise ContractViolationError(
                f"Signal '{signal_name}' cannot be NaN or Infinite, got: {val_float}"
            )

        if not (0.0 <= val_float <= 1.0):
            raise ContractViolationError(
                f"Signal '{signal_name}' must be strictly in [0.0, 1.0], got: {val_float}"
            )

        return val_float

    def compute_weighted_fusion(
        self,
        ml: float,
        velocity: float,
        behavioral: float,
        rules: float,
        anomaly: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute deterministic linear weighted fusion:
        normalized_score = 0.45*ml + 0.15*vel + 0.15*beh + 0.15*rules + 0.10*anomaly.
        
        Returns:
            Tuple of (normalized_score, contributions_dict).
        """
        ml_c = ml * self.weights.w_ml
        vel_c = velocity * self.weights.w_velocity
        beh_c = behavioral * self.weights.w_behavioral
        rules_c = rules * self.weights.w_rules
        anom_c = anomaly * self.weights.w_anomaly

        normalized_score = ml_c + vel_c + beh_c + rules_c + anom_c

        contributions = {
            "ml_contribution": round(ml_c, 6),
            "velocity_contribution": round(vel_c, 6),
            "behavioral_contribution": round(beh_c, 6),
            "rules_contribution": round(rules_c, 6),
            "anomaly_contribution": round(anom_c, 6)
        }

        return normalized_score, contributions

    def evaluate_triggers(
        self,
        ml: float,
        velocity: float,
        behavioral: float,
        rules: float
    ) -> Tuple[List[str], int]:
        """
        Evaluate active compounding triggers using the established thresholds:
        - ml > 0.40
        - velocity > 0.30
        - behavioral > 0.30
        - rules > 0.40
        """
        active = []
        if ml > self.compounding_policy.ml_threshold:
            active.append("ml")
        if velocity > self.compounding_policy.velocity_threshold:
            active.append("velocity")
        if behavioral > self.compounding_policy.behavioral_threshold:
            active.append("behavioral")
        if rules > self.compounding_policy.rules_threshold:
            active.append("rules")

        return active, len(active)

    def apply_hybrid_policy(
        self,
        raw_risk_score: float,
        hard_block: bool = False
    ) -> Tuple[Literal["APPROVE", "REVIEW", "BLOCK"], Literal["LOW", "MEDIUM", "HIGH"], float]:
        """
        Apply R5 hybrid decision and risk-level policies.

        Boundaries:
            risk_score < 30.0  -> APPROVE, LOW
            30.0 <= risk_score < 70.0 -> REVIEW, MEDIUM
            risk_score >= 70.0 -> BLOCK, HIGH

        Hard Block Override:
            If hard_block is True, decision is strictly BLOCK, risk_level is HIGH,
            and score is bounded to at least hard_block_min_score.
        """
        if hard_block:
            effective_score = max(raw_risk_score, self.hard_block_min_score)
            return "BLOCK", "HIGH", effective_score

        if raw_risk_score >= self.thresholds.tau_block:
            return "BLOCK", "HIGH", raw_risk_score
        elif raw_risk_score >= self.thresholds.tau_review:
            return "REVIEW", "MEDIUM", raw_risk_score
        else:
            return "APPROVE", "LOW", raw_risk_score

    def evaluate(
        self,
        calibrated_probability: Optional[float] = None,
        velocity: Optional[float] = None,
        behavioral: Optional[float] = None,
        rules: Optional[float] = None,
        anomaly: Optional[float] = None,
        transaction_id: str = "unknown_tx",
        model_version: Optional[str] = None,
        ml_decision: Optional[str] = None,
        reasons: Optional[List[str]] = None,
        hard_block: bool = False,
        apply_compounding: Optional[bool] = None,
        signals: Optional[Union[NormalizedRiskSignals, Dict[str, Any]]] = None,
        r4_output: Optional[Union[R4InferenceOutput, Any]] = None,
        rule_result: Optional[Union[Any, Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> HybridRiskResult:
        """
        Execute deterministic hybrid risk evaluation with structured diagnostics.

        Parameters:
            calibrated_probability: Canonical R4 Platt-calibrated ML probability in [0, 1].
            velocity: Validated velocity signal in [0, 1].
            behavioral: Validated behavioral signal in [0, 1].
            rules: Validated rules signal in [0, 1].
            anomaly: Validated anomaly signal in [0, 1].
            transaction_id: Unique transaction identifier.
            model_version: Production model version (defaults to R4 contract or finpulse-v3).
            ml_decision: Independent R4 ML decision (APPROVE / REVIEW / BLOCK).
            reasons: Feature attribution or rule violation reasons list.
            hard_block: Deterministic safety override flag.
            apply_compounding: Explicit override for compounding escalation evaluation.
            signals: Optional NormalizedRiskSignals or dictionary containing the 5 signals.
            r4_output: Optional R4InferenceOutput or InferenceResult instance.
        """
        # 1. Resolve R4 metadata & probability from structured r4_output if provided
        resolved_tx_id = transaction_id
        resolved_version = model_version
        resolved_ml_decision = ml_decision

        if r4_output is not None:
            if hasattr(r4_output, "calibrated_probability"):
                if calibrated_probability is None:
                    calibrated_probability = float(r4_output.calibrated_probability)
                if resolved_tx_id == "unknown_tx" and hasattr(r4_output, "transaction_id"):
                    resolved_tx_id = str(r4_output.transaction_id)
                if resolved_version is None and hasattr(r4_output, "model_version"):
                    resolved_version = str(r4_output.model_version)
                if resolved_ml_decision is None and hasattr(r4_output, "ml_decision"):
                    resolved_ml_decision = str(r4_output.ml_decision)
                elif resolved_ml_decision is None and hasattr(r4_output, "decision"):
                    resolved_ml_decision = str(r4_output.decision)
            elif isinstance(r4_output, dict):
                if calibrated_probability is None:
                    calibrated_probability = r4_output.get("calibrated_probability", r4_output.get("ml_prob"))
                if resolved_tx_id == "unknown_tx":
                    resolved_tx_id = str(r4_output.get("transaction_id", "unknown_tx"))
                if resolved_version is None:
                    resolved_version = r4_output.get("model_version")
                if resolved_ml_decision is None:
                    resolved_ml_decision = r4_output.get("ml_decision", r4_output.get("decision"))

        # Fallback for model_version
        if resolved_version is None:
            resolved_version = self.model_version

        # 2. Resolve signals from signals container or kwargs
        if signals is not None:
            if isinstance(signals, NormalizedRiskSignals):
                if calibrated_probability is None:
                    calibrated_probability = signals.calibrated_probability
                if velocity is None:
                    velocity = signals.velocity_signal
                if behavioral is None:
                    behavioral = signals.behavioral_signal
                if rules is None:
                    rules = signals.rules_signal
                if anomaly is None:
                    anomaly = signals.anomaly_signal
            elif isinstance(signals, dict):
                if calibrated_probability is None:
                    calibrated_probability = signals.get("calibrated_probability", signals.get("ml", signals.get("ml_prob")))
                if velocity is None:
                    velocity = signals.get("velocity", signals.get("velocity_signal", signals.get("velocity_risk")))
                if behavioral is None:
                    behavioral = signals.get("behavioral", signals.get("behavioral_signal", signals.get("behavioral_risk")))
                if rules is None:
                    rules = signals.get("rules", signals.get("rules_signal", signals.get("rule_risk")))
                if anomaly is None:
                    anomaly = signals.get("anomaly", signals.get("anomaly_signal", signals.get("anomaly_score")))

        # Support alias kwargs
        if calibrated_probability is None and "ml" in kwargs:
            calibrated_probability = kwargs["ml"]
        if velocity is None and "velocity_signal" in kwargs:
            velocity = kwargs["velocity_signal"]
        if behavioral is None and "behavioral_signal" in kwargs:
            behavioral = kwargs["behavioral_signal"]
        if rules is None and "rules_signal" in kwargs:
            rules = kwargs["rules_signal"]
        if anomaly is None and "anomaly_signal" in kwargs:
            anomaly = kwargs["anomaly_signal"]

        # Resolve rules, hard_block, and reasons from rule_result if provided
        resolved_metadata: Dict[str, Any] = {}
        if rule_result is not None:
            if hasattr(rule_result, "to_dict"):
                resolved_metadata["rule_result"] = rule_result.to_dict()
            elif isinstance(rule_result, dict):
                resolved_metadata["rule_result"] = dict(rule_result)

            if rules is None:
                if hasattr(rule_result, "rules_signal"):
                    rules = float(rule_result.rules_signal)
                elif isinstance(rule_result, dict) and "rules_signal" in rule_result:
                    rules = float(rule_result["rules_signal"])

            if not hard_block:
                if hasattr(rule_result, "hard_block"):
                    hard_block = bool(rule_result.hard_block)
                elif isinstance(rule_result, dict) and "hard_block" in rule_result:
                    hard_block = bool(rule_result["hard_block"])

            rr_reasons = []
            if hasattr(rule_result, "reasons"):
                rr_reasons = list(rule_result.reasons)
            elif isinstance(rule_result, dict) and "reasons" in rule_result:
                rr_reasons = list(rule_result["reasons"])
            if rr_reasons:
                merged = list(reasons or []) + rr_reasons
                reasons = list(dict.fromkeys(merged))

        # 3. Strict Input Validation (Deterministic rejection of invalid values)
        v_ml = self.validate_signal("calibrated_probability", calibrated_probability)
        v_velocity = self.validate_signal("velocity", velocity)
        v_behavioral = self.validate_signal("behavioral", behavioral)
        v_rules = self.validate_signal("rules", rules)
        v_anomaly = self.validate_signal("anomaly", anomaly)

        # 4. Deterministic Weighted Fusion
        norm_score, raw_contributions = self.compute_weighted_fusion(
            ml=v_ml,
            velocity=v_velocity,
            behavioral=v_behavioral,
            rules=v_rules,
            anomaly=v_anomaly
        )

        weighted_score_raw = norm_score * 100.0
        pre_escalation_score = round(weighted_score_raw, 4)

        # 5. Evaluate Multi-Signal Compounding Triggers & Escalation
        active_triggers, trigger_count = self.evaluate_triggers(
            ml=v_ml,
            velocity=v_velocity,
            behavioral=v_behavioral,
            rules=v_rules
        )

        should_compound = self.enable_compounding if apply_compounding is None else bool(apply_compounding)

        if should_compound and trigger_count >= self.compounding_policy.min_triggers:
            norm_boost = trigger_count * self.compounding_policy.boost_per_trigger
            escalation_boost = round(norm_boost * 100.0, 4)
        else:
            norm_boost = 0.0
            escalation_boost = 0.0

        # Pre-clamp combined score
        pre_clamp_score = round(pre_escalation_score + escalation_boost, 4)
        clamped_score = min(100.0, pre_clamp_score)

        # 6. Independent R4 ML Policy Evaluation
        if resolved_ml_decision is None:
            resolved_ml_decision = self.ml_thresholds.apply_policy(v_ml)

        # 7. Apply R5 Hybrid Decision & Risk Level Policies
        decision, risk_level, effective_score = self.apply_hybrid_policy(
            raw_risk_score=clamped_score,
            hard_block=hard_block
        )

        final_risk_score = round(float(effective_score), 4)

        # 8. Diagnostic Attribution Breakdown
        # Scaled contributions on 0-100 scale: individual_i = signal_i * weight_i * 100
        diag_contributions = {
            "ml": round(v_ml * self.weights.w_ml * 100.0, 4),
            "velocity": round(v_velocity * self.weights.w_velocity * 100.0, 4),
            "behavioral": round(v_behavioral * self.weights.w_behavioral * 100.0, 4),
            "rules": round(v_rules * self.weights.w_rules * 100.0, 4),
            "anomaly": round(v_anomaly * self.weights.w_anomaly * 100.0, 4)
        }

        # Contribution percentages: individual / weighted_score * 100
        if pre_escalation_score > 0.0:
            diag_percentages = {
                k: round((v / pre_escalation_score) * 100.0, 4)
                for k, v in diag_contributions.items()
            }
        else:
            diag_percentages = {
                "ml": 0.0,
                "velocity": 0.0,
                "behavioral": 0.0,
                "rules": 0.0,
                "anomaly": 0.0
            }

        diagnostics_obj = HybridRiskDiagnostics(
            weighted_score=round(pre_escalation_score, 2),
            escalation_boost=round(escalation_boost, 2),
            trigger_count=trigger_count,
            active_triggers=active_triggers,
            contributions={k: round(v, 2) for k, v in diag_contributions.items()},
            contribution_percentages={k: round(v, 2) for k, v in diag_percentages.items()},
            hard_block=bool(hard_block),
            pre_escalation_score=round(pre_escalation_score, 2),
            final_score=round(final_risk_score, 2),
            pre_clamp_score=round(pre_clamp_score, 2)
        )

        # Structure signals dictionary supporting both canonical and short keys
        signals_dict = {
            "ml": round(v_ml, 4),
            "calibrated_probability": round(v_ml, 4),
            "velocity": round(v_velocity, 4),
            "velocity_signal": round(v_velocity, 4),
            "behavioral": round(v_behavioral, 4),
            "behavioral_signal": round(v_behavioral, 4),
            "rules": round(v_rules, 4),
            "rules_signal": round(v_rules, 4),
            "anomaly": round(v_anomaly, 4),
            "anomaly_signal": round(v_anomaly, 4)
        }

        return HybridRiskResult(
            transaction_id=str(resolved_tx_id),
            model_version=str(resolved_version),
            calibrated_probability=round(v_ml, 4),
            signals=signals_dict,
            risk_score=final_risk_score,
            risk_level=risk_level,
            decision=decision,
            ml_decision=resolved_ml_decision,
            contributions=raw_contributions,
            diagnostics=diagnostics_obj.to_dict(),
            reasons=list(reasons or []),
            hard_block=hard_block,
            metadata=resolved_metadata
        )
