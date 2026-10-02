"""R4 → R5 Interface Contract and Boundary Specification for FinPulse.

Defines the formal boundary between:
- R4: Real-time ML model inference, Platt calibration, and ML decision policy.
- R5: Hybrid Risk Engine (multi-signal weighted fusion, compounding, overrides).

Key Invariants:
1. Canonical ML Probability:
   - R5 consumes `calibrated_probability` (float in [0.0, 1.0]).
   - Ambiguous `ml_prob` is retained strictly as a deprecated backward-compatible alias.
2. Production Model Version:
   - Production model version is strictly reported as "finpulse-v3".
   - "finpulse-v2.0" is deprecated and removed from active production paths.
3. Separation of Decision Policies:
   - R4 ML Decision Policy: [0.0, 1.0] domain, thresholds: tau_review=0.1580, tau_block=0.5516.
   - R5 Hybrid Risk Policy: [0.0, 100.0] domain, thresholds: tau_review=30.0, tau_block=70.0.
   - The policies operate on different scales and must never be conflated.
4. Anomaly-Score Contract:
   - Sourced from IsolationForestAnomalyDetector (in src.models.anomaly) or incoming anomaly signal.
   - Raw score represents decision_function(X) where positive=normal, negative=outlier.
   - Normalized via sigmoid inversion into [0.0, 1.0], where 1.0 = severe anomaly, 0.0 = legitimate inlier.
   - Safe neutral fallback on missing/NaN is 0.0.
5. Signal Normalization Contract:
   - All 5 hybrid input signals (ML, Velocity, Behavioral, Rules, Anomaly) are guaranteed in [0.0, 1.0].
6. Compounding Escalation Policy:
   - Multi-signal synergy escalation is applied post-weighted-fusion:
     If >= 2 trigger conditions fire (velocity > 0.3, behavioral > 0.3, rules > 0.4, ml > 0.4),
     an explicit escalation of (+0.12 * triggers) is compounded, capped at 1.0.
7. Initial Weight Configuration:
   - Initial engineering configuration: ML=45%, Velocity=15%, Behavioral=15%, Rules=15%, Anomaly=10%.
   - Sum equals 1.0 (100%). Explicitly treated as baseline engineering configuration.
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Literal, Union, Tuple


class ContractViolationError(ValueError):
    """Raised when an incoming payload or signal violates the R4 -> R5 boundary contract."""
    pass


# ==============================================================================
# 1. R4 ML Inference Output Contract
# ==============================================================================

@dataclass(frozen=True)
class R4InferenceOutput:
    """
    Standardized R4 output contract passed across the boundary to R5.

    Specifications:
    - field name: `calibrated_probability` (canonical)
    - data type: `float`
    - valid range: `[0.0, 1.0]`
    - meaning: Posterior fraud probability calibrated via Platt scaling on frozen CatBoost output
    - producer: R4 ProductionModelService / ML scoring pipeline
    - consumer: R5 FinPulseRiskEngine (hybrid fusion layer)
    - calibrated: True
    - represents: Probability (0.0 to 1.0), distinct from raw score and distinct from decision
    """
    transaction_id: str
    calibrated_probability: float
    model_version: str = "finpulse-v3"
    feature_schema_version: str = "2.0"
    raw_probability: Optional[float] = None
    ml_decision: Literal["APPROVE", "REVIEW", "BLOCK"] = "APPROVE"
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.calibrated_probability is None or math.isnan(self.calibrated_probability) or math.isinf(self.calibrated_probability):
            raise ContractViolationError(f"calibrated_probability must be a valid float, got: {self.calibrated_probability}")
        if not (0.0 <= self.calibrated_probability <= 1.0):
            raise ContractViolationError(f"calibrated_probability must be in [0.0, 1.0], got: {self.calibrated_probability}")

    @property
    def ml_prob(self) -> float:
        """Deprecated backward-compatible alias for calibrated_probability."""
        return self.calibrated_probability

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        res = {
            "transaction_id": self.transaction_id,
            "model_version": self.model_version,
            "feature_schema_version": self.feature_schema_version,
            "calibrated_probability": round(float(self.calibrated_probability), 6),
            "ml_decision": self.ml_decision
        }
        if self.raw_probability is not None:
            res["raw_probability"] = round(float(self.raw_probability), 6)
        if self.latency_ms is not None:
            res["latency_ms"] = round(float(self.latency_ms), 3)
        return res

    @classmethod
    def from_inference_result(cls, inf_result: Any) -> "R4InferenceOutput":
        """Instantiate contract from R4 InferenceResult instance or dict."""
        if hasattr(inf_result, "calibrated_probability"):
            return cls(
                transaction_id=str(getattr(inf_result, "transaction_id", "unknown")),
                calibrated_probability=float(inf_result.calibrated_probability),
                model_version=str(getattr(inf_result, "model_version", "finpulse-v3")),
                feature_schema_version=str(getattr(inf_result, "feature_schema_version", "2.0")),
                raw_probability=float(inf_result.raw_probability) if hasattr(inf_result, "raw_probability") else None,
                ml_decision=getattr(inf_result, "decision", "APPROVE"),
                latency_ms=getattr(inf_result, "latency_ms", None),
                metadata=getattr(inf_result, "metadata", {})
            )
        elif isinstance(inf_result, dict):
            prob = inf_result.get("calibrated_probability", inf_result.get("ml_prob"))
            if prob is None:
                raise ContractViolationError("Inference dictionary missing 'calibrated_probability' key")
            return cls(
                transaction_id=str(inf_result.get("transaction_id", "unknown")),
                calibrated_probability=float(prob),
                model_version=str(inf_result.get("model_version", "finpulse-v3")),
                feature_schema_version=str(inf_result.get("feature_schema_version", "2.0")),
                raw_probability=float(inf_result["raw_probability"]) if "raw_probability" in inf_result else None,
                ml_decision=inf_result.get("ml_decision", inf_result.get("decision", "APPROVE")),
                latency_ms=inf_result.get("latency_ms")
            )
        raise ContractViolationError(f"Unsupported inference result object type: {type(inf_result)}")


# ==============================================================================
# 2. Normalized Hybrid Risk Signals Contract
# ==============================================================================

@dataclass(frozen=True)
class NormalizedRiskSignals:
    """
    Guarantees that all 5 hybrid risk signals entering the fusion engine
    are strictly bounded in [0.0, 1.0] and contain no NaN/infinite values.
    """
    calibrated_probability: float
    velocity_signal: float
    behavioral_signal: float
    rules_signal: float
    anomaly_signal: float

    def __post_init__(self):
        signals = {
            "calibrated_probability": self.calibrated_probability,
            "velocity_signal": self.velocity_signal,
            "behavioral_signal": self.behavioral_signal,
            "rules_signal": self.rules_signal,
            "anomaly_signal": self.anomaly_signal
        }
        for name, val in signals.items():
            if val is None or math.isnan(val) or math.isinf(val):
                raise ContractViolationError(f"Signal '{name}' must be a valid float, got: {val}")
            if not (0.0 <= val <= 1.0):
                raise ContractViolationError(f"Signal '{name}' must be in [0.0, 1.0], got: {val}")

    # Backward compatibility aliases
    @property
    def ml_prob(self) -> float:
        return self.calibrated_probability

    @property
    def ml_probability(self) -> float:
        return self.calibrated_probability

    @property
    def velocity_risk(self) -> float:
        return self.velocity_signal

    @property
    def behavioral_risk(self) -> float:
        return self.behavioral_signal

    @property
    def rule_risk(self) -> float:
        return self.rules_signal

    @property
    def anomaly_score(self) -> float:
        return self.anomaly_signal

    def to_dict(self) -> Dict[str, float]:
        """Produce dictionary with both canonical signal names and legacy backward-compatible keys."""
        return {
            "calibrated_probability": round(float(self.calibrated_probability), 4),
            "ml_probability": round(float(self.calibrated_probability), 4),
            "velocity_signal": round(float(self.velocity_signal), 4),
            "velocity_risk": round(float(self.velocity_signal), 4),
            "behavioral_signal": round(float(self.behavioral_signal), 4),
            "behavioral_risk": round(float(self.behavioral_signal), 4),
            "rules_signal": round(float(self.rules_signal), 4),
            "rule_risk": round(float(self.rules_signal), 4),
            "anomaly_signal": round(float(self.anomaly_signal), 4),
            "anomaly_score": round(float(self.anomaly_signal), 4)
        }


# ==============================================================================
# 3. Policy Thresholds (Separation of ML vs Hybrid Decisioning)
# ==============================================================================

@dataclass(frozen=True)
class R4MLThresholds:
    """
    Frozen R4 ML Decision Policy Thresholds.
    Domain: Calibrated Probability in [0.0, 1.0].
    
    tau_review: 0.1580
    tau_block:  0.5516
    """
    tau_review: float = 0.1580
    tau_block: float = 0.5516

    def apply_policy(self, calibrated_prob: float) -> Literal["APPROVE", "REVIEW", "BLOCK"]:
        """Assign ML decision based exclusively on R4 ML thresholds."""
        if math.isnan(calibrated_prob) or math.isinf(calibrated_prob) or not (0.0 <= calibrated_prob <= 1.0):
            raise ContractViolationError(f"Cannot apply ML policy to invalid probability: {calibrated_prob}")
        if calibrated_prob < self.tau_review:
            return "APPROVE"
        elif calibrated_prob < self.tau_block:
            return "REVIEW"
        else:
            return "BLOCK"


@dataclass(frozen=True)
class HybridRiskThresholds:
    """
    R5 Hybrid Risk Engine Thresholds.
    Domain: Composite Risk Score in [0.0, 100.0].
    
    tau_review: 30.0
    tau_block:  70.0
    
    NOTE: These thresholds operate on the 0-100 risk score domain and must never
    be confused with R4 ML probability thresholds (0.1580 / 0.5516).
    """
    tau_review: float = 30.0
    tau_block: float = 70.0

    def __post_init__(self):
        if not (0.0 <= self.tau_review < self.tau_block <= 100.0):
            raise ContractViolationError(
                f"Invalid hybrid thresholds: 0.0 <= tau_review ({self.tau_review}) < tau_block ({self.tau_block}) <= 100.0"
            )

    def apply_policy(self, risk_score: float) -> Tuple[Literal["APPROVE", "REVIEW", "BLOCK"], Literal["LOW", "MEDIUM", "HIGH"]]:
        """Assign hybrid risk decision and risk level based on R5 hybrid thresholds."""
        if math.isnan(risk_score) or math.isinf(risk_score) or not (0.0 <= risk_score <= 100.0):
            raise ContractViolationError(f"Cannot apply hybrid policy to invalid risk score: {risk_score}")
        if risk_score >= self.tau_block:
            return "BLOCK", "HIGH"
        elif risk_score >= self.tau_review:
            return "REVIEW", "MEDIUM"
        else:
            return "APPROVE", "LOW"


# ==============================================================================
# 4. Initial Weight Configuration
# ==============================================================================

@dataclass(frozen=True)
class HybridRiskWeights:
    """
    R5 Hybrid Risk Fusion Weights.

    Initial Weight Configuration:
    - ML / calibrated probability = 45% (0.45)
    - Velocity                    = 15% (0.15)
    - Behavioral                  = 15% (0.15)
    - Rules                       = 15% (0.15)
    - Anomaly                     = 10% (0.10)
    Total: 100% (1.00)

    IMPORTANT NOTICE:
    These weights represent an explicit initial engineering configuration for deterministic
    fusion. They are NOT claimed to be scientifically validated or empirically optimal.
    Rigorous weight optimization and tuning remain subject to future validation studies.
    """
    w_ml: float = 0.45
    w_velocity: float = 0.15
    w_behavioral: float = 0.15
    w_rules: float = 0.15
    w_anomaly: float = 0.10

    def __post_init__(self):
        total = self.w_ml + self.w_velocity + self.w_behavioral + self.w_rules + self.w_anomaly
        if not math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ContractViolationError(f"Hybrid weights must sum to exactly 1.0, got {total}")
        for k, v in [
            ("w_ml", self.w_ml),
            ("w_velocity", self.w_velocity),
            ("w_behavioral", self.w_behavioral),
            ("w_rules", self.w_rules),
            ("w_anomaly", self.w_anomaly)
        ]:
            if not (0.0 <= v <= 1.0):
                raise ContractViolationError(f"Weight '{k}' must be in [0.0, 1.0], got: {v}")

    def compute_weighted_fusion(self, signals: NormalizedRiskSignals) -> float:
        """Compute linear deterministic weighted fusion of normalized signals."""
        return (
            self.w_ml * signals.calibrated_probability +
            self.w_velocity * signals.velocity_signal +
            self.w_behavioral * signals.behavioral_signal +
            self.w_rules * signals.rules_signal +
            self.w_anomaly * signals.anomaly_signal
        )


# ==============================================================================
# 5. Explicit Compounding Policy
# ==============================================================================

@dataclass(frozen=True)
class CompoundingEvaluation:
    """Result of multi-signal trigger compounding policy evaluation."""
    trigger_count: int
    trigger_flags: Dict[str, bool]
    escalation_boost: float
    is_active: bool


@dataclass(frozen=True)
class CompoundingPolicy:
    """
    Multi-Signal Compounding and Risk Escalation Policy.

    Policy Definition:
    - Compounding is evaluated strictly AFTER linear weighted fusion.
    - Each normalized signal is checked against its designated trigger threshold:
      * Velocity trigger:     velocity_signal > 0.30
      * Behavioral trigger:   behavioral_signal > 0.30
      * Rules trigger:        rules_signal > 0.40
      * ML trigger:           calibrated_probability > 0.40
    - Multi-signal synergy escalation:
      If trigger_count >= 2:
          escalation_boost = trigger_count * 0.12
      If trigger_count < 2:
          escalation_boost = 0.0
    - Final combined score before percentage scaling:
      combined_score = min(1.0, weighted_fusion + escalation_boost)
    """
    velocity_threshold: float = 0.30
    behavioral_threshold: float = 0.30
    rules_threshold: float = 0.40
    ml_threshold: float = 0.40
    min_triggers: int = 2
    boost_per_trigger: float = 0.12

    def evaluate(self, signals: NormalizedRiskSignals) -> CompoundingEvaluation:
        """Deterministic evaluation of trigger conditions and resulting escalation boost."""
        flags = {
            "ml": signals.calibrated_probability > self.ml_threshold,
            "velocity": signals.velocity_signal > self.velocity_threshold,
            "behavioral": signals.behavioral_signal > self.behavioral_threshold,
            "rules": signals.rules_signal > self.rules_threshold
        }
        trigger_count = sum(1 for triggered in flags.values() if triggered)
        is_active = trigger_count >= self.min_triggers
        boost = (trigger_count * self.boost_per_trigger) if is_active else 0.0

        return CompoundingEvaluation(
            trigger_count=trigger_count,
            trigger_flags=flags,
            escalation_boost=round(boost, 4),
            is_active=is_active
        )


# ==============================================================================
# 6. Explicit Signal Normalization Functions
# ==============================================================================

def normalize_velocity_signal(tx_count_5m: Any, tx_count_1h: Any) -> float:
    """
    Normalize Redis sliding-window transaction counts into velocity_signal in [0.0, 1.0].

    Source: Redis sliding window rolling counters.
    Formula: min(1.0, max(0.0, (tx_count_5m / 5.0) * 0.6 + (tx_count_1h / 15.0) * 0.4))
    Fallback: Invalid, missing, None, or NaN counts default to 0.0.
    """
    try:
        c_5m = float(tx_count_5m) if tx_count_5m is not None else 0.0
        c_1h = float(tx_count_1h) if tx_count_1h is not None else 0.0
        if math.isnan(c_5m) or math.isinf(c_5m) or c_5m < 0.0:
            c_5m = 0.0
        if math.isnan(c_1h) or math.isinf(c_1h) or c_1h < 0.0:
            c_1h = 0.0
    except (ValueError, TypeError):
        c_5m, c_1h = 0.0, 0.0

    raw = (c_5m / 5.0) * 0.6 + (c_1h / 15.0) * 0.4
    return float(max(0.0, min(1.0, raw)))


def normalize_behavioral_signal(amount_zscore: Any, is_new_device: Any, is_new_location: Any) -> float:
    """
    Normalize customer behavioral deviation signals into behavioral_signal in [0.0, 1.0].

    Source: Feature vector features (amount_zscore, is_new_device, is_new_location).
    Formula: min(1.0, max(0.0, (|amount_zscore| / 4.0) * 0.5 + is_new_device * 0.3 + is_new_location * 0.2))
    Fallback: Invalid, missing, None, or NaN values default to 0.0.
    """
    try:
        z = abs(float(amount_zscore)) if amount_zscore is not None else 0.0
        dev = float(is_new_device) if is_new_device is not None else 0.0
        loc = float(is_new_location) if is_new_location is not None else 0.0
        if math.isnan(z) or math.isinf(z):
            z = 0.0
        if math.isnan(dev) or math.isinf(dev) or dev < 0.0:
            dev = 0.0
        if math.isnan(loc) or math.isinf(loc) or loc < 0.0:
            loc = 0.0
    except (ValueError, TypeError):
        z, dev, loc = 0.0, 0.0, 0.0

    raw = (z / 4.0) * 0.5 + min(1.0, dev) * 0.3 + min(1.0, loc) * 0.2
    return float(max(0.0, min(1.0, raw)))


def normalize_rules_signal(raw_rule_risk: Any) -> float:
    """
    Normalize deterministic rule engine output into rules_signal in [0.0, 1.0].

    Source: DeterministicRuleEngine.evaluate_rules().
    Formula: Bounded clipping of rule activation penalty sum in [0.0, 1.0].
    Fallback: Invalid, missing, None, or NaN values default to 0.0.
    """
    try:
        r = float(raw_rule_risk) if raw_rule_risk is not None else 0.0
        if math.isnan(r) or math.isinf(r) or r < 0.0:
            r = 0.0
    except (ValueError, TypeError):
        r = 0.0
    return float(max(0.0, min(1.0, r)))


def normalize_anomaly_signal(raw_anomaly: Any, fallback: float = 0.0) -> float:
    """
    Normalize anomaly score into anomaly_signal in [0.0, 1.0].

    Source: IsolationForestAnomalyDetector (in src.models.anomaly) or pre-computed anomaly score.
    Algorithm: Unsupervised IsolationForest decision_function with sigmoid inversion.
    Meaning: Higher means MORE anomalous (0.0 = completely normal inlier, 1.0 = severe outlier).
    Fallback: If score is None, NaN, inf, or missing, safely defaults to fallback (default 0.0).
    """
    if raw_anomaly is None:
        return float(fallback)
    try:
        a = float(raw_anomaly)
        if math.isnan(a) or math.isinf(a):
            return float(fallback)
    except (ValueError, TypeError):
        return float(fallback)
    return float(max(0.0, min(1.0, a)))


def validate_calibrated_probability(prob: Any) -> float:
    """
    Validate canonical R4 calibrated probability input for R5 consumption.
    Strictly enforces float in [0.0, 1.0]; raises ContractViolationError on violation.
    """
    if prob is None:
        raise ContractViolationError("calibrated_probability cannot be None")
    try:
        p = float(prob)
    except (ValueError, TypeError) as e:
        raise ContractViolationError(f"calibrated_probability must be numeric float: {e}")

    if math.isnan(p) or math.isinf(p):
        raise ContractViolationError(f"calibrated_probability cannot be NaN or Inf, got: {p}")
    if not (0.0 <= p <= 1.0):
        raise ContractViolationError(f"calibrated_probability must be in [0.0, 1.0], got: {p}")

    return p
