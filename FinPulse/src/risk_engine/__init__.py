"""FinPulse Risk Engine module providing hybrid risk scoring and R4 -> R5 contracts."""
from .contract import (
    R4InferenceOutput,
    NormalizedRiskSignals,
    R4MLThresholds,
    HybridRiskThresholds,
    HybridRiskWeights,
    CompoundingPolicy,
    CompoundingEvaluation,
    ContractViolationError,
    normalize_velocity_signal,
    normalize_behavioral_signal,
    normalize_rules_signal,
    normalize_anomaly_signal,
    validate_calibrated_probability,
)
from .rules import (
    BusinessRuleEngine,
    DeterministicRuleEngine,
    BusinessRule,
    MatchedRule,
    RuleResult,
)
from .engine import FinPulseRiskEngine
from .hybrid import HybridRiskEngine, HybridRiskResult, HybridRiskDiagnostics
from .decision_event import DecisionEvent

__all__ = [
    "DecisionEvent",
    "HybridRiskEngine",
    "HybridRiskResult",
    "HybridRiskDiagnostics",
    "FinPulseRiskEngine",
    "BusinessRuleEngine",
    "DeterministicRuleEngine",
    "BusinessRule",
    "MatchedRule",
    "RuleResult",
    "R4InferenceOutput",
    "NormalizedRiskSignals",
    "R4MLThresholds",
    "HybridRiskThresholds",
    "HybridRiskWeights",
    "CompoundingPolicy",
    "CompoundingEvaluation",
    "ContractViolationError",
    "normalize_velocity_signal",
    "normalize_behavioral_signal",
    "normalize_rules_signal",
    "normalize_anomaly_signal",
    "validate_calibrated_probability",
]
