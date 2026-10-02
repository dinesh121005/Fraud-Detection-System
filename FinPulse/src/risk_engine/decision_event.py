"""R5.4 — DecisionEvent v1.0 Contract.

Canonical, versioned, deterministic event schema for publishing completed
R5 hybrid risk decisions to Kafka.

Invariants:
1. schema_version is strictly "1.0"
2. event_type is strictly "finpulse.fraud_decision"
3. event_id is deterministic and stable across retries
4. transaction_id serves as the business idempotency key
5. Full 32-feature vector is NEVER included
6. Independent R4 ML decision (ml_decision) and R5 hybrid decision (decision) are both preserved
7. Structured R5.2 diagnostics and R5.3 rule attributions are preserved
8. Serialization is deterministic UTF-8 JSON
"""

import json
import uuid
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Literal, Union

from .hybrid import HybridRiskResult
from .rules import RuleResult


@dataclass(frozen=True)
class DecisionEvent:
    """
    R5.4 Canonical DecisionEvent v1.0 Contract.
    
    Published to Kafka 'predictions' and 'fraud-alerts' topics.
    """
    transaction_id: str
    customer_id: str
    decision: Literal["APPROVE", "REVIEW", "BLOCK"]
    risk_score: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    ml_decision: Literal["APPROVE", "REVIEW", "BLOCK"]
    calibrated_probability: float
    signals: Dict[str, float]
    diagnostics: Dict[str, Any]
    reasons: List[str]
    event_id: str = field(default="")
    timestamp: float = field(default=0.0)
    schema_version: str = "1.0"
    event_type: str = "finpulse.fraud_decision"
    model_version: str = "finpulse-v3"
    feature_schema_version: str = "2.0"
    matched_rules: List[Dict[str, Any]] = field(default_factory=list)
    hard_block: bool = False
    hard_block_rules: List[str] = field(default_factory=list)
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # 1. Validation: transaction_id and customer_id must be non-empty strings
        if not self.transaction_id or not str(self.transaction_id).strip():
            raise ValueError("Field 'transaction_id' cannot be empty or whitespace.")
        if not self.customer_id or not str(self.customer_id).strip():
            raise ValueError("Field 'customer_id' cannot be empty or whitespace.")

        # 2. Validation: calibrated_probability in [0.0, 1.0]
        p = float(self.calibrated_probability)
        if math.isnan(p) or math.isinf(p) or not (0.0 <= p <= 1.0):
            raise ValueError(f"Field 'calibrated_probability' must be in [0.0, 1.0], got {p}")

        # 3. Validation: risk_score in [0.0, 100.0]
        s = float(self.risk_score)
        if math.isnan(s) or math.isinf(s) or not (0.0 <= s <= 100.0):
            raise ValueError(f"Field 'risk_score' must be in [0.0, 100.0], got {s}")

        # 4. Strict exclusion of 32-feature vector
        for forbidden in ("features", "feature_vector", "ordered_vec", "features_dict"):
            if forbidden in self.metadata:
                raise ValueError(f"Forbidden feature vector key '{forbidden}' found in metadata.")

        # 5. Deterministic event_id if not explicitly provided
        if not self.event_id:
            ts_repr = f"{self.timestamp:.4f}"
            unique_key = f"{self.transaction_id}:{self.model_version}:{ts_repr}"
            generated_id = str(uuid.uuid5(uuid.NAMESPACE_OID, unique_key))
            object.__setattr__(self, "event_id", generated_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to clean dictionary contract."""
        return {
            "schema_version": self.schema_version,
            "event_type": self.event_type,
            "event_id": self.event_id,
            "timestamp": round(float(self.timestamp), 4),
            "transaction_id": self.transaction_id,
            "customer_id": self.customer_id,
            "decision": self.decision,
            "risk_score": round(float(self.risk_score), 4),
            "risk_level": self.risk_level,
            "ml_decision": self.ml_decision,
            "calibrated_probability": round(float(self.calibrated_probability), 6),
            "model_version": self.model_version,
            "feature_schema_version": self.feature_schema_version,
            "signals": {k: round(float(v), 4) for k, v in self.signals.items()},
            "diagnostics": dict(self.diagnostics),
            "matched_rules": list(self.matched_rules),
            "hard_block": bool(self.hard_block),
            "hard_block_rules": list(self.hard_block_rules),
            "reasons": list(self.reasons),
            "latency_ms": round(float(self.latency_ms), 3) if self.latency_ms is not None else None,
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        """Deterministic UTF-8 JSON serialization with sorted keys."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_bytes(self) -> bytes:
        """Deterministic UTF-8 encoded bytes for Kafka transmission."""
        return self.to_json().encode("utf-8")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionEvent":
        """Deserialize and validate from dictionary."""
        d = data.copy()
        # Verify schema version and event type
        s_ver = d.get("schema_version", "1.0")
        if s_ver != "1.0":
            raise ValueError(f"Unsupported schema_version '{s_ver}', expected '1.0'")
        e_type = d.get("event_type", "finpulse.fraud_decision")
        if e_type != "finpulse.fraud_decision":
            raise ValueError(f"Invalid event_type '{e_type}', expected 'finpulse.fraud_decision'")

        return cls(
            transaction_id=str(d["transaction_id"]),
            customer_id=str(d["customer_id"]),
            decision=d["decision"],
            risk_score=float(d["risk_score"]),
            risk_level=d["risk_level"],
            ml_decision=d["ml_decision"],
            calibrated_probability=float(d["calibrated_probability"]),
            signals=dict(d.get("signals", {})),
            diagnostics=dict(d.get("diagnostics", {})),
            reasons=list(d.get("reasons", [])),
            event_id=str(d.get("event_id", "")),
            timestamp=float(d.get("timestamp", 0.0)),
            schema_version=s_ver,
            event_type=e_type,
            model_version=str(d.get("model_version", "finpulse-v3")),
            feature_schema_version=str(d.get("feature_schema_version", "2.0")),
            matched_rules=list(d.get("matched_rules", [])),
            hard_block=bool(d.get("hard_block", False)),
            hard_block_rules=list(d.get("hard_block_rules", [])),
            latency_ms=float(d["latency_ms"]) if d.get("latency_ms") is not None else None,
            metadata=dict(d.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "DecisionEvent":
        """Deserialize from JSON string with strict validation."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON for DecisionEvent: {e}") from e
        return cls.from_dict(data)

    @classmethod
    def from_bytes(cls, raw_bytes: bytes) -> "DecisionEvent":
        """Deserialize from raw Kafka message bytes."""
        try:
            decoded = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"DecisionEvent payload is not valid UTF-8: {e}") from e
        return cls.from_json(decoded)

    @classmethod
    def from_hybrid_result(
        cls,
        result: HybridRiskResult,
        customer_id: str,
        timestamp: float = 0.0,
        latency_ms: Optional[float] = None,
        feature_schema_version: str = "2.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "DecisionEvent":
        """
        Factory to construct DecisionEvent directly from completed HybridRiskResult.
        """
        # Canonical 5 signals
        sig_map = {
            "calibrated_probability": result.calibrated_probability,
            "velocity_signal": result.signals.get("velocity_signal", result.signals.get("velocity", 0.0)),
            "behavioral_signal": result.signals.get("behavioral_signal", result.signals.get("behavioral", 0.0)),
            "rules_signal": result.signals.get("rules_signal", result.signals.get("rules", 0.0)),
            "anomaly_signal": result.signals.get("anomaly_signal", result.signals.get("anomaly", 0.0)),
        }

        # Rule attributions
        rule_meta = result.metadata.get("rule_result", {}) if isinstance(result.metadata, dict) else {}
        matched_rules = rule_meta.get("matched_rules", [])
        hard_block_rules = rule_meta.get("hard_block_rules", [])

        # Sanitized metadata (exclude any feature vectors)
        safe_meta = dict(metadata or {})
        for k in list(safe_meta.keys()):
            if k in ("features", "feature_vector", "ordered_vec", "features_dict"):
                del safe_meta[k]

        return cls(
            transaction_id=result.transaction_id,
            customer_id=customer_id,
            decision=result.decision,
            risk_score=result.risk_score,
            risk_level=result.risk_level,
            ml_decision=result.ml_decision,
            calibrated_probability=result.calibrated_probability,
            signals=sig_map,
            diagnostics=result.diagnostics,
            reasons=list(result.reasons),
            timestamp=timestamp,
            model_version=result.model_version,
            feature_schema_version=feature_schema_version,
            matched_rules=matched_rules,
            hard_block=result.hard_block,
            hard_block_rules=hard_block_rules,
            latency_ms=latency_ms,
            metadata=safe_meta,
        )

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict()
