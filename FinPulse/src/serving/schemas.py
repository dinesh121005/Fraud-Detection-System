"""Pydantic schemas for FastAPI serving endpoint contracts with R6 Security Hardening."""
import math
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, List, Optional

class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., max_length=128, json_schema_extra={"example": "tx_9823482104"})
    timestamp: float = Field(..., json_schema_extra={"example": 1727725200.0})
    amount: float = Field(..., gt=0.0, le=100_000_000.0, json_schema_extra={"example": 14500.0})
    customer_id: str = Field(..., max_length=128, json_schema_extra={"example": "C982341"})
    merchant_id: str = Field(..., max_length=128, json_schema_extra={"example": "M847291"})
    category: str = Field("electronics", max_length=64, json_schema_extra={"example": "electronics"})
    payment_type: str = Field("TRANSFER", max_length=32, json_schema_extra={"example": "TRANSFER"})
    origin_balance: float = Field(15000.0, ge=0.0, json_schema_extra={"example": 15000.0})
    latitude: float = Field(37.7749, ge=-90.0, le=90.0, json_schema_extra={"example": 37.7749})
    longitude: float = Field(-122.4194, ge=-180.0, le=180.0, json_schema_extra={"example": -122.4194})
    device_id: str = Field("dev_mac_8392", max_length=128, json_schema_extra={"example": "dev_mac_8392"})
    auth_verified: bool = Field(True, json_schema_extra={"example": True})

    @field_validator("transaction_id", "customer_id", "merchant_id")
    @classmethod
    def check_non_empty_string(cls, v: str, info) -> str:
        if not v or not str(v).strip():
            raise ValueError(f"Field '{info.field_name}' cannot be empty or whitespace.")
        return str(v).strip()

    @field_validator("amount", "timestamp", "origin_balance", "latitude", "longitude")
    @classmethod
    def check_non_nan_finite(cls, v: float, info) -> float:
        val = float(v)
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"Field '{info.field_name}' must be a finite number, cannot be NaN or Infinity.")
        return val

    @field_validator("timestamp")
    @classmethod
    def check_positive_timestamp(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError("Timestamp must be a positive unix epoch time in seconds.")
        return v


class SignalsSchema(BaseModel):
    # Canonical R4 -> R5 Probability
    calibrated_probability: float = Field(..., description="Canonical Platt-calibrated ML fraud probability in [0, 1]")
    ml_probability: Optional[float] = Field(None, description="Deprecated backward-compatible alias for calibrated_probability")
    
    # Anomaly Signal
    anomaly_signal: Optional[float] = Field(None, description="Canonical normalized anomaly signal in [0, 1]")
    anomaly_score: float = Field(..., description="Legacy alias for anomaly_signal")
    
    # Velocity Signal
    velocity_signal: Optional[float] = Field(None, description="Canonical normalized velocity signal in [0, 1]")
    velocity_risk: float = Field(..., description="Legacy alias for velocity_signal")
    
    # Behavioral Signal
    behavioral_signal: Optional[float] = Field(None, description="Canonical normalized behavioral signal in [0, 1]")
    behavioral_risk: float = Field(..., description="Legacy alias for behavioral_signal")
    
    # Rules Signal
    rules_signal: Optional[float] = Field(None, description="Canonical normalized rules signal in [0, 1]")
    rule_risk: float = Field(..., description="Legacy alias for rules_signal")


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    calibrated_probability: Optional[float] = None
    risk_score: float
    risk_level: str # LOW, MEDIUM, HIGH
    decision: str   # APPROVE, REVIEW, BLOCK
    ml_decision: Optional[str] = None # R4 ML decision for policy separation auditability
    model_version: str = "finpulse-v3"
    signals: SignalsSchema
    compounding: Optional[Dict[str, Any]] = None
    diagnostics: Optional[Dict[str, Any]] = None
    top_reasons: List[str]
    latency_ms: float
