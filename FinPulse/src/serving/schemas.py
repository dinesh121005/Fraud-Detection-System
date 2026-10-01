"""Pydantic schemas for FastAPI serving endpoint contracts."""
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., example="tx_9823482104")
    timestamp: float = Field(..., example=1727725200.0)
    amount: float = Field(..., gt=0.0, example=14500.0)
    customer_id: str = Field(..., example="C982341")
    merchant_id: str = Field(..., example="M847291")
    category: str = Field("electronics", example="electronics")
    payment_type: str = Field("TRANSFER", example="TRANSFER")
    origin_balance: float = Field(15000.0, example=15000.0)
    latitude: float = Field(37.7749, example=37.7749)
    longitude: float = Field(-122.4194, example=-122.4194)
    device_id: str = Field("dev_mac_8392", example="dev_mac_8392")
    auth_verified: bool = Field(True, example=True)

class SignalsSchema(BaseModel):
    ml_probability: float
    anomaly_score: float
    velocity_risk: float
    behavioral_risk: float
    rule_risk: float

class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_score: float
    risk_level: str # LOW, MEDIUM, HIGH
    decision: str   # APPROVE, REVIEW, BLOCK
    model_version: str = "finpulse-v2.0"
    signals: SignalsSchema
    top_reasons: List[str]
    latency_ms: float
