"""Transaction Event Contract for Real-Time Streaming Ingestion.

Defines the canonical, versioned, serializable event schema transported
through Kafka to support downstream state tracking and inference workers.
"""
import json
import time
import hashlib
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from src.features.schema import CommonTransactionSchema, LocationSchema

class TransactionEvent(BaseModel):
    """
    Canonical Streaming Transaction Contract (v1.0).
    Guarantees schema versioning, strict serialization/deserialization,
    and bi-directional compatibility with CommonTransactionSchema.
    """
    schema_version: str = Field("1.0", description="Contract schema version")
    transaction_id: str = Field(..., description="Unique transaction identifier")
    timestamp: float = Field(default_factory=time.time, description="Unix epoch timestamp in seconds")
    amount: float = Field(..., gt=0.0, description="Transaction monetary value strictly greater than 0")
    customer_id: str = Field(..., description="Cardholder or account identifier")
    merchant_id: str = Field(..., description="Merchant or POS terminal identifier")
    category: str = Field("general", description="Merchant classification or product category")
    payment_type: str = Field("TRANSFER", description="Payment rail / instrument")
    origin_balance: float = Field(0.0, ge=0.0, description="Pre-transaction customer balance or limit proxy")
    latitude: float = Field(0.0, description="Transaction GPS latitude (-90 to 90)")
    longitude: float = Field(0.0, description="Transaction GPS longitude (-180 to 180)")
    home_latitude: Optional[float] = Field(None, description="Cardholder home GPS latitude")
    home_longitude: Optional[float] = Field(None, description="Cardholder home GPS longitude")
    device_id: str = Field("unknown_device", description="Hardware or digital device identifier")
    auth_verified: bool = Field(True, description="Whether 2FA / biometric step-up authentication succeeded")
    is_fraud: Optional[int] = Field(None, description="Optional ground truth label for validation streams")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extensible metadata without schema drift")

    @field_validator("transaction_id", "customer_id", "merchant_id")
    @classmethod
    def check_non_empty_string(cls, v: str, info) -> str:
        if not v or not v.strip():
            raise ValueError(f"Field '{info.field_name}' cannot be empty or whitespace")
        return v.strip()

    @field_validator("amount")
    @classmethod
    def check_amount_positive(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError("Transaction amount must be strictly positive (> 0.0)")
        return float(v)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dictionary."""
        return self.model_dump()

    def to_json(self) -> str:
        """Serialize event to canonical JSON string."""
        return self.model_dump_json()

    def to_bytes(self) -> bytes:
        """Serialize event to UTF-8 encoded bytes for Kafka payload."""
        return self.to_json().encode("utf-8")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransactionEvent":
        """Deserialize and validate from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "TransactionEvent":
        """Deserialize and validate from JSON string with clear validation errors."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON payload: {e}") from e
        return cls.from_dict(data)

    @classmethod
    def from_bytes(cls, raw_bytes: bytes) -> "TransactionEvent":
        """Deserialize and validate from raw Kafka message bytes."""
        try:
            decoded = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"Payload not valid UTF-8: {e}") from e
        return cls.from_json(decoded)

    def to_common_schema(self) -> CommonTransactionSchema:
        """Convert event to canonical FinPulse CommonTransactionSchema."""
        return CommonTransactionSchema(
            transaction_id=self.transaction_id,
            timestamp=self.timestamp,
            amount=self.amount,
            customer_id=self.customer_id,
            merchant_id=self.merchant_id,
            category=self.category,
            payment_type=self.payment_type,
            origin_balance=self.origin_balance,
            location=LocationSchema(latitude=self.latitude, longitude=self.longitude),
            home_location=LocationSchema(latitude=self.home_latitude, longitude=self.home_longitude)
            if (self.home_latitude is not None and self.home_longitude is not None) else None,
            device_id=self.device_id,
            auth_verified=self.auth_verified,
            is_fraud=self.is_fraud,
            dataset_source="streaming",
            raw_metadata=self.metadata
        )

    @classmethod
    def from_common_schema(cls, common: CommonTransactionSchema) -> "TransactionEvent":
        """Build TransactionEvent from CommonTransactionSchema."""
        return cls(
            schema_version="1.0",
            transaction_id=common.transaction_id,
            timestamp=common.timestamp,
            amount=common.amount,
            customer_id=common.customer_id,
            merchant_id=common.merchant_id,
            category=common.category,
            payment_type=common.payment_type,
            origin_balance=common.origin_balance,
            latitude=common.location.latitude if common.location else 0.0,
            longitude=common.location.longitude if common.location else 0.0,
            home_latitude=common.home_location.latitude if common.home_location else None,
            home_longitude=common.home_location.longitude if common.home_location else None,
            device_id=common.device_id,
            auth_verified=common.auth_verified,
            is_fraud=common.is_fraud,
            metadata=common.raw_metadata
        )

def create_sample_transaction(
    transaction_id: str = "tx_sample_001",
    amount: float = 125.50,
    customer_id: str = "cust_sample_42",
    merchant_id: str = "merch_sample_99",
    category: str = "grocery_pos",
    payment_type: str = "CREDIT_CARD",
    timestamp: Optional[float] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    device_id: Optional[str] = None
) -> TransactionEvent:
    """Helper to produce a deterministic, valid TransactionEvent for testing and simulation."""
    return TransactionEvent(
        schema_version="1.0",
        transaction_id=transaction_id,
        timestamp=timestamp if timestamp is not None else 1704067200.0,
        amount=amount,
        customer_id=customer_id,
        merchant_id=merchant_id,
        category=category,
        payment_type=payment_type,
        origin_balance=5000.0,
        latitude=latitude if latitude is not None else 37.7749,
        longitude=longitude if longitude is not None else -122.4194,
        home_latitude=37.7749,
        home_longitude=-122.4194,
        device_id=device_id if device_id is not None else f"pos_term_{customer_id}",
        auth_verified=True,
        is_fraud=0,
        metadata={"source": "sample_generator"}
    )
