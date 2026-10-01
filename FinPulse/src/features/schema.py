"""Pydantic and tabular schema definitions for FinPulse transactions and features."""
from typing import Optional, Dict, Any, List, ClassVar
from pydantic import BaseModel, Field, model_validator
import numpy as np

class LocationSchema(BaseModel):
    """Geographic location coordinates."""
    latitude: float = Field(0.0, description="GPS Latitude (-90.0 to 90.0)")
    longitude: float = Field(0.0, description="GPS Longitude (-180.0 to 180.0)")

class CommonTransactionSchema(BaseModel):
    """
    Canonical Transaction Contract for Ingestion, Dataset Adapters, and API Serving.
    Normalizes concepts from PaySim, Sparkov, and IEEE-CIS while preserving
    source-specific metadata.
    """
    transaction_id: str = Field(..., description="Unique transaction identifier")
    timestamp: float = Field(..., description="Unix epoch timestamp in seconds")
    amount: float = Field(..., gt=0.0, description="Transaction monetary amount")
    customer_id: str = Field(..., description="Customer, cardholder, or sender account identifier")
    merchant_id: str = Field(..., description="Merchant, terminal, or destination identifier")
    category: str = Field("general", description="Merchant category or transaction classification")
    payment_type: str = Field("TRANSFER", description="Payment rail / instrument (TRANSFER, PAYMENT, CREDIT_CARD, etc.)")
    origin_balance: float = Field(0.0, ge=0.0, description="Pre-authorization available balance or card limit")
    location: LocationSchema = Field(default_factory=LocationSchema)
    home_location: Optional[LocationSchema] = Field(default=None, description="Cardholder/account home residence location if known")
    device_id: str = Field("unknown_device", description="Hardware or digital device identifier")
    auth_verified: bool = Field(True, description="Whether 2FA / biometric step-up authentication succeeded")
    is_fraud: Optional[int] = Field(None, description="Ground truth label when available (0: legit, 1: fraud)")
    dataset_source: str = Field("unknown", description="Source dataset origin (paysim, sparkov, ieee_cis, streaming)")
    raw_metadata: Dict[str, Any] = Field(default_factory=dict, description="Raw source-specific attributes preserved without loss")

    @model_validator(mode="before")
    @classmethod
    def populate_locations(cls, data: Any) -> Any:
        if isinstance(data, dict):
            loc = data.get("location")
            if not loc:
                lat = float(data.get("latitude", 0.0))
                lon = float(data.get("longitude", 0.0))
                if lat != 0.0 or lon != 0.0:
                    data["location"] = LocationSchema(latitude=lat, longitude=lon)
            hloc = data.get("home_location")
            if not hloc:
                hlat = data.get("home_latitude")
                hlon = data.get("home_longitude")
                if hlat is not None and hlon is not None:
                    data["home_location"] = LocationSchema(latitude=float(hlat), longitude=float(hlon))
        return data

class FinPulseFeatureVector(BaseModel):
    """
    Unified 32-Feature Output Contract.
    Strictly defined and ordered to guarantee offline-training and online-serving parity.
    """
    FEATURE_NAMES: ClassVar[List[str]] = [
        # Group A: Transaction features (5)
        "amount", "log_amount", "payment_type_enc", "merchant_category_enc", "amount_to_balance_ratio",
        # Group B: Temporal features (6)
        "hour_of_day", "day_of_week", "is_weekend", "is_night", "cyclic_hour_sin", "cyclic_hour_cos",
        # Group C: Velocity features (7)
        "tx_count_1m", "tx_count_5m", "tx_count_15m", "tx_count_1h",
        "amount_sum_5m", "amount_sum_15m", "amount_sum_1h",
        # Group D: Behavioral features (5)
        "user_avg_amount_30d", "user_std_amount_30d", "amount_zscore", "user_category_frequency", "user_hourly_tx_deviation",
        # Group E: Contextual & Location features (6)
        "distance_from_home_km", "distance_from_prev_loc_km", "speed_kmh_from_prev_tx",
        "is_new_device", "is_new_location", "device_user_count_24h",
        # Group F: Anomaly & Rule features (3)
        "auth_factor_verified", "deterministic_rule_count", "isolation_forest_score"
    ]
    model_config = {"arbitrary_types_allowed": True}

    # Group A: Transaction Features (5)
    amount: float
    log_amount: float
    payment_type_enc: int
    merchant_category_enc: int
    amount_to_balance_ratio: float

    # Group B: Temporal Features (6)
    hour_of_day: int
    day_of_week: int
    is_weekend: int
    is_night: int
    cyclic_hour_sin: float
    cyclic_hour_cos: float

    # Group C: Velocity Features (7)
    tx_count_1m: int
    tx_count_5m: int
    tx_count_15m: int
    tx_count_1h: int
    amount_sum_5m: float
    amount_sum_15m: float
    amount_sum_1h: float

    # Group D: Behavioral Features (5)
    user_avg_amount_30d: float
    user_std_amount_30d: float
    amount_zscore: float
    user_category_frequency: float
    user_hourly_tx_deviation: float

    # Group E: Context & Location Features (6)
    distance_from_home_km: float
    distance_from_prev_loc_km: float
    speed_kmh_from_prev_tx: float
    is_new_device: int
    is_new_location: int
    device_user_count_24h: int

    # Group F: Anomaly & Rules (3)
    auth_factor_verified: int
    deterministic_rule_count: int
    isolation_forest_score: float

    def to_ordered_vector(self) -> List[float]:
        """Convert pydantic schema to strict 32-dimension float list for model inference."""
        return [
            float(self.amount),
            float(self.log_amount),
            float(self.payment_type_enc),
            float(self.merchant_category_enc),
            float(self.amount_to_balance_ratio),
            float(self.hour_of_day),
            float(self.day_of_week),
            float(self.is_weekend),
            float(self.is_night),
            float(self.cyclic_hour_sin),
            float(self.cyclic_hour_cos),
            float(self.tx_count_1m),
            float(self.tx_count_5m),
            float(self.tx_count_15m),
            float(self.tx_count_1h),
            float(self.amount_sum_5m),
            float(self.amount_sum_15m),
            float(self.amount_sum_1h),
            float(self.user_avg_amount_30d),
            float(self.user_std_amount_30d),
            float(self.amount_zscore),
            float(self.user_category_frequency),
            float(self.user_hourly_tx_deviation),
            float(self.distance_from_home_km),
            float(self.distance_from_prev_loc_km),
            float(self.speed_kmh_from_prev_tx),
            float(self.is_new_device),
            float(self.is_new_location),
            float(self.device_user_count_24h),
            float(self.auth_factor_verified),
            float(self.deterministic_rule_count),
            float(self.isolation_forest_score)
        ]

    def to_dict(self) -> Dict[str, float]:
        """Convert features to dictionary keyed by canonical feature names."""
        vec = self.to_ordered_vector()
        return dict(zip(self.FEATURE_NAMES, vec))

from dataclasses import dataclass, field, asdict

@dataclass
class FeatureDefinition:
    """Authoritative metadata specification for a single feature."""
    name: str
    index: int
    group: str
    expected_type: str
    calculation_definition: str
    cold_start_default: float

FEATURE_SPECIFICATIONS: List[FeatureDefinition] = [
    FeatureDefinition("amount", 0, "group_a_transaction", "float", "Raw transaction monetary amount in dollars", 0.0),
    FeatureDefinition("log_amount", 1, "group_a_transaction", "float", "Natural logarithm log1p(max(amount, 0.0))", 0.0),
    FeatureDefinition("payment_type_enc", 2, "group_a_transaction", "int", "Categorical integer ordinal encoding of payment method", 6.0),
    FeatureDefinition("merchant_category_enc", 3, "group_a_transaction", "int", "Target frequency or hash encoding of merchant category", 0.0),
    FeatureDefinition("amount_to_balance_ratio", 4, "group_a_transaction", "float", "amount / (origin_balance + 1e-5)", 0.0),
    FeatureDefinition("hour_of_day", 5, "group_b_temporal", "int", "UTC hour of transaction (0-23)", 12.0),
    FeatureDefinition("day_of_week", 6, "group_b_temporal", "int", "Day of week (0=Monday, 6=Sunday)", 3.0),
    FeatureDefinition("is_weekend", 7, "group_b_temporal", "int", "Binary flag: 1 if day_of_week >= 5 else 0", 0.0),
    FeatureDefinition("is_night", 8, "group_b_temporal", "int", "Binary flag: 1 if hour < 6 or hour > 22 else 0", 0.0),
    FeatureDefinition("cyclic_hour_sin", 9, "group_b_temporal", "float", "sin(2 * pi * hour / 24.0)", 0.0),
    FeatureDefinition("cyclic_hour_cos", 10, "group_b_temporal", "float", "cos(2 * pi * hour / 24.0)", 1.0),
    FeatureDefinition("tx_count_1m", 11, "group_c_velocity", "int", "Transaction count in prior 1 minute [t - 60s, t)", 0.0),
    FeatureDefinition("tx_count_5m", 12, "group_c_velocity", "int", "Transaction count in prior 5 minutes [t - 300s, t)", 0.0),
    FeatureDefinition("tx_count_15m", 13, "group_c_velocity", "int", "Transaction count in prior 15 minutes [t - 900s, t)", 0.0),
    FeatureDefinition("tx_count_1h", 14, "group_c_velocity", "int", "Transaction count in prior 1 hour [t - 3600s, t)", 0.0),
    FeatureDefinition("amount_sum_5m", 15, "group_c_velocity", "float", "Total monetary spend in prior 5 minutes [t - 300s, t)", 0.0),
    FeatureDefinition("amount_sum_15m", 16, "group_c_velocity", "float", "Total monetary spend in prior 15 minutes [t - 900s, t)", 0.0),
    FeatureDefinition("amount_sum_1h", 17, "group_c_velocity", "float", "Total monetary spend in prior 1 hour [t - 3600s, t)", 0.0),
    FeatureDefinition("user_avg_amount_30d", 18, "group_d_behavioral", "float", "Customer mean transaction amount over prior 30 days (default = current amount)", 0.0),
    FeatureDefinition("user_std_amount_30d", 19, "group_d_behavioral", "float", "Customer sample standard deviation over prior 30 days (ddof=1)", 0.0),
    FeatureDefinition("amount_zscore", 20, "group_d_behavioral", "float", "(amount - mean) / max(std, 1e-4), clipped [-5.0, 10.0]", 0.0),
    FeatureDefinition("user_category_frequency", 21, "group_d_behavioral", "float", "Frequency of current category in prior 30-day transactions", 1.0),
    FeatureDefinition("user_hourly_tx_deviation", 22, "group_d_behavioral", "float", "Absolute circular deviation between current hour and historical mean hour", 0.0),
    FeatureDefinition("distance_from_home_km", 23, "group_e_context_location", "float", "Haversine distance in km from account registered home coordinates", 0.0),
    FeatureDefinition("distance_from_prev_loc_km", 24, "group_e_context_location", "float", "Haversine distance in km from immediately preceding transaction location", 0.0),
    FeatureDefinition("speed_kmh_from_prev_tx", 25, "group_e_context_location", "float", "Travel speed in km/h: distance_from_prev_loc_km / (time_diff_hours)", 0.0),
    FeatureDefinition("is_new_device", 26, "group_e_context_location", "int", "1 if device not previously seen for this customer, else 0 (cold start = 0)", 0.0),
    FeatureDefinition("is_new_location", 27, "group_e_context_location", "int", "1 if >50km from all prior locations for customer, else 0 (cold start = 0)", 0.0),
    FeatureDefinition("device_user_count_24h", 28, "group_e_context_location", "int", "Unique customer count transacting on this device in prior 24 hours", 1.0),
    FeatureDefinition("auth_factor_verified", 29, "group_f_anomaly_rule", "int", "1 if multi-factor or biometric authentication was verified, else 0", 1.0),
    FeatureDefinition("deterministic_rule_count", 30, "group_f_anomaly_rule", "int", "Count of triggered deterministic risk rules", 0.0),
    FeatureDefinition("isolation_forest_score", 31, "group_f_anomaly_rule", "float", "Unsupervised anomaly score from Isolation Forest", 0.10)
]

def validate_feature_vector(
    vector: Any,
    expected_dim: int = 32
) -> np.ndarray:
    """
    Strict numerical and schema validation for FinPulse feature vectors.
    Enforces:
    1. Dimension is strictly equal to 32 (rejects 31, 33, etc.)
    2. Zero NaN values
    3. Zero Inf / -Inf values
    4. Numeric float32 type
    Returns validated np.ndarray of shape (32,).
    """
    if isinstance(vector, (list, tuple)):
        arr = np.array(vector, dtype=np.float32)
    elif isinstance(vector, np.ndarray):
        arr = vector.astype(np.float32)
    elif hasattr(vector, "to_ordered_vector"):
        arr = np.array(vector.to_ordered_vector(), dtype=np.float32)
    else:
        raise TypeError(f"Unsupported feature vector type: {type(vector)}")

    if arr.ndim == 2:
        if arr.shape[0] == 1:
            arr = arr.flatten()
        elif arr.shape[1] == expected_dim:
            if np.isnan(arr).any():
                raise ValueError("Feature matrix contains NaN values")
            if np.isinf(arr).any():
                raise ValueError("Feature matrix contains Infinite values")
            return arr

    if arr.shape != (expected_dim,):
        raise ValueError(
            f"Invalid feature dimension: expected ({expected_dim},), got {arr.shape}. "
            "Feature vector must contain strictly 32 elements."
        )

    if np.isnan(arr).any():
        nan_indices = np.where(np.isnan(arr))[0].tolist()
        raise ValueError(f"Feature vector contains NaN at indices: {nan_indices}")

    if np.isinf(arr).any():
        inf_indices = np.where(np.isinf(arr))[0].tolist()
        raise ValueError(f"Feature vector contains Infinite values at indices: {inf_indices}")

    return arr

@dataclass
class FeatureExtractionResult:
    """
    Standardized R3 output contract emitted by the live feature engine.
    Wraps the validated 32-feature vector with transaction identifiers for downstream consumption.
    """
    transaction_id: str
    customer_id: str
    timestamp: float
    features: List[float]
    feature_names: List[str] = field(default_factory=lambda: list(FinPulseFeatureVector.FEATURE_NAMES))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate 32 dimensions on construction
        validate_feature_vector(self.features, expected_dim=32)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to standard R3 output JSON structure."""
        return {
            "transaction_id": self.transaction_id,
            "customer_id": self.customer_id,
            "timestamp": self.timestamp,
            "features": self.features,
            "feature_count": len(self.features)
        }

    def to_numpy(self) -> np.ndarray:
        return np.array(self.features, dtype=np.float32)

