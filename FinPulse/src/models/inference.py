"""Production Model Inference and Real-Time Decisioning Service (R4 Foundation).

Enforces:
1. Production bundle integrity verification against SHA-256 manifest before loading.
2. Independent model-boundary 32-feature schema validation (0 NaNs, 0 Infs, strict ordering).
3. Frozen CatBoost model inference producing deterministic raw fraud probability.
4. Frozen Platt calibrator producing well-calibrated posterior fraud probability.
5. Frozen ML threshold policy mapping calibrated probability to APPROVE / REVIEW / BLOCK.
6. Safe failure behavior with zero silent degradation or fabricated predictions.
"""
import os
import sys
import json
import time
import math
import hashlib
import joblib
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union, Literal

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.features.schema import FinPulseFeatureVector, FeatureExtractionResult

class ModelBundleIntegrityError(Exception):
    """Raised when production bundle artifacts are missing, unreadable, or fail checksum validation."""
    pass

class FeatureContractError(Exception):
    """Raised when an incoming feature vector violates the production 32-feature contract."""
    pass

class InferenceExecutionError(Exception):
    """Raised when model inference, probability calibration, or policy evaluation fails."""
    pass

@dataclass
class InferenceResult:
    """Canonical R4 output contract emitted upon successful model scoring and policy evaluation."""
    transaction_id: str
    model_version: str
    feature_schema_version: str
    raw_probability: float
    calibrated_probability: float
    decision: Literal["APPROVE", "REVIEW", "BLOCK"]
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to standard R4 JSON contract."""
        res = {
            "transaction_id": self.transaction_id,
            "model_version": self.model_version,
            "feature_schema_version": self.feature_schema_version,
            "raw_probability": round(float(self.raw_probability), 6),
            "calibrated_probability": round(float(self.calibrated_probability), 6),
            "decision": self.decision
        }
        if self.latency_ms is not None:
            res["latency_ms"] = round(float(self.latency_ms), 3)
        return res

class ProductionModelService:
    """
    R4 Production Model Service:
    Loads frozen finpulse-v3 bundle, executes inference, applies Platt calibration,
    and assigns decisions using the configured threshold policy.
    """

    DEFAULT_BUNDLE_DIR = os.path.join(FINPULSE_DIR, "models", "production", "finpulse-v3")

    def __init__(self, bundle_dir: Optional[str] = None, enforce_checksum: bool = True):
        self.bundle_dir = os.path.abspath(bundle_dir or self.DEFAULT_BUNDLE_DIR)
        self.enforce_checksum = enforce_checksum
        
        self.model = None
        self.calibrator = None
        self.feature_schema = None
        self.policy_data = None
        self.metadata = None
        
        self.model_version = "finpulse-v3"
        self.feature_schema_version = "2.0"
        self.feature_names: List[str] = []
        self.tau_review: float = 0.1580
        self.tau_block: float = 0.5516

        self._load_and_verify_bundle()

    def _verify_checksums(self) -> Dict[str, str]:
        """Verify SHA-256 checksums of all production bundle artifacts."""
        checksum_file = os.path.join(self.bundle_dir, "checksum.sha256")
        if not os.path.exists(checksum_file):
            raise ModelBundleIntegrityError(f"Checksum manifest missing at: {checksum_file}")

        try:
            with open(checksum_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            raise ModelBundleIntegrityError(f"Failed to read checksum manifest: {e}")

        verified = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            expected_sha, fname = parts[0].strip(), parts[1].strip()
            fpath = os.path.join(self.bundle_dir, fname)

            if not os.path.exists(fpath):
                raise ModelBundleIntegrityError(f"Production artifact missing: {fpath}")

            hasher = hashlib.sha256()
            try:
                with open(fpath, "rb") as af:
                    while chunk := af.read(65536):
                        hasher.update(chunk)
            except Exception as e:
                raise ModelBundleIntegrityError(f"Error reading artifact {fname}: {e}")

            actual_sha = hasher.hexdigest()
            if actual_sha != expected_sha:
                raise ModelBundleIntegrityError(
                    f"Checksum mismatch for artifact '{fname}': expected {expected_sha}, got {actual_sha}"
                )
            verified[fname] = actual_sha

        return verified

    def _load_and_verify_bundle(self):
        """Validate integrity and deserialize frozen artifacts into memory."""
        if not os.path.exists(self.bundle_dir):
            raise ModelBundleIntegrityError(f"Production bundle directory does not exist: {self.bundle_dir}")

        # 1. Verify Checksum Manifest
        if self.enforce_checksum:
            self._verify_checksums()

        # 2. Load Feature Schema
        schema_path = os.path.join(self.bundle_dir, "feature_schema.json")
        if not os.path.exists(schema_path):
            raise ModelBundleIntegrityError(f"feature_schema.json missing at: {schema_path}")
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                self.feature_schema = json.load(f)
            self.feature_schema_version = str(self.feature_schema.get("feature_schema_version", "2.0"))
            self.feature_names = list(self.feature_schema.get("feature_names", []))
            if len(self.feature_names) != 32:
                raise FeatureContractError(
                    f"Feature schema must define strictly 32 features, found {len(self.feature_names)}"
                )
        except Exception as e:
            if isinstance(e, (ModelBundleIntegrityError, FeatureContractError)):
                raise
            raise ModelBundleIntegrityError(f"Failed to parse feature_schema.json: {e}")

        # 3. Load Threshold Policy
        policy_path = os.path.join(self.bundle_dir, "threshold_policy.json")
        if not os.path.exists(policy_path):
            raise ModelBundleIntegrityError(f"threshold_policy.json missing at: {policy_path}")
        try:
            with open(policy_path, "r", encoding="utf-8") as f:
                self.policy_data = json.load(f)
            
            tau_r = self.policy_data.get("tau_review", self.policy_data.get("review_threshold"))
            tau_b = self.policy_data.get("tau_block", self.policy_data.get("block_threshold"))
            
            if tau_r is None or tau_b is None:
                raise ModelBundleIntegrityError("threshold_policy.json missing required review or block threshold keys")
            
            self.tau_review = float(tau_r)
            self.tau_block = float(tau_b)
            if self.tau_review >= self.tau_block:
                raise ModelBundleIntegrityError(
                    f"Invalid threshold policy: tau_review ({self.tau_review}) >= tau_block ({self.tau_block})"
                )
        except Exception as e:
            if isinstance(e, ModelBundleIntegrityError):
                raise
            raise ModelBundleIntegrityError(f"Failed to parse threshold_policy.json: {e}")

        # 4. Load Model Metadata
        meta_path = os.path.join(self.bundle_dir, "model_metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                self.model_version = str(self.metadata.get("model_version", self.model_version))
            except Exception:
                pass

        # 5. Load Frozen CatBoost Model
        model_path = os.path.join(self.bundle_dir, "model.pkl")
        if not os.path.exists(model_path):
            raise ModelBundleIntegrityError(f"model.pkl missing at: {model_path}")
        try:
            self.model = joblib.load(model_path)
            if not hasattr(self.model, "predict_proba"):
                raise ModelBundleIntegrityError("Loaded model does not implement predict_proba")
        except Exception as e:
            if isinstance(e, ModelBundleIntegrityError):
                raise
            raise ModelBundleIntegrityError(f"Failed to deserialize model.pkl: {e}")

        # 6. Load Frozen Platt Calibrator
        cal_path = os.path.join(self.bundle_dir, "calibrator.pkl")
        if not os.path.exists(cal_path):
            raise ModelBundleIntegrityError(f"calibrator.pkl missing at: {cal_path}")
        try:
            self.calibrator = joblib.load(cal_path)
            if not hasattr(self.calibrator, "predict"):
                raise ModelBundleIntegrityError("Loaded calibrator does not implement predict")
        except Exception as e:
            if isinstance(e, ModelBundleIntegrityError):
                raise
            raise ModelBundleIntegrityError(f"Failed to deserialize calibrator.pkl: {e}")

    def validate_features(
        self,
        features: Union[List[float], np.ndarray, Dict[str, float], FeatureExtractionResult, FinPulseFeatureVector]
    ) -> np.ndarray:
        """
        Independently validate incoming feature vector against the production contract.
        Enforces:
        - strictly 32 features
        - correct feature names and order if provided as dict
        - no NaN values
        - no Infinite values
        - valid float32 conversion
        Returns 2D np.ndarray of shape (1, 32).
        """
        if isinstance(features, FeatureExtractionResult):
            arr = np.array(features.features, dtype=np.float32)
        elif isinstance(features, FinPulseFeatureVector):
            arr = np.array(features.to_ordered_vector(), dtype=np.float32)
        elif isinstance(features, dict):
            # Check keys match production feature names
            missing_keys = [k for k in self.feature_names if k not in features]
            if missing_keys:
                raise FeatureContractError(f"Missing required production features in dict: {missing_keys}")
            unexpected_keys = [k for k in features if k not in self.feature_names]
            if unexpected_keys:
                raise FeatureContractError(f"Unexpected features found in dict: {unexpected_keys}")
            ordered_vals = [float(features[k]) for k in self.feature_names]
            arr = np.array(ordered_vals, dtype=np.float32)
        elif isinstance(features, (list, tuple)):
            arr = np.array(features, dtype=np.float32)
        elif isinstance(features, np.ndarray):
            arr = features.astype(np.float32)
        else:
            raise FeatureContractError(f"Unsupported feature type: {type(features)}")

        # Reshape & validate dimensions
        if arr.ndim == 1:
            if arr.shape[0] != 32:
                raise FeatureContractError(
                    f"Feature vector length must be strictly 32, got {arr.shape[0]}"
                )
            arr = arr.reshape(1, 32)
        elif arr.ndim == 2:
            if arr.shape[1] != 32:
                raise FeatureContractError(
                    f"Feature matrix column count must be strictly 32, got {arr.shape[1]}"
                )
            if arr.shape[0] != 1:
                arr = arr[:1]  # Take single transaction for real-time scoring
        else:
            raise FeatureContractError(f"Feature array must be 1D or 2D, got shape {arr.shape}")

        if np.isnan(arr).any():
            nan_indices = np.where(np.isnan(arr[0]))[0].tolist()
            nan_features = [self.feature_names[i] for i in nan_indices if i < len(self.feature_names)]
            raise FeatureContractError(
                f"Feature vector contains NaN at indices {nan_indices} (features: {nan_features})"
            )

        if np.isinf(arr).any():
            inf_indices = np.where(np.isinf(arr[0]))[0].tolist()
            inf_features = [self.feature_names[i] for i in inf_indices if i < len(self.feature_names)]
            raise FeatureContractError(
                f"Feature vector contains Infinite values at indices {inf_indices} (features: {inf_features})"
            )

        return arr

    def predict_raw(self, X: np.ndarray) -> float:
        """Run frozen CatBoost model to obtain raw probability of fraud."""
        try:
            raw_output = self.model.predict_proba(X)
        except Exception as e:
            raise InferenceExecutionError(f"CatBoost model predict_proba failed: {e}")

        if isinstance(raw_output, np.ndarray):
            if raw_output.ndim == 2:
                raw_prob = float(raw_output[0, 1])
            else:
                raw_prob = float(raw_output[0])
        else:
            raw_prob = float(raw_output)

        if math.isnan(raw_prob) or math.isinf(raw_prob) or not (0.0 <= raw_prob <= 1.0):
            raise InferenceExecutionError(f"Invalid raw probability generated: {raw_prob}")

        return raw_prob

    def calibrate(self, raw_prob: float) -> float:
        """Transform raw probability into calibrated probability using frozen Platt calibrator."""
        if math.isnan(raw_prob) or math.isinf(raw_prob) or not (0.0 <= raw_prob <= 1.0):
            raise InferenceExecutionError(f"Cannot calibrate invalid raw probability: {raw_prob}")

        try:
            cal_input = np.array([raw_prob], dtype=np.float64)
            cal_output = self.calibrator.predict(cal_input)
            cal_prob = float(cal_output[0]) if hasattr(cal_output, "__getitem__") else float(cal_output)
        except Exception as e:
            raise InferenceExecutionError(f"Platt calibrator failed: {e}")

        if math.isnan(cal_prob) or math.isinf(cal_prob) or not (0.0 <= cal_prob <= 1.0):
            raise InferenceExecutionError(f"Invalid calibrated probability generated: {cal_prob}")

        return cal_prob

    def apply_decision_policy(self, calibrated_prob: float) -> Literal["APPROVE", "REVIEW", "BLOCK"]:
        """
        Evaluate calibrated probability against frozen ML threshold policy:
        P(fraud) < tau_review -> APPROVE
        tau_review <= P(fraud) < tau_block -> REVIEW
        P(fraud) >= tau_block -> BLOCK
        """
        if math.isnan(calibrated_prob) or math.isinf(calibrated_prob) or not (0.0 <= calibrated_prob <= 1.0):
            raise InferenceExecutionError(f"Cannot apply policy to invalid probability: {calibrated_prob}")

        if calibrated_prob < self.tau_review:
            return "APPROVE"
        elif calibrated_prob < self.tau_block:
            return "REVIEW"
        else:
            return "BLOCK"

    def score_features(
        self,
        features: Union[List[float], np.ndarray, Dict[str, float], FeatureExtractionResult, FinPulseFeatureVector],
        transaction_id: str = "unknown"
    ) -> InferenceResult:
        """
        Execute complete R4 inference flow on pre-computed feature representation:
        Validate 32 Features -> Frozen CatBoost -> Raw P -> Platt Calibrator -> Calibrated P -> ML Policy.
        """
        t0 = time.perf_counter()

        # 1. Production feature contract validation
        X = self.validate_features(features)

        # 2. CatBoost inference
        raw_p = self.predict_raw(X)

        # 3. Platt calibration
        cal_p = self.calibrate(raw_p)

        # 4. Frozen ML decision policy
        decision = self.apply_decision_policy(cal_p)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return InferenceResult(
            transaction_id=str(transaction_id),
            model_version=self.model_version,
            feature_schema_version=self.feature_schema_version,
            raw_probability=raw_p,
            calibrated_probability=cal_p,
            decision=decision,
            latency_ms=latency_ms
        )
