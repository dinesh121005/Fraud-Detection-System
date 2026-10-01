"""Comprehensive Test Suite for R4 Real-Time Model Inference & Decisioning.

Validates:
1. R4.1 Production Bundle Integrity & Checksum Verification (tamper detection, missing artifacts).
2. R4.2 Model Boundary Feature Contract (32 dimensions, exact order, NaN/Inf rejection, type checking).
3. R4.3 Frozen CatBoost Model Inference (deterministic raw fraud probabilities, bounds [0, 1]).
4. R4.4 Frozen Platt Calibrator Transformation (posterior calibration, bounds [0, 1]).
5. R4.5 Frozen ML Decision Policy Boundaries (0.1579->APPROVE, 0.1580->REVIEW, 0.5515->REVIEW, 0.5516->BLOCK).
6. Comprehensive Error Handling (corrupted artifacts, malformed schemas, safe failures).
7. End-to-End Streaming Pipeline Integration (Kafka -> Redis -> R3 Feature Engine -> CatBoost -> Platt -> Policy).
"""
import os
import sys
import json
import time
import math
import shutil
import tempfile
import pytest
import numpy as np

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.models.inference import (
    ProductionModelService,
    InferenceResult,
    ModelBundleIntegrityError,
    FeatureContractError,
    InferenceExecutionError,
)
from src.features.schema import FinPulseFeatureVector, FeatureExtractionResult
from src.features.engine import FinPulseFeatureEngine
from src.state.manager import RedisStateManager
from src.state.redis_client import InMemoryRedisMock
from src.streaming.schema import TransactionEvent, create_sample_transaction
from src.streaming.consumer import TransactionConsumer
from src.streaming.config import StreamingConfig

PRODUCTION_BUNDLE_DIR = os.path.join(FINPULSE_DIR, "models", "production", "finpulse-v3")

@pytest.fixture
def model_service():
    """Provides an initialized ProductionModelService pointing to the registered production bundle."""
    return ProductionModelService(bundle_dir=PRODUCTION_BUNDLE_DIR, enforce_checksum=True)

@pytest.fixture
def mock_redis():
    """Provides an isolated InMemoryRedisMock."""
    client = InMemoryRedisMock()
    yield client
    client.flushall()

@pytest.fixture
def state_manager(mock_redis):
    """Provides a RedisStateManager backed by mock_redis."""
    return RedisStateManager(redis_client=mock_redis)

@pytest.fixture
def feature_engine(state_manager):
    """Provides an R3 Feature Engine backed by state_manager."""
    return FinPulseFeatureEngine(state_manager=state_manager)

# ==============================================================================
# R4.1 — Production Bundle Integrity & Checksum Verification
# ==============================================================================

def test_production_bundle_exists_and_verified(model_service):
    """Verifies that the frozen finpulse-v3 bundle exists, passes checksums, and loads successfully."""
    assert os.path.isdir(PRODUCTION_BUNDLE_DIR)
    assert model_service.model is not None
    assert model_service.calibrator is not None
    assert model_service.feature_schema is not None
    assert model_service.policy_data is not None
    assert model_service.model_version == "finpulse-v3"
    assert model_service.feature_schema_version == "2.0"
    assert len(model_service.feature_names) == 32
    assert model_service.tau_review == pytest.approx(0.1580, abs=1e-5)
    assert model_service.tau_block == pytest.approx(0.5516, abs=1e-5)

def test_production_checksum_manifest_verification():
    """Verifies that all 5 artifact files in the bundle match their SHA-256 hashes."""
    import hashlib
    manifest_path = os.path.join(PRODUCTION_BUNDLE_DIR, "checksum.sha256")
    assert os.path.exists(manifest_path)

    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    artifact_count = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        expected_sha, fname = line.split(None, 1)
        fpath = os.path.join(PRODUCTION_BUNDLE_DIR, fname)
        assert os.path.exists(fpath), f"Artifact missing: {fname}"

        hasher = hashlib.sha256()
        with open(fpath, "rb") as af:
            while chunk := af.read(65536):
                hasher.update(chunk)
        assert hasher.hexdigest() == expected_sha, f"Checksum mismatch on {fname}"
        artifact_count += 1

    assert artifact_count >= 5

def test_checksum_tamper_detection_fails_safely():
    """Verifies that tampering with any bundle file immediately halts loading with ModelBundleIntegrityError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy production bundle into temp directory
        for fname in os.listdir(PRODUCTION_BUNDLE_DIR):
            src_file = os.path.join(PRODUCTION_BUNDLE_DIR, fname)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, os.path.join(tmpdir, fname))

        # Tamper with model.pkl
        tampered_model = os.path.join(tmpdir, "model.pkl")
        with open(tampered_model, "ab") as f:
            f.write(b"CORRUPTED_BYTES")

        with pytest.raises(ModelBundleIntegrityError) as exc_info:
            ProductionModelService(bundle_dir=tmpdir, enforce_checksum=True)
        assert "Checksum mismatch" in str(exc_info.value)

def test_missing_checksum_manifest_fails_safely():
    """Verifies that missing checksum.sha256 raises ModelBundleIntegrityError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy production bundle except checksum.sha256
        for fname in os.listdir(PRODUCTION_BUNDLE_DIR):
            if fname != "checksum.sha256":
                src_file = os.path.join(PRODUCTION_BUNDLE_DIR, fname)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, os.path.join(tmpdir, fname))

        with pytest.raises(ModelBundleIntegrityError) as exc_info:
            ProductionModelService(bundle_dir=tmpdir, enforce_checksum=True)
        assert "Checksum manifest missing" in str(exc_info.value)

def test_missing_artifact_file_fails_safely():
    """Verifies that a missing artifact file raises ModelBundleIntegrityError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for fname in os.listdir(PRODUCTION_BUNDLE_DIR):
            if fname != "calibrator.pkl":
                src_file = os.path.join(PRODUCTION_BUNDLE_DIR, fname)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, os.path.join(tmpdir, fname))

        with pytest.raises(ModelBundleIntegrityError) as exc_info:
            ProductionModelService(bundle_dir=tmpdir, enforce_checksum=True)
        assert "Production artifact missing" in str(exc_info.value)

# ==============================================================================
# R4.2 — Production Feature Contract Validation
# ==============================================================================

def test_validate_features_valid_array(model_service):
    """Verifies that a valid 32-element list or 1D/2D array is normalized to (1, 32) float32."""
    valid_vec = [0.0] * 32
    valid_vec[0] = 50.0  # amount
    valid_vec[1] = 3.91  # amount_log

    X = model_service.validate_features(valid_vec)
    assert isinstance(X, np.ndarray)
    assert X.shape == (1, 32)
    assert X.dtype == np.float32
    assert X[0, 0] == pytest.approx(50.0)

def test_validate_features_valid_dict(model_service):
    """Verifies that a dict containing all 32 production feature names is converted with exact schema order."""
    feature_dict = {name: float(i * 1.5) for i, name in enumerate(model_service.feature_names)}
    X = model_service.validate_features(feature_dict)

    assert X.shape == (1, 32)
    for i, name in enumerate(model_service.feature_names):
        assert X[0, i] == pytest.approx(float(i * 1.5), abs=1e-5)

def test_validate_features_finpulse_feature_vector_object(model_service, feature_engine):
    """Verifies that FinPulseFeatureVector dataclass is accepted seamlessly."""
    tx = create_sample_transaction(
        transaction_id="tx_vec_test",
        amount=120.0,
        customer_id="cust_vec"
    )
    vec = feature_engine.compute_features(tx)
    assert isinstance(vec, FinPulseFeatureVector)
    X = model_service.validate_features(vec)
    assert X.shape == (1, 32)
    assert X[0, 0] == pytest.approx(120.0)
    assert X[0, 1] == pytest.approx(math.log(121.0), abs=1e-4)

def test_validate_features_rejects_31_features(model_service):
    """Hard Requirement: Under-length feature vector (31 features) must fail safely."""
    invalid_31 = [0.0] * 31
    with pytest.raises(FeatureContractError) as exc:
        model_service.validate_features(invalid_31)
    assert "must be strictly 32" in str(exc.value)

def test_validate_features_rejects_33_features(model_service):
    """Hard Requirement: Over-length feature vector (33 features) must fail safely."""
    invalid_33 = [0.0] * 33
    with pytest.raises(FeatureContractError) as exc:
        model_service.validate_features(invalid_33)
    assert "must be strictly 32" in str(exc.value)

def test_validate_features_rejects_nan(model_service):
    """Hard Requirement: Feature vector containing NaN values must fail safely."""
    vec_with_nan = [0.0] * 32
    vec_with_nan[14] = float("nan")

    with pytest.raises(FeatureContractError) as exc:
        model_service.validate_features(vec_with_nan)
    assert "contains NaN" in str(exc.value)

def test_validate_features_rejects_inf(model_service):
    """Hard Requirement: Feature vector containing Inf values must fail safely."""
    vec_with_inf = [0.0] * 32
    vec_with_inf[25] = float("inf")

    with pytest.raises(FeatureContractError) as exc:
        model_service.validate_features(vec_with_inf)
    assert "contains Infinite" in str(exc.value)

def test_validate_features_rejects_missing_keys_in_dict(model_service):
    """Hard Requirement: Dict missing required feature keys must fail safely."""
    partial_dict = {name: 1.0 for name in model_service.feature_names[:30]}  # Missing 2 keys
    with pytest.raises(FeatureContractError) as exc:
        model_service.validate_features(partial_dict)
    assert "Missing required production features" in str(exc.value)

def test_validate_features_rejects_unexpected_keys_in_dict(model_service):
    """Hard Requirement: Dict with extraneous keys must fail safely."""
    dict_with_extra = {name: 1.0 for name in model_service.feature_names}
    dict_with_extra["unauthorized_bonus_feature"] = 999.0
    with pytest.raises(FeatureContractError) as exc:
        model_service.validate_features(dict_with_extra)
    assert "Unexpected features found" in str(exc.value)

# ==============================================================================
# R4.3 — CatBoost Model Inference
# ==============================================================================

def test_catboost_inference_deterministic(model_service):
    """
    Verifies that the frozen CatBoost model produces deterministic, finite raw probabilities
    in [0, 1] for identical inputs.
    """
    vec = [0.0] * 32
    vec[0] = 75.0
    vec[1] = math.log(76.0)

    X = model_service.validate_features(vec)
    raw_p1 = model_service.predict_raw(X)
    raw_p2 = model_service.predict_raw(X)

    assert isinstance(raw_p1, float)
    assert raw_p1 == pytest.approx(raw_p2, abs=1e-9)
    assert not math.isnan(raw_p1)
    assert not math.isinf(raw_p1)
    assert 0.0 <= raw_p1 <= 1.0

def test_catboost_distinct_inputs_produce_distinct_probabilities(model_service):
    """Verifies that varied inputs produce expected varying fraud risk probabilities."""
    # Low-risk baseline: small amount, normal velocity
    low_risk = [0.0] * 32
    low_risk[0] = 12.0
    low_risk[1] = math.log(13.0)

    # High-risk profile: huge amount, high velocity, speed anomaly
    high_risk = [0.0] * 32
    high_risk[0] = 9500.0
    high_risk[1] = math.log(9501.0)
    high_risk[14] = 25.0  # tx_count_1h
    high_risk[15] = 15000.0  # amount_sum_1h
    high_risk[25] = 850.0  # speed_kmh

    X_low = model_service.validate_features(low_risk)
    X_high = model_service.validate_features(high_risk)

    p_low = model_service.predict_raw(X_low)
    p_high = model_service.predict_raw(X_high)

    assert 0.0 <= p_low <= 1.0
    assert 0.0 <= p_high <= 1.0
    assert p_high > p_low

# ==============================================================================
# R4.4 — Frozen Platt Calibration
# ==============================================================================

def test_platt_calibration_bounds_and_validity(model_service):
    """
    Verifies that passing raw probabilities through the frozen Platt calibrator
    yields valid posterior probabilities strictly within [0, 1].
    """
    test_raw_probs = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
    for raw_p in test_raw_probs:
        cal_p = model_service.calibrate(raw_p)
        assert isinstance(cal_p, float)
        assert not math.isnan(cal_p)
        assert not math.isinf(cal_p)
        assert 0.0 <= cal_p <= 1.0

def test_platt_calibration_monotonicity(model_service):
    """Verifies that Platt calibration preserves monotonic probability ordering."""
    p_lower = 0.20
    p_higher = 0.80

    cal_lower = model_service.calibrate(p_lower)
    cal_higher = model_service.calibrate(p_higher)

    assert cal_higher > cal_lower

def test_platt_calibration_rejects_invalid_inputs(model_service):
    """Verifies that invalid probability inputs to calibrator raise InferenceExecutionError."""
    with pytest.raises(InferenceExecutionError):
        model_service.calibrate(-0.1)

    with pytest.raises(InferenceExecutionError):
        model_service.calibrate(1.1)

    with pytest.raises(InferenceExecutionError):
        model_service.calibrate(float("nan"))

# ==============================================================================
# R4.5 — Frozen ML Decision Policy Boundaries
# ==============================================================================

def test_ml_decision_policy_exact_boundaries(model_service):
    """
    Explicitly tests the frozen ML decision boundaries:
    0.1579 -> APPROVE
    0.1580 -> REVIEW
    0.5515 -> REVIEW
    0.5516 -> BLOCK
    """
    # Just below tau_review (0.1580)
    assert model_service.apply_decision_policy(0.1579) == "APPROVE"
    assert model_service.apply_decision_policy(0.157999) == "APPROVE"

    # Exactly at tau_review
    assert model_service.apply_decision_policy(0.1580) == "REVIEW"
    assert model_service.apply_decision_policy(0.158001) == "REVIEW"

    # Just below tau_block (0.5516)
    assert model_service.apply_decision_policy(0.5515) == "REVIEW"
    assert model_service.apply_decision_policy(0.551599) == "REVIEW"

    # Exactly at tau_block
    assert model_service.apply_decision_policy(0.5516) == "BLOCK"
    assert model_service.apply_decision_policy(0.551601) == "BLOCK"

    # Extreme bounds
    assert model_service.apply_decision_policy(0.0) == "APPROVE"
    assert model_service.apply_decision_policy(1.0) == "BLOCK"

def test_ml_decision_policy_rejects_invalid_values(model_service):
    """Verifies that applying decision policy on out-of-range or NaN values fails safely."""
    with pytest.raises(InferenceExecutionError):
        model_service.apply_decision_policy(-0.01)

    with pytest.raises(InferenceExecutionError):
        model_service.apply_decision_policy(1.01)

    with pytest.raises(InferenceExecutionError):
        model_service.apply_decision_policy(float("nan"))

# ==============================================================================
# R4.6 — Output Contract & End-to-End Model Scoring
# ==============================================================================

def test_score_features_canonical_contract(model_service):
    """Verifies that score_features produces a compliant InferenceResult matching the R4 contract."""
    vec = [0.0] * 32
    res = model_service.score_features(vec, transaction_id="tx_test_001")

    assert isinstance(res, InferenceResult)
    assert res.transaction_id == "tx_test_001"
    assert res.model_version == "finpulse-v3"
    assert res.feature_schema_version == "2.0"
    assert isinstance(res.raw_probability, float)
    assert isinstance(res.calibrated_probability, float)
    assert res.decision in ("APPROVE", "REVIEW", "BLOCK")
    assert res.latency_ms is not None and res.latency_ms > 0

    contract_dict = res.to_dict()
    assert set(contract_dict.keys()) >= {
        "transaction_id",
        "model_version",
        "feature_schema_version",
        "raw_probability",
        "calibrated_probability",
        "decision"
    }

def test_malformed_threshold_policy_fails_safely():
    """Verifies that an inverted policy (tau_review >= tau_block) raises ModelBundleIntegrityError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for fname in os.listdir(PRODUCTION_BUNDLE_DIR):
            src_file = os.path.join(PRODUCTION_BUNDLE_DIR, fname)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, os.path.join(tmpdir, fname))

        # Invert policy thresholds
        invalid_policy = {"tau_review": 0.80, "tau_block": 0.20}
        with open(os.path.join(tmpdir, "threshold_policy.json"), "w") as pf:
            json.dump(invalid_policy, pf)

        # Update checksum manifest for temp dir so it passes checksum check but fails schema validation
        import hashlib
        hasher = hashlib.sha256()
        with open(os.path.join(tmpdir, "threshold_policy.json"), "rb") as pf:
            hasher.update(pf.read())
        new_sha = hasher.hexdigest()

        with open(os.path.join(tmpdir, "checksum.sha256"), "r") as mf:
            lines = mf.readlines()
        with open(os.path.join(tmpdir, "checksum.sha256"), "w") as mf:
            for line in lines:
                if "threshold_policy.json" in line:
                    mf.write(f"{new_sha}  threshold_policy.json\n")
                else:
                    mf.write(line)

        with pytest.raises(ModelBundleIntegrityError) as exc:
            ProductionModelService(bundle_dir=tmpdir, enforce_checksum=True)
        assert "Invalid threshold policy" in str(exc.value)

# ==============================================================================
# R4.7 — End-to-End Pipeline Integration (Kafka -> Redis -> R3 -> R4 Decision)
# ==============================================================================

def test_full_e2e_streaming_pipeline_integration(state_manager, feature_engine, model_service):
    """
    Demonstrates complete end-to-end integration:
    Kafka Transaction -> Redis State -> R3 Feature Engine (32 Features) ->
    Production CatBoost -> Raw P -> Platt Calibrator -> Calibrated P -> ML Policy -> Decision.
    """
    captured_decisions = []

    def mock_sink_handler(event, context, feature_result, inference_result):
        assert event is not None
        assert feature_result is not None
        assert len(feature_result.features) == 32
        assert inference_result is not None
        assert inference_result.decision in ("APPROVE", "REVIEW", "BLOCK")
        captured_decisions.append({
            "tx_id": event.transaction_id,
            "decision": inference_result.decision,
            "cal_p": inference_result.calibrated_probability,
            "raw_p": inference_result.raw_probability,
            "latency_ms": inference_result.latency_ms
        })

    config = StreamingConfig()
    consumer = TransactionConsumer(
        config=config,
        handler=mock_sink_handler,
        state_manager=state_manager,
        feature_engine=feature_engine,
        model_service=model_service
    )

    # 1. Normal benign transaction
    tx_normal = create_sample_transaction(
        transaction_id="tx_normal_001",
        customer_id="cust_normal_99",
        amount=35.0,
        category="groceries",
        payment_type="chip",
        timestamp=1700000000.0,
        latitude=40.7128,
        longitude=-74.0060
    )
    raw_bytes = tx_normal.to_bytes()
    event = consumer.process_raw_message(raw_bytes)
    assert event is not None

    evt, ctx, feat_res, inf_res = consumer.process_event_with_state(event)
    assert inf_res is not None
    assert inf_res.decision in ("APPROVE", "REVIEW")
    assert inf_res.model_version == "finpulse-v3"
    assert inf_res.feature_schema_version == "2.0"

    # 2. High-velocity anomalous transaction
    # Pre-populate state with multiple rapid transactions
    t_base = 1700000050.0
    for i in range(12):
        prior_tx = create_sample_transaction(
            transaction_id=f"tx_prior_{i}",
            customer_id="cust_attacker_1",
            amount=1500.0,
            category="electronics",
            payment_type="online",
            timestamp=t_base + i * 5.0,
            latitude=51.5074,
            longitude=-0.1278
        )
        consumer.process_event_with_state(prior_tx)

    # Incoming attack transaction
    tx_attack = create_sample_transaction(
        transaction_id="tx_attack_999",
        customer_id="cust_attacker_1",
        amount=4800.0,
        category="electronics",
        payment_type="online",
        timestamp=t_base + 100.0,
        latitude=35.6762,  # Tokyo (teleportation anomaly)
        longitude=139.6503
    )

    evt2, ctx2, feat_res2, inf_res2 = consumer.process_event_with_state(tx_attack)
    assert inf_res2 is not None
    assert inf_res2.raw_probability > 0.0
    assert inf_res2.calibrated_probability > 0.0
    assert len(captured_decisions) >= 2

def test_inference_performance_latency(model_service, feature_engine, state_manager):
    """
    Measures the real-time inference latency of the model service.
    Requirement: Sub-5ms scoring per transaction.
    """
    tx = create_sample_transaction(customer_id="cust_perf", amount=150.0)
    feat_res = feature_engine.extract_features(tx)

    latencies = []
    # Warmup
    for _ in range(5):
        model_service.score_features(feat_res.features, tx.transaction_id)

    # Benchmarking 50 iterations
    for _ in range(50):
        t0 = time.perf_counter()
        res = model_service.score_features(feat_res.features, tx.transaction_id)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)
        assert res.decision in ("APPROVE", "REVIEW", "BLOCK")

    p95_latency = np.percentile(latencies, 95)
    mean_latency = np.mean(latencies)
    assert mean_latency < 5.0, f"Average inference latency too high: {mean_latency:.2f}ms"
    assert p95_latency < 10.0, f"P95 inference latency too high: {p95_latency:.2f}ms"
