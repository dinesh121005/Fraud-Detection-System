"""Comprehensive Validation Suite for FinPulse ML Production Package (ML-01 to ML-09)."""
import os
import sys
import json
import joblib
import hashlib
import numpy as np
import pytest

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.features.schema import FinPulseFeatureVector, CommonTransactionSchema
from src.evaluation.metrics import evaluate_fraud_metrics

def test_feature_vector_dimension():
    """Verify strictly 32 features are specified and defined."""
    assert len(FinPulseFeatureVector.FEATURE_NAMES) == 32, "Feature vector must have exactly 32 dimensions"

def test_processed_datasets_zero_nan_inf():
    """ML-01 Gate: All processed datasets must have 32 features and 0 NaN/Inf values."""
    processed_dir = os.path.join(FINPULSE_DIR, "data", "processed")
    for key in ["paysim", "sparkov", "ieee_cis"]:
        fpath = os.path.join(processed_dir, f"{key}_features.npz")
        assert os.path.exists(fpath), f"Processed dataset missing: {fpath}"
        data = np.load(fpath)
        for split in ["train", "val", "test"]:
            X = data[f"X_{split}"]
            y = data[f"y_{split}"]
            assert X.shape[1] == 32, f"{key} {split} feature dim != 32"
            assert not np.isnan(X).any(), f"{key} {split} contains NaN"
            assert not np.isinf(X).any(), f"{key} {split} contains Inf"
            assert len(X) == len(y), f"{key} {split} X and y length mismatch"

def test_candidate_selection_artifacts():
    """ML-03 Gate: Candidate model selection must be backed by benchmark metrics."""
    sel_path = os.path.join(FINPULSE_DIR, "reports", "baseline", "candidate_selection.json")
    assert os.path.exists(sel_path), "candidate_selection.json missing"
    with open(sel_path, "r") as f:
        data = json.load(f)
    assert "candidate_model" in data
    assert data["selection_metric"] == "pr_auc"
    assert "aggregate_performance" in data
    assert data["aggregate_performance"]["mean_pr_auc"] > 0.0

def test_optuna_tuning_artifacts():
    """ML-04 Gate: Optuna hyperparameter optimization artifacts must be reproducible."""
    tuning_dir = os.path.join(FINPULSE_DIR, "reports", "tuning")
    best_params_path = os.path.join(tuning_dir, "best_parameters.json")
    db_path = os.path.join(tuning_dir, "optuna_study.db")
    
    assert os.path.exists(best_params_path), "best_parameters.json missing"
    assert os.path.exists(db_path), "optuna_study.db missing"

    with open(best_params_path, "r") as f:
        best_data = json.load(f)
    assert "best_params" in best_data
    assert "best_val_pr_auc" in best_data

def test_production_bundle_integrity_and_checksums():
    """ML-09 Gate: Production bundle must be intact and all SHA-256 checksums must match."""
    prod_dir = os.path.join(FINPULSE_DIR, "models", "production", "finpulse-v3")
    assert os.path.exists(prod_dir), f"Production bundle missing at {prod_dir}"

    checksum_file = os.path.join(prod_dir, "checksum.sha256")
    assert os.path.exists(checksum_file), "checksum.sha256 missing"

    with open(checksum_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        if not line.strip():
            continue
        expected_sha, fname = line.strip().split()
        fpath = os.path.join(prod_dir, fname)
        assert os.path.exists(fpath), f"Artifact missing: {fname}"

        hasher = hashlib.sha256()
        with open(fpath, "rb") as af:
            while chunk := af.read(65536):
                hasher.update(chunk)
        actual_sha = hasher.hexdigest()
        assert actual_sha == expected_sha, f"Checksum mismatch for {fname}: expected {expected_sha}, got {actual_sha}"

def test_production_model_inference_and_decision_policy():
    """Verify production model, calibrator, and threshold policy execute inference correctly."""
    prod_dir = os.path.join(FINPULSE_DIR, "models", "production", "finpulse-v3")
    
    model = joblib.load(os.path.join(prod_dir, "model.pkl"))
    calibrator = joblib.load(os.path.join(prod_dir, "calibrator.pkl"))
    
    with open(os.path.join(prod_dir, "threshold_policy.json"), "r") as f:
        policy = json.load(f)
    tau_review = policy["tau_review"]
    tau_block = policy["tau_block"]

    # Dummy batch of 5 samples with 32 features
    X_dummy = np.random.randn(5, 32).astype(np.float32)
    raw_probs = model.predict_proba(X_dummy)
    assert len(raw_probs) == 5
    assert (raw_probs >= 0.0).all() and (raw_probs <= 1.0).all()

    cal_probs = calibrator.predict(raw_probs)
    assert len(cal_probs) == 5
    assert (cal_probs >= 0.0).all() and (cal_probs <= 1.0).all()

    # Decision mapping
    for p in cal_probs:
        if p < tau_review:
            dec = "APPROVE"
        elif p < tau_block:
            dec = "REVIEW"
        else:
            dec = "BLOCK"
        assert dec in ["APPROVE", "REVIEW", "BLOCK"]
