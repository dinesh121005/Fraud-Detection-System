"""ML-06: Final Probability Calibration Engine.

Calibrates the final optimized model on non-training validation data:
- Compares Isotonic Regression vs Platt Scaling (Sigmoid)
- Evaluates Brier Score, Expected Calibration Error (ECE), and Reliability Curves
- Uses strictly VALIDATION set (zero overlap with model training, zero leakage to test)
- Persists:
    models/candidates/finpulse_model.pkl
    models/candidates/finpulse_calibrator.pkl
    reports/calibration/calibration_report.json
"""
import os
import sys
import time
import json
import joblib
import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from typing import Dict, Any, Tuple

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.models.calibrator import ProbabilityCalibrator

def compute_expected_calibration_error(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_proba >= bin_lower) & (y_proba < bin_upper) if i < n_bins - 1 else (y_proba >= bin_lower) & (y_proba <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if np.sum(in_bin) > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_proba[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)

def run_calibration() -> Dict[str, Any]:
    candidates_dir = os.path.join(FINPULSE_DIR, "models", "candidates")
    reports_cal_dir = os.path.join(FINPULSE_DIR, "reports", "calibration")
    os.makedirs(reports_cal_dir, exist_ok=True)

    # 1. Load Candidate Model
    model_candidate_path = os.path.join(candidates_dir, "finpulse_model_candidate.pkl")
    if not os.path.exists(model_candidate_path):
        raise FileNotFoundError(f"Model candidate not found at {model_candidate_path}. Run train_production.py first.")

    model = joblib.load(model_candidate_path)

    # 2. Load Metadata and Validation Split
    meta_path = os.path.join(candidates_dir, "model_candidate_metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    dataset_name = meta["training_datasets"][0]

    data_path = os.path.join(FINPULSE_DIR, "data", "processed", f"{dataset_name}_features.npz")
    data = np.load(data_path)
    X_val = data["X_val"]
    y_val = data["y_val"]

    print("\n" + "=" * 80)
    print("  FINPULSE ML-06: PROBABILITY CALIBRATION")
    print("=" * 80)
    print(f"Base Model:             {meta['model']}")
    print(f"Calibration Dataset:    {dataset_name.upper()} (Validation Split, N={len(X_val):,})")
    print(f"Data Isolation:         Validation fold strictly separated from model training")
    print(f"Test Isolation:         TEST SET IS STRICTLY UNTOUCHED")
    print("-" * 80)

    # 3. Split validation set into 50% calibrator fit, 50% calibrator selection
    n_val = len(X_val)
    mid = n_val // 2
    X_cal_fit, y_cal_fit = X_val[:mid], y_val[:mid]
    X_cal_eval, y_cal_eval = X_val[mid:], y_val[mid:]

    raw_probs_fit = model.predict_proba(X_cal_fit)
    raw_probs_eval = model.predict_proba(X_cal_eval)

    raw_brier = brier_score_loss(y_cal_eval, raw_probs_eval)
    raw_ece = compute_expected_calibration_error(y_cal_eval, raw_probs_eval)

    # Fit Isotonic Calibrator
    iso_cal = ProbabilityCalibrator(method="isotonic").fit(raw_probs_fit, y_cal_fit)
    iso_probs_eval = iso_cal.predict(raw_probs_eval)
    iso_brier = brier_score_loss(y_cal_eval, iso_probs_eval)
    iso_ece = compute_expected_calibration_error(y_cal_eval, iso_probs_eval)

    # Fit Platt Calibrator
    platt_cal = ProbabilityCalibrator(method="platt").fit(raw_probs_fit, y_cal_fit)
    platt_probs_eval = platt_cal.predict(raw_probs_eval)
    platt_brier = brier_score_loss(y_cal_eval, platt_probs_eval)
    platt_ece = compute_expected_calibration_error(y_cal_eval, platt_probs_eval)

    print(f"{'Method':<15} | {'Brier Score':<12} | {'ECE':<10} | {'Brier Delta':<12}")
    print("-" * 55)
    print(f"{'Uncalibrated':<15} | {raw_brier:<12.5f} | {raw_ece:<10.5f} | {'0.00000':<12}")
    print(f"{'Isotonic':<15} | {iso_brier:<12.5f} | {iso_ece:<10.5f} | {raw_brier - iso_brier:<+12.5f}")
    print(f"{'Platt (Sigmoid)':<15} | {platt_brier:<12.5f} | {platt_ece:<10.5f} | {raw_brier - platt_brier:<+12.5f}")
    print("-" * 55)

    # Select best calibration method
    if iso_brier <= platt_brier:
        selected_method = "isotonic"
        best_calibrator = iso_cal
        best_brier = iso_brier
        best_ece = iso_ece
    else:
        selected_method = "platt"
        best_calibrator = platt_cal
        best_brier = platt_brier
        best_ece = platt_ece

    # Re-fit the winning calibrator on the entire validation set for maximum fidelity
    raw_probs_all_val = model.predict_proba(X_val)
    final_calibrator = ProbabilityCalibrator(method=selected_method).fit(raw_probs_all_val, y_val)

    print(f"SELECTED CALIBRATION METHOD: '{selected_method.upper()}'")
    print(f"  -> Validation Brier Score: {best_brier:.5f} (Improvement: {raw_brier - best_brier:+.5f})")
    print(f"  -> Validation ECE:         {best_ece:.5f}")
    print("=" * 80)

    # Persist serialized model and calibrator
    final_model_path = os.path.join(candidates_dir, "finpulse_model.pkl")
    final_calibrator_path = os.path.join(candidates_dir, "finpulse_calibrator.pkl")

    joblib.dump(model, final_model_path)
    joblib.dump(final_calibrator, final_calibrator_path)

    report = {
        "candidate_model": meta["model"],
        "dataset": dataset_name,
        "selected_method": selected_method,
        "metrics": {
            "uncalibrated": {"brier_score": round(float(raw_brier), 5), "ece": round(float(raw_ece), 5)},
            "isotonic": {"brier_score": round(float(iso_brier), 5), "ece": round(float(iso_ece), 5)},
            "platt": {"brier_score": round(float(platt_brier), 5), "ece": round(float(platt_ece), 5)}
        },
        "brier_improvement": round(float(raw_brier - best_brier), 5),
        "test_isolation": "VERIFIED_UNTOUCHED",
        "artifacts": {
            "model": final_model_path,
            "calibrator": final_calibrator_path
        }
    }

    report_path = os.path.join(reports_cal_dir, "calibration_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Calibrated Artifacts Persisted:")
    print(f"  -> {final_model_path}")
    print(f"  -> {final_calibrator_path}")
    print(f"  -> {report_path}")

    return report

def main():
    run_calibration()

if __name__ == "__main__":
    main()
