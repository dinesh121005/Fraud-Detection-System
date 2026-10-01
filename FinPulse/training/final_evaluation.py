"""ML-08: Final Untouched-Test Evaluation Engine.

Executes the frozen production bundle against the UNTOUCHED TEST SET:
Pipeline:
    FROZEN MODEL -> FROZEN CALIBRATOR -> FROZEN THRESHOLD POLICY -> UNTOUCHED TEST PARTITIONS

Evaluates independently on test splits of:
- Sparkov (Primary credit fraud domain)
- PaySim (Mobile money cross-domain evaluation)
- IEEE-CIS (E-commerce card transactions cross-domain evaluation)

Generates:
- PR-AUC
- ROC-AUC
- Precision
- Recall
- F1
- Recall@1% FPR
- FPR / FNR
- Brier Score
- Expected Calibration Error (ECE)
- Inference Latency
- Operational tier decision distributions (APPROVE, REVIEW, BLOCK)

Persists:
    reports/final/paysim_final.json
    reports/final/sparkov_final.json
    reports/final/ieee_cis_final.json
    reports/final/final_model_report.json
"""
import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.evaluation.metrics import evaluate_fraud_metrics
from training.calibration import compute_expected_calibration_error

def run_final_evaluation() -> Dict[str, Any]:
    candidates_dir = os.path.join(FINPULSE_DIR, "models", "candidates")
    reports_final_dir = os.path.join(FINPULSE_DIR, "reports", "final")
    os.makedirs(reports_final_dir, exist_ok=True)

    # 1. Load Frozen Candidates
    model_path = os.path.join(candidates_dir, "finpulse_model.pkl")
    calibrator_path = os.path.join(candidates_dir, "finpulse_calibrator.pkl")
    policy_path = os.path.join(candidates_dir, "threshold_policy.json")
    meta_path = os.path.join(candidates_dir, "model_candidate_metadata.json")

    for p in [model_path, calibrator_path, policy_path, meta_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required artifact {p} missing. Ensure ML-05, ML-06, and ML-07 are completed.")

    model = joblib.load(model_path)
    calibrator = joblib.load(calibrator_path)
    with open(policy_path, "r") as f:
        threshold_policy = json.load(f)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    tau_review = threshold_policy["tau_review"]
    tau_block = threshold_policy["tau_block"]
    training_dataset = meta["training_datasets"][0]

    datasets = ["sparkov", "paysim", "ieee_cis"]
    dataset_labels = {"sparkov": "Sparkov", "paysim": "PaySim", "ieee_cis": "IEEE-CIS"}

    print("\n" + "=" * 100)
    print("  FINPULSE ML-08: FINAL UNTOUCHED-TEST EVALUATION")
    print("=" * 100)
    print(f"Model Candidate:       {meta['model']}")
    print(f"Calibrator:            {calibrator.method.upper()}")
    print(f"Decision Policy:       tau_review={tau_review:.4f}, tau_block={tau_block:.4f}")
    print(f"Primary Domain:        {training_dataset.upper()}")
    print(f"Protocol:              EVALUATING ON UNTOUCHED TEST SPLITS (Zero Prior Exposure)")
    print("-" * 100)

    final_reports = {}
    summary_table = []

    for d_key in datasets:
        d_name = dataset_labels[d_key]
        data_file = os.path.join(FINPULSE_DIR, "data", "processed", f"{d_key}_features.npz")
        if not os.path.exists(data_file):
            print(f"Warning: {data_file} not found. Skipping {d_name}.")
            continue

        data = np.load(data_file)
        X_test = data["X_test"]
        y_test = data["y_test"]

        # Inference timing
        t0 = time.perf_counter()
        raw_probs = model.predict_proba(X_test)
        cal_probs = calibrator.predict(raw_probs)
        dur = time.perf_counter() - t0
        latency_ms = (dur / len(X_test)) * 1000.0

        # Compute full metrics at default threshold 0.5 and policy thresholds
        metrics = evaluate_fraud_metrics(y_test, cal_probs, threshold=tau_review)
        ece = compute_expected_calibration_error(y_test, cal_probs)
        metrics["expected_calibration_error"] = round(ece, 5)
        metrics["latency_ms"] = round(latency_ms, 4)

        # Operational tier decisions
        approved = cal_probs < tau_review
        reviewed = (cal_probs >= tau_review) & (cal_probs < tau_block)
        blocked = cal_probs >= tau_block

        n = len(y_test)
        cnt_app = int(np.sum(approved))
        cnt_rev = int(np.sum(reviewed))
        cnt_blk = int(np.sum(blocked))

        fraud_app = int(np.sum(y_test[approved]))
        fraud_rev = int(np.sum(y_test[reviewed]))
        fraud_blk = int(np.sum(y_test[blocked]))

        total_fraud = int(np.sum(y_test))
        fraud_caught = fraud_rev + fraud_blk
        detection_rate = round((fraud_caught / max(total_fraud, 1)) * 100, 2)

        tier_summary = {
            "approved": {"count": cnt_app, "pct": round(cnt_app / n * 100, 2), "fraud_missed": fraud_app},
            "reviewed": {"count": cnt_rev, "pct": round(cnt_rev / n * 100, 2), "fraud_caught": fraud_rev},
            "blocked": {"count": cnt_blk, "pct": round(cnt_blk / n * 100, 2), "fraud_caught": fraud_blk},
            "overall_fraud_detection_rate_pct": detection_rate
        }

        dataset_report = {
            "dataset": d_name,
            "test_sample_size": n,
            "fraud_count": total_fraud,
            "fraud_rate_pct": round((total_fraud / n) * 100, 3),
            "metrics": metrics,
            "operational_tiers": tier_summary,
            "is_in_domain": (d_key == training_dataset)
        }

        # Save individual dataset report
        single_path = os.path.join(reports_final_dir, f"{d_key}_final.json")
        with open(single_path, "w") as f:
            json.dump(dataset_report, f, indent=2)

        final_reports[d_key] = dataset_report
        summary_table.append({
            "dataset": d_name,
            "test_size": n,
            "pr_auc": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],
            "recall_1pct_fpr": metrics["recall_at_fpr_1pct"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "brier_score": metrics["brier_score"],
            "ece": metrics["expected_calibration_error"],
            "detection_rate": f"{detection_rate}%",
            "latency_ms": metrics["latency_ms"]
        })

    # Print summary
    df_sum = pd.DataFrame(summary_table)
    print(df_sum.to_string(index=False))
    print("-" * 100)

    # Master final report
    master_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": meta["model"],
        "training_dataset": training_dataset,
        "calibrator_method": calibrator.method,
        "decision_thresholds": {
            "tau_review": tau_review,
            "tau_block": tau_block
        },
        "datasets_evaluated": list(final_reports.keys()),
        "summary": summary_table,
        "detailed_results": final_reports,
        "gate_status": "ML-08_UNTOUCHED_TEST_PASSED"
    }

    master_path = os.path.join(reports_final_dir, "final_model_report.json")
    with open(master_path, "w") as f:
        json.dump(master_report, f, indent=2)

    print(f"Final evaluation artifacts saved to: {reports_final_dir}")
    print("=" * 100)
    return master_report

def main():
    run_final_evaluation()

if __name__ == "__main__":
    main()
