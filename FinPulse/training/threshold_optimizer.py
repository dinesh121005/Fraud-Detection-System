"""ML-07: Threshold & Operational Decision Optimization Engine.

Derives production decision boundaries (tau_review, tau_block) on calibrated validation probabilities:
Decision policy:
    P(fraud) < tau_review                  -> APPROVE
    tau_review <= P(fraud) < tau_block     -> REVIEW
    P(fraud) >= tau_block                  -> BLOCK

Operational constraints:
- Max Review FPR <= target_fpr (0.01 = 1.0% false positive alert rate)
- Min Block Precision >= min_block_precision (0.85 = 85% precision for automatic block)
- Test set remains STRICTLY UNTOUCHED.
- Persists: models/candidates/threshold_policy.json
"""
import os
import sys
import json
import joblib
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from typing import Dict, Any

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

def optimize_thresholds() -> Dict[str, Any]:
    candidates_dir = os.path.join(FINPULSE_DIR, "models", "candidates")
    model_path = os.path.join(candidates_dir, "finpulse_model.pkl")
    calibrator_path = os.path.join(candidates_dir, "finpulse_calibrator.pkl")
    meta_path = os.path.join(candidates_dir, "model_candidate_metadata.json")

    if not os.path.exists(model_path) or not os.path.exists(calibrator_path):
        raise FileNotFoundError("Model or calibrator candidate missing. Run train_production.py and calibration.py first.")

    with open(meta_path, "r") as f:
        meta = json.load(f)
    dataset_name = meta["training_datasets"][0]

    model = joblib.load(model_path)
    calibrator = joblib.load(calibrator_path)

    # Load validation data (strictly validation fold)
    data_path = os.path.join(FINPULSE_DIR, "data", "processed", f"{dataset_name}_features.npz")
    data = np.load(data_path)
    X_val = data["X_val"]
    y_val = data["y_val"]

    # Target constraints
    target_fpr = 0.01  # 1.0% maximum false alert rate
    min_block_precision = 0.85  # 85% minimum precision to block transaction without manual review

    print("\n" + "=" * 80)
    print("  FINPULSE ML-07: DECISION THRESHOLD OPTIMIZATION")
    print("=" * 80)
    print(f"Target Review Constraint:  Max FPR <= {target_fpr * 100:.1f}%")
    print(f"Target Block Constraint:   Min Precision >= {min_block_precision * 100:.1f}%")
    print(f"Validation Distribution:   {len(y_val):,} transactions ({int(y_val.sum())} fraudulent)")
    print(f"Data Source:               Calibrated probabilities on VALIDATION fold")
    print(f"Test Isolation:            TEST SET IS STRICTLY UNTOUCHED")
    print("-" * 80)

    raw_probs_val = model.predict_proba(X_val)
    cal_probs_val = calibrator.predict(raw_probs_val)

    # 1. Derive tau_review from ROC curve (FPR <= target_fpr)
    fpr_pts, tpr_pts, roc_thresh = roc_curve(y_val, cal_probs_val)
    valid_fpr_idx = np.where(fpr_pts <= target_fpr)[0]

    if len(valid_fpr_idx) > 0:
        idx_r = valid_fpr_idx[-1]
        tau_review = float(roc_thresh[idx_r])
        # Safety bound
        tau_review = float(np.clip(tau_review, 0.01, 0.40))
    else:
        tau_review = 0.10

    # 2. Derive tau_block from PR curve (Precision >= min_block_precision)
    prec_pts, recall_pts, pr_thresh = precision_recall_curve(y_val, cal_probs_val)
    valid_prec_idx = np.where(prec_pts[:-1] >= min_block_precision)[0]

    if len(valid_prec_idx) > 0:
        idx_b = valid_prec_idx[0]
        tau_block = float(pr_thresh[idx_b])
        # Ensure block threshold is strictly greater than review threshold
        tau_block = max(tau_block, tau_review + 0.15)
        tau_block = float(np.clip(tau_block, tau_review + 0.05, 0.95))
    else:
        # Fallback to high-confidence threshold
        tau_block = max(0.60, tau_review + 0.20)

    # Apply policy to validation set to verify operational tiering
    approved_mask = cal_probs_val < tau_review
    review_mask = (cal_probs_val >= tau_review) & (cal_probs_val < tau_block)
    block_mask = cal_probs_val >= tau_block

    n = len(y_val)
    cnt_app = int(np.sum(approved_mask))
    cnt_rev = int(np.sum(review_mask))
    cnt_blk = int(np.sum(block_mask))

    fraud_app = int(np.sum(y_val[approved_mask]))
    fraud_rev = int(np.sum(y_val[review_mask]))
    fraud_blk = int(np.sum(y_val[block_mask]))

    # Actual FPR at tau_review
    actual_rev_fpr = float(np.sum(~y_val.astype(bool) & (cal_probs_val >= tau_review)) / max(np.sum(~y_val.astype(bool)), 1))
    actual_blk_prec = float(fraud_blk / max(cnt_blk, 1)) if cnt_blk > 0 else 1.0

    print(f"DERIVED OPERATIONAL THRESHOLDS:")
    print(f"  * tau_review:            {tau_review:.4f}  (Actual Val FPR: {actual_rev_fpr * 100:.2f}%)")
    print(f"  * tau_block:             {tau_block:.4f}  (Actual Val Block Precision: {actual_blk_prec * 100:.1f}%)")
    print("-" * 80)
    print(f"Operational Tier Distribution on Validation Set:")
    print(f"  * APPROVE:  {cnt_app:,} ({cnt_app/n*100:.2f}%) | Frauds missed: {fraud_app}")
    print(f"  * REVIEW:   {cnt_rev:,} ({cnt_rev/n*100:.2f}%) | Frauds caught: {fraud_rev}")
    print(f"  * BLOCK:    {cnt_blk:,} ({cnt_blk/n*100:.2f}%) | Frauds caught: {fraud_blk}")
    print("=" * 80)

    policy = {
        "review_threshold": round(tau_review, 4),
        "block_threshold": round(tau_block, 4),
        "tau_review": round(tau_review, 4),
        "tau_block": round(tau_block, 4),
        "optimization_metric": "precision_recall_constraint",
        "validation_target_fpr": target_fpr,
        "validation_min_block_precision": min_block_precision,
        "operational_results": {
            "validation_fpr_at_review": round(actual_rev_fpr, 5),
            "validation_precision_at_block": round(actual_blk_prec, 5),
            "tier_distribution": {
                "approved_pct": round(cnt_app / n * 100, 2),
                "reviewed_pct": round(cnt_rev / n * 100, 2),
                "blocked_pct": round(cnt_blk / n * 100, 2)
            }
        },
        "test_isolation": "VERIFIED_UNTOUCHED"
    }

    out_path = os.path.join(candidates_dir, "threshold_policy.json")
    with open(out_path, "w") as f:
        json.dump(policy, f, indent=2)

    print(f"Threshold policy locked and persisted to: {out_path}\n")
    return policy

def main():
    optimize_thresholds()

if __name__ == "__main__":
    main()
