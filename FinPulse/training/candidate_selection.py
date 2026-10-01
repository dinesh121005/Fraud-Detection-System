"""ML-03: Candidate Model Selection Engine.

Selects the best performing model from the actual 3-dataset benchmark:
- Primary Metric: Mean PR-AUC across datasets
- Secondary Metrics: Recall@1% FPR, Precision, F1, Latency
- Outputs: reports/baseline/candidate_selection.json
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

def select_candidate() -> Dict[str, Any]:
    baseline_dir = os.path.join(FINPULSE_DIR, "reports", "baseline")
    csv_path = os.path.join(baseline_dir, "multi_dataset_benchmark.csv")
    json_path = os.path.join(baseline_dir, "multi_dataset_benchmark.json")

    if not os.path.exists(csv_path) or not os.path.exists(json_path):
        raise FileNotFoundError(f"Benchmark results not found in {baseline_dir}. Run benchmark.py first.")

    df = pd.read_csv(csv_path)

    # Compute mean metrics per model across datasets
    grouped = df.groupby("model").agg({
        "pr_auc": "mean",
        "roc_auc": "mean",
        "recall_at_fpr_1pct": "mean",
        "precision": "mean",
        "recall": "mean",
        "f1": "mean",
        "latency_ms": "mean",
        "fit_time_sec": "mean"
    }).round(4)

    # Primary rank by PR-AUC, breaking ties with recall@1%FPR then latency
    grouped["rank_score"] = grouped["pr_auc"] * 0.70 + grouped["recall_at_fpr_1pct"] * 0.20 + (1.0 / (grouped["latency_ms"] + 0.1)) * 0.10
    sorted_models = grouped.sort_values(by=["pr_auc", "recall_at_fpr_1pct"], ascending=False)

    best_model_name = sorted_models.index[0]
    best_row = sorted_models.loc[best_model_name]

    print("\n" + "=" * 90)
    print("  FINPULSE ML-03: CANDIDATE MODEL SELECTION")
    print("=" * 90)
    print(sorted_models[["pr_auc", "roc_auc", "recall_at_fpr_1pct", "precision", "f1", "latency_ms"]].to_string())
    print("-" * 90)
    print(f"SELECTED CANDIDATE: '{best_model_name}'")
    print(f"  -> Mean PR-AUC:            {best_row['pr_auc']:.4f}")
    print(f"  -> Mean Recall@1% FPR:     {best_row['recall_at_fpr_1pct']:.4f}")
    print(f"  -> Mean Precision:         {best_row['precision']:.4f}")
    print(f"  -> Mean F1:                {best_row['f1']:.4f}")
    print(f"  -> Mean Inference Latency: {best_row['latency_ms']:.3f} ms")
    print("=" * 90)

    # Breakdown per dataset
    per_dataset_metrics = {}
    for d_name in df["dataset"].unique():
        sub = df[(df["dataset"] == d_name) & (df["model"] == best_model_name)].iloc[0]
        per_dataset_metrics[d_name] = {
            "pr_auc": float(sub["pr_auc"]),
            "roc_auc": float(sub["roc_auc"]),
            "recall_at_fpr_1pct": float(sub["recall_at_fpr_1pct"]),
            "precision": float(sub["precision"]),
            "recall": float(sub["recall"]),
            "f1": float(sub["f1"]),
            "latency_ms": float(sub["latency_ms"])
        }

    selection_record = {
        "candidate_model": best_model_name,
        "selection_metric": "pr_auc",
        "secondary_metrics": [
            "recall_at_1pct_fpr",
            "precision",
            "latency"
        ],
        "aggregate_performance": {
            "mean_pr_auc": float(best_row["pr_auc"]),
            "mean_roc_auc": float(best_row["roc_auc"]),
            "mean_recall_at_1pct_fpr": float(best_row["recall_at_fpr_1pct"]),
            "mean_precision": float(best_row["precision"]),
            "mean_recall": float(best_row["recall"]),
            "mean_f1": float(best_row["f1"]),
            "mean_latency_ms": float(best_row["latency_ms"]),
            "mean_fit_time_sec": float(best_row["fit_time_sec"])
        },
        "per_dataset_breakdown": per_dataset_metrics,
        "comparison_table": grouped.to_dict(orient="index")
    }

    out_file = os.path.join(baseline_dir, "candidate_selection.json")
    with open(out_file, "w") as f:
        json.dump(selection_record, f, indent=2)

    print(f"Candidate selection record persisted to: {out_file}\n")
    return selection_record

def main():
    select_candidate()

if __name__ == "__main__":
    main()
