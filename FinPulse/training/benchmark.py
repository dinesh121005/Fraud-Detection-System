"""ML-02: Three-Dataset Baseline Benchmark Suite.

Evaluates 5 candidate models across 3 independent datasets:
Datasets: PaySim, Sparkov, IEEE-CIS
Models:
1. Logistic Regression
2. Random Forest
3. XGBoost
4. LightGBM
5. CatBoost

Metrics per model/dataset:
- PR-AUC (Primary)
- ROC-AUC
- Precision
- Recall
- F1
- Recall@1% FPR
- FPR
- FNR
- Brier Score
- Inference Latency (ms/sample)
"""
import os
import sys
import time
import json
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.models.baselines import LogisticRegressionBaseline, RandomForestBaseline
from src.models.gbm_models import XGBoostModel, LightGBMModel, CatBoostModel
from src.models.imbalance import ImbalanceHandler
from src.evaluation.metrics import evaluate_fraud_metrics
from training.dataset_builder import process_single_dataset, ADAPTER_MAP, load_configs, save_processed_dataset

MODELS_ORDER = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "CatBoost"
]

def load_or_generate_dataset(dataset_key: str, sample_size: int = 30000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load cached features or build them via dataset adapter and feature engine."""
    processed_file = os.path.join(FINPULSE_DIR, "data", "processed", f"{dataset_key}_features.npz")
    
    if os.path.exists(processed_file):
        data = np.load(processed_file)
        return data["X_train"], data["y_train"], data["X_val"], data["y_val"]

    # Fallback to generating directly
    datasets_cfg, _ = load_configs()
    adapter_cls, display_name = ADAPTER_MAP[dataset_key]
    print(f"  -> Features not pre-cached for {display_name}. Extracting now...")
    res = process_single_dataset(dataset_key, adapter_cls, datasets_cfg[dataset_key], sample_size=sample_size)
    save_processed_dataset(dataset_key, res, os.path.join(FINPULSE_DIR, "data", "processed"))
    return res["splits"]["train"]["X"], res["splits"]["train"]["y"], res["splits"]["val"]["X"], res["splits"]["val"]["y"]

def create_model_suite(scale_pos_weight: float = 1.0) -> Dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegressionBaseline(C=1.0, random_state=42),
        "Random Forest": RandomForestBaseline(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": XGBoostModel(n_estimators=150, max_depth=5, scale_pos_weight=scale_pos_weight, random_state=42),
        "LightGBM": LightGBMModel(n_estimators=150, num_leaves=31, scale_pos_weight=scale_pos_weight, random_state=42),
        "CatBoost": CatBoostModel(iterations=250, depth=5, random_state=42)
    }

def run_benchmark(sample_size: int = 30000):
    reports_dir = os.path.join(FINPULSE_DIR, "reports", "baseline")
    os.makedirs(reports_dir, exist_ok=True)

    datasets = ["paysim", "sparkov", "ieee_cis"]
    dataset_names = {"paysim": "PaySim", "sparkov": "Sparkov", "ieee_cis": "IEEE-CIS"}

    benchmark_all: Dict[str, Dict[str, Any]] = {d: {} for d in datasets}
    all_records = []

    print("\n" + "=" * 115)
    print("  FINPULSE ML-02: THREE-DATASET 5-MODEL BASELINE BENCHMARK")
    print("=" * 115)

    for d_key in datasets:
        d_name = dataset_names[d_key]
        print(f"\n[{d_name.upper()} DATASET] Loading Train/Validation Splits...")
        X_train, y_train, X_val, y_val = load_or_generate_dataset(d_key, sample_size=sample_size)
        
        scale_pos = ImbalanceHandler.calculate_scale_pos_weight(y_train)
        print(f"  -> Train samples: {len(X_train):,} (Fraud: {int(y_train.sum())}) | Val samples: {len(X_val):,} (Fraud: {int(y_val.sum())})")
        print(f"  -> Imbalance ratio (scale_pos_weight): {scale_pos:.2f}")

        models = create_model_suite(scale_pos_weight=scale_pos)

        print("-" * 115)
        print(f"{'Model Candidate':<22} | {'PR-AUC':<7} | {'ROC-AUC':<7} | {'Recall@1%FPR':<12} | {'Precision':<9} | {'Recall':<7} | {'F1':<7} | {'Latency':<8}")
        print("-" * 115)

        d_report = {
            "dataset": d_name,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "train_fraud": int(y_train.sum()),
            "val_fraud": int(y_val.sum()),
            "scale_pos_weight": round(scale_pos, 2),
            "models": {}
        }

        for m_name in MODELS_ORDER:
            model = models[m_name]
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            fit_time = time.perf_counter() - t0

            t_infer = time.perf_counter()
            probs_val = model.predict_proba(X_val)
            infer_latency_ms = ((time.perf_counter() - t_infer) / len(X_val)) * 1000.0

            metrics = evaluate_fraud_metrics(y_val, probs_val)
            metrics["latency_ms"] = round(infer_latency_ms, 4)
            metrics["fit_time_sec"] = round(fit_time, 2)

            d_report["models"][m_name] = metrics
            benchmark_all[d_key][m_name] = metrics

            all_records.append({
                "dataset": d_name,
                "model": m_name,
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
                "recall_at_fpr_1pct": metrics["recall_at_fpr_1pct"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "fpr": metrics["fpr"],
                "fnr": metrics["fnr"],
                "brier_score": metrics["brier_score"],
                "latency_ms": metrics["latency_ms"],
                "fit_time_sec": metrics["fit_time_sec"]
            })

            print(f"{m_name:<22} | {metrics['pr_auc']:<7.4f} | {metrics['roc_auc']:<7.4f} | {metrics['recall_at_fpr_1pct']:<12.4f} | {metrics['precision']:<9.4f} | {metrics['recall']:<7.4f} | {metrics['f1']:<7.4f} | {metrics['latency_ms']:<5.3f} ms")

        # Save single dataset report
        single_path = os.path.join(reports_dir, f"{d_key}.json")
        with open(single_path, "w") as f:
            json.dump(d_report, f, indent=2)

    # Multi-dataset summary table
    df_results = pd.DataFrame(all_records)
    csv_path = os.path.join(reports_dir, "multi_dataset_benchmark.csv")
    df_results.to_csv(csv_path, index=False)

    # Pivot table for comparison
    pivot_prauc = df_results.pivot(index="model", columns="dataset", values="pr_auc")
    pivot_prauc["Mean PR-AUC"] = pivot_prauc.mean(axis=1).round(4)
    pivot_prauc = pivot_prauc.reindex(MODELS_ORDER)

    pivot_latency = df_results.pivot(index="model", columns="dataset", values="latency_ms")
    pivot_latency["Mean Latency (ms)"] = pivot_latency.mean(axis=1).round(4)
    pivot_latency = pivot_latency.reindex(MODELS_ORDER)

    print("\n" + "=" * 80)
    print("  MULTI-DATASET AGGREGATE BENCHMARK (PR-AUC)")
    print("=" * 80)
    print(pivot_prauc.to_string())
    print("\n" + "=" * 80)
    print("  MULTI-DATASET AGGREGATE BENCHMARK (INFERENCE LATENCY ms/sample)")
    print("=" * 80)
    print(pivot_latency.to_string())

    # Build comprehensive multi-dataset JSON
    multi_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "datasets": dataset_names,
        "models": MODELS_ORDER,
        "aggregate_pr_auc": pivot_prauc.to_dict(),
        "aggregate_latency_ms": pivot_latency.to_dict(),
        "detailed_results": benchmark_all
    }

    multi_json_path = os.path.join(reports_dir, "multi_dataset_benchmark.json")
    with open(multi_json_path, "w") as f:
        json.dump(multi_report, f, indent=2)

    print(f"\nArtifacts saved:")
    print(f"  -> {csv_path}")
    print(f"  -> {multi_json_path}")
    print("=" * 80)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="FinPulse 5-Model Multi-Dataset Benchmark")
    parser.add_argument("--sample-size", type=int, default=30000, help="Row sample size per dataset")
    args = parser.parse_args()

    run_benchmark(sample_size=args.sample_size)

if __name__ == "__main__":
    main()
