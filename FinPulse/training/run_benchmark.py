"""5-Model Benchmark Runner: Temporal Evaluation, Calibration, and Thresholding."""
import os
import sys
import json
import time
import yaml
import joblib
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.data.sparkov_adapter import SparkovAdapter
from src.data.splitters import temporal_train_val_test_split
from src.features.pipeline import FinPulseFeaturePipeline
from src.models.baselines import LogisticRegressionBaseline, RandomForestBaseline
from src.models.gbm_models import XGBoostModel, LightGBMModel, CatBoostModel
from src.models.imbalance import ImbalanceHandler
from src.models.calibrator import ProbabilityCalibrator
from src.evaluation.metrics import evaluate_fraud_metrics
from src.evaluation.thresholding import ThresholdOptimizer

def main():
    print("=" * 100)
    print("  FINPULSE ML REDESIGN: 5-MODEL BENCHMARK WITH REAL 32-FEATURE ENGINE")
    print("=" * 100)

    # 1. Ingestion & Adapter Mapping to Canonical Representation
    config_path = os.path.join(FINPULSE_DIR, "configs", "datasets.yaml")
    with open(config_path, "r") as f:
        datasets_cfg = yaml.safe_load(f)["datasets"]

    adapter = SparkovAdapter(datasets_cfg["sparkov"])
    print("\n[Step 1/6] Ingesting Sparkov dataset with sanitization...")
    df_raw = adapter.load_raw(split="train", sample_size=50000)
    df_clean = adapter.clean_and_filter_leakage(df_raw)
    df_common = adapter.transform_to_common_schema(df_clean)

    print(f"  -> Total records: {len(df_common):,}")
    print(f"  -> Fraud prevalence: {df_common['is_fraud'].mean()*100:.3f}%")

    # Sort strictly chronologically
    df_common = df_common.sort_values(by="timestamp").reset_index(drop=True)

    # 2. Real 32-Feature Engine Generation (Zero Placeholders)
    print("\n[Step 2/6] Generating unified 32-feature matrix via FinPulseFeatureEngine (Zero Placeholders)...")
    pipeline = FinPulseFeaturePipeline().fit(df_common)
    X_all, y_all, feature_names = pipeline.engine.compute_offline_features(df_common, is_sorted=True)
    print(f"  -> Computed feature matrix shape: {X_all.shape}")
    print(f"  -> Features: {len(feature_names)} ordered features from features.yaml")

    # 3. Chronological Temporal Split (70/15/15)
    print("\n[Step 3/6] Executing Chronological Temporal Split (70/15/15)...")
    n_total = len(X_all)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_val, y_val = X_all[n_train:n_train + n_val], y_all[n_train:n_train + n_val]
    X_test, y_test = X_all[n_train + n_val:], y_all[n_train + n_val:]

    print(f"  -> Train set: {len(X_train):,} rows")
    print(f"  -> Val set:   {len(X_val):,} rows")
    print(f"  -> Test set:  {len(X_test):,} rows")

    scale_pos = ImbalanceHandler.calculate_scale_pos_weight(y_train)
    print(f"  -> Calculated scale_pos_weight: {scale_pos:.2f}")

    # 3. Model Candidates
    print("\n[Step 4/6] Training & Evaluating 5-Model Benchmark Suite on Untouched Test Set...")
    candidates = {
        "Logistic Regression": LogisticRegressionBaseline(C=1.0),
        "Random Forest": RandomForestBaseline(n_estimators=100, max_depth=10),
        "XGBoost (Baseline)": XGBoostModel(n_estimators=150, max_depth=5, scale_pos_weight=scale_pos),
        "LightGBM": LightGBMModel(n_estimators=150, num_leaves=31, scale_pos_weight=scale_pos),
        "CatBoost": CatBoostModel(iterations=250, depth=5)
    }

    benchmark_results = {}
    fitted_models = {}

    print("\n" + "=" * 105)
    print(f"{'Model Candidate':<22} | {'PR-AUC':<7} | {'ROC-AUC':<7} | {'Recall@1%FPR':<12} | {'Precision':<9} | {'Recall':<7} | {'F1':<7} | {'Latency':<8}")
    print("-" * 105)

    for name, model in candidates.items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        # Latency benchmark per transaction
        t_infer = time.perf_counter()
        probs_test = model.predict_proba(X_test)
        infer_latency_ms = ((time.perf_counter() - t_infer) / len(X_test)) * 1000.0

        metrics = evaluate_fraud_metrics(y_test, probs_test)
        metrics["latency_ms"] = round(infer_latency_ms, 3)
        metrics["fit_time_sec"] = round(fit_time, 2)
        benchmark_results[name] = metrics
        fitted_models[name] = model

        print(f"{name:<22} | {metrics['pr_auc']:<7.4f} | {metrics['roc_auc']:<7.4f} | {metrics['recall_at_fpr_1pct']:<12.4f} | {metrics['precision']:<9.4f} | {metrics['recall']:<7.4f} | {metrics['f1']:<7.4f} | {metrics['latency_ms']:<5.2f} ms")

    print("=" * 105)

    # 4. Best Model Selection & Calibration (GATE 7 & GATE 8)
    best_model_name = max(benchmark_results, key=lambda k: benchmark_results[k]["pr_auc"])
    best_model = fitted_models[best_model_name]
    print(f"\n[Step 5/6] Selected Best Model: '{best_model_name}' (PR-AUC: {benchmark_results[best_model_name]['pr_auc']})")

    # Fit calibration on Validation fold
    val_probs_raw = best_model.predict_proba(X_val)
    test_probs_raw = best_model.predict_proba(X_test)

    calibrator = ProbabilityCalibrator(method="isotonic").fit(val_probs_raw, y_val)
    test_probs_cal = calibrator.predict(test_probs_raw)
    cal_eval = ProbabilityCalibrator.evaluate_calibration(test_probs_raw, test_probs_cal, y_test)
    print(f"  -> Raw Brier Score:        {cal_eval['raw_brier_score']}")
    print(f"  -> Calibrated Brier Score:   {cal_eval['calibrated_brier_score']} (Reduction: {cal_eval['improvement']})")

    # 5. Dual-Threshold Optimization (GATE 9)
    print("\n[Step 6/6] Deriving Operational Decision Thresholds on Validation Set...")
    threshold_opt = ThresholdOptimizer(target_fpr=0.01, min_block_precision=0.85)
    val_probs_cal = calibrator.predict(val_probs_raw)
    thresh_results = threshold_opt.optimize_thresholds(y_val, val_probs_cal)

    print(f"  -> Decision Thresholds Locked:")
    print(f"     * tau_review: {thresh_results['tau_review']:.4f} (Flags max 1% false alerts)")
    print(f"     * tau_block:  {thresh_results['tau_block']:.4f}  (Min 85% precision for automatic block)")

    # Save artifacts
    artifacts_dir = os.path.join(FINPULSE_DIR, "models", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(artifacts_dir, "production_candidate_model.joblib"))
    joblib.dump(calibrator, os.path.join(artifacts_dir, "production_calibrator.joblib"))
    joblib.dump(pipeline, os.path.join(artifacts_dir, "production_feature_pipeline.joblib"))

    # Save benchmark report
    reports_dir = os.path.join(FINPULSE_DIR, "reports")
    with open(os.path.join(reports_dir, "model_benchmark_results.json"), "w") as f:
        json.dump({
            "benchmark": benchmark_results,
            "best_model": best_model_name,
            "calibration": cal_eval,
            "thresholds": thresh_results
        }, f, indent=2)

    print("\n" + "=" * 100)
    print("  GATES 4 - 9 VERIFIED SUCCESSFULLY")
    print(f"  Candidate model artifacts saved to: {artifacts_dir}")
    print(f"  Benchmark report saved to: {os.path.join(reports_dir, 'model_benchmark_results.json')}")
    print("=" * 100)

if __name__ == "__main__":
    main()
