"""ML-01: Pipeline Freeze & Reproducibility Dataset Builder.

Processes PaySim, Sparkov, and IEEE-CIS datasets through:
1. Canonical Dataset Adapters with Leakage Removal
2. Canonical CommonTransactionSchema Mapping
3. FinPulse 32-Feature Engine (Chronological, Zero-Leakage)
4. Chronological 70/15/15 Temporal Splitting
5. Strict Integrity Validation (Feature dim=32, NaN/Inf=0, Temporal Order Valid)
6. Reproducible Feature Matrix Freezing
"""
import os
import sys
import time
import json
import argparse
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.data.paysim_adapter import PaySimAdapter
from src.data.sparkov_adapter import SparkovAdapter
from src.data.ieee_adapter import IEEECISAdapter
from src.data.splitters import temporal_train_val_test_split
from src.features.engine import FinPulseFeatureEngine
from src.features.schema import FinPulseFeatureVector, CommonTransactionSchema

RANDOM_SEED = 42
FEATURE_SCHEMA_VERSION = "2.0"
FEATURE_COUNT = 32

ADAPTER_MAP = {
    "paysim": (PaySimAdapter, "PaySim"),
    "sparkov": (SparkovAdapter, "Sparkov"),
    "ieee_cis": (IEEECISAdapter, "IEEE-CIS")
}

def load_configs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    datasets_cfg_path = os.path.join(FINPULSE_DIR, "configs", "datasets.yaml")
    eval_cfg_path = os.path.join(FINPULSE_DIR, "configs", "evaluation.yaml")

    with open(datasets_cfg_path, "r") as f:
        datasets_cfg = yaml.safe_load(f)["datasets"]

    with open(eval_cfg_path, "r") as f:
        eval_cfg = yaml.safe_load(f)["evaluation"]

    return datasets_cfg, eval_cfg

def process_single_dataset(
    dataset_key: str,
    adapter_cls,
    dataset_cfg: dict,
    sample_size: Optional[int] = 40000
) -> Dict[str, Any]:
    """Process a single raw dataset into clean chronological 32-feature matrices."""
    adapter = adapter_cls(dataset_cfg)
    
    # 1. Raw Ingestion
    df_raw = adapter.load_raw(sample_size=sample_size)
    raw_count = len(df_raw)

    # 2. Leakage Elimination
    df_clean = adapter.clean_and_filter_leakage(df_raw)
    clean_count = len(df_clean)

    # 3. Canonical Schema Transformation
    df_common = adapter.transform_to_common_schema(df_clean)

    # 4. Strict Chronological Sort
    df_common = df_common.sort_values(by="timestamp").reset_index(drop=True)

    # 5. Chronological 70/15/15 Split
    train_df, val_df, test_df = temporal_train_val_test_split(
        df_common,
        time_col="timestamp",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15
    )

    # 6. Feature Extraction using FinPulseFeatureEngine
    engine = FinPulseFeatureEngine()
    engine.fit(train_df)

    X_train, y_train, feat_names = engine.compute_offline_features(train_df, is_sorted=True)
    X_val, y_val, _ = engine.compute_offline_features(val_df, is_sorted=True)
    X_test, y_test, _ = engine.compute_offline_features(test_df, is_sorted=True)

    # Verify temporal ordering
    t_train_max = float(train_df["timestamp"].max())
    t_val_min = float(val_df["timestamp"].min())
    t_val_max = float(val_df["timestamp"].max())
    t_test_min = float(test_df["timestamp"].min())

    temporal_valid = (t_train_max <= t_val_min) and (t_val_max <= t_test_min)

    # NaN / Inf checks
    nan_count = int(np.isnan(X_train).sum() + np.isnan(X_val).sum() + np.isnan(X_test).sum())
    inf_count = int(np.isinf(X_train).sum() + np.isinf(X_val).sum() + np.isinf(X_test).sum())
    feat_dim = int(X_train.shape[1])

    return {
        "dataset_key": dataset_key,
        "raw_count": raw_count,
        "clean_count": clean_count,
        "feature_dim": feat_dim,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "temporal_valid": temporal_valid,
        "feature_names": feat_names,
        "splits": {
            "train": {"X": X_train, "y": y_train, "count": len(y_train), "fraud": int(y_train.sum()), "t_range": (float(train_df["timestamp"].min()), t_train_max)},
            "val": {"X": X_val, "y": y_val, "count": len(y_val), "fraud": int(y_val.sum()), "t_range": (t_val_min, t_val_max)},
            "test": {"X": X_test, "y": y_test, "count": len(y_test), "fraud": int(y_test.sum()), "t_range": (t_test_min, float(test_df["timestamp"].max()))}
        }
    }

def save_processed_dataset(dataset_key: str, data: Dict[str, Any], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{dataset_key}_features.npz")
    np.savez_compressed(
        out_file,
        X_train=data["splits"]["train"]["X"],
        y_train=data["splits"]["train"]["y"],
        X_val=data["splits"]["val"]["X"],
        y_val=data["splits"]["val"]["y"],
        X_test=data["splits"]["test"]["X"],
        y_test=data["splits"]["test"]["y"],
        feature_names=np.array(data["feature_names"])
    )
    print(f"  -> Persisted feature bundle to: {out_file}")

def validate_datasets(sample_size: int = 10000) -> Dict[str, Any]:
    datasets_cfg, _ = load_configs()
    validation_summary = {}

    print("\n" + "=" * 80)
    print("  FINPULSE ML-01: PIPELINE FREEZE & REPRODUCIBILITY VALIDATION")
    print("=" * 80)
    print(f"{'Dataset':<12} | {'Rows':<8} | {'Feat Dim':<9} | {'NaN/Inf':<8} | {'Temporal Order':<15} | {'Status':<8}")
    print("-" * 80)

    all_passed = True
    for key, (adapter_cls, display_name) in ADAPTER_MAP.items():
        t0 = time.time()
        result = process_single_dataset(key, adapter_cls, datasets_cfg[key], sample_size=sample_size)
        dur = round(time.time() - t0, 2)

        is_valid = (
            result["feature_dim"] == FEATURE_COUNT and
            result["nan_count"] == 0 and
            result["inf_count"] == 0 and
            result["temporal_valid"]
        )
        if not is_valid:
            all_passed = False

        status_str = "VALID" if is_valid else "FAILED"
        print(f"{display_name:<12} | {result['clean_count']:<8} | {result['feature_dim']}/32    | {result['nan_count']}/{result['inf_count']:<5} | {'VALID' if result['temporal_valid'] else 'INVALID':<15} | {status_str:<8}")

        validation_summary[key] = {
            "dataset": display_name,
            "sample_size": sample_size,
            "rows_processed": result["clean_count"],
            "feature_dim": result["feature_dim"],
            "expected_feature_dim": FEATURE_COUNT,
            "nan_count": result["nan_count"],
            "inf_count": result["inf_count"],
            "temporal_order_valid": result["temporal_valid"],
            "status": status_str,
            "splits": {
                "train_rows": result["splits"]["train"]["count"],
                "train_fraud": result["splits"]["train"]["fraud"],
                "val_rows": result["splits"]["val"]["count"],
                "val_fraud": result["splits"]["val"]["fraud"],
                "test_rows": result["splits"]["test"]["count"],
                "test_fraud": result["splits"]["test"]["fraud"]
            },
            "processing_seconds": dur
        }

    print("=" * 80)
    print(f"Overall Gate Status: {'PASSED [GATE ML-01 CLEARED]' if all_passed else 'FAILED'}")
    print("=" * 80)

    # Save reports
    reports_dir = os.path.join(FINPULSE_DIR, "reports", "baseline")
    os.makedirs(reports_dir, exist_ok=True)
    report_file = os.path.join(reports_dir, "dataset_validation.json")
    with open(report_file, "w") as f:
        json.dump(validation_summary, f, indent=2)

    return validation_summary

def build_datasets(sample_sizes: Optional[Dict[str, int]] = None):
    datasets_cfg, eval_cfg = load_configs()
    sizes = sample_sizes or eval_cfg.get("benchmark_sample_sizes", {"paysim": 40000, "sparkov": 40000, "ieee_cis": 40000})

    processed_dir = os.path.join(FINPULSE_DIR, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("  FINPULSE ML-01: FREEZING REPRODUCIBLE DATASET MATRICES")
    print("=" * 80)

    build_metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "random_seed": RANDOM_SEED,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_count": FEATURE_COUNT,
        "datasets": {}
    }

    for key, (adapter_cls, display_name) in ADAPTER_MAP.items():
        sample_size = sizes.get(key, 40000)
        print(f"\nProcessing {display_name} (sample_size={sample_size})...")
        t0 = time.time()
        result = process_single_dataset(key, adapter_cls, datasets_cfg[key], sample_size=sample_size)
        save_processed_dataset(key, result, processed_dir)
        dur = round(time.time() - t0, 2)
        print(f"  -> {display_name} completed in {dur}s: Train={result['splits']['train']['count']}, Val={result['splits']['val']['count']}, Test={result['splits']['test']['count']}")

        build_metadata["datasets"][key] = {
            "name": display_name,
            "sample_size": sample_size,
            "train_size": result["splits"]["train"]["count"],
            "val_size": result["splits"]["val"]["count"],
            "test_size": result["splits"]["test"]["count"],
            "train_fraud": result["splits"]["train"]["fraud"],
            "val_fraud": result["splits"]["val"]["fraud"],
            "test_fraud": result["splits"]["test"]["fraud"],
            "processing_seconds": dur
        }

    meta_path = os.path.join(processed_dir, "dataset_build_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(build_metadata, f, indent=2)
    print(f"\nDataset build metadata saved to: {meta_path}")

def main():
    parser = argparse.ArgumentParser(description="FinPulse Dataset Builder & Validator")
    parser.add_argument("--validate", action="store_true", help="Run ML-01 dataset validation check")
    parser.add_argument("--build", action="store_true", help="Build and freeze processed dataset matrices")
    parser.add_argument("--sample-size", type=int, default=None, help="Override sample size for validation or build")
    args = parser.parse_args()

    # Default to both validate and build if neither specified, or execute as requested
    if args.validate and not args.build:
        sample_size = args.sample_size or 10000
        validate_datasets(sample_size=sample_size)
    elif args.build and not args.validate:
        sample_sizes = {k: args.sample_size for k in ADAPTER_MAP} if args.sample_size else None
        build_datasets(sample_sizes=sample_sizes)
    else:
        sample_size = args.sample_size or 10000
        validate_datasets(sample_size=sample_size)
        sample_sizes = {k: args.sample_size for k in ADAPTER_MAP} if args.sample_size else None
        build_datasets(sample_sizes=sample_sizes)

if __name__ == "__main__":
    main()
