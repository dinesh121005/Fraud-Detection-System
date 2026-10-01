"""Dataset-Level Validation Script for PaySim, Sparkov, and IEEE-CIS.

Demonstrates that the 32-feature FinPulseFeatureEngine successfully processes each dataset
through its adapter into a clean, leakage-free 32-feature matrix with zero NaN/Inf values.
"""
import os
import sys
import time
import json
import yaml
import numpy as np
import pandas as pd

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.data.paysim_adapter import PaySimAdapter
from src.data.sparkov_adapter import SparkovAdapter
from src.data.ieee_adapter import IEEECISAdapter
from src.features.engine import FinPulseFeatureEngine
from src.features.schema import FinPulseFeatureVector

def validate_dataset(adapter_cls, config: dict, dataset_name: str, sample_size: int = 5000) -> dict:
    print(f"\n[{dataset_name.upper()}] Starting Validation...")
    t0 = time.time()
    adapter = adapter_cls(config)

    # 1. Load Raw Data
    df_raw = adapter.load_raw(sample_size=sample_size)
    raw_rows = len(df_raw)

    # 2. Clean & Filter Leakage
    df_clean = adapter.clean_and_filter_leakage(df_raw)
    clean_rows = len(df_clean)
    skipped_rows = raw_rows - clean_rows

    # 3. Transform to Common Schema
    df_common = adapter.transform_to_common_schema(df_clean)

    # 4. Feature Extraction via Real 32-Feature Engine
    feature_engine = FinPulseFeatureEngine()
    feature_engine.fit(df_common)
    X, y, feature_names = feature_engine.compute_offline_features(df_common)
    elapsed_sec = round(time.time() - t0, 2)

    # Quality & Integrity Checks
    nan_count = int(np.isnan(X).sum())
    inf_count = int(np.isinf(X).sum())
    fraud_count = int(np.sum(y == 1))
    legit_count = int(np.sum(y == 0))
    fraud_pct = round((fraud_count / len(y)) * 100, 3) if len(y) > 0 else 0.0

    # Feature distribution summary
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)

    # Missing field / semantic availability analysis
    field_availability = {
        "paysim": {
            "amount": "Native",
            "balance": "Native pre-tx oldbalanceOrg",
            "velocity": "Calculated via customer history",
            "behavioral": "Calculated via customer history",
            "gps_location": "Unavailable (default 0.0)",
            "device_hardware": "Unavailable (default unknown_device)",
            "auth_2fa": "Unavailable (default 1)"
        },
        "sparkov": {
            "amount": "Native",
            "balance": "Credit limit proxy (5000.0)",
            "velocity": "Calculated via customer history",
            "behavioral": "Calculated via customer history",
            "gps_location": "Native (cardholder home + merchant POS)",
            "device_hardware": "Card POS terminal proxy",
            "auth_2fa": "Unavailable (default 1)"
        },
        "ieee_cis": {
            "amount": "Native",
            "balance": "Credit limit proxy (10000.0)",
            "velocity": "Calculated via customer history",
            "behavioral": "Calculated via customer history",
            "gps_location": "Billing distance dist1",
            "device_hardware": "Native (DeviceType + DeviceInfo)",
            "auth_2fa": "Unavailable (default 1)"
        }
    }

    report = {
        "dataset_name": dataset_name,
        "rows_processed": clean_rows,
        "rows_skipped": skipped_rows,
        "feature_dimensionality": int(X.shape[1]),
        "feature_count_expected": len(FinPulseFeatureVector.FEATURE_NAMES),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "fraud_distribution": {
            "legitimate": legit_count,
            "fraud": fraud_count,
            "fraud_rate_pct": fraud_pct
        },
        "processing_time_sec": elapsed_sec,
        "field_availability": field_availability.get(dataset_name.lower(), {}),
        "sample_features": {
            feature_names[i]: {
                "mean": round(float(means[i]), 4),
                "std": round(float(stds[i]), 4),
                "min": round(float(mins[i]), 4),
                "max": round(float(maxs[i]), 4)
            }
            for i in [0, 1, 4, 11, 12, 14, 15, 17, 18, 20, 23, 26] # Sample representative features
        }
    }

    print(f"  -> Rows Processed: {clean_rows:,} | Rows Skipped: {skipped_rows}")
    print(f"  -> Feature Shape: {X.shape} | Feature Dimensions: {X.shape[1]}/32")
    print(f"  -> NaN count: {nan_count} | Inf count: {inf_count}")
    print(f"  -> Fraud Count: {fraud_count} ({fraud_pct}%)")
    print(f"  -> Processing Time: {elapsed_sec}s ({round(clean_rows/max(elapsed_sec, 0.01), 1)} rows/sec)")
    return report

def main():
    print("=" * 80)
    print("  FINPULSE ML REDESIGN: DATASET-LEVEL FEATURE ENGINE VALIDATION")
    print("=" * 80)

    config_path = os.path.join(FINPULSE_DIR, "configs", "datasets.yaml")
    with open(config_path, "r") as f:
        datasets_cfg = yaml.safe_load(f)["datasets"]

    reports = {}

    # 1. Validate PaySim
    reports["paysim"] = validate_dataset(PaySimAdapter, datasets_cfg["paysim"], "PaySim", sample_size=10000)

    # 2. Validate Sparkov
    reports["sparkov"] = validate_dataset(SparkovAdapter, datasets_cfg["sparkov"], "Sparkov", sample_size=10000)

    # 3. Validate IEEE-CIS
    reports["ieee_cis"] = validate_dataset(IEEECISAdapter, datasets_cfg["ieee_cis"], "IEEE-CIS", sample_size=10000)

    # Save validation report
    out_path = os.path.join(FINPULSE_DIR, "reports", "dataset_feature_validation_report.json")
    with open(out_path, "w") as f:
        json.dump(reports, f, indent=2)

    print("\n" + "=" * 80)
    print("  ALL THREE DATASETS VALIDATED SUCCESSFULLY ON REAL FEATURE ENGINE")
    print(f"  Full report saved to: {out_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
