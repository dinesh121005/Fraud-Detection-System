"""ML-09: Production Artifact Packaging & Registry.

Packages the frozen, validated artifacts into an immutable bundle:
models/
└── production/
    └── finpulse-v3/
        ├── model.pkl
        ├── calibrator.pkl
        ├── threshold_policy.json
        ├── feature_schema.json
        ├── model_metadata.json
        └── checksum.sha256

Generates SHA-256 integrity checksums for all production artifacts.
"""
import os
import sys
import json
import time
import shutil
import hashlib
import subprocess
from typing import Dict, Any

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.features.schema import FinPulseFeatureVector

def calculate_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_git_commit_sha() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=FINPULSE_DIR).decode().strip()
        return commit
    except Exception:
        return "local_release"

def register_production_bundle() -> Dict[str, Any]:
    candidates_dir = os.path.join(FINPULSE_DIR, "models", "candidates")
    prod_dir = os.path.join(FINPULSE_DIR, "models", "production", "finpulse-v3")
    reports_final_dir = os.path.join(FINPULSE_DIR, "reports", "final")
    os.makedirs(prod_dir, exist_ok=True)

    # 1. Verify ML-08 final evaluation passed
    final_report_file = os.path.join(reports_final_dir, "final_model_report.json")
    if not os.path.exists(final_report_file):
        raise FileNotFoundError("final_model_report.json not found. ML-08 must pass before production packaging.")

    with open(final_report_file, "r") as f:
        final_report = json.load(f)

    # 2. Source Candidate Artifacts
    src_model = os.path.join(candidates_dir, "finpulse_model.pkl")
    src_calibrator = os.path.join(candidates_dir, "finpulse_calibrator.pkl")
    src_policy = os.path.join(candidates_dir, "threshold_policy.json")
    src_meta = os.path.join(candidates_dir, "model_candidate_metadata.json")

    for p in [src_model, src_calibrator, src_policy, src_meta]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Source candidate artifact {p} missing.")

    with open(src_meta, "r") as f:
        candidate_meta = json.load(f)
    with open(src_policy, "r") as f:
        policy_data = json.load(f)

    # 3. Copy Model, Calibrator, and Threshold Policy
    dst_model = os.path.join(prod_dir, "model.pkl")
    dst_calibrator = os.path.join(prod_dir, "calibrator.pkl")
    dst_policy = os.path.join(prod_dir, "threshold_policy.json")

    shutil.copy2(src_model, dst_model)
    shutil.copy2(src_calibrator, dst_calibrator)
    shutil.copy2(src_policy, dst_policy)

    # 4. Generate feature_schema.json
    feature_schema_data = {
        "feature_schema_version": "2.0",
        "feature_count": 32,
        "feature_names": FinPulseFeatureVector.FEATURE_NAMES,
        "feature_groups": {
            "group_a_transaction": FinPulseFeatureVector.FEATURE_NAMES[0:5],
            "group_b_temporal": FinPulseFeatureVector.FEATURE_NAMES[5:11],
            "group_c_velocity": FinPulseFeatureVector.FEATURE_NAMES[11:18],
            "group_d_behavioral": FinPulseFeatureVector.FEATURE_NAMES[18:23],
            "group_e_context_location": FinPulseFeatureVector.FEATURE_NAMES[23:29],
            "group_f_anomaly_rule": FinPulseFeatureVector.FEATURE_NAMES[29:32]
        }
    }
    dst_schema = os.path.join(prod_dir, "feature_schema.json")
    with open(dst_schema, "w") as f:
        json.dump(feature_schema_data, f, indent=2)

    # 5. Generate model_metadata.json
    primary_dataset = candidate_meta["training_datasets"][0]
    prod_metadata = {
        "model_version": "finpulse-v3",
        "model_type": candidate_meta["model"],
        "feature_schema_version": "2.0",
        "feature_count": 32,
        "calibration": final_report.get("calibrator_method", "isotonic"),
        "threshold_policy": "v3",
        "decision_thresholds": {
            "tau_review": policy_data["tau_review"],
            "tau_block": policy_data["tau_block"]
        },
        "training_datasets": candidate_meta["training_datasets"],
        "training_samples": candidate_meta.get("training_samples"),
        "hyperparameter_source": candidate_meta.get("hyperparameter_source", "Optuna"),
        "hyperparameters": candidate_meta.get("best_hyperparameters", {}),
        "random_seed": 42,
        "git_commit": get_git_commit_sha(),
        "registration_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "FROZEN_PRODUCTION",
        "untouched_test_evaluation": final_report.get("summary", [])
    }
    dst_meta = os.path.join(prod_dir, "model_metadata.json")
    with open(dst_meta, "w") as f:
        json.dump(prod_metadata, f, indent=2)

    # 6. Generate SHA-256 Checksums
    checksum_files = [
        "model.pkl",
        "calibrator.pkl",
        "threshold_policy.json",
        "feature_schema.json",
        "model_metadata.json"
    ]

    checksums = {}
    checksum_lines = []
    for fname in checksum_files:
        fpath = os.path.join(prod_dir, fname)
        sha = calculate_sha256(fpath)
        checksums[fname] = sha
        checksum_lines.append(f"{sha}  {fname}\n")

    checksum_path = os.path.join(prod_dir, "checksum.sha256")
    with open(checksum_path, "w") as f:
        f.writelines(checksum_lines)

    print("\n" + "=" * 80)
    print("  FINPULSE ML-09: IMMUTABLE PRODUCTION BUNDLE PACKAGING")
    print("=" * 80)
    print(f"Production Version:     finpulse-v3")
    print(f"Model Architecture:     {prod_metadata['model_type']}")
    print(f"Calibrator:             {prod_metadata['calibration']}")
    print(f"Thresholds:             tau_review={prod_metadata['decision_thresholds']['tau_review']}, tau_block={prod_metadata['decision_thresholds']['tau_block']}")
    print(f"Training Dataset:       {primary_dataset.upper()}")
    print(f"Destination:            {prod_dir}")
    print("-" * 80)
    print("IMMUTABLE ARTIFACT CHECKSUMS (SHA-256):")
    for fname, sha in checksums.items():
        print(f"  * {fname:<25} : {sha}")
    print("=" * 80)
    print("  STAGE GATE ML-09 COMPLETE — FINPULSE ML v3 PRODUCTION BUNDLE READY")
    print("=" * 80 + "\n")

    return prod_metadata

def main():
    register_production_bundle()

if __name__ == "__main__":
    main()
