"""ML-05: Final Model Training Engine.

Trains the winning candidate model using best hyperparameters found by Optuna.
Data protocol:
- Fits underlying model on TRAIN partition (holding VALIDATION strictly for calibration, and TEST untouched).
- Saves candidate model to models/candidates/finpulse_model_candidate.pkl and metadata.
"""
import os
import sys
import time
import json
import joblib
import yaml
import numpy as np
from typing import Dict, Any

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from training.tuning import build_model_with_params
from src.features.schema import FinPulseFeatureVector

def train_production_model() -> Dict[str, Any]:
    candidates_dir = os.path.join(FINPULSE_DIR, "models", "candidates")
    os.makedirs(candidates_dir, exist_ok=True)

    # 1. Load best parameters from tuning
    tuning_file = os.path.join(FINPULSE_DIR, "reports", "tuning", "best_parameters.json")
    if not os.path.exists(tuning_file):
        raise FileNotFoundError(f"Tuning file not found at {tuning_file}. Run tuning.py first.")

    with open(tuning_file, "r") as f:
        tuning_info = json.load(f)

    model_name = tuning_info["candidate_model"]
    best_params = tuning_info["best_params"]
    tuning_dataset = tuning_info.get("tuning_dataset", "sparkov")

    # 2. Load dataset
    data_path = os.path.join(FINPULSE_DIR, "data", "processed", f"{tuning_dataset}_features.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at {data_path}")

    data = np.load(data_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    # TEST remains untouched!

    scale_pos = tuning_info.get("scale_pos_weight", 1.0)

    print("\n" + "=" * 80)
    print("  FINPULSE ML-05: FINAL MODEL TRAINING")
    print("=" * 80)
    print(f"Candidate Model:        '{model_name}'")
    print(f"Training Dataset:       {tuning_dataset.upper()}")
    print(f"Training Samples:       {len(X_train):,} (Fraud: {int(y_train.sum())})")
    print(f"Feature Dimensions:     {X_train.shape[1]}")
    print(f"Hyperparameter Source:  Optuna (Trial #{tuning_info.get('best_trial_number', 0)})")
    print(f"Scale Pos Weight:       {scale_pos:.2f}")
    print(f"Strict Boundary:        TEST SET IS COMPLETELY UNTOUCHED")
    print("-" * 80)

    model = build_model_with_params(model_name, best_params, scale_pos_weight=scale_pos)
    
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    print(f"Training completed successfully in {fit_time:.2f}s.")

    # 3. Save Candidate Model
    model_candidate_path = os.path.join(candidates_dir, "finpulse_model_candidate.pkl")
    joblib.dump(model, model_candidate_path)

    metadata = {
        "model": model_name,
        "feature_count": int(X_train.shape[1]),
        "feature_schema_version": "2.0",
        "training_datasets": [tuning_dataset],
        "training_samples": len(X_train),
        "hyperparameter_source": "Optuna",
        "best_hyperparameters": best_params,
        "fit_time_seconds": round(fit_time, 2),
        "random_seed": 42,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    meta_path = os.path.join(candidates_dir, "model_candidate_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nTrained Candidate Model Persisted:")
    print(f"  -> Model Artifact:   {model_candidate_path}")
    print(f"  -> Model Metadata:   {meta_path}")
    print("=" * 80)

    return metadata

def main():
    train_production_model()

if __name__ == "__main__":
    main()
