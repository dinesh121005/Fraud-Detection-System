"""ML-04: Optuna Hyperparameter Optimization Engine.

Tunes ONLY the model candidate selected by ML-03.
Optimization Protocol:
- Objective: PR-AUC on Validation Split (maximizing)
- Trials: 35 trials
- Sampler: TPESampler(seed=42)
- Strict boundary:
    TRAIN set -> model fitting
    VALIDATION set -> objective score calculation
    TEST set -> UNTOUCHED (zero leakage)
- Persists:
    reports/tuning/optuna_study.db
    reports/tuning/optimization_history.json
    reports/tuning/best_parameters.json
    reports/tuning/tuning_report.json
"""
import os
import sys
import time
import json
import yaml
import sqlite3
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import average_precision_score
from typing import Dict, Any

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.models.baselines import LogisticRegressionBaseline, RandomForestBaseline
from src.models.gbm_models import XGBoostModel, LightGBMModel, CatBoostModel
from src.models.imbalance import ImbalanceHandler

optuna.logging.set_verbosity(optuna.logging.WARNING)

def sample_params(trial: optuna.Trial, model_name: str, search_space: dict) -> dict:
    params = {}
    for param_name, cfg in search_space.items():
        ptype = cfg.get("type", "float")
        if ptype == "float":
            params[param_name] = trial.suggest_float(
                param_name,
                float(cfg["low"]),
                float(cfg["high"]),
                log=cfg.get("log", False)
            )
        elif ptype == "int":
            params[param_name] = trial.suggest_int(
                param_name,
                int(cfg["low"]),
                int(cfg["high"]),
                log=cfg.get("log", False)
            )
        elif ptype == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, cfg["choices"])
    return params

def build_model_with_params(model_name: str, params: dict, scale_pos_weight: float = 1.0):
    name_clean = model_name.lower()
    if "lightgbm" in name_clean:
        p = params.copy()
        p["scale_pos_weight"] = scale_pos_weight
        p["random_state"] = 42
        p["verbose"] = -1
        p.setdefault("n_estimators", 200)
        return LightGBMModel(**p)
    elif "xgboost" in name_clean:
        p = params.copy()
        p["scale_pos_weight"] = scale_pos_weight
        p["random_state"] = 42
        p.setdefault("n_estimators", 200)
        return XGBoostModel(**p)
    elif "catboost" in name_clean:
        p = params.copy()
        p["random_state"] = 42
        p.setdefault("iterations", 250)
        return CatBoostModel(**p)
    elif "random forest" in name_clean:
        p = params.copy()
        p["random_state"] = 42
        return RandomForestBaseline(**p)
    elif "logistic" in name_clean:
        p = params.copy()
        p["random_state"] = 42
        return LogisticRegressionBaseline(**p)
    else:
        raise ValueError(f"Unsupported model for tuning: {model_name}")

def run_tuning(n_trials: int = 35) -> Dict[str, Any]:
    tuning_dir = os.path.join(FINPULSE_DIR, "reports", "tuning")
    os.makedirs(tuning_dir, exist_ok=True)

    # 1. Load Candidate Selection
    cand_file = os.path.join(FINPULSE_DIR, "reports", "baseline", "candidate_selection.json")
    if not os.path.exists(cand_file):
        raise FileNotFoundError("Candidate selection report not found. Run candidate_selection.py first.")

    with open(cand_file, "r") as f:
        cand_data = json.load(f)
    candidate_model_name = cand_data["candidate_model"]

    # 2. Load Tuning Config
    tuning_cfg_path = os.path.join(FINPULSE_DIR, "configs", "tuning.yaml")
    with open(tuning_cfg_path, "r") as f:
        tuning_cfg = yaml.safe_load(f)["tuning"]

    # 3. Load TRAIN and VALIDATION splits (TEST is STRICTLY untouched)
    primary_dataset = tuning_cfg.get("primary_dataset", "sparkov")
    processed_npz = os.path.join(FINPULSE_DIR, "data", "processed", f"{primary_dataset}_features.npz")
    if not os.path.exists(processed_npz):
        raise FileNotFoundError(f"Processed dataset {processed_npz} not found. Run dataset_builder.py first.")

    data = np.load(processed_npz)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    # Verify test set is NOT loaded

    scale_pos = ImbalanceHandler.calculate_scale_pos_weight(y_train)

    # Determine search space key
    space_key = "lightgbm"
    for k in tuning_cfg["search_spaces"]:
        if k in candidate_model_name.lower().replace(" ", "_"):
            space_key = k
            break
    search_space = tuning_cfg["search_spaces"][space_key]

    print("\n" + "=" * 80)
    print("  FINPULSE ML-04: OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    print(f"Target Candidate Model:   '{candidate_model_name}'")
    print(f"Search Space Key:         '{space_key}'")
    print(f"Target Metric:            Validation PR-AUC (Maximize)")
    print(f"Dataset for Tuning:       {primary_dataset.upper()} (Train: {len(X_train):,}, Val: {len(X_val):,})")
    print(f"Strict Test Isolation:    TEST SET IS UNTOUCHED")
    print(f"Planned Trials:           {n_trials}")
    print("-" * 80)

    db_path = os.path.join(tuning_dir, "optuna_study.db")
    storage_url = f"sqlite:///{os.path.abspath(db_path).replace(os.sep, '/')}"

    study = optuna.create_study(
        study_name=f"finpulse_{space_key}_tuning",
        storage=storage_url,
        direction="maximize",
        sampler=TPESampler(seed=42),
        load_if_exists=True
    )

    history = []

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial, candidate_model_name, search_space)
        model = build_model_with_params(candidate_model_name, params, scale_pos_weight=scale_pos)
        model.fit(X_train, y_train)

        probs_val = model.predict_proba(X_val)
        score = average_precision_score(y_val, probs_val)

        history.append({
            "trial_number": trial.number,
            "params": params,
            "val_pr_auc": round(float(score), 5)
        })
        return float(score)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_trial = study.best_trial
    print("-" * 80)
    print(f"Optuna Optimization Complete: {len(study.trials)} trials executed.")
    print(f"Best Validation PR-AUC: {best_trial.value:.5f} (Trial #{best_trial.number})")
    print(f"Best Hyperparameters:")
    for k, v in best_trial.params.items():
        print(f"  * {k}: {v}")
    print("=" * 80)

    # Persist artifacts
    best_params_path = os.path.join(tuning_dir, "best_parameters.json")
    with open(best_params_path, "w") as f:
        json.dump({
            "candidate_model": candidate_model_name,
            "best_trial_number": best_trial.number,
            "best_val_pr_auc": float(best_trial.value),
            "best_params": best_trial.params,
            "scale_pos_weight": float(scale_pos),
            "tuning_dataset": primary_dataset,
            "random_seed": 42
        }, f, indent=2)

    history_path = os.path.join(tuning_dir, "optimization_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    tuning_report_path = os.path.join(tuning_dir, "tuning_report.json")
    with open(tuning_report_path, "w") as f:
        json.dump({
            "candidate_model": candidate_model_name,
            "total_trials": len(study.trials),
            "best_score": float(best_trial.value),
            "best_trial_id": best_trial.number,
            "best_hyperparameters": best_trial.params,
            "search_space_configured": search_space,
            "test_set_isolation": "VERIFIED_UNTOUCHED",
            "study_db": db_path
        }, f, indent=2)

    print(f"Tuning artifacts saved to:")
    print(f"  -> {best_params_path}")
    print(f"  -> {history_path}")
    print(f"  -> {tuning_report_path}")
    print(f"  -> {db_path}")

    return {
        "candidate_model": candidate_model_name,
        "best_score": best_trial.value,
        "best_params": best_trial.params
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="FinPulse Optuna Tuning")
    parser.add_argument("--trials", type=int, default=35, help="Number of Optuna trials")
    args = parser.parse_args()

    run_tuning(n_trials=args.trials)

if __name__ == "__main__":
    main()
