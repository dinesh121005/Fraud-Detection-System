# scripts/train_model.py

import os
import pandas as pd
import numpy as np
import random
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

# --------------------------
# 1. Setup
# --------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Modern NumPy RNG (to avoid legacy API warnings)
rng = np.random.default_rng(seed=RANDOM_SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "../data/onlinefraud.csv")
MODEL_DIR = os.path.join(SCRIPT_DIR, "../models")

os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------
# 2. Load dataset
# --------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Make sure 'onlinefraud.csv' is in 'data/' folder.")

df = pd.read_csv(DATA_PATH)

# --------------------------
# 3. Ensure target label
# --------------------------
if "isFraud" not in df.columns:
    df["isFraud"] = 0  # safe default

# --------------------------
# 4. Add contextual features
# --------------------------
def add_context(df):
    n = len(df)
    df["auth_verified"] = rng.choice([0, 1], size=n, p=[0.05, 0.95])
    df["device_known"] = rng.choice([0, 1], size=n, p=[0.1, 0.9])
    df["geo_known"] = rng.choice([0, 1], size=n, p=[0.1, 0.9])
    df["initiated_by"] = rng.choice(["sender", "third_party"], size=n, p=[0.98, 0.02])
    df["receiver_risk_score"] = np.round(rng.random(n), 3)
    df["sender_hourly_tx_count"] = rng.poisson(1, size=n)
    df["sender_daily_tx_count"] = rng.poisson(2, size=n)
    return df

df = add_context(df)

# --------------------------
# 5. Inject synthetic fraud attacks
# --------------------------
def inject_attacks(df, frac=0.02):
    df = df.copy()
    n = len(df)
    k = int(n * frac)
    idx = rng.choice(df.index, replace=False, size=k)

    df.loc[idx, "auth_verified"] = 0
    df.loc[idx, "initiated_by"] = "third_party"
    df.loc[idx, "device_known"] = 0
    df.loc[idx, "geo_known"] = 0
    df.loc[idx, "amount"] = df.loc[idx, "oldbalanceOrg"] + rng.uniform(1, 5000, size=k)
    df.loc[idx, "receiver_risk_score"] = np.clip(
        df.loc[idx, "receiver_risk_score"] + rng.uniform(0.2, 0.8, size=k), 0, 1
    )
    df.loc[idx, "isFraud"] = 1
    return df

df = inject_attacks(df, frac=0.02)

# --------------------------
# 6. Feature Engineering
# --------------------------
def feature_engineer(df):
    df = df.copy()
    df["amount_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1e-6)
    df["unusual_device"] = (df["device_known"] == 0).astype(int)
    df["unusual_location"] = (df["geo_known"] == 0).astype(int)
    df["initiated_by_third"] = (df["initiated_by"] == "third_party").astype(int)
    return df

df = feature_engineer(df)

# --------------------------
# 7. Prepare dataset
# --------------------------
features = [
    "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "amount_balance_ratio", "unusual_device", "unusual_location",
    "sender_hourly_tx_count", "sender_daily_tx_count",
    "receiver_risk_score", "initiated_by_third", "auth_verified"
]

df = df.dropna(subset=features + ["isFraud"])

# Encode transaction type
le_type = LabelEncoder()
df["type_enc"] = le_type.fit_transform(df["type"].astype(str))
df["type"] = df["type_enc"]

X = df[features].copy()
y = df["isFraud"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

# --------------------------
# 8. Train XGBoost model
# --------------------------
model = XGBClassifier(
    n_estimators=200,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=RANDOM_SEED
)

model.fit(X_train, y_train)

# --------------------------
# 9. Evaluate
# --------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n📊 Model Performance:")
print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_proba))

# --------------------------
# 10. Save model & encoder
# --------------------------
model_path = os.path.join(MODEL_DIR, "fraud_model.json")
le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

model.save_model(model_path)   # save model as JSON
joblib.dump(le_type, le_path)

print(f"\n✅ Saved model to: {model_path}")
print(f"✅ Saved label encoder to: {le_path}")
