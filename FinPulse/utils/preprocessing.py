import pandas as pd
import numpy as np

def preprocess_transaction(tx):
    df = pd.DataFrame([tx])
    # derived
    df["amount_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1e-6)
    df["unusual_device"] = (df["device_known"] == 0).astype(int)
    df["unusual_location"] = (df["geo_known"] == 0).astype(int)
    df["initiated_by_third"] = (df["initiated_by"] == "third_party").astype(int)
    
    # Ensure numeric fill
    df = df.fillna(0)
    
    # Remove columns that are completely NaN (if any)
    df = df.dropna(axis=1, how='all')
    
    feature_order = [
        "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "amount_balance_ratio", "unusual_device", "unusual_location",
        "sender_hourly_tx_count", "sender_daily_tx_count",
        "receiver_risk_score", "initiated_by_third", "auth_verified"
    ]
    
    # Some transactions may not have all keys - ensure columns exist
    for c in feature_order:
        if c not in df.columns:
            df[c] = 0
            
    # Reorder columns according to the feature_order
    df = df[feature_order]
    
    return df
