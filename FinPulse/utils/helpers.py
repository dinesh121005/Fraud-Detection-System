# utils/helpers.py
import numpy as np

def color_flag(val):
    """Streamlit table coloring for fraud."""
    if val == 1:
        return "background-color: red; color: white;"
    return ""

def compute_heuristic_risk(tx: dict) -> float:
    """
    Returns a heuristic risk score (0.0 - 1.0) based on simple rules.
    tx dict must contain:
        - amount
        - oldbalanceOrg
        - newbalanceOrig
        - type
        - auth_verified (0/1)
        - device_known (0/1)
        - geo_known (0/1)
        - sender_hourly_tx_count
        - sender_daily_tx_count
        - receiver_risk_score
    """
    score = 0.0

    # Amount vs origin balance
    if tx["oldbalanceOrg"] > 0:
        ratio = tx["amount"] / (tx["oldbalanceOrg"] + 1e-5)
        if ratio > 1.0:
            score += min(0.4, ratio * 0.1)
    else:
        if tx["amount"] > 0:
            score += 0.3

    # Transaction type
    if tx["type"] in ["CASH_OUT", "TRANSFER"]:
        score += 0.2

    # Auth/device/geo
    if tx.get("auth_verified", 1) == 0:
        score += 0.2
    if tx.get("device_known", 1) == 0:
        score += 0.1
    if tx.get("geo_known", 1) == 0:
        score += 0.1

    # Velocity / risk
    if tx.get("sender_hourly_tx_count", 0) > 5:
        score += 0.1
    if tx.get("sender_daily_tx_count", 0) > 20:
        score += 0.1
    if tx.get("receiver_risk_score", 0) > 0.7:
        score += 0.1

    return min(score, 1.0)

def explain_heuristic(tx: dict) -> list:
    """
    Returns a list of reasons why the transaction is flagged (heuristic explanation)
    """
    reasons = []

    if tx.get("auth_verified", 1) == 0:
        reasons.append("Auth not verified (OTP/biometric missing)")

    if tx.get("device_known", 1) == 0:
        reasons.append("Unknown device for sender")

    if tx.get("geo_known", 1) == 0:
        reasons.append("Unknown geo location for sender")

    if tx.get("receiver_risk_score", 0) > 0.7:
        reasons.append("Receiver has high risk score")

    if tx.get("sender_hourly_tx_count", 0) > 5:
        reasons.append("Sender exceeded hourly transaction count")

    if tx.get("sender_daily_tx_count", 0) > 20:
        reasons.append("Sender exceeded daily transaction count")

    if tx.get("oldbalanceOrg", 0) == 0 and tx.get("amount", 0) > 0:
        reasons.append("Large amount sent from zero balance account")

    if len(reasons) == 0:
        reasons.append("No suspicious behavior detected")

    return reasons
